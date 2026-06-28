"""Unified frame iterator over all input types.

Yields (frame_index, bgr_frame) pairs as numpy uint8 arrays.
"""
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

Frame = np.ndarray  # shape (H, W, 3), dtype uint8, BGR
FrameStream = Generator[Tuple[int, Frame], None, None]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def open_source(config: Dict[str, Any]) -> FrameStream:
    src_type = config.get("type", "synthetic")

    if src_type == "synthetic":
        yield from _synthetic(config)
    elif src_type == "image_dir":
        yield from _image_dir(config)
    elif src_type in ("file", "usb_camera", "csi_camera"):
        yield from _gstreamer(config)
    else:
        raise ValueError(f"Unknown input type '{src_type}'")


def _synthetic(config: Dict[str, Any]) -> FrameStream:
    n = config.get("synthetic_frames", 100)
    w = config.get("width", 1280)
    h = config.get("height", 720)
    logger.info("Synthetic source: %d frames at %dx%d", n, w, h)
    rng = np.random.default_rng(42)
    for i in range(n):
        yield i, rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _image_dir(config: Dict[str, Any]) -> FrameStream:
    import cv2
    src = Path(config["source"])
    if not src.is_dir():
        raise FileNotFoundError(f"image_dir not found: {src}")

    paths = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in _IMAGE_EXTS
    )
    max_frames: Optional[int] = config.get("max_frames")
    if max_frames:
        paths = paths[:max_frames]

    logger.info("Image dir source: %d images from %s", len(paths), src)
    for i, p in enumerate(paths):
        frame = cv2.imread(str(p))
        if frame is None:
            logger.warning("Could not read %s — skipping", p)
            continue
        yield i, frame


def _gstreamer(config: Dict[str, Any]) -> FrameStream:
    import cv2
    from src.input.gstreamer_pipeline import build_pipeline

    pipeline = build_pipeline(config)
    logger.info("GStreamer pipeline: %s", pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {config.get('source')}\n"
            "GStreamer pipeline failed to open. If using --input-type file, ensure the "
            "file path is accessible inside the container and the required decoder is "
            "available.\nTo use software decode instead of hardware, pass: "
            "--decoder sw"
        )

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        decoder = config.get("decoder", "hw")
        raise RuntimeError(
            f"GStreamer pipeline opened but produced no frames (decoder={decoder}).\n"
            + (
                "Hardware decode (nvv4l2decoder) failed — the Jetson decoder devices "
                "may not be accessible in this environment.\n"
                "Pass --decoder sw to use software decode instead. Note: this will "
                "affect timing results."
                if decoder == "hw"
                else "Software decode (decodebin) also failed — check the file format."
            )
        )

    max_frames: Optional[int] = config.get("max_frames")
    i = 0
    try:
        yield i, first_frame
        i += 1
        while not (max_frames and i >= max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()
