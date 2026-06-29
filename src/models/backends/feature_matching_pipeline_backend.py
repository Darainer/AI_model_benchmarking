"""PROTOTYPE — overlapped SuperPoint(DLA) ∥ LightGlue(GPU) streaming pipeline.

⚠️  UNTESTED ON HARDWARE. Runs anywhere via fallback; true DLA/GPU overlap only
    engages on Jetson Orin NX in "trt-dla" mode (native TensorRT on a CUDA stream).

This is the variant that targets the actual win: keep both the NVDLA and the GPU
busy at the same time on a *stream* of frames, matching each frame to the
previous one (the real feature-matching use case — odometry / tracking), instead
of matching a frame to itself.

Software pipeline (1-frame latency, double-buffered)
----------------------------------------------------
Per frame N, two CUDA streams are driven before a single sync:

    dla_stream :  backbone(N)            ── NVDLA, zero-copy, async
    gpu_stream :  postproc(N-1) + match(N-2,N-1)  ── GPU, FP16

Because backbone(N) and the GPU work for the previous frame are issued to
different streams, they execute concurrently:

    frame:     N-1        N          N+1
    DLA  :   bbone(N-1) bbone(N)   bbone(N+1)
    GPU  :     …        post(N-1)   post(N)
                        match(N-2)  match(N-1)
               └── overlap ──┘

Net effect we expect on Orin NX:
  * The conv backbone leaves the GPU 3D engine → measured GR3D% (gpu_util_avg)
    drops vs the all-GPU `lightglue` backend.
  * Wall-clock per frame trends toward max(DLA_backbone, GPU_rest) rather than
    their sum — freeing GPU headroom for a *second* concurrent model.

In "ort-dla" / "gpu" modes the backbone call is synchronous, so there is no
overlap — the pipeline still produces correct consecutive matches, just without
the concurrency win. `backend_name()` / logs report the active mode.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.models.backends._superpoint_dla import SuperPointDLAExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LightGlueDLAPipelineModel(BaseModel):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._extractor: Optional[SuperPointDLAExtractor] = None
        self._matcher = None
        self._device: Optional[str] = None
        self._overlap = False                  # true only in trt-dla on CUDA

        self._dla_stream = None
        self._gpu_stream = None

        # double-buffer state
        self._pending = None                   # backbone outputs of frame N-1 (+event)
        self._prev_feats = None                # features of frame N-2

        # GPU-busy accounting (CUDA events), exponential moving averages
        self._gpu_busy_ema = 0.0
        self._frame_ema = 0.0
        self._frames = 0

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
        except ImportError as exc:
            raise RuntimeError(
                "lightglue or torch not found.\nInstall: pip install lightglue torch"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        max_kpts = self.config.get("max_keypoints", 512)
        logger.info("%s: loading overlapped SuperPoint(DLA)∥LightGlue(GPU) on %s",
                    self.name, self._device)

        superpoint = SuperPoint(max_num_keypoints=max_kpts).eval().to(self._device)
        self._matcher = LightGlue(features="superpoint").eval().to(self._device)
        self._extractor = SuperPointDLAExtractor(
            superpoint, self.input_shape, self.config, self._device
        )
        self._extractor.setup()

        if self._device == "cuda":
            self._dla_stream = torch.cuda.Stream()
            self._gpu_stream = torch.cuda.Stream()
            self._overlap = self._extractor.mode == "trt-dla"
        logger.info("%s: overlap=%s (mode=%s)", self.name, self._overlap, self._extractor.mode)

    # ── per-frame pipeline step ───────────────────────────────────────────────
    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import cv2
        import torch

        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        img = (torch.from_numpy(gray).float().div(255.0)
               .unsqueeze(0).unsqueeze(0).to(self._device))

        if not self._overlap:
            return self._infer_serial(img, (h, w))
        return self._infer_overlapped(img, (h, w))

    def _infer_overlapped(self, img, hw) -> List[np.ndarray]:
        """trt-dla path: backbone(N) on dla_stream ∥ GPU work for frame N-1."""
        import torch

        result: List[np.ndarray] = [np.empty((0, 2), dtype=np.int32)]
        gpu_t0, gpu_t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            # 1. launch backbone(N) on the DLA stream (async, zero-copy)
            with torch.cuda.stream(self._dla_stream):
                scores_N, desc_N = self._extractor.run_backbone(img, stream=self._dla_stream)
            bb_event = torch.cuda.Event()
            bb_event.record(self._dla_stream)

            # 2. on the GPU stream, process the PREVIOUS frame — overlaps step 1
            with torch.cuda.stream(self._gpu_stream):
                if self._pending is not None:
                    # don't read backbone outputs until the DLA has written them
                    self._gpu_stream.wait_event(self._pending["event"])
                gpu_t0.record(self._gpu_stream)
                if self._pending is not None:
                    feats_prev = self._extractor.postprocess(
                        self._pending["scores"], self._pending["desc"], hw)
                    if self._prev_feats is not None:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            m = self._matcher({"image0": self._prev_feats, "image1": feats_prev})
                        out = m.get("matches", m.get("matches0"))
                        if out is not None and hasattr(out, "cpu"):
                            result = [out.cpu().numpy()]
                    self._prev_feats = feats_prev
                gpu_t1.record(self._gpu_stream)

            # 3. single sync point — both engines have run concurrently up to here
            torch.cuda.synchronize()

        self._pending = {"scores": scores_N, "desc": desc_N, "event": bb_event, "img": img}
        self._record_busy(gpu_t0, gpu_t1)
        return result

    def _infer_serial(self, img, hw) -> List[np.ndarray]:
        """ort-dla / gpu path: no overlap, still consecutive-frame matching."""
        import torch

        result: List[np.ndarray] = [np.empty((0, 2), dtype=np.int32)]
        with torch.no_grad():
            feats = self._extractor.extract(img)
            if self._prev_feats is not None:
                if self._device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        m = self._matcher({"image0": self._prev_feats, "image1": feats})
                else:
                    m = self._matcher({"image0": self._prev_feats, "image1": feats})
                out = m.get("matches", m.get("matches0"))
                if out is not None and hasattr(out, "cpu"):
                    result = [out.cpu().numpy()]
            self._prev_feats = feats
        return result

    def _record_busy(self, gpu_t0, gpu_t1) -> None:
        """Track GPU-busy fraction via CUDA events — evidence for 'GPU util down'."""
        try:
            gpu_ms = gpu_t0.elapsed_time(gpu_t1)
        except Exception:
            return
        self._frames += 1
        a = 0.05
        self._gpu_busy_ema = (1 - a) * self._gpu_busy_ema + a * gpu_ms if self._frames > 1 else gpu_ms
        if self._frames % 25 == 0:
            logger.info("%s: GPU-busy/frame ≈ %.2f ms (EMA) over %d frames",
                        self.name, self._gpu_busy_ema, self._frames)

    def backend_name(self) -> str:
        dev = (self._device or "unknown").upper()
        mode = self._extractor.mode if self._extractor else "?"
        tag = "overlap" if self._overlap else "serial"
        return f"SP-{mode}∥LG[{tag}]/{dev}"
