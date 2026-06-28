"""GStreamer pipeline strings for Jetson Nano.

OpenCV is used as the GStreamer host:
    cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)

All pipelines end with `appsink` so OpenCV receives BGR frames.
"""
from typing import Dict, Any


def build_pipeline(config: Dict[str, Any]) -> str:
    src_type = config.get("type", "synthetic")
    width = config.get("width", 1280)
    height = config.get("height", 720)
    fps = config.get("fps", 30)

    if src_type == "csi_camera":
        flip = config.get("flip_method", 0)
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1,format=NV12 ! "
            f"nvvidconv flip-method={flip} ! "
            f"video/x-raw,width={width},height={height},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )

    elif src_type == "usb_camera":
        device = config.get("device", "/dev/video0")
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )

    elif src_type == "file":
        source = config["source"]
        decoder = config.get("decoder", "hw")
        if decoder == "hw":
            # Hardware H.264/H.265 decode on Jetson
            decode_chain = "qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR"
        else:
            decode_chain = "decodebin ! videoconvert ! video/x-raw,format=BGR"
        return (
            f"filesrc location={source} ! "
            f"{decode_chain} ! "
            f"appsink max-buffers=1 drop=true"
        )

    raise ValueError(f"Cannot build GStreamer pipeline for type '{src_type}'. "
                     "Use frame_source.py for image_dir and synthetic inputs.")
