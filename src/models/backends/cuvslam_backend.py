"""NVIDIA Isaac ROS cuVSLAM visual SLAM benchmark backend.

cuVSLAM (CUDA Visual SLAM) is NVIDIA's GPU-accelerated monocular/stereo
visual odometry library, shipped as part of Isaac ROS.

This backend wraps the cuVSLAM C library directly via ctypes so it can be
timed inside our standard benchmark loop without a running ROS 2 daemon.

Installation (Jetson Orin / Isaac ROS ≥ 3.x):
    sudo apt install ros-humble-isaac-ros-visual-slam
    # or pull the Isaac ROS container:
    # docker pull nvcr.io/nvidia/isaac/ros:aarch64-visual_slam-3.2.1

The shared library is expected at one of these paths (searched in order):
    /opt/ros/humble/lib/libcuvslam.so
    /usr/local/lib/libcuvslam.so
    /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/lib/libcuvslam.so
    /usr/lib/aarch64-linux-gnu/libcuvslam.so

Intrinsics can be overridden in configs/models.yaml (camera_fx etc.).
The defaults match a generic 640×480 fisheye camera — tune them to yours.
"""
import ctypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Known install locations for libcuvslam.so on Isaac ROS hosts
_LIB_SEARCH_PATHS = [
    "/opt/ros/humble/lib/libcuvslam.so",
    "/opt/ros/jazzy/lib/libcuvslam.so",
    "/usr/local/lib/libcuvslam.so",
    "/usr/lib/aarch64-linux-gnu/libcuvslam.so",
    "/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/lib/libcuvslam.so",
]

# ── cuVSLAM C API types ────────────────────────────────────────────────────
# Defined in cuvslam.h (Isaac ROS ≥ 3.x).
# Only the subset needed for track + destroy is wired here.

class _CuvSLAMCameraRig(ctypes.Structure):
    _fields_ = [
        ("num_cameras", ctypes.c_int),
        ("width",       ctypes.c_int),
        ("height",      ctypes.c_int),
        ("fx",          ctypes.c_double),
        ("fy",          ctypes.c_double),
        ("cx",          ctypes.c_double),
        ("cy",          ctypes.c_double),
    ]


class _CuvSLAMPose(ctypes.Structure):
    _fields_ = [
        ("tx", ctypes.c_float), ("ty", ctypes.c_float), ("tz", ctypes.c_float),
        ("qx", ctypes.c_float), ("qy", ctypes.c_float),
        ("qz", ctypes.c_float), ("qw", ctypes.c_float),
    ]


def _find_lib() -> Optional[str]:
    # Also check LD_LIBRARY_PATH entries
    extra = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    candidates = _LIB_SEARCH_PATHS + [
        os.path.join(d, "libcuvslam.so") for d in extra if d
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class CuVSLAMModel(BaseModel):
    """
    Times the cuVSLAM per-frame visual tracking latency on GPU.

    Because cuVSLAM maintains internal state (feature map, pose graph), the
    `warmup()` calls serve a dual purpose: they let the SLAM system build
    an initial map before the timed window begins, giving realistic numbers
    rather than first-frame bootstrap overhead.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._lib: Optional[ctypes.CDLL] = None
        self._handle: Optional[ctypes.c_void_p] = None
        self._frame_idx: int = 0

    # ── load ──────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        lib_path = _find_lib()
        if lib_path is None:
            raise RuntimeError(
                "cuVSLAM library (libcuvslam.so) not found.\n\n"
                "Install Isaac ROS Visual SLAM:\n"
                "  sudo apt install ros-humble-isaac-ros-visual-slam\n"
                "or use the Isaac ROS container:\n"
                "  nvcr.io/nvidia/isaac/ros:aarch64-visual_slam\n\n"
                "Searched paths:\n  " + "\n  ".join(_LIB_SEARCH_PATHS)
            )

        logger.info("%s: loading cuVSLAM from %s", self.name, lib_path)
        self._lib = ctypes.CDLL(lib_path)

        _, _, h, w = self.input_shape
        rig = _CuvSLAMCameraRig(
            num_cameras=1,
            width=w,
            height=h,
            fx=self.config.get("camera_fx", w * 0.8),
            fy=self.config.get("camera_fy", w * 0.8),
            cx=self.config.get("camera_cx", w / 2.0),
            cy=self.config.get("camera_cy", h / 2.0),
        )

        # CUVSLAM_CreateTracker(handle*, rig*, use_imu=0, debug=0) → int status
        self._lib.CUVSLAM_CreateTracker.restype  = ctypes.c_int
        self._lib.CUVSLAM_CreateTracker.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_CuvSLAMCameraRig),
            ctypes.c_int,
            ctypes.c_int,
        ]
        handle = ctypes.c_void_p(None)
        status = self._lib.CUVSLAM_CreateTracker(ctypes.byref(handle), ctypes.byref(rig), 0, 0)
        if status != 0:
            raise RuntimeError(f"CUVSLAM_CreateTracker returned status {status}")
        self._handle = handle

        # Wire up Track and Destroy
        self._lib.CUVSLAM_Track.restype  = ctypes.c_int
        self._lib.CUVSLAM_Track.argtypes = [
            ctypes.c_void_p,          # handle
            ctypes.c_int,             # timestamp_ns (truncated to int for ctypes)
            ctypes.c_void_p,          # image data (uint8, row-major)
            ctypes.POINTER(_CuvSLAMPose),  # pose out
        ]
        self._lib.CUVSLAM_DestroyTracker.restype  = ctypes.c_int
        self._lib.CUVSLAM_DestroyTracker.argtypes = [ctypes.c_void_p]

        logger.info("%s: cuVSLAM tracker initialised (%dx%d)", self.name, w, h)

    # ── inference ──────────────────────────────────────────────────────────
    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import cv2, time

        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        pose = _CuvSLAMPose()
        ts_ns = ctypes.c_int(self._frame_idx)   # monotonic frame counter as timestamp
        status = self._lib.CUVSLAM_Track(
            self._handle,
            ts_ns,
            gray.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(pose),
        )
        self._frame_idx += 1

        if status not in (0, 1):   # 0 = OK, 1 = tracking (valid pose)
            logger.debug("%s: CUVSLAM_Track status=%d", self.name, status)

        return [np.array([pose.tx, pose.ty, pose.tz,
                          pose.qx, pose.qy, pose.qz, pose.qw], dtype=np.float32)]

    def warmup(self, runs: int = 10) -> None:
        dummy = np.zeros((self.input_shape[2], self.input_shape[3], 3), dtype=np.uint8)
        for _ in range(runs):
            self.infer(dummy)

    def __del__(self):
        if self._lib and self._handle:
            try:
                self._lib.CUVSLAM_DestroyTracker(self._handle)
            except Exception:
                pass

    def backend_name(self) -> str:
        return "cuVSLAM/CUDA"
