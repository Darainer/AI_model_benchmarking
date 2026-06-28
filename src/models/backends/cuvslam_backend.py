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

# ── cuVSLAM C API types (cuvslam.h, Isaac ROS ≥ 3.x) ────────────────────────


class _CuvSLAMCamera(ctypes.Structure):
    """Per-camera intrinsics + extrinsics (CUVSLAM_Camera in cuvslam.h)."""
    _fields_ = [
        ("fx",               ctypes.c_double),
        ("fy",               ctypes.c_double),
        ("cx",               ctypes.c_double),
        ("cy",               ctypes.c_double),
        # 8 distortion coefficients — all zeros for the default pinhole model
        ("k",                ctypes.c_double * 8),
        # 3×4 [R|t] extrinsics, row-major; identity for the sole monocular camera
        ("extrinsics_rt",    ctypes.c_double * 12),
        ("width",            ctypes.c_int),
        ("height",           ctypes.c_int),
        # 0=pinhole (no distortion), 1=radtan, 2=equidistant
        ("distortion_model", ctypes.c_int),
    ]


class _CuvSLAMCameraRig(ctypes.Structure):
    """Camera rig descriptor (CUVSLAM_CameraRig in cuvslam.h)."""
    _fields_ = [
        ("cameras",     ctypes.POINTER(_CuvSLAMCamera)),
        ("num_cameras", ctypes.c_int),
    ]


class _CuvSLAMPose(ctypes.Structure):
    """SE3 pose (CUVSLAM_Pose in cuvslam.h): rotation matrix + translation."""
    _fields_ = [
        ("r", ctypes.c_float * 9),   # row-major 3×3 rotation matrix
        ("t", ctypes.c_float * 3),   # translation vector (metres)
    ]


class _CuvSLAMImage(ctypes.Structure):
    """Single-camera image frame (CUVSLAM_Image in cuvslam.h)."""
    _fields_ = [
        ("timestamp_ns",       ctypes.c_int64),
        ("pixels",             ctypes.c_void_p),
        ("width",              ctypes.c_int),
        ("height",             ctypes.c_int),
        ("pitch",              ctypes.c_int),    # row stride in bytes
        ("num_bytes_in_pixel", ctypes.c_int),
        ("camera_index",       ctypes.c_int),
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
        self._cam: Optional[_CuvSLAMCamera] = None    # keep-alive for rig pointer
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

        # Build monocular pinhole camera descriptor with identity extrinsics
        self._cam = _CuvSLAMCamera(
            fx=self.config.get("camera_fx", w * 0.8),
            fy=self.config.get("camera_fy", w * 0.8),
            cx=self.config.get("camera_cx", w / 2.0),
            cy=self.config.get("camera_cy", h / 2.0),
            k=(ctypes.c_double * 8)(*([0.0] * 8)),
            extrinsics_rt=(ctypes.c_double * 12)(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ),
            width=w,
            height=h,
            distortion_model=0,   # pinhole
        )
        rig = _CuvSLAMCameraRig(cameras=ctypes.pointer(self._cam), num_cameras=1)

        # CUVSLAM_CreateTracker(handle*, rig*, cfg*) — pass NULL cfg for defaults
        self._lib.CUVSLAM_CreateTracker.restype = ctypes.c_int
        self._lib.CUVSLAM_CreateTracker.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_CuvSLAMCameraRig),
            ctypes.c_void_p,   # CUVSLAM_Configuration* — NULL = library defaults
        ]
        handle = ctypes.c_void_p(None)
        status = self._lib.CUVSLAM_CreateTracker(
            ctypes.byref(handle), ctypes.byref(rig), None
        )
        if status != 0:
            raise RuntimeError(f"CUVSLAM_CreateTracker returned status {status}")
        self._handle = handle

        # CUVSLAM_Track(handle, images*, num_images, predicted_pose*, pose_estimate*)
        self._lib.CUVSLAM_Track.restype = ctypes.c_int
        self._lib.CUVSLAM_Track.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CuvSLAMImage),
            ctypes.c_size_t,
            ctypes.POINTER(_CuvSLAMPose),   # predicted_pose — NULL = free-running
            ctypes.POINTER(_CuvSLAMPose),   # pose_estimate out
        ]
        self._lib.CUVSLAM_DestroyTracker.restype = ctypes.c_int
        self._lib.CUVSLAM_DestroyTracker.argtypes = [ctypes.c_void_p]

        logger.info("%s: cuVSLAM tracker initialised (%dx%d)", self.name, w, h)

    # ── inference ──────────────────────────────────────────────────────────
    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import cv2

        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        img = _CuvSLAMImage(
            timestamp_ns=self._frame_idx * 33_000_000,   # synthetic ~30 fps timestamps
            pixels=gray.ctypes.data,
            width=w,
            height=h,
            pitch=w,              # grayscale: 1 byte/pixel, no padding
            num_bytes_in_pixel=1,
            camera_index=0,
        )
        pose = _CuvSLAMPose()
        status = self._lib.CUVSLAM_Track(
            self._handle,
            ctypes.byref(img),
            ctypes.c_size_t(1),
            None,                 # no pose prediction — free-running odometry
            ctypes.byref(pose),
        )
        self._frame_idx += 1

        if status not in (0, 1):   # 0=OK, 1=tracking (valid pose)
            logger.debug("%s: CUVSLAM_Track status=%d", self.name, status)

        # Return flattened [r0..r8, tx, ty, tz] for easy logging
        return [np.array(list(pose.r) + list(pose.t), dtype=np.float32)]

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
