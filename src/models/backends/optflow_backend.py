"""CUDA-accelerated sparse optical flow visual odometry backend.

Uses OpenCV's CUDA Lucas-Kanade sparse optical flow (SparsePyrLKOpticalFlow)
for GPU-accelerated feature tracking across consecutive frames. Relative camera
pose (R, t) is estimated from tracked correspondences via Essential Matrix +
RANSAC (CPU, since OpenCV has no CUDA path for this).

No additional packages are required — OpenCV is compiled with CUDA support in
all JetPack releases on Jetson hardware.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_CORNERS  = 500
_QUALITY      = 0.01
_MIN_DIST     = 10.0
_REDETECT_THR = 80   # re-detect when fewer than this many points survive tracking


class CudaOptFlowModel(BaseModel):
    """
    GPU-accelerated monocular visual odometry — CUDA Lucas-Kanade tracking.

    Per-frame pipeline:
      1. Upload current grayscale frame to GPU
      2. CUDA Pyramid LK optical flow to track keypoints from previous frame
      3. Download tracked points to CPU
      4. Essential Matrix + RANSAC → relative pose [R|t]

    The warmup phase lets the tracker accumulate a stable feature set before
    timed inference begins.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._lk = None
        self._K: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None

    def load(self, model_path: str) -> None:
        import cv2

        if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError(
                "OpenCV CUDA is not available.\n"
                "This backend requires OpenCV compiled with CUDA support,\n"
                "which is included in all JetPack releases on Jetson hardware."
            )

        self._lk = cv2.cuda.SparsePyrLKOpticalFlow_create(
            winSize=(21, 21),
            maxLevel=3,
            iters=30,
            useInitialFlow=False,
        )

        _, _, h, w = self.input_shape
        self._K = np.array([
            [self.config.get("camera_fx", w * 0.8), 0.0, self.config.get("camera_cx", w / 2.0)],
            [0.0, self.config.get("camera_fy", w * 0.8), self.config.get("camera_cy", h / 2.0)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        logger.info(
            "%s: CUDA optical flow VO ready  "
            "CUDA devices=%d  resolution=%dx%d  "
            "K=[[%.1f, %.1f], [%.1f, %.1f]]",
            self.name,
            cv2.cuda.getCudaEnabledDeviceCount(),
            w, h,
            self._K[0, 0], self._K[0, 2],
            self._K[1, 1], self._K[1, 2],
        )

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        # Resize + grayscale — untimed preprocessing.
        _, _, h, w = self.input_shape
        return cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)

    def infer_prepared(self, gray: np.ndarray) -> List[np.ndarray]:
        import cv2

        identity = np.eye(3, 4, dtype=np.float32).ravel()  # 12-element [R|t]

        # First frame or feature set depleted: initialise and return identity pose
        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < _REDETECT_THR:
            self._prev_gray = gray
            self._prev_pts  = self._detect(gray)
            return [identity]

        # ── GPU: pyramid Lucas-Kanade tracking ─────────────────────────────
        prev_gpu = cv2.cuda_GpuMat()
        curr_gpu = cv2.cuda_GpuMat()
        prev_gpu.upload(self._prev_gray)
        curr_gpu.upload(gray)

        # SparsePyrLK on CUDA needs points as [1, N, 2] float32
        pts_gpu = cv2.cuda_GpuMat()
        pts_gpu.upload(self._prev_pts.reshape(1, -1, 2).astype(np.float32))

        next_pts_gpu, status_gpu, _ = self._lk.calc(prev_gpu, curr_gpu, pts_gpu, None)

        # ── CPU: filter tracked points and recover pose ─────────────────────
        next_pts = next_pts_gpu.download().reshape(-1, 2)
        status   = status_gpu.download().ravel().astype(bool)

        good_prev = self._prev_pts.reshape(-1, 2)[status]
        good_next = next_pts[status]

        pose = identity.copy()
        if len(good_prev) >= 5:
            E, _ = cv2.findEssentialMat(
                good_prev, good_next, self._K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0,
            )
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, good_prev, good_next, self._K)
                pose = np.concatenate([R.ravel(), t.ravel()]).astype(np.float32)

        # ── Update state ────────────────────────────────────────────────────
        self._prev_gray = gray
        self._prev_pts  = (
            self._detect(gray)
            if status.sum() < _REDETECT_THR
            else good_next.reshape(-1, 1, 2).astype(np.float32)
        )

        return [pose]

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        import cv2
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=_MAX_CORNERS,
            qualityLevel=_QUALITY,
            minDistance=_MIN_DIST,
        )
        return pts if pts is not None else np.zeros((0, 1, 2), dtype=np.float32)

    def backend_name(self) -> str:
        return "CUDA/OptFlow"
