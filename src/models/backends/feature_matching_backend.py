"""SuperPoint + LightGlue feature matching pipeline backend.

SuperPoint (grayscale CNN) extracts keypoints and descriptors from each image.
LightGlue (transformer matcher) matches the descriptors between two images.

For benchmark timing the same frame is used as both image0 and image1 —
this gives an upper-bound on LightGlue compute (maximum possible match count).

ONNX export (run once via download_models.py):
    pip install lightglue
    # lightglue.onnx module required for LightGlue ONNX export:
    # pip install git+https://github.com/cvg/LightGlue.git
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_batch(arr: np.ndarray, ndim: int = 3) -> np.ndarray:
    """Add a batch dimension if the array is missing it."""
    while arr.ndim < ndim:
        arr = arr[np.newaxis]
    return arr


class LightGlueModel(BaseModel):
    """Two-stage feature matching: SuperPoint extractor + LightGlue matcher."""

    def __init__(self, config: Dict[str, Any], ort_providers: Optional[List] = None):
        super().__init__(config)
        self._extractor = None
        self._matcher   = None
        self._ext_input_name: Optional[str]  = None
        self._ext_output_names: Optional[List[str]] = None
        self._mat_input_names:  Optional[List[str]] = None
        self._mat_output_names: Optional[List[str]] = None
        self._active_provider: Optional[str] = None
        self._requested_providers = ort_providers or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    # ── load ──────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime not found.") from exc

        available = ort.get_available_providers()
        providers  = [p for p in self._requested_providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # ── SuperPoint (extractor) ────────────────────────────────────────
        self._extractor = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self._ext_input_name   = self._extractor.get_inputs()[0].name
        self._ext_output_names = [o.name for o in self._extractor.get_outputs()]
        self._active_provider  = self._extractor.get_providers()[0]
        logger.info(
            "%s: SuperPoint loaded — input=%s  outputs=%s  provider=%s",
            self.name, self._ext_input_name, self._ext_output_names, self._active_provider,
        )

        # ── LightGlue (matcher) ───────────────────────────────────────────
        matcher_name = Path(self.config["local_path_matcher"]).name
        matcher_path = str(Path(model_path).parent / matcher_name)
        self._matcher = ort.InferenceSession(matcher_path, sess_options=opts, providers=providers)
        self._mat_input_names  = [i.name for i in self._matcher.get_inputs()]
        self._mat_output_names = [o.name for o in self._matcher.get_outputs()]
        logger.info(
            "%s: LightGlue loaded — inputs=%s  outputs=%s",
            self.name, self._mat_input_names, self._mat_output_names,
        )

    # ── preprocessing ──────────────────────────────────────────────────────
    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 → float32 [1, 1, H, W] in [0, 1]."""
        import cv2
        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        return (gray.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]  # [1, 1, H, W]

    # ── inference ──────────────────────────────────────────────────────────
    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        gray = self._to_gray(frame)
        h, w = int(gray.shape[2]), int(gray.shape[3])

        # Run SuperPoint twice (same frame used as both images for timing)
        raw0 = self._extractor.run(self._ext_output_names, {self._ext_input_name: gray})
        raw1 = self._extractor.run(self._ext_output_names, {self._ext_input_name: gray})

        # SuperPoint outputs: [keypoints [N,2], scores [N], descriptors [N,256]]
        # Ensure batch dimension is present
        kpts0, scores0, desc0 = [_ensure_batch(x, 3) for x in raw0]
        kpts1, scores1, desc1 = [_ensure_batch(x, 3) for x in raw1]

        img_size = np.array([[h, w]], dtype=np.float32)

        # Build LightGlue input dict by matching on known field name patterns
        mat_in: Dict[str, np.ndarray] = {}
        for name in self._mat_input_names:
            n = name.lower()
            if   "kpts0"   in n: mat_in[name] = kpts0
            elif "kpts1"   in n: mat_in[name] = kpts1
            elif "desc0"   in n: mat_in[name] = desc0
            elif "desc1"   in n: mat_in[name] = desc1
            elif "size0"   in n or ("image0" in n and "size" in n): mat_in[name] = img_size
            elif "size1"   in n or ("image1" in n and "size" in n): mat_in[name] = img_size
            else:
                logger.warning("%s: unknown LightGlue input '%s' — skipping", self.name, name)

        return self._matcher.run(self._mat_output_names, mat_in)

    def warmup(self, runs: int = 10) -> None:
        dummy = np.zeros((self.input_shape[2], self.input_shape[3], 3), dtype=np.uint8)
        for _ in range(runs):
            self.infer(dummy)

    def backend_name(self) -> str:
        short = {
            "TensorrtExecutionProvider": "ORT/TRT",
            "CUDAExecutionProvider":     "ORT/CUDA",
            "CPUExecutionProvider":      "ORT/CPU",
        }
        return f"LightGlue/{short.get(self._active_provider or '', 'unknown')}"
