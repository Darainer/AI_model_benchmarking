"""SuperPoint + LightGlue feature matching pipeline — native PyTorch backend.

Uses the official lightglue pip package directly, without ONNX export.
Weights are downloaded automatically from HuggingFace on first use.

Install: pip install lightglue
         pip install torch  (included in the Jetson JetPack base image)
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LightGlueModel(BaseModel):
    """Two-stage feature matching: SuperPoint extractor + LightGlue matcher.

    Both stages run as native PyTorch on CUDA (or CPU as fallback). The same
    frame is used as both image0 and image1 to time the full pipeline at the
    maximum possible match count.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._extractor = None
        self._matcher = None
        self._device: Optional[str] = None

    def load(self, model_path: str) -> None:
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
        except ImportError as exc:
            raise RuntimeError(
                "lightglue or torch not found.\n"
                "Install: pip install lightglue torch"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        max_kpts = self.config.get("max_keypoints", 512)

        logger.info(
            "%s: loading SuperPoint + LightGlue on %s (max_keypoints=%d)",
            self.name, self._device, max_kpts,
        )
        self._extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(self._device)
        self._matcher   = LightGlue(features="superpoint").eval().to(self._device)

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import cv2
        import torch

        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        # SuperPoint expects [1, 1, H, W] float32 in [0, 1]
        img = (
            torch.from_numpy(gray)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            feats0 = self._extractor.extract(img)
            feats1 = self._extractor.extract(img)  # same frame — maximises match count
            matches01 = self._matcher({"image0": feats0, "image1": feats1})

        # matches01['matches'] is [K, 2] in current LightGlue; some versions use 'matches0'
        out = matches01.get("matches", matches01.get("matches0"))
        if out is not None and hasattr(out, "cpu"):
            return [out.cpu().numpy()]
        return [np.empty((0, 2), dtype=np.int32)]

    def backend_name(self) -> str:
        return f"LightGlue/{(self._device or 'unknown').upper()}"
