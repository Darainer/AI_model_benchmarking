"""PROTOTYPE — SuperPoint(DLA) + LightGlue(GPU) single-frame backend (Orin NX).

⚠️  UNTESTED ON HARDWARE. Loads and runs anywhere via the GPU fallback; the DLA
    offload only engages on Jetson Orin NX with JetPack TensorRT.

This is the *single-frame, self-match* variant kept for benchmark parity with
the stock `lightglue` backend — it feeds the same frame as image0 and image1 so
the matcher runs at worst-case match count. For the real DLA+GPU concurrency win
(overlapped, consecutive-frame matching) see feature_matching_pipeline_backend.py
and docs/DLA_SUPERPOINT_PROTOTYPE.md.

Zero-copy: the SuperPoint conv backbone runs on the DLA and writes its outputs
directly into torch CUDA tensors (no host round-trip) via SuperPointDLAExtractor.
The LightGlue matcher runs on the GPU in FP16 autocast.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.models.backends._superpoint_dla import SuperPointDLAExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LightGlueDLAModel(BaseModel):
    """SuperPoint conv backbone on DLA (zero-copy), matcher on GPU (FP16)."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._extractor: Optional[SuperPointDLAExtractor] = None
        self._matcher = None
        self._device: Optional[str] = None

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
        logger.info("%s: loading SuperPoint(DLA) + LightGlue on %s (max_keypoints=%d)",
                    self.name, self._device, max_kpts)

        superpoint = SuperPoint(max_num_keypoints=max_kpts).eval().to(self._device)
        self._matcher = LightGlue(features="superpoint").eval().to(self._device)

        self._extractor = SuperPointDLAExtractor(
            superpoint, self.input_shape, self.config, self._device
        )
        self._extractor.setup()

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import cv2
        import torch

        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        img = (torch.from_numpy(gray).float().div(255.0)
               .unsqueeze(0).unsqueeze(0).to(self._device))

        with torch.no_grad():
            feats0 = self._extractor.extract(img)
            feats1 = self._extractor.extract(img)   # same frame → worst-case match count
            if self._device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    matches01 = self._matcher({"image0": feats0, "image1": feats1})
            else:
                matches01 = self._matcher({"image0": feats0, "image1": feats1})

        out = matches01.get("matches", matches01.get("matches0"))
        if out is not None and hasattr(out, "cpu"):
            return [out.cpu().numpy()]
        return [np.empty((0, 2), dtype=np.int32)]

    def backend_name(self) -> str:
        dev = (self._device or "unknown").upper()
        mode = self._extractor.mode if self._extractor else "?"
        return f"SP-{mode}+LG/{dev}"
