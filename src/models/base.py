from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseModel(ABC):
    """Common interface for all inference backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config["name"]
        self.task = config["task"]
        self.input_shape = config["input_shape"]  # [N, C, H, W]
        self._input_mean = np.array(config.get("input_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._input_std = np.array(config.get("input_std", [1.0, 1.0, 1.0]), dtype=np.float32)

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model weights / engine from disk."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        """Run inference on a single pre-processed frame.

        Args:
            frame: uint8 BGR image, shape (H, W, 3).

        Returns:
            List of raw output tensors as numpy arrays.
        """

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize → RGB → normalize → CHW → batch."""
        import cv2
        _, _, h, w = self.input_shape
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self._input_mean) / self._input_std
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def preprocess_letterbox(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox resize for detection models (preserves aspect ratio)."""
        import cv2
        _, _, target_h, target_w = self.input_shape
        src_h, src_w = frame.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def warmup(self, runs: int = 10) -> None:
        dummy = np.zeros((self.input_shape[0], self.input_shape[2], self.input_shape[3], 3), dtype=np.uint8)
        for _ in range(runs):
            self.infer(dummy[0])

    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier (e.g. 'onnxruntime/CUDA')."""
