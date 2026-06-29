"""PyTorch inference backend — useful as a baseline / debug path."""
from typing import Any, Dict, List
import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TorchModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model = None
        self._device = None

    def load(self, model_path: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch not installed.") from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("%s: loading PyTorch model on %s", self.name, self._device)

        if model_path.endswith(".pt") or model_path.endswith(".pth"):
            self._model = torch.load(model_path, map_location=self._device)
        else:
            # TorchScript
            self._model = torch.jit.load(model_path, map_location=self._device)

        self._model.eval()
        logger.info("%s: model loaded", self.name)

    def prepare(self, frame: np.ndarray):
        import torch
        # Preprocess + host→device copy happen here, outside the timer.
        return torch.from_numpy(self.preprocess(frame)).to(self._device)

    def infer_prepared(self, blob) -> List[np.ndarray]:
        import torch
        with torch.no_grad():
            outputs = self._model(blob)
        # .cpu() forces the CUDA stream to finish, so the timer captures the full
        # forward pass even though kernel launches are async.
        if isinstance(outputs, torch.Tensor):
            return [outputs.cpu().numpy()]
        return [o.cpu().numpy() for o in outputs]

    def backend_name(self) -> str:
        import torch
        return f"PyTorch/{'CUDA' if torch.cuda.is_available() else 'CPU'}"
