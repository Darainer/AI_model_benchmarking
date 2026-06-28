from typing import Any, Dict, List, Optional
import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OnnxModel(BaseModel):
    """ONNX Runtime inference backend.

    Provider priority (from pipeline.yaml) is tried in order; first one
    available in the current onnxruntime build wins. On Jetson with the
    onnxruntime-gpu wheel this will typically resolve to
    TensorrtExecutionProvider or CUDAExecutionProvider.
    """

    def __init__(self, config: Dict[str, Any], ort_providers: Optional[List] = None):
        super().__init__(config)
        self._session = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._active_provider: Optional[str] = None
        self._requested_providers = ort_providers or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def load(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime not found. Install onnxruntime-gpu (Jetson) "
                "or onnxruntime (CPU only)."
            ) from exc

        available = ort.get_available_providers()
        providers = [p for p in self._requested_providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        logger.info(
            "%s: loading with providers %s (available: %s)",
            self.name, providers, available,
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._active_provider = self._session.get_providers()[0]
        logger.info(
            "%s: active provider = %s  inputs=%s  outputs=%s",
            self.name, self._active_provider, self._input_name, self._output_names,
        )

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        if self.config["task"] == "detection":
            blob = self.preprocess_letterbox(frame)
        else:
            blob = self.preprocess(frame)
        return self._session.run(self._output_names, {self._input_name: blob})

    def backend_name(self) -> str:
        provider = self._active_provider or "unknown"
        # Shorten long provider names for display
        short = {
            "TensorrtExecutionProvider": "ORT/TRT",
            "CUDAExecutionProvider": "ORT/CUDA",
            "CPUExecutionProvider": "ORT/CPU",
        }
        return short.get(provider, provider)
