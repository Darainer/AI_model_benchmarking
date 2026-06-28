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

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._active_provider = self._session.get_providers()[0]

        # Detect NHWC layout: shape is [N, H, W, C] — last dim is the channel count
        shape = inp.shape
        self._nhwc = (
            len(shape) == 4
            and isinstance(shape[3], int)
            and shape[3] in (1, 3, 4)
            and isinstance(shape[1], int)
            and shape[1] not in (1, 3, 4)
        )
        logger.info(
            "%s: active provider = %s  inputs=%s  outputs=%s  layout=%s",
            self.name, self._active_provider, self._input_name, self._output_names,
            "NHWC" if self._nhwc else "NCHW",
        )

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        if self.config["task"] == "detection":
            blob = self.preprocess_letterbox(frame)
        else:
            blob = self.preprocess(frame)
        if self._nhwc:
            blob = np.transpose(blob, (0, 2, 3, 1))  # NCHW → NHWC
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
