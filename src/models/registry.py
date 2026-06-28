"""Load a model config and return the appropriate backend instance."""
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_model(
    config: Dict[str, Any],
    model_path: Path,
    ort_providers: Optional[List[str]] = None,
) -> BaseModel:
    backend = config.get("backend", "onnx")

    if backend == "onnx":
        from src.models.backends.onnx_backend import OnnxModel
        m = OnnxModel(config, ort_providers=ort_providers)

    elif backend == "tensorrt":
        from src.models.backends.tensorrt_backend import TensorRTModel
        m = TensorRTModel(config)

    elif backend == "torch":
        from src.models.backends.torch_backend import TorchModel
        m = TorchModel(config)

    elif backend == "lightglue":
        from src.models.backends.feature_matching_backend import LightGlueModel
        m = LightGlueModel(config, ort_providers=ort_providers)

    elif backend == "cuvslam":
        from src.models.backends.cuvslam_backend import CuVSLAMModel
        m = CuVSLAMModel(config)

    elif backend == "cuda_optflow":
        from src.models.backends.optflow_backend import CudaOptFlowModel
        m = CudaOptFlowModel(config)

    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{config['name']}'")

    m.load(str(model_path))
    return m
