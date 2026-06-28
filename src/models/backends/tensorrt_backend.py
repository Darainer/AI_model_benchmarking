"""Native TensorRT inference backend.

Requires:
  - JetPack / TensorRT installed on the host (provides `tensorrt` Python pkg)
  - pycuda: pip install pycuda

Engine files (.trt / .engine) can be built from ONNX with:
  trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
"""
from typing import Any, Dict, List
import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TensorRTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._engine = None
        self._context = None
        self._bindings: List = []
        self._host_inputs: List[np.ndarray] = []
        self._host_outputs: List[np.ndarray] = []
        self._cuda_inputs = []
        self._cuda_outputs = []
        self._stream = None

    def load(self, model_path: str) -> None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401 — initializes CUDA driver
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT backend requires tensorrt and pycuda. "
                "On Jetson: sudo apt install python3-libnvinfer python3-libnvinfer-dev "
                "and pip install pycuda."
            ) from exc

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        self._bindings = []
        self._host_inputs = []
        self._host_outputs = []
        self._cuda_inputs = []
        self._cuda_outputs = []

        for i in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = np.float32
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(cuda_mem))
            if self._engine.binding_is_input(i):
                self._host_inputs.append(host_mem)
                self._cuda_inputs.append(cuda_mem)
            else:
                self._host_outputs.append(host_mem)
                self._cuda_outputs.append(cuda_mem)

        logger.info(
            "%s: TensorRT engine loaded — %d bindings, %d inputs, %d outputs",
            self.name,
            self._engine.num_bindings,
            len(self._host_inputs),
            len(self._host_outputs),
        )

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import pycuda.driver as cuda

        blob = self.preprocess(frame).ravel()
        np.copyto(self._host_inputs[0], blob)
        cuda.memcpy_htod_async(self._cuda_inputs[0], self._host_inputs[0], self._stream)
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        for host, device in zip(self._host_outputs, self._cuda_outputs):
            cuda.memcpy_dtoh_async(host, device, self._stream)
        self._stream.synchronize()
        return [np.copy(o) for o in self._host_outputs]

    def backend_name(self) -> str:
        return "TensorRT"
