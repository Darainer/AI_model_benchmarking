"""Zero-copy SuperPoint conv backbone on the NVDLA (Orin NX) + GPU post-proc.

⚠️  UNTESTED ON HARDWARE — the DLA paths require Jetson Orin NX silicon with
    JetPack TensorRT. Off-Jetson the extractor transparently falls back to the
    stock GPU SuperPoint so everything still runs.

This module is shared by:
  * feature_matching_dla_backend.py      (single-frame, self-match benchmark)
  * feature_matching_pipeline_backend.py (streaming, consecutive-frame, overlapped)

Three execution modes, picked at setup() in priority order:
  "trt-dla"  native TensorRT engine, DLA core + GPU fallback, FP16.
             Zero-copy: TRT reads/writes torch CUDA tensors directly via
             set_tensor_address(); execute_async_v3() runs on a caller-supplied
             CUDA stream, so the backbone can overlap GPU work. ← the real path.
  "ort-dla"  onnxruntime TensorRT EP with trt_dla_enable. Zero-copy via
             io_binding bound to torch tensor data_ptr()s, but run is
             synchronous (no cross-engine overlap). A simpler fallback.
  "gpu"      stock lightglue SuperPoint on the GPU. No DLA, no zero-copy needed.

The split point: only the static-shape conv stack (conv1a … convPb/convDb)
goes to the DLA. softmax → depth-to-space → NMS → top-k → grid_sample produce
dynamic shapes and always run on the GPU here.
"""
from typing import Any, Dict, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)

# SuperPoint conv-stack attribute names in the `lightglue` package.
_BACKBONE_CONVS = (
    "conv1a", "conv1b", "conv2a", "conv2b",
    "conv3a", "conv3b", "conv4a", "conv4b",
    "convPa", "convPb",   # detector (score) head
    "convDa", "convDb",   # descriptor head
)


# ── GPU post-processing primitives (dynamic shapes — never on the DLA) ───────

def simple_nms(scores, radius: int):
    """SuperPoint non-max suppression via iterated max-pool."""
    import torch
    import torch.nn.functional as F

    assert radius >= 0

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Bilinear-sample dense descriptors at keypoint locations (SuperPoint)."""
    import torch
    import torch.nn.functional as F

    b, c, h, w = descriptors.shape
    kpts = keypoints - s / 2 + 0.5
    kpts = kpts / torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)], device=kpts.device
    )
    kpts = kpts * 2 - 1
    descriptors = F.grid_sample(
        descriptors, kpts.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    )
    return F.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)


class _SuperPointBackbone:
    """Runs only the conv stack of a lightglue SuperPoint → the two raw dense
    conv outputs that are static-shape and DLA-friendly."""

    def __init__(self, superpoint):
        import torch.nn as nn

        self.sp = superpoint
        self.relu = nn.ReLU(inplace=True)
        missing = [c for c in _BACKBONE_CONVS if not hasattr(superpoint, c)]
        if missing:
            raise AttributeError(
                f"lightglue SuperPoint missing conv layers {missing}; "
                "package layout changed — cannot extract DLA backbone."
            )

    def forward(self, img):
        import torch.nn.functional as F

        sp = self.sp
        x = self.relu(sp.conv1a(img));  x = self.relu(sp.conv1b(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(sp.conv2a(x));    x = self.relu(sp.conv2b(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(sp.conv3a(x));    x = self.relu(sp.conv3b(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(sp.conv4a(x));    x = self.relu(sp.conv4b(x))

        scores_logits = sp.convPb(self.relu(sp.convPa(x)))   # [B, 65, H/8, W/8]
        desc_raw = sp.convDb(self.relu(sp.convDa(x)))        # [B, 256, H/8, W/8]
        return scores_logits, desc_raw


class SuperPointDLAExtractor:
    """SuperPoint feature extraction with the conv backbone optionally on the DLA.

    Lifecycle:
        ex = SuperPointDLAExtractor(superpoint, input_shape, config, device)
        ex.setup()                       # picks trt-dla / ort-dla / gpu
        feats = ex.extract(img_gray)     # synchronous convenience

    For overlapped pipelining use the async primitives:
        s, d = ex.run_backbone(img, stream)   # async on `stream` (trt-dla)
        feats = ex.postprocess(s, d)          # GPU, on current torch stream
    """

    def __init__(self, superpoint, input_shape, config: Dict[str, Any], device: str):
        self.sp = superpoint
        self.input_shape = input_shape           # [N, C, H, W]
        self.config = config
        self.device = device
        self.mode: Optional[str] = None

        _, _, h, w = input_shape
        self._scores_shape = (1, 65, h // 8, w // 8)
        self._desc_shape = (1, 256, h // 8, w // 8)

        self._backbone: Optional[_SuperPointBackbone] = None
        self._onnx_path: Optional[str] = None
        # trt-dla
        self._trt_engine = None
        self._trt_context = None
        # ort-dla
        self._ort_session = None

    # ── setup ────────────────────────────────────────────────────────────────
    def setup(self) -> str:
        if self.config.get("force_gpu_fallback", False):
            self.mode = "gpu"
            logger.info("SuperPointDLAExtractor: force_gpu_fallback → mode=gpu")
            return self.mode
        try:
            self._backbone = _SuperPointBackbone(self.sp)
            self._export_onnx()
            if self._try_trt_dla():
                self.mode = "trt-dla"
            elif self._try_ort_dla():
                self.mode = "ort-dla"
            else:
                self.mode = "gpu"
        except Exception as exc:  # noqa: BLE001 — prototype: any failure → GPU
            logger.warning("SuperPointDLAExtractor: DLA setup failed (%s) → gpu", exc)
            self.mode = "gpu"
        logger.info("SuperPointDLAExtractor: mode=%s", self.mode)
        return self.mode

    def _export_onnx(self) -> None:
        import os
        import torch

        base = os.path.splitext(self.config["local_path"])[0]
        self._onnx_path = base + "_backbone.onnx"
        if os.path.exists(self._onnx_path):
            return
        _, _, h, w = self.input_shape

        class _Wrap(torch.nn.Module):
            def __init__(self, bb):
                super().__init__()
                self.bb = bb

            def forward(self, img):
                return self.bb.forward(img)

        logger.info("Exporting SuperPoint backbone → %s", self._onnx_path)
        wrap = _Wrap(self._backbone).eval().to(self.device)
        dummy = torch.zeros(1, 1, h, w, device=self.device)
        torch.onnx.export(
            wrap, dummy, self._onnx_path,
            input_names=["image"], output_names=["scores_logits", "desc_raw"],
            opset_version=16, do_constant_folding=True,
        )

    # ── trt-dla (native, zero-copy, async) ────────────────────────────────────
    def _try_trt_dla(self) -> bool:
        try:
            import os
            import tensorrt as trt
        except ImportError:
            logger.info("tensorrt python pkg not present → skip trt-dla")
            return False

        dla_core = int(self.config.get("dla_core", 0))
        workspace = int(self.config.get("trt_max_workspace_size", 1 << 30))
        engine_path = os.path.splitext(self._onnx_path)[0] + f".dla{dla_core}.trt"

        trt_logger = trt.Logger(trt.Logger.WARNING)
        if os.path.exists(engine_path):
            with open(engine_path, "rb") as f:
                engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        else:
            logger.info("Building DLA TRT engine (core=%d) → %s", dla_core, engine_path)
            builder = trt.Builder(trt_logger)
            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, trt_logger)
            with open(self._onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("ONNX parse: %s", parser.get_error(i))
                    return False
            cfg = builder.create_builder_config()
            cfg.set_flag(trt.BuilderFlag.FP16)          # DLA is FP16/INT8 only
            cfg.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # spill rejected layers to GPU
            cfg.default_device_type = trt.DeviceType.DLA
            cfg.DLA_core = dla_core
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
            serialized = builder.build_serialized_network(network, cfg)
            if serialized is None:
                logger.warning("DLA engine build returned None → skip trt-dla")
                return False
            with open(engine_path, "wb") as f:
                f.write(serialized)
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(serialized)

        self._trt_engine = engine
        self._trt_context = engine.create_execution_context()
        self._trt_context.set_input_shape("image", (1, 1, self.input_shape[2], self.input_shape[3]))
        return True

    def run_backbone(self, img, stream=None):
        """Async backbone on `stream` (trt-dla). Returns (scores_t, desc_t) torch
        CUDA tensors that become valid once `stream` reaches this point.

        Zero-copy: TRT writes directly into the returned torch tensors.
        """
        import torch

        scores_t = torch.empty(self._scores_shape, dtype=torch.float32, device=self.device)
        desc_t = torch.empty(self._desc_shape, dtype=torch.float32, device=self.device)
        ctx = self._trt_context
        ctx.set_tensor_address("image", int(img.contiguous().data_ptr()))
        ctx.set_tensor_address("scores_logits", int(scores_t.data_ptr()))
        ctx.set_tensor_address("desc_raw", int(desc_t.data_ptr()))
        stream_handle = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        ctx.execute_async_v3(stream_handle=stream_handle)
        return scores_t, desc_t

    # ── ort-dla (zero-copy io-binding, synchronous) ───────────────────────────
    def _try_ort_dla(self) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            logger.info("ORT build lacks TensorrtExecutionProvider → skip ort-dla")
            return False
        dla_core = str(self.config.get("dla_core", 0))
        cache = self.config.get("trt_engine_cache_path", "./models/trt_cache")
        trt_opts = {
            "trt_dla_enable": "1", "trt_dla_core": dla_core, "trt_fp16_enable": "1",
            "trt_engine_cache_enable": "1", "trt_engine_cache_path": cache,
        }
        providers = [("TensorrtExecutionProvider", trt_opts),
                     "CUDAExecutionProvider", "CPUExecutionProvider"]
        self._ort_session = ort.InferenceSession(self._onnx_path, providers=providers)
        return "TensorrtExecutionProvider" in self._ort_session.get_providers()

    def _run_backbone_ort(self, img):
        """Zero-copy ORT io-binding: bind torch CUDA tensors directly, no host copy."""
        import numpy as _np
        import torch

        scores_t = torch.empty(self._scores_shape, dtype=torch.float32, device=self.device)
        desc_t = torch.empty(self._desc_shape, dtype=torch.float32, device=self.device)
        img = img.contiguous()
        io = self._ort_session.io_binding()
        dev = "cuda" if self.device == "cuda" else "cpu"
        io.bind_input("image", dev, 0, _np.float32, tuple(img.shape), img.data_ptr())
        io.bind_output("scores_logits", dev, 0, _np.float32, self._scores_shape, scores_t.data_ptr())
        io.bind_output("desc_raw", dev, 0, _np.float32, self._desc_shape, desc_t.data_ptr())
        self._ort_session.run_with_iobinding(io)
        return scores_t, desc_t

    # ── GPU post-processing → lightglue feats dict ────────────────────────────
    def postprocess(self, scores_logits, desc_raw, image_hw: Tuple[int, int]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        max_kpts = self.config.get("max_keypoints", 512)
        nms_radius = int(self.config.get("nms_radius", 4))
        thresh = float(self.config.get("detection_threshold", 0.0005))

        scores = F.softmax(scores_logits, dim=1)[:, :-1]      # drop dustbin channel
        b, _, hc, wc = scores.shape
        scores = (scores.permute(0, 2, 3, 1).reshape(b, hc, wc, 8, 8)
                  .permute(0, 1, 3, 2, 4).reshape(b, hc * 8, wc * 8))
        scores = simple_nms(scores, nms_radius)

        s = scores[0]
        ys, xs = torch.where(s > thresh)
        kpt_scores = s[ys, xs]
        if kpt_scores.numel() > max_kpts:
            kpt_scores, idx = torch.topk(kpt_scores, max_kpts)
            ys, xs = ys[idx], xs[idx]
        keypoints = torch.stack([xs, ys], dim=-1).float()      # (N, 2) x,y

        descriptors = sample_descriptors(keypoints[None], F.normalize(desc_raw, p=2, dim=1), 8)
        h, w = image_hw
        return {
            "keypoints": keypoints[None],
            "keypoint_scores": kpt_scores[None],
            "descriptors": descriptors.transpose(1, 2),        # [1, N, 256]
            "image_size": torch.tensor([[w, h]], device=self.device).float(),
        }

    # ── synchronous convenience ───────────────────────────────────────────────
    def extract(self, img) -> Dict[str, Any]:
        """Synchronous feature extraction (used by the single-frame backend)."""
        import torch

        if self.mode == "gpu":
            return self.sp.extract(img)

        if self.mode == "trt-dla":
            scores_t, desc_t = self.run_backbone(img, stream=torch.cuda.current_stream())
            torch.cuda.current_stream().synchronize()
        else:  # ort-dla
            scores_t, desc_t = self._run_backbone_ort(img)

        _, _, h, w = img.shape
        return self.postprocess(scores_t, desc_t, (h, w))
