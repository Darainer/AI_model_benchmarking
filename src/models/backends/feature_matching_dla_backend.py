"""PROTOTYPE — SuperPoint(DLA) + LightGlue(GPU) split pipeline for Orin NX.

⚠️  UNTESTED ON HARDWARE. This is an architectural prototype written to be run
    on a Jetson Orin NX (which has 2× NVDLA v2 cores). It will *load and run*
    on any CUDA machine via the full-GPU fallback path, but the DLA offload
    path can only be exercised on real Orin NX silicon with JetPack's
    TensorRT + onnxruntime-gpu (TensorRT EP) stack.

Motivation
----------
The stock `lightglue` backend (see feature_matching_backend.py) runs both
stages on the GPU. Profiling logic in HARDWARE.md flags the pipeline as
bandwidth-bound: the SuperPoint CNN under-fills the GPU while the LightGlue
attention matcher is latency/BW-bound. On Orin NX the idle NVDLA cores are
free real estate.

The split implemented here
--------------------------
  SuperPoint = [ dense conv backbone ]  +  [ keypoint post-processing ]
                        │                            │
                        ▼                            ▼
                  DLA (FP16, ONNX/TRT EP)      GPU (dynamic shapes)
                                                     │
                                                     ▼
                                  LightGlue matcher  GPU (FP16 autocast)

Why the split is *here* and not elsewhere:
  - The conv backbone (conv1a … convPb / convDb) is static-shape FP16 — the
    canonical DLA-friendly workload. GPU_FALLBACK lets unsupported layers
    spill to the GPU automatically.
  - softmax → depth-to-space → NMS → top-k → grid_sample produce DYNAMIC
    shapes (variable keypoint count). DLA cannot run these; they stay on GPU.
  - The matcher is attention over ≤512 nodes — tiny matmuls + softmax that
    never fill the GPU. FP16 autocast is the cheap win here, not a different
    chip.

If anything in the DLA path is unavailable (no onnxruntime, export failure,
no DLA core, CPU-only host) the model silently falls back to running the
stock SuperPoint extractor fully on the GPU so the benchmark still produces a
number. `backend_name()` reports which path is live.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Attribute names of the SuperPoint conv stack in the `lightglue` package.
# Used to extract the DLA-able backbone. If the package renames these, export
# fails gracefully and we fall back to the full-GPU path.
_BACKBONE_CONVS = (
    "conv1a", "conv1b", "conv2a", "conv2b",
    "conv3a", "conv3b", "conv4a", "conv4b",
    "convPa", "convPb",   # detector (score) head
    "convDa", "convDb",   # descriptor head
)


def _superpoint_dla_ep_options(config: Dict[str, Any]) -> List:
    """Build the ORT provider list that pins the SuperPoint backbone to the DLA.

    DLA on Jetson is reached through the TensorRT EP with trt_dla_enable. DLA
    only supports FP16/INT8, and GPU fallback must stay on for the layers the
    DLA core cannot handle.
    """
    dla_core = str(config.get("dla_core", 0))
    workspace = str(config.get("trt_max_workspace_size", 1 << 30))
    cache_path = config.get("trt_engine_cache_path", "./models/trt_cache")
    trt_opts = {
        "trt_dla_enable": "1",
        "trt_dla_core": dla_core,
        "trt_fp16_enable": "1",          # DLA is FP16/INT8 only
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": cache_path,
        "trt_max_workspace_size": workspace,
        # GPU_FALLBACK is implied by the EP when DLA is enabled; layers the
        # DLA rejects (e.g. the final reshape) run on the GPU automatically.
    }
    return [
        ("TensorrtExecutionProvider", trt_opts),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


class _SuperPointBackbone:
    """Thin wrapper that runs only the conv stack of a lightglue SuperPoint.

    Returns the two raw dense conv outputs — exactly the tensors that are
    static-shape and DLA-friendly:
        scores_logits : [B, 65, H/8, W/8]   (convPb output, pre-softmax)
        desc_raw      : [B, 256, H/8, W/8]  (convDb output, pre-normalize)
    """

    def __init__(self, superpoint):
        import torch.nn as nn

        self.sp = superpoint
        self.relu = nn.ReLU(inplace=True)
        # Verify the expected submodules exist before we promise a DLA path.
        missing = [c for c in _BACKBONE_CONVS if not hasattr(superpoint, c)]
        if missing:
            raise AttributeError(
                f"lightglue SuperPoint is missing expected conv layers {missing}; "
                "cannot extract DLA backbone (package layout changed)."
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

        cpa = self.relu(sp.convPa(x))
        scores_logits = sp.convPb(cpa)          # [B, 65, H/8, W/8]

        cda = self.relu(sp.convDa(x))
        desc_raw = sp.convDb(cda)               # [B, 256, H/8, W/8]
        return scores_logits, desc_raw


def _simple_nms(scores, radius: int):
    """SuperPoint non-max suppression via iterated max-pool (GPU postproc)."""
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


class LightGlueDLAModel(BaseModel):
    """SuperPoint conv backbone on DLA, post-proc + LightGlue matcher on GPU.

    Config knobs (configs/models.yaml):
        max_keypoints : top-k keypoints (default 512)
        dla_core      : 0 or 1 — which NVDLA core to target (default 0)
        nms_radius    : NMS radius in pixels (default 4, SuperPoint default)
        detection_threshold : keypoint score floor (default 0.0005)
        force_gpu_fallback  : true → skip DLA path entirely (A/B baseline)
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._extractor = None          # stock SuperPoint (fallback + weights source)
        self._backbone = None           # _SuperPointBackbone (DLA path)
        self._matcher = None
        self._ort_session = None        # ORT session pinned to DLA
        self._device: Optional[str] = None
        self._dla_active = False

    # ------------------------------------------------------------------ load
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
            "%s: loading SuperPoint(DLA-split) + LightGlue on %s (max_keypoints=%d)",
            self.name, self._device, max_kpts,
        )
        self._extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(self._device)
        self._matcher = LightGlue(features="superpoint").eval().to(self._device)

        if self.config.get("force_gpu_fallback", False):
            logger.warning("%s: force_gpu_fallback set — DLA path disabled.", self.name)
            return

        try:
            self._setup_dla_path()
            self._dla_active = True
            logger.info("%s: DLA backbone path ACTIVE (core=%s).",
                        self.name, self.config.get("dla_core", 0))
        except Exception as exc:  # noqa: BLE001 — prototype: any failure → fallback
            logger.warning(
                "%s: DLA path unavailable (%s). Falling back to full-GPU SuperPoint.",
                self.name, exc,
            )
            self._dla_active = False

    def _setup_dla_path(self) -> None:
        """Export the SuperPoint conv backbone to ONNX and pin it to the DLA."""
        import os
        import torch
        import onnxruntime as ort

        _, _, h, w = self.input_shape
        onnx_path = os.path.splitext(self.config["local_path"])[0] + "_backbone.onnx"

        self._backbone = _SuperPointBackbone(self._extractor)

        if not os.path.exists(onnx_path):
            logger.info("%s: exporting SuperPoint backbone → %s", self.name, onnx_path)

            class _ExportWrapper(torch.nn.Module):
                def __init__(self, bb):
                    super().__init__()
                    self.bb = bb

                def forward(self, img):
                    return self.bb.forward(img)

            wrapper = _ExportWrapper(self._backbone).eval().to(self._device)
            dummy = torch.zeros(1, 1, h, w, device=self._device)
            torch.onnx.export(
                wrapper, dummy, onnx_path,
                input_names=["image"],
                output_names=["scores_logits", "desc_raw"],
                opset_version=16,
                do_constant_folding=True,
            )

        providers = _superpoint_dla_ep_options(self.config)
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                f"TensorrtExecutionProvider not in ORT build ({available}); "
                "DLA is only reachable via the TensorRT EP on Jetson."
            )
        self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
        logger.info("%s: backbone ORT session providers=%s",
                    self.name, self._ort_session.get_providers())

    # ----------------------------------------------------------------- infer
    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        import cv2
        import torch

        _, _, h, w = self.input_shape
        gray = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY)
        img = (
            torch.from_numpy(gray).float().div(255.0)
            .unsqueeze(0).unsqueeze(0).to(self._device)
        )

        with torch.no_grad():
            if self._dla_active:
                feats0 = self._extract_dla(img)
                feats1 = self._extract_dla(img)   # same frame → worst-case match count
            else:
                feats0 = self._extractor.extract(img)
                feats1 = self._extractor.extract(img)

            # Matcher in FP16 — tiny matmuls; autocast keeps it on Tensor Cores.
            if self._device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    matches01 = self._matcher({"image0": feats0, "image1": feats1})
            else:
                matches01 = self._matcher({"image0": feats0, "image1": feats1})

        out = matches01.get("matches", matches01.get("matches0"))
        if out is not None and hasattr(out, "cpu"):
            return [out.cpu().numpy()]
        return [np.empty((0, 2), dtype=np.int32)]

    def _extract_dla(self, img) -> Dict[str, Any]:
        """Run the DLA backbone, then SuperPoint post-processing on the GPU.

        Mirrors lightglue.SuperPoint's keypoint head so the matcher receives
        the same feats dict it expects: keypoints [1,N,2], keypoint_scores
        [1,N], descriptors [1,N,256], image_size [1,2].
        """
        import torch
        import torch.nn.functional as F

        max_kpts = self.config.get("max_keypoints", 512)
        nms_radius = int(self.config.get("nms_radius", 4))
        thresh = float(self.config.get("detection_threshold", 0.0005))

        # --- DLA backbone (ONNX/TRT EP). Inputs/outputs cross PCIe-less SoC mem.
        np_img = img.detach().cpu().numpy().astype(np.float32)
        scores_logits, desc_raw = self._ort_session.run(
            ["scores_logits", "desc_raw"], {"image": np_img}
        )
        scores_logits = torch.from_numpy(scores_logits).to(self._device)
        desc_raw = torch.from_numpy(desc_raw).to(self._device)

        # --- GPU post-processing (dynamic shapes — cannot live on the DLA) ---
        # scores: softmax over 65 channels, drop dustbin, depth-to-space ×8.
        scores = F.softmax(scores_logits, dim=1)[:, :-1]
        b, _, hc, wc = scores.shape
        scores = (
            scores.permute(0, 2, 3, 1)
            .reshape(b, hc, wc, 8, 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(b, hc * 8, wc * 8)
        )
        scores = _simple_nms(scores, nms_radius)

        # top-k keypoints (single image in the benchmark, b == 1)
        s = scores[0]
        ys, xs = torch.where(s > thresh)
        kpt_scores = s[ys, xs]
        if kpt_scores.numel() > max_kpts:
            kpt_scores, idx = torch.topk(kpt_scores, max_kpts)
            ys, xs = ys[idx], xs[idx]
        keypoints = torch.stack([xs, ys], dim=-1).float()      # (N, 2) x,y

        # descriptors: normalize, bilinear-sample at keypoints.
        descriptors = F.normalize(desc_raw, p=2, dim=1)
        descriptors = self._sample_descriptors(keypoints[None], descriptors, 8)

        _, _, H, W = img.shape
        return {
            "keypoints": keypoints[None],                       # [1, N, 2]
            "keypoint_scores": kpt_scores[None],                # [1, N]
            "descriptors": descriptors.transpose(1, 2),         # [1, N, 256]
            "image_size": torch.tensor([[W, H]], device=self._device).float(),
        }

    @staticmethod
    def _sample_descriptors(keypoints, descriptors, s: int):
        """Bilinear-sample dense descriptors at keypoint locations (SuperPoint)."""
        import torch
        import torch.nn.functional as F

        b, c, h, w = descriptors.shape
        kpts = keypoints - s / 2 + 0.5
        kpts = kpts / torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                                   device=kpts.device)
        kpts = kpts * 2 - 1                                      # → [-1, 1] grid
        descriptors = F.grid_sample(
            descriptors, kpts.view(b, 1, -1, 2),
            mode="bilinear", align_corners=True,
        )
        descriptors = F.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def backend_name(self) -> str:
        dev = (self._device or "unknown").upper()
        path = "SuperPoint@DLA+LightGlue@GPU/FP16" if self._dla_active else "FullGPU"
        return f"LightGlue-DLA[{path}]/{dev}"
