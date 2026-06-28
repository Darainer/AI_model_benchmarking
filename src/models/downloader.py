"""Download model checkpoints to the local models/ directory."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _download_url(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, file=sys.stdout
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def ensure_model(config: Dict[str, Any], models_root: Optional[Path] = None) -> Path:
    """Return local path to the model file, downloading / exporting if necessary."""
    root = models_root or MODELS_DIR
    root.mkdir(parents=True, exist_ok=True)
    local_path = root / Path(config["local_path"]).name

    source = config.get("source", "url")

    # lightglue produces TWO files (extractor + matcher) — check both
    if source == "lightglue":
        matcher_path = root / Path(config["local_path_matcher"]).name
        if local_path.exists() and matcher_path.exists():
            logger.info("%s: both ONNX files found, skipping export", config["name"])
            return local_path
        _download_lightglue(config, extractor_dest=local_path, matcher_dest=matcher_path)
        return local_path

    # system-installed runtime (cuVSLAM etc.) — create a sentinel, no download
    if source == "system":
        if not local_path.exists():
            local_path.touch()
        logger.info("%s: system-installed runtime, sentinel at %s", config["name"], local_path)
        return local_path

    if local_path.exists():
        logger.info("%s: found at %s (skipping download)", config["name"], local_path)
        return local_path

    if source == "url":
        url = config.get("url")
        if not url:
            raise ValueError(f"Model '{config['name']}' has source=url but no 'url' field.")
        _download_url(url, local_path)

    elif source == "torchvision":
        _download_from_torchvision(config, local_path)

    elif source == "ultralytics":
        _download_from_ultralytics(config, local_path)

    elif source == "huggingface":
        _download_from_huggingface(config, local_path)

    elif source == "smp":
        _download_from_smp(config, local_path)

    elif source == "rfdetr":
        _download_rfdetr(config, local_path)

    elif source == "local":
        raise FileNotFoundError(
            f"Model '{config['name']}' has source=local but file not found: {local_path}"
        )

    else:
        raise ValueError(f"Unknown source '{source}' for model '{config['name']}'.")

    return local_path


def _download_from_torchvision(config: Dict[str, Any], dest: Path) -> None:
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("torchvision required for source=torchvision") from exc

    name = config["name"]
    logger.info("%s: exporting from torchvision to ONNX at %s", name, dest)
    model_fn = getattr(torchvision.models, name, None)
    if model_fn is None:
        raise ValueError(f"torchvision has no model named '{name}'")

    model = model_fn(pretrained=True).eval()
    _, _, h, w = config["input_shape"]
    dummy = torch.zeros(1, 3, h, w)
    torch.onnx.export(
        model, dummy, str(dest),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info("%s: ONNX export done → %s", name, dest)


def _download_from_ultralytics(config: Dict[str, Any], dest: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics package required for source=ultralytics") from exc

    name = config["name"]
    logger.info("%s: downloading via ultralytics and exporting to ONNX", name)
    model = YOLO(f"{name}.pt")
    exported = model.export(format="onnx", imgsz=config["input_shape"][2])
    Path(exported).rename(dest)
    logger.info("%s: exported to %s", name, dest)


def _download_from_huggingface(config: Dict[str, Any], dest: Path) -> None:
    """Download a HuggingFace segmentation model and export it to ONNX.

    The HuggingFace model returns a dataclass output, so we wrap it in a thin
    nn.Module that exposes only the logits tensor — ONNX export requires plain
    tensors.

    Install deps: pip install transformers torch
    """
    try:
        import torch
        from transformers import AutoModelForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch required for source=huggingface. "
            "Install: pip install transformers torch"
        ) from exc

    model_id = config["hf_model_id"]
    name = config["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%s: downloading %s from HuggingFace Hub", name, model_id)

    hf_model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
    hf_model.eval()

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.m(pixel_values=pixel_values).logits

    wrapper = _LogitsWrapper(hf_model)
    _, _, h, w = config["input_shape"]
    dummy = torch.zeros(1, 3, h, w)

    logger.info("%s: exporting to ONNX → %s (this may take a minute)", name, dest)
    torch.onnx.export(
        wrapper,
        dummy,
        str(dest),
        opset_version=12,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    )
    logger.info("%s: HuggingFace ONNX export done", name)


def _download_from_smp(config: Dict[str, Any], dest: Path) -> None:
    """Build a segmentation_models_pytorch model and export it to ONNX.

    The model uses ImageNet-pretrained weights for the encoder so it can be
    used directly for transfer learning or as a timing baseline.

    Install deps: pip install segmentation-models-pytorch torch
    """
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation_models_pytorch and torch required for source=smp. "
            "Install: pip install segmentation-models-pytorch torch"
        ) from exc

    arch = config.get("smp_arch", "Unet")
    encoder = config.get("smp_encoder", "resnet34")
    weights = config.get("smp_encoder_weights", "imagenet")
    classes = config.get("smp_classes", 21)
    name = config["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    model_cls = getattr(smp, arch, None)
    if model_cls is None:
        raise ValueError(f"segmentation_models_pytorch has no architecture '{arch}'")

    logger.info(
        "%s: building smp.%s(encoder=%s, weights=%s, classes=%d)",
        name, arch, encoder, weights, classes,
    )
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=classes,
    ).eval()

    _, _, h, w = config["input_shape"]
    dummy = torch.zeros(1, 3, h, w)

    logger.info("%s: exporting to ONNX → %s", name, dest)
    torch.onnx.export(
        model,
        dummy,
        str(dest),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info("%s: smp ONNX export done", name)


def _download_lightglue(
    config: Dict[str, Any],
    extractor_dest: Path,
    matcher_dest: Path,
) -> None:
    """Export SuperPoint (extractor) and LightGlue (matcher) to ONNX.

    Requires the full LightGlue package with onnx support:
        pip install git+https://github.com/cvg/LightGlue.git

    The `lightglue.onnx` submodule (OnnxSuperPoint / OnnxLightGlue) provides
    ONNX-exportable wrapper classes that handle dynamic keypoint counts.
    """
    try:
        import torch
        from lightglue.onnx import OnnxSuperPoint, OnnxLightGlue  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LightGlue onnx module not found.\n"
            "Install: pip install git+https://github.com/cvg/LightGlue.git\n"
            "(The plain 'pip install lightglue' may not include the onnx submodule.)"
        ) from exc

    name        = config["name"]
    max_kpts    = config.get("max_keypoints", 512)
    _, _, h, w  = config.get("input_shape", [1, 1, 480, 640])

    extractor_dest.parent.mkdir(parents=True, exist_ok=True)

    # ── SuperPoint ─────────────────────────────────────────────────────────
    logger.info("%s: exporting SuperPoint → %s", name, extractor_dest)
    sp = OnnxSuperPoint(max_num_keypoints=max_kpts)
    dummy_img = torch.zeros(1, 1, h, w)
    torch.onnx.export(
        sp, dummy_img, str(extractor_dest),
        opset_version=17,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes={"image": {2: "height", 3: "width"}},
    )
    logger.info("%s: SuperPoint export done", name)

    # ── LightGlue ──────────────────────────────────────────────────────────
    logger.info("%s: exporting LightGlue → %s", name, matcher_dest)
    lg = OnnxLightGlue(extractor_type="superpoint")
    n           = max_kpts
    dummy_kpts  = torch.zeros(1, n, 2)
    dummy_desc  = torch.zeros(1, n, 256)
    dummy_size  = torch.tensor([[h, w]], dtype=torch.float32)
    torch.onnx.export(
        lg,
        (dummy_kpts, dummy_kpts, dummy_desc, dummy_desc, dummy_size, dummy_size),
        str(matcher_dest),
        opset_version=17,
        input_names=["kpts0", "kpts1", "desc0", "desc1", "image0_size", "image1_size"],
        output_names=["matches0", "matches1", "mscores0", "mscores1"],
        dynamic_axes={
            "kpts0": {1: "n_kpts0"}, "kpts1": {1: "n_kpts1"},
            "desc0": {1: "n_kpts0"}, "desc1": {1: "n_kpts1"},
        },
    )
    logger.info("%s: LightGlue export done", name)


def _download_rfdetr(config: Dict[str, Any], dest: Path) -> None:
    """Download RF-DETR pretrained weights and export to ONNX.

    Requires: pip install rfdetr
    RF-DETR (Roboflow Detection Transformer) is a real-time DETR-based
    object detector.
    """
    try:
        from rfdetr import RFDETRBase, RFDETRLarge  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rfdetr not installed. Install: pip install rfdetr"
        ) from exc

    import glob, shutil, tempfile

    name = config["name"]
    size = config.get("rfdetr_size", "base")
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("%s: downloading RF-DETR-%s weights and exporting to ONNX", name, size)
    model = RFDETRBase() if size == "base" else RFDETRLarge()

    with tempfile.TemporaryDirectory() as tmpdir:
        # rfdetr.export() signature varies by version; try common forms
        try:
            model.export(output_dir=tmpdir)
        except TypeError:
            try:
                model.export(tmpdir)
            except TypeError:
                model.export()

        onnx_files = glob.glob(f"{tmpdir}/**/*.onnx", recursive=True)
        if not onnx_files:
            onnx_files = glob.glob("rf_detr*.onnx") + glob.glob("*.onnx")
        if not onnx_files:
            raise RuntimeError(
                "RF-DETR export produced no ONNX file. "
                "Check your rfdetr version or export manually."
            )
        shutil.copy(onnx_files[0], dest)

    logger.info("%s: RF-DETR ONNX exported → %s", name, dest)
