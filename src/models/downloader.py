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
    """Return local path to the model file, downloading if necessary."""
    root = models_root or MODELS_DIR
    local_path = root / Path(config["local_path"]).name

    if local_path.exists():
        logger.info("%s: found at %s (skipping download)", config["name"], local_path)
        return local_path

    source = config.get("source", "url")

    if source == "url":
        url = config.get("url")
        if not url:
            raise ValueError(f"Model '{config['name']}' has source=url but no 'url' field.")
        _download_url(url, local_path)

    elif source == "torchvision":
        _download_from_torchvision(config, local_path)

    elif source == "ultralytics":
        _download_from_ultralytics(config, local_path)

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
