#!/usr/bin/env python3
"""Main entry point for the inference benchmark pipeline.

Examples:
  # MP4 video (most common use case — HW-decoded on Jetson via nvv4l2decoder)
  python scripts/run_benchmark.py --video /path/to/clip.mp4

  # Synthetic frames (no camera / file required — pure inference timing)
  python scripts/run_benchmark.py

  # CSI camera (Jetson ribbon connector)
  python scripts/run_benchmark.py --input-type csi_camera

  # USB camera
  python scripts/run_benchmark.py --input-type usb_camera --device /dev/video0

  # Directory of images
  python scripts/run_benchmark.py --input-type image_dir --source /path/to/images/

  # Run only specific models
  python scripts/run_benchmark.py --video clip.mp4 --filter mobilenet_v2 yolov8n

  # Force CPU (no GPU / no CUDA)
  python scripts/run_benchmark.py --providers CPUExecutionProvider
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.benchmark.runner import run_all
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference benchmark across multiple model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config files
    p.add_argument("--models-config", default="configs/models.yaml",
                   help="Path to models YAML (default: configs/models.yaml)")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml",
                   help="Path to pipeline YAML (default: configs/pipeline.yaml)")

    # Input overrides
    p.add_argument("--video", metavar="PATH",
                   help="MP4 / video file to benchmark against (shorthand for "
                        "--input-type file --source PATH)")
    p.add_argument("--input-type",
                   choices=["synthetic", "file", "image_dir", "usb_camera", "csi_camera"],
                   help="Override input.type from pipeline config")
    p.add_argument("--source",
                   help="Video file path or image directory (overrides pipeline config)")
    p.add_argument("--device", default="/dev/video0",
                   help="Camera device node (default: /dev/video0)")
    p.add_argument("--width", type=int, help="Override frame width")
    p.add_argument("--height", type=int, help="Override frame height")
    p.add_argument("--frames", type=int,
                   help="Max frames to process per model (overrides benchmark_runs)")

    # Backend
    p.add_argument("--providers", nargs="+",
                   metavar="PROVIDER",
                   default=None,
                   help="ORT providers in priority order "
                        "(e.g. TensorrtExecutionProvider CUDAExecutionProvider)")

    # Model filter
    p.add_argument("--filter", nargs="+", metavar="NAME",
                   help="Only benchmark models whose name matches one of these")

    # Output
    p.add_argument("--results-dir", default=None,
                   help="Directory to save results (default: ./results)")
    p.add_argument("--models-dir", default="models",
                   help="Directory containing downloaded model files")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_log_level("DEBUG" if args.verbose else "INFO")

    # Load configs
    models_cfg_path = Path(args.models_config)
    pipeline_cfg_path = Path(args.pipeline_config)

    if not models_cfg_path.exists():
        logger.error("Models config not found: %s", models_cfg_path)
        sys.exit(1)
    if not pipeline_cfg_path.exists():
        logger.error("Pipeline config not found: %s", pipeline_cfg_path)
        sys.exit(1)

    with open(models_cfg_path) as f:
        models_data = yaml.safe_load(f)
    with open(pipeline_cfg_path) as f:
        pipeline_cfg = yaml.safe_load(f)

    models = models_data.get("models", [])

    # Apply CLI overrides to pipeline config
    input_cfg = pipeline_cfg.setdefault("input", {})

    # --video is the preferred shorthand for MP4 benchmarking
    if args.video:
        input_cfg["type"] = "file"
        input_cfg["source"] = args.video
    if args.input_type:
        input_cfg["type"] = args.input_type
    if args.source:
        input_cfg["source"] = args.source
    if args.device:
        input_cfg["device"] = args.device
    if args.width:
        input_cfg["width"] = args.width
    if args.height:
        input_cfg["height"] = args.height

    bench_cfg = pipeline_cfg.setdefault("benchmark", {})
    if args.providers:
        bench_cfg["onnx_providers"] = args.providers

    # Apply per-model frame count override
    if args.frames:
        for m in models:
            m["benchmark_runs"] = args.frames

    # Filter models
    if args.filter:
        models = [m for m in models if m["name"] in args.filter]
        if not models:
            logger.error("No models matched filter: %s", args.filter)
            sys.exit(1)

    results_dir = Path(args.results_dir) if args.results_dir else None
    models_root = Path(args.models_dir)

    logger.info("Benchmarking %d model(s), input type: %s",
                len(models), input_cfg.get("type", "synthetic"))

    metrics = run_all(
        models_config=models,
        pipeline_config=pipeline_cfg,
        results_dir=results_dir,
        models_root=models_root,
    )

    if not metrics:
        logger.error("No models ran successfully.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Benchmark complete: {len(metrics)} model(s)")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
