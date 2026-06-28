#!/usr/bin/env python3
"""Download all (or selected) model checkpoints defined in configs/models.yaml."""
import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.models.downloader import ensure_model
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Download model checkpoints")
    p.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to models YAML config (default: configs/models.yaml)",
    )
    p.add_argument(
        "--models-dir",
        default="models",
        help="Local directory to store downloaded models (default: models/)",
    )
    p.add_argument(
        "--filter",
        nargs="+",
        metavar="NAME",
        help="Only download models whose name matches one of these strings",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        set_log_level("DEBUG")
    else:
        set_log_level("INFO")

    config_path = Path(args.models_config)
    if not config_path.exists():
        logger.error("Models config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    models = data.get("models", [])
    if args.filter:
        models = [m for m in models if m["name"] in args.filter]
        if not models:
            logger.error("No models matched filter: %s", args.filter)
            sys.exit(1)

    models_root = Path(args.models_dir)
    ok = []
    failed = []

    for cfg in models:
        try:
            path = ensure_model(cfg, models_root=models_root)
            logger.info("✓  %s → %s", cfg["name"], path)
            ok.append(cfg["name"])
        except Exception as exc:
            logger.error("✗  %s: %s", cfg["name"], exc)
            failed.append(cfg["name"])

    print(f"\nDownloaded: {len(ok)}  Failed: {len(failed)}")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
