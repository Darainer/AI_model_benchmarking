#!/usr/bin/env python3
"""Per-stage profiler for the SuperPoint + LightGlue feature-matching pipeline.

LightGlue is by far the heaviest model in the benchmark, and its end-to-end
``infer()`` latency bundles CPU preprocessing, host↔device copies, and the GPU
network together. This script breaks that single number into its stages so you
can see whether the cost is **CPU preprocess** or **GPU compute**:

    preprocess_cpu  cv2.resize + BGR→gray + build CPU float tensor   (CPU)
    h2d             host → device copy (.to(cuda))                   (PCIe/iGPU)
    extract_gpu     SuperPoint ×2                                    (GPU)
    match_gpu       LightGlue                                        (GPU)
    d2h             device → host copy of match indices (.cpu())     (PCIe/iGPU)

GPU stages are timed with ``torch.cuda.Event`` and every stage boundary is
``torch.cuda.synchronize()``-fenced, so async kernel launches are charged to the
stage that actually runs them (a plain wall-clock timer would otherwise bill all
GPU work to whatever stage finally touches the result on the host).

Usage:
  # synthetic frame (pure timing, no I/O)
  python scripts/profile_lightglue.py

  # real image — gives realistic keypoint/match counts
  python scripts/profile_lightglue.py --source path/to/frame.jpg

  # first frame of a video
  python scripts/profile_lightglue.py --source clip.mp4

  python scripts/profile_lightglue.py --runs 200 --warmup 20
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.models.backends.feature_matching_backend import LightGlueModel
from src.models.downloader import ensure_model
from src.models.registry import build_model
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def _percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def _load_frame(source, h, w):
    """Return a uint8 BGR frame: from an image/video file if given, else synthetic."""
    import cv2

    if not source:
        # Match the benchmark's synthetic input — random uint8 BGR.
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"--source not found: {source}")

    img = cv2.imread(str(p))
    if img is not None:
        return img

    # Not a still image — try to grab the first video frame.
    cap = cv2.VideoCapture(str(p))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read an image or video frame from {source}")
    return frame


def _find_config(models_config_path, name):
    with open(models_config_path) as f:
        models = yaml.safe_load(f).get("models", [])
    if name:
        for m in models:
            if m["name"] == name:
                return m
        raise SystemExit(f"No model named '{name}' in {models_config_path}")
    for m in models:
        if m.get("backend") == "lightglue":
            return m
    raise SystemExit(f"No backend: lightglue model found in {models_config_path}")


class _Stopwatch:
    """Times stages, preferring CUDA events on GPU and perf_counter on CPU."""

    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        if use_cuda:
            import torch
            self._torch = torch

    def sync(self):
        if self.use_cuda:
            self._torch.cuda.synchronize()

    def time(self, fn):
        """Run fn(), return (result, elapsed_ms) with the stage properly fenced."""
        if self.use_cuda:
            start = self._torch.cuda.Event(enable_timing=True)
            end = self._torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            self._torch.cuda.synchronize()
            return result, start.elapsed_time(end)
        t0 = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - t0) * 1e3


def profile(model: LightGlueModel, frame, runs, warmup):
    import torch

    use_cuda = torch.cuda.is_available() and model.device == "cuda"
    sw = _Stopwatch(use_cuda)

    # Warmup (not measured) — lets cuDNN/TRT autotune and the clocks ramp.
    for _ in range(warmup):
        model.infer(frame)
    sw.sync()

    stages = {k: [] for k in ("preprocess_cpu", "h2d", "extract_gpu", "match_gpu", "d2h")}
    last_match_count = 0

    for _ in range(runs):
        sw.sync()
        cpu_img, ms = sw.time(lambda: model.preprocess_cpu(frame))
        stages["preprocess_cpu"].append(ms)

        img, ms = sw.time(lambda: model.to_device(cpu_img))
        stages["h2d"].append(ms)

        feats, ms = sw.time(lambda: model.extract(img))
        stages["extract_gpu"].append(ms)
        feats0, feats1 = feats

        matches01, ms = sw.time(lambda: model.match(feats0, feats1))
        stages["match_gpu"].append(ms)

        out, ms = sw.time(lambda: model.postprocess(matches01))
        stages["d2h"].append(ms)
        last_match_count = int(out[0].shape[0]) if out and out[0].size else 0

    return stages, use_cuda, last_match_count


def _report(stages, use_cuda, match_count, model, runs):
    order = ["preprocess_cpu", "h2d", "extract_gpu", "match_gpu", "d2h"]
    kind = {
        "preprocess_cpu": "CPU",
        "h2d": "copy",
        "extract_gpu": "GPU",
        "match_gpu": "GPU",
        "d2h": "copy",
    }
    avgs = {k: float(np.mean(stages[k])) for k in order}
    total = sum(avgs.values())

    print(f"\n{'='*68}")
    print(f"  LightGlue per-stage profile  [{model.backend_name()}]")
    print(f"  runs={runs}  matches/frame≈{match_count}  "
          f"timing={'CUDA events' if use_cuda else 'wall clock (CPU)'}")
    print(f"{'='*68}")
    print(f"  {'stage':<16}{'kind':<6}{'avg ms':>9}{'p50':>9}{'p95':>9}{'% total':>9}")
    print("  " + "─" * 64)
    for k in order:
        print(f"  {k:<16}{kind[k]:<6}"
              f"{avgs[k]:>9.3f}{_percentile(stages[k], 50):>9.3f}"
              f"{_percentile(stages[k], 95):>9.3f}"
              f"{(100 * avgs[k] / total if total else 0):>8.1f}%")
    print("  " + "─" * 64)
    print(f"  {'TOTAL infer':<22}{total:>9.3f} ms  "
          f"({1000.0 / total if total else 0:.1f} inf/s)")

    cpu_ms = avgs["preprocess_cpu"]
    copy_ms = avgs["h2d"] + avgs["d2h"]
    gpu_ms = avgs["extract_gpu"] + avgs["match_gpu"]
    print(f"\n  Where the time goes:")
    print(f"    CPU preprocess : {cpu_ms:>8.3f} ms  ({100*cpu_ms/total if total else 0:.1f}%)")
    print(f"    H2D + D2H copy : {copy_ms:>8.3f} ms  ({100*copy_ms/total if total else 0:.1f}%)")
    print(f"    GPU compute    : {gpu_ms:>8.3f} ms  ({100*gpu_ms/total if total else 0:.1f}%)")
    verdict = "GPU-compute" if gpu_ms >= max(cpu_ms, copy_ms) else (
        "CPU-preprocess" if cpu_ms >= copy_ms else "host↔device copy")
    print(f"    → dominated by: {verdict}")
    if not use_cuda:
        print("\n  NOTE: CUDA not active — these are CPU-only numbers. Run on the "
              "Jetson (or a CUDA host) for a real CPU-vs-GPU split.")
    print(f"{'='*68}\n")


def main():
    p = argparse.ArgumentParser(
        description="Profile the SuperPoint+LightGlue pipeline stage by stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--name", default=None,
                   help="Model name in the config (default: first lightglue backend)")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--source", default=None,
                   help="Image or video file; omitted → synthetic random frame")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--width", type=int, help="Override input width")
    p.add_argument("--height", type=int, help="Override input height")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    set_log_level("DEBUG" if args.verbose else "INFO")

    cfg = _find_config(Path(args.models_config), args.name)
    if args.width:
        cfg["input_shape"][3] = args.width
    if args.height:
        cfg["input_shape"][2] = args.height

    model_path = ensure_model(cfg, models_root=Path(args.models_dir))
    model = build_model(cfg, model_path)
    if not isinstance(model, LightGlueModel):
        raise SystemExit(
            f"Model '{cfg['name']}' uses backend '{cfg.get('backend')}', not lightglue. "
            "This profiler targets the LightGlue stage breakdown.")

    _, _, h, w = cfg["input_shape"]
    frame = _load_frame(args.source, h, w)

    logger.info("Profiling %s on %s (%d runs, %d warmup, frame=%dx%d, source=%s)",
                cfg["name"], model.device, args.runs, args.warmup, w, h,
                args.source or "synthetic")

    stages, use_cuda, match_count = profile(model, frame, args.runs, args.warmup)
    _report(stages, use_cuda, match_count, model, args.runs)


if __name__ == "__main__":
    main()
