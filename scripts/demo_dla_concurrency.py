#!/usr/bin/env python3
"""Demo: does offloading SuperPoint to the DLA free GPU headroom for a 2nd model?

The thesis from docs/DLA_SUPERPOINT_PROTOTYPE.md is that the DLA win is a
*system* win: with the SuperPoint conv backbone on the NVDLA, the GPU stops
doing those convolutions, so a second GPU model (a detector / classifier) can
run alongside feature-matching with less contention.

This script makes that measurable. It runs three scenarios on a shared synthetic
frame stream, sampling GPU utilization (HardwareMonitor / tegrastats) throughout:

  A. GPU model ALONE                         → baseline GPU%, baseline FPS
  B. GPU model + stock LightGlue (all-GPU)   → both contend for the GPU
  C. GPU model + DLA pipeline (backbone@DLA) → backbone offloaded to the NVDLA

The headline comparison is B vs C:
  * C should show LOWER combined GPU% than B (convolutions left the GR3D engine).
  * The GPU model should keep MORE of its standalone FPS in C than in B
    (less GPU contention), while feature-matching FPS holds.

Off-Jetson (no DLA) the pipeline falls back to GPU, so C ≈ B — the script says so.

Usage:
  python scripts/demo_dla_concurrency.py
  python scripts/demo_dla_concurrency.py --gpu-model resnet50_v1 --duration 10
  python scripts/demo_dla_concurrency.py --gpu-model rf_detr_base
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.benchmark.hardware_monitor import HardwareMonitor
from src.models.downloader import ensure_model
from src.models.registry import build_model
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure GPU headroom freed by offloading SuperPoint to the DLA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--gpu-model", default="mobilenet_v2",
                   help="The co-tenant GPU model run alongside feature matching "
                        "(default: mobilenet_v2). Try resnet50_v1 or rf_detr_base.")
    p.add_argument("--dla-feature-model", default="superpoint_lightglue_dla_pipeline",
                   help="Overlapped DLA pipeline backend (scenario C).")
    p.add_argument("--baseline-feature-model", default="superpoint_lightglue",
                   help="Stock all-GPU feature matcher (scenario B).")
    p.add_argument("--duration", type=float, default=8.0,
                   help="Seconds to run each scenario (default: 8).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--providers", nargs="+", default=None,
                   help="ORT providers in priority order.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load(name, models, models_root, providers):
    cfg = next((m for m in models if m["name"] == name), None)
    if cfg is None:
        raise KeyError(f"model '{name}' not found in models config")
    path = ensure_model(cfg, models_root=models_root)
    model = build_model(cfg, path, ort_providers=providers)
    model.warmup(cfg.get("warmup_runs", 5))
    return model


class _Worker(threading.Thread):
    """Hammers model.infer(frame) in a loop until stopped; counts iterations."""

    def __init__(self, model, frame):
        super().__init__(daemon=True)
        self.model = model
        self.frame = frame
        self._stop = threading.Event()
        self.iters = 0
        self.err = None

    def run(self):
        try:
            while not self._stop.is_set():
                self.model.infer(self.frame)
                self.iters += 1
        except Exception as exc:  # noqa: BLE001
            self.err = exc

    def stop(self):
        self._stop.set()


def run_scenario(label, models_with_frames, monitor, duration):
    """Run the given models concurrently (one thread each) for `duration` seconds.

    Returns (fps_per_model: dict, hw_summary: dict).
    """
    workers = [(_Worker(m, f), name) for name, m, f in models_with_frames]

    monitor.start()
    t0 = time.perf_counter()
    for w, _ in workers:
        w.start()
    time.sleep(duration)
    for w, _ in workers:
        w.stop()
    for w, _ in workers:
        w.join(timeout=30)
    elapsed = time.perf_counter() - t0
    monitor.stop()
    hw = monitor.summary()

    fps = {}
    for w, name in workers:
        if w.err is not None:
            logger.error("[%s] worker %s errored: %s", label, name, w.err)
            fps[name] = float("nan")
        else:
            fps[name] = w.iters / elapsed
    return fps, hw


def _fmt(v, fmt=".1f", fallback="—"):
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


def main():
    args = parse_args()
    set_log_level("DEBUG" if args.verbose else "INFO")

    with open(args.models_config) as f:
        models = yaml.safe_load(f).get("models", [])
    models_root = Path(args.models_dir)

    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    # ── load models (each scenario reuses these) ──────────────────────────────
    logger.info("Loading GPU co-tenant model: %s", args.gpu_model)
    gpu_model = _load(args.gpu_model, models, models_root, args.providers)

    logger.info("Loading baseline (all-GPU) feature matcher: %s", args.baseline_feature_model)
    try:
        baseline_feat = _load(args.baseline_feature_model, models, models_root, args.providers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Baseline feature model unavailable (%s) — skipping scenario B.", exc)
        baseline_feat = None

    logger.info("Loading DLA pipeline feature matcher: %s", args.dla_feature_model)
    dla_feat = _load(args.dla_feature_model, models, models_root, args.providers)
    dla_mode = getattr(getattr(dla_feat, "_extractor", None), "mode", "?")
    logger.info("DLA pipeline extractor mode = %s", dla_mode)

    monitor = HardwareMonitor(interval_ms=100)
    results = {}

    # ── A: GPU model alone ────────────────────────────────────────────────────
    logger.info("── Scenario A: %s alone (%.0fs) ──", args.gpu_model, args.duration)
    results["A"] = run_scenario("A", [(args.gpu_model, gpu_model, frame)], monitor, args.duration)

    # ── B: GPU model + stock all-GPU LightGlue ────────────────────────────────
    if baseline_feat is not None:
        logger.info("── Scenario B: %s + %s (all-GPU) ──", args.gpu_model, args.baseline_feature_model)
        results["B"] = run_scenario(
            "B",
            [(args.gpu_model, gpu_model, frame),
             (args.baseline_feature_model, baseline_feat, frame)],
            monitor, args.duration,
        )

    # ── C: GPU model + DLA pipeline ───────────────────────────────────────────
    logger.info("── Scenario C: %s + %s (backbone@DLA) ──", args.gpu_model, args.dla_feature_model)
    results["C"] = run_scenario(
        "C",
        [(args.gpu_model, gpu_model, frame),
         (args.dla_feature_model, dla_feat, frame)],
        monitor, args.duration,
    )

    _report(args, results, dla_mode, monitor.backend)


def _report(args, results, dla_mode, monitor_backend):
    bar = "=" * 78
    print(f"\n{bar}")
    print("  DLA Concurrency Demo — does the NVDLA free GPU headroom?")
    print(f"  GPU co-tenant: {args.gpu_model}   |   DLA extractor mode: {dla_mode}")
    print(f"  Monitor: {monitor_backend}   |   {args.duration:.0f}s/scenario   "
          f"frame: {args.width}x{args.height}")
    print(bar)

    hdr = f"  {'Scenario':<44}  {'GPU% avg':>8}  {'GPU% pk':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    def gpu_pct(scn):
        return results[scn][1].get("gpu_util_avg_pct")

    def row(scn, desc):
        if scn not in results:
            return
        fps, hw = results[scn]
        print(f"  {desc:<44}  {_fmt(hw.get('gpu_util_avg_pct')):>8}  "
              f"{_fmt(hw.get('gpu_util_peak_pct')):>7}")
        for name, v in fps.items():
            print(f"      └ {name:<38} {_fmt(v):>6} fps")

    row("A", f"A. {args.gpu_model} alone")
    row("B", f"B. + {args.baseline_feature_model} (all-GPU)")
    row("C", f"C. + {args.dla_feature_model} (DLA)")

    # ── interpretation ────────────────────────────────────────────────────────
    print("  " + "─" * (len(hdr) - 2))
    if "B" in results and "C" in results:
        gb, gc = gpu_pct("B"), gpu_pct("C")
        fb = results["B"][0].get(args.gpu_model)
        fc = results["C"][0].get(args.gpu_model)
        if gb is not None and gc is not None:
            delta = gb - gc
            print(f"  Δ GPU% (B→C): {_fmt(gb)} → {_fmt(gc)}  "
                  f"({'-' if delta >= 0 else '+'}{_fmt(abs(delta))} pts)")
        if fb and fc and fb == fb and fc == fc:  # not NaN
            keep = (fc / fb - 1.0) * 100.0 if fb else 0.0
            print(f"  {args.gpu_model} FPS under contention (B→C): "
                  f"{_fmt(fb)} → {_fmt(fc)}  ({'+' if keep >= 0 else ''}{_fmt(keep)}%)")
        if dla_mode != "trt-dla":
            print("  NOTE: DLA path not active (mode != trt-dla) — C≈B is expected.")
            print("        Run on a Jetson Orin NX with TensorRT for the real result.")
        else:
            print("  Lower GPU% + retained co-tenant FPS in C ⇒ the DLA freed GPU headroom.")
    else:
        print("  (Scenario B unavailable — install lightglue for the full comparison.)")
    print(bar + "\n")


if __name__ == "__main__":
    main()
