"""Benchmark runner — feeds video/image frames into each model and reports stats."""
import csv
import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmark.hardware_monitor import HardwareMonitor
from src.benchmark.metrics import RunMetrics, Timer
from src.input.frame_source import open_source
from src.models.downloader import ensure_model
from src.models.registry import build_model
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_all(
    models_config: List[Dict[str, Any]],
    pipeline_config: Dict[str, Any],
    results_dir: Optional[Path] = None,
    models_root: Optional[Path] = None,
) -> List[RunMetrics]:
    """Download (if needed), load, warmup, and benchmark every model.

    For each model:
      1. Download / export checkpoint if not cached locally
      2. Load the backend and run warmup passes
      3. Start HardwareMonitor in background
      4. Feed frames from the configured source, time each inference
      5. Stop monitor, attach hw stats, print per-model summary
    End: print consolidated report table and save CSV.
    """
    out_cfg   = pipeline_config.get("output", {})
    bench_cfg = pipeline_config.get("benchmark", {})
    input_cfg = pipeline_config.get("input", {})

    results_dir = results_dir or Path(out_cfg.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    ort_providers = bench_cfg.get("onnx_providers", [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])

    monitor = HardwareMonitor(interval_ms=bench_cfg.get("monitor_interval_ms", 200))
    all_metrics: List[RunMetrics] = []
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model_cfg in models_config:
        name = model_cfg["name"]
        logger.info("─── %s ───", name)

        # Per-model provider override (e.g. to skip TRT EP for specific models)
        model_providers = model_cfg.get("onnx_providers", ort_providers)

        # ── load ────────────────────────────────────────────────────────
        try:
            model_path = ensure_model(model_cfg, models_root=models_root)
            model = build_model(model_cfg, model_path, ort_providers=model_providers)
        except Exception as exc:
            logger.error("Failed to load %s: %s", name, exc)
            continue

        warmup_runs    = model_cfg.get("warmup_runs", 10)
        benchmark_runs = model_cfg.get("benchmark_runs", 100)

        logger.info("%s: warmup (%d passes)…", name, warmup_runs)
        model.warmup(warmup_runs)

        metrics = RunMetrics(
            model_name=name,
            backend=model.backend_name(),
            task=model_cfg["task"],
            run_timestamp=run_ts,
        )

        # ── benchmark ────────────────────────────────────────────────────
        logger.info("%s: benchmarking (%d frames)…", name, benchmark_runs)
        monitor.start()
        run_start = time.perf_counter()

        for _, frame in open_source(input_cfg):
            with Timer() as t:
                model.infer(frame)
            metrics.record(t.elapsed_ms)
            if metrics.frames_processed >= benchmark_runs:
                break

        metrics.wall_time_s = time.perf_counter() - run_start
        monitor.stop()
        metrics.hw = monitor.summary()

        _print_model_summary(metrics)
        all_metrics.append(metrics)

    # ── consolidated report ──────────────────────────────────────────────
    if all_metrics:
        _print_report_table(all_metrics, input_cfg, monitor.backend)
        if out_cfg.get("csv", True):
            ts_file = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            _save_csv(all_metrics, results_dir / f"benchmark_{ts_file}.csv")

    return all_metrics


# ── per-model inline summary ───────────────────────────────────────────────

def _print_model_summary(m: RunMetrics) -> None:
    hw = m.hw
    bw_str = ""
    if hw.get("mem_bw_avg_gb_s") is not None:
        bw_str = (f"  BW      : avg {hw['mem_bw_avg_gb_s']:.1f} GB/s  "
                  f"peak {hw['mem_bw_peak_gb_s']:.1f} GB/s")
        if hw.get("mem_bw_peak_capacity_gb_s"):
            bw_str += f"  (capacity {hw['mem_bw_peak_capacity_gb_s']:.1f} GB/s)"

    print(f"\n{'='*64}")
    print(f"  {m.model_name}  [{m.backend}]  task={m.task}")
    print(f"  Latency : avg {m.avg_latency_ms:.1f} ms  "
          f"p50 {m.p50_latency_ms:.1f} ms  "
          f"p95 {m.p95_latency_ms:.1f} ms  "
          f"p99 {m.p99_latency_ms:.1f} ms")
    print(f"  FPS     : {m.throughput_fps:.1f}  ({m.frames_processed} frames in {m.wall_time_s:.1f}s)")
    if hw.get("gpu_util_avg_pct") is not None:
        print(f"  GPU     : avg {hw['gpu_util_avg_pct']:.0f}%  peak {hw['gpu_util_peak_pct']:.0f}%"
              + (f"  @ {hw['gpu_clock_avg_mhz']:.0f} MHz" if hw.get("gpu_clock_avg_mhz") else ""))
    if hw.get("gpu_mem_used_peak_mb") is not None:
        print(f"  Mem     : avg {hw['gpu_mem_used_avg_mb']:.0f} MB  "
              f"peak {hw['gpu_mem_used_peak_mb']:.0f} MB"
              + (f"  / {hw['gpu_mem_total_mb']:.0f} MB total" if hw.get("gpu_mem_total_mb") else ""))
    if bw_str:
        print(bw_str)
    if hw.get("gpu_power_avg_mw") is not None:
        print(f"  Power   : avg {hw['gpu_power_avg_mw']:.0f} mW  "
              f"peak {hw['gpu_power_peak_mw']:.0f} mW  (CPU+GPU+CV rail)")
    print(f"{'='*64}")


# ── consolidated end-of-run table ─────────────────────────────────────────

def _fmt(v, fmt=".1f", fallback="  —  "):
    return format(v, fmt) if v is not None else fallback

_COL_W = 22  # model name column width

def _print_report_table(
    metrics: List[RunMetrics],
    input_cfg: Dict[str, Any],
    monitor_backend: str,
) -> None:
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    src  = input_cfg.get("source", input_cfg.get("type", "synthetic"))
    bar  = "=" * 100

    print(f"\n{bar}")
    print(f"  Benchmark Report — {ts}")
    print(f"  Input  : {src}  |  Monitor: {monitor_backend}")
    print(bar)

    # header
    h = (f"  {'Model':<{_COL_W}}  {'Backend':<10}  {'Task':<14}  "
         f"{'Frms':>5}  {'avg ms':>7}  {'p95 ms':>7}  {'FPS':>6}  "
         f"{'GPU%':>5}  {'Mem MB':>7}  {'Pwr mW':>7}")
    print(h)
    print("  " + "─" * (len(h) - 2))

    for m in metrics:
        hw = m.hw
        print(
            f"  {m.model_name:<{_COL_W}}"
            f"  {m.backend:<10}"
            f"  {m.task:<14}"
            f"  {m.frames_processed:>5}"
            f"  {_fmt(m.avg_latency_ms):>7}"
            f"  {_fmt(m.p95_latency_ms):>7}"
            f"  {_fmt(m.throughput_fps):>6}"
            f"  {_fmt(hw.get('gpu_util_avg_pct'), '.0f'):>5}"
            f"  {_fmt(hw.get('gpu_mem_used_peak_mb'), '.0f'):>7}"
            f"  {_fmt(hw.get('gpu_power_avg_mw'), '.0f'):>7}"
        )

    # bandwidth footer (if available)
    capacity = next(
        (m.hw.get("mem_bw_peak_capacity_gb_s") for m in metrics if m.hw.get("mem_bw_peak_capacity_gb_s")),
        None,
    )
    if capacity:
        print("  " + "─" * (len(h) - 2))
        print(f"  Measured peak mem BW (CuPy d2d): {capacity:.1f} GB/s")

    print(bar + "\n")


# ── CSV export ─────────────────────────────────────────────────────────────

def _save_csv(metrics_list: List[RunMetrics], path: Path) -> None:
    rows = [m.flat_dict() for m in metrics_list]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results saved → %s", path)
