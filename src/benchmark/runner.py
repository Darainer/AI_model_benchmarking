"""Benchmark runner — feeds frames from the input pipeline into each model."""
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmark.metrics import RunMetrics, Timer, sample_gpu_stats
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
    """Download (if needed), load, and benchmark every model in the list."""
    out_cfg = pipeline_config.get("output", {})
    bench_cfg = pipeline_config.get("benchmark", {})
    results_dir = results_dir or Path(out_cfg.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    ort_providers = bench_cfg.get("onnx_providers", [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])

    all_metrics: List[RunMetrics] = []

    for model_cfg in models_config:
        name = model_cfg["name"]
        logger.info("─── %s ───", name)

        try:
            model_path = ensure_model(model_cfg, models_root=models_root)
            model = build_model(model_cfg, model_path, ort_providers=ort_providers)
        except Exception as exc:
            logger.error("Failed to load %s: %s", name, exc)
            continue

        warmup_runs = model_cfg.get("warmup_runs", 10)
        benchmark_runs = model_cfg.get("benchmark_runs", 100)

        logger.info("%s: warming up (%d passes)…", name, warmup_runs)
        model.warmup(warmup_runs)

        metrics = RunMetrics(
            model_name=name,
            backend=model.backend_name(),
            task=model_cfg["task"],
        )

        input_cfg = pipeline_config.get("input", {})
        frame_iter = open_source(input_cfg)

        logger.info("%s: benchmarking (%d passes)…", name, benchmark_runs)
        run_start = time.perf_counter()
        frames_seen = 0

        for frame_idx, frame in frame_iter:
            with Timer() as t:
                model.infer(frame)
            metrics.record(t.elapsed_ms)
            frames_seen += 1
            if frames_seen >= benchmark_runs:
                break

        metrics.wall_time_s = time.perf_counter() - run_start

        gpu = sample_gpu_stats()
        if gpu:
            metrics.gpu_memory_mb = gpu["gpu_memory_mb"]
            metrics.gpu_util_pct = gpu["gpu_util_pct"]

        metrics.print_summary()
        all_metrics.append(metrics)

    if out_cfg.get("csv", True):
        _save_csv(all_metrics, results_dir / "benchmark_results.csv")

    return all_metrics


def _save_csv(metrics_list: List[RunMetrics], path: Path) -> None:
    if not metrics_list:
        return
    rows = [m.summary_dict() for m in metrics_list]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results saved to %s", path)
