"""Latency, throughput, and hardware metrics for a single benchmark run."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class RunMetrics:
    model_name: str
    backend: str
    task: str
    latencies_ms: List[float] = field(default_factory=list)
    frames_processed: int = 0
    wall_time_s: float = 0.0
    run_timestamp: str = ""

    # Weight bytes the model must stream from DRAM per inference (FP16 params for
    # ONNX/TRT models, file size otherwise). Used for the effective-bandwidth roofline.
    model_bytes: Optional[int] = None

    # Populated by HardwareMonitor.summary() after inference completes
    hw: Dict[str, Any] = field(default_factory=dict)

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.frames_processed += 1

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def throughput_fps(self) -> float:
        return self.frames_processed / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def eff_mem_bw_gb_s(self) -> Optional[float]:
        """Lower-bound effective DRAM bandwidth = weight bytes streamed per inference
        / avg latency. Real lower bound (weights are read at least once per inference;
        activations add more), and the figure that matters on bandwidth-bound SoCs.
        Compare against hw['mem_bw_peak_capacity_gb_s'] (the measured device ceiling).

        NOTE: avg_latency_ms is timed around the full model.infer() call, which runs
        in-model preprocessing (resize / colour-convert / normalize) before the backend
        executes. So this denominator is end-to-end (preprocess + inference), not the
        isolated backend execution time — making the figure an *extra-conservative*
        lower bound. A model could be more bandwidth-bound at the kernel level than this
        number suggests; do not read a low value as proof a model is not bandwidth-bound."""
        if not self.model_bytes or not self.latencies_ms:
            return None
        avg_s = (self.avg_latency_ms / 1000.0)
        if avg_s <= 0:
            return None
        return round(self.model_bytes / avg_s / 1e9, 2)

    def flat_dict(self) -> dict:
        """Flat dict suitable for CSV row — merges latency fields with hw stats."""
        d = {
            "run_timestamp":   self.run_timestamp,
            "model":           self.model_name,
            "backend":         self.backend,
            "task":            self.task,
            "frames":          self.frames_processed,
            "avg_latency_ms":  round(self.avg_latency_ms, 2),
            "p50_latency_ms":  round(self.p50_latency_ms, 2),
            "p95_latency_ms":  round(self.p95_latency_ms, 2),
            "p99_latency_ms":  round(self.p99_latency_ms, 2),
            "throughput_fps":  round(self.throughput_fps, 2),
            # memory: weight bytes streamed/inference and the resulting effective DRAM BW.
            # eff_mem_bw uses end-to-end infer() latency (incl. in-model preprocessing),
            # so it is an extra-conservative lower bound — see eff_mem_bw_gb_s docstring.
            "model_weight_mb":      round(self.model_bytes / 1e6, 2) if self.model_bytes else None,
            "eff_mem_bw_gb_s":      self.eff_mem_bw_gb_s,
            # hardware (None when monitor unavailable)
            "gpu_util_avg_pct":      self.hw.get("gpu_util_avg_pct"),
            "gpu_util_peak_pct":     self.hw.get("gpu_util_peak_pct"),
            "gpu_mem_used_avg_mb":   self.hw.get("gpu_mem_used_avg_mb"),
            "gpu_mem_used_peak_mb":  self.hw.get("gpu_mem_used_peak_mb"),
            "mem_bw_avg_gb_s":       self.hw.get("mem_bw_avg_gb_s"),
            "mem_bw_peak_gb_s":      self.hw.get("mem_bw_peak_gb_s"),
            "mem_bw_peak_capacity_gb_s": self.hw.get("mem_bw_peak_capacity_gb_s"),
            "mem_bw_source":         self.hw.get("mem_bw_source"),
            "gpu_clock_avg_mhz":     self.hw.get("gpu_clock_avg_mhz"),
            "gpu_clock_peak_mhz":    self.hw.get("gpu_clock_peak_mhz"),
            "gpu_power_avg_mw":      self.hw.get("gpu_power_avg_mw"),
            "gpu_power_peak_mw":     self.hw.get("gpu_power_peak_mw"),
        }
        return d


class Timer:
    """Context-manager wall-clock timer (perf_counter precision)."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1_000.0
