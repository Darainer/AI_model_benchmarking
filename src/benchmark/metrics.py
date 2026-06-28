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

    def flat_dict(self) -> dict:
        """Flat dict suitable for CSV row — merges latency fields with hw stats."""
        d = {
            "model":           self.model_name,
            "backend":         self.backend,
            "task":            self.task,
            "frames":          self.frames_processed,
            "avg_latency_ms":  round(self.avg_latency_ms, 2),
            "p50_latency_ms":  round(self.p50_latency_ms, 2),
            "p95_latency_ms":  round(self.p95_latency_ms, 2),
            "p99_latency_ms":  round(self.p99_latency_ms, 2),
            "throughput_fps":  round(self.throughput_fps, 2),
            # hardware (None when monitor unavailable)
            "gpu_util_avg_pct":      self.hw.get("gpu_util_avg_pct"),
            "gpu_util_peak_pct":     self.hw.get("gpu_util_peak_pct"),
            "gpu_mem_used_avg_mb":   self.hw.get("gpu_mem_used_avg_mb"),
            "gpu_mem_used_peak_mb":  self.hw.get("gpu_mem_used_peak_mb"),
            "mem_bw_avg_gb_s":       self.hw.get("mem_bw_avg_gb_s"),
            "mem_bw_peak_gb_s":      self.hw.get("mem_bw_peak_gb_s"),
            "mem_bw_theoretical_gb_s": self.hw.get("mem_bw_theoretical_gb_s"),
        }
        return d


class Timer:
    """Context-manager wall-clock timer (perf_counter precision)."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1_000.0
