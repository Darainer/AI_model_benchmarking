"""Latency and throughput metrics collected during a benchmark run."""
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class RunMetrics:
    model_name: str
    backend: str
    task: str
    latencies_ms: List[float] = field(default_factory=list)
    frames_processed: int = 0
    wall_time_s: float = 0.0

    # Optional hardware counters (populated if nvidia-smi / tegrastats available)
    gpu_memory_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None

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

    def summary_dict(self) -> dict:
        return {
            "model": self.model_name,
            "backend": self.backend,
            "task": self.task,
            "frames": self.frames_processed,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "throughput_fps": round(self.throughput_fps, 2),
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_util_pct": self.gpu_util_pct,
        }

    def print_summary(self) -> None:
        d = self.summary_dict()
        print(f"\n{'='*60}")
        print(f"  Model   : {d['model']}")
        print(f"  Backend : {d['backend']}")
        print(f"  Task    : {d['task']}")
        print(f"  Frames  : {d['frames']}")
        print(f"  Latency : avg={d['avg_latency_ms']} ms  "
              f"p50={d['p50_latency_ms']} ms  "
              f"p95={d['p95_latency_ms']} ms  "
              f"p99={d['p99_latency_ms']} ms")
        print(f"  FPS     : {d['throughput_fps']}")
        if d["gpu_memory_mb"] is not None:
            print(f"  GPU Mem : {d['gpu_memory_mb']:.0f} MB")
        print(f"{'='*60}")


class Timer:
    """Simple wall-clock timer using time.perf_counter."""

    def __init__(self):
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def sample_gpu_stats() -> Optional[dict]:
    """Try to read GPU stats via nvidia-smi or tegrastats (non-blocking)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        mem_mb, util_pct = out.split(",")
        return {"gpu_memory_mb": float(mem_mb), "gpu_util_pct": float(util_pct)}
    except Exception:
        return None
