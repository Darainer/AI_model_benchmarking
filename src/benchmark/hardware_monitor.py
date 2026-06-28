"""Background hardware stats sampler for GPU and memory bandwidth.

Auto-detects backend at startup:
  tegrastats  — Jetson devices (EMC % → GB/s via DDR bus formula)
  nvidia-smi  — discrete NVIDIA GPUs (loop query; GB/s via pynvml if available)
  none        — no-op, stats will be empty

Typical usage::
    mon = HardwareMonitor(interval_ms=200)
    mon.start()
    # ... run inference frames ...
    mon.stop()
    stats = mon.summary()
"""
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Jetson Nano / Orin use LPDDR4/5 on a 64-bit external memory bus.
# Overrideable via JETSON_BUS_BITS env var if you're on a different Jetson.
_JETSON_BUS_BITS = 64

# ── sample dataclass ───────────────────────────────────────────────────────

@dataclass
class HardwareSample:
    timestamp_s: float
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    # mem_bw_util_pct: EMC% on Jetson (JetPack ≤5), utilization.memory on nvidia-smi
    #   Not available on JetPack 6 / Orin inside a non-privileged container.
    mem_bw_util_pct: Optional[float] = None
    mem_bw_gb_s: Optional[float] = None
    gpu_clock_mhz: Optional[float] = None
    # CPU+GPU+CV power rail (mW); available on JetPack 6 Orin via tegrastats VDD fields
    gpu_power_mw: Optional[float] = None


# ── tegrastats parser ──────────────────────────────────────────────────────
_TS_RAM  = re.compile(r"RAM (\d+)/(\d+)MB")
# JetPack 5-: "GR3D X%@CLOCK"  JetPack 6/Orin: "GR3D_FREQ X%"
_TS_GR3D = re.compile(r"GR3D(?:_FREQ)? (\d+)%(?:@(\d+))?")
# EMC field dropped in JetPack 6 on Orin; kept for older Jetson boards
_TS_EMC  = re.compile(r"EMC(?:_FREQ)? (\d+)%@(\d+)")
# GPU-related power rail (CPU+GPU+CV subsystem); format: "VDD_CPU_GPU_CV CURmW/AVGmW"
_TS_VDD_GPU = re.compile(r"VDD_CPU_GPU_CV (\d+)mW/(\d+)mW")


def _parse_tegrastats(line: str) -> HardwareSample:
    s = HardwareSample(timestamp_s=time.monotonic())

    m = _TS_RAM.search(line)
    if m:
        # Jetson uses unified (CPU+GPU) memory — RAM field covers both
        s.gpu_mem_used_mb  = float(m.group(1))
        s.gpu_mem_total_mb = float(m.group(2))

    m = _TS_GR3D.search(line)
    if m:
        s.gpu_util_pct = float(m.group(1))
        if m.group(2):
            s.gpu_clock_mhz = float(m.group(2))

    m = _TS_VDD_GPU.search(line)
    if m:
        s.gpu_power_mw = float(m.group(1))  # instantaneous reading

    m = _TS_EMC.search(line)
    if m:
        emc_pct = float(m.group(1))
        emc_mhz = float(m.group(2))
        # DDR peak bandwidth = 2 × clock_MHz × (bus_bits / 8) / 1000  [GB/s]
        peak_gb_s = 2 * emc_mhz * (_JETSON_BUS_BITS / 8) / 1_000
        s.mem_bw_util_pct = emc_pct
        s.mem_bw_gb_s     = emc_pct / 100.0 * peak_gb_s

    return s


# ── nvidia-smi parser ──────────────────────────────────────────────────────
# Fields: utilization.gpu, utilization.memory, memory.used, memory.total, clocks.gr
def _parse_nvidiasmi(line: str, theoretical_gb_s: Optional[float]) -> Optional[HardwareSample]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        s = HardwareSample(timestamp_s=time.monotonic())
        s.gpu_util_pct     = float(parts[0])
        s.mem_bw_util_pct  = float(parts[1])   # % of cycles memory controller active
        s.gpu_mem_used_mb  = float(parts[2])
        s.gpu_mem_total_mb = float(parts[3])
        if len(parts) > 4 and parts[4] != "N/A":
            s.gpu_clock_mhz = float(parts[4])
        if theoretical_gb_s is not None:
            s.mem_bw_gb_s = s.mem_bw_util_pct / 100.0 * theoretical_gb_s
        return s
    except (ValueError, IndexError):
        return None


def _query_theoretical_bw_pynvml() -> Optional[float]:
    """Return theoretical memory bandwidth in GB/s using pynvml if available."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        bus_bits  = pynvml.nvmlDeviceGetMemoryBusWidth(h)
        max_mem_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        gb_s = 2 * max_mem_mhz * 1e6 * bus_bits / 8 / 1e9
        logger.info("pynvml: theoretical mem BW = %.1f GB/s (%d-bit bus @ %d MHz)",
                    gb_s, bus_bits, max_mem_mhz)
        return gb_s
    except Exception:
        return None


_EMC_CAP_PATH = "/sys/kernel/nvpmodel_clk_cap/emc"

def _measure_bw_cupy() -> Optional[float]:
    """Measure peak GPU memory bandwidth via a CuPy device-to-device memcpy benchmark.

    Uses 256 MB transfers × 5 iterations so the measurement takes ~100 ms and
    is stable enough for a one-time baseline. Returns GB/s or None on failure.
    """
    try:
        import cupy as cp  # type: ignore
        n = 256 * 1024 * 1024 // 4  # float32 elements → 256 MB
        src = cp.random.rand(n, dtype=cp.float32)
        dst = cp.empty_like(src)
        # Warmup pass
        cp.copyto(dst, src)
        cp.cuda.Stream.null.synchronize()
        # Timed passes
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            cp.copyto(dst, src)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        # Each iteration reads 256 MB and writes 256 MB → 512 MB per iter
        bw_gb_s = iters * 512 / 1024 / elapsed
        del src, dst
        return round(bw_gb_s, 1)
    except Exception:
        return None


# ── monitor ────────────────────────────────────────────────────────────────

class HardwareMonitor:
    """Polls GPU/memory stats in a background thread during inference."""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self._samples: List[HardwareSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self.backend, self._theoretical_gb_s = self._detect_backend()

    # ── detection ──────────────────────────────────────────────────────────
    def _detect_backend(self):
        # tegrastats wins on Jetson
        try:
            r = subprocess.run(["which", "tegrastats"],
                               capture_output=True, timeout=2)
            if r.returncode == 0:
                bw = _measure_bw_cupy()
                logger.info(
                    "HardwareMonitor: backend=tegrastats  measured_peak_bw=%s",
                    f"{bw:.1f} GB/s" if bw else "unavailable",
                )
                return "tegrastats", bw
        except Exception:
            pass

        # Fallback: nvidia-smi (discrete GPU)
        try:
            r = subprocess.run(["nvidia-smi", "-L"],
                               capture_output=True, timeout=3)
            if r.returncode == 0:
                bw = _query_theoretical_bw_pynvml()
                logger.info(
                    "HardwareMonitor: backend=nvidia-smi  theoretical_bw=%s",
                    f"{bw:.1f} GB/s" if bw else "unknown",
                )
                return "nvidia-smi", bw
        except Exception:
            pass

        logger.warning("HardwareMonitor: no GPU monitoring backend found — hw stats disabled")
        return "none", None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        with self._lock:
            self._samples.clear()

        if self.backend == "tegrastats":
            target = self._run_tegrastats
        elif self.backend == "nvidia-smi":
            target = self._run_nvidiasmi
        else:
            return

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ── sampling loops ─────────────────────────────────────────────────────
    def _run_tegrastats(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                sample = _parse_tegrastats(line)
                with self._lock:
                    self._samples.append(sample)
        except Exception as exc:
            logger.debug("tegrastats loop error: %s", exc)

    def _run_nvidiasmi(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,"
            "memory.used,memory.total,clocks.gr",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                sample = _parse_nvidiasmi(line, self._theoretical_gb_s)
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)
        except Exception as exc:
            logger.debug("nvidia-smi loop error: %s", exc)

    # ── statistics ─────────────────────────────────────────────────────────
    def summary(self) -> dict:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {"backend": self.backend, "samples": 0}

        def avg(vals):
            v = [x for x in vals if x is not None]
            return round(sum(v) / len(v), 2) if v else None

        def peak(vals):
            v = [x for x in vals if x is not None]
            return round(max(v), 2) if v else None

        return {
            "backend":                  self.backend,
            "samples":                  len(samples),
            "gpu_util_avg_pct":         avg(s.gpu_util_pct for s in samples),
            "gpu_util_peak_pct":        peak(s.gpu_util_pct for s in samples),
            "gpu_mem_used_avg_mb":      avg(s.gpu_mem_used_mb for s in samples),
            "gpu_mem_used_peak_mb":     peak(s.gpu_mem_used_mb for s in samples),
            "gpu_mem_total_mb":         peak(s.gpu_mem_total_mb for s in samples),
            "mem_bw_util_avg_pct":      avg(s.mem_bw_util_pct for s in samples),
            "mem_bw_util_peak_pct":     peak(s.mem_bw_util_pct for s in samples),
            "mem_bw_avg_gb_s":          avg(s.mem_bw_gb_s for s in samples),
            "mem_bw_peak_gb_s":         peak(s.mem_bw_gb_s for s in samples),
            # measured peak via CuPy benchmark (Jetson) or pynvml theoretical (discrete)
            "mem_bw_peak_capacity_gb_s": self._theoretical_gb_s,
            "gpu_clock_avg_mhz":        avg(s.gpu_clock_mhz for s in samples),
            "gpu_power_avg_mw":         avg(s.gpu_power_mw for s in samples),
            "gpu_power_peak_mw":        peak(s.gpu_power_mw for s in samples),
        }
