# Jetson Hardware Comparison — Orin Family & Thor

This document covers the NVIDIA Jetson hardware generations relevant to this
benchmark suite, with estimated inference throughput for the models included in
`configs/models.yaml`.  All performance figures are estimates derived from
published hardware specs and published NVIDIA benchmark data; actual numbers
depend on JetPack version, power mode, thermal conditions, and TensorRT build.

---

## Platform Overview

| Module | GPU Arch | CUDA Cores | Tensor Cores | Memory | Bandwidth | DLA Cores | Peak TOPS | TDP |
|---|---|---|---|---|---|---|---|---|
| **Orin Nano 8 GB** | Ampere | 1024 | 32 | 8 GB LPDDR5 | 68 GB/s | 0 | 40 | 7–15 W |
| **Orin NX 8 GB** | Ampere | 1024 | 32 | 8 GB LPDDR5 | 68 GB/s | 2 | 70 | 10–25 W |
| **Orin NX 16 GB** | Ampere | 1024 | 32 | 16 GB LPDDR5 | 102 GB/s | 2 | 100 | 10–25 W |
| **AGX Orin 64 GB** *(EGX)* | Ampere | 2048 | 64 | 64 GB LPDDR5 | 205 GB/s | 2 | 275 | 15–60 W |
| **Thor** *(future)* | Blackwell | ~4096 est. | ~128 est. | ≥ 64 GB | ~900 GB/s | 2 | ~2000 | ~100 W |

> **TOPS definition:** "Tera-Operations Per Second" counts INT8 MAC operations
> across GPU **and** DLA.  GPU-only ONNX Runtime workloads (the default in this
> repo) use only the GPU column; DLA TOPS only matter when you route layers
> through the TensorRT DLA execution provider.

---

## Module Deep-Dives

### Orin Nano 8 GB — your current target

The entry-level Orin module.  The Nano line removes the two on-chip DLA
(Deep Learning Accelerator) engines, so all AI work lands on the 1024-core
Ampere GPU.  The 68 GB/s LPDDR5 bus is the main bottleneck for large models —
once a tensor doesn't fit in the L2 cache hierarchy, you pay the full DRAM
round-trip cost on every layer.

Key constraints:
- **No DLA** — TensorRT's `DLA execution provider` option has no effect; the
  runtime silently falls back to GPU.
- **Shared memory** — the 8 GB pool is split between the OS, CPU inference
  buffers, and GPU.  Running multiple large models concurrently can exhaust it.
- **68 GB/s bandwidth** — enough for classification and small detection models
  at real-time rates, but segmentation models (SegFormer-B0 at 512×512) will
  be noticeably memory-bound.
- Power envelope of 7–15 W makes this the right choice for battery-powered
  or thermally constrained platforms.

---

### Orin NX 8 GB

Same GPU die and memory capacity as the Nano 8 GB, but with two DLA cores
re-enabled and a higher power envelope.  For ONNX Runtime workloads the
**GPU performance is identical** to the Nano 8 GB — the extra TOPS number
(70 vs 40) comes entirely from the DLAs.

Where NX 8 GB beats Nano 8 GB:
- **DLA offload:** if you build a TensorRT engine with `--useDLACore=0` or use
  the `DlaExecutionProvider` in ORT, supported layers run on the DLA at 26 TOPS
  each while the GPU handles unsupported ops — effectively ~2× throughput for
  DLA-friendly CNNs (ResNet, MobileNet, EfficientNet-Lite).
- **CPU:** two additional Cortex-A78AE cores (8 vs 6) improve pre/post-
  processing parallelism — relevant for detection post-processing (NMS) and
  LightGlue match filtering.
- **Higher TDP ceiling** allows sustained compute without thermal throttle on
  dense workloads like cuVSLAM + detection running in parallel.

For this benchmarking repo (pure GPU path):
- Expect similar `avg_latency_ms` to Orin Nano 8 GB for all models.
- Add the `DlaExecutionProvider` to `onnx_providers` in `pipeline.yaml` to
  unlock the NX-specific advantage.

---

### Orin NX 16 GB

The 16 GB variant upgrades the memory bus to **102 GB/s** (1.5× the 8 GB
modules), which is the single most impactful difference for inference latency
on larger models.  The GPU core count is the same (1024 Ampere + 32 Tensor
Cores), so single-batch latency improvements come from memory bandwidth, not
compute throughput.

Expected improvements over Orin Nano 8 GB (GPU-only, FP16 TensorRT):

| Model | Nano 8 GB (est.) | NX 16 GB (est.) | Reason |
|---|---|---|---|
| MobileNetV2 | ~2 ms | ~2 ms | compute-bound, BW doesn't help |
| ResNet-50 | ~4 ms | ~4 ms | compute-bound |
| EfficientNet-Lite4 | ~5 ms | ~4 ms | slightly BW-bound |
| YOLOv8n (640×640) | ~8 ms | ~6 ms | BW-bound at larger input |
| SegFormer-B0 (512×512) | ~25 ms | ~17 ms | heavily BW-bound |
| UNet-ResNet34 (512×512) | ~30 ms | ~20 ms | heavily BW-bound |
| SuperPoint+LightGlue | ~15 ms | ~12 ms | two ONNX sessions, BW matters |
| cuVSLAM (480×640) | ~5 ms | ~4 ms | mostly compute |

The 16 GB address space gives headroom for larger individual models or
higher-resolution inputs without hitting OOM — relevant if you add bigger
segmentation backbones (e.g. SegFormer-B2/B5) to `configs/models.yaml`.

---

### AGX Orin 64 GB — EGX / Enterprise Edge

The AGX Orin is the top-of-stack embedded Orin module and the basis for the
NVIDIA EGX edge-server platforms.  It doubles the GPU to **2048 Ampere CUDA
cores / 64 Tensor Cores** and triples the memory bandwidth to **205 GB/s**
versus the Nano.

This platform closes the gap to a desktop GPU for many inference workloads:

| Capability | Nano 8 GB | AGX Orin 64 GB | Ratio |
|---|---|---|---|
| CUDA cores | 1024 | 2048 | 2× |
| Memory BW | 68 GB/s | 205 GB/s | 3× |
| Total TOPS (INT8) | 40 | 275 | ~7× |
| Max TDP | 15 W | 60 W | 4× |

Practical implications for this repo:
- All segmentation models (SegFormer, UNet) become real-time at 512×512.
- cuVSLAM can be paired with a stereo camera rig at full frame rate alongside
  a detection model — the AGX has headroom for concurrent pipelines.
- TensorRT engine build times are ~3× faster, reducing the first-run penalty.
- The 64 GB pool can accommodate very large individual models (e.g. full
  SegFormer-B5 or large detection transformers) that would OOM on Nano/NX.

The "EGX" designation refers to the AGX Orin being integrated into NVIDIA's
EGX edge-server reference designs (e.g. IGX Orin for industrial, AGX Orin
DevKit for development).  The underlying SoC is the same; the difference is
the carrier board, I/O, and certification (IEC 62368, IPC-A-610, etc.).

---

## Performance Scaling Summary

Normalised to Orin Nano 8 GB for GPU-only ONNX/TensorRT workloads:

```
Platform             BW mult   Compute mult   Expected FPS mult
─────────────────────────────────────────────────────────────────
Orin Nano 8 GB       1.0×       1.0×           1.0×  (baseline)
Orin NX   8 GB       1.0×       1.0×           1.0×  GPU-only
                                               +DLA offload → ~1.5–2×
Orin NX  16 GB       1.5×       1.0×           1.2–1.6× (BW-bound models)
AGX Orin 64 GB       3.0×       2.0×           2.5–5×  depending on model
```

Rule of thumb: **classification models are compute-bound** (ResNet, MobileNet
→ benefit from more CUDA cores); **segmentation and LightGlue are BW-bound**
(benefit more from higher memory bandwidth than from more cores).

---

## Thor — Next Generation (Future)

NVIDIA **Thor** is the successor SoC to Orin, built on the Blackwell GPU
microarchitecture.  It is positioned primarily for autonomous vehicles and
high-performance robotics edge computing.

Published figures (from NVIDIA GTC announcements, subject to revision before
GA release):

| Attribute | Thor |
|---|---|
| GPU architecture | Blackwell (successor to Ampere) |
| AI performance | ~2000 TOPS (INT8, GPU + accelerators) |
| Memory | ≥ 64 GB LPDDR5X (automotive config up to 128 GB) |
| Memory bandwidth | ~900 GB/s |
| CPU | 12× Arm Cortex-X925 (high-performance) |
| DLA | Next-gen, 2 engines |
| Target TDP | ~100 W (automotive module) |

**Why the jump is so large:** Thor's Blackwell GPU generation adds:
- Transformer Engine (native FP8 + mixed-precision) — transformers like
  SegFormer run at lower precision with no quality loss.
- Second-generation Tensor Cores with higher throughput per clock.
- The ~4× bandwidth increase (vs AGX Orin) removes the memory wall for
  large-batch or high-resolution inference.

### Thor example: YOLO11n at 1080p

On Orin Nano 8 GB, YOLO11n benchmarks at 640×640 (the native training
resolution) at roughly 15–20 FPS in FP16 TensorRT.  Scaling to 1080p is
impractical — the input resize and larger feature maps push it below real-time.

On Thor (estimated, Blackwell FP8 TensorRT):
- **1080p input, FP8 TensorRT:** ~120–150 FPS estimated.
- The 900 GB/s bus means the feature pyramid at 1080p fits comfortably in
  the sustained bandwidth budget.
- The Transformer Engine handles YOLO11's C2f attention blocks natively in
  FP8 vs FP16 on Orin, gaining ~1.5× on those layers.

To run this benchmark suite on Thor when hardware is available, the only
changes needed are:
1. Update `onnx_providers` in `pipeline.yaml` to include
   `TensorrtExecutionProvider` (Thor will support TensorRT 10+).
2. Optionally lower precision: add `fp8: true` to `tensorrt_ep_options`
   once the ORT TRT EP exposes FP8 (expected with TensorRT ≥ 10.3).
3. Update the GStreamer pipeline resolution to 1920×1080 in
   `gstreamer_pipeline.py` — the synthetic input generator already supports
   arbitrary `--width`/`--height`.

---

## Choosing a Platform

| Use case | Recommended module |
|---|---|
| Battery-powered mobile robot, SLAM only | Orin Nano 8 GB |
| Mobile robot + detection at 30 FPS | Orin NX 8 GB (use DLA) |
| Segmentation in real-time on-board | Orin NX 16 GB |
| Multi-model concurrent pipelines, stereo SLAM | AGX Orin 32/64 GB |
| Full-resolution (1080p+) detection + SLAM + seg | Thor (future) |
