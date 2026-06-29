# DLA-split SuperPoint + LightGlue — Orin NX Prototype

> **Status: PROTOTYPE / UNTESTED ON HARDWARE.**
> All code loads and runs on any CUDA machine via a full-GPU fallback, but the
> NVDLA offload + DLA/GPU overlap can only be *exercised and timed* on real
> Jetson Orin NX silicon with the JetPack TensorRT stack. Numbers below are
> reasoned estimates, not measured results.

## The thesis

The DLA does **not** make SuperPoint faster in isolation — DLA v2 is a few TOPS
versus the GPU's tens. The win is a **system** win:

1. **Move the SuperPoint conv backbone off the GPU onto the idle NVDLA.**
2. **Overlap** it with the GPU work (matcher) on a *stream* of frames.
3. Result: the GPU 3D engine stops doing convolutions → **measured GPU
   utilization drops**, and that reclaimed GPU headroom can run a second model
   (detection / segmentation) concurrently, or the whole pipeline runs cooler.

So the metric that matters here is **`gpu_util_avg_pct` going down while
throughput holds**, not single-model latency going down.

## Architecture

SuperPoint splits at the static/dynamic boundary:

```
SuperPoint
├── dense conv backbone     conv1a … convPb / convDb     STATIC shapes, FP16
│                                                         → NVDLA (zero-copy)
└── keypoint head           softmax → depth-to-space →    DYNAMIC shapes
                            NMS → top-k → grid_sample      → GPU (always)

LightGlue attention matcher                                → GPU, FP16 autocast
```

Two backends are provided:

| Backend (`configs/models.yaml`) | What it does | Use |
|---|---|---|
| `lightglue_dla` | Single frame, self-match (image0==image1). | Benchmark parity vs stock `lightglue`; verifies the DLA path works. |
| `lightglue_dla_pipeline` | **Overlapped** streaming, matches consecutive frames. | The real win — DLA ∥ GPU concurrency. |

## Zero-copy: how tensors stay on the device

The earlier draft round-tripped GPU→CPU→ORT→GPU. That's gone. The shared
extractor (`src/models/backends/_superpoint_dla.py`) keeps everything in
device memory, with three modes picked at `setup()`:

### `trt-dla` — native TensorRT (the real path)
- Builds (and disk-caches) a TensorRT engine from the exported backbone ONNX
  with `default_device_type = DLA`, `DLA_core = N`, `FP16`, and `GPU_FALLBACK`.
- At inference, **no copies**: the input and both outputs are torch CUDA
  tensors whose pointers are handed to TRT via
  `context.set_tensor_address(name, tensor.data_ptr())`.
- Runs with `execute_async_v3(stream_handle=...)` on a caller-supplied CUDA
  stream → **asynchronous**, so it can overlap GPU work (see pipeline below).

```python
scores_t = torch.empty((1,65,H//8,W//8), device="cuda")   # TRT writes here
desc_t   = torch.empty((1,256,H//8,W//8), device="cuda")
ctx.set_tensor_address("image",          img.data_ptr())
ctx.set_tensor_address("scores_logits",  scores_t.data_ptr())
ctx.set_tensor_address("desc_raw",       desc_t.data_ptr())
ctx.execute_async_v3(stream_handle=dla_stream.cuda_stream)
```

### `ort-dla` — onnxruntime TensorRT EP (simpler fallback)
- `trt_dla_enable=1`, `trt_dla_core=N`, `trt_fp16_enable=1`.
- Still zero-copy via `io_binding` bound to torch tensor `data_ptr()`s, but
  `run_with_iobinding` is **synchronous** — correct, but no DLA/GPU overlap.

### `gpu` — stock lightglue SuperPoint
- No DLA available → run the full extractor on the GPU. Everything still works.

`backend_name()` reports the active mode (`SP-trt-dla∥LG[overlap]/CUDA`, etc.)
and it lands in the CSV `backend` column.

## The overlapped pipeline

`feature_matching_pipeline_backend.py` drives **two CUDA streams** with a
one-frame software pipeline (double-buffered):

```
Per frame N (single sync at the end):
  dla_stream :  backbone(N)                       ── NVDLA, async, zero-copy
  gpu_stream :  wait(event N-1)
                postproc(N-1)                      ── GPU
                match(N-2, N-1)  [FP16 autocast]   ── GPU
```

Because backbone(N) and the previous frame's GPU work are issued to different
streams before the sync, they run **concurrently**:

```
frame:     N-1         N           N+1
DLA  :   bbone(N-1)  bbone(N)    bbone(N+1)
GPU  :      …        post(N-1)    post(N)
                     match(N-2)   match(N-1)
            └── overlap ──┘
```

A `torch.cuda.Event` recorded on `dla_stream` after backbone(N) gates the GPU
stream so it never reads backbone outputs before the DLA has written them —
correctness without a global sync between the engines.

The matcher now matches **consecutive frames** (N-2 ↔ N-1), the real
odometry/tracking use case, instead of a frame against itself. First two frames
return empty (pipeline fill).

> In `ort-dla`/`gpu` modes the pipeline runs serially (no overlap) but still
> produces correct consecutive matches — so the backend is safe to benchmark
> anywhere. `backend_name()` shows `[overlap]` vs `[serial]`.

## Getting GPU utilization down — how to measure it

Two independent measurements, both already plumbed:

### 1. System GR3D% via the existing HardwareMonitor
`run_benchmark.py` samples `tegrastats` and records `gpu_util_avg_pct`
(GR3D_FREQ — the GPU 3D-engine busy %). Compare three rows in one run:

```bash
python scripts/run_benchmark.py --filter \
  superpoint_lightglue \
  superpoint_lightglue_dla \
  superpoint_lightglue_dla_pipeline
```

Expectation on Orin NX:

| Row | GPU% (GR3D) | Why |
|---|---|---|
| `superpoint_lightglue` (stock) | highest | backbone + matcher both on GPU |
| `superpoint_lightglue_dla` | lower | backbone on DLA, but serial |
| `superpoint_lightglue_dla_pipeline` | **lowest** | backbone on DLA *and* overlapped |

### 2. Per-frame GPU-busy time via CUDA events
The pipeline brackets its GPU-stream work with `cudaEvent`s and logs an EMA:

```
superpoint_lightglue_dla_pipeline: GPU-busy/frame ≈ 6.20 ms (EMA) over 25 frames
```

If `GPU-busy/frame` is well below `avg_latency_ms`, the GPU is idle for the
remainder of each frame — that idle slice is the headroom the DLA bought you,
available for a concurrent model.

### 3. Confirm the DLA is actually doing the work
GR3D% dropping is only meaningful if the DLA picked up the backbone:

```bash
# DLA engine power-gated state (should be 'active' during the run)
cat /sys/devices/platform/host1x/15880000.nvdla0/power/runtime_status
sudo tegrastats              # watch for DLA / NVDLA activity, falling GR3D_FREQ
jtop                         # jetson-stats: per-engine GPU/DLA/CPU gauges

# Verify layer placement at build time
trtexec --onnx=models/.lightglue_dla_sentinel_backbone.onnx \
        --useDLACore=0 --fp16 --allowGPUFallback --verbose 2>&1 | grep -i dla
```

> Note: raw `tegrastats` does not expose a reliable DLA-utilization percentage
> on all JetPack builds; `jtop` (jetson-stats) is the most dependable per-engine
> view. The benchmark CSV captures GR3D% (GPU) — the DLA side is confirmed
> qualitatively via the tools above.

## Expected outcome (estimate, not measured)

- **Latency:** roughly flat, possibly a small win from the FP16 matcher. The
  DLA backbone is not faster than the GPU backbone in isolation.
- **GPU utilization:** **down** — the headline result. The conv work leaves
  GR3D; with overlap the GPU is busy only for postproc+matcher.
- **System throughput:** the real payoff appears when you run a *second* GPU
  model alongside this one. The freed GPU slice means the detector/segmenter
  no longer contends with SuperPoint's convolutions.

## What's in the repo

| File | Role |
|---|---|
| `src/models/backends/_superpoint_dla.py` | Shared zero-copy extractor (trt-dla / ort-dla / gpu), GPU post-proc. |
| `src/models/backends/feature_matching_dla_backend.py` | `lightglue_dla` — single-frame self-match. |
| `src/models/backends/feature_matching_pipeline_backend.py` | `lightglue_dla_pipeline` — overlapped consecutive-frame streaming. |
| `src/models/registry.py` | Registers both backends. |
| `configs/models.yaml` | `superpoint_lightglue_dla` + `superpoint_lightglue_dla_pipeline`. |
| `docs/DLA_SUPERPOINT_PROTOTYPE.md` | This document. |

### Config knobs

```yaml
max_keypoints: 512
dla_core: 0                 # 0 or 1 — Orin NX has two NVDLA cores
nms_radius: 4
detection_threshold: 0.0005
force_gpu_fallback: false   # true → skip DLA, pure-GPU baseline for A/B
```

## Known limitations / TODO before this is real

- **Hardware validation.** None of the timing/utilization claims are measured.
  Run the three-row comparison on an Orin NX and record real GR3D% deltas.
- **Post-processing parity.** The GPU NMS/top-k/sample mirrors lightglue's
  SuperPoint but has not been numerically diffed against `extractor.extract()`
  on hardware. Validate that match counts agree before trusting timing.
- **ORT stream overlap.** `ort-dla` mode is synchronous — only `trt-dla`
  overlaps. That's intentional (ORT user-compute-stream wiring is brittle), but
  it means the concurrency win requires the native TRT path.
- **Two DLA cores.** Only `dla_core: 0` is used. A fuller system could place a
  second model's CNN on core 1 to load-balance the two NVDLA engines.
- **INT8 DLA.** Only FP16 is wired up; INT8 (≈2× DLA throughput) needs a
  calibration cache — out of scope for the prototype.
- **Backbone-attribute coupling.** `_SuperPointBackbone` reaches into lightglue
  SuperPoint submodule names; pin the `lightglue` commit or vendor the forward.

## References

- Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural
  Networks*, CVPR 2020.
- Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed*,
  ICCV 2023 (the dependency actually used here).
- DeTone et al., *SuperPoint: Self-Supervised Interest Point Detection and
  Description*, CVPRW 2018.
- NVIDIA TensorRT Developer Guide — *Working with DLA* (`set_tensor_address`,
  `execute_async_v3`, `DLA_core`, `GPU_FALLBACK`, FP16/INT8 constraints).
