# DLA-split SuperPoint + LightGlue — Orin NX Prototype

> **Status: PROTOTYPE / UNTESTED ON HARDWARE.**
> The code loads and runs on any CUDA machine via a full-GPU fallback, but the
> NVDLA offload path can only be *exercised and timed* on real Jetson Orin NX
> silicon with the JetPack TensorRT + `onnxruntime-gpu` (TensorRT EP) stack.
> Numbers below are reasoned estimates, not measured results.

## Why

The stock pipeline (`backend: lightglue`) runs **both** stages on the GPU:

```
SuperPoint (CNN)  ──►  LightGlue (attention matcher)   ── all on GPU
```

`HARDWARE.md` flags this workload as **bandwidth-bound**, not compute-bound.
Two facts drive that:

1. **SuperPoint under-fills the GPU.** It's a small dense CNN at 480×640 — it
   does not saturate the 1024 Ampere cores on an Orin NX.
2. **The matcher is latency/BW-bound.** Attention over ≤512 keypoints is a pile
   of *tiny* matmuls (512×256, 512×512) plus softmax. The arithmetic is trivial;
   you pay for memory traffic and kernel launches, not FLOPs.

Meanwhile, the **two NVDLA v2 cores on the Orin NX sit idle.** The DLA is a
fixed-function CNN accelerator — exactly what SuperPoint's conv backbone wants.
Moving that backbone off the GPU frees the GPU to do nothing but the matcher,
and lets the two engines overlap.

## The split

SuperPoint is not one monolithic block — it's a conv backbone followed by a
keypoint-extraction head:

```
SuperPoint
├── dense conv backbone        conv1a … convPb / convDb     ← STATIC shapes, FP16
│                                                              → DLA-friendly
└── keypoint head              softmax → depth-to-space →    ← DYNAMIC shapes
                               NMS → top-k → grid_sample        (variable #kpts)
                                                              → must stay on GPU
```

So the prototype routes work like this:

```
        image (1×1×480×640, FP16)
                 │
                 ▼
   ┌─────────────────────────────┐
   │ SuperPoint conv backbone     │   NVDLA core 0   (TensorRT EP, FP16)
   │ → scores_logits [1,65,60,80] │   GPU_FALLBACK handles any rejected layer
   │ → desc_raw      [1,256,60,80]│
   └─────────────────────────────┘
                 │   (dense conv maps)
                 ▼
   ┌─────────────────────────────┐
   │ keypoint post-processing     │   GPU   (dynamic shapes)
   │ softmax→d2s→NMS→top-k→sample │
   └─────────────────────────────┘
                 │   (keypoints + descriptors)
                 ▼
   ┌─────────────────────────────┐
   │ LightGlue attention matcher  │   GPU   (FP16 autocast → Tensor Cores)
   └─────────────────────────────┘
                 │
                 ▼
              matches
```

### Why the cut is *exactly here*

- **DLA only runs static-shape FP16/INT8 layers.** The conv stack qualifies; the
  keypoint head (whose output length depends on the image) does not. Trying to
  push NMS/top-k onto the DLA would fail the build or silently fall back.
- **`GPU_FALLBACK` is mandatory, not optional.** A couple of ops in the backbone
  (notably the final reshape / depth-to-space if you fold it in) may not be
  DLA-supported. With GPU fallback on, TensorRT places those on the GPU and only
  the conv-heavy core lands on the DLA. The prototype leaves the depth-to-space
  *out* of the exported graph for this reason — it's done in the GPU post-proc.
- **FP16 on the matcher is the cheap win.** The stock backend runs the matcher
  in FP32. Wrapping it in `torch.autocast(float16)` puts the small matmuls on
  Tensor Cores for ~1.5–2× with no algorithmic change. This is independent of
  the DLA work and is the single highest-value line in the prototype.

## What's in the repo

| File | Change |
|---|---|
| `src/models/backends/feature_matching_dla_backend.py` | New `LightGlueDLAModel` backend (the prototype). |
| `src/models/registry.py` | Registers `backend: lightglue_dla`. |
| `configs/models.yaml` | New `superpoint_lightglue_dla` entry. |
| `docs/DLA_SUPERPOINT_PROTOTYPE.md` | This document. |

The new backend reuses the existing `BaseModel` interface and the
`source: system` sentinel flow, so `download_models.py` and `run_benchmark.py`
need no changes.

### Config knobs (`configs/models.yaml`)

```yaml
- name: superpoint_lightglue_dla
  backend: lightglue_dla
  max_keypoints: 512
  dla_core: 0                 # 0 or 1 — Orin NX has two NVDLA cores
  nms_radius: 4               # SuperPoint default
  detection_threshold: 0.0005
  force_gpu_fallback: false   # true → skip DLA, pure-GPU baseline for A/B
```

## How to run

```bash
# Anywhere (will use the full-GPU fallback off-Jetson):
python scripts/run_benchmark.py --filter superpoint_lightglue_dla

# A/B against the stock GPU pipeline in one run:
python scripts/run_benchmark.py --filter superpoint_lightglue superpoint_lightglue_dla
```

On a real Orin NX, watch the log line on load:

```
superpoint_lightglue_dla: DLA backbone path ACTIVE (core=0).
superpoint_lightglue_dla: backbone ORT session providers=['TensorrtExecutionProvider', ...]
```

If you instead see `DLA path unavailable (...). Falling back to full-GPU
SuperPoint.` the prototype still benchmarks — it just measured the GPU path.
The `backend` column in the CSV disambiguates:
`LightGlue-DLA[SuperPoint@DLA+LightGlue@GPU/FP16]/CUDA` vs `...[FullGPU]/CUDA`.

## Verifying the DLA is actually used

ORT reporting the TensorRT EP is necessary but not sufficient — confirm the DLA
core is busy while the backbone runs:

```bash
# DLA utilization counters
cat /sys/devices/platform/host1x/15880000.nvdla0/power/runtime_status
sudo tegrastats        # watch the NVDLA / DLA_FREQ fields
```

You can also dump the TensorRT engine layer placement:

```bash
trtexec --onnx=models/superpoint_lightglue_dla_backbone.onnx \
        --useDLACore=0 --fp16 --allowGPUFallback --verbose 2>&1 | grep -i dla
```

## Expected outcome (estimate, not measured)

Stock pipeline on Orin NX 16 GB is ~12 ms (HARDWARE.md). For the DLA split,
the realistic expectation:

- The **DLA is slower per-op than the GPU** for the backbone in isolation (DLA
  v2 ≈ a few TOPS vs the GPU's tens). So the win is **not** a faster backbone.
- The win is **concurrency + freeing the GPU**: backbone on DLA overlaps with
  the matcher on GPU, and the matcher gets the GPU to itself in FP16.
- Net effect is workload-dependent. Likely outcomes, in order of probability:
  1. **Latency roughly flat, GPU headroom freed** — best when you run another
     GPU model concurrently (detection, segmentation). This is the real use
     case: the DLA split is a *system throughput* optimization, not a
     single-model latency one.
  2. **Small latency win** from the FP16 matcher dominating.
  3. **Slight regression** if the SoC-memory round-trip (GPU tensor → numpy →
     ORT → GPU tensor in `_extract_dla`) costs more than the DLA saves. See
     limitations.

**Bottom line the analysis already reached:** the matcher is a mediocre GPU
target and the DLA can't help it; the DLA's value is taking the *backbone* off
the GPU so the whole-system pipeline (matcher + other models) has more GPU to
work with. Treat this prototype as a throughput experiment, not a latency
silver bullet.

## Known limitations / TODO before this is real

- **Host round-trip in `_extract_dla`.** The current code moves the image
  GPU→CPU→ORT and the results back, which on a unified-memory SoC is wasteful.
  Production path: ORT IO-binding with CUDA pointers, or run the backbone via
  native TensorRT with the DLA core set and keep everything in device memory.
- **Backbone-attribute coupling.** `_SuperPointBackbone` reaches into the
  `lightglue` SuperPoint submodule names (`conv1a`…`convDb`). If the package
  refactors, export raises `AttributeError` and we fall back. Pin the
  `lightglue` commit, or vendor the SuperPoint forward.
- **Post-processing parity.** The GPU NMS/top-k/sample mirrors SuperPoint but
  has **not** been numerically diffed against `extractor.extract()` on hardware.
  Validate match counts agree before trusting timing.
- **INT8 DLA.** Only FP16 is wired up. INT8 on the DLA would roughly double its
  throughput but needs a calibration cache — out of scope for the prototype.
- **Two DLA cores.** `dla_core: 0` only. A fuller system could put a second
  model's CNN on core 1 to load-balance.

## References

- Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural
  Networks*, CVPR 2020 (the optimal-transport matcher this lineage starts from).
- Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed*,
  ICCV 2023 (adaptive-depth successor; the dependency actually used here).
- DeTone et al., *SuperPoint: Self-Supervised Interest Point Detection and
  Description*, CVPRW 2018.
- NVIDIA, *Working with DLA* (TensorRT Developer Guide) — `--useDLACore`,
  `--allowGPUFallback`, FP16/INT8 constraints.
