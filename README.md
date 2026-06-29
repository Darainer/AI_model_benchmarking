# AI Model Benchmarking — Jetson Nano / Orin

Config-driven inference pipeline for NVIDIA Jetson devices. Download model
checkpoints, feed them a standardized MP4 video (or camera stream), and get
per-frame latency, FPS, GPU utilization, and memory bandwidth stats — all from
a single command.

## What's in the box

| Layer | What it does |
|---|---|
| `configs/models.yaml` | Model registry — add/remove checkpoints here |
| `configs/pipeline.yaml` | Input source, GStreamer settings, ORT provider priority |
| `src/input/` | Unified frame iterator (synthetic / MP4 / image dir / camera) via GStreamer |
| `src/models/backends/` | ONNX Runtime (CUDA + TRT EP), native TensorRT, PyTorch |
| `src/models/downloader.py` | Auto-download or ONNX-export from URL / HuggingFace / torchvision / smp |
| `src/benchmark/hardware_monitor.py` | Background sampler — tegrastats (Jetson) or nvidia-smi (desktop) |
| `src/benchmark/runner.py` | Warmup → timed inference → report table + CSV |
| `scripts/run_benchmark.py` | Main CLI |
| `scripts/download_models.py` | Pre-download checkpoints before first run |
| `Dockerfile` / `docker-compose.yml` | Container built on `dustynv/l4t-ml:r36.4.0` (JetPack 6 / CUDA 12.6) |

### Default model registry

| Model | Task | Backend | Input | Source |
|---|---|---|---|---|
| MobileNetV2 | Classification | ONNX | 224×224 | ONNX Model Zoo |
| ResNet50-v1 | Classification | ONNX | 224×224 | ONNX Model Zoo |
| EfficientNet-Lite4 | Classification | ONNX | 224×224 | ONNX Model Zoo |
| SqueezeNet 1.1 | Classification | ONNX | 224×224 | ONNX Model Zoo |
| YOLOv8n | Detection | ONNX | 640×640 | Ultralytics |
| SegFormer-B0 (ADE20K) | Segmentation | ONNX | 512×512 | HuggingFace → ONNX export |
| UNet ResNet34 | Segmentation | ONNX | 512×512 | segmentation-models-pytorch → ONNX export |

---

## Quickstart — Docker (recommended on Orin / JetPack 6)

The container inherits the Jetson GPU stack (PyTorch, ONNX Runtime GPU,
TensorRT, CUDA-enabled OpenCV) from `dustynv/l4t-ml:r36.4.0`, so there's no
risk of pip clobbering the aarch64 GPU builds.

```bash
# 1. Pull the base image (several GB — one time only)
make pull

# 2. Build the benchmark image on top
make build

# 3. Confirm GPU providers are visible
make gpu-check
# → providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 4. Download all model checkpoints into ./models/
make download

# 5. Run benchmark against an MP4 video
make bench-file VIDEO=/workspace/clip.mp4

# 6. Or run on synthetic frames (no video needed — pure inference timing)
make bench

# 7. Interactive shell inside the container
make shell
```

### Makefile targets

| Target | Command |
|---|---|
| `make venv` | Host venv setup (shares deps with Docker — see [VENV.md](VENV.md)) |
| `make pull` | Pull base image (`dustynv/l4t-ml:r36.4.0`) |
| `make build` | Build `ai-model-benchmarking:orin` |
| `make gpu-check` | Verify GPU providers inside container |
| `make download` | Download / export all model checkpoints |
| `make bench` | Synthetic frame benchmark |
| `make bench-file VIDEO=…` | Benchmark a video file |
| `make bench-cpu` | Force CPU-only (no CUDA) |
| `make shell` | Interactive bash session |
| `make clean` | Remove the built image |

---

## Quickstart — Host venv (Orin / JetPack 6 — best for video/camera)

A `--system-site-packages` venv inherits the JetPack GPU stack and installs the
project deps via the **same shared script the Docker build uses**
(`scripts/install_deps.sh`), so the two environments stay consistent. Prefer
this path for HW-decoded **video/camera** runs — the host NVDEC path works
directly, with no container-toolkit layer to break `nvv4l2decoder`. Full
details and the two-layer design in [VENV.md](VENV.md).

```bash
make venv                 # verify host GPU stack → create .venv → install deps
source .venv/bin/activate
python scripts/download_models.py
python scripts/run_benchmark.py --video /path/to/clip.mp4
```

---

## Quickstart — Host install (Nano / JetPack 4.x)

```bash
# Install Python deps (onnxruntime-gpu comes from JetPack or Jetson Zoo)
pip install -r requirements.txt

# Download all checkpoints (auto-exports SegFormer and UNet to ONNX on first run)
python scripts/download_models.py

# Run on an MP4 video
python scripts/run_benchmark.py --video /path/to/clip.mp4

# Synthetic frames (no video required)
python scripts/run_benchmark.py

# Specific models only
python scripts/run_benchmark.py --video clip.mp4 --filter mobilenet_v2 yolov8n

# Force CPU
python scripts/run_benchmark.py --providers CPUExecutionProvider
```

---

## Input sources

```bash
# MP4 / video file (HW-decoded via nvv4l2decoder on Jetson)
python scripts/run_benchmark.py --video /path/to/clip.mp4

# CSI camera (ribbon connector — uses nvarguscamerasrc)
python scripts/run_benchmark.py --input-type csi_camera

# USB camera
python scripts/run_benchmark.py --input-type usb_camera --device /dev/video0

# Directory of images
python scripts/run_benchmark.py --input-type image_dir --source /path/to/frames/

# Synthetic (random frames — isolates pure inference time from I/O)
python scripts/run_benchmark.py
```

For camera passthrough inside Docker, uncomment the relevant `devices:` /
`volumes:` block in `docker-compose.yml`.

---

## Output

Each run prints a per-model block as it runs, then a consolidated table:

```
================================================================
  Benchmark Report — 2026-06-28 14:32:01
  Input  : clip.mp4  |  Monitor: tegrastats
================================================================
  Model                   Backend     Task            Frms  avg ms  p95 ms    FPS  GPU%  Mem MB  BW avg   BW peak
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  mobilenet_v2            ORT/TRT     classification   100     4.1     4.8  243.9   42%     210   4.3 GB/s   6.1 GB/s
  resnet50_v1             ORT/TRT     classification   100     9.2    10.1  108.7   61%     310   8.7 GB/s  12.4 GB/s
  yolov8n                 ORT/CUDA    detection         50    12.3    13.5   81.3   55%     280   7.1 GB/s   9.8 GB/s
  segformer_b0_ade        ORT/CUDA    segmentation      50    22.4    24.1   44.6   76%     520  15.2 GB/s  19.3 GB/s
  unet_resnet34           ORT/CUDA    segmentation      50    17.8    19.2   56.2   68%     390  11.4 GB/s  15.7 GB/s
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Memory Bandwidth across all models: avg 9.3 GB/s  peak 19.3 GB/s  (theoretical 25.6 GB/s)
================================================================
```

**Hardware stats:**
- **Jetson (`tegrastats`)** — unified RAM, GR3D%, EMC% → GB/s via `2 × EMC_MHz × 64-bit bus / 8 / 1000`
- **Desktop (`nvidia-smi`)** — VRAM used, GPU%, memory controller utilization %; GB/s via pynvml if available

Results are also saved to `results/benchmark_results.csv`.

---

## TensorRT (best performance)

### Option A — ORT TensorRT EP (automatic, no extra tooling)

Set provider priority in `configs/pipeline.yaml`:

```yaml
benchmark:
  onnx_providers:
    - TensorrtExecutionProvider   # builds & caches engine on first run
    - CUDAExecutionProvider
    - CPUExecutionProvider
  tensorrt_ep_options:
    trt_fp16_enable: "1"
    trt_engine_cache_enable: "1"
    trt_engine_cache_path: "./models/trt_cache"
```

First run is slow (engine build). Subsequent runs load the cached `.trt` from
`models/trt_cache/`.

### Option B — native TensorRT backend (`trtexec`)

```bash
# Build FP16 engine once
trtexec \
  --onnx=models/mobilenet_v2.onnx \
  --saveEngine=models/mobilenet_v2.trt \
  --fp16

# Add to configs/models.yaml:
- name: mobilenet_v2_trt
  task: classification
  backend: tensorrt
  source: local
  local_path: "models/mobilenet_v2.trt"
  input_shape: [1, 3, 224, 224]
  input_mean: [0.485, 0.456, 0.406]
  input_std:  [0.229, 0.224, 0.225]
  warmup_runs: 10
  benchmark_runs: 100
```

---

## Adding a model

Edit `configs/models.yaml`:

```yaml
- name: my_model
  task: classification          # classification | detection | segmentation
  backend: onnx                 # onnx | tensorrt | torch
  source: url                   # url | torchvision | ultralytics | huggingface | smp | local
  url: "https://..."
  local_path: "models/my_model.onnx"
  input_shape: [1, 3, 224, 224]
  input_mean: [0.485, 0.456, 0.406]
  input_std:  [0.229, 0.224, 0.225]
  warmup_runs: 10
  benchmark_runs: 100
```

Then download it:

```bash
python scripts/download_models.py --filter my_model
# or inside Docker:
docker compose run --rm benchmark python3 scripts/download_models.py --filter my_model
```

### Segmentation — source types

**`source: huggingface`** — downloads from HuggingFace and exports to ONNX automatically.
Requires `pip install transformers torch`.

```yaml
- name: segformer_b0_ade
  source: huggingface
  hf_model_id: "nvidia/segformer-b0-finetuned-ade-512-512"
  num_classes: 150
```

**`source: smp`** — builds with `segmentation-models-pytorch` and exports to ONNX.
Requires `pip install segmentation-models-pytorch torch`. Supports `Unet`, `FPN`,
`DeepLabV3Plus`, and all other `smp` architectures.

```yaml
- name: unet_resnet34
  source: smp
  smp_arch: Unet
  smp_encoder: resnet34
  smp_encoder_weights: imagenet
  smp_classes: 21
```

---

## Repo layout

```
configs/
  models.yaml             # model registry
  pipeline.yaml           # input + GStreamer + backend + monitor settings
scripts/
  download_models.py      # pre-download / export all checkpoints
  run_benchmark.py        # main CLI
src/
  input/
    frame_source.py       # unified frame iterator
    gstreamer_pipeline.py # GStreamer string builder (HW decode, CSI/USB camera)
  models/
    base.py               # BaseModel ABC — preprocess / infer / warmup
    registry.py           # config → backend factory
    downloader.py         # download / ONNX-export from multiple sources
    backends/
      onnx_backend.py     # ONNX Runtime (TRT EP / CUDA / CPU)
      tensorrt_backend.py # native TensorRT (.trt engine files)
      torch_backend.py    # PyTorch (.pt / TorchScript)
  benchmark/
    hardware_monitor.py   # background tegrastats / nvidia-smi sampler
    runner.py             # orchestrates load → warmup → time → report → CSV
    metrics.py            # RunMetrics dataclass (latency, FPS, hw stats)
  utils/
    logging.py
Dockerfile                # dustynv/l4t-ml:r36.4.0 base (JetPack 6 / CUDA 12.6)
docker-compose.yml
Makefile                  # build / pull / bench / shell shortcuts
models/                   # gitignored — populated by download_models.py
results/                  # gitignored — benchmark CSVs written here
```
