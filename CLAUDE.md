# AI Model Benchmarking — Jetson Nano

Inference pipeline that downloads model checkpoints and benchmarks them against a standardized video/image input on NVIDIA Jetson Nano (or any CUDA-capable machine).

## Repo layout

```
configs/
  models.yaml        # registry of model checkpoints to benchmark
  pipeline.yaml      # input source + GStreamer + backend settings
scripts/
  download_models.py # download all checkpoints before first run
  run_benchmark.py   # main entry point
src/
  input/
    frame_source.py        # unified frame iterator (synthetic/file/dir/camera)
    gstreamer_pipeline.py  # GStreamer pipeline string builder (Jetson)
  models/
    base.py                # BaseModel ABC (preprocess/infer/warmup)
    registry.py            # factory: config → backend instance
    downloader.py          # download/export models to models/ dir
    backends/
      onnx_backend.py      # ONNX Runtime (CUDA / TensorRT EP / CPU)
      tensorrt_backend.py  # native TensorRT (.trt engine files)
      torch_backend.py     # PyTorch (.pt / TorchScript)
  benchmark/
    runner.py    # orchestrates download → load → warmup → time → save
    metrics.py   # latency / throughput / GPU stats dataclass
models/          # gitignored — populated by download_models.py
results/         # gitignored — benchmark CSVs written here
```

## Docker (recommended on Jetson)

For a reproducible GPU stack, run inside the container instead of a host venv —
see [DOCKER.md](DOCKER.md). TL;DR: `make pull && make build && make bench`.

## Quickstart (host install — Jetson Nano / JetPack 4.x+)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
# onnxruntime-gpu wheel for Jetson ships with JetPack or from Jetson Zoo:
# https://elinux.org/Jetson_Zoo#ONNX_Runtime

# 2. Download model checkpoints
python scripts/download_models.py

# 3. Run benchmark on synthetic frames
python scripts/run_benchmark.py

# 4. Run on a real video file (hardware-decoded)
python scripts/run_benchmark.py --input-type file --source /path/to/clip.mp4

# 5. Run on CSI camera (Jetson ribbon connector)
python scripts/run_benchmark.py --input-type csi_camera

# 6. Run on USB camera
python scripts/run_benchmark.py --input-type usb_camera --device /dev/video0

# 7. Only benchmark specific models
python scripts/run_benchmark.py --filter mobilenet_v2 yolov8n

# 8. Force CPU only (no CUDA needed)
python scripts/run_benchmark.py --providers CPUExecutionProvider
```

## Adding a new model

Edit `configs/models.yaml` and add an entry:

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

Then run `python scripts/download_models.py --filter my_model`.

### Segmentation models

**SegFormer** (transformer, ADE20K 150 classes): requires `pip install transformers torch`.
On first download the model is exported from HuggingFace to ONNX automatically.
Output shape: `[N, 150, H/4, W/4]` — upscale with `F.interpolate` for full-res masks.

```yaml
- name: segformer_b0_ade
  task: segmentation
  backend: onnx
  source: huggingface
  hf_model_id: "nvidia/segformer-b0-finetuned-ade-512-512"
  local_path: "models/segformer_b0_ade.onnx"
  input_shape: [1, 3, 512, 512]
  input_mean: [0.485, 0.456, 0.406]
  input_std:  [0.229, 0.224, 0.225]
  num_classes: 150
```

**UNet** (classic encoder-decoder): requires `pip install segmentation-models-pytorch torch`.
`smp_arch` can be `Unet`, `FPN`, `DeepLabV3Plus`, etc.

```yaml
- name: unet_resnet34
  task: segmentation
  backend: onnx
  source: smp
  smp_arch: Unet
  smp_encoder: resnet34
  smp_encoder_weights: imagenet
  smp_classes: 21
  local_path: "models/unet_resnet34.onnx"
  input_shape: [1, 3, 512, 512]
  input_mean: [0.485, 0.456, 0.406]
  input_std:  [0.229, 0.224, 0.225]
  num_classes: 21
```

## TensorRT engine workflow

```bash
# Build FP16 engine from an ONNX model (run once, cached on disk)
trtexec \
  --onnx=models/mobilenet_v2.onnx \
  --saveEngine=models/mobilenet_v2.trt \
  --fp16

# Add to configs/models.yaml with backend: tensorrt, local_path: models/mobilenet_v2.trt
```

Or use the ORT TensorRT EP (automatic, no trtexec needed — set in `pipeline.yaml`):
```yaml
benchmark:
  onnx_providers:
    - TensorrtExecutionProvider   # builds & caches engine automatically
    - CUDAExecutionProvider
    - CPUExecutionProvider
```

## Results

CSV saved to `results/benchmark_results.csv` after each run:

| model | backend | task | frames | avg_latency_ms | p95_latency_ms | throughput_fps |
|-------|---------|------|--------|---------------|---------------|---------------|

## Development notes

- `BaseModel.preprocess()` handles standard ImageNet normalization (CHW, float32)
- `BaseModel.preprocess_letterbox()` is used for detection models (YOLO-style)
- The `synthetic` input type generates random frames — useful for pure inference timing without I/O overhead
- The ORT TensorRT EP caches built engines under `models/trt_cache/`; first run is slow (engine build), subsequent runs are fast
- GStreamer fallback: if `cv2.CAP_GSTREAMER` fails, frame_source falls back to `cv2.VideoCapture` native
