#!/usr/bin/env bash
# scripts/setup_venv.sh
# =============================================================================
# LAYER 2a — host venv bootstrap (Jetson / JetPack 6.x).
#
# Creates a `--system-site-packages` venv so it INHERITS the JetPack-provided
# GPU stack (CUDA+GStreamer OpenCV via apt python3-opencv, TensorRT bindings via
# apt python3-libnvinfer, and the Jetson torch / onnxruntime-gpu wheels), then
# layers this project's pure-python deps on top via the SHARED installer
# (scripts/install_deps.sh) — the exact same script the Docker build runs. That
# shared layer is what keeps the venv and the container in sync.
#
# Why a venv (not Docker) for video/camera runs: the host NVDEC path
# (nvv4l2decoder) works directly here — there is no nvidia-container-toolkit
# layer to fail the `S_EXT_CTRLS for CUDA_GPU_ID` ioctl that breaks HW decode
# inside the container. See open_issues.md.
#
# A venv only isolates Python packages; it does NOT supply the GPU builds. So
# this script first verifies the host already has them, then builds on top.
#
# Env overrides:
#   PYTHON     host interpreter to base the venv on   (default: python3)
#   VENV_DIR   where to create the venv               (default: <repo>/.venv)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

echo "================================================================"
echo " Host venv setup — AI Model Benchmarking (Jetson)"
echo "================================================================"

# --- Prerequisite check: the GPU stack must exist on the HOST first ----------
# The venv inherits these via --system-site-packages; it cannot create them.
echo "==> Checking host GPU stack (required before creating the venv)"
required_missing=0
optional_missing=0

check() {  # $1=label  $2=python-import-test  $3=fix-hint  $4=required(1/0)
  if "$PY" -c "$2" >/dev/null 2>&1; then
    echo "  [ok]   $1"
  else
    if [ "$4" = "1" ]; then
      echo "  [MISS] $1 (required) — $3"
      required_missing=1
    else
      echo "  [warn] $1 (optional) — $3"
      optional_missing=1
    fi
  fi
}

check "onnxruntime-gpu (CUDA provider)" \
  "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()" \
  "install the Jetson onnxruntime-gpu wheel (Jetson Zoo / NVIDIA index)" 1
check "OpenCV built WITH GStreamer" \
  "import cv2, re; m=re.search(r'GStreamer:\s*(YES|NO)', cv2.getBuildInformation()); assert m and m.group(1)=='YES'" \
  "sudo apt-get install python3-opencv  (JetPack GStreamer build; do NOT pip install opencv-python)" 1
check "torch (CUDA)" \
  "import torch; assert torch.cuda.is_available()" \
  "install the Jetson torch wheel (only needed for source=torchvision/huggingface/smp)" 0
check "tensorrt bindings" \
  "import tensorrt" \
  "sudo apt-get install python3-libnvinfer  (only needed for backend=tensorrt)" 0

if [ "$required_missing" -ne 0 ]; then
  echo
  echo "  !! Required GPU packages are missing on the host. The venv would not"
  echo "  !! have a working GPU stack. Install the [MISS] items above, then"
  echo "  !! re-run this script. (torch/tensorrt warnings are fine to skip if"
  echo "  !! you don't use those backends.)"
  exit 1
fi

# --- Create the venv with access to the system (JetPack) site-packages --------
echo "==> Creating venv (--system-site-packages) at $VENV_DIR"
"$PY" -m venv --system-site-packages "$VENV_DIR"

# --- Install project deps via the SHARED installer (Layer 1) ------------------
echo "==> Installing project deps via scripts/install_deps.sh (shared with Docker)"
PYTHON="$VENV_DIR/bin/python" bash "$REPO_ROOT/scripts/install_deps.sh"

# --- Post-install verification (providers visible from inside the venv) -------
echo "==> Verifying GPU execution providers from inside the venv"
"$VENV_DIR/bin/python" -c \
  "import onnxruntime as o; print('   providers:', o.get_available_providers())" || true

echo
echo "================================================================"
echo " Done. Activate the venv with:"
echo "     source $VENV_DIR/bin/activate"
echo " Then run, e.g.:"
echo "     python scripts/download_models.py"
echo "     python scripts/run_benchmark.py --video /path/to/clip.mp4"
echo "================================================================"
