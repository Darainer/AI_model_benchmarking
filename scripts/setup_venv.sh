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

# --- Probe the GPU stack: host now, Docker base for comparison ---------------
# The venv inherits the host GPU stack via --system-site-packages; it cannot
# create it. We probe the host with scripts/_env_report.py and probe the Docker
# base with the SAME script (piped into `docker run`), then show them side by
# side so it's obvious whether the venv will match the container.
REPORT="$REPO_ROOT/scripts/_env_report.py"

echo "==> Probing host GPU stack"
HOST_REPORT="$("$PY" "$REPORT" 2>/dev/null || true)"
declare -A HOST
while IFS='=' read -r k v; do [ -n "$k" ] && HOST["$k"]="$v"; done <<< "$HOST_REPORT"

echo "==> Probing Docker base for comparison (best effort)"
DOCKER_REPORT=""
DOCKER_SRC="(not available)"
if command -v docker >/dev/null 2>&1; then
  for img in ai-model-benchmarking:orin dustynv/l4t-ml:r36.4.0; do
    docker image inspect "$img" >/dev/null 2>&1 || continue
    # Try the nvidia runtime first so capability flags (cuda/providers) are
    # accurate; fall back to the default runtime for version-only info.
    DOCKER_REPORT="$(docker run --rm -i --runtime=nvidia "$img" python3 - < "$REPORT" 2>/dev/null \
                     || docker run --rm -i "$img" python3 - < "$REPORT" 2>/dev/null || true)"
    if [ -n "$DOCKER_REPORT" ]; then DOCKER_SRC="$img"; break; fi
  done
else
  DOCKER_SRC="(docker not installed)"
fi
declare -A DOCK
while IFS='=' read -r k v; do [ -n "$k" ] && DOCK["$k"]="$v"; done <<< "$DOCKER_REPORT"

# --- Render the comparison table ---------------------------------------------
echo
echo "  Dependency comparison — host venv vs Docker"
echo "  Docker source: $DOCKER_SRC"
printf "  %-22s %-20s %-20s %s\n" "component" "host (venv)" "docker" ""
printf "  %-22s %-20s %-20s %s\n" "----------------------" "--------------------" "--------------------" "----"
row() {  # $1=label  $2=key
  local h="${HOST[$2]:--}" d="${DOCK[$2]:--}" mark=""
  if [ -z "$DOCKER_REPORT" ]; then mark=""
  elif [ "$h" = "$d" ]; then mark="✓ match"
  else mark="✗ differ"; fi
  printf "  %-22s %-20s %-20s %s\n" "$1" "$h" "$d" "$mark"
}
row "python"             python
row "numpy"              numpy
row "torch"              torch
row "torchvision"        torchvision
row "onnxruntime"        onnxruntime
row "opencv"             opencv
row "tensorrt"           tensorrt
row "pandas"             pandas
row "torch CUDA"         torch_cuda
row "ORT CUDA provider"  ort_cuda
row "ORT TRT provider"   ort_trt
row "opencv GStreamer"   opencv_gstreamer
row "opencv CUDA"        opencv_cuda
if [ -z "$DOCKER_REPORT" ]; then
  echo
  echo "  (No Docker image probed — '$DOCKER_SRC'. Run 'make pull && make build'"
  echo "   to compare against the container, or 'make pull' for the base image.)"
fi

# --- Verdict: required components must be present on the host -----------------
echo
echo "==> Host prerequisite verdict"
required_missing=0
verdict() {  # $1=label  $2=condition(0=ok)  $3=required(1/0)  $4=fix-hint
  if [ "$2" = "ok" ]; then
    echo "  [ok]   $1"
  elif [ "$3" = "1" ]; then
    echo "  [MISS] $1 (required) — $4"
    required_missing=1
  else
    echo "  [warn] $1 (optional) — $4"
  fi
}
[ "${HOST[ort_cuda]:-}" = "yes" ] && r1=ok || r1=no
[ "${HOST[opencv_gstreamer]:-}" = "yes" ] && r2=ok || r2=no
[ "${HOST[torch_cuda]:-}" = "yes" ] && r3=ok || r3=no
[ "${HOST[tensorrt]:--}" != "-" ] && r4=ok || r4=no
verdict "onnxruntime-gpu (CUDA provider)" "$r1" 1 \
  "install the Jetson onnxruntime-gpu wheel (Jetson Zoo / NVIDIA index)"
verdict "OpenCV built WITH GStreamer" "$r2" 1 \
  "sudo apt-get install python3-opencv  (JetPack build; do NOT pip install opencv-python)"
verdict "torch (CUDA)" "$r3" 0 \
  "install the Jetson torch wheel (only for source=torchvision/huggingface/smp)"
verdict "tensorrt bindings" "$r4" 0 \
  "sudo apt-get install python3-libnvinfer  (only for backend=tensorrt)"

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
