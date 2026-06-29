#!/usr/bin/env bash
# scripts/install_deps.sh
# =============================================================================
# LAYER 1 (shared) — the single source of truth for installing this project's
# pure-python dependencies ON TOP OF an existing Jetson GPU stack, without
# letting pip clobber the GPU-built packages.
#
# This is the ONE place dependency installation lives. It is invoked by BOTH:
#   - the host venv bootstrap   (scripts/setup_venv.sh), and
#   - the Docker image build     (Dockerfile),
# so the two environments resolve to the same dependency set. Change the deps
# in requirements-deps.txt; change *how* they're installed here.
#
# Assumption: the GPU stack (torch, torchvision, onnxruntime-gpu, CUDA+GStreamer
# OpenCV, TensorRT bindings) is ALREADY present in the active interpreter — from
# the dustynv/l4t-ml base image in Docker, or from JetPack apt + a
# --system-site-packages venv on the host. This script never installs those
# itself; it only layers the pure-python deps on top.
#
# Env overrides:
#   PYTHON        interpreter to install into   (default: python3)
#   REQ_FILE      requirements file to install  (default: <repo>/requirements-deps.txt)
#   CONSTRAINTS   constraints output path       (default: $TMPDIR/jetson-constraints.txt)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
REQ_FILE="${REQ_FILE:-$REPO_ROOT/requirements-deps.txt}"
CONSTRAINTS="${CONSTRAINTS:-${TMPDIR:-/tmp}/jetson-constraints.txt}"

echo "==> install_deps (shared layer)"
echo "    interpreter : $("$PY" -c 'import sys; print(sys.executable)')"
echo "    requirements: $REQ_FILE"
echo "    constraints : $CONSTRAINTS"

"$PY" -m pip install --no-cache-dir --upgrade pip

# --- Pin the GPU-built packages already in the environment --------------------
# transformers / segmentation-models-pytorch / rfdetr / ... list torch,
# torchvision, opencv etc. as dependencies. Freezing the installed versions into
# a constraints file makes pip treat them as already satisfied, so it reuses the
# Jetson (GPU-enabled, aarch64) builds instead of pulling generic — and
# broken-on-aarch64 — PyPI wheels.
echo "==> Freezing existing GPU packages into constraints"
"$PY" -m pip freeze 2>/dev/null \
  | grep -iE '^(torch|torchvision|torchaudio|onnxruntime|onnxruntime-gpu|numpy|opencv|opencv-python|opencv-python-headless|pandas|tensorrt)' \
  > "$CONSTRAINTS" || true
echo "----- pinned (kept) packages -----"
cat "$CONSTRAINTS" || true
echo "----------------------------------"

# --- Install the pure-python deps under those constraints ---------------------
echo "==> Installing pure-python deps from $REQ_FILE"
"$PY" -m pip install --no-cache-dir \
    -c "$CONSTRAINTS" \
    --extra-index-url https://pypi.org/simple \
    -r "$REQ_FILE"

echo "==> install_deps: done"
