# syntax=docker/dockerfile:1
# AI Model Benchmarking — Jetson Orin (JetPack 6.x / L4T r36.x, CUDA 12.6)
#
# Base: dustynv/l4t-ml — prebuilt, Jetson-compiled GPU stack:
#   PyTorch, torchvision, onnxruntime-gpu, TensorRT bindings, and
#   OpenCV built WITH CUDA + GStreamer (needed for hw video decode / CSI).
#
# Host is L4T r36.5.0; nearest published dustynv tag is r36.4.0. This is safe:
# CUDA 12.6 is constant across JP6.1/6.2 and the actual driver libs are mounted
# from the host at runtime by the nvidia container runtime.
FROM dustynv/l4t-ml:r36.4.0

# --- Install deps via the SHARED installer (Layer 1) -------------------------
# scripts/install_deps.sh is the single source of truth for dependency setup —
# the host venv (scripts/setup_venv.sh) runs the exact same script, so the
# container and the venv resolve to the same dependency set. It freezes the
# Jetson-built GPU packages already in this base image into a constraints file
# and installs the pure-python deps under it, so pip never replaces the
# (GPU-enabled, aarch64) torch/torchvision/onnxruntime-gpu/opencv builds.
COPY requirements-deps.txt /tmp/requirements-deps.txt
COPY scripts/install_deps.sh /tmp/install_deps.sh
RUN --network=host REQ_FILE=/tmp/requirements-deps.txt \
        CONSTRAINTS=/opt/jetson-constraints.txt \
        bash /tmp/install_deps.sh

# Project code. The repo is also bind-mounted over /workspace at runtime (see
# docker-compose.yml) so live edits don't need a rebuild; this COPY just makes
# the image self-contained if run without the mount.
WORKDIR /workspace
COPY . /workspace

# Run scripts from the repo root; `src` is importable via PYTHONPATH (avoids
# relying on `pip install -e .`, whose metadata would be hidden by the mount).
ENV PYTHONPATH=/workspace \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

CMD ["python3", "scripts/run_benchmark.py"]
