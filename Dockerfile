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

# --- The critical bit: do NOT let pip replace the Jetson-built GPU packages ---
# transformers/smp/etc. list torch, torchvision, opencv etc. as dependencies.
# Pinning the already-installed versions as constraints makes pip treat them as
# satisfied and leaves the (GPU-enabled, aarch64) builds untouched.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip freeze 2>/dev/null \
      | grep -iE '^(torch|torchvision|torchaudio|onnxruntime|onnxruntime-gpu|numpy|opencv|pandas)' \
      > /opt/jetson-constraints.txt || true && \
    echo "----- pinned (kept) packages -----" && cat /opt/jetson-constraints.txt

# Pure-python deps our pipeline adds on top of the base image.
# opencv-python and onnxruntime-gpu are deliberately absent — they come from
# the base and must not be reinstalled from PyPI.
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -c /opt/jetson-constraints.txt \
        -r /tmp/requirements-docker.txt

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
