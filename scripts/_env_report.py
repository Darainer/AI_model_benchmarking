#!/usr/bin/env python3
# scripts/_env_report.py
# Emit a flat key=value manifest of the GPU/ML stack in the *current*
# interpreter. Used by scripts/setup_venv.sh to compare the host venv against
# the Docker base image (the same script is piped into `docker run`), so both
# columns of the comparison are produced by identical probing logic.
#
# Output is one `key=value` line per fact. Missing packages report `-`.
import importlib
import re
import sys

# label -> import module name
VERSIONED = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("onnxruntime", "onnxruntime"),
    ("opencv", "cv2"),
    ("tensorrt", "tensorrt"),
    ("pandas", "pandas"),
]


def version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return "-"


def main():
    print("python=%d.%d.%d" % sys.version_info[:3])
    for label, mod in VERSIONED:
        print("%s=%s" % (label, version(mod)))

    # Capability flags (runtime-dependent — for the container these reflect the
    # runtime the `docker run` used).
    try:
        import torch
        print("torch_cuda=%s" % ("yes" if torch.cuda.is_available() else "no"))
    except Exception:
        print("torch_cuda=-")

    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        print("ort_cuda=%s" % ("yes" if "CUDAExecutionProvider" in provs else "no"))
        print("ort_trt=%s" % ("yes" if "TensorrtExecutionProvider" in provs else "no"))
    except Exception:
        print("ort_cuda=-")
        print("ort_trt=-")

    try:
        import cv2
        bi = cv2.getBuildInformation()
        g = re.search(r"GStreamer:\s*(YES|NO)", bi)
        c = re.search(r"NVIDIA CUDA:\s*(YES|NO)", bi)
        print("opencv_gstreamer=%s" % ("yes" if g and g.group(1) == "YES" else "no"))
        print("opencv_cuda=%s" % ("yes" if c and c.group(1) == "YES" else "no"))
    except Exception:
        print("opencv_gstreamer=-")
        print("opencv_cuda=-")


if __name__ == "__main__":
    main()
