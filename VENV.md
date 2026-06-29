# Host venv setup (Jetson Orin / JetPack 6.x)

A host virtual environment that **inherits the JetPack GPU stack** instead of
re-containerizing it. Use this for HW-decoded **video / camera** benchmarks:
the host NVDEC path (`nvv4l2decoder`) works directly here, with no
`nvidia-container-toolkit` layer to fail the `S_EXT_CTRLS for CUDA_GPU_ID`
ioctl that breaks HW decode inside the container (see [open_issues.md](open_issues.md)).

## Two-layer design (shared with Docker)

Dependency installation lives in **one** place and is reused by both
environments, so the venv and the container stay consistent:

```
                requirements-deps.txt        ← the shared dependency list
                          │
              scripts/install_deps.sh        ← LAYER 1: the shared installer
                  (freeze GPU pkgs → constraints → pip install deps)
                    ┌─────┴─────────┐
   scripts/setup_venv.sh         Dockerfile  ← LAYER 2: thin callers
        (host venv)               (container)
```

- **Layer 1 — `scripts/install_deps.sh`:** the single source of truth. Given an
  interpreter that already has the Jetson GPU stack, it freezes those GPU
  packages into a constraints file and installs `requirements-deps.txt` under
  it, so pip never replaces the aarch64 GPU builds.
- **Layer 2a — `scripts/setup_venv.sh`:** creates a `--system-site-packages`
  venv (so it sees the JetPack GPU stack), then calls Layer 1.
- **Layer 2b — `Dockerfile`:** `COPY`s and runs the *same* Layer 1 script on
  top of the `dustynv/l4t-ml` base.

Change deps in `requirements-deps.txt` once; both environments pick them up.

## Quickstart

```bash
make venv                 # verify host GPU stack, create .venv, install deps
source .venv/bin/activate
python scripts/download_models.py
python scripts/run_benchmark.py --video /path/to/clip.mp4   # HW-decoded
```

`make venv` is just `bash scripts/setup_venv.sh`.

## Prerequisites (must exist on the host first)

A venv isolates **Python packages only** — it does not supply the GPU builds.
`setup_venv.sh` checks for these before creating the venv and refuses if a
required one is missing:

| Package | Required? | How to get it on Jetson |
|---|---|---|
| `onnxruntime-gpu` (CUDA EP) | **yes** | Jetson Zoo / NVIDIA wheel index (not PyPI) |
| OpenCV built **with GStreamer** | **yes** | `sudo apt-get install python3-opencv` (JetPack build) |
| `torch` (+ `torchvision`) | optional | Jetson torch wheel — only for `source=torchvision/huggingface/smp` |
| `tensorrt` bindings | optional | `sudo apt-get install python3-libnvinfer` — only for `backend=tensorrt` |

> **Never `pip install opencv-python`.** The PyPI wheel has no GStreamer and
> would shadow the JetPack build, breaking HW decode / CSI capture. This is also
> why `ultralytics` (which hard-depends on `opencv-python`) is left out of
> `requirements-deps.txt`. The constraints file in Layer 1 guards against
> transitive pulls, but don't add it by hand.

## venv vs Docker — when to use which

| | Host venv | Docker |
|---|---|---|
| HW decode (`nvv4l2decoder`) | ✅ works (host NVDEC, no toolkit layer) | ❌ broken (container-toolkit ioctl) |
| GPU ML stack | inherited from JetPack (apt + NVIDIA wheels) | prebuilt in `dustynv/l4t-ml` |
| Reproducibility / portability | tied to the flashed JetPack | pinned image, JetPack-agnostic |
| Footprint / iteration | light, fast | ~10–15 GB base, rebuilds |

Rule of thumb: **venv for video/camera (valid HW-decode throughput); Docker for
synthetic / pure-inference runs where decode never enters the path** and a
pinned, portable image matters. Both install the same deps via Layer 1.

## Friction points

1. **`--system-site-packages` is leaky.** The venv can see *all* host
   site-packages. The Layer 1 constraints file pins torch/opencv/onnxruntime/
   numpy so a transitive dependency can't silently upgrade and break the ABI —
   but don't manually `pip install` generic GPU wheels into the venv.
2. **GPU stack is the host's responsibility.** If `make venv` reports a required
   package missing, install it via apt / the NVIDIA wheel index and re-run; the
   venv cannot create it.
3. **Constraints come from `pip freeze`.** Packages built from source without
   pip metadata (some base OpenCV builds) may not appear in the freeze and thus
   aren't pinned — another reason to never introduce `opencv-python` as a dep.
