# Dockerized setup (Jetson Orin / JetPack 6.x)

Containerized so the GPU benchmark stack (onnxruntime-gpu, TensorRT, torch,
GStreamer-enabled OpenCV) is reproducible and isolated from the JetPack system
Python — which is the whole point of a *benchmark*.

## Quickstart

```bash
make pull        # one-time: pull the ~10–15 GB dustynv base image
make build       # build our flavor on top (fast — just pip + COPY)
make gpu-check   # confirm onnxruntime sees CUDA/TensorRT providers
make download    # download/export model checkpoints into ./models
make bench       # run the synthetic benchmark
```

Common runs:

```bash
make bench-cpu                              # force CPU only
make bench-file VIDEO=/workspace/clip.mp4   # benchmark a real video
make shell                                  # interactive shell in the container
```

## How it's built

- **Base:** `dustynv/l4t-ml:r36.4.0` — a prebuilt, Jetson-compiled image with the
  full GPU ML stack already working on aarch64. We add only pure-python deps and
  our code on top.
- **Shared dependency install:** the Dockerfile does not install deps inline — it
  `COPY`s and runs `scripts/install_deps.sh` (with `requirements-deps.txt`), the
  **same script the host venv uses** (`scripts/setup_venv.sh`). One source of
  truth keeps the container and the venv consistent. See [VENV.md](VENV.md) for
  the two-layer design and when to prefer a venv (HW-decoded video/camera runs).
- **Live edits:** `docker-compose.yml` bind-mounts the repo at `/workspace`, so
  code/config changes take effect without rebuilding. `models/` and `results/`
  persist on the host as subdirs of that mount.
- **GPU access:** the host already has the `nvidia` container runtime registered
  (`/etc/docker/daemon.json`), so `runtime: nvidia` is all that's needed.

## Friction points (read before changing the Dockerfile)

1. **Tag mismatch (benign).** Host is L4T `r36.5.0`; nearest published base tag is
   `r36.4.0`. Fine because CUDA 12.6 is constant across JP6.1/6.2 and the real
   driver libs are mounted from the host at runtime.

2. **Never let pip replace the Jetson GPU packages.** The base ships
   Jetson-built `torch`, `torchvision`, `onnxruntime-gpu`, and `opencv`. Packages
   like `transformers`/`segmentation-models-pytorch` list these as dependencies;
   `scripts/install_deps.sh` pins the installed versions via a constraints file
   so pip reuses them instead of pulling generic (broken-on-aarch64) PyPI wheels.

3. **OpenCV + GStreamer.** Hardware video decode / CSI camera relies on the
   GStreamer-enabled OpenCV in the base. Do **not** add `opencv-python` to any
   requirements file — it would clobber that build. (This is why `ultralytics`,
   which hard-depends on `opencv-python`, is left out of the default image.)

4. **GPU at build time.** The default Docker runtime here is `runc`, not `nvidia`.
   That's fine: our build only pip-installs pure python (no GPU needed). If you
   later want to pre-build TensorRT engines *inside* the image, flip the daemon's
   default-runtime to `nvidia` first.

## Cameras / display

Uncomment the relevant block in `docker-compose.yml`:
- USB camera → pass through `/dev/video0`
- CSI camera → also mount `/tmp/argus_socket` (host Argus daemon)
- On-screen display → pass `DISPLAY` + `/tmp/.X11-unix`
