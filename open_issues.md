# Open Issues

## HW video decode (`nvv4l2decoder`) fails **inside the container** (host is fine)

**Status:** root cause isolated — **container GPU/runtime layer**, *not* the host and
*not* the decoder hardware. HW decode works on the host. Container fix still to be
selected (investigation phase). Use `--decoder sw` meanwhile (works, but decode-bound).

> ⚠️ Always run HW-decode tests **bounded** (`gst-launch … num-buffers=N`, wrapped in
> `timeout 40 …`, and an outer `timeout` on any `docker` invocation). An unbounded
> `nvv4l2decoder` failure spins in a GStreamer error loop and has previously hung the
> Orin hard enough to need a restart.

### Symptom (container only)
HW decode (`--decoder hw`, the default for file inputs) hangs at preroll:
```
Opening in BLOCKING MODE
Error while setting IOCTL
S_EXT_CTRLS for CUDA_GPU_ID failed
```
It never reaches `NvMMLiteOpen` (the NVDEC block is never created) and stalls at preroll.

### Root cause (CONFIRMED by direct host-vs-container A/B test, 2026-06-28)
The earlier write-up in this file was a **misdiagnosis** and has been corrected. The host
NVDEC stack is fully intact; the failure is specific to the **container runtime layer**
(`nvidia-container-toolkit`). The `S_EXT_CTRLS for CUDA_GPU_ID` ioctl is `libnvv4l2.so`
telling the decoder which GPU to bind; that GPU-id negotiation fails inside the container
even though general CUDA (inference) works there.

**A/B evidence — identical bounded pipeline, identical `/dev/v4l2-nvdec` (c 1 3), identical libs:**

| | Host | Container |
|---|---|---|
| `… ! qtdemux ! h264parse ! nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! fakesink` | ✅ `NvMMLiteOpen BlockType=261`, NV12 1920×1080, **Got EOS, exit 0** | ❌ `S_EXT_CTRLS for CUDA_GPU_ID failed`, hang at preroll (killed by `timeout`) |

Host test (works):
```bash
timeout 40 gst-launch-1.0 -v filesrc location=/home/ryan/Downloads/DJI_0118.MP4 \
  ! qtdemux ! h264parse ! nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! fakesink num-buffers=30
```
Container test (reproduces the failure):
```bash
timeout 90 docker compose run --rm --no-deps -v /home/ryan/Downloads:/videos:ro \
  --entrypoint bash benchmark -lc \
  "timeout 40 gst-launch-1.0 -v filesrc location=/videos/DJI_0118.MP4 \
   ! qtdemux ! h264parse ! nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! fakesink num-buffers=30"
```

### Why the previous "host is broken" diagnosis was wrong
- **`/dev/v4l2-nvdec` = `c 1 3` is correct by design, not an accidental `/dev/null`.**
  NVIDIA's own udev rule creates it that way when the nvdec driver binds:
  `/etc/udev/rules.d/99-tegra-devices.rules` →
  `KERNEL=="15480000.nvdec", DRIVER=="tegra-nvdec", ACTION=="bind", RUN+="/bin/mknod -m 666 /dev/v4l2-nvdec c 1 3"`.
  A healthy Orin Nano has this exact node; the major/minor collision with `/dev/null` is
  a red herring. The node existing means the rule fired → the driver bound.
- **The NVDEC kernel driver is present and BOUND** (built into the stock Image, so no `.ko`):
  `/sys/bus/platform/drivers/tegra-nvdec/15480000.nvdec` is bound.
- **All multimedia userspace is already installed** (`nvidia-l4t-gstreamer`,
  `nvidia-l4t-multimedia`, `nvidia-l4t-jetson-multimedia-api`, `nvidia-l4t-kernel-oot-modules`,
  all `36.5.0`); `gst-inspect-1.0 nvv4l2decoder` works on the host. Kernel is **stock**
  NVIDIA `5.15.185-tegra` (buildbrain build), not custom. (`KERNEL_VARIANT: oot` is the
  normal JP6 arrangement, not a custom kernel.)
- The Orin Nano **has NVDEC** (decodes up to 8K); it only lacks NVENC *encode*.

### Dead ends (do NOT pursue — verified unavailable on this platform)
- **NVCUVID / `cv2.cudacodec`**: `libnvcuvid.so.1` is absent (only `libnvcuvidv4l2.so`).
- **PyNvVideoCodec**: NVIDIA dGPU-only; explicitly **not supported on Jetson**.
- Both, plus `jetson_multimedia_api` and ffmpeg `nvmpi`, route through the same NVDEC path
  that already works fine on the host — so there is nothing to "bypass". The only thing
  broken is container access; fix that, not the API.

### Secondary host finding (fixed this session)
The host was missing `h264parse` (package `gstreamer1.0-plugins-bad` — only `good`/`ugly`
were installed) and has no `ffmpeg`, so a host-side H.264 HW pipeline couldn't even be
built before. Installed `gstreamer1.0-plugins-bad` on the host to run the A/B test. (The
container image already ships `h264parse`.)

### Key realization about the "SW" run (unchanged, still true)
`--decoder sw` uses `decodebin`, which auto-selects software `avdec_h264`. So that
benchmark is fully CPU-decoded; classification models cluster at ~21–28 FPS on video while
squeezenet hits ~102 FPS on synthetic — **SW decode is the bottleneck**, masking true
inference throughput. Getting HW decode working in the container matters for validity.

### Next: select the container-side fix (investigation phase — not yet chosen)
Reproduce with the bounded container command above, then bisect:
1. **Container-toolkit version** — installed: `nvidia-container-toolkit 1.16.2-1`,
   `libnvidia-container1 1.16.2-1`, runtime `mode = "auto"`. The
   `S_EXT_CTRLS for CUDA_GPU_ID` hang is a documented `nvv4l2*`-in-container toolkit
   regression; validate 1.16.2 against the known-good line for L4T r36.5 (test a different
   version).
2. **Runtime mode** — try `mode = "csv"` vs `auto` in
   `/etc/nvidia-container-runtime/config.toml`.
3. **GPU/device access for the CUDA_GPU_ID path** — confirm the iGPU control nodes the
   ioctl touches are fully accessible (cgroup) in the container; compare a minimal
   `docker run --runtime nvidia <l4t-base>` repro to isolate from our compose setup.

### Code state
- `scripts/run_benchmark.py` has a `--decoder {hw,sw}` flag overriding `input.decoder`.
- Pipeline builder: `src/input/gstreamer_pipeline.py` (`hw` → explicit `nvv4l2decoder`
  chain; `sw` → `decodebin`). No code change needed for HW decode — it's an environment fix.
- SW-decode full run results: `results/benchmark_2026-06-28_19-19-45.csv` (8 models, DJI_0118.MP4).
