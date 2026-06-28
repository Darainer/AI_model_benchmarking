# Open Issues

## HW video decode (`nvv4l2decoder`) does not work in the container

**Status:** unresolved. Use `--decoder sw` for now (works, but decode-bound).

### Symptom
Running the benchmark on a video file with HW decode (`--decoder hw`, the default
for file inputs) hangs at the first frame read. Logs show:
```
Opening in BLOCKING MODE
Error while setting IOCTL
S_EXT_CTRLS for CUDA_GPU_ID failed
... v4l2 Unable to try format: Unknown error -1
... cannot query video width/height
(65s later) gst_sample_get_caps() returns NULL
```
This previously hung the Orin hard enough to require a restart (multiple stuck
containers spinning in the GStreamer error loop). Always run HW-decode tests
**bounded** (`gst-launch ... num-buffers=N`, `timeout 40 ...`) — never unbounded.

### Root cause (confirmed)
`/dev/v4l2-nvdec` is **`/dev/null` (major 1, minor 3)** on the **host itself**,
not just in the container. The NVDEC V4L2 device node does not really exist:
- No `/dev/video*` nodes on the host at all; no `/dev/v4l/` dir.
- Host kernel is **out-of-tree** (`KERNEL_VARIANT: oot`, L4T r36.5.0) — the NVDEC
  V4L2 kernel driver that would create a real node appears to be absent.
- The nvidia-container-toolkit injects `/dev/v4l2-nvdec` from its CSV
  (`/etc/nvidia-container-runtime/host-files-for-container.d/devices.csv`), but
  since the host node is `/dev/null`, the container gets a `/dev/null` too.

The gst plugin `libgstnvvideo4linux2.so` issues **real** v4l2 ioctls on this fd
(it does **not** link `libnvv4l2.so`, the userspace emulation). Every ioctl on
`/dev/null` returns -1 → format probe fails → decoder never negotiates → stall.

### Ruled out
- **Not** a codec mismatch: file is H.264 (`avc1`), 1920×1080, 30fps; `h264parse`
  parses it fine. (`/home/ryan/Downloads/DJI_0118.MP4`)
- **Not** a userspace/kernel lib ABI mismatch: container tegra libs
  (`libnvcuvidv4l2.so`, `libnvv4l2.so`, gst plugin) are byte-identical to the
  host's (same sizes, same Jan-16 build; 1h timestamp diff is just timezone).
- **Not** the compose `devices:` mounts: a plain `docker run --runtime nvidia`
  with **no** manual device mounts produces the identical `/dev/null` node and
  the same failure.
- `CUDA_GPU_ID`/IOCTL errors alone are **non-fatal** — they also appear in the
  working SW run (where `decodebin` silently falls back to `avdec_h264`).

### Key realization about the "SW" run
`--decoder sw` uses `decodebin`, which **auto-selects software `avdec_h264`**
(HW `nvv4l2decoder` fails to plug). So the working benchmark is fully CPU-decoded.
This is why all classification models cluster at ~21–28 FPS on video despite
squeezenet hitting ~102 FPS on synthetic frames — **SW decode is the bottleneck**,
masking true inference throughput. Fixing HW decode genuinely matters for
benchmark validity.

### Next thing to try (was interrupted before testing)
Bypass the broken V4L2 path entirely and decode via the **CUDA NVCUVID API**:
- Check `cv2.cudacodec.createVideoReader()` (OpenCV CUDA video decode → NVDEC via
  libnvcuvid → `/dev/nvidia0`, which DOES work — cupy sees the Orin GPU).
- Note: a quick `ldd` showed only the v4l2 shim `libnvcuvidv4l2.so` present; the
  **core `libnvcuvid.so.1` was not found** — so cudacodec may also be unavailable
  and may need installing/mounting. Verify first.
- Alternative: `PyNvVideoCodec` (NVIDIA's Python NVDEC binding).
- If neither NVDEC path is available on this host, HW decode may be genuinely
  impossible without fixing the host JetPack multimedia stack (install/repair
  `nvidia-l4t-multimedia`, or boot the in-tree kernel so the NVDEC V4L2 node is
  created).

### Decision needed
1. Pursue NVCUVID/cudacodec HW decode, OR
2. Accept SW decode and **document it clearly** in results (decode-bound numbers),
   OR benchmark with `--input-type synthetic` / `image_dir` to isolate pure
   inference (no decode bottleneck) when HW decode isn't the point.

### Code state
- `scripts/run_benchmark.py` now has a `--decoder {hw,sw}` flag (added this
  session, not yet committed) that overrides `input.decoder`.
- Pipeline builder: `src/input/gstreamer_pipeline.py` (`hw` → explicit
  `nvv4l2decoder` chain; `sw` → `decodebin`).
- SW-decode full run results saved:
  `results/benchmark_2026-06-28_19-19-45.csv` (8 models, DJI_0118.MP4).
</content>
