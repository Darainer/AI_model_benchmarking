#!/usr/bin/env bash
# Extract the first N consecutive frames of an H.264 MP4 into a folder of JPEGs.
#
# Host-side helper (GStreamer, hardware decode via nvv4l2decoder — which works on
# the host even though it currently fails inside the container; see open_issues.md).
# The output folder is meant to be used as an `image_dir` benchmark input:
#
#   python scripts/run_benchmark.py --input-type image_dir --source data/frames_dji_0118
#
# image_dir feeds pre-decoded images via cv2.imread, so the timed loop measures
# pure model inference with no live video decode in it.
#
# Usage: scripts/extract_frames.sh VIDEO OUTDIR [COUNT]   (default COUNT=100)
set -euo pipefail

VIDEO="${1:?usage: extract_frames.sh VIDEO OUTDIR [COUNT]}"
OUTDIR="${2:?usage: extract_frames.sh VIDEO OUTDIR [COUNT]}"
COUNT="${3:-100}"

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/frame_*.jpg

# Decode a bit past COUNT, then trim to exactly COUNT. `identity eos-after` can be
# off-by-one and can linger after EOS, so we bound the pipeline with `timeout`;
# multifilesink closes each JPEG as it writes, so every file on disk is complete.
status=0
timeout 90 gst-launch-1.0 -e filesrc location="$VIDEO" \
  ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw,format=I420' \
  ! jpegenc quality=95 ! identity eos-after="$((COUNT + 5))" \
  ! multifilesink location="$OUTDIR/frame_%04d.jpg" >/dev/null 2>&1 || status=$?

# `timeout` kills the bounded pipeline with 124 once enough frames are written — that is
# the expected exit here and is not a failure. Any other non-zero status is real: 127 =
# gst-launch-1.0 not installed, others = unsupported codec / decoder init / bad input.
if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
  echo "extract_frames: gst-launch-1.0 failed (exit $status) — check that GStreamer is" \
       "installed and '$VIDEO' is a readable H.264 MP4." >&2
  exit "$status"
fi

# Keep only the first COUNT frames (sorted by sequential filename). nullglob → empty
# array (not a literal pattern) when the pipeline wrote nothing.
shopt -s nullglob
frames=("$OUTDIR"/frame_*.jpg)
if [ "${#frames[@]}" -gt "$COUNT" ]; then
  rm -f -- "${frames[@]:COUNT}"
  frames=("${frames[@]:0:COUNT}")
fi

# Don't report success on an empty/partial extraction — that would feed a zero-frame or
# under-sampled image_dir into the benchmark and record meaningless results.
written=${#frames[@]}
if [ "$written" -lt "$COUNT" ]; then
  echo "extract_frames: only $written/$COUNT frame(s) extracted to $OUTDIR —" \
       "extraction incomplete (clip shorter than COUNT, decode failure, or wrong input?)." >&2
  exit 1
fi

echo "Wrote $written frame(s) to $OUTDIR"
