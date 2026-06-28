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
timeout 90 gst-launch-1.0 -e filesrc location="$VIDEO" \
  ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw,format=I420' \
  ! jpegenc quality=95 ! identity eos-after="$((COUNT + 5))" \
  ! multifilesink location="$OUTDIR/frame_%04d.jpg" >/dev/null 2>&1 || true

# Keep only the first COUNT frames (sorted by sequential filename).
mapfile -t frames < <(ls "$OUTDIR"/frame_*.jpg 2>/dev/null | sort)
if [ "${#frames[@]}" -gt "$COUNT" ]; then
  printf '%s\n' "${frames[@]:COUNT}" | xargs -r rm -f
fi

echo "Wrote $(ls "$OUTDIR"/frame_*.jpg 2>/dev/null | wc -l) frame(s) to $OUTDIR"
