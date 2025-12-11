#!/usr/bin/env bash
set -e

UFPR_ROOT="UFPR-ALPR dataset"
OUT_ROOT="videos/ufpr_tracks"

mkdir -p "$OUT_ROOT"

# Loop over splits and tracks
for split in training validation testing; do
  split_dir="$UFPR_ROOT/$split"
  [ -d "$split_dir" ] || continue

  echo "Processing split: $split"

  for track_dir in "$split_dir"/track*; do
    [ -d "$track_dir" ] || continue

    track_name=$(basename "$track_dir")
    out_dir="$OUT_ROOT/$split"
    mkdir -p "$out_dir"

    out_file="$out_dir/${track_name}.mp4"

    echo "  - $track_name -> $out_file"

    ffmpeg -y \
      -framerate 30 \
      -pattern_type glob \
      -i "$track_dir/*.png" \
      -c:v libx264 \
      -pix_fmt yuv420p \
      "$out_file"
  done
done

echo "Done. Videos are under $OUT_ROOT"
