#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Use your virtualenv
source env/bin/activate

# Optional: start with a clean logs folder once
# Comment this out if you want to append to existing logs
rm -rf logs

# Make filename globbing fail gracefully if no match
shopt -s nullglob

# Extensions you want to process
VIDEO_DIR="videos/ufpr_tracks"
FILES=( "$VIDEO_DIR"/*.mp4 "$VIDEO_DIR"/*.MP4 "$VIDEO_DIR"/*.mkv "$VIDEO_DIR"/*.avi )

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No video files found in $VIDEO_DIR"
  exit 1
fi

for vid in "${FILES[@]}"; do
  echo "==============================="
  echo "Processing: $vid"
  echo "==============================="

  python vehicle_detection_pipeline.py \
    --source "$vid" \
    --no-show \
    --dedupe-minutes 10
    # --dedupe-minutes 0 add to remove duplicated car feature

  echo
done

echo "All videos processed."
echo "Raw logs:    $(realpath logs/raw/vehicle_events.csv)"
echo "Refined log: $(realpath logs/vehicles_refined.csv)"
echo "Images:      $(realpath logs/images)"
