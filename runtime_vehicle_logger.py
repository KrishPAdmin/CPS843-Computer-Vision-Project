#!/usr/bin/env python3
"""
Runtime vehicle logger

How to run:

1) Webcam (default camera 0):
   python runtime_vehicle_logger.py --source 0

2) Video file:
   python runtime_vehicle_logger.py --source /path/to/video.mp4

3) Disable window display (headless):
   python runtime_vehicle_logger.py --source /path/to/video.mp4 --no-show

4) Test mode (reset logs folder first):
   python runtime_vehicle_logger.py --source /path/to/video.mp4 --no-show --test 1
"""

import argparse
import csv
import datetime
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


# Classes we consider as "vehicles"
VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorbike", "motorcycle"}


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle tracking and logging")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (0,1,2,...) or video file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="CSV log file path (default: ./logs/vehicle_events.csv)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable display window",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="If 1, delete existing logs folder before running (test mode)",
    )
    return parser.parse_args()


def init_log(log_path: Path):
    """
    Open the log file in append mode.
    If it does not exist, write header.
    Returns (file_handle, csv_writer).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    f = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(
            [
                "run_id",
                "event_time_utc",
                "video_time_sec",
                "source",
                "track_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "vehicle_crop_path",
            ]
        )
        f.flush()

    return f, writer


def estimate_fps_for_file(source_path: str) -> float:
    """
    Try to estimate FPS for a video file using OpenCV.
    Fallback to 30.0 if it fails.
    """
    fps_default = 30.0
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return fps_default
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return fps_default
    return float(fps)


def now_utc_iso():
    """Return timezone aware UTC timestamp as ISO string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_run_id():
    """Return a compact UTC run id string."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")


def main():
    args = parse_args()

    # Resolve script directory and default log path
    script_dir = Path(__file__).resolve().parent
    if args.log_path is None:
        log_path = script_dir / "logs" / "vehicle_events.csv"
    else:
        log_path = Path(args.log_path).expanduser().resolve()

    logs_dir = log_path.parent

    # Test mode: delete entire logs folder first
    if args.test == 1 and logs_dir.exists():
        print(f"[TEST] Removing logs folder: {logs_dir}")
        shutil.rmtree(logs_dir, ignore_errors=True)

    # Decide how to pass the source to YOLO
    source_str = args.source.strip()

    # If source looks like a camera index (pure digit and short), convert to int
    if source_str.isdigit() and len(source_str) == 1:
        source = int(source_str)
        is_webcam = True
    else:
        source = source_str
        is_webcam = False

    # Estimate FPS for video files
    if is_webcam:
        fps = 30.0
    else:
        fps = estimate_fps_for_file(source_str)

    # Prepare logging
    log_file, log_writer = init_log(log_path)
    run_id = new_run_id()
    crops_dir = logs_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # Class index to name
    class_names = model.names

    seen_ids = set()
    frame_idx = 0
    show = not args.no_show

    print(f"[INFO] Starting tracking on source: {source}")
    print(f"[INFO] Log file: {log_path}")
    print(f"[INFO] Crops folder: {crops_dir}")
    if show:
        print("[INFO] Press 'q' in the window to quit.")
    else:
        print("[INFO] Display is disabled (no-show).")

    try:
        results_generator = model.track(
            source=source,
            imgsz=args.imgsz,
            stream=True,
            persist=True,
            verbose=False,
            show=False,  # we handle display ourselves if enabled
        )

        for res in results_generator:
            frame_idx += 1
            video_time_sec = frame_idx / fps if fps > 0 else 0.0

            # Original frame for drawing and cropping
            frame = res.orig_img

            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                # No detections in this frame
                if show:
                    try:
                        cv2.imshow("Vehicle tracking", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except Exception as e:
                        print(f"[WARN] Display error, disabling show: {e}")
                        show = False
                continue

            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id  # tensor or None

            event_time_utc = now_utc_iso()

            for i in range(len(boxes)):
                class_id = int(cls[i])
                class_name = class_names.get(class_id, str(class_id))

                # Only log vehicle classes
                if class_name not in VEHICLE_CLASS_NAMES:
                    continue

                if ids is None:
                    continue

                track_id_tensor = ids[i]
                if track_id_tensor is None:
                    continue
                try:
                    track_id = int(track_id_tensor.item())
                except Exception:
                    track_id = int(track_id_tensor)

                # Only log first time we see this track id
                if track_id in seen_ids:
                    continue
                seen_ids.add(track_id)

                x1, y1, x2, y2 = map(int, xyxy[i])
                conf_i = float(conf[i])

                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                crop = frame[y1:y2, x1:x2].copy()
                crop_name = f"{run_id}_veh{track_id}_f{frame_idx}.jpg"
                crop_path = crops_dir / crop_name

                if crop.size > 0:
                    try:
                        cv2.imwrite(str(crop_path), crop)
                    except Exception as e:
                        print(f"[WARN] Could not write crop {crop_path}: {e}")
                        crop_path = Path("")
                else:
                    crop_path = Path("")

                # Append to CSV log
                log_writer.writerow(
                    [
                        run_id,
                        event_time_utc,
                        f"{video_time_sec:.3f}",
                        str(source),
                        track_id,
                        class_name,
                        f"{conf_i:.3f}",
                        x1,
                        y1,
                        x2,
                        y2,
                        str(crop_path),
                    ]
                )
                log_file.flush()

                print(
                    f"[EVENT] run={run_id} id={track_id} class={class_name} "
                    f"conf={conf_i:.2f} t={video_time_sec:.2f}s"
                )

            if show:
                try:
                    annotated_frame = res.plot()
                    cv2.imshow("Vehicle tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception as e:
                    print(f"[WARN] Display error, disabling show: {e}")
                    show = False

    finally:
        log_file.close()
        if show:
            cv2.destroyAllWindows()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
