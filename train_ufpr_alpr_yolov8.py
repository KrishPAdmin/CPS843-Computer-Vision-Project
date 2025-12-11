#!/usr/bin/env python3
"""
Train a YOLOv8 license plate detector on the UFPR-ALPR dataset.

Assumes YOLO-format data under:

    /home/krishadmin/vehicle_project/datasets/ufpr_alpr_yolo/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test

and a data.yaml file at:

    /home/krishadmin/vehicle_project/datasets/ufpr_alpr_yolo/data.yaml
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 license plate detector on UFPR-ALPR."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="/home/krishadmin/vehicle_project/datasets/ufpr_alpr_yolo/data.yaml",
        help="Path to YOLO data.yaml for UFPR-ALPR.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Base weights to start from (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (square).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (adjust if you hit OOM).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="/home/krishadmin/vehicle_project/runs/ufpr_alpr",
        help="Root folder for YOLO training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n_plate",
        help="Name of this specific training run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help=(
            "Device to use. '0' for first GPU, '1' for second GPU, 'cpu' for CPU. "
            "Default: auto-detect GPU if available."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("[UFPR-ALPR] YOLOv8 license plate training")
    print("=" * 60)
    print(f"[INFO] data.yaml   : {args.data}")
    print(f"[INFO] weights     : {args.weights}")
    print(f"[INFO] epochs      : {args.epochs}")
    print(f"[INFO] batch size  : {args.batch}")
    print(f"[INFO] image size  : {args.imgsz}")
    print(f"[INFO] project dir : {args.project}")
    print(f"[INFO] run name    : {args.name}")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA is available. Using device: {args.device}")
    else:
        print("[WARN] CUDA is NOT available. Training will run on CPU.")

    # Load base model (pretrained on COCO by default)
    model = YOLO(args.weights)

    # Make sure project directory exists
    Path(args.project).mkdir(parents=True, exist_ok=True)

    # Kick off training
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,  # "0" will use your 4060 Ti
        project=args.project,
        name=args.name,
        workers=8,
        pretrained=True,  # fine-tune from given weights
        exist_ok=True,    # overwrite previous run with same name
    )

    run_dir = Path(args.project) / args.name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    print("=" * 60)
    print("[DONE] Training complete.")
    print(f"[INFO] Best model: {best_weights}")
    print(f"[INFO] Last model: {last_weights}")
    print("=" * 60)
    print("[TIP] Use 'best.pt' in your pipeline for license plate detection.")


if __name__ == "__main__":
    main()
