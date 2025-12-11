import argparse
import csv
from pathlib import Path

# Project root = folder where this script lives
ROOT = Path(__file__).resolve().parent

# Default UFPR dataset root (note the space in the directory name – Path handles it fine)
DEFAULT_DATASET_ROOT = ROOT / "UFPR-ALPR dataset"

# Default output CSV
DEFAULT_OUTPUT = ROOT / "data" / "ufpr_alpr" / "ufpr_groundtruth_plates.csv"


def parse_annotation_file(txt_path: Path):
    """
    Parse a single UFPR-ALPR annotation .txt file.

    Returns a dict with fields:
      camera, vehicle_type, vehicle_make, vehicle_model, vehicle_year,
      plate, vehicle_x, vehicle_y, vehicle_w, vehicle_h, plate_corners
    Any missing fields are left as empty strings.
    """
    camera = ""
    vehicle_type = ""
    vehicle_make = ""
    vehicle_model = ""
    vehicle_year = ""
    plate = ""
    vehicle_x = vehicle_y = vehicle_w = vehicle_h = ""
    plate_corners = ""

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Normalize indentation
                low = line.lower()

                if low.startswith("camera:"):
                    camera = line.split(":", 1)[1].strip()

                elif low.startswith("position_vehicle:"):
                    # position_vehicle: x y w h
                    coords = line.split(":", 1)[1].strip().split()
                    if len(coords) == 4:
                        vehicle_x, vehicle_y, vehicle_w, vehicle_h = coords

                elif "type:" in low and low.startswith(("type:", "\ttype:", "type :")):
                    vehicle_type = line.split(":", 1)[1].strip()

                elif "make:" in low and low.startswith(("make:", "\tmake:", "make :")):
                    vehicle_make = line.split(":", 1)[1].strip()

                elif "model:" in low and low.startswith(("model:", "\tmodel:", "model :")):
                    vehicle_model = line.split(":", 1)[1].strip()

                elif "year:" in low and low.startswith(("year:", "\tyear:", "year :")):
                    vehicle_year = line.split(":", 1)[1].strip()

                elif low.startswith("plate:"):
                    plate = line.split(":", 1)[1].strip()

                elif low.startswith("corners:"):
                    # corners: x1,y1 x2,y2 x3,y3 x4,y4
                    plate_corners = line.split(":", 1)[1].strip()

    except Exception as e:
        print(f"[WARN] Failed to parse {txt_path}: {e}")

    return {
        "camera": camera,
        "vehicle_type": vehicle_type,
        "vehicle_make": vehicle_make,
        "vehicle_model": vehicle_model,
        "vehicle_year": vehicle_year,
        "plate": plate,
        "vehicle_x": vehicle_x,
        "vehicle_y": vehicle_y,
        "vehicle_w": vehicle_w,
        "vehicle_h": vehicle_h,
        "plate_corners": plate_corners,
    }


def build_csv(dataset_root: Path, output_csv: Path):
    """
    Walk UFPR-ALPR dataset (training / validation / testing),
    parse all .txt annotation files, and write a single CSV.
    """
    splits = ["training", "validation", "testing"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "track_dir",
            "image_filename",
            "image_rel_path",
            "camera",
            "vehicle_type",
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
            "plate",
            "vehicle_x",
            "vehicle_y",
            "vehicle_w",
            "vehicle_h",
            "plate_corners",
        ])

        total = 0

        for split in splits:
            split_dir = dataset_root / split
            if not split_dir.is_dir():
                print(f"[WARN] Split folder missing, skipping: {split_dir}")
                continue

            # e.g. training/track0001, training/track0002, ...
            for track_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for txt_path in sorted(track_dir.glob("*.txt")):
                    # Corresponding image: same basename with .png or .jpg
                    stem = txt_path.stem  # e.g. 'track0001[01]'
                    img_path = txt_path.with_suffix(".png")
                    if not img_path.is_file():
                        jpg_candidate = txt_path.with_suffix(".jpg")
                        if jpg_candidate.is_file():
                            img_path = jpg_candidate

                    # Relative path to image from dataset root (for easy joining later)
                    if img_path.is_file():
                        image_rel_path = img_path.relative_to(dataset_root)
                        image_filename = img_path.name
                    else:
                        image_rel_path = txt_path.relative_to(dataset_root)
                        image_filename = stem
                        print(f"[WARN] Image file missing for {txt_path}, using txt path instead.")

                    meta = parse_annotation_file(txt_path)

                    writer.writerow([
                        split,
                        track_dir.name,
                        image_filename,
                        str(image_rel_path),
                        meta["camera"],
                        meta["vehicle_type"],
                        meta["vehicle_make"],
                        meta["vehicle_model"],
                        meta["vehicle_year"],
                        meta["plate"],
                        meta["vehicle_x"],
                        meta["vehicle_y"],
                        meta["vehicle_w"],
                        meta["vehicle_h"],
                        meta["plate_corners"],
                    ])
                    total += 1

        print(f"[OK] Wrote {total} annotations to {output_csv}")


def main():
    ap = argparse.ArgumentParser(
        description="Flatten UFPR-ALPR annotation .txt files into a single CSV."
    )
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to 'UFPR-ALPR dataset' directory.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output)

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    build_csv(dataset_root, output_csv)


if __name__ == "__main__":
    main()
