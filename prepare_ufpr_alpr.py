import csv
from pathlib import Path
import cv2


UFPR_ROOT = Path("UFPR-ALPR dataset")  # your folder with spaces
OUT_ROOT = Path("data/ufpr_alpr")

SPLITS = ["training", "validation", "testing"]


def parse_annotation(txt_path: Path):
    """
    Parse a UFPR-ALPR annotation .txt file into a dict.

    Assumes format like:

      camera: GoPro Hero4 Silver
      position_vehicle: 804 367 279 244
          type: car
          make: GM Chevrolet
          model: Classic
          year: 2009
      plate: AQY6388
      corners: 909,483 986,484 987,507 909,507
          char 1: 912 491 9 15
          ...
    """
    data = {
        "camera": None,
        "vehicle_bbox": None,   # (x, y, w, h)
        "vehicle_type": None,
        "vehicle_make": None,
        "vehicle_model": None,
        "vehicle_year": None,
        "plate_text": None,
        "plate_corners": [],   # [(x1,y1), ...]
        "chars": [],           # list of {index, char, x,y,w,h}
    }

    with txt_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        if ln.startswith("camera:"):
            data["camera"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("position_vehicle:"):
            parts = ln.split(":", 1)[1].split()
            x, y, w, h = map(int, parts)
            data["vehicle_bbox"] = (x, y, w, h)

        elif ln.startswith("type:"):
            data["vehicle_type"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("make:"):
            data["vehicle_make"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("model:"):
            data["vehicle_model"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("year:"):
            data["vehicle_year"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("plate:"):
            data["plate_text"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("corners:"):
            # four "x,y" tokens
            rest = ln.split(":", 1)[1].strip()
            pts = []
            for token in rest.split():
                xs, ys = token.split(",")
                pts.append((int(xs), int(ys)))
            data["plate_corners"] = pts

        elif ln.startswith("char "):
            # e.g. "char 1: 912 491 9 15"
            left, right = ln.split(":", 1)
            idx = int(left.split()[1])
            coords = list(map(int, right.split()))
            x, y, w, h = coords
            data["chars"].append(
                {
                    "index": idx,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    # attach character symbols using plate_text
    plate = data["plate_text"]
    if plate is not None:
        for ch in data["chars"]:
            idx = ch["index"] - 1  # char 1 is first
            if 0 <= idx < len(plate):
                ch["char"] = plate[idx]
            else:
                ch["char"] = None

    return data


def crop_plate(image, corners):
    """
    Take four corner points and return a simple axis-aligned crop.
    You can later replace this with a perspective transform if needed.
    """
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    plates_root = OUT_ROOT / "plates"
    plates_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_root = UFPR_ROOT / split
        if not split_root.exists():
            print(f"[WARN] Split folder missing: {split_root}")
            continue

        split_plate_dir = plates_root / split
        split_plate_dir.mkdir(parents=True, exist_ok=True)

        meta_path = OUT_ROOT / f"meta_{split}.csv"
        with meta_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "split",
                    "track",
                    "image_rel_path",
                    "plate_crop_rel_path",
                    "camera",
                    "vehicle_type",
                    "vehicle_make",
                    "vehicle_model",
                    "vehicle_year",
                    "plate_text",
                    "img_width",
                    "img_height",
                    "plate_x_min",
                    "plate_y_min",
                    "plate_x_max",
                    "plate_y_max",
                ]
            )

            for track_dir in sorted(split_root.glob("track*")):
                for txt_path in sorted(track_dir.glob("*.txt")):
                    img_path = txt_path.with_suffix(".png")
                    if not img_path.exists():
                        # some images might be .jpg in other datasets
                        jpg_path = txt_path.with_suffix(".jpg")
                        if jpg_path.exists():
                            img_path = jpg_path
                        else:
                            print(f"[WARN] No image for annotation: {txt_path}")
                            continue

                    ann = parse_annotation(txt_path)

                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"[WARN] Could not read image: {img_path}")
                        continue
                    h, w = img.shape[:2]

                    if not ann["plate_corners"]:
                        print(f"[WARN] No plate corners in: {txt_path}")
                        continue

                    plate_img, (x_min, y_min, x_max, y_max) = crop_plate(
                        img, ann["plate_corners"]
                    )

                    crop_name = f"{track_dir.name}_{img_path.stem}_plate.png"
                    crop_path = split_plate_dir / crop_name
                    cv2.imwrite(str(crop_path), plate_img)

                    writer.writerow(
                        [
                            split,
                            track_dir.name,
                            str(img_path.relative_to(UFPR_ROOT)),
                            str(crop_path.relative_to(OUT_ROOT)),
                            ann["camera"],
                            ann["vehicle_type"],
                            ann["vehicle_make"],
                            ann["vehicle_model"],
                            ann["vehicle_year"],
                            ann["plate_text"],
                            w,
                            h,
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                        ]
                    )

        print(f"[OK] Wrote {meta_path}")


if __name__ == "__main__":
    main()
