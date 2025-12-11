import os
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")  # silence NNPACK spam

import argparse
import csv
import datetime
import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None

# ----------------------------
# GPU / device diagnostics
# ----------------------------

def print_device_info():
    print(f"[DEVICE] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            print(f"[DEVICE] CUDA devices: {count}")
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                print(f"[DEVICE]  - GPU {i}: {name}")
        except Exception as e:
            print(f"[DEVICE] Failed to query CUDA devices: {e}")
    else:
        print("[DEVICE] No CUDA GPU detected, running on CPU.")


# ----------------------------
# Global device selection
# ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    try:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

print_device_info()
print(f"[DEVICE] DEVICE={DEVICE}, YOLO_DEVICE={YOLO_DEVICE}")

# ----------------------------
# Paths & constants
# ----------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FINAL_DATA = ROOT / "final_data"
LOGS = ROOT / "logs"
RAW = LOGS / "raw"
RAW_CROPS = RAW / "crops"
REFINED_IMG = LOGS / "images"

RAW_CSV = RAW / "vehicle_events.csv"
REFINED_CSV = LOGS / "vehicles_refined.csv"

YOLO_WEIGHTS = FINAL_DATA / "yolo" / "vehicle_detector.pt"

VEHICLE_CLASSES = {2, 3, 5, 7}  # YOLO classes: car, motorcycle, bus, truck


# ----------------------------
# Utility functions
# ----------------------------

def ensure_dirs():
    LOGS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    RAW_CROPS.mkdir(parents=True, exist_ok=True)
    REFINED_IMG.mkdir(parents=True, exist_ok=True)


def current_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def estimate_color(bgr: np.ndarray) -> str:
    if bgr is None or bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = [float(x.mean()) for x in cv2.split(hsv)]
    if v < 50:
        return "black"
    if v > 200 and s < 40:
        return "white"
    if s < 40:
        return "gray"
    if h < 10 or h >= 160:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    if 130 <= h < 160:
        return "purple"
    return "unknown"


def clean_plate_text(t: str) -> str:
    import re
    t = str(t).upper()
    return re.sub(r"[^A-Z0-9]", "", t)


def parse_car_label(label: str) -> Tuple[str, str, str]:
    """
    Parse a label string like 'Volkswagen Fox Car 2015' or
    'Acura TL Sedan 2012' into (make, model, body_type).
    """
    toks = label.strip().split()
    if not toks:
        return "unknown", "unknown", "unknown"

    # drop trailing 4-digit year if present
    if toks[-1].isdigit() and len(toks[-1]) == 4:
        toks = toks[:-1]

    body_words = {
        "Sedan", "SUV", "Coupe", "Convertible", "Hatchback",
        "Wagon", "Minivan", "Van", "Pickup", "Roadster",
        # UFPR / generic body types
        "Car", "Motorcycle", "Bike", "Truck", "Bus"
    }

    body = "unknown"
    remain = []
    for t in toks:
        if t in body_words and body == "unknown":
            body = t
        else:
            remain.append(t)

    if not remain:
        return "unknown", "unknown", body

    make = remain[0]
    model = " ".join(remain[1:]) if len(remain) > 1 else "unknown"
    return make, model, body


def is_plate_like(cleaned: str) -> bool:
    """
    Decide whether a cleaned OCR string looks like a plate.

    - length 5–8
    - must contain at least one letter and one digit
    """
    if not cleaned:
        return False
    n = len(cleaned)
    if n < 5 or n > 8:
        return False
    has_letter = any(c.isalpha() for c in cleaned)
    has_digit = any(c.isdigit() for c in cleaned)
    if not (has_letter and has_digit):
        return False
    return True


def normalize_plate_for_dedupe(plate: str) -> str:
    """
    Normalize plate for deduplication so that small OCR confusions like
    I/L/1 and O/0 don't create separate logical plates.

    This is ONLY used for dedupe keys; the logged plate_text stays unmodified.
    """
    plate = plate.upper()
    trans = str.maketrans({
        "I": "L",
        "1": "L",
        "O": "0",
        "Q": "0",
    })
    return plate.translate(trans)


# Simple brand dictionary for OCR-based make override
BRAND_KEYWORDS = {
    "SUZUKI": "Suzuki",
    "VOLKSWAGEN": "Volkswagen",
    "VW": "Volkswagen",
    "CHEVROLET": "Chevrolet",
    "CHEVY": "Chevrolet",
    "FORD": "Ford",
    "FIAT": "Fiat",
    "HONDA": "Honda",
    "TOYOTA": "Toyota",
    "NISSAN": "Nissan",
    "RENAULT": "Renault",
    "PEUGEOT": "Peugeot",
    "CITROEN": "Citroen",
    "MERCEDES": "Mercedes-Benz",
    "BENZ": "Mercedes-Benz",
    "BMW": "BMW",
    "AUDI": "Audi",
    "KIA": "Kia",
    "HYUNDAI": "Hyundai",
}


def extract_brand_from_ocr(results) -> Optional[str]:
    """
    Look through EasyOCR results for brand names like 'SUZUKI'.
    Uses fuzzy matching so even slightly wrong OCR (e.g. 'SU2UKI')
    still maps to SUZUKI.
    """
    import re
    import difflib

    best_brand = None
    best_score = 0.0

    for _, raw_txt, prob in results:
        letters = re.sub(r"[^A-Z]", "", str(raw_txt).upper())
        if not letters:
            continue

        for key, pretty in BRAND_KEYWORDS.items():
            # Fuzzy similarity between OCR letters and brand key
            score = difflib.SequenceMatcher(None, letters, key).ratio()
            # Require at least 0.6 similarity to consider it
            if score >= 0.6 and score > best_score:
                best_score = score
                best_brand = pretty

    return best_brand


# ----------------------------
# Car make/model classifier
# ----------------------------

def _find_car_classifier_files() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Try several likely locations for weights and class mapping, in order:

    1) Environment variables (highest priority)
    2) final_data/car_classifier/...
    3) models/...
    4) last-resort: search for any class_mapping.json under ROOT
    """
    # 1) Env override
    env_w = os.environ.get("CAR_CLS_WEIGHTS")
    env_m = os.environ.get("CAR_CLS_CLASSMAP")
    if env_w and env_m:
        w = Path(env_w)
        m = Path(env_m)
        if w.exists() and m.exists():
            print(f"[CAR-CLF] Using weights/mapping from env:\n"
                  f"  weights: {w}\n  mapping: {m}")
            return w, m

    # 2 + 3) explicit candidates
    candidates = [
        (
            FINAL_DATA / "car_classifier" / "car_classifier_resnet50_hf.pt",
            FINAL_DATA / "car_classifier" / "class_mapping.json",
        ),
        (
            FINAL_DATA / "car_classifier" / "car_classifier_best.pth",
            FINAL_DATA / "car_classifier" / "class_mapping.json",
        ),
        (
            ROOT / "models" / "car_classifier_resnet50_hf.pt",
            ROOT / "models" / "class_mapping.json",
        ),
        (
            ROOT / "models" / "car_classifier_best.pth",
            ROOT / "models" / "class_mapping.json",
        ),
    ]

    for w, m in candidates:
        if w.exists() and m.exists():
            print(f"[CAR-CLF] Found weights/mapping:\n"
                  f"  weights: {w}\n  mapping: {m}")
            return w, m

    # 4) last resort – look for any class_mapping.json under ROOT
    mapping_candidates = list(ROOT.rglob("class_mapping.json"))
    if mapping_candidates:
        m = mapping_candidates[0]
        # Try to pair it with your known weights file if present
        default_weights = ROOT / "models" / "car_classifier_resnet50_hf.pt"
        if default_weights.exists():
            print(f"[CAR-CLF] Found mapping at {m} and weights at {default_weights}")
            return default_weights, m

    print("[CAR-CLF] Could not find any valid (weights, mapping) pair.\n"
          "  Looked in:\n"
          f"  - {FINAL_DATA / 'car_classifier'}\n"
          f"  - {ROOT / 'models'}\n"
          f"  - {ROOT}/**/class_mapping.json\n"
          "  Please ensure that:\n"
          "    models/car_classifier_resnet50_hf.pt\n"
          "    models/class_mapping.json\n"
          "  (or the final_data equivalents) exist, or set\n"
          "    CAR_CLS_WEIGHTS and CAR_CLS_CLASSMAP env vars.")
    return None, None


class CarMakeModelClassifier:
    """
    Wraps your fine-grained car classifier trained on Stanford Cars + UFPR.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or DEVICE)
        self.model = None
        self.transform = None
        self.class_map: Dict[str, str] = {}
        self.available = False
        self._init_model()

    def _init_model(self):
        weights_path, mapping_path = _find_car_classifier_files()
        if weights_path is None or mapping_path is None:
            print("[CAR-CLF] Classifier unavailable (missing weights or mapping).")
            return

        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.class_map = json.load(f)

            num_classes = len(self.class_map)
            print(f"[CAR-CLF] Loading car classifier ({num_classes} classes) "
                  f"from {weights_path} on device {self.device}...")

            base = models.resnet50(weights=None)
            base.fc = torch.nn.Sequential(
                torch.nn.Linear(base.fc.in_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, num_classes),
            )

            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]

            base.load_state_dict(state, strict=True)
            base.eval()
            base.to(self.device)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ])
            self.model = base
            self.available = True
            print(f"[CAR-CLF] Car classifier loaded and ready on {self.device}.")
        except Exception as e:
            print(f"[CAR-CLF] Failed to load classifier: {e}")
            self.available = False

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> Tuple[str, str, str, float]:
        if not self.available or bgr is None or bgr.size == 0:
            return "unknown", "unknown", "unknown", 0.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inp = self.transform(pil).unsqueeze(0).to(self.device, non_blocking=True)
        out = self.model(inp)
        prob = torch.softmax(out, dim=1)
        conf, idx = torch.max(prob, dim=1)
        label = self.class_map.get(str(int(idx.item())), "unknown")
        make, model, body = parse_car_label(label)
        return make, model, body, float(conf.item())


# ----------------------------
# Refiner Thread (OCR + classifier)
# ----------------------------

class Refiner(threading.Thread):
    def __init__(self, q: "queue.Queue", dedupe_min: float = 10.0):
        super().__init__(daemon=True)
        self.q = q
        self.stop = False
        self.reader = None
        self.classifier = None
        self.last_seen: Dict[str, datetime.datetime] = {}
        self.dedupe = dedupe_min

    def run(self):
        self._init_ocr()
        print("[REFINER] Initializing car make/model classifier...")
        self.classifier = CarMakeModelClassifier()
        if not self.classifier.available:
            print("[REFINER] Car classifier unavailable – make/model/body will stay 'unknown'.")
        f, writer = self._csv()
        try:
            while not self.stop:
                try:
                    job = self.q.get(timeout=1)
                except queue.Empty:
                    if self.stop:
                        break
                    continue
                if job is None:
                    break
                self._process(job, writer)
                f.flush()
        finally:
            f.close()
            print("[REFINER] Stopped.")

    def _init_ocr(self):
        if easyocr is None:
            print("[REFINER] easyocr not installed. Plate OCR disabled.")
            return
        for i in range(3):
            try:
                use_gpu = (DEVICE == "cuda")
                print(f"[REFINER] Initializing EasyOCR (attempt {i+1}/3, gpu={use_gpu})...")
                self.reader = easyocr.Reader(["en"], gpu=use_gpu)
                print("[REFINER] EasyOCR ready.")
                return
            except Exception as e:
                print(f"[REFINER] EasyOCR init failed: {e}")
                time.sleep(3)
        print("[REFINER] Failed to init EasyOCR. Plate OCR disabled.")
        self.reader = None

    def _csv(self):
        new = not REFINED_CSV.exists()
        f = open(REFINED_CSV, "a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if new:
            w.writerow([
                "run_id", "event_time_utc", "video_time_sec", "source", "track_id",
                "plate_text", "plate_conf", "detected_class", "body_type",
                "make", "model", "color", "final_image_path", "make_model_confidence"
            ])
        return f, w

    def _skip(self, plate: str, ts: datetime.datetime) -> bool:
        """
        Deduplicate based on a *normalized* plate string, so that
        small OCR confusions (I vs L, O vs 0) don't create duplicates.
        """
        if not plate or self.dedupe <= 0:
            return False

        norm = normalize_plate_for_dedupe(plate)
        last = self.last_seen.get(norm)
        if not last:
            self.last_seen[norm] = ts
            return False

        if (ts - last).total_seconds() / 60.0 < self.dedupe:
            return True

        self.last_seen[norm] = ts
        return False

    def _process(self, job, writer):
        if self.reader is None:
            return
        p = Path(job["raw_crop_path"])
        if not p.exists():
            return
        img = cv2.imread(str(p))
        if img is None:
            return

        # Run EasyOCR on the cropped vehicle image
        results = self.reader.readtext(img)
        if not results:
            return

        # --- Brand from OCR (for overriding make if we see SUZUKI, etc.) ---
        ocr_brand = extract_brand_from_ocr(results)

        # --- Plate OCR: filter to plate-like strings only ---
        best_plate = ""
        best_plate_conf = 0.0
        for _, raw_txt, prob in results:
            cleaned = clean_plate_text(raw_txt)
            if not is_plate_like(cleaned):
                continue
            prob = float(prob)
            if prob > best_plate_conf:
                best_plate = cleaned
                best_plate_conf = prob

        if not best_plate:
            # No plate-like text found
            return

        t = datetime.datetime.fromisoformat(job["event_time_utc"])
        if self._skip(best_plate, t):
            print(f"[REFINER] Skip duplicate plate {best_plate}.")
            return

        color = estimate_color(img)

        # --- Make / model classifier ---
        make, model, body, c2 = "unknown", "unknown", "unknown", 0.0
        if self.classifier and self.classifier.available:
            try:
                make, model, body, c2 = self.classifier.predict(img)

                # Confidence gating:
                # - c2 < 0.4: too uncertain -> everything unknown
                # - 0.4 <= c2 < 0.7: trust make/body more than model
                if c2 < 0.4:
                    make, model, body = "unknown", "unknown", "unknown"
                elif c2 < 0.7:
                    model = "unknown"

            except Exception as e:
                print("[REFINER] Classifier error:", e)
                make, model, body, c2 = "unknown", "unknown", "unknown", 0.0

        # --- Override make using OCR brand text when available ---
        if ocr_brand:
            # If classifier is low-ish confidence or disagrees, trust brand text
            if make == "unknown" or make.lower() != ocr_brand.lower() or c2 < 0.9:
                if make.lower() != ocr_brand.lower():
                    # we no longer trust the model for the exact model name
                    model = "unknown"
                make = ocr_brand

        fn = f"{job['run_id']}_veh{job['track_id']}.jpg"
        dst_rel = Path("images") / fn
        dst_abs = REFINED_IMG / dst_rel.name
        cv2.imwrite(str(dst_abs), img)

        writer.writerow([
            job["run_id"],
            job["event_time_utc"],
            job["video_time_sec"],
            job["source"],
            job["track_id"],
            best_plate,
            f"{best_plate_conf:.3f}",
            job["detected_class"],
            body,
            make,
            model,
            color,
            str(dst_rel),
            round(c2, 3),
        ])
        print(
            f"[REFINER] Refined {best_plate} → {make} {model} ({color}), "
            f"plate_conf={best_plate_conf:.3f}, make_model_conf={c2:.3f}"
        )


# ----------------------------
# Main runtime
# ----------------------------

def write_raw_header():
    new = not RAW_CSV.exists()
    f = open(RAW_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow([
            "run_id", "event_time_utc", "video_time_sec", "source", "track_id",
            "detected_class", "confidence", "x1", "y1", "x2", "y2", "raw_crop_path"
        ])
    return f, w


def load_yolo_model():
    """
    Load custom vehicle YOLO if present, else fall back to yolov8n.pt.
    """
    if YOLO_WEIGHTS.exists():
        print(f"[YOLO] Using custom weights from {YOLO_WEIGHTS}")
        return YOLO(str(YOLO_WEIGHTS))
    else:
        print("[YOLO] Custom weights not found in final_data/yolo.")
        print("[YOLO] Falling back to default 'yolov8n.pt' (COCO pretrained).")
        return YOLO("yolov8n.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source for YOLO (e.g. '0' for webcam or path to video)",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Reserved for future use (no display window).",
    )
    ap.add_argument(
        "--test",
        type=int,
        default=0,
        help="If 1, delete existing logs folder before run.",
    )
    ap.add_argument(
        "--dedupe-minutes",
        type=float,
        default=10.0,
        help="Minimum minutes between logging the same plate in the refined CSV.",
    )
    args = ap.parse_args()

    src = 0 if args.source == "0" else args.source

    if args.test == 1 and LOGS.exists():
        print(f"[TEST] Removing {LOGS}")
        shutil.rmtree(LOGS)
    ensure_dirs()

    q = queue.Queue()
    ref = Refiner(q, dedupe_min=args.dedupe_minutes)
    ref.start()

    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    print(f"[INFO] Run id: {run_id}")
    print(f"[INFO] Loading YOLO model; will run on device={YOLO_DEVICE}")

    model = load_yolo_model()

    print(f"[INFO] Starting source: {args.source}")
    f, w = write_raw_header()
    seen = set()
    frame_idx = 0
    fps = 30.0  # nominal FPS for video_time_sec; adjust if needed

    try:
        results = model.track(
            source=src,
            stream=True,
            verbose=False,
            persist=True,
            device=YOLO_DEVICE,
        )
        for res in results:
            frame_idx += 1
            t = frame_idx / fps

            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes
            ids = boxes.id
            cls = boxes.cls
            confs = boxes.conf
            xyxy = boxes.xyxy

            if ids is None:
                continue

            frame = res.orig_img
            h, wf = frame.shape[:2]

            for i in range(len(boxes)):
                tid = int(ids[i].item())
                if tid in seen:
                    continue
                seen.add(tid)

                cid = int(cls[i].item())
                if cid not in VEHICLE_CLASSES:
                    continue

                conf = float(confs[i].item())
                x1, y1, x2, y2 = map(int, xyxy[i].tolist())

                # Slightly enlarge the crop so we capture logos like "SUZUKI"
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(0.10 * box_w)
                pad_y = int(0.15 * box_h)

                x1p = max(0, x1 - pad_x)
                x2p = min(wf - 1, x2 + pad_x)
                y1p = max(0, y1 - pad_y)
                y2p = min(h - 1, y2 + pad_y)

                if x2p <= x1p or y2p <= y1p:
                    continue

                crop = frame[y1p:y2p, x1p:x2p]
                if crop is None or crop.size == 0:
                    continue

                fn = f"{run_id}_veh{tid}_f{frame_idx}.jpg"
                path = RAW_CROPS / fn
                cv2.imwrite(str(path), crop)

                evt = current_utc_iso()
                w.writerow([
                    run_id,
                    evt,
                    f"{t:.3f}",
                    args.source,
                    tid,
                    cid,
                    f"{conf:.3f}",
                    x1,
                    y1,
                    x2,
                    y2,
                    str(path),
                ])
                print(
                    f"[EVENT-RAW] run={run_id} id={tid} class={cid} "
                    f"conf={conf:.2f} t={t:.2f}s"
                )

                q.put({
                    "run_id": run_id,
                    "event_time_utc": evt,
                    "video_time_sec": f"{t:.3f}",
                    "source": args.source,
                    "track_id": tid,
                    "detected_class": str(cid),
                    "raw_crop_path": str(path),
                })
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        f.close()
        ref.stop = True
        q.put(None)
        ref.join()
        print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()
