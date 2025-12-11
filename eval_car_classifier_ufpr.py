import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import cv2
from torchvision import models, transforms
from PIL import Image

# --- Project paths (adjust if your layout differs) ---
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
UFPR_META_DIR = ROOT / "data" / "ufpr_alpr"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_car_label(label: str) -> Tuple[str, str, str]:
    toks = label.strip().split()
    if not toks:
        return "unknown", "unknown", "unknown"
    if toks[-1].isdigit() and len(toks[-1]) == 4:
        toks = toks[:-1]
    body_words = {
        "Sedan", "SUV", "Coupe", "Convertible", "Hatchback",
        "Wagon", "Minivan", "Van", "Pickup", "Roadster",
        "Car", "Motorcycle", "Truck", "Bus"
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


def ufpr_label_from_row(row: pd.Series,
                        make_col: str,
                        model_col: str,
                        body_col: Optional[str],
                        year_col: Optional[str]) -> str:
    make = str(row[make_col]).strip()
    model = str(row[model_col]).strip()

    parts = []
    if make and make.lower() != "nan":
        parts.append(make)
    if model and model.lower() != "nan":
        parts.append(model)

    if body_col:
        body = str(row[body_col]).strip()
        if body and body.lower() != "nan":
            parts.append(body.capitalize())

    if year_col:
        year = str(row[year_col]).strip()
        if year.isdigit() and len(year) == 4:
            parts.append(year)

    return " ".join(parts)


def infer_ufpr_columns(df: pd.DataFrame):
    cols = list(df.columns)

    def find(keywords):
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in keywords):
                return c
        return None

    image_col = find(["image", "path", "file"])
    make_col = find(["manufacturer", "make", "brand"])
    model_col = find(["model"])
    body_col = find(["type", "vehicle_type", "body"])
    year_col = find(["year"])

    if image_col is None or make_col is None or model_col is None:
        raise RuntimeError(
            f"Could not infer necessary UFPR columns from {cols}"
        )

    return image_col, make_col, model_col, body_col, year_col


def load_classifier() -> Tuple[torch.nn.Module, Dict[str, str]]:
    weights_path = MODELS_DIR / "car_classifier_resnet50_hf.pt"
    mapping_path = MODELS_DIR / "class_mapping.json"

    if not weights_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing weights or mapping:\n  {weights_path}\n  {mapping_path}"
        )

    with open(mapping_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    num_classes = len(class_map)
    print(f"[MODEL] Loading classifier with {num_classes} classes from {weights_path}")

    base = models.resnet50(weights=None)
    base.fc = torch.nn.Sequential(
        torch.nn.Linear(base.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, num_classes),
    )

    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    base.load_state_dict(state, strict=True)
    base.eval()
    base.to(DEVICE)

    return base, class_map


def main():
    meta_test = UFPR_META_DIR / "meta_testing.csv"
    if not meta_test.exists():
        raise FileNotFoundError(meta_test)

    df = pd.read_csv(meta_test)
    print(f"[DATA] meta_testing: {len(df)} rows")

    image_col, make_col, model_col, body_col, year_col = infer_ufpr_columns(df)

    model, class_map = load_classifier()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    total = 0
    full_correct = 0
    make_correct = 0
    body_correct = 0

    with torch.no_grad():
        for idx, row in df.iterrows():
            img_rel = str(row[image_col])
            img_path = Path(img_rel)
            if not img_path.is_file():
                # try relative to dataset root
                candidates = [
                    ROOT / img_rel,
                    ROOT / "UFPR-ALPR dataset" / img_rel,
                    ROOT / "data" / img_rel,
                ]
                for c in candidates:
                    if c.is_file():
                        img_path = c
                        break
            if not img_path.is_file():
                print(f"[WARN] Missing image for row {idx}: {img_rel}")
                continue

            img = Image.open(img_path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(DEVICE)

            out = model(inp)
            prob = torch.softmax(out, dim=1)
            conf, pred_idx = torch.max(prob, dim=1)
            pred_label = class_map.get(str(int(pred_idx.item())), "unknown")

            # ground truth label string in the same format as training
            true_label = ufpr_label_from_row(row, make_col, model_col, body_col, year_col)

            true_make, true_model, true_body = parse_car_label(true_label)
            pred_make, pred_model, pred_body = parse_car_label(pred_label)

            total += 1
            if pred_label == true_label:
                full_correct += 1
            if pred_make.lower() == true_make.lower():
                make_correct += 1
            if true_body != "unknown" and pred_body.lower() == true_body.lower():
                body_correct += 1

    print(f"[RESULT] Samples evaluated: {total}")
    if total > 0:
        print(f"[RESULT] Full label accuracy: {full_correct/total:.4f}")
        print(f"[RESULT] Make-only accuracy: {make_correct/total:.4f}")
        print(f"[RESULT] Body-type accuracy (where known): {body_correct/total:.4f}")


if __name__ == "__main__":
    main()
