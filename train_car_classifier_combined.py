import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image

import pandas as pd
from datasets import load_dataset  # HuggingFace "tanganke/stanford_cars"

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
UFPR_META_DIR = ROOT / "data" / "ufpr_alpr"


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def infer_ufpr_columns(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    """
    Heuristically infer UFPR meta CSV column names.

    Returns (image_col, make_col, model_col, body_col, year_col)
    body_col and year_col may be None.
    """
    cols = list(df.columns)

    def find(keywords) -> Optional[str]:
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

    missing = []
    if image_col is None:
        missing.append("image/path/file column")
    if make_col is None:
        missing.append("manufacturer/make/brand column")
    if model_col is None:
        missing.append("model column")
    if missing:
        raise RuntimeError(
            "Could not infer UFPR meta CSV columns: "
            + ", ".join(missing)
            + f". Columns present: {cols}"
        )

    return image_col, make_col, model_col, body_col, year_col


def ufpr_label_from_row(row: pd.Series,
                        make_col: str,
                        model_col: str,
                        body_col: Optional[str],
                        year_col: Optional[str]) -> str:
    """
    Build a label string like:
      'Volkswagen Fox Car 2015'
    This matches what parse_car_label() expects: Make Model Body Year.
    """
    make = str(row[make_col]).strip()
    model = str(row[model_col]).strip()

    parts: List[str] = []
    if make and make.lower() != "nan":
        parts.append(make)
    if model and model.lower() != "nan":
        parts.append(model)

    if body_col:
        body = str(row[body_col]).strip()
        if body and body.lower() != "nan":
            # Normalize to capitalized, e.g. Car, Motorcycle
            parts.append(body.capitalize())

    if year_col:
        year = str(row[year_col]).strip()
        if year.isdigit() and len(year) == 4:
            parts.append(year)

    return " ".join(parts)


def resolve_image_path(path_str: str) -> Path:
    """
    Try to resolve UFPR image path robustly.

    Supports:
      - absolute paths
      - relative to project ROOT
      - relative to UFPR-ALPR dataset folder
    """
    p = Path(path_str)
    if p.is_file():
        return p

    candidates_root = [
        ROOT,
        ROOT / "UFPR-ALPR dataset",
        ROOT / "UFPR-ALPR_dataset",
        ROOT / "UFPR-ALPR",
        ROOT / "data",
    ]
    for base in candidates_root:
        candidate = base / p
        if candidate.is_file():
            return candidate

    # Fallback: return unresolved path (will raise when opened)
    return p


# -----------------------------
# Datasets
# -----------------------------
class StanfordCarsDataset(Dataset):
    def __init__(self,
                 hf_split,
                 label2id: Dict[str, int],
                 label_names: List[str],
                 transform):
        self.ds = hf_split
        self.label2id = label2id
        self.label_names = label_names
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]

        # *** IMPORTANT FIX: always convert to RGB so we have 3 channels ***
        img = img.convert("RGB")

        label_idx = int(row["label"])
        label_str = self.label_names[label_idx]
        y = self.label2id[label_str]
        if self.transform:
            img = self.transform(img)
        return img, y

class UFPRCarsDataset(Dataset):
    def __init__(self,
                 csv_path: Path,
                 label2id: Dict[str, int],
                 transform):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            print(f"[UFPR-DS] WARNING: {csv_path} is empty.")
        self.label2id = label2id
        self.transform = transform

        (self.image_col,
         self.make_col,
         self.model_col,
         self.body_col,
         self.year_col) = infer_ufpr_columns(self.df)

        print(f"[UFPR-DS] {csv_path.name}: using columns - "
              f"image='{self.image_col}', make='{self.make_col}', "
              f"model='{self.model_col}', body='{self.body_col}', year='{self.year_col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path_str = str(row[self.image_col])
        img_path = resolve_image_path(img_path_str)
        img = Image.open(img_path).convert("RGB")

        label_str = ufpr_label_from_row(
            row,
            self.make_col,
            self.model_col,
            self.body_col,
            self.year_col,
        )
        if label_str not in self.label2id:
            raise KeyError(
                f"Label '{label_str}' from {self.csv_path} not found in label2id. "
                f"Ensure all UFPR meta files were included when building the mapping."
            )
        y = self.label2id[label_str]
        if self.transform:
            img = self.transform(img)
        return img, y


# -----------------------------
# Label mapping (Stanford + UFPR)
# -----------------------------
def build_label_mapping(cache_dir: str,
                        use_stanford: bool,
                        use_ufpr: bool):
    """
    Build global label2id mapping from:
      - Stanford Cars HF labels
      - UFPR meta CSVs (make/model/type/year)
    Returns (label2id, id2label, aux_info)
    """
    label_strings: set = set()
    aux = {
        "hf": None,
        "hf_label_names": None,
    }

    # Stanford Cars
    if use_stanford:
        print("[LABELS] Loading Stanford Cars HF dataset labels...")
        hf = load_dataset("tanganke/stanford_cars",
                          cache_dir=cache_dir)
        aux["hf"] = hf
        label_names = list(hf["train"].features["label"].names)
        aux["hf_label_names"] = label_names
        print(f"[LABELS] Stanford Cars: {len(label_names)} classes.")
        label_strings.update(label_names)

    # UFPR meta
    if use_ufpr:
        print("[LABELS] Scanning UFPR meta CSVs for labels...")
        meta_names = ["meta_training.csv", "meta_validation.csv", "meta_testing.csv"]
        for name in meta_names:
            p = UFPR_META_DIR / name
            if not p.exists():
                print(f"[UFPR] {p} not found, skipping.")
                continue
            df = pd.read_csv(p)
            if df.empty:
                print(f"[UFPR] {p} is empty, skipping.")
                continue

            image_col, make_col, model_col, body_col, year_col = infer_ufpr_columns(df)
            for _, row in df.iterrows():
                label_str = ufpr_label_from_row(row, make_col, model_col, body_col, year_col)
                if label_str:
                    label_strings.add(label_str)

        print(f"[LABELS] After UFPR: total unique class strings = {len(label_strings)}")

    if not label_strings:
        raise RuntimeError(
            "No labels collected from any dataset. "
            "Enable at least one of --no-stanford / --no-ufpr."
        )

    all_labels = sorted(label_strings)
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {str(idx): label for label, idx in label2id.items()}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = MODELS_DIR / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)
    print(f"[LABELS] Wrote mapping for {len(all_labels)} classes to {mapping_path}")

    return label2id, id2label, aux


def build_datasets(label2id: Dict[str, int],
                   aux_info,
                   use_stanford: bool,
                   use_ufpr: bool,
                   img_size: int = 224):
    """
    Build train and val datasets as ConcatDataset of:
      - Stanford train / test
      - UFPR meta_training / meta_validation
    """
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_parts = []
    val_parts = []

    # Stanford Cars
    if use_stanford:
        hf = aux_info["hf"]
        label_names = aux_info["hf_label_names"]

        stanford_train = StanfordCarsDataset(
            hf_split=hf["train"],
            label2id=label2id,
            label_names=label_names,
            transform=transform_train,
        )
        stanford_val = StanfordCarsDataset(
            hf_split=hf["test"],   # use HF test as val
            label2id=label2id,
            label_names=label_names,
            transform=transform_val,
        )
        print(f"[DATA] Stanford Cars: train={len(stanford_train)}, val={len(stanford_val)}")
        train_parts.append(stanford_train)
        val_parts.append(stanford_val)

    # UFPR-ALPR
    if use_ufpr:
        meta_train = UFPR_META_DIR / "meta_training.csv"
        meta_val = UFPR_META_DIR / "meta_validation.csv"

        if meta_train.exists():
            ufpr_train = UFPRCarsDataset(
                csv_path=meta_train,
                label2id=label2id,
                transform=transform_train,
            )
            print(f"[DATA] UFPR-ALPR train: {len(ufpr_train)}")
            train_parts.append(ufpr_train)
        else:
            print(f"[UFPR] {meta_train} does not exist; UFPR train skipped.")

        if meta_val.exists():
            ufpr_val = UFPRCarsDataset(
                csv_path=meta_val,
                label2id=label2id,
                transform=transform_val,
            )
            print(f"[DATA] UFPR-ALPR val: {len(ufpr_val)}")
            val_parts.append(ufpr_val)
        else:
            print(f"[UFPR] {meta_val} does not exist; UFPR val skipped.")

    if not train_parts or not val_parts:
        raise RuntimeError(
            "Train or val dataset is empty. "
            "Check that at least one source (Stanford/UFPR) is enabled and has data."
        )

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)
    print(f"[DATA] Combined train size: {len(train_ds)}, val size: {len(val_ds)}")
    return train_ds, val_ds


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        base = models.resnet50(weights=None)
    in_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return base


# -----------------------------
# Training loop
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")

    set_seed(args.seed)

    # 1) Global label mapping (Stanford + UFPR)
    label2id, id2label, aux_info = build_label_mapping(
        cache_dir=args.cache_dir,
        use_stanford=not args.no_stanford,
        use_ufpr=not args.no_ufpr,
    )

    num_classes = len(label2id)
    print(f"[INFO] Total classes: {num_classes}")

    # 2) Datasets
    train_ds, val_ds = build_datasets(
        label2id=label2id,
        aux_info=aux_info,
        use_stanford=not args.no_stanford,
        use_ufpr=not args.no_ufpr,
        img_size=args.img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 3) Model
    model = create_model(num_classes=num_classes,
                         pretrained=not args.no_imagenet_pretrain)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_val_acc = 0.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total if total > 0 else 0.0

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            weights_path = MODELS_DIR / "car_classifier_resnet50_hf.pt"
            torch.save(model.state_dict(), weights_path)
            print(f"--> Saved new best model to {weights_path} (val_acc={best_val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"Training complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Total training time: {elapsed/60:.1f} min")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train car make/model classifier on Stanford Cars + UFPR-ALPR."
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT / "data" / "stanford_cars_hf"),
        help="Cache dir for HuggingFace Stanford Cars dataset.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size for the classifier.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--no-imagenet-pretrain",
        action="store_true",
        help="If set, do NOT initialize from ImageNet weights.",
    )
    p.add_argument(
        "--no-stanford",
        action="store_true",
        help="If set, do NOT use Stanford Cars dataset (only UFPR).",
    )
    p.add_argument(
        "--no-ufpr",
        action="store_true",
        help="If set, do NOT use UFPR dataset (only Stanford).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
