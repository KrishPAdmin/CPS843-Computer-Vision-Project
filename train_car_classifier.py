import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets import load_dataset, concatenate_datasets


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        image = item["image"]
        # HF stores images as PIL via datasets.Image
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        label = int(item["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    image_size: int,
    base_splits_only: bool = False,
):
    """
    Uses Hugging Face dataset:
    - By default: **all splits** (train + test + corruption splits)
      so you’re really using 100% of the provided images.
    - If base_splits_only=True: just train+test.
    """
    ds = load_dataset("tanganke/stanford_cars", cache_dir=str(cache_dir))

    base_splits: List[str] = ["train", "test"]
    corruption_splits: List[str] = [
        "contrast",
        "gaussian_noise",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "spatter",
    ]

    splits_to_use = base_splits if base_splits_only else base_splits + corruption_splits

    datasets_to_concat = [ds[split] for split in splits_to_use]
    full_ds = concatenate_datasets(datasets_to_concat)

    # Shuffle & internal train/val split (no extra copies on disk)
    full_ds = full_ds.shuffle(seed=42)
    split_ds = full_ds.train_test_split(
        test_size=val_ratio,
        stratify_by_column="label",
        seed=42,
    )

    train_hf = split_ds["train"]
    val_hf = split_ds["test"]

    num_classes = len(set(full_ds["label"]))

    train_tf, val_tf = build_transforms(image_size=image_size)

    train_ds = HFDataset(train_hf, transform=train_tf)
    val_ds = HFDataset(val_hf, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes, len(full_ds), splits_to_use


def build_model(num_classes: int, pretrained: bool = True):
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train car make/model classifier on Stanford Cars (HF)."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/stanford_cars_hf",
        help="Cache directory for Hugging Face stanford_cars dataset.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--base-splits-only",
        action="store_true",
        help="Use only the original train+test splits (no corruption splits).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/car_classifier_resnet50_hf.pt",
        help="Path to save best model checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, num_classes, total_images, splits_used = create_dataloaders(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        base_splits_only=args.base_splits_only,
    )

    print(f"HF splits used: {', '.join(splits_used)}")
    print(f"Total images used (train + val): {total_images}")
    print(f"Number of classes: {num_classes}")

    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_epoch = -1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "splits_used": splits_used,
                },
                output_path,
            )
            print(f"--> Saved new best model to {output_path} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
