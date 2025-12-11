from pathlib import Path
from datasets import load_dataset

root = Path("data")
root.mkdir(exist_ok=True)

print("Downloading tanganke/stanford_cars ...")
cars = load_dataset("tanganke/stanford_cars")
cars.save_to_disk(root / "stanford_cars")

print("Downloading keremberke/license-plate-object-detection (full split) ...")
lp_det = load_dataset("keremberke/license-plate-object-detection", name="full")
lp_det.save_to_disk(root / "license_plate_detection")

print("All done.")
