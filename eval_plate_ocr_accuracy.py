import argparse
import math
import re
from pathlib import Path

import pandas as pd


def clean_plate(s: str) -> str:
    """Uppercase and strip non-alphanumerics."""
    if not isinstance(s, str):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            s = ""
        else:
            s = str(s)
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def normalize_plate(s: str) -> str:
    """
    Normalize plate for 'lenient' matching:
    - I, 1, L -> L
    - O, Q, 0 -> 0
    """
    s = clean_plate(s)
    trans = str.maketrans({"I": "L", "1": "L", "O": "0", "Q": "0"})
    return s.translate(trans)


def get_track_from_source(source: str) -> str:
    """
    Extract track id from source path, e.g.
    'videos/ufpr_tracks/track0001.mp4' -> 'track0001'
    """
    m = re.search(r"(track\d{4})", str(source))
    if m:
        return m.group(1)
    return Path(str(source)).stem


def build_gt_maps(gt_df: pd.DataFrame):
    """
    Build dicts mapping track_dir -> set of plates (strict + normalized).
    """
    strict = {}
    norm = {}

    for track, sub in gt_df.groupby("track_dir"):
        strict_set = set()
        norm_set = set()
        for p in sub["plate"].tolist():
            cp = clean_plate(p)
            if not cp:
                continue
            strict_set.add(cp)
            norm_set.add(normalize_plate(cp))
        if strict_set:
            strict[track] = strict_set
            norm[track] = norm_set

    return strict, norm


def compute_stats(pred_df: pd.DataFrame, gt_strict, gt_norm, threshold: float):
    total = 0
    strict_hits = 0
    norm_hits = 0

    for _, row in pred_df.iterrows():
        try:
            conf = float(row["plate_conf"])
        except Exception:
            continue
        if conf < threshold:
            continue

        track = get_track_from_source(row["source"])
        if track not in gt_strict:
            continue

        p_strict = clean_plate(row["plate_text"])
        if not p_strict:
            continue
        p_norm = normalize_plate(row["plate_text"])

        total += 1
        if p_strict in gt_strict[track]:
            strict_hits += 1
        if p_norm in gt_norm[track]:
            norm_hits += 1

    strict_acc = strict_hits / total if total > 0 else 0.0
    norm_acc = norm_hits / total if total > 0 else 0.0

    return total, strict_hits, norm_hits, strict_acc, norm_acc


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate plate OCR accuracy against UFPR ground truth."
    )
    ap.add_argument(
        "--groundtruth",
        type=str,
        default="data/ufpr_alpr/ufpr_groundtruth_plates.csv",
        help="Path to ufpr_groundtruth_plates.csv",
    )
    ap.add_argument(
        "--predictions",
        type=str,
        default="logs/vehicles_refined.csv",
        help="Path to vehicles_refined.csv",
    )
    args = ap.parse_args()

    gt_path = Path(args.groundtruth)
    pred_path = Path(args.predictions)

    if not gt_path.is_file():
        raise SystemExit(f"Groundtruth CSV not found: {gt_path}")
    if not pred_path.is_file():
        raise SystemExit(f"Prediction CSV not found: {pred_path}")

    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    gt_strict, gt_norm = build_gt_maps(gt_df)

    print(f"[INFO] Groundtruth rows: {len(gt_df)}")
    print(f"[INFO] Prediction rows: {len(pred_df)}")
    print(f"[INFO] Tracks in GT: {len(gt_strict)}")
    print()

    thresholds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print("thr\tN\tstrict_hits\tlenient_hits\tstrict_acc\tlenient_acc")
    for thr in thresholds:
        total, sh, nh, sa, na = compute_stats(pred_df, gt_strict, gt_norm, thr)
        print(
            f"{thr:.1f}\t{total}\t{sh}\t\t{nh}\t\t{sa:.4f}\t{na:.4f}"
        )


if __name__ == "__main__":
    main()
