import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Dict, Any

import difflib
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
    Normalize plate for matching:
    - I, 1, L -> L
    - O, Q, 0 -> 0
    """
    s = clean_plate(s)
    trans = str.maketrans({"I": "L", "1": "L", "O": "0", "Q": "0"})
    return s.translate(trans)


def load_hotlist(path: Path) -> List[Dict[str, Any]]:
    """
    Load hotlist CSV and return a list of entries with precomputed
    cleaned / normalized plates.
    """
    df = pd.read_csv(path)

    entries = []
    for _, row in df.iterrows():
        plate_raw = row.get("plate", "")
        clean = clean_plate(plate_raw)
        norm = normalize_plate(plate_raw)
        if not norm:
            continue
        entries.append({
            "plate_raw": plate_raw,
            "plate_clean": clean,
            "plate_norm": norm,
            "make": str(row.get("make", "")),
            "model": str(row.get("model", "")),
            "color": str(row.get("color", "")),
            "reason": str(row.get("reason", "")),
            "priority": str(row.get("priority", "")),
        })

    return entries


def find_matches(norm_pred: str,
                 hotlist: List[Dict[str, Any]],
                 fuzzy_threshold: float) -> List[Dict[str, Any]]:
    """
    Return all hotlist entries that match this predicted plate.

    - First, look for exact normalized matches.
    - If none, fall back to fuzzy matching using SequenceMatcher
      with ratio >= fuzzy_threshold (e.g. 0.8).
    """
    if not norm_pred:
        return []

    # 1) exact matches
    exact = [h for h in hotlist if h["plate_norm"] == norm_pred]
    if exact:
        return exact

    # 2) fuzzy matches
    candidates = []
    for h in hotlist:
        # Only compare similar-length plates to avoid silly matches
        if abs(len(norm_pred) - len(h["plate_norm"])) > 1:
            continue
        ratio = difflib.SequenceMatcher(None, norm_pred, h["plate_norm"]).ratio()
        if ratio >= fuzzy_threshold:
            candidates.append(h)

    return candidates


def main():
    ap = argparse.ArgumentParser(
        description="Cross-reference vehicles_refined.csv with a hotlist of wanted vehicles."
    )
    ap.add_argument(
        "--hotlist",
        type=str,
        default="hotlist_sample.csv",
        help="Path to hotlist CSV (columns: plate,make,model,color,reason,priority).",
    )
    ap.add_argument(
        "--refined",
        type=str,
        default="logs/vehicles_refined.csv",
        help="Path to vehicles_refined.csv from the pipeline.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="logs/alerts.csv",
        help="Where to write alert matches.",
    )
    ap.add_argument(
        "--min-plate-conf",
        type=float,
        default=0.5,
        help="Minimum plate_conf required to consider a detection.",
    )
    ap.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help=(
            "Fuzzy match threshold (0–1). "
            "Lower values give more matches but more false positives."
        ),
    )
    args = ap.parse_args()

    hotlist_path = Path(args.hotlist)
    refined_path = Path(args.refined)
    output_path = Path(args.output)

    if not hotlist_path.is_file():
        raise SystemExit(f"Hotlist CSV not found: {hotlist_path}")
    if not refined_path.is_file():
        raise SystemExit(f"Refined CSV not found: {refined_path}")

    hotlist = load_hotlist(hotlist_path)
    print(f"[INFO] Hotlist entries: {len(hotlist)}")

    df_ref = pd.read_csv(refined_path)

    alerts = []

    for _, row in df_ref.iterrows():
        # 1) confidence filter
        try:
            conf = float(row.get("plate_conf", 0.0))
        except Exception:
            conf = 0.0
        if conf < args.min_plate_conf:
            continue

        # 2) normalize predicted plate
        plate_pred = row.get("plate_text", "")
        plate_clean = clean_plate(plate_pred)
        plate_norm = normalize_plate(plate_pred)
        if not plate_norm:
            continue

        # 3) look for matches in hotlist (exact or fuzzy)
        matches = find_matches(plate_norm, hotlist, args.fuzzy_threshold)
        if not matches:
            continue

        for hl in matches:
            alert = {
                "run_id": row.get("run_id", ""),
                "event_time_utc": row.get("event_time_utc", ""),
                "video_time_sec": row.get("video_time_sec", ""),
                "source": row.get("source", ""),
                "track_id": row.get("track_id", ""),
                "plate_text": plate_clean,
                "plate_conf": row.get("plate_conf", ""),
                "detected_class": row.get("detected_class", ""),
                "body_type": row.get("body_type", ""),
                "make_pred": row.get("make", ""),
                "model_pred": row.get("model", ""),
                "color_pred": row.get("color", ""),
                "final_image_path": row.get("final_image_path", ""),
                "make_model_confidence": row.get("make_model_confidence", ""),
                # hotlist info
                "hotlist_plate": hl["plate_clean"],
                "hotlist_make": hl["make"],
                "hotlist_model": hl["model"],
                "hotlist_color": hl["color"],
                "reason": hl["reason"],
                "priority": hl["priority"],
            }
            alerts.append(alert)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Always write a header, even if there are 0 alerts
    fieldnames = [
        "run_id",
        "event_time_utc",
        "video_time_sec",
        "source",
        "track_id",
        "plate_text",
        "plate_conf",
        "detected_class",
        "body_type",
        "make_pred",
        "model_pred",
        "color_pred",
        "final_image_path",
        "make_model_confidence",
        "hotlist_plate",
        "hotlist_make",
        "hotlist_model",
        "hotlist_color",
        "reason",
        "priority",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in alerts:
            writer.writerow(a)

    print(f"[INFO] Alerts found: {len(alerts)}")
    print(f"[INFO] Alerts written to: {output_path}")

    for a in alerts:
        print(
            f"[ALERT] {a['priority'].upper()} | Plate {a['plate_text']} "
            f"({a['make_pred']} {a['model_pred']}, {a['color_pred']}) "
            f"matches hotlist {a['hotlist_plate']} "
            f"({a['reason']}) source={a['source']} t={a['video_time_sec']}s"
        )


if __name__ == "__main__":
    main()
