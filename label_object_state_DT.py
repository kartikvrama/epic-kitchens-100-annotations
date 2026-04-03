"""
For each row in object_last_action_*.json, scan segment narrations in order and find
the last narration whose EPIC CSV *noun* matches the Visor object *name* (same rules
as generate_object_last_action.narration_noun_matches_active_name). Use that row's
*verb*. If no narration qualifies, count the row under verb "unknown".

Requires EPIC low-level CSVs (same as generate_object_last_action.load_narration_df).
"""
from __future__ import annotations

import argparse
from ast import literal_eval
import re
import sys
import json
import math
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from generate_object_last_action import load_narration_df, match_noun_names
from utils import hhmmss_to_seconds

DEFAULT_INPUT_DIR = "object_last_action_deduped"
DEFAULT_TIME_TOL = 1

VERBS_TIDY_IDLE = {}
VERBS_IN_USE = {}

rejected_objects = []


def parse_narration_line(line: str) -> tuple[str, float | None, float | None]:
    """Return (narration_text, start_sec, stop_sec). Times None if missing or invalid."""
    parts = str(line).strip().split(";")
    text = parts[0].strip() if parts else ""
    if len(parts) < 3:
        return text, None, None
    try:
        return text, float(parts[1]), float(parts[2])
    except (TypeError, ValueError):
        return text, None, None


def segment_time_bounds(seg: dict) -> tuple[float | None, float | None]:
    """Segment [start, end] in seconds; supports end_time or legacy stop_time."""
    try:
        st = float(seg["start_time"])
    except (KeyError, TypeError, ValueError):
        st = None
    en = seg.get("end_time")
    if en is None:
        en = seg.get("stop_time")
    try:
        en = float(en)
    except (TypeError, ValueError):
        en = None
    return st, en


def csv_rows_for_narration_line(
    vn: pd.DataFrame,
    text: str,
    start_sec: float | None,
    stop_sec: float | None,
    tol: float,
    seg_start: float | None,
    seg_end: float | None,
) -> list[dict]:
    """Match EPIC rows: strict narration timestamps first, else overlap with active segment."""
    if not text:
        return []
    base = vn[vn["narration"].astype(str).str.strip() == text]
    if base.empty:
        return []
    if start_sec is not None and stop_sec is not None:
        m = base[
            (base["start_sec"] >= start_sec - tol) & (base["stop_sec"] <= stop_sec + tol)
        ]
        if not m.empty:
            return m.to_dict(orient="records")
    if seg_start is not None and seg_end is not None:
        m = base[
            (base["start_sec"] <= seg_end + tol) & (base["stop_sec"] >= seg_start - tol)
        ]
        if not m.empty:
            return m.to_dict(orient="records")
    # No time hints: avoid picking an arbitrary occurrence elsewhere in the video
    if seg_start is None or seg_end is None:
        return base.to_dict(orient="records")
    return []


def normalize_verb(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    s = str(v).strip().lower()
    return s if s else None


def categorize_verb(verb: str) -> str:
    if verb in VERBS_TIDY_IDLE:
        return True
    if verb in VERBS_IN_USE:
        return False
    # raise ValueError(f"Unknown verb: {verb}")
    return "unknown" ## TODO: temporary fix


def _get_last_verb(vn: pd.DataFrame, obj: dict) -> str:
    """Get the last verb for an object from the narration dataframe."""
    hits = [
        r
        for r in vn.to_dict(orient="records")
        if match_noun_names(r.get("noun"), obj["name"]) and obj["class_id"] in literal_eval(r.get("all_noun_classes"))
    ]
    hits.sort(key=lambda r: (r["start_sec"] is None, r["start_sec"] or 0.0))
    if not hits:
        import pdb; pdb.set_trace()
        raise ValueError(f"No hits for object {obj}")
    return "_action_b4_active | " + hits[-1]["verb"], hits[-1]["verb_class"]


def label_verb(
    entry: dict,
    vn: pd.DataFrame,
) -> str:
    global rejected_objects
    """Last matching narration's verb, else 'unknown'."""
    obj = entry.get("object") or {}
    name = (obj.get("name") or "").strip()
    seg = entry.get("last_active_segment") or {}
    _narration_found = entry.get("_narration_found")

    seg_start, seg_end = segment_time_bounds(seg)
    narrations = vn[(vn["start_sec"] >= seg_start) & (vn["start_sec"] <= seg_end)]

    ## Object is tidy if it does not exist in any of the narrations
    ## TODO: This can potentially be more nuanced.
    if not _narration_found:
        return True, "_narration_not_found | None", None

    chosen_verb: str | None = None
    hits = [
        r
        for r in narrations.to_dict(orient="records")
        if match_noun_names(r.get("noun"), name) and obj["class_id"] in literal_eval(r.get("all_noun_classes"))
    ]

    ## Object is not tidy if narration (action) occurs before the active segment
    if not hits:
        return False, *_get_last_verb(vn, obj)

    hits.sort(key=lambda r: (r["start_sec"] is None, r["start_sec"] or 0.0))
    row = hits[-1]
    chosen_verb = row.get("verb")
    chosen_verb_class = row.get("verb_class")
    return categorize_verb(chosen_verb), chosen_verb, chosen_verb_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory with object_last_action_<video_id>.json",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print("Loading narration CSVs...")
    narration_df = load_narration_df()
    narration_df = narration_df.copy()
    narration_df["start_sec"] = narration_df["start_timestamp"].apply(hhmmss_to_seconds)
    narration_df["stop_sec"] = narration_df["stop_timestamp"].apply(hhmmss_to_seconds)

    by_verb = Counter()
    n_rows = 0
    labels = []
    for path in sorted(root.glob("object_last_action_*.json")):
        video_id = path.stem.replace("object_last_action_", "")
        vn = narration_df[narration_df["video_id"] == video_id]
        if vn.empty:
            print(f"warning: no narrations for video_id={video_id}, skipping file {path.name}")
            continue
        with open(path) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            n_rows += 1
            is_tidy_at_end, verb, verb_class = label_verb(entry, vn)
            labels.append({
                "video_id": video_id,
                "segment_id": entry["last_active_segment"]["segment_id"],
                "class_id": entry["object"]["class_id"],
                "name": entry["object"]["name"],
                "is_tidy_at_end": is_tidy_at_end,
                "last_action_verb": verb,
                "last_action_verb_class": verb_class,
            })

    tidy_labels = [x["is_tidy_at_end"] for x in labels]
    verbs = [x["last_action_verb"] for x in labels]

    # Save labels to a JSON file inside the input-dir
    out_json_path = root / "object_last_action_labels.json"
    with open(out_json_path, "w") as fout:
        json.dump(labels, fout, indent=2)
    print(f"Saved {len(labels)} labels to {out_json_path}")


    # Count each unique verb
    verb_counts = Counter(verbs)
    print("\nCounts for each unique verb:")
    for verb, count in sorted(verb_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {repr(verb)}: {count}")
    print("-" * 40)
    print(f"  Total: {sum(verb_counts.values())}")


if __name__ == "__main__":
    main()
