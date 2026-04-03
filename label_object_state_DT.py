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

from generate_object_last_action import load_narration_df, narration_noun_matches_active_name
from utils import hhmmss_to_seconds

DEFAULT_INPUT_DIR = "object_last_action_combined"
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


# def _match_noun_names(name1, name2):
#     ## Split both names by : and space
#     parts1 = re.split(r"[:| ]", name1)
#     parts2 = re.split(r"[:| ]", name2)
#     ## Remove empty parts
#     parts1 = set([p for p in parts1 if p])
#     parts2 = set([p for p in parts2 if p])
#     ## return true if parts1 is subset of parts2 or vice versa
#     iou = len(parts1 & parts2) / len(parts1 | parts2)
#     return iou >= 0.25


# def _find_last_narration_verb(obj: dict, vn: pd.DataFrame) -> str:
#     class_id = obj.get("class_id")
#     name = (obj.get("name") or "").strip()
#     matching_arr = [
#         (row["start_sec"], row["stop_sec"], row["verb"], row["narration"])
#         for _, row in vn.iterrows()
#         if class_id in literal_eval(row["all_noun_classes"])
#         and any([_match_noun_names(name, noun_item) for noun_item in literal_eval(row["all_nouns"])])
#     ]
#     if not matching_arr:
#         print(f"No matching narrations found for object={obj}")
#         import pdb; pdb.set_trace()
#         return None, None, None, None
#     matching_arr.sort(key=lambda x: x[0], reverse=True)
#     return matching_arr[0]


def categorize_verb(verb: str) -> str:
    if verb in VERBS_TIDY_IDLE:
        return True
    if verb in VERBS_IN_USE:
        return False
    # raise ValueError(f"Unknown verb: {verb}")
    return True ## TODO: temporary fix


def label_verb(
    entry: dict,
    vn: pd.DataFrame,
    tol: float,
) -> str:
    global rejected_objects
    """Last matching narration's verb, else 'unknown'."""
    obj = entry.get("object") or {}
    name = (obj.get("name") or "").strip()
    seg = entry.get("last_active_segment") or {}

    seg_start, seg_end = segment_time_bounds(seg)
    narrations = vn[(vn["start_sec"] >= seg_start) & (vn["start_sec"] <= seg_end)]

    chosen_verb: str | None = None
    for line in narrations:
        text, t0, t1 = parse_narration_line(line)
        rows = csv_rows_for_narration_line(vn, text, t0, t1, tol, seg_start, seg_end)
        hits = [
            r
            for r in rows
            if narration_noun_matches_active_name(r.get("noun"), name)
        ]
        if not hits:
            continue
        hits.sort(key=lambda r: (r.get("start_sec") is None, r.get("start_sec") or 0.0))
        row = hits[-1]
        v = normalize_verb(row.get("verb"))
        chosen_verb = v if v is not None else "(no verb)"

    if chosen_verb: 
        return categorize_verb(chosen_verb), chosen_verb
    
    else:
        start_sec_last, stop_sec_last, verb, narration_last = _find_last_narration_verb(obj, vn)
        if verb:
            if not start_sec_last < seg_end:
                print(f"Warning: last narration occuring after end of active segment: {obj}")
                import pdb; pdb.set_trace()
            return False, verb
        else:
            rejected_objects.append({
                "video_id": vn["video_id"].iloc[0],
                "segment_id": seg["segment_id"],
                "class_id": obj["class_id"],
                "name": obj["name"],
                "reason": "No narrations found for this object"
            })
            ## TODO: this can potentially be true.
            return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory with object_last_action_<video_id>.json",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=DEFAULT_TIME_TOL,
        help="Seconds tolerance when matching narration timestamps to CSV rows",
    )
    parser.add_argument(
        "--json-summary",
        default="",
        help="If set, write verb counts as JSON to this path",
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
            is_tidy_at_end, verb = label_verb(entry, vn, args.time_tol)
            labels.append({
                "video_id": video_id,
                "segment_id": entry["last_active_segment"]["segment_id"],
                "class_id": entry["object"]["class_id"],
                "name": entry["object"]["name"],
                "is_tidy_at_end": is_tidy_at_end,
                "last_action_verb": verb,
            })

    import matplotlib.pyplot as plt

    # Bubble plot: x-axis = verbs, y-axis = is_tidy_at_end (True/False), bubble size = frequency, frequency inside bubble
    import numpy as np
    # Count frequencies for each (verb, is_tidy_at_end) combination
    from collections import defaultdict

    bubble_counts = defaultdict(int)
    all_verbs = set()
    for label in labels:
        verb = label["last_action_verb"]
        is_tidy = label["is_tidy_at_end"]
        bubble_counts[(verb, is_tidy)] += 1
        all_verbs.add(verb)

    # Sort verbs for x axis
    verbs_sorted = sorted(all_verbs, key=lambda v: (v is None, str(v)))
    y_states = [True, False]

    # Positions for plotting
    x_positions = np.arange(len(verbs_sorted))
    y_position_map = {True: 1, False: 0}

    fig, ax = plt.subplots(figsize=(max(7, len(verbs_sorted) * 0.7), 5))

    # Scatter bubbles
    for xi, verb in enumerate(verbs_sorted):
        for yi, is_tidy in enumerate(y_states):
            freq = bubble_counts.get((verb, is_tidy), 0)
            if freq > 0:
                # Bubble area proportional to freq
                ax.scatter(
                    xi, y_position_map[is_tidy],
                    s=400 + freq * 200,  # base size + scaled by freq
                    alpha=0.6,
                    color='skyblue',
                    edgecolor='k'
                )
                ax.text(
                    xi, y_position_map[is_tidy],
                    str(freq),
                    va="center", ha="center", fontsize=10, fontweight="bold"
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(verbs_sorted, rotation=45, ha='right')
    ax.set_yticks([y_position_map[True], y_position_map[False]])
    ax.set_yticklabels(['True', 'False'])
    ax.set_xlabel('Verb')
    ax.set_ylabel('is_tidy_at_end')
    ax.set_title("Verb vs is_tidy_at_end Bubble Plot (Bubble Size = Frequency)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
