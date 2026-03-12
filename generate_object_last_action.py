"""
For every video_id and each object in active_objects, find the last action (narration)
done by the user on that object from NARRATION_LOW_LEVEL_FILES. Output a list of
final narrations with timestamps and the corresponding active segment.
Objects in objects_to_exclude_vlm.OBJECTS_TO_EXCLUDE_FROM_VLM are skipped.
"""
import argparse
import json
import os
from collections import Counter

import pandas as pd
from utils import hhmmss_to_seconds
from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from objects_to_include import CATEGORIES_TO_INCLUDE

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
NARRATION_LOW_LEVEL_FILES = [
    "EPIC_100_train.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_test_timestamps.csv",
]
ACTIVE_OBJECTS_DIR = "active_objects"
OUTPUT_DIR = "object_last_action"


def load_narration_df():
    """Load and concatenate all narration low-level CSVs. Drops rows without noun_class."""
    dfs = []
    for path in NARRATION_LOW_LEVEL_FILES:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        # test_timestamps may not have narration/noun_class
        if "noun_class" not in df.columns:
            continue
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"None of {NARRATION_LOW_LEVEL_FILES} found with noun_class")
    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["noun_class"])
    out["noun_class"] = out["noun_class"].astype(int)
    return out


def get_unique_objects_from_active(segments, exclude_keys=None, include_categories=None):
    """Return unique objects (class_id, name, subclass_name, category) from all segments.
    Objects whose subclass_name is in exclude_keys are skipped.
    If include_categories is non-empty, only objects whose category is in that set are kept."""
    exclude_keys = exclude_keys or set()
    include_categories = include_categories or set()
    seen = set()
    unique = []
    for seg in segments:
        for obj in seg.get("objects_in_sequence", []):
            subclass = obj.get("subclass_name", "")
            if subclass in exclude_keys:
                continue
            category = obj.get("category", "")
            if include_categories and category not in include_categories:
                continue
            key = (obj["class_id"], obj["name"])
            if key in seen:
                continue
            seen.add(key)
            unique.append({
                "class_id": obj["class_id"],
                "name": obj["name"],
                "subclass_name": subclass,
                "category": category,
            })
    return unique


def find_segment_for_narration(segments, start_ts, stop_ts, narration_str):
    """Find the active segment that contains this narration (by narration list or time overlap)."""
    # First pass: match by narration string and timestamps (format "narration;start_sec;stop_sec")
    for seg in segments:
        for n in seg.get("narrations", []):
            parts = n.split(";")
            if len(parts) >= 3 and parts[0].strip() == narration_str:
                try:
                    n_start = float(parts[1])
                    n_stop = float(parts[2])
                    if abs(n_start - start_ts) < 0.01 and abs(n_stop - stop_ts) < 0.01:
                        return seg
                except ValueError:
                    pass
    # Second pass: segment whose time range contains this narration
    for seg in segments:
        seg_start = seg["start_time"]
        seg_end = seg["end_time"]
        if start_ts >= seg_start and stop_ts <= seg_end:
            return seg
        if start_ts <= seg_end and stop_ts >= seg_start:
            return seg
    return None


def process_video(video_id, narration_df, active_objects_path, output_dir):
    """For one video: compute last action per object and corresponding segment; save JSON."""
    if not os.path.isfile(active_objects_path):
        return False
    with open(active_objects_path) as f:
        segments = json.load(f)

    video_narrations = narration_df[narration_df["video_id"] == video_id]
    if video_narrations.empty:
        return False

    # Convert timestamps to seconds for comparison
    video_narrations = video_narrations.copy()
    video_narrations["start_sec"] = video_narrations["start_timestamp"].apply(hhmmss_to_seconds)
    video_narrations["stop_sec"] = video_narrations["stop_timestamp"].apply(hhmmss_to_seconds)

    unique_objects = get_unique_objects_from_active(
        segments,
        exclude_keys=OBJECTS_TO_EXCLUDE_FROM_VLM,
        include_categories=CATEGORIES_TO_INCLUDE,
    )
    results = []

    for obj in unique_objects:
        class_id = obj["class_id"]
        obj_narrations = video_narrations[video_narrations["noun_class"] == class_id]
        if obj_narrations.empty:
            continue
        # Last action = row with maximum stop_timestamp
        last_row = obj_narrations.loc[obj_narrations["stop_sec"].idxmax()]
        narration_text = last_row.get("narration", "")
        if pd.isna(narration_text):
            narration_text = ""
        start_ts = float(last_row["start_sec"])
        stop_ts = float(last_row["stop_sec"])

        segment = find_segment_for_narration(
            segments, start_ts, stop_ts, narration_text
        )
        if segment is None:
            segment = None
        else:
            segment = {
                "segment_id": segment["segment_id"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "start_frame": segment.get("start_frame"),
                "end_frame": segment.get("end_frame"),
            }

        def _safe(val):
            if pd.isna(val):
                return None
            if hasattr(val, "item"):  # numpy scalar
                return val.item()
            return val

        results.append({
            "object": obj,
            "last_narration": {
                "narration": narration_text,
                "verb": _safe(last_row.get("verb")),
                "verb_class": int(last_row["verb_class"]) if not pd.isna(last_row.get("verb_class")) else None,
                "noun": _safe(last_row.get("noun")),
                "noun_class": int(last_row["noun_class"]) if not pd.isna(last_row.get("noun_class")) else None,
                "start_timestamp": start_ts,
                "stop_timestamp": stop_ts,
                "start_timestamp_str": last_row["start_timestamp"],
                "stop_timestamp_str": last_row["stop_timestamp"],
            },
            "active_segment": segment,
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"object_last_action_{video_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Find last action per object from narrations and link to active segments."
    )
    parser.add_argument(
        "--active-objects-dir",
        default=ACTIVE_OBJECTS_DIR,
        help="Directory containing active_objects_<video_id>.json files",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for object_last_action JSON files",
    )
    args = parser.parse_args()

    print("Loading narration low-level files...")
    narration_df = load_narration_df()
    print(f"Loaded {len(narration_df)} narrations with noun_class")

    count = 0
    for fname in sorted(os.listdir(args.active_objects_dir)):
        if not fname.endswith(".json") or not fname.startswith("active_objects_"):
            continue
        video_id = fname.replace("active_objects_", "").replace(".json", "")
        active_path = os.path.join(args.active_objects_dir, fname)
        if process_video(video_id, narration_df, active_path, args.output_dir):
            count += 1
            print(f"Wrote object_last_action for {video_id}")

    print(f"Done. Wrote {count} files to {args.output_dir}/")

    # Print object counts: total and by category
    by_category = Counter()
    total_objects = 0
    for fname in sorted(os.listdir(args.output_dir)):
        if not fname.endswith(".json") or not fname.startswith("object_last_action_"):
            continue
        with open(os.path.join(args.output_dir, fname)) as f:
            entries = json.load(f)
        for entry in entries:
            cat = entry.get("object", {}).get("category", "unknown")
            by_category[cat] += 1
            total_objects += 1
    print("\nObjects (included categories only):")
    print("-" * 50)
    for cat, n in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")
    print("-" * 50)
    print(f"  Total: {total_objects}")


if __name__ == "__main__":
    main()
