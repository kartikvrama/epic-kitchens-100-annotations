"""
For every video_id and each object in active_objects, find the last action (narration)
done by the user on that object from NARRATION_LOW_LEVEL_FILES. Output a list of
final narrations with timestamps and the corresponding active segment.
Objects in objects_to_exclude_vlm.SUBCLASSES_EXCLUDED are skipped.
"""
import numpy as np
import argparse
from ast import literal_eval
import json
import os
from collections import Counter

import pandas as pd
from utils import hhmmss_to_seconds, include_object, load_noun_class_names

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
NARRATION_LOW_LEVEL_FILES = [
    "EPIC_100_train.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_test_timestamps.csv",
]
ACTIVE_OBJECTS_DIR = "active_objects"
OUTPUT_DIR = "object_last_action"

NOUN_CLASS_NAMES = load_noun_class_names()


def _noun_token_bag(s):
    """Bag of lowercase tokens from EPIC/Visor noun strings (handles `base:qualifier`, `/`, spaces)."""
    if not s:
        return frozenset()
    tokens = []
    for part in str(s).lower().replace(",", " ").split("/"):
        part = part.strip()
        for w in part.split():
            for t in w.split(":"):
                t = t.strip()
                if t:
                    tokens.append(t)
    return frozenset(tokens)


def narration_noun_matches_active_name(noun_csv, active_name):
    """True if CSV narration noun refers to the same instance label as Visor active object name.

    EPIC nouns often use ``head:modifier`` (e.g. ``container:tofu``) while Visor uses English NP order
    (``tofu container``). Comparing token bags makes these align without requiring 1:1 string equality.
    """
    if not noun_csv or not active_name:
        return False
    if noun_csv == active_name:
        return True
    bag_n = _noun_token_bag(noun_csv)
    if not bag_n:
        return False
    for variant in str(active_name).split("/"):
        bag_a = _noun_token_bag(variant.strip())
        if bag_n == bag_a:
            return True
    return False


def disambiguate_narration_keys(matching_keys, active_name):
    """Narrow ``matching_keys`` (full 4-tuples) using exact then token-bag match to ``active_name``."""
    if len(matching_keys) <= 1:
        return matching_keys
    exact = [k for k in matching_keys if k[2] == active_name]
    if len(exact) == 1:
        return exact
    if len(exact) > 1:
        return exact
    token_hits = [k for k in matching_keys if narration_noun_matches_active_name(k[2], active_name)]
    if len(token_hits) == 1:
        return token_hits
    if len(token_hits) > 1:
        token_hits.sort(key=lambda k: (-len(k[2]), k[2]))
        return [token_hits[0]]
    return []


def get_tidy_label(verb, all_noun_classes):
    """Get the tidy label for a verb and a list of noun classes."""
    raise NotImplementedError()


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


def get_object_last_usage(segments):
    """Return unique objects (class_id, name, subclass_name, category) from all segments."""
    object_last_usage = {}
    for seg in segments:
        for obj in seg.get("objects_in_sequence", []):
            category = obj.get("category")
            subclass = obj.get("subclass_name")
            name = obj.get("name")
            if not include_object(category, subclass, name):
                continue
            key = (int(obj["class_id"]), obj["subclass_name"], obj["name"], category)
            object_last_usage[key] = {
                "segment_id": seg.get("segment_id"),
                "start_time": seg.get("start_time"),
                "end_time": seg.get("end_time"),
                "start_frame": seg.get("start_frame"),
                "stop_frame": seg.get("end_frame"),
                "narrations": seg.get("narrations", []),
            }
    return object_last_usage


def get_object_last_narrations(video_narrations, object_list):
    """Return the last narration for each object in object_last_usage."""
    object_last_narrations = {}
    ## Read narrations in descending order of start_frame to get the last narration for each object
    for _, video_narration in video_narrations.sort_values("start_frame", ascending=False).iterrows():
        narration_str = video_narration["narration"]
        start_ts = video_narration["start_sec"]
        stop_ts = video_narration["stop_sec"]
        noun_name = video_narration["noun"]
        class_id = int(video_narration["noun_class"])
        noun_dict = NOUN_CLASS_NAMES[class_id]
        category = noun_dict["category"]
        subclass_name = noun_dict["key"]
        key = (class_id, subclass_name, noun_name, category)
        if key not in object_last_narrations:
            object_last_narrations[key] = {
                "segment_id": None,
                "start_time": start_ts,
                "stop_time": stop_ts,
                "start_frame": video_narration["start_frame"],
                "stop_frame": video_narration["stop_frame"],
                "narrations": [narration_str],
            }
    return object_last_narrations


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

    object_last_active_segments = get_object_last_usage(segments)
    object_last_narrations = get_object_last_narrations(video_narrations, list(object_last_active_segments.keys()))
    results = []
    _delta_narration_active_segment = []
    for obj_key in object_last_active_segments.keys():
        class_id, subclass_name, name, category = obj_key
        ## Find last active segment that uses this object
        last_active_segment = object_last_active_segments[obj_key]
        last_segment_stop_timestamp = last_active_segment["end_time"]
        matching_arr = [k for k in object_last_narrations.keys() if k[0] == class_id and k[1] == subclass_name and k[3] == category]
        # EPIC CSV `noun` uses colon-separated qualifiers (e.g. ``container:tofu``); Visor ``name`` is often
        # plain English order (e.g. ``tofu container``). So many objects share the same
        # (class_id, subclass_name, category) with several last-narration keys — not 1:1 until disambiguated.
        matching_arr = disambiguate_narration_keys(matching_arr, name)
        if not matching_arr:
            ## Use last active segment as last action
            # print(f"Warning: no last narration found for object {obj_key}")
            results.append({
                "object": {
                    "class_id": class_id,
                    "subclass_name": subclass_name,
                    "name": name,
                    "category": category,
                },
                "last_active_segment": {
                    **last_active_segment,
                    "tidy_label": None
                },
            })
            continue
        if len(matching_arr) > 1:
            # print(f"Warning: {len(matching_arr)} last narrations found for object {obj_key}")
            matching_arr = [matching_arr[0]]
        matching_key_narr = matching_arr[0]
        last_narration = object_last_narrations[matching_key_narr]
        last_narration_stop_time = last_narration["stop_time"]


        _delta_narration_active_segment.append(abs(last_narration_stop_time - last_segment_stop_timestamp))
        if last_narration_stop_time > last_segment_stop_timestamp:
            obj_data = last_narration
        else:
            obj_data = last_active_segment
        
        results.append({
            "object": {
                "class_id": class_id,
                "subclass_name": subclass_name,
                "name": name,
                "category": category,
            },
            "last_active_segment": {
                **obj_data,
                "tidy_label": None
            },
        })
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"object_last_action_{video_id}.json")
    print(f"{video_id}: Delta narration active segment: {np.mean(_delta_narration_active_segment)/60:.2f} +/- {np.std(_delta_narration_active_segment)/60:.2f}")
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
