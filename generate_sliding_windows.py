"""
Generate sliding-window segment assignments per video and per object.
Writes data_stats/sliding_window_results.json (labels_per_object with behind/ahead/within).
Consumed by stats_sliding_window_labels.py for per-video per-object label counts.
"""
import os
import json
import csv
from math import ceil, floor
from collections import defaultdict
from tqdm import tqdm

from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from utils import (
    load_noun_class_names,
    load_inactive_segments,
    load_inactive_annotations,
    get_visor_object_appearance_times,
)
from utils import (
    MIN_DURATION_INACTIVE_SEGMENT,
    ACTIVE_OBJECTS_DIR,
    VIDEO_ID_FILE,
    VIDEO_INFO_FILE,
)

WINDOW_SIZE = 12  # seconds
STEP_SIZE = 3  # seconds
HISTORY_SIZE = 3 * WINDOW_SIZE  # seconds
WINDOW_BEHIND = 3 * WINDOW_SIZE  # seconds
WINDOW_AHEAD = 3 * WINDOW_SIZE  # seconds

INACTIVE_ANNOTATIONS_DIR = "vlm_annotations_maxImages1_minSpacing3_20260226"
OUTPUT_DIR = "data_stats"
SLIDING_WINDOW_RESULTS_FILE = "sliding_window_results.json"

NOUN_CLASS_NAMES = load_noun_class_names()


import pdb

def _object_appears_in_range(key, t_low, t_high, object_appearance_times_by_key):
    """Return True if the object has at least one VISOR appearance in [t_low, t_high]."""
    timestamps = object_appearance_times_by_key.get(key, [])
    return any(t_low <= t <= t_high for t in timestamps)


def _object_status_in_interval(
    obj_key,
    t_low,
    t_high,
    annotations_active_per_segment,
    annotations_inactive_per_object,
):
    """
    Return one of "active", "passive", "both", "unused" for obj_key (category/subclass/name)
    over the time interval [t_low, t_high], based on overlapping active and inactive annotations.
    Non-zero overlap is required.
    """
    has_active = False
    has_passive = False
    for seg in annotations_active_per_segment:
        seg_start = seg.get("start_time")
        seg_end = seg.get("end_time")
        if seg_start is None or seg_end is None:
            continue
        if seg_start >= t_high or seg_end <= t_low:
            continue
        overlap = min(seg_end, t_high) - max(seg_start, t_low)
        if overlap <= 0:
            continue
        for obj in seg.get("objects_in_sequence", []):
            if obj.get("subclass_name") in OBJECTS_TO_EXCLUDE_FROM_VLM:
                continue
            key = f"{obj.get('category')}/{obj.get('subclass_name')}/{obj.get('name')}"
            if key == obj_key:
                has_active = True
                break
        if has_active:
            break
    for ann in annotations_inactive_per_object:
        if ann.get("is_passive_usage") is not True:
            continue
        if "query_object" not in ann:
            continue
        ann_start = ann.get("start_time")
        ann_end = ann.get("end_time")
        if ann_start is None or ann_end is None:
            continue
        if ann_start >= t_high or ann_end <= t_low:
            continue
        overlap = min(ann_end, t_high) - max(ann_start, t_low)
        if overlap <= 0:
            continue
        try:
            obj_id, subclass, name = ann.get("query_object").split("/", maxsplit=2)
            category = NOUN_CLASS_NAMES[int(obj_id)]["category"]
            key = f"{category}/{subclass}/{name}"
        except (ValueError, KeyError, TypeError):
            print(f"Error parsing query_object: {ann.get('query_object')}")
            pdb.set_trace()
            continue
        if key == obj_key:
            has_passive = True
            break
    if has_active and has_passive:
        return "both"
    if has_active:
        return "active"
    if has_passive:
        return "passive"
    return "unused"


def main():
    with open(VIDEO_INFO_FILE, "r") as f:
        reader = csv.DictReader(f)
        video_info = list(reader)

    with open(VIDEO_ID_FILE, "r") as f:
        reader = csv.reader(f)
        video_ids = [row[0] for row in reader][1:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    for video_id in tqdm(video_ids):
        video_info_row = next((v for v in video_info if v["video_id"] == video_id), None)
        if not video_info_row:
            raise ValueError(f"Video {video_id} not found in video_info")
        video_length = floor(float(video_info_row["duration"]))

        # Per-video VISOR appearance times: key by category/subclass/name for lookup
        visor_appearance_list = get_visor_object_appearance_times(video_id)
        object_appearance_times_by_key = {}
        if visor_appearance_list:
            for item in visor_appearance_list:
                obj_key = item["object_key"]
                parts = obj_key.split("/", 2)
                if len(parts) >= 3:
                    try:
                        class_id = int(parts[0])
                        subclass, name = parts[1], parts[2]
                        if subclass in OBJECTS_TO_EXCLUDE_FROM_VLM:
                            continue
                        category = NOUN_CLASS_NAMES[class_id]["category"]
                        key = f"{category}/{subclass}/{name}"
                        object_appearance_times_by_key[key] = item["timestamps"]
                    except (ValueError, KeyError):
                        print(f"Error parsing object_key: {obj_key}")
                        pdb.set_trace()
                        pass
        if not object_appearance_times_by_key:
            raise ValueError(
                f"object_appearance_times_by_key is empty for video_id={video_id} "
                "(missing VISOR annotations or visor-frames_to_timestamps.json)"
            )

        annotations_path = os.path.join(
            INACTIVE_ANNOTATIONS_DIR, f"inactive_segments_{video_id}_labels.jsonl"
        )
        annotations_inactive_per_object, keys_missing, keys_null = load_inactive_annotations(
            video_id,
            annotations_filepath=annotations_path,
            object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM,
            min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT,
        )
        if keys_missing:
            print(
                f"Video {video_id}: {len(keys_missing)} missing keys in VLM annotations, skipping video..."
            )
            continue

        with open(
            os.path.join(ACTIVE_OBJECTS_DIR, f"active_objects_{video_id}.json"), "r"
        ) as f:
            annotations_active_per_segment = json.load(f)

        total_segments = ceil(video_length / STEP_SIZE)
        all_object_keys = set(object_appearance_times_by_key.keys())
        labels_per_object = {
            key: {
                "labels_behind_to_seg": {},
                "labels_ahead_of_seg": {},
            }
            for key in all_object_keys
        }
        # Recompute labels for each valid (object, segment_idx)
        for segment_idx in range(total_segments):
            window_start = segment_idx * STEP_SIZE
            window_end = min(window_start + WINDOW_SIZE, video_length)
            t = window_end
            t_history = t - HISTORY_SIZE
            t_behind = t - WINDOW_BEHIND ## t - N
            t_ahead = t + WINDOW_AHEAD ## t + N
            for key in all_object_keys:
                subclass_name = key.split("/")[1]
                if subclass_name in OBJECTS_TO_EXCLUDE_FROM_VLM:
                    continue
                if _object_appears_in_range(
                    key, t_history, t, object_appearance_times_by_key
                ):
                    label_behind = _object_status_in_interval(
                        key, t_behind, t,
                        annotations_active_per_segment, annotations_inactive_per_object,
                    )
                    label_ahead = _object_status_in_interval(
                        key, t, t_ahead,
                        annotations_active_per_segment, annotations_inactive_per_object,
                    )
                    labels_per_object[key]["labels_behind_to_seg"][segment_idx] = label_behind
                    labels_per_object[key]["labels_ahead_of_seg"][segment_idx] = label_ahead

        num_objects = len(labels_per_object)
        results[video_id] = {
            "video_length": video_length,
            "num_objects": num_objects,
            "labels_per_object": labels_per_object,
        }

    out_path = os.path.join(OUTPUT_DIR, SLIDING_WINDOW_RESULTS_FILE)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
