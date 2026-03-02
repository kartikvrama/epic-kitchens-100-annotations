"""
Generate sliding-window segment assignments per video and per object.
Writes data_stats/sliding_window_results.json for consumption by stats_sliding_window.py.
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
OVERLAP_TOL = 0.0  # minimum overlap between segment and window to be considered "used"

WINDOW_BEHIND = 3 * WINDOW_SIZE  # seconds
WINDOW_AHEAD = 3 * WINDOW_SIZE  # seconds

INACTIVE_ANNOTATIONS_DIR = "vlm_annotations_maxImages1_minSpacing3_20260226"
OUTPUT_DIR = "data_stats"
SLIDING_WINDOW_RESULTS_FILE = "sliding_window_results.json"

NOUN_CLASS_NAMES = load_noun_class_names()


def _object_appears_in_range(key, t_low, t_high, object_appearance_times_by_key):
    """Return True if the object has at least one VISOR appearance in [t_low, t_high]."""
    timestamps = object_appearance_times_by_key.get(key, [])
    return any(t_low <= t <= t_high for t in timestamps)


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
                        category = NOUN_CLASS_NAMES[class_id]["category"]
                        key = f"{category}/{subclass}/{name}"
                        object_appearance_times_by_key[key] = item["timestamps"]
                    except (ValueError, KeyError):
                        pass
        if not object_appearance_times_by_key:
            raise ValueError(
                f"object_appearance_times_by_key is empty for video_id={video_id} "
                "(missing VISOR annotations or visor-frames_to_timestamps.json)"
            )

        segments_per_object_active = defaultdict(list)
        segments_per_object_passive = defaultdict(list)
        segments_per_object_unused = defaultdict(list)
        appearance_time_per_object = defaultdict(float)

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

        segment_idx = 0
        for window_start in range(0, video_length, STEP_SIZE):
            window_end = min(window_start + WINDOW_SIZE, video_length)
            if window_end > video_length:
                break

            # Find overlapping active segments between window_start and window_end
            for seg in annotations_active_per_segment:
                seg_start = seg.get("start_time")
                seg_end = seg.get("end_time")
                if seg_start < window_end and seg_end > window_start:
                    overlap_start = max(seg_start, window_start)
                    overlap_end = min(seg_end, window_end)
                    overlap = max(0, overlap_end - overlap_start)
                    seg_duration = seg_end - seg_start
                    if seg_duration == 0:
                        continue
                    overlap_frac = overlap / seg_duration
                    if overlap_frac < OVERLAP_TOL:
                        continue
                    for obj in seg.get("objects_in_sequence", []):
                        if obj["subclass_name"] in OBJECTS_TO_EXCLUDE_FROM_VLM:
                            continue
                        key = f"{obj['category']}/{obj['subclass_name']}/{obj['name']}"
                        t_low = seg_start - WINDOW_BEHIND
                        t_high = seg_end
                        if key not in object_appearance_times_by_key or not _object_appears_in_range(
                            key, t_low, t_high, object_appearance_times_by_key
                        ):
                            continue
                        segments_per_object_active[key].append({
                            "segment_idx": segment_idx,
                            "overlap_frac": overlap_frac,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                        })
                        if key not in appearance_time_per_object:
                            appearance_time_per_object[key] = seg_start

            for ann in annotations_inactive_per_object:
                if "query_object" not in ann:
                    continue
                usage_label = ann.get("is_passive_usage")
                ann_start = ann.get("start_time")
                ann_end = ann.get("end_time")
                if not all(
                    isinstance(x, (float, int))
                    for x in [ann_start, ann_end, window_start, window_end]
                ):
                    raise ValueError(f"Invalid annotation: {ann}")
                if ann_start < window_end and ann_end > window_start:
                    overlap_start = max(ann_start, window_start)
                    overlap_end = min(ann_end, window_end)
                    overlap = max(0, overlap_end - overlap_start)
                    ann_duration = ann_end - ann_start
                    if ann_duration == 0:
                        continue
                    overlap_frac = overlap / ann_duration
                    obj_id, subclass, name = ann.get("query_object").split("/", maxsplit=2)
                    category = NOUN_CLASS_NAMES[int(obj_id)]["category"]
                    key = f"{category}/{subclass}/{name}"
                    t_low = ann_start - WINDOW_BEHIND
                    t_high = ann_end
                    if usage_label is True:
                        if overlap_frac >= OVERLAP_TOL:
                            if key in object_appearance_times_by_key and _object_appears_in_range(
                                key, t_low, t_high, object_appearance_times_by_key
                            ):
                                segments_per_object_passive[key].append({
                                    "segment_idx": segment_idx,
                                    "overlap_frac": overlap_frac,
                                    "overlap_start": overlap_start,
                                    "overlap_end": overlap_end,
                                })
                    elif usage_label is False:
                        if overlap_frac >= OVERLAP_TOL:
                            if key in object_appearance_times_by_key and _object_appears_in_range(
                                key, t_low, t_high, object_appearance_times_by_key
                            ):
                                active_segment_indices = {
                                    seg["segment_idx"]
                                    for seg in segments_per_object_active.get(key, [])
                                }
                                if segment_idx not in active_segment_indices:
                                    segments_per_object_unused[key].append({
                                        "segment_idx": segment_idx
                                    })
            segment_idx += 1

        results[video_id] = {
            "video_length": video_length,
            "total_segments": ceil(video_length / STEP_SIZE),
            "segments_per_object_active": dict(segments_per_object_active),
            "segments_per_object_passive": dict(segments_per_object_passive),
            "segments_per_object_unused": dict(segments_per_object_unused),
            "appearance_time_per_object": dict(appearance_time_per_object),
        }

    out_path = os.path.join(OUTPUT_DIR, SLIDING_WINDOW_RESULTS_FILE)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
