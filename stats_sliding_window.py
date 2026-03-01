import os
import json
import csv
import statistics
from math import ceil, floor
from collections import defaultdict

from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from utils import load_noun_class_names, load_inactive_segments, load_inactive_annotations
from utils import MIN_DURATION_INACTIVE_SEGMENT, ACTIVE_OBJECTS_DIR, VIDEO_ID_FILE, VIDEO_INFO_FILE

WINDOW_SIZE = 12 # seconds
STEP_SIZE = 3 # seconds
OVERLAP_TOL = 0.25 # minimum overlap between segment and window to be considered

INACTIVE_ANNOTATIONS_DIR = "vlm_annotations_maxImages1_minSpacing3_20260226"
NOUN_CLASS_NAMES = load_noun_class_names()


import pdb
def main():
    with open(VIDEO_INFO_FILE, "r") as f:
        reader = csv.DictReader(f)
        video_info = list(reader)

    with open(VIDEO_ID_FILE, "r") as f:
        reader = csv.reader(f)
        video_ids = [row[0] for row in reader][1:]

    # Save all segment usage stats in a JSON file
    output_dir = "data_stats"
    os.makedirs(output_dir, exist_ok=True)


    all_video_stats = defaultdict(dict)
    all_object_stats = defaultdict(dict)

    for video_id in video_ids:
        video_info_row = next((v for v in video_info if v['video_id'] == video_id), None)
        if not video_info_row:
            print(f"Video {video_id} not found in video_info")
            pdb.set_trace()
            raise ValueError(f"Video {video_id} not found in video_info")
        video_length = floor(float(video_info_row['duration']))

        segments_per_object_active = defaultdict(list[dict])
        segments_per_object_passive = defaultdict(list[dict])
        segments_per_object_unused = defaultdict(list[dict])
        appearance_time_per_object = defaultdict(float)

        segment_idx = 0
        for window_start in range(0, video_length, STEP_SIZE):
            window_end = min(window_start + WINDOW_SIZE, video_length)
            if window_end > video_length:
                break

            with open(os.path.join(ACTIVE_OBJECTS_DIR, f"active_objects_{video_id}.json"), "r") as f:
                annotations_active_per_segment = json.load(f)

            segments_inactive = load_inactive_segments(
                video_id,
                object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM,
                min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT
            )

            annotations_path= os.path.join(INACTIVE_ANNOTATIONS_DIR, f"inactive_segments_{video_id}_labels.jsonl")

            annotations_inactive_per_object = load_inactive_annotations(
                annotations_path,
                segments_inactive,
                object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM,
                min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT
            )

            # Find overlapping active segments between window_start and window_end
            for seg in annotations_active_per_segment:
                seg_start = seg.get("start_time")
                seg_end = seg.get("end_time")
                # Check for overlap
                if seg_start < window_end and seg_end > window_start:
                    # Calculate overlap between segment and window
                    overlap_start = max(seg_start, window_start)
                    overlap_end = min(seg_end, window_end)
                    overlap = max(0, overlap_end - overlap_start)
                    seg_duration = seg_end - seg_start
                    if seg_duration == 0:  # avoid division by zero
                        continue
                    overlap_frac = overlap / seg_duration
                    if overlap_frac < OVERLAP_TOL:
                        continue
                    for obj in seg.get("objects_in_sequence", []):
                        if obj['subclass_name'] in OBJECTS_TO_EXCLUDE_FROM_VLM:
                            continue
                        key = f"{obj['category']}/{obj['subclass_name']}/{obj['name']}"
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
                # Check for overlap
                if not all(isinstance(x, float) or isinstance(x, int) for x in [ann_start, ann_end, window_start, window_end]):
                    print(f"Invalid annotation: {ann}")
                    pdb.set_trace()
                    raise ValueError(f"Invalid annotation: {ann}")
                if ann_start < window_end and ann_end > window_start:
                    # Calculate overlap between annotation and window
                    overlap_start = max(ann_start, window_start)
                    overlap_end = min(ann_end, window_end)
                    overlap = max(0, overlap_end - overlap_start)
                    ann_duration = ann_end - ann_start
                    if ann_duration == 0:  # avoid division by zero
                        continue
                    overlap_frac = overlap / ann_duration
                    if overlap_frac < OVERLAP_TOL: ## TODO: Is this correct?
                        continue
                    obj_id, subclass, name = ann.get("query_object").split("/", maxsplit=2)
                    category = NOUN_CLASS_NAMES[int(obj_id)]["category"]
                    key = f"{category}/{subclass}/{name}"
                    if usage_label is True:
                        segments_per_object_passive[key].append(
                            {
                                "segment_idx": segment_idx,
                                "overlap_frac": overlap_frac,
                                "overlap_start": overlap_start,
                                "overlap_end": overlap_end,
                            }
                        )
                    else:
                        segments_per_object_unused[key].append(
                            {
                                "segment_idx": segment_idx,
                                "overlap_frac": overlap_frac,
                                "overlap_start": overlap_start,
                                "overlap_end": overlap_end,
                            }
                        )

            segment_idx += 1



        all_keys = set(
            list(segments_per_object_active.keys())
            + list(segments_per_object_passive.keys())
            + list(segments_per_object_unused.keys())
        )

        all_video_stats[video_id] = {
            "total_objects": len(appearance_time_per_object),
            "video_length": video_length,
            "total_segments": ceil(video_length / STEP_SIZE),
        }

        for key in all_keys:
            active_segments = set([seg["segment_idx"] for seg in segments_per_object_active.get(key, [])])
            passive_segments = set([seg["segment_idx"] for seg in segments_per_object_passive.get(key, [])])
            unused_segments = set([seg["segment_idx"] for seg in segments_per_object_unused.get(key, [])])

            all_segments = active_segments | passive_segments | unused_segments

            # Segments where object is in use (active or passive)
            in_use_segments = active_segments | passive_segments
            # Segments only active
            only_active_segments = active_segments - passive_segments
            # Segments only passive
            only_passive_segments = passive_segments - active_segments
            # Segments both active and passive
            both_active_passive_segments = active_segments & passive_segments
            # Segments not in use
            not_in_use_segments = unused_segments

            key_global = f"{video_id}/{key}"

            all_object_stats[key_global] = {
                "total_segments": len(all_segments),
                "in_use_segments": len(in_use_segments),
                "only_active_segments": len(only_active_segments),
                "only_passive_segments": len(only_passive_segments),
                "both_active_and_passive_segments": len(both_active_passive_segments),
                "not_in_use_segments": len(not_in_use_segments),
            }


    # Usage stats: mean ± std, min, max per stat
    def agg(values):
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    global_stat_names = list(all_video_stats.values())[0].keys() if all_video_stats else []
    usage_global = {name: agg([d[name] for d in all_video_stats.values()]) for name in global_stat_names}

    object_stat_names = list(all_object_stats.values())[0].keys() if all_object_stats else []
    usage_per_object = {name: agg([d[name] for d in all_object_stats.values()]) for name in object_stat_names}

    summary = {
        "per_video": {name: {"mean_std": f"{u['mean']:.4f} ± {u['std']:.4f}", "min": u["min"], "max": u["max"]} for name, u in usage_global.items()},
        "per_object_per_video": {name: {"mean_std": f"{u['mean']:.4f} ± {u['std']:.4f}", "min": u["min"], "max": u["max"]} for name, u in usage_per_object.items()},
    }
    with open(os.path.join(output_dir, "usage_stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ## Write to markdown file
    with open(os.path.join(output_dir, "usage_stats_summary.md"), "w") as f:
        f.write("# Usage Stats Summary\n")
        f.write("## Per Video\n")
        for name, u in usage_global.items():
            f.write(f"### {name}\n")
            f.write(f"Mean: {u['mean']:.4f} ± {u['std']:.4f}\n")
            f.write(f"Min: {u['min']:.4f}\n")
            f.write(f"Max: {u['max']:.4f}\n")
        
        f.write("## Per Object per Video\n")
        for name, u in usage_per_object.items():
            f.write(f"### {name}\n")
            f.write(f"Mean: {u['mean']:.4f} ± {u['std']:.4f}\n")
            f.write(f"Min: {u['min']:.4f}\n")
            f.write(f"Max: {u['max']:.4f}\n")

if __name__ == "__main__":
    main()
