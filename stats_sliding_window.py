import os
import json
import csv
import statistics
from math import ceil, floor
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # no display needed when saving to file
import matplotlib.pyplot as plt
from tqdm import tqdm

from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from utils import load_noun_class_names, load_inactive_segments, load_inactive_annotations, get_visor_object_appearance_times
from utils import MIN_DURATION_INACTIVE_SEGMENT, ACTIVE_OBJECTS_DIR, VIDEO_ID_FILE, VIDEO_INFO_FILE

WINDOW_SIZE = 12 # seconds
STEP_SIZE = 3 # seconds
OVERLAP_TOL = 0. # minimum overlap between segment and window to be considered "used"

WINDOW_BEHIND = 3*WINDOW_SIZE # seconds

INACTIVE_ANNOTATIONS_DIR = "vlm_annotations_maxImages1_minSpacing3_20260226"
NOUN_CLASS_NAMES = load_noun_class_names()


def _object_appears_in_range(key, t_low, t_high, object_appearance_times_by_key):
    """Return True if the object has at least one VISOR appearance in [t_low, t_high]."""
    timestamps = object_appearance_times_by_key.get(key, [])
    return any(t_low <= t <= t_high for t in timestamps)


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
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_video_stats = defaultdict(dict)
    all_object_stats = defaultdict(dict)

    for video_id in tqdm(video_ids):
        video_info_row = next((v for v in video_info if v['video_id'] == video_id), None)
        if not video_info_row:
            print(f"Video {video_id} not found in video_info")
            pdb.set_trace()
            raise ValueError(f"Video {video_id} not found in video_info")
        video_length = floor(float(video_info_row['duration']))

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
            raise ValueError(f"object_appearance_times_by_key is empty for video_id={video_id} (missing VISOR annotations or visor-frames_to_timestamps.json)")

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
            annotations_path= os.path.join(INACTIVE_ANNOTATIONS_DIR, f"inactive_segments_{video_id}_labels.jsonl")

            inactive_segments = load_inactive_segments(
                video_id,
                object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM,
                min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT
            )
            annotations_inactive_per_object, keys_missing, keys_null = load_inactive_annotations(
                video_id,
                annotations_filepath=annotations_path,
                object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM,
                min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT
            )

            if keys_missing:
                print(f"Video {video_id}: {len(keys_missing)} missing keys in VLM annotations, skipping video...")
                continue

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
                        t_low = seg_start - WINDOW_BEHIND
                        t_high = seg_end
                        if key not in object_appearance_times_by_key or not _object_appears_in_range(key, t_low, t_high, object_appearance_times_by_key):
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
                # Check for overlap
                if not all(
                    isinstance(x, float) or isinstance(x, int)
                    for x in [ann_start, ann_end, window_start, window_end]
                ):
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
                    obj_id, subclass, name = ann.get("query_object").split("/", maxsplit=2)
                    category = NOUN_CLASS_NAMES[int(obj_id)]["category"]
                    key = f"{category}/{subclass}/{name}"
                    t_low = ann_start - WINDOW_BEHIND
                    t_high = ann_end
                    if usage_label is True:
                        if overlap_frac >= OVERLAP_TOL:
                            if key in object_appearance_times_by_key and _object_appears_in_range(key, t_low, t_high, object_appearance_times_by_key):
                                segments_per_object_passive[key].append(
                                    {"segment_idx": segment_idx, "overlap_frac": overlap_frac, "overlap_start": overlap_start, "overlap_end": overlap_end}
                                )
                    elif usage_label is False:
                        if overlap_frac >= OVERLAP_TOL:
                            if key in object_appearance_times_by_key and _object_appears_in_range(key, t_low, t_high, object_appearance_times_by_key):
                                active_segment_indices = {seg["segment_idx"] for seg in segments_per_object_active.get(key, [])}
                                if segment_idx not in active_segment_indices:
                                    segments_per_object_unused[key].append(
                                        {"segment_idx": segment_idx}
                                    )
            segment_idx += 1

        all_keys = set(
            list(segments_per_object_active.keys())
            + list(segments_per_object_passive.keys())
            + list(segments_per_object_unused.keys())
        )

        ## Calculate video-level stats
        all_video_stats[video_id] = {
            "total_objects": len(appearance_time_per_object),
            "video_length": video_length,
            "total_segments": ceil(video_length / STEP_SIZE),
        }

        # Calculate object-level stats and prepare data for stacked bar chart
        per_object_counts = {}
        fig, ax = plt.subplots(figsize=(10, 5))

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

            # Overlap fraction mean and std for active and passive segments
            if len(active_segments) > 0:
                active_overlap_frac_mean = statistics.mean(
                    [seg.get("overlap_frac") for seg in segments_per_object_active.get(key, [])]
                )
                active_overlap_frac_std = statistics.stdev(
                    [seg.get("overlap_frac") for seg in segments_per_object_active.get(key, [])]
                )
            else:
                active_overlap_frac_mean = None
                active_overlap_frac_std = None
            if len(passive_segments) > 0:
                passive_overlap_frac_mean = statistics.mean(
                    [seg.get("overlap_frac") for seg in segments_per_object_passive.get(key, [])]
                )
                passive_overlap_frac_std = statistics.stdev(
                    [seg.get("overlap_frac") for seg in segments_per_object_passive.get(key, [])]
                )
            else:
                passive_overlap_frac_mean = None
                passive_overlap_frac_std = None

            # Add to all_object_stats
            key_global = f"{video_id}/{key}"
            total_segments_obj = len(all_segments)
            all_object_stats[key_global] = {
                "total_segments": total_segments_obj,
                "in_use_segments": len(in_use_segments),
                "only_active_segments": len(only_active_segments),
                "only_passive_segments": len(only_passive_segments),
                "both_active_and_passive_segments": len(both_active_passive_segments),
                "not_in_use_segments": len(not_in_use_segments),
                "active_overlap_fraction_mean": active_overlap_frac_mean,
                "active_overlap_fraction_std": active_overlap_frac_std,
                "passive_overlap_fraction_mean": passive_overlap_frac_mean,
                "passive_overlap_fraction_std": passive_overlap_frac_std,
            }

            # Store counts for plotting proportions later
            per_object_counts[key] = {
                "total": total_segments_obj,
                "in_use": len(in_use_segments),
                "only_active": len(only_active_segments),
                "only_passive": len(only_passive_segments),
                "both_active_and_passive": len(both_active_passive_segments),
                "not_in_use": len(not_in_use_segments),
            }

        # Prepare stacked bar chart (proportion of segments per object)
        if per_object_counts:
            # Sort object keys by total number of segments (descending)
            sorted_keys = list(per_object_counts.keys())

            x = list(range(len(sorted_keys)))
            only_active_vals = []
            only_passive_vals = []
            both_active_passive_vals = []
            in_use_vals = []
            not_in_use_vals = []
            for k in sorted_keys:
                total = per_object_counts[k]["total"]
                in_use_vals.append(per_object_counts[k]["in_use"])
                only_active_vals.append(per_object_counts[k]["only_active"])
                only_passive_vals.append(per_object_counts[k]["only_passive"])
                both_active_passive_vals.append(per_object_counts[k]["both_active_and_passive"])
                not_in_use_vals.append(per_object_counts[k]["not_in_use"])

            bottom = [0.0] * len(sorted_keys)
            ax.bar(x, only_active_vals, bottom=bottom, color="#b2df8a", label="Only Active")  # pastel green
            bottom = [b + v for b, v in zip(bottom, only_active_vals)]

            ax.bar(x, only_passive_vals, bottom=bottom, color="#a6cee3", label="Only Passive")  # pastel blue
            bottom = [b + v for b, v in zip(bottom, only_passive_vals)]

            ax.bar(
                x,
                both_active_passive_vals,
                bottom=bottom,
                color="#cab2d6",  # pastel purple
                label="Both Active and Passive",
                hatch="///",
            )
            bottom = [b + v for b, v in zip(bottom, both_active_passive_vals)]

            ax.bar(x, not_in_use_vals, bottom=bottom, color="#fdbf6f", label="Not in Use")  # pastel orange

            ax.set_ylabel("Proportion of segments")
            ax.set_xlabel("Object key")
            ax.set_title(f"Segment usage breakdown per object (video {video_id})")
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_keys, rotation=90, fontsize=6)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"stacked_bar_chart_{video_id}.png"))
            plt.close(fig)


    # Usage stats: mean ± std, min, max per stat
    def agg(values):
        values = [v for v in values if v is not None]
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

    # Save all_object_stats and all_video_stats as JSON files
    with open(os.path.join(output_dir, "all_object_stats.json"), "w") as f:
        json.dump(all_object_stats, f, indent=2)

    with open(os.path.join(output_dir, "all_video_stats.json"), "w") as f:
        json.dump(all_video_stats, f, indent=2)

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

    # Plot histograms for each usage_per_object metric, 3 per row
    n_metrics = len(usage_per_object)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), constrained_layout=True)
    axs = axs.flatten() if n_metrics > 1 else [axs]

    for i, (name, u) in enumerate(usage_per_object.items()):
        # Gather all values for this metric across all objects
        values = [d[name] for d in all_object_stats.values() if d[name] is not None]
        axs[i].hist(values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        axs[i].set_title(f"Histogram of {name} (per object per video)")
        axs[i].set_xlabel(name)
        axs[i].set_ylabel("Count")

    # Hide any remaining subplots if n_metrics is not a multiple of n_cols
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("Histograms of Per Object Per Video Usage Metrics")
    plt.savefig(os.path.join(output_dir, "usage_stats_histograms.png"))


if __name__ == "__main__":
    main()
