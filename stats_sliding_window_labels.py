"""
Per-video per-object counts for behind, ahead, and within labels.
Reads data_stats/sliding_window_results.json (produced by generate_sliding_windows.py).
Writes data_stats/label_counts_per_video_per_object.json and data_stats/label_counts_summary.md.
"""
import os
import json
import statistics
from collections import Counter, defaultdict

OUTPUT_DIR = "data_stats"
SLIDING_WINDOW_RESULTS_FILE = "sliding_window_results.json"
LABEL_COUNTS_JSON = "label_counts_per_video_per_object.json"
LABEL_COUNTS_SUMMARY_MD = "label_counts_summary.md"

LABELS = ("active", "passive", "both", "unused")
INTERVALS = ("behind", "ahead")
LABEL_LIST_KEYS = ("labels_behind_to_seg", "labels_ahead_of_seg")


def main():
    results_path = os.path.join(OUTPUT_DIR, SLIDING_WINDOW_RESULTS_FILE)
    if not os.path.isfile(results_path):
        raise FileNotFoundError(
            f"Sliding window results not found: {results_path}. "
            "Run generate_sliding_windows.py first."
        )
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # per_video_per_object[video_id][object_key] = { "behind": {active: n, ...}, "ahead": {...}, "within": {...} }
    per_object = defaultdict(dict)

    ## structure of labels_per_object: labels_per_object["object_key"] = {"labels_behind_to_seg": {"segment_idx": "label"}, "labels_ahead_of_seg": {"segment_idx": "label"}}
    objects_per_video = []
    segments_per_object_behind = []
    segments_per_object_ahead = []
    for video_id, data in results.items():
        labels_per_object = data.get("labels_per_object") or {}
        objects_per_video.append(len(labels_per_object))


        for obj_key, label_dict in labels_per_object.items():
            segments_per_object_behind.append(len(label_dict.get("labels_behind_to_seg")))
            segments_per_object_ahead.append(len(label_dict.get("labels_ahead_of_seg")))
            counts = {}
            for interval, list_key in zip(INTERVALS, LABEL_LIST_KEYS):
                lst = label_dict.get(list_key) or []
                counts[interval] = dict(Counter(lst.values()))
            per_object[video_id][obj_key] = counts

    # Convert to plain dict for JSON
    out = {"per_object": dict(per_object)}

    json_path = os.path.join(OUTPUT_DIR, LABEL_COUNTS_JSON)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")

    # Summary: aggregate counts across all videos/objects per interval
    summary = {}
    for interval in INTERVALS:
        totals = {label: 0 for label in LABELS}
        for _, objs in per_object.items():
            for _, counts in objs.items():
                for label in LABELS:
                    totals[label] += counts[interval].get(label, 0)
        summary[interval] = totals

    md_path = os.path.join(OUTPUT_DIR, LABEL_COUNTS_SUMMARY_MD)
    with open(md_path, "w") as f:
        f.write("# Label counts summary (behind / ahead / within)\n\n")
        f.write("Totals across all videos and all objects.\n\n")
        for interval in INTERVALS:
            f.write(f"## {interval}\n")
            for label in LABELS:
                f.write(f"- **{label}**: {summary[interval][label]}\n")
            f.write("\n")
        f.write(f"Number of objects per video: {statistics.mean(objects_per_video)} +- {statistics.stdev(objects_per_video)}\n")
        f.write(f"Number of segments per object behind: {statistics.mean(segments_per_object_behind)} +- {statistics.stdev(segments_per_object_behind)}\n")
        f.write(f"Number of segments per object ahead: {statistics.mean(segments_per_object_ahead)} +- {statistics.stdev(segments_per_object_ahead)}\n")
    print(f"Wrote {md_path}")
    print(f"Number of objects per video: {statistics.mean(objects_per_video)} +- {statistics.stdev(objects_per_video)}")
    print(f"Number of segments per object behind: {statistics.mean(segments_per_object_behind)} +- {statistics.stdev(segments_per_object_behind)}")
    print(f"Number of segments per object ahead: {statistics.mean(segments_per_object_ahead)} +- {statistics.stdev(segments_per_object_ahead)}")


if __name__ == "__main__":
    main()
