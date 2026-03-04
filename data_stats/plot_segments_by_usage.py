#!/usr/bin/env python3
"""
Pivot label_counts_per_video_per_object.json to compute mean & std of segment counts
for behind/ahead, split by usage (active, both, passive, unused), and plot grouped bars.
"""

import json
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

import matplotlib.pyplot as plt
from pathlib import Path



USAGE_LABELS = ["active", "both", "passive", "unused"]
SEGMENT_TYPES = ["behind", "ahead"]


def load_and_aggregate(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    # Collect counts per (segment_type, usage): list of counts (one per video-object pair)
    # Use 0 when a usage key is missing for that object
    buckets = {
        st: {u: [] for u in USAGE_LABELS}
        for st in SEGMENT_TYPES
    }

    per_object = data.get("per_object", data)
    for video_id, objects in per_object.items():
        for obj_name, segments in objects.items():
            for seg_type in SEGMENT_TYPES:
                usage_counts = segments.get(seg_type, {})
                for u in USAGE_LABELS:
                    count = usage_counts.get(u, 0)
                    buckets[seg_type][u].append(count)

    # Compute mean and std for each (segment_type, usage)
    means = {st: {} for st in SEGMENT_TYPES}
    stds = {st: {} for st in SEGMENT_TYPES}
    for st in SEGMENT_TYPES:
        for u in USAGE_LABELS:
            arr = np.array(buckets[st][u])
            means[st][u] = float(np.mean(arr))
            stds[st][u] = float(np.std(arr)) if len(arr) > 1 else 0.0

    return means, stds


def plot_grouped_bars(means, stds, out_path: str):
    x_labels = SEGMENT_TYPES
    x = np.arange(len(x_labels))
    n_groups = len(USAGE_LABELS)
    total_width = 0.8
    bar_width = total_width / n_groups
    offset = (np.arange(n_groups) - (n_groups - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(12, 8))
    # Pastel colors: blue, purple, red, orange for active, both, passive, unused
    colors = ["#aec7e8", "#c5b0d5", "#ff9896", "#ffbb78"]

    usage_labels = {
        "active": "Active Only",
        "both": "Active + Passive",
        "passive": "Passive Only",
        "unused": "Not in Use",
    }
    segment_type_labels = {"behind": "q^past", "ahead": "q^future"}
    for i, usage in enumerate(USAGE_LABELS):
        vals = [means[st][usage] for st in SEGMENT_TYPES]
        errs = [stds[st][usage] for st in SEGMENT_TYPES]
        ax.bar(
            x + offset[i],
            vals,
            bar_width,
            yerr=errs,
            label=usage_labels[usage],
            color=colors[i],
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([segment_type_labels[st] for st in SEGMENT_TYPES])
    ax.legend(title="Usage")
    ax.set_title("Mean (± std) segments per video-object by type and usage")
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    base = Path(__file__).resolve().parent.parent
    json_path = base / "data_stats" / "label_counts_per_video_per_object.json"
    out_path = base / "data_stats" / "plots" / "segments_by_usage_bar.png"

    means, stds = load_and_aggregate(str(json_path))

    # Persist pivot (mean & std)
    pivot_path = base / "data_stats" / "segments_by_usage_mean_std.json"
    pivot = {
        st: {
            u: {"mean": means[st][u], "std": stds[st][u]}
            for u in USAGE_LABELS
        }
        for st in SEGMENT_TYPES
    }
    with open(pivot_path, "w") as f:
        json.dump(pivot, f, indent=2)
    print(f"Pivot saved: {pivot_path}")

    # Print summary
    print("Mean (std) number of segments per video-object:")
    for st in SEGMENT_TYPES:
        print(f"  {st}:")
        for u in USAGE_LABELS:
            print(f"    {u}: {means[st][u]:.2f} ({stds[st][u]:.2f})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plot_grouped_bars(means, stds, str(out_path))


if __name__ == "__main__":
    main()
