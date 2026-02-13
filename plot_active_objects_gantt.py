#!/usr/bin/env python3
"""Plot active object usage from active_objects JSON as a Gantt chart."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed when saving to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import argparse



def load_active_objects(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def build_object_intervals(segments: list) -> dict[str, list[tuple[float, float]]]:
    """For each object, collect all (start, end) time intervals where it appears."""
    obj_intervals: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        start, end = seg["start_time"], seg["end_time"]
        for obj in seg["objects_in_sequence"]:
            obj_intervals.setdefault(obj, []).append((start, end))
    return obj_intervals


def plot_gantt(segments: list, out_path: str | None = None) -> None:
    obj_intervals = build_object_intervals(segments)
    # Order objects by first occurrence time (then by name for ties)
    def first_time(obj: str) -> float:
        return min(s for s, e in obj_intervals[obj])

    objects = sorted(obj_intervals.keys(), key=lambda o: (first_time(o), o))
    if not objects:
        print("No objects found.")
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(objects) * 0.35)))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    np.random.seed(42)
    np.random.shuffle(colors)

    for i, obj in enumerate(objects):
        for start, end in obj_intervals[obj]:
            width = end - start
            ax.barh(i, width, left=start, height=0.7, color=colors[i % 20], edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects, fontsize=9)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Active object usage (P06_13)")
    ax.set_xlim(0, max(e for intervals in obj_intervals.values() for s, e in intervals) * 1.02)
    max_time = max(e for intervals in obj_intervals.values() for s, e in intervals) * 1.02
    xticks = np.arange(0, max_time + 1e-6, 30)
    ax.set_xticks(xticks)

    def seconds_to_hms(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02}"

    ax.set_xticklabels([seconds_to_hms(t) for t in xticks], rotation=90)
    ax.invert_yaxis()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot active object usage from active_objects JSON as a Gantt chart.")
    parser.add_argument("video_id", help="Video ID to plot (e.g. P06_13)")
    parser.add_argument(
        "--active-objects-dir",
        type=str,
        default="active_objects",
        help="Directory containing active_objects JSON files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the output plot (e.g. plot.png); if not specified, uses active_objects_<video_id>.png",
    )
    args = parser.parse_args()

    json_path = Path(args.active_objects_dir) / f"active_objects_{args.video_id}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    segments = load_active_objects(str(json_path))
    out_path = args.out or Path(args.active_objects_dir) / f"plots/active_objects_{args.video_id}.png"
    plot_gantt(segments, out_path=out_path)


if __name__ == "__main__":
    main()
