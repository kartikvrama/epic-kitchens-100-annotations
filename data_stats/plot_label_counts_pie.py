#!/usr/bin/env python3
"""Generate two pie charts from label_counts_summary: behind and ahead usage breakdown."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Reuse labels from stats_sliding_window_labels
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stats_sliding_window_labels import LABELS

# From label_counts_summary.md (behind / ahead sections)
BEHIND = {"active": 59567, "passive": 108, "both": 50940, "unused": 39922}
AHEAD = {"active": 40689, "passive": 19984, "both": 34916, "unused": 54948}
# Pastel: blue, purple, red, orange
COLORS = ["#aec7e8", "#c5b0d5", "#ff9896", "#ffbb78"]
# Same display labels as plot_segments_by_usage.py
USAGE_LABELS = {
    "active": "Active Only",
    "both": "Active + Passive",
    "passive": "Passive Only",
    "unused": "Not in Use",
}
SEGMENT_TYPE_LABELS = {"behind": "q^past", "ahead": "q^future"}


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "data_stats" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fontsize = 16
    for seg_type, data in [("behind", BEHIND), ("ahead", AHEAD)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sizes = [data[k] for k in LABELS]
        ax.pie(
            sizes,
            labels=[USAGE_LABELS[k] for k in LABELS],
            colors=COLORS,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": fontsize},
        )
        ax.set_title(SEGMENT_TYPE_LABELS[seg_type], fontsize=fontsize)
        out_path = out_dir / f"label_counts_pie_{seg_type}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
