"""
Compute statistics and plots from precomputed sliding-window results.
Reads data_stats/sliding_window_results.json (produced by generate_sliding_windows.py).
"""
import os
import json
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "data_stats"
SLIDING_WINDOW_RESULTS_FILE = "sliding_window_results.json"


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
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_video_stats = {}
    all_object_stats = {}

    for video_id, data in results.items():
        video_length = data["video_length"]
        total_segments = data["total_segments"]
        segments_per_object_active = data["segments_per_object_active"]
        segments_per_object_passive = data["segments_per_object_passive"]
        segments_per_object_unused = data["segments_per_object_unused"]
        appearance_time_per_object = data["appearance_time_per_object"]

        all_keys = set(
            list(segments_per_object_active.keys())
            + list(segments_per_object_passive.keys())
            + list(segments_per_object_unused.keys())
        )

        all_video_stats[video_id] = {
            "total_objects": len(appearance_time_per_object),
            "video_length": video_length,
            "total_segments": total_segments,
        }

        per_object_counts = {}
        fig, ax = plt.subplots(figsize=(10, 5))

        for key in all_keys:
            active_segments = set(
                seg["segment_idx"] for seg in segments_per_object_active.get(key, [])
            )
            passive_segments = set(
                seg["segment_idx"] for seg in segments_per_object_passive.get(key, [])
            )
            unused_segments = set(
                seg["segment_idx"] for seg in segments_per_object_unused.get(key, [])
            )

            all_segments = active_segments | passive_segments | unused_segments
            in_use_segments = active_segments | passive_segments
            only_active_segments = active_segments - passive_segments
            only_passive_segments = passive_segments - active_segments
            both_active_passive_segments = active_segments & passive_segments
            not_in_use_segments = unused_segments

            if len(active_segments) > 0:
                active_overlap_frac_mean = statistics.mean(
                    seg.get("overlap_frac") for seg in segments_per_object_active.get(key, [])
                )
                active_overlap_frac_std = statistics.stdev(
                    seg.get("overlap_frac") for seg in segments_per_object_active.get(key, [])
                )
            else:
                active_overlap_frac_mean = None
                active_overlap_frac_std = None
            if len(passive_segments) > 0:
                passive_overlap_frac_mean = statistics.mean(
                    seg.get("overlap_frac") for seg in segments_per_object_passive.get(key, [])
                )
                passive_overlap_frac_std = statistics.stdev(
                    seg.get("overlap_frac") for seg in segments_per_object_passive.get(key, [])
                )
            else:
                passive_overlap_frac_mean = None
                passive_overlap_frac_std = None

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

            per_object_counts[key] = {
                "total": total_segments_obj,
                "in_use": len(in_use_segments),
                "only_active": len(only_active_segments),
                "only_passive": len(only_passive_segments),
                "both_active_and_passive": len(both_active_passive_segments),
                "not_in_use": len(not_in_use_segments),
            }

        if per_object_counts:
            sorted_keys = list(per_object_counts.keys())
            x = list(range(len(sorted_keys)))
            only_active_vals = [per_object_counts[k]["only_active"] for k in sorted_keys]
            only_passive_vals = [per_object_counts[k]["only_passive"] for k in sorted_keys]
            both_active_passive_vals = [
                per_object_counts[k]["both_active_and_passive"] for k in sorted_keys
            ]
            not_in_use_vals = [per_object_counts[k]["not_in_use"] for k in sorted_keys]

            bottom = [0.0] * len(sorted_keys)
            ax.bar(x, only_active_vals, bottom=bottom, color="#b2df8a", label="Only Active")
            bottom = [b + v for b, v in zip(bottom, only_active_vals)]
            ax.bar(x, only_passive_vals, bottom=bottom, color="#a6cee3", label="Only Passive")
            bottom = [b + v for b, v in zip(bottom, only_passive_vals)]
            ax.bar(
                x,
                both_active_passive_vals,
                bottom=bottom,
                color="#cab2d6",
                label="Both Active and Passive",
                hatch="///",
            )
            bottom = [b + v for b, v in zip(bottom, both_active_passive_vals)]
            ax.bar(x, not_in_use_vals, bottom=bottom, color="#fdbf6f", label="Not in Use")

            ax.set_ylabel("Proportion of segments")
            ax.set_xlabel("Object key")
            ax.set_title(f"Segment usage breakdown per object (video {video_id})")
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_keys, rotation=90, fontsize=6)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"stacked_bar_chart_{video_id}.png"))
            plt.close(fig)

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
    usage_global = {
        name: agg([d[name] for d in all_video_stats.values()])
        for name in global_stat_names
    }

    object_stat_names = (
        list(all_object_stats.values())[0].keys() if all_object_stats else []
    )
    usage_per_object = {
        name: agg([d[name] for d in all_object_stats.values()])
        for name in object_stat_names
    }

    with open(os.path.join(OUTPUT_DIR, "all_object_stats.json"), "w") as f:
        json.dump(all_object_stats, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "all_video_stats.json"), "w") as f:
        json.dump(all_video_stats, f, indent=2)

    summary = {
        "per_video": {
            name: {
                "mean_std": f"{u['mean']:.4f} ± {u['std']:.4f}",
                "min": u["min"],
                "max": u["max"],
            }
            for name, u in usage_global.items()
        },
        "per_object_per_video": {
            name: {
                "mean_std": f"{u['mean']:.4f} ± {u['std']:.4f}",
                "min": u["min"],
                "max": u["max"],
            }
            for name, u in usage_per_object.items()
        },
    }
    with open(os.path.join(OUTPUT_DIR, "usage_stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "usage_stats_summary.md"), "w") as f:
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

    n_metrics = len(usage_per_object)
    if n_metrics > 0:
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), constrained_layout=True
        )
        axs = axs.flatten() if n_metrics > 1 else [axs]
        for i, (name, u) in enumerate(usage_per_object.items()):
            values = [d[name] for d in all_object_stats.values() if d[name] is not None]
            axs[i].hist(values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
            axs[i].set_title(f"Histogram of {name} (per object per video)")
            axs[i].set_xlabel(name)
            axs[i].set_ylabel("Count")
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")
        plt.suptitle("Histograms of Per Object Per Video Usage Metrics")
        plt.savefig(os.path.join(OUTPUT_DIR, "usage_stats_histograms.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
