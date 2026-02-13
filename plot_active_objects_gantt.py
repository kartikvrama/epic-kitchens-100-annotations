#!/usr/bin/env python3
"""Plot active object usage from active_objects JSON as a Gantt chart."""
import os
import json
from pathlib import Path

import cv2
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


def sample_frames_from_video(video_path: str, times_sec: list[float]) -> list[np.ndarray] | None:
    """Extract frames at given timestamps (in seconds). Returns list of RGB images or None if video open fails."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    for t in times_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))  # placeholder on read failure
    cap.release()
    return frames


def segments_in_window(segments: list, window_start: float, window_end: float) -> list[dict]:
    """Return segments overlapping [window_start, window_end], with times clipped and shifted to 0..(window_end-window_start)."""
    width = window_end - window_start
    out = []
    for seg in segments:
        s, e = seg["start_time"], seg["end_time"]
        if e <= window_start or s >= window_end:
            continue
        s_clip = max(0, s - window_start)
        e_clip = min(width, e - window_start)
        out.append({
            "start_time": s_clip,
            "end_time": e_clip,
            "objects_in_sequence": seg["objects_in_sequence"],
        })
    return out


def _draw_gantt_ax(
    ax,
    segments: list,
    title: str,
    xlim: tuple[float, float] | None = None,
    video_id: str = "",
) -> None:
    """Draw Gantt chart on an existing axes. If xlim is None, use full extent of segments."""
    obj_intervals = build_object_intervals(segments)
    if not obj_intervals:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No active objects in this window", ha="center", va="center", transform=ax.transAxes)
        return

    def first_time(obj: str) -> float:
        return min(s for s, e in obj_intervals[obj])

    objects = sorted(obj_intervals.keys(), key=lambda o: (first_time(o), o))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    np.random.seed(42)
    np.random.shuffle(colors)

    for i, obj in enumerate(objects):
        for start, end in obj_intervals[obj]:
            width = end - start
            ax.barh(i, width, left=start, height=0.7, color=colors[i % 20], edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1] * 1.02 if xlim[1] > 0 else 1)
    else:
        max_time = max(e for intervals in obj_intervals.values() for s, e in intervals) or 1
        ax.set_xlim(0, max_time * 1.02)
    ax.invert_yaxis()


def plot_gantt(segments: list, out_path: str | None = None, video_id: str = "") -> None:
    obj_intervals = build_object_intervals(segments)
    if not obj_intervals:
        print("No objects found.")
        return

    def first_time(obj: str) -> float:
        return min(s for s, e in obj_intervals[obj])

    objects = sorted(obj_intervals.keys(), key=lambda o: (first_time(o), o))
    fig, ax = plt.subplots(figsize=(14, max(6, len(objects) * 0.35)))
    _draw_gantt_ax(ax, segments, f"Active object usage ({video_id})" if video_id else "Active object usage", video_id=video_id)

    max_time = max(e for intervals in obj_intervals.values() for s, e in intervals)
    xticks = np.arange(0, max_time + 1e-6, 30)
    ax.set_xticks(xticks)
 
    def seconds_to_hms(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02}"

    ax.set_xticklabels([seconds_to_hms(t) for t in xticks], rotation=90)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")


def plot_minute_frames_and_gantt(
    video_path: str,
    segments: list,
    video_id: str,
    out_dir: str | Path,
    num_frames: int = 8,
) -> None:
    """For each full minute, create a 2x1 plot: top = uniformly sampled frames, bottom = Gantt for that minute."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_time = max(seg["end_time"] for seg in segments) if segments else 0
    num_minutes = max(1, int(max_time // 60) + 1)

    for minute_idx in range(num_minutes):
        window_start = minute_idx * 60.0
        window_end = (minute_idx + 1) * 60.0

        # Uniform sample times in [window_start, window_end)
        if num_frames <= 0:
            num_frames = 1
        times_sec = [window_start + (window_end - window_start) * (i + 1) / (num_frames + 1) for i in range(num_frames)]

        frames = sample_frames_from_video(video_path, times_sec)
        if frames is None:
            print(f"Could not open video: {video_path}. Skipping minute plots.")
            return
        # Handle video shorter than this minute
        while len(frames) < num_frames:
            frames.append(np.zeros_like(frames[0]) if frames else np.zeros((480, 640, 3), dtype=np.uint8))

        window_segments = segments_in_window(segments, window_start, window_end)

        max_cols = 4
        n_rows = (num_frames + max_cols - 1) // max_cols
        n_cols = min(num_frames, max_cols)

        fig = plt.figure(figsize=(3.5 * n_cols, 3 * n_rows + 4))
        gs = fig.add_gridspec(2, 1, height_ratios=[n_rows, 1.2], hspace=0.3)
        top_gs = gs[0].subgridspec(n_rows, n_cols)
        for i in range(num_frames):
            r, c = i // n_cols, i % n_cols
            ax = fig.add_subplot(top_gs[r, c])
            ax.imshow(frames[i])
            ax.set_axis_off()
            ax.set_title(f"{times_sec[i] - window_start:.1f}s", fontsize=9)
        ax_gantt = fig.add_subplot(gs[1])
        _draw_gantt_ax(
            ax_gantt,
            window_segments,
            title=f"Minute {minute_idx} ({window_start:.0f}s–{window_end:.0f}s)",
            xlim=(0, 60),
        )

        fig.suptitle(f"{video_id} — minute {minute_idx}", fontsize=12)
        plt.tight_layout()
        out_path = out_dir / f"{video_id}_minute_{minute_idx:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
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
        help="Path to save the output plot (e.g. plot.png); if not specified, uses active_objects/plots/active_objects_<video_id>.png",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to the video file (required for --minute-plots)",
    )
    parser.add_argument(
        "--minute-plots",
        action="store_true",
        help="For each minute, create a 2x1 plot: top = sampled frames from video, bottom = Gantt for that minute",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample per minute (default: 8)",
    )
    args = parser.parse_args()

    json_path = Path(args.active_objects_dir) / f"active_objects_{args.video_id}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    segments = load_active_objects(str(json_path))

    if args.minute_plots:
        if not args.video_path or not Path(args.video_path).exists():
            print("--minute-plots requires --video-path to a valid video file.")
            return
        out_dir = Path(args.active_objects_dir) / "plots" / "minutes"
        plot_minute_frames_and_gantt(
            args.video_path,
            segments,
            args.video_id,
            out_dir,
            num_frames=args.num_frames,
        )
    else:
        out_path = args.out or Path(args.active_objects_dir) / f"plots/active_objects_{args.video_id}.png"
        ## Create directory if it doesn't exist
        os.makedirs(out_path.parent, exist_ok=True)
        plot_gantt(segments, out_path=out_path, video_id=args.video_id)


if __name__ == "__main__":
    main()
