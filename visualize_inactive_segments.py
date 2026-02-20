#!/usr/bin/env python3
"""
Iteratively visualize inactive segment prompt information:
  - All event history frames (start/end per narration)
  - Frame extracted from the video at frame_after_gap_start

Navigation: arrow keys (left/right), q to quit.

Usage:
    python visualize_inactive_segments.py <video_path> [--segments-dir inactive_segments]
"""

import argparse
import json
import math
import os
import textwrap

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def extract_frames(video_path, frame_numbers):
    """Extract multiple frames in one pass, returned as {frame_number: rgb_array}."""
    cap = cv2.VideoCapture(video_path)
    results = {}
    for fn in sorted(set(frame_numbers)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        ## resize frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        results[fn] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    cap.release()
    return results


def parse_event_history(event_history):
    """Group event_history into (narration, start_frame, end_frame) triples."""
    entries = []
    i = 0
    while i < len(event_history):
        item = event_history[i]
        if item.startswith("narration:"):
            narration = item.split("narration:", 1)[1]
            start_frame = end_frame = None
            if i + 1 < len(event_history) and event_history[i + 1].startswith("frame_id:"):
                start_frame = int(event_history[i + 1].split("frame_id:", 1)[1])
                i += 1
            if i + 1 < len(event_history) and event_history[i + 1].startswith("frame_id:"):
                end_frame = int(event_history[i + 1].split("frame_id:", 1)[1])
                i += 1
            entries.append((narration, start_frame, end_frame))
        i += 1
    return entries


def build_flat_list(data):
    """Flatten {object: [segments]} into [(object, segment_index, segment)] list."""
    items = []
    for obj in sorted(data.keys()):
        for idx, seg in enumerate(data[obj]):
            items.append((obj, idx, seg))
    return items


def render(fig, obj_name, seg_idx, seg, total_for_obj, video_path, current, total):
    fig.clf()

    entries = parse_event_history(seg["event_history"])
    frame_after = seg["frame_after_gap_start"]
    gap_start = seg["start_time"]
    gap_end = seg["end_time"]
    duration = seg["duration_sec"]

    # Build list of (frame_number, label, is_gap_frame) to display
    frame_items = []
    for narration, sf, ef in entries:
        short = textwrap.fill(narration, width=28)
        if sf is not None:
            frame_items.append((sf, f"{short}\n(start: {sf})", False))
        if ef is not None:
            frame_items.append((ef, f"{short}\n(end: {ef})", False))
    frame_items.append((frame_after, f"FRAME AFTER GAP START\n({frame_after})", True))

    all_frame_nums = [fn for fn, _, _ in frame_items]
    frames = extract_frames(video_path, all_frame_nums)

    n_frames = len(frame_items)
    n_cols = min(n_frames, 4)
    n_img_rows = math.ceil(n_frames / n_cols)

    gs = gridspec.GridSpec(
        1 + n_img_rows, n_cols,
        height_ratios=[0.35] + [1] * n_img_rows,
        hspace=0.45, wspace=0.25,
    )

    # --- text info panel ---
    ax_text = fig.add_subplot(gs[0, :])
    narration_bullets = "  |  ".join(
        f"\u2022 {n}" for n, _, _ in entries
    )
    info = (
        f"Object: {obj_name}   (segment {seg_idx + 1}/{total_for_obj})        "
        f"Gap: {gap_start:.2f}s \u2013 {gap_end:.2f}s  ({duration}s)\n"
        f"{narration_bullets}"
    )
    ax_text.text(
        0.02, 0.5, info,
        transform=ax_text.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )
    ax_text.set_axis_off()

    # --- frame grid ---
    for i, (fn, label, is_gap) in enumerate(frame_items):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        img = frames.get(fn)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Frame {fn}\nunavailable",
                    ha="center", va="center", fontsize=10, color="red",
                    transform=ax.transAxes)
        title_color = "#b71c1c" if is_gap else "black"
        ax.set_title(label, fontsize=8, fontweight="bold" if is_gap else "normal",
                     color=title_color)
        if is_gap:
            for spine in ax.spines.values():
                spine.set_edgecolor("#b71c1c")
                spine.set_linewidth(3)
                spine.set_visible(True)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.set_axis_off()

    # hide leftover grid cells
    for i in range(n_frames, n_img_rows * n_cols):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_axis_off()

    fig.suptitle(
        f"Inactive Segment {current + 1} / {total}   "
        f"[\u2190 prev]  [\u2192 next]  [q quit]",
        fontsize=12, fontweight="bold",
    )
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Visualize inactive segment prompts.")
    parser.add_argument("video_path", help="Path to the video file (e.g. data/videos/P37_101.MP4)")
    parser.add_argument("--segments-dir", default="inactive_segments",
                        help="Directory containing inactive segment JSON files")
    args = parser.parse_args()

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    print(f"Video basename: {video_basename}")
    json_path = os.path.join(args.segments_dir, f"inactive_segments_{video_basename}.json")
    if not os.path.exists(json_path):
        print(f"No inactive segments file found at {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    ignore_keys = set()
    if os.path.exists("nouns_to_ignore_keys.txt"):
        with open("nouns_to_ignore_keys.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                ignore_keys.add(line)

    filtered_data = {}
    for obj_key in data.keys():
        if obj_key.split("/")[1] in ignore_keys or obj_key.split("/")[2] in ignore_keys:
            continue
        filtered_data[obj_key] = data[obj_key]
    data = filtered_data

    items = build_flat_list(data)
    if not items:
        print("No inactive segments to visualize.")
        return

    obj_counts = {}
    
    for obj in data:
        obj_counts[obj] = len(data[obj])

    current = [0]

    fig = plt.figure(figsize=(18, 9))

    def show(idx):
        idx = idx % len(items)
        current[0] = idx
        obj, seg_idx, seg = items[idx]
        render(fig, obj, seg_idx, seg, obj_counts[obj],
               args.video_path, idx, len(items))

    def on_key(event):
        if event.key == "right":
            show(current[0] + 1)
        elif event.key == "left":
            show(current[0] - 1)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(0)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


if __name__ == "__main__":
    main()
