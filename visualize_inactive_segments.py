#!/usr/bin/env python3
"""
Iteratively visualize inactive segment prompt information:
  - Object crops (from crop_from_previous_active, when present; multiple per segment)
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
import re
import textwrap

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def extract_frames(video_path, frame_numbers, max_size=400):
    """Extract multiple frames in one pass; optionally scale so longest side is max_size. Returns {frame_number: rgb_array}."""
    cap = cv2.VideoCapture(video_path)
    results = {}
    for fn in sorted(set(frame_numbers)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            results[fn] = None
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if max_size and max(frame.shape[:2]) > max_size:
            h, w = frame.shape[:2]
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        results[fn] = frame
    cap.release()
    return results


def extract_frames_full_res(video_path, frame_numbers):
    """Extract frames at original resolution, returned as {frame_number: rgb_array}."""
    cap = cv2.VideoCapture(video_path)
    results = {}
    for fn in sorted(set(frame_numbers)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        results[fn] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    cap.release()
    return results


def crop_bbox_from_image(rgb_array, bbox):
    """Crop bbox [x, y, w, h] from an RGB image. Returns RGB crop or None."""
    if rgb_array is None or rgb_array.size == 0:
        return None
    h_img, w_img = rgb_array.shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(x1 + 1, min(x + w, w_img))
    y2 = max(y1 + 1, min(y + h, h_img))
    crop = rgb_array[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def parse_crop_specs(crop_from_previous_active):
    """
    Parse crop_from_previous_active into list of (frame_id, bbox).
    Supports: list of {frame_path, bbox} or legacy single {crop_frame_id, crop_bbox}.
    """
    if not crop_from_previous_active:
        return []
    specs = []
    items = crop_from_previous_active if isinstance(crop_from_previous_active, list) else [crop_from_previous_active]
    for item in items:
        if "crop_frame_id" in item and "crop_bbox" in item:
            specs.append((item["crop_frame_id"], item["crop_bbox"]))
        elif "frame_path" in item and "bbox" in item:
            # e.g. "frame_0000003581.jpg" -> 3581
            match = re.search(r"frame_(\d+)", item["frame_path"])
            if match:
                frame_id = int(match.group(1))
                specs.append((frame_id, item["bbox"]))
    return specs


def extract_object_crops(video_path, crop_specs):
    """Extract multiple object crops. crop_specs is list of (frame_id, bbox). Returns list of RGB arrays."""
    if not crop_specs:
        return []
    unique_frames = list({fid for fid, _ in crop_specs})
    full_res = extract_frames_full_res(video_path, unique_frames)
    return [crop_bbox_from_image(full_res.get(fid), bbox) for fid, bbox in crop_specs]


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

    # Object crops from previous active (can be multiple per segment)
    crop_specs = parse_crop_specs(seg.get("crop_from_previous_active"))
    object_crop_rgbs = extract_object_crops(video_path, crop_specs)

    # Collect all frame numbers: event start/end per narration, and frame_after_gap_start
    event_frame_nums = []
    for _, sf, ef in entries:
        if sf is not None:
            event_frame_nums.append(sf)
        if ef is not None:
            event_frame_nums.append(ef)
    all_frame_nums = list(set(event_frame_nums + [frame_after]))
    frames = extract_frames(video_path, all_frame_nums, max_size=400)

    n_crop = len(object_crop_rgbs)
    n_events = len(entries)
    n_cols = max(3, min(n_crop, 6))  # event rows use 3 cols; crop row may need up to 6
    n_crop_rows = math.ceil(n_crop / n_cols) if n_crop else 0
    # Rows: header, "Object crops" label, crop row(s), "Event history" label, n_events event rows, "Frame after gap" label, gap frame
    n_header = 1
    n_crop_label = 1
    n_event_label = 1
    n_gap_label = 1
    n_gap_row = 1
    n_rows = n_header + n_crop_label + n_crop_rows + n_event_label + n_events + n_gap_label + n_gap_row
    height_ratios = [0.3] + [0.15] + [1.2] * n_crop_rows + [0.2] + [1.0] * n_events + [0.15] + [1.4]
    gs = gridspec.GridSpec(n_rows, n_cols, height_ratios=height_ratios, hspace=0.35, wspace=0.2)

    row = 0
    # --- Header: object, segment, gap times ---
    ax_header = fig.add_subplot(gs[row, :])
    narration_bullets = "  |  ".join(f"\u2022 {n}" for n, _, _ in entries)
    info = (
        f"Object: {obj_name}   Segment {seg_idx + 1}/{total_for_obj}   "
        f"Gap: {gap_start:.2f}s \u2013 {gap_end:.2f}s  ({duration:.1f}s)\n"
        f"Event narrations: {narration_bullets}"
    )
    ax_header.text(
        0.02, 0.5, info,
        transform=ax_header.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="center",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#ccc"),
    )
    ax_header.set_axis_off()
    row += 1

    # --- Section: Query object crops ---
    ax_crop_title = fig.add_subplot(gs[row, :])
    ax_crop_title.text(0.02, 0.5, "Query object crops (from previous active)", fontsize=11, fontweight="bold", color="#0d47a1")
    ax_crop_title.set_axis_off()
    row += 1

    for i, crop_rgb in enumerate(object_crop_rgbs):
        r, c = row + i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[r, c])
        if crop_rgb is not None:
            ax.imshow(crop_rgb)
        else:
            ax.text(0.5, 0.5, "Crop\nunavailable", ha="center", va="center", fontsize=9, color="red", transform=ax.transAxes)
        ax.set_title(f"Crop {i + 1}", fontsize=8, fontweight="bold", color="#0d47a1")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0d47a1")
            spine.set_linewidth(2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    row += n_crop_rows

    # --- Section: Event history (narration + scene images) ---
    ax_ev_title = fig.add_subplot(gs[row, :])
    ax_ev_title.text(0.02, 0.5, "Event history: narrations and scene frames", fontsize=11, fontweight="bold", color="#333")
    ax_ev_title.set_axis_off()
    row += 1

    for ev_idx, (narration, sf, ef) in enumerate(entries):
        # One row per event: [narration text] [start frame] [end frame]
        ax_nar = fig.add_subplot(gs[row, 0])
        ax_nar.text(
            0.05, 0.95, textwrap.fill(narration, width=24),
            transform=ax_nar.transAxes, fontsize=9, verticalalignment="top", wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#81c784"),
        )
        ax_nar.set_axis_off()

        ax_start = fig.add_subplot(gs[row, 1])
        if sf is not None:
            img = frames.get(sf)
            if img is not None:
                ax_start.imshow(img)
            else:
                ax_start.text(0.5, 0.5, f"Frame {sf}\nunavailable", ha="center", va="center", fontsize=8, color="red", transform=ax_start.transAxes)
        else:
            ax_start.text(0.5, 0.5, "No start frame", ha="center", va="center", fontsize=8, transform=ax_start.transAxes)
        ax_start.set_title(f"Start frame {sf}" if sf is not None else "Start", fontsize=8)
        ax_start.set_axis_off()

        ax_end = fig.add_subplot(gs[row, 2])
        if ef is not None:
            img = frames.get(ef)
            if img is not None:
                ax_end.imshow(img)
            else:
                ax_end.text(0.5, 0.5, f"Frame {ef}\nunavailable", ha="center", va="center", fontsize=8, color="red", transform=ax_end.transAxes)
        else:
            ax_end.text(0.5, 0.5, "No end frame", ha="center", va="center", fontsize=8, transform=ax_end.transAxes)
        ax_end.set_title(f"End frame {ef}" if ef is not None else "End", fontsize=8)
        ax_end.set_axis_off()
        row += 1

    # --- Section: Frame after gap start ---
    ax_gap_title = fig.add_subplot(gs[row, :])
    ax_gap_title.text(0.02, 0.5, "Frame after gap start", fontsize=11, fontweight="bold", color="#b71c1c")
    ax_gap_title.set_axis_off()
    row += 1

    ax_gap = fig.add_subplot(gs[row, :])
    gap_img = frames.get(frame_after)
    if gap_img is not None:
        ax_gap.imshow(gap_img)
    else:
        ax_gap.text(0.5, 0.5, f"Frame {frame_after} unavailable", ha="center", va="center", fontsize=12, color="red", transform=ax_gap.transAxes)
    ax_gap.set_title(f"Frame {frame_after} (first frame after gap start)", fontsize=10, fontweight="bold", color="#b71c1c")
    for spine in ax_gap.spines.values():
        spine.set_edgecolor("#b71c1c")
        spine.set_linewidth(2)
    ax_gap.set_axis_off()

    fig.suptitle(
        f"Inactive Segment {current + 1} / {total}   [\u2190 prev]  [\u2192 next]  [q quit]",
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
