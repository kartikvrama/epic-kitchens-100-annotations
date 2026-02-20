#!/usr/bin/env python3
"""
Visualize active segments from active_objects JSON:
  - Frames with bounding boxes around each active object
  - Action narrations and list of active objects

Navigation: arrow keys (left/right) to change segment, q to quit.

Usage:
    python visualize_active_segments.py <video_path> [--active-objects-dir active_objects] [--video-id VIDEO_ID]
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


# Distinct colors (R, G, B) for drawing bboxes per object
OBJECT_COLORS = [
    (31, 119, 180),   # blue
    (255, 127, 14),   # orange
    (44, 160, 44),    # green
    (214, 39, 40),    # red
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
    (227, 119, 194),  # pink
    (127, 127, 127),  # gray
    (188, 189, 34),   # olive
    (23, 190, 207),   # cyan
]


def extract_frames(video_path, frame_numbers, max_size=720):
    """Extract frames from video by frame index. Returns {frame_number: rgb_array}. Optionally scale longest side to max_size."""
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


def frame_id_from_path(frame_path):
    """Extract frame id from frame_path e.g. 'frame_0000000297.jpg' -> 297."""
    match = re.search(r"frame_(\d+)", frame_path)
    return int(match.group(1)) if match else None


def build_frame_bboxes(segment):
    """Build frame_to_specs: {frame_id: [(bbox, label, color), ...]} for all frames with bboxes in this segment."""
    frame_to_specs = {}
    color_idx = 0
    for obj in segment.get("objects_in_sequence", []):
        label = f"{obj.get('class_name', '')} ({obj.get('name', '')})".strip(" ()") or obj.get("name", "?")
        color = OBJECT_COLORS[color_idx % len(OBJECT_COLORS)]
        color_idx += 1
        for crop in obj.get("image_crops", []):
            frame_path = crop.get("frame_path")
            bbox = crop.get("bbox")
            if not frame_path or not bbox or len(bbox) != 4:
                continue
            fid = frame_id_from_path(frame_path)
            if fid is None:
                continue
            if fid not in frame_to_specs:
                frame_to_specs[fid] = []
            frame_to_specs[fid].append((bbox, label, color))
    return frame_to_specs


def draw_bboxes(rgb_array, frame_specs):
    """Draw bboxes and labels on rgb_array. frame_specs = [(bbox, label, color), ...]. bbox is [x, y, w, h]. Returns new RGB array."""
    if rgb_array is None or rgb_array.size == 0:
        return rgb_array
    out = rgb_array.copy()
    h_img, w_img = out.shape[:2]
    for bbox, label, color in frame_specs:
        x, y, w, h = [int(round(v)) for v in bbox]
        x1 = max(0, min(x, w_img - 1))
        y1 = max(0, min(y, h_img - 1))
        x2 = max(x1 + 1, min(x + w, w_img))
        y2 = max(y1 + 1, min(y + h, h_img))
        # cv2 uses BGR
        bgr = (color[2], color[1], color[0])
        thickness = max(1, min(3, max(w_img, h_img) // 400))
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)
        # Label above box (large font for visibility at full res; scales with image size)
        font_scale = max(1.0, min(2.5, 1200 / max(w_img, h_img)))
        font_thickness = max(2, max(w_img, h_img) // 600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        pad = max(4, font_thickness)
        ty = max(y1 - pad, th + pad)
        cv2.rectangle(out, (x1, ty - th - pad), (x1 + tw + pad * 2, ty + pad), bgr, -1)
        cv2.putText(out, label, (x1 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return out


def resize_for_display(rgb_array, max_size=720):
    """Resize image so longest side is max_size. Returns new RGB array or original if already small enough."""
    if rgb_array is None or rgb_array.size == 0 or not max_size:
        return rgb_array
    h, w = rgb_array.shape[:2]
    if max(h, w) <= max_size:
        return rgb_array
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(rgb_array, (new_w, new_h), interpolation=cv2.INTER_AREA)


def render(fig, segment, segment_idx, total_segments, video_path, current, total):
    fig.clf()

    segment_id = segment.get("segment_id", f"Segment_{segment_idx + 1}")
    start_time = segment.get("start_time", 0)
    end_time = segment.get("end_time", 0)
    narrations = segment.get("narrations", [])
    objects_in_sequence = segment.get("objects_in_sequence", [])

    # Active objects list (unique display names)
    active_object_names = []
    for obj in objects_in_sequence:
        name = f"{obj.get('class_name', '')} ({obj.get('name', '')})".strip(" ()") or obj.get("name", "?")
        if name and name not in active_object_names:
            active_object_names.append(name)

    frame_to_specs = build_frame_bboxes(segment)
    frame_ids = sorted(frame_to_specs.keys())
    if not frame_ids:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No frames with bboxes in this segment", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(f"{segment_id}   [{current + 1} / {total}]   [← prev]  [→ next]  [q quit]", fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()
        return

    # Option A: extract full-res, draw bboxes, then resize for display
    frames_full = extract_frames(video_path, frame_ids, max_size=None)
    frames_drawn = {}
    for fid in frame_ids:
        rgb = frames_full.get(fid)
        specs = frame_to_specs.get(fid, [])
        drawn = draw_bboxes(rgb, specs) if rgb is not None else None
        frames_drawn[fid] = resize_for_display(drawn, max_size=720) if drawn is not None else None

    n_frames = len(frame_ids)
    n_cols = min(4, n_frames)
    n_rows = math.ceil(n_frames / n_cols) if n_frames else 0

    # Layout: header (narrations + active objects), then grid of frames
    n_header_rows = 2
    height_ratios = [0.35, 0.25] + [1.0] * n_rows
    gs = gridspec.GridSpec(n_header_rows + n_rows, n_cols, height_ratios=height_ratios, hspace=0.28, wspace=0.15)

    row = 0
    # --- Header: segment id, time range, narrations ---
    ax_header = fig.add_subplot(gs[row, :])
    nar_lines = []
    for n in narrations:
        parts = n.split(";")
        verb_noun = parts[0] if parts else n
        times = f" ({parts[1]}–{parts[2]}s)" if len(parts) >= 3 else ""
        nar_lines.append(f"• {verb_noun}{times}")
    narrations_text = "\n".join(textwrap.fill(line, width=70) for line in nar_lines)
    info = (
        f"Segment: {segment_id}   Time: {start_time:.2f}s – {end_time:.2f}s\n"
        f"Action narrations:\n{narrations_text}"
    )
    ax_header.text(
        0.02, 0.5, info,
        transform=ax_header.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="center",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd", edgecolor="#1976d2"),
    )
    ax_header.set_axis_off()
    row += 1

    # --- Active objects list ---
    ax_objs = fig.add_subplot(gs[row, :])
    objs_text = "Active objects: " + ", ".join(active_object_names) if active_object_names else "Active objects: (none)"
    ax_objs.text(
        0.02, 0.5, textwrap.fill(objs_text, width=100),
        transform=ax_objs.transAxes,
        fontsize=10, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#388e3c"),
    )
    ax_objs.set_axis_off()
    row += 1

    # --- Frames with bboxes ---
    for i, fid in enumerate(frame_ids):
        r, c = row + i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[r, c])
        img = frames_drawn.get(fid)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Frame {fid}\nunavailable", ha="center", va="center", fontsize=9, color="red", transform=ax.transAxes)
        ax.set_title(f"Frame {fid}", fontsize=9, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle(
        f"Active Segment {current + 1} / {total}   [← prev]  [→ next]  [q quit]",
        fontsize=12, fontweight="bold",
    )
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize active segments: frames with bboxes, narrations, and active objects list.",
    )
    parser.add_argument("video_path", help="Path to the video file (e.g. data/videos/P01_01.MP4)")
    parser.add_argument(
        "--active-objects-dir",
        default="active_objects",
        help="Directory containing active_objects_<video_id>.json",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Video ID (e.g. P01_01). Default: derived from video_path basename without extension.",
    )
    args = parser.parse_args()

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    video_id = args.video_id or video_basename
    json_path = os.path.join(args.active_objects_dir, f"active_objects_{video_id}.json")

    if not os.path.exists(json_path):
        print(f"Active objects file not found: {json_path}")
        return

    with open(json_path) as f:
        segments = json.load(f)

    if not segments:
        print("No segments in file.")
        return

    current = [0]
    fig = plt.figure(figsize=(16, 10))

    def show(idx):
        idx = max(0, min(idx, len(segments) - 1))
        current[0] = idx
        render(fig, segments[idx], idx, len(segments), args.video_path, current[0], len(segments))

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
