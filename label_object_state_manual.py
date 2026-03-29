"""
Label last-action narrations as "tidy" or "not tidy".
Loads samples for a single video, shows each narration, and saves labels to a JSON file.
Shows matplotlib: last 3 actions, 1 image 1s before, 3 during, 1 image 1s after the last action.
Resumes from existing labels if present.
"""
import argparse
import csv
import json
import os
import sys
from math import ceil, floor

import cv2
import matplotlib.pyplot as plt

from generate_object_last_action import load_narration_df
from utils import VIDEO_INFO_FILE
from utils import hhmmss_to_seconds

OBJECT_LAST_ACTION_DIR = "object_last_action"
LABELS_PATH = "object_last_action/labels_tidy.json"
NUM_PAST_ACTIONS = 3
NUM_FUTURE_ACTIONS = 3
NUM_IMAGES_DURING = 3
SEC_BEFORE = 5.0
SEC_AFTER = 5.0


def load_existing_labels_and_records(labels_path):
    """Load existing labels file. Returns (existing_dict, labeled_records_list)."""
    existing = {}
    labeled_records = []
    if not os.path.isfile(labels_path):
        return existing, labeled_records
    try:
        with open(labels_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return existing, labeled_records
    if isinstance(data, list):
        labeled_records = data
        existing = {(r["source"], r["segment_id"], r["noun_class"]): r["label"] for r in data}
    return existing, labeled_records


def save_labels(labels_path, labeled_records):
    """Save labeled records to JSON (list of dicts with source, segment_id, noun_class, narration, label)."""
    os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labeled_records, f, indent=2)


def load_last_action_samples_for_video(data_dir, video_id):
    """
    Load object_last_action JSON for a given video_id; return list of
    (source_basename, segment_id, noun_class, narration, noun_name, verb).
    The video_id is assumed to be part of the file name.
    """
    entries = []
    json_file_name = f"object_last_action_{video_id}.json"
    if not os.path.isfile(os.path.join(data_dir, json_file_name)):
        raise FileNotFoundError(f"File {json_file_name} not found in {data_dir}")
    with open(os.path.join(data_dir, json_file_name)) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"File {json_file_name} is not a list")
    for entry in data:
        obj = entry.get("object") or {}
        noun_class = obj.get("class_id")
        noun_name = (obj.get("name") or "").strip()
        last = entry.get("last_narration") or {}
        verb = (last.get("verb") or "").strip()
        if not verb:
            continue
        narration = (last.get("narration") or "").strip()
        seg = entry.get("active_segment") or {}
        segment_id = seg.get("segment_id") or ""
        if narration and segment_id and noun_class is not None:
            entries.append((json_file_name, segment_id, noun_class, narration, noun_name, verb))
    return entries


def load_video_fps(video_info_csv_path, video_id):
    """Load FPS for the given video_id from EPIC video info CSV."""
    if not video_info_csv_path or not os.path.isfile(video_info_csv_path):
        return None
    with open(video_info_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_id") == video_id:
                try:
                    return float(row["fps"])
                except (ValueError, KeyError):
                    return None
    return None


def get_past_actions(video_narrations_sorted, before_timestamp, num_actions=NUM_PAST_ACTIONS):
    """Return the num_actions narrations immediately preceding before_timestamp (seconds)."""
    preceding = video_narrations_sorted[video_narrations_sorted["stop_sec"] <= before_timestamp]
    tail = preceding.tail(num_actions)
    return tail["narration"].tolist()


def get_future_actions(video_narrations_sorted, after_timestamp, num_actions=NUM_FUTURE_ACTIONS):
    """Return the num_actions narrations immediately following after_timestamp (seconds)."""
    following = video_narrations_sorted[video_narrations_sorted["start_sec"] >= after_timestamp]
    head = following.head(num_actions)
    return head["narration"].tolist()


def extract_frames(video_cap, frame_ids):
    """Extract frames at given indices; return dict frame_id -> RGB array (or None if read failed)."""
    result = {}
    for fid in sorted(set(frame_ids)):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_cap.read()
        result[fid] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    return result


def get_context_frame_ids(start_ts, stop_ts, fps, active_segment=None):
    """
    Return (before_frame_id, during_frame_ids, after_frame_id).
    Before = 1 sec before start; after = 1 sec after stop; during = 3 uniformly sampled.
    """
    start_frame = ceil(start_ts * fps)
    end_frame = floor(stop_ts * fps)

    before_frame_id = max(0, ceil((start_ts - SEC_BEFORE) * fps))
    after_frame_id = floor((stop_ts + SEC_AFTER) * fps)

    total_during = end_frame - start_frame
    if total_during <= 0:
        during_ids = [start_frame]
    elif total_during <= NUM_IMAGES_DURING:
        step = max(1, total_during // (NUM_IMAGES_DURING + 1))
        during_ids = list(range(start_frame + step, end_frame, step))[:NUM_IMAGES_DURING]
        if not during_ids:
            during_ids = [(start_frame + end_frame) // 2]
    else:
        step = total_during // (NUM_IMAGES_DURING + 1)
        during_ids = [start_frame + step * (i + 1) for i in range(NUM_IMAGES_DURING)]

    return before_frame_id, during_ids, after_frame_id


def show_context_plot(past_actions, future_actions, narration, before_rgb, during_rgbs, after_rgb, noun_name, verb):
    """Show matplotlib figure: past 3 actions text, then row of 5 images (before, 3 during, after)."""
    ncols = 5  # before, during1, during2, during3, after
    # Pad to exactly 3 during-images
    during = list(during_rgbs) + [None] * (NUM_IMAGES_DURING - len(during_rgbs))
    during = during[:NUM_IMAGES_DURING]
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 5))
    titles = [f"{SEC_BEFORE} sec before", "During 1", "During 2", "During 3", f"{SEC_AFTER} sec after"]
    imgs = [before_rgb] + during + [after_rgb]
    for ax, title, img in zip(axes[0], titles, imgs):
        ax.axis("off")
        ax.set_title(title)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
    text_lines = [
        f"Object: {noun_name}  |  verb: {verb}",
        "",
        "Last 3 actions before:",
    ]
    if past_actions:
        for i, a in enumerate(past_actions, 1):
            text_lines.append(f"  {i}. {a}")
    else:
        text_lines.append("  (none)")
    text_lines.append("")
    text_lines.append(f"Last action (TO LABEL): \"{narration}\"")
    text_lines.append("")
    text_lines.append("Future actions:")
    if future_actions:
        for i, a in enumerate(future_actions, 1):
            text_lines.append(f"  {i}. {a}")
    else:
        text_lines.append("  (none)")
    text_block = "\n".join(text_lines)
    for ax in axes[1]:
        ax.axis("off")
    axes[1, 0].text(0, 1, text_block, va="top", ha="left", fontsize=10, family="sans-serif")
    fig.suptitle("Context for labeling: tidy or idle / in use", fontsize=12)
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively label tidy or idle / in use for one video's last-action narrations."
    )
    parser.add_argument(
        "video_id",
        help="Video id suffix used in file name, e.g. P01_01 for object_last_action_P01_01.json",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the video file (e.g. P01_01.MP4) for extracting before/during/after frames.",
    )
    parser.add_argument(
        "--data-dir",
        default=OBJECT_LAST_ACTION_DIR,
        help=f"Directory containing object_last_action_*.json files (default: {OBJECT_LAST_ACTION_DIR})",
    )
    parser.add_argument(
        "--labels-path",
        default=LABELS_PATH,
        help=f"Path to labels JSON output (default: {LABELS_PATH})",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
        raise SystemExit(1)

    fps = load_video_fps(VIDEO_INFO_FILE, args.video_id)
    if fps is None:
        print(f"Error: FPS not found for {args.video_id} in {VIDEO_INFO_FILE}", file=sys.stderr)
        raise SystemExit(1)

    # Load full raw entries to get timestamps and active_segment
    json_path = os.path.join(args.data_dir, f"object_last_action_{args.video_id}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path) as f:
        raw_entries = json.load(f)
    if not isinstance(raw_entries, list):
        raise ValueError(f"{json_path} is not a list")

    to_label = []
    seen = set()
    json_file_name = os.path.basename(json_path)
    for entry in raw_entries:
        obj = entry.get("object") or {}
        last = entry.get("last_narration") or {}
        seg = entry.get("active_segment") or {}
        segment_id = seg.get("segment_id") or ""
        noun_class = obj.get("class_id")
        narration = (last.get("narration") or "").strip()
        verb = (last.get("verb") or "").strip()
        if not verb or not narration or not segment_id or noun_class is None:
            continue
        key = (segment_id, noun_class, narration)
        if key in seen:
            continue
        seen.add(key)
        start_ts = last.get("start_timestamp")
        stop_ts = last.get("stop_timestamp")
        if start_ts is None or stop_ts is None:
            continue
        to_label.append({
            "source": json_file_name,
            "segment_id": segment_id,
            "noun_class": noun_class,
            "narration": narration,
            "noun_name": (obj.get("name") or "").strip(),
            "verb": verb,
            "start_ts": float(start_ts),
            "stop_ts": float(stop_ts),
            "active_segment": seg,
        })

    labels_path = os.path.join(script_dir, args.labels_path)
    existing, labeled_records = load_existing_labels_and_records(labels_path)

    def record_key(r):
        return (r["source"], r["segment_id"], r["noun_class"])

    pending = [r for r in to_label if record_key(r) not in existing]
    already_done = len(to_label) - len(pending)

    print(f"Last-action narration labeling for video {args.video_id}: tidy or idle / in use")
    print("Commands: t = tidy/idle, u = in use, s = skip, q = quit & save")
    print(f"Already labeled: {already_done}. Remaining: {len(pending)}")
    print()

    # Load narrations for past-actions context
    narration_df = load_narration_df()
    video_narrations = narration_df[narration_df["video_id"] == args.video_id].copy()
    video_narrations["start_sec"] = video_narrations["start_timestamp"].apply(
        lambda ts: hhmmss_to_seconds(ts) if isinstance(ts, str) else float(ts)
    )
    video_narrations["stop_sec"] = video_narrations["stop_timestamp"].apply(
        lambda ts: hhmmss_to_seconds(ts) if isinstance(ts, str) else float(ts)
    )
    video_narrations = video_narrations.sort_values("start_sec").reset_index(drop=True)

    cap = cv2.VideoCapture(args.video_path)
    try:
        for i, r in enumerate(pending):
            past_actions = get_past_actions(video_narrations, r["start_ts"], NUM_PAST_ACTIONS)
            future_actions = get_future_actions(video_narrations, r["stop_ts"], NUM_FUTURE_ACTIONS)
            before_id, during_ids, after_id = get_context_frame_ids(
                r["start_ts"], r["stop_ts"], fps, r.get("active_segment")
            )
            all_frame_ids = [before_id] + during_ids + [after_id]
            frames = extract_frames(cap, all_frame_ids)
            before_rgb = frames.get(before_id)
            during_rgbs = [frames.get(fid) for fid in during_ids]
            after_rgb = frames.get(after_id)
            show_context_plot(
                past_actions,
                future_actions,
                r["narration"],
                before_rgb,
                during_rgbs,
                after_rgb,
                r["noun_name"],
                r["verb"],
            )
            source, segment_id, noun_class = r["source"], r["segment_id"], r["noun_class"]
            narration, noun_name, verb = r["narration"], r["noun_name"], r["verb"]
            print(f"[{i + 1}/{len(pending)}] verb={verb} noun_class={noun_class} ({noun_name})")
            print(f"  \"{narration}\"")
            while True:
                reply = input("  Label (t/u/s/q): ").strip().lower()
                if reply in ("t"):
                    label = "tidy/idle"
                    break
                if reply in ("u"):
                    label = "in use"
                    break
                if reply in ("s", "skip"):
                    label = None
                    break
                if reply in ("q", "quit"):
                    print("Quitting and saving...")
                    save_labels(labels_path, labeled_records)
                    return
                print("  Invalid. Use t (tidy), n (not tidy), s (skip), q (quit).")
            if label:
                existing[record_key(r)] = label
                labeled_records.append({
                    "source": source,
                    "segment_id": segment_id,
                    "noun_class": noun_class,
                    "narration": narration,
                    "noun_name": noun_name,
                    "verb": verb,
                    "label": label,
                })
                save_labels(labels_path, labeled_records)
    finally:
        cap.release()

    print("All done.")
    save_labels(labels_path, labeled_records)


if __name__ == "__main__":
    main()
