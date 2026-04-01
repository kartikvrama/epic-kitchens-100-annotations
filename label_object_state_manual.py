"""
Label last-action narrations as "tidy" or "not tidy".
Loads samples for a single video from object_last_action_{video_id}.json (e.g. under
object_last_action_combined/), shows each narration, and saves labels to
labels_tidy_{video_id}.json (by default beside those files). Use --labels-path to override.
Expects combined format (last_active_segment + narrations).
Shows matplotlib: last 3 actions, frames before/during/after the last action.
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

OBJECT_LAST_ACTION_DIR = "object_last_action_combined"
NARRATION_TIME_MATCH_TOL = 0.35
NUM_PAST_ACTIONS = 3
NUM_FUTURE_ACTIONS = 3
NUM_IMAGES_DURING = 3
SEC_BEFORE = 5.0
SEC_AFTER = 5.0


def make_object_key(obj):
    """Stable dict identity for an entry's object (matches combined JSON object fields)."""
    obj = obj or {}
    return {
        "class_id": obj.get("class_id"),
        "subclass_name": (obj.get("subclass_name") or "").strip(),
        "name": (obj.get("name") or "").strip(),
        "category": (obj.get("category") or "").strip(),
    }


def object_key_identity(ok):
    """Hashable tuple for resume keys (ok is a make_object_key dict or loaded JSON object_key)."""
    if not isinstance(ok, dict):
        return None
    return (
        ok.get("class_id"),
        ok.get("subclass_name") or "",
        ok.get("name") or "",
        ok.get("category") or "",
    )


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
        for r in data:
            if object_key_identity(r.get("object_key")) is None:
                continue
            rec = {k: v for k, v in r.items() if k != "source"}
            labeled_records.append(rec)
            oid = object_key_identity(rec["object_key"])
            key = (oid, rec.get("segment_id") or "", rec["noun_class"], rec.get("narration", ""))
            existing[key] = rec["label"]
    return existing, labeled_records


def save_labels(labels_path, labeled_records):
    """Save labeled records to JSON (object_key, segment_id, noun_class, narration, verb, label)."""
    os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labeled_records, f, indent=2)


# def parse_last_action_from_segment(seg):
#     """From last_active_segment, return (narration_text, start_sec, stop_sec) for the last line."""
#     if not seg:
#         return None
#     narrations = seg.get("narrations") or []
#     if not narrations:
#         return None
#     last_line = str(narrations[-1]).strip()
#     parts = last_line.split(";")
#     narration_text = parts[0].strip() if parts else last_line
#     if len(parts) >= 3:
#         try:
#             start_ts = float(parts[1])
#             stop_ts = float(parts[2])
#             return narration_text, start_ts, stop_ts
#         except ValueError:
#             pass
#     start_ts = seg.get("start_time")
#     stop_ts = seg.get("stop_time")
#     if stop_ts is None:
#         stop_ts = seg.get("end_time")
#     try:
#         start_ts = float(start_ts)
#         stop_ts = float(stop_ts)
#     except (TypeError, ValueError):
#         return None
#     return narration_text, start_ts, stop_ts


def lookup_narration_row(video_narrations, narrations, start_ts, stop_ts, tol=NARRATION_TIME_MATCH_TOL):
    """Find the EPIC CSV row for this narration text and time span (seconds)."""
    cand = video_narrations[
        video_narrations["narration"].astype(str).str.strip().isin(narrations) &
        (video_narrations["start_sec"] >= start_ts - tol) &
        (video_narrations["stop_sec"] <= stop_ts + tol)
    ]
    if cand.empty:
        return None
    return cand.to_dict(orient="records")


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


def show_context_plot(past_actions, future_actions, narrations, before_rgb, during_rgbs, after_rgb, noun_name, verbs):
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
        f"Object: {noun_name}  |  verbs: {verbs}",
        "",
        "Last 3 actions before:",
    ]
    if past_actions:
        for i, a in enumerate(past_actions, 1):
            text_lines.append(f"  {i}. {a}")
    else:
        text_lines.append("  (none)")
    text_lines.append("")
    text_lines.append("Last active segment (TO LABEL):")
    for i, n in enumerate(narrations, 1):
        text_lines.append(f"  {i}. {n}")
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
        "--last-action-dir",
        dest="last_action_dir",
        default=OBJECT_LAST_ACTION_DIR,
        help=(
            "Directory with object_last_action_*.json (e.g. object_last_action_combined; "
            f"default: {OBJECT_LAST_ACTION_DIR})"
        ),
    )
    parser.add_argument(
        "--labels-dir",
        default=OBJECT_LAST_ACTION_DIR,
        help=(
            "Directory for per-video labels (default: same as --last-action-dir default). "
            "Ignored if --labels-path is set."
        ),
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help=(
            "Full path to labels JSON for this run. "
            "Default: <labels-dir>/labels_tidy_<video_id>.json"
        ),
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

    # Load full raw entries (combined: last_active_segment; legacy: last_narration + active_segment)
    json_path = os.path.join(args.last_action_dir, f"object_last_action_{args.video_id}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path) as f:
        raw_entries = json.load(f)
    if not isinstance(raw_entries, list):
        raise ValueError(f"{json_path} is not a list")

    narration_df = load_narration_df()
    video_narrations = narration_df[narration_df["video_id"] == args.video_id].copy()
    video_narrations["start_sec"] = video_narrations["start_timestamp"].apply(
        lambda ts: hhmmss_to_seconds(ts) if isinstance(ts, str) else float(ts)
    )
    video_narrations["stop_sec"] = video_narrations["stop_timestamp"].apply(
        lambda ts: hhmmss_to_seconds(ts) if isinstance(ts, str) else float(ts)
    )
    video_narrations = video_narrations.sort_values("start_sec").reset_index(drop=True)

    to_label = []
    seen = set()
    for entry in raw_entries:
        obj = entry.get("object") or {}
        noun_class = obj.get("class_id")
        if noun_class is None:
            continue
        object_key = make_object_key(obj)
        oid = object_key_identity(object_key)

        seg_combined = entry.get("last_active_segment")
        if seg_combined is None:
            raise ValueError(f"Legacy format not supported: {entry}")

        segment_id = seg_combined.get("segment_id", "None")
        narrations = [n.strip().split(";")[0] for n in seg_combined.get("narrations", [])]
        start_ts = seg_combined.get("start_time")
        stop_ts = seg_combined.get("end_time") or seg_combined.get("stop_time")

        if not start_ts or not stop_ts:
            print(f"Warning: skip entry (no start or stop time): {entry}")
            import pdb; pdb.set_trace()
        # seg_for_entry = None

        # parsed = parse_last_action_from_segment(seg_combined)
        # if not parsed:
        #     continue
        # narration, start_ts, stop_ts = parsed
        # narration = narration.strip()
        # if not narration:
        #     continue
        # segment_id = seg_combined.get("segment_id") or ""
        # seg_for_entry = seg_combined
        rows = lookup_narration_row(video_narrations, narrations, start_ts, stop_ts)
        # import pdb; pdb.set_trace()
        narration_nouns = []
        narration_verbs = []
        if rows is not None:
            for r in rows:
                narration_nouns.append(str(r["noun"]).strip())
                narration_verbs.append(str(r["verb"]).strip())

        key = ("-".join([str(v) for v in oid]), segment_id, noun_class, "-".join(narrations), start_ts, stop_ts)
        print(key)
        if key in seen:
            continue
        seen.add(key)
        to_label.append({
            "object_key": object_key,
            "segment_id": segment_id,
            "noun_class": noun_class,
            "narrations": narrations,
            "noun_name": obj.get("name"),
            "narration_nouns": narration_nouns,
            "narration_verbs": narration_verbs,
            "start_ts": float(start_ts),
            "stop_ts": float(stop_ts),
            "active_segment": seg_combined,
        })

    if args.labels_path is not None:
        labels_rel = args.labels_path
    else:
        labels_rel = os.path.join(args.labels_dir, f"labels_tidy_{args.video_id}.json")
    labels_path = os.path.join(script_dir, labels_rel)

    existing, labeled_records = load_existing_labels_and_records(labels_path)

    def record_key(r):
        oid = object_key_identity(r["object_key"])
        return ("-".join([str(v) for v in oid]), r.get("segment_id") or "", r["noun_class"], "-".join(r["narrations"]))

    pending = [r for r in to_label if record_key(r) not in existing]
    already_done = len(to_label) - len(pending)

    print(f"Last-action narration labeling for video {args.video_id}: tidy or idle / in use")
    print(f"Labels file: {labels_path}")
    print("Commands: t = tidy/idle, u = in use, s = skip, q = quit & save")
    print(f"Already labeled: {already_done}. Remaining: {len(pending)}")
    print()

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
                r["narrations"],
                before_rgb,
                during_rgbs,
                after_rgb,
                r["noun_name"],
                r["narration_verbs"],
            )
            segment_id, noun_class = r["segment_id"], r["noun_class"]
            narrations, noun_name, narration_nouns, narration_verbs = r["narrations"], r["noun_name"], r["narration_nouns"], r["narration_verbs"]
            object_key = r["object_key"]
            print(
                f"[{i + 1}/{len(pending)}] noun_class={noun_class} ({noun_name}) | "
                f"object_key={object_key}"
            )
            print("Last segment narrations:")
            for n, nn, nv, i in zip(narrations, narration_nouns, narration_verbs, range(1, len(narrations) + 1)):
                print(f"  {i}. \"{n}\" ({nn} {nv})")
            print()
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
                print("  Invalid. Use t (tidy/idle), u (in use), s (skip), q (quit).")
            if label:
                existing[record_key(r)] = label
                labeled_records.append({
                    "record_key": record_key(r),
                    "object_key": object_key,
                    "segment_id": segment_id,
                    "noun_class": noun_class,
                    "narrations": narrations,
                    "narration_nouns": narration_nouns,
                    "narration_verbs": narration_verbs,
                    "label": label,
                })
                save_labels(labels_path, labeled_records)
    finally:
        cap.release()

    print("All done.")
    save_labels(labels_path, labeled_records)


if __name__ == "__main__":
    main()
