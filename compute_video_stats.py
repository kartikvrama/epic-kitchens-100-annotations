#!/usr/bin/env python3
"""
Per-video statistics: unique objects (VISOR), active/passive segment stats,
active time per object, objects per sequence. Single-file, extensible via STATS below.

Usage:
  python compute_video_stats.py [--active-objects-dir active_objects] [--output video_stats]
"""

import argparse
import json
import os
import statistics
from pathlib import Path


# --- Paths (override via CLI) ---
DEFAULTS = {
    "visor_annotations_dir": "visor_annotations",
    "active_objects_dir": "active_objects",
    "inactive_segments_dir": "inactive_segments",
}

HAND_CLASS_IDS = {300, 301}  # exclude left/right hand from unique object count


def _load_json(path):
    if path is None or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _visor_path(video_id, cfg):
    for split in ("train", "val"):
        p = Path(cfg["visor_annotations_dir"]) / split / f"{video_id}.json"
        if p.is_file():
            return str(p)
    return None


# --- Stat compute functions: each returns a number or dict with mean/std, or None if missing data ---

def num_unique_objects(video_id, cfg):
    path = _visor_path(video_id, cfg)
    data = _load_json(path)
    if not data:
        return None
    seen = set()
    for frame in data.get("video_annotations", []):
        for ann in frame.get("annotations", []):
            cid = ann.get("class_id")
            if cid in HAND_CLASS_IDS:
                continue
            seen.add((cid, ann.get("name", "")))
    return len(seen)


def num_active_segments(video_id, cfg):
    path = Path(cfg["active_objects_dir"]) / f"active_objects_{video_id}.json"
    data = _load_json(str(path))
    if not data:
        return None
    return len(data)


def passive_segment_length_per_object(video_id, cfg):
    path = Path(cfg["inactive_segments_dir"]) / f"inactive_segments_{video_id}.json"
    data = _load_json(str(path))
    if not data or not isinstance(data, dict):
        return None
    per_object_means = []
    for segments in data.values():
        if not segments:
            continue
        lengths = [s["duration_sec"] for s in segments if "duration_sec" in s]
        if lengths:
            per_object_means.append(statistics.mean(lengths))
    if not per_object_means:
        return None
    return {"mean": statistics.mean(per_object_means), "std": statistics.stdev(per_object_means) if len(per_object_means) > 1 else 0.0}


def active_time_per_object(video_id, cfg):
    path = Path(cfg["active_objects_dir"]) / f"active_objects_{video_id}.json"
    data = _load_json(str(path))
    if not data:
        return None
    total_by_obj = {}
    for seg in data:
        duration = seg["end_time"] - seg["start_time"]
        for obj in seg.get("objects_in_sequence", []):
            key = (obj["class_id"], obj.get("class_name", ""), obj.get("name", ""))
            total_by_obj[key] = total_by_obj.get(key, 0) + duration
    if not total_by_obj:
        return None
    totals = list(total_by_obj.values())
    return {"mean": statistics.mean(totals), "std": statistics.stdev(totals) if len(totals) > 1 else 0.0}


def objects_per_sequence(video_id, cfg):
    path = Path(cfg["active_objects_dir"]) / f"active_objects_{video_id}.json"
    data = _load_json(str(path))
    if not data:
        return None
    counts = [len(seg.get("objects_in_sequence", [])) for seg in data]
    if not counts:
        return None
    return {"mean": statistics.mean(counts), "std": statistics.stdev(counts) if len(counts) > 1 else 0.0}


# --- Registry: add new stats here (key -> compute_fn). Flatten dict stats into columns with _mean / _std. ---
STATS = [
    ("num_unique_objects", num_unique_objects),
    ("num_active_segments", num_active_segments),
    ("passive_segment_length_per_object", passive_segment_length_per_object),
    ("active_time_per_object", active_time_per_object),
    ("objects_per_sequence", objects_per_sequence),
]


def flatten_value(stat_key, v):
    if v is None:
        return {}
    if isinstance(v, dict) and "mean" in v and "std" in v:
        return {f"{stat_key}_mean": v["mean"], f"{stat_key}_std": v["std"]}
    return {stat_key: v}


def discover_video_ids(active_objects_dir):
    ids = []
    p = Path(active_objects_dir)
    if not p.is_dir():
        return ids
    for f in p.glob("active_objects_*.json"):
        vid = f.stem.replace("active_objects_", "")
        if vid:
            ids.append(vid)
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser(description="Compute per-video statistics.")
    ap.add_argument("--visor-annotations-dir", default=DEFAULTS["visor_annotations_dir"], help="VISOR annotations root (train/val under it)")
    ap.add_argument("--active-objects-dir", default=DEFAULTS["active_objects_dir"], help="Directory of active_objects_{video_id}.json")
    ap.add_argument("--inactive-segments-dir", default=DEFAULTS["inactive_segments_dir"], help="Directory of inactive_segments_{video_id}.json")
    ap.add_argument("--output", default="video_stats", help="Output path prefix (writes video_stats.csv and video_stats.json)")
    args = ap.parse_args()
    cfg = {
        "visor_annotations_dir": args.visor_annotations_dir,
        "active_objects_dir": args.active_objects_dir,
        "inactive_segments_dir": args.inactive_segments_dir,
    }
    video_ids = discover_video_ids(cfg["active_objects_dir"])
    if not video_ids:
        print("No active_objects_*.json found; nothing to do.")
        return
    rows = []
    for video_id in video_ids:
        row = {"video_id": video_id}
        for key, fn in STATS:
            try:
                v = fn(video_id, cfg)
            except Exception as e:
                v = None
                print(f"Warning: {video_id} {key}: {e}")
            for col, val in flatten_value(key, v).items():
                row[col] = val
        rows.append(row)
    out = args.output

    with open(f"{out}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out}.json")


if __name__ == "__main__":
    main()
