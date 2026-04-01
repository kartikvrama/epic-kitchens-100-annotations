"""
Deduplicate active_objects JSON per video by EPIC noun class (class_id).

- Keeps only objects whose class_id exists in EPIC_100_noun_classes_v2.csv (id column).
- If a class_id appears under a single distinct ``name`` across the video, every such
  object entry is retained.
- If a class_id appears under multiple distinct ``name`` values, exactly one object
  is kept: the first in segment order whose ``name`` is the most specific, defined
  as the name with the largest word count; ties break on longer string length, then
  lexicographic order.

Input files: active_objects_<video_id>.json
Output: same filenames under --output-dir (default: active_objects_deduped).

Rejected instances (invalid class_id, non-chosen name for multi-name class, or
extra keepers after the first) are appended to a text log (--rejections-log).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

NOUN_CLASSES_FILE = "EPIC_100_noun_classes_v2.csv"


def load_valid_class_ids(path: str) -> set[int]:
    with open(path) as f:
        return {int(row["id"]) for row in csv.DictReader(f)}


def name_specificity_key(name: str) -> tuple[int, int, str]:
    """Higher is more specific: word count, then character length, then name for ties."""
    s = (name or "").strip()
    n_words = len(s.split()) if s else 0
    return (n_words, len(s), s)


def winning_name(names: set[str]) -> str:
    return max(names, key=name_specificity_key)


def collect_names_per_class_id(segments: list[dict], valid_ids: set[int]) -> dict[int, set[str]]:
    by_cid: dict[int, set[str]] = defaultdict(set)
    for seg in segments:
        for obj in seg.get("objects_in_sequence") or []:
            cid = obj.get("class_id")
            name = obj.get("name")
            if cid is None or name is None:
                continue
            cid = int(cid)
            if cid not in valid_ids:
                continue
            by_cid[cid].add(str(name))
    return by_cid


def filter_segments(
    segments: list[dict],
    valid_ids: set[int],
    names_per_cid: dict[int, set[str]],
) -> tuple[list[dict], list[dict]]:
    multi_name_cids = {cid for cid, names in names_per_cid.items() if len(names) > 1}
    chosen_name: dict[int, str] = {cid: winning_name(names_per_cid[cid]) for cid in multi_name_cids}
    rejections: list[dict] = []

    out_segments: list[dict] = []
    for seg in segments:
        segment_id = seg.get("segment_id", "")
        new_seg = dict(seg)
        objs_in = seg.get("objects_in_sequence") or []
        new_objs: list[dict] = []
        for obj in objs_in:
            cid_raw = obj.get("class_id")
            name_raw = obj.get("name")
            if cid_raw is None:
                rejections.append(
                    {
                        "segment_id": segment_id,
                        "class_id": "",
                        "name": name_raw,
                        "reason": "missing_class_id",
                        "note": "",
                    }
                )
                continue
            cid = int(cid_raw)
            if cid not in valid_ids:
                rejections.append(
                    {
                        "segment_id": segment_id,
                        "class_id": cid,
                        "name": name_raw,
                        "reason": "class_id_not_in_noun_csv",
                        "note": "",
                    }
                )
                continue
            names = names_per_cid.get(cid, set())
            if len(names) <= 1:
                new_objs.append(obj)
                continue
            nm = str(name_raw or "")
            win = chosen_name[cid]
            if nm != win:
                rejections.append(
                    {
                        "segment_id": segment_id,
                        "class_id": cid,
                        "name": name_raw,
                        "reason": "multi_name_not_chosen",
                        "note": f"kept_name={win!r}",
                    }
                )
                continue
            new_objs.append(obj)
        new_seg["objects_in_sequence"] = new_objs
        out_segments.append(new_seg)
    return out_segments, rejections


def process_file(
    in_path: str,
    out_path: str,
    valid_ids: set[int],
    video_id: str,
) -> dict:
    with open(in_path) as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError(f"Expected list of segments in {in_path}")

    names_per_cid = collect_names_per_class_id(segments, valid_ids)
    out, rejections = filter_segments(segments, valid_ids, names_per_cid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    n_in = sum(len(seg.get("objects_in_sequence") or []) for seg in segments)
    n_out = sum(len(seg.get("objects_in_sequence") or []) for seg in out)
    multi = {
        cid: sorted(names) for cid, names in names_per_cid.items() if len(names) > 1
    }
    return {
        "video_id": video_id,
        "path": in_path,
        "objects_in": n_in,
        "objects_out": n_out,
        "multi_name_class_ids": multi,
        "rejections": rejections,
    }


def write_rejections_log(path: str, summaries: list[dict]) -> None:
    lines = [
        "# active_objects dedupe: rejected object instances",
        "# Fields: video_id | segment_id | class_id | name | reason | note",
        "",
    ]
    for s in summaries:
        vid = s["video_id"]
        rej = s["rejections"]
        lines.append("=" * 80)
        lines.append(f"video_id: {vid}")
        lines.append(f"rejected_count: {len(rej)}")
        lines.append("=" * 80)
        for r in rej:
            cid = r["class_id"]
            cid_s = "" if cid == "" else str(cid)
            name = r["name"]
            name_s = "" if name is None else str(name).replace("\n", " ")
            note = r.get("note") or ""
            lines.append(
                f"{vid}\t{r.get('segment_id', '')}\t{cid_s}\t{name_s}\t{r['reason']}\t{note}"
            )
        lines.append("")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="active_objects",
        help="Directory with active_objects_<video_id>.json",
    )
    parser.add_argument(
        "--output-dir",
        default="active_objects_deduped",
        help="Directory for filtered JSON (created if missing)",
    )
    parser.add_argument(
        "--noun-classes",
        default=NOUN_CLASSES_FILE,
        help="EPIC noun classes CSV (id column = class_id)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file summary and multi-name class_ids",
    )
    parser.add_argument(
        "--rejections-log",
        default="active_objects_dedupe_rejections.txt",
        help="Text file listing every rejected instance, grouped by video_id",
    )
    args = parser.parse_args()

    valid_ids = load_valid_class_ids(args.noun_classes)
    os.makedirs(args.output_dir, exist_ok=True)

    fnames = sorted(
        f
        for f in os.listdir(args.input_dir)
        if f.startswith("active_objects_") and f.endswith(".json")
    )
    if not fnames:
        raise SystemExit(f"No active_objects_*.json in {args.input_dir!r}")

    summaries = []
    for fname in fnames:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        video_id = fname.replace("active_objects_", "").replace(".json", "")
        summaries.append(process_file(in_path, out_path, valid_ids, video_id))

    write_rejections_log(args.rejections_log, summaries)

    total_in = sum(s["objects_in"] for s in summaries)
    total_out = sum(s["objects_out"] for s in summaries)
    total_rej = sum(len(s["rejections"]) for s in summaries)
    print(f"Processed {len(summaries)} files -> {args.output_dir!r}")
    print(f"Total object entries: {total_in} -> {total_out} (dropped {total_in - total_out})")
    print(f"Rejections log: {args.rejections_log!r} ({total_rej} rows)")

    if args.verbose:
        for s in summaries:
            print(f"\n{s['path']}: {s['objects_in']} -> {s['objects_out']}")
            if s["multi_name_class_ids"]:
                print(f"  multi-name class_ids: {s['multi_name_class_ids']}")


if __name__ == "__main__":
    main()
