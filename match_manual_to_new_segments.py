#!/usr/bin/env python3
"""
Match manual annotations (from manual_labels/inactive_segments_P37_101.jsonl) to
the new segment set (inactive_segments/inactive_segments_P37_101.json) by finding
the manual annotation whose start_time and end_time are closest to each new segment.

Distance = |new_start - manual_start| + |new_end - manual_end| (L1).
Only annotations with the same query_object are considered.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

MIN_DURATION = 6 # half the length of average active segment
from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM


def load_manual_annotations(path: str) -> dict[str, list[dict]]:
    """Load JSONL manual labels, grouped by query_object."""
    by_query = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            q = ann["query_object"]
            by_query[q].append(
                {
                    "start_time": ann["start_time"],
                    "end_time": ann["end_time"],
                    "used": ann["used"],
                }
            )
    return dict(by_query)


def load_new_segments(path: str) -> dict[str, list[dict]]:
    """Load new segments JSON: query_object -> list of segment dicts with start_time, end_time."""
    with open(path) as f:
        data = json.load(f)
    return data


def distance(new_start: float, new_end: float, manual_start: float, manual_end: float) -> float:
    """L1 distance between (new_start, new_end) and (manual_start, manual_end)."""
    return abs(new_start - manual_start) + abs(new_end - manual_end)


def find_closest_manual(
    new_start: float,
    new_end: float,
    manual_list: list[dict],
):
    """Among manual annotations with same query_object, return closest and its distance."""
    if not manual_list:
        return None, float("inf")
    best = None
    best_dist = float("inf")
    for m in manual_list:
        d = distance(new_start, new_end, m["start_time"], m["end_time"])
        if d < best_dist:
            best_dist = d
            best = m
    return best, best_dist


def run(
    manual_path: str,
    new_segments_path: str,
    out_mapping_path: str = None,
    out_annotated_json_path: str = None,
    max_distance: float = None,
):
    manual_by_query = load_manual_annotations(manual_path)
    new_by_query = load_new_segments(new_segments_path)

    # For each new segment, find closest manual and record
    mapping = []
    annotated_segments = {}

    for query_object, segments in new_by_query.items():
        manual_list = manual_by_query.get(query_object, [])
        annotated_segments[query_object] = []

        print(f"Processing query_object: {query_object} ... {query_object.split('/', maxsplit=2)}")
        _, subclass_name, _ = query_object.split("/", maxsplit=2)

        for i, seg in enumerate(segments):

            if subclass_name in OBJECTS_TO_EXCLUDE_FROM_VLM:
                continue

            new_start = seg["start_time"]
            new_end = seg["end_time"]

            if new_end - new_start < MIN_DURATION:
                continue

            closest, dist = find_closest_manual(new_start, new_end, manual_list)

            used = None
            manual_start = manual_start_end = None
            if closest is not None and (max_distance is None or dist <= max_distance):
                used = closest["used"]
                manual_start = closest["start_time"]
                manual_start_end = closest["end_time"]

            mapping.append({
                "query_object": query_object,
                "segment_index": i,
                "start_time": new_start,
                "end_time": new_end,
                "previous_start_time": manual_start,
                "previous_end_time": manual_start_end,
                "used": used,
                "distance": dist if closest else None,
            })

            # Build annotated segment: copy full segment and add "used" if we had a match
            seg_copy = dict(seg)
            if used is not None:
                seg_copy["used"] = used
            annotated_segments[query_object].append(seg_copy)

    # Write mapping JSONL
    if out_mapping_path:
        Path(out_mapping_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_mapping_path, "w") as f:
            for row in mapping:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote mapping to {out_mapping_path} ({len(mapping)} rows)")

    # Write new segments JSON with "used" from closest manual
    if out_annotated_json_path:
        Path(out_annotated_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_annotated_json_path, "w") as f:
            json.dump(annotated_segments, f, indent=2)
        print(f"Wrote annotated segments to {out_annotated_json_path}")

    # Summary
    with_used = sum(1 for r in mapping if r["used"] is not None)
    without_match = sum(1 for r in mapping if r["manual_start_time"] is None)
    print(f"Segments with a matched manual 'used' value: {with_used} / {len(mapping)}")
    print(f"Segments with no manual annotation for same query_object: {without_match}")

    return mapping, annotated_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual",
        default="manual_labels/inactive_segments_P37_101.jsonl",
        help="Path to manual labels JSONL",
    )
    parser.add_argument(
        "--new",
        default="inactive_segments/inactive_segments_P37_101.json",
        help="Path to new segments JSON",
    )
    parser.add_argument(
        "--out-mapping",
        default="results/manual_to_new_segments_mapping_P37_101.jsonl",
        help="Output path for mapping JSONL",
    )
    parser.add_argument(
        "--out-annotated",
        default=None,
        help="Output path for new segments JSON with 'used' from closest manual",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="If set, only assign 'used' when L1 distance <= this (seconds)",
    )
    args = parser.parse_args()
    run(
        manual_path=args.manual,
        new_segments_path=args.new,
        out_mapping_path=args.out_mapping,
        out_annotated_json_path=args.out_annotated,
        max_distance=args.max_distance,
    )


if __name__ == "__main__":
    main()
