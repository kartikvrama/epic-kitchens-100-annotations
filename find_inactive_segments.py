#!/usr/bin/env python3
"""
Find temporal segments (>= min_duration seconds) where each object is NOT active
in an active_objects JSON file. Takes the JSON file path as a command-line argument.
"""

import argparse
import json
from collections import defaultdict


def inactive_segments(active_intervals, video_start, video_end, min_duration):
    """
    Given active intervals (non-overlapping by design) within [video_start, video_end],
    return list of (start, end) where the object is inactive for at least min_duration seconds.
    """
    if not active_intervals:
        gap_start, gap_end = video_start, video_end
        if gap_end - gap_start >= min_duration:
            return [(gap_start, gap_end)]
        return []

    active_intervals = sorted(active_intervals, key=lambda x: x[0])
    gaps = []
    # Gaps between active intervals (exclude time before first active)
    for i in range(len(active_intervals) - 1):
        gap_start = active_intervals[i][1]
        gap_end = active_intervals[i + 1][0]
        if gap_end - gap_start >= min_duration:
            gaps.append((gap_start, gap_end))
    # Gap after last active
    if video_end - active_intervals[-1][1] >= min_duration:
        gaps.append((active_intervals[-1][1], video_end))

    return gaps


def main():
    parser = argparse.ArgumentParser(
        description="Find segments (>= 12s) where each object is not active in an active_objects JSON file."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to active_objects JSON file (e.g. active_objects/active_objects_P01_01.json)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=12.0,
        help="Minimum duration (seconds) for an inactive segment (default: 12)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write results as JSON",
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        segments = json.load(f)

    if not segments:
        print("No segments in JSON file.")
        return

    video_start = segments[0]["start_time"]
    video_end = segments[-1]["end_time"]

    # For each object, collect (start_time, end_time) of segments where it appears
    object_active_intervals = defaultdict(list)
    for seg in segments:
        start, end = seg["start_time"], seg["end_time"]
        for obj in seg.get("objects_in_sequence", []):
            object_active_intervals[obj].append((start, end))

    # Compute inactive segments >= min_duration per object (active segments are non-overlapping by design)
    result = {}
    for obj in sorted(object_active_intervals.keys()):
        active = object_active_intervals[obj]
        inactive = inactive_segments(active, video_start, video_end, args.min_duration)
        if inactive:
            result[obj] = [{"start_time": s, "end_time": e, "duration_sec": round(e - s, 2)} for s, e in inactive]

    # Print summary
    print(f"Video time range: {video_start:.2f}s - {video_end:.2f}s")
    print(f"Inactive segments (duration >= {args.min_duration}s):\n")
    for obj in sorted(result.keys()):
        segments_list = result[obj]
        print(f"  {obj}: {len(segments_list)} segment(s)")
        for seg in segments_list:
            print(f"    [{seg['start_time']:.2f}, {seg['end_time']:.2f}] ({seg['duration_sec']}s)")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
