#!/usr/bin/env python3
"""
Find temporal segments (>= min_duration seconds) where each object is NOT active
in an active_objects JSON file. Takes the JSON file path as a command-line argument.
"""

import argparse
import json
import csv
import math
from collections import defaultdict
video_info = []

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
BUFFER_TIME = 5 # seconds

def generate_event_history(narrations, fps):
    """Sort narrations and frame ids by timestamp to generate a history of events."""
    event_history = []
    for narration in narrations:
        narration, start_time, end_time = narration.split(";")
        event_history.append(f"narration:{narration}")
        start_time = float(start_time)
        frame_after_start = math.ceil(start_time * fps)
        event_history.append(f"frame_id:{frame_after_start}")
        end_time = float(end_time)
        frame_before_end = math.floor(end_time * fps)
        event_history.append(f"frame_id:{frame_before_end}")
    return event_history


def inactive_segments(active_intervals, video_start, video_end, min_duration, fps):
    """
    Given active intervals (non-overlapping by design) within [video_start, video_end],
    return list of (start, end) where the object is inactive for at least min_duration seconds.
    """
    global video_info
    if not active_intervals:
        gap_start, gap_end = video_start, video_end
        if gap_end - gap_start >= min_duration:
            return [(gap_start, gap_end)]
        return []

    active_intervals = sorted(active_intervals, key=lambda x: x["start_time"])
    gaps = []
    # Gaps between active intervals (exclude time before first active)
    for i in range(len(active_intervals) - 1):
        gap_start = active_intervals[i]["end_time"]
        gap_end = active_intervals[i + 1]["start_time"]
        prev_active_narrations = active_intervals[i]["narrations"]
        event_history = generate_event_history(prev_active_narrations, fps)
        frame_after_gap_start = math.ceil((BUFFER_TIME + gap_start) * fps)
        if gap_end - gap_start >= min_duration:
            gaps.append(
                {
                    "start_time": gap_start,
                    "end_time": gap_end,
                    "duration_sec": round(gap_end - gap_start, 2),
                    "event_history": event_history,
                    "frame_after_gap_start": frame_after_gap_start,
                }
            )
    # Gap after last active
    if video_end - active_intervals[-1]["end_time"] >= min_duration:
        event_history = generate_event_history(active_intervals[-1]["narrations"], fps)
        frame_after_gap_start = math.ceil(active_intervals[-1]["end_time"] * fps)
        gaps.append(
            {
                "start_time": active_intervals[-1]["end_time"],
                "end_time": video_end,
                "duration_sec": round(video_end - active_intervals[-1]["end_time"], 2),
                "event_history": event_history,
                "frame_after_gap_start": frame_after_gap_start,
            }
        )
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
        "--video-path",
        type=str,
        default=None,
        help="Path to the video file (e.g. data/videos/P01_01.mp4)",
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

    with open(VIDEO_INFO_FILE) as f:
        video_info = list(csv.DictReader(f))
    video_info = {v["video_id"]: v for v in video_info}

    video_id = args.json_file.split("/")[-1].split(".")[0].split("active_objects_")[1]
    print(f"Video ID: {video_id}")

    with open(args.json_file) as f:
        segments = json.load(f)

    if not segments:
        print("No segments in JSON file.")
        return

    video_start = segments[0]["start_time"]
    video_end = segments[-1]["end_time"]

    # For each object, collect (start_time, end_time) of segments where it appears
    object_active_segment_mapping = defaultdict(list)
    for idx, seg in enumerate(segments):
        for obj in seg.get("objects_in_sequence", []):
            object_active_segment_mapping[obj].append((idx, seg["segment_id"]))

    # Compute inactive segments >= min_duration per object (active segments are non-overlapping by design)
    result = {}
    for obj in sorted(object_active_segment_mapping.keys()):
        active = [
            segments[idx[0]] for idx in object_active_segment_mapping[obj]
        ]
        inactive = inactive_segments(active, video_start, video_end, args.min_duration, float(video_info[video_id]["fps"]))
        if inactive:
            result[obj] = inactive

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
