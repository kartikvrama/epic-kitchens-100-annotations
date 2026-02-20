#!/usr/bin/env python3
"""
Find temporal segments (>= min_duration seconds) where each object is NOT active
in an active_objects JSON file. Takes the JSON file path as a command-line argument.
"""

import os
import json
import csv
import math
from collections import defaultdict
import argparse

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
BUFFER_TIME = 1 # seconds

video_info = []


def sample_frames(frame_after_start, frame_before_end):
    ## Sample frames at 25 % after start and 25 % behind end
    duration = frame_before_end - frame_after_start
    if duration > 50:
        sampled_frames = [frame_after_start + math.floor(0.25 * duration), frame_before_end - math.floor(0.25 * duration)]
    else: ## choose middle frame
        sampled_frames = [math.floor((frame_after_start + frame_before_end) / 2)]
    return sampled_frames


def generate_event_history(narrations, fps):
    """Sort narrations and frame ids by timestamp to generate a history of events."""
    event_history = []
    for narration in narrations:
        narration, start_time, end_time = narration.split(";")
        event_history.append(f"narration:{narration}")
        start_time = float(start_time)
        frame_after_start = math.ceil(start_time * fps)
        end_time = float(end_time)
        frame_before_end = math.floor(end_time * fps)
        sampled_frames = sample_frames(frame_after_start, frame_before_end)
        for frame in sampled_frames:
            event_history.append(f"frame_id:{frame}")
    return event_history


def inactive_segments(active_intervals, object_crop_data, video_start, video_end, min_duration, fps):
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
        object_crop_data_start = object_crop_data[i]["image_crops"]
        gap_end = active_intervals[i + 1]["start_time"]
        prev_active_narrations = active_intervals[i]["narrations"]
        event_history = generate_event_history(prev_active_narrations, fps)
        ## TODO: Is it ok to choose a frame after the gap start like this?
        frame_after_gap_start = math.ceil((BUFFER_TIME + gap_start) * fps)
        if gap_end - gap_start >= min_duration:
            gaps.append(
                {
                    "start_time": gap_start,
                    "end_time": gap_end,
                    "duration_sec": round(gap_end - gap_start, 2),
                    "event_history": event_history,
                    "frame_after_gap_start": frame_after_gap_start,
                    "crop_from_previous_active": object_crop_data_start,
                }
            )
    # Gap after last active
    if video_end - active_intervals[-1]["end_time"] >= min_duration:
        event_history = generate_event_history(active_intervals[-1]["narrations"], fps)
        frame_after_gap_start = math.ceil(active_intervals[-1]["end_time"] * fps)
        object_crop_data_end = object_crop_data[-1]["image_crops"]
        gaps.append(
            {
                "start_time": active_intervals[-1]["end_time"],
                "end_time": video_end,
                "duration_sec": round(video_end - active_intervals[-1]["end_time"], 2),
                "event_history": event_history,
                "frame_after_gap_start": frame_after_gap_start,
                "crop_from_previous_active": object_crop_data_end,
            }
        )
    return gaps


def main():
    parser = argparse.ArgumentParser(
        description="Find segments (>= 12s) where each object is not active in an active_objects JSON file."
    )
    parser.add_argument(
        "active_object_folder",
        type=str,
        help="Path to active_objects folder (e.g. active_objects)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0,
        help="Minimum duration (seconds) for an inactive segment (default: 12)",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="inactive_segments",
        help="Optional path to write results as JSON",
    )
    args = parser.parse_args()

    with open(VIDEO_INFO_FILE) as f:
        video_info = list(csv.DictReader(f))
    video_info = {v["video_id"]: v for v in video_info}

    for active_objects_file in os.listdir(args.active_object_folder):
        if not active_objects_file.endswith(".json"):
            continue
        video_id = active_objects_file.split("active_objects_")[1].split(".")[0]
        print(f"Video ID: {video_id}")
        print(f"Active objects file: {active_objects_file}")
        with open(os.path.join(args.active_object_folder, active_objects_file)) as f:
            segments = json.load(f)
        video_start = segments[0]["start_time"]
        video_end = segments[-1]["end_time"]

        # For each object, collect (start_time, end_time) of segments where it appears
        object_active_segment_mapping = defaultdict(list)
        for idx, seg in enumerate(segments):
            for obj in seg.get("objects_in_sequence", []):
                key = f"{obj['class_id']}/{obj['class_name']}/{obj['name']}"
                object_active_segment_mapping[key].append(idx)

        # Compute inactive segments >= min_duration per object (active segments are non-overlapping by design)
        result = {}
        for obj_key in sorted(object_active_segment_mapping.keys()):
            active = [
                segments[idx] for idx in object_active_segment_mapping[obj_key]
            ]
            # One crop per active segment: the object's crop in that segment
            object_crop_data = [
                next(
                    (elem for elem in seg.get("objects_in_sequence", [])
                     if obj_key == f"{elem['class_id']}/{elem['class_name']}/{elem['name']}"),
                    None,
                )
                for seg in active
            ]
            inactive = inactive_segments(active, object_crop_data, video_start, video_end, args.min_duration, float(video_info[video_id]["fps"]))

            if inactive:
                result[obj_key] = inactive

        # Print summary
        print(f"Video time range: {video_start:.2f}s - {video_end:.2f}s")
        print(f"Inactive segments (duration >= {args.min_duration}s):\n")
        for obj_key in sorted(result.keys()):
            segments_list = result[obj_key]
            print(f"  {obj_key}: {len(segments_list)} segment(s)")
            for seg in segments_list:
                print(f"    [{seg['start_time']:.2f}, {seg['end_time']:.2f}] ({seg['duration_sec']}s)")
            print()

        os.makedirs(args.output_folder, exist_ok=True)
        with open(os.path.join(args.output_folder, f"inactive_segments_{video_id}.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote results to {os.path.join(args.output_folder, f'inactive_segments_{video_id}.json')}")


if __name__ == "__main__":
    main()
