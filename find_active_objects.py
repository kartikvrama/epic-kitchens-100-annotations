import os
import csv
import json
import pandas as pd
from math import floor, ceil

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
NARRATION_LOW_LEVEL_FILES = ["EPIC_100_train.csv", "EPIC_100_validation.csv", "EPIC_100_test_timestamps.csv"]

def hhmmss_to_seconds(hhmmss):
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def save_active_objects(video_id, video_info, narration_low_level_df, output_dir="./"):
    visor_annotations_file = f"visor_annotations/train/{video_id}.json"
    if not os.path.exists(visor_annotations_file):
        visor_annotations_file = f"visor_annotations/val/{video_id}.json"
        if not os.path.exists(visor_annotations_file):
            print(f"Visor annotations not found for {video_id}")
            return
    with open(visor_annotations_file) as f:
        visor_data = json.load(f)
    visor_annotations = visor_data["video_annotations"]
    ## Add frame_id and objects to each frame
    for frame in visor_annotations:
        frame["frame_id"] = int(frame["image"]["name"].split("_")[-1].split(".")[0])
        frame["objects"] = [k["name"] for k in frame["annotations"]]

    fps = float([v for v in video_info if v["video_id"] == video_id][0]["fps"])
    narrations_low_level_filtered = narration_low_level_df[narration_low_level_df["video_id"] == video_id]
    ## Sort by start_timestamp
    narrations_low_level_filtered = narrations_low_level_filtered.sort_values(by="start_timestamp")

    # Build sequences of 3 non-overlapping actions; collect (sequence_start, sequence_end, objects, frame_ids)
    sequences = []
    sequence_index = 1
    prev_stop = None
    objects_in_sequence = []
    sequence_start_time = None
    sequence_end_time = None
    non_overlapping_count = 1

    os.makedirs(output_dir, exist_ok=True)
    txt_path = f"{output_dir}/active_objects_{video_id}.txt"
    json_path = f"{output_dir}/active_objects_{video_id}.json"

    with open(txt_path, "w") as txt_file:
        verb_noun_pairs = []
        for _, row in narrations_low_level_filtered.iterrows():
            start_timestamp = hhmmss_to_seconds(row["start_timestamp"])
            stop_timestamp = hhmmss_to_seconds(row["stop_timestamp"])

            # Write one block per narration
            txt_file.write(f"Start timestamp: {start_timestamp}, Stop timestamp: {stop_timestamp}\n")
            txt_file.write(f"Narration: {row['narration']}\n")
            txt_file.write("------\n")

            verb_noun_pair = f"{row['verb']};{row['noun']};{start_timestamp};{stop_timestamp}"

            # Count non-overlapping narrations; each sequence = 3 non-overlapping actions
            if prev_stop is None or start_timestamp >= prev_stop:
                if non_overlapping_count == 1:
                    sequence_start_time = start_timestamp
                    verb_noun_pairs = []
                non_overlapping_count += 1
                sequence_end_time = stop_timestamp

                if non_overlapping_count > 3:
                    # Sequence complete: [sequence_start_time, sequence_end_time] with 3 actions
                    txt_file.write("************\n")
                    frames_in_sequence = [
                        frame for frame in visor_annotations
                        if int(frame["image"]["subsequence"].split("_")[-1]) == sequence_index
                    ]
                    objects_in_sequence = list(set(sum([f["objects"] for f in frames_in_sequence], [])))
                    if "left hand" in objects_in_sequence:
                        objects_in_sequence.remove("left hand")
                    if "right hand" in objects_in_sequence:
                        objects_in_sequence.remove("right hand")
                    frame_ids_in_sequence = sorted([f["frame_id"] for f in frames_in_sequence])
                    txt_file.write(f"Frame IDs in sequence: {frame_ids_in_sequence}\n")
                    txt_file.write(f"Objects in sequence: {objects_in_sequence}\n")
                    txt_file.write(f"Verb-noun pairs in sequence: {verb_noun_pairs}\n")
                    txt_file.write("************\n")

                    sequences.append({
                        "segment_id": f"{video_id}_ActiveUsage_{sequence_index:04d}",
                        "start_time": sequence_start_time,
                        "end_time": sequence_end_time,
                        "objects_in_sequence": sorted(objects_in_sequence),
                        "frame_ids": frame_ids_in_sequence,
                        "verb_noun_pairs": verb_noun_pairs,
                    })
                    ## Reset objects_in_sequence and non-overlapping count
                    non_overlapping_count = 1
                    sequence_index += 1
                prev_stop = stop_timestamp

            verb_noun_pairs.append(verb_noun_pair)

    with open(json_path, "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"Wrote {len(sequences)} sequences to {json_path} and narration log to {txt_path}")

def main():

    print("Reading video info")
    with open(VIDEO_INFO_FILE) as f:
        video_info = list(csv.DictReader(f))

    print("Reading narration low level")
    narration_low_level = []
    for file in NARRATION_LOW_LEVEL_FILES:
        narration_file = pd.read_csv(file)
        narration_low_level.append(narration_file)
    narration_low_level_df = pd.concat(narration_low_level)

    for row in video_info:
        print(f"Processing video {row['video_id']}")
        save_active_objects(row["video_id"], video_info, narration_low_level_df, output_dir="active_objects")
    
    # ## DEBUG
    # save_active_objects("P37_101", video_info, narration_low_level_df, output_dir="active_objects")


if __name__ == "__main__":
    main()