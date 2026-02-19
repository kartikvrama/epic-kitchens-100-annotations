import os
import csv
import json
import pandas as pd
from math import floor, ceil

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
NARRATION_LOW_LEVEL_FILES = ["EPIC_100_train.csv", "EPIC_100_validation.csv", "EPIC_100_test_timestamps.csv"]
NOUN_CLASSES_FILE = "EPIC_100_noun_classes_v2.csv"

def load_noun_class_names():
    """Load class_id -> class name (key) from EPIC_100_noun_classes_v2.csv."""
    with open(NOUN_CLASSES_FILE) as f:
        reader = csv.DictReader(f)
        return {int(row["id"]): row["key"] for row in reader}

def hhmmss_to_seconds(hhmmss):
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def save_active_objects(video_id, video_info, narration_low_level_df, noun_class_names, output_dir="./"):
    visor_annotations_file = f"visor_annotations/train/{video_id}.json"
    if not os.path.exists(visor_annotations_file):
        visor_annotations_file = f"visor_annotations/val/{video_id}.json"
        if not os.path.exists(visor_annotations_file):
            # print(f"Visor annotations not found for {video_id}")
            return

    with open(visor_annotations_file) as f:
        visor_data = json.load(f)
    visor_annotations = visor_data["video_annotations"]

    with open(f"visor-frame-mapping.json", "r") as f:
        visor_frame_mapping = json.load(f)

    ## Add frame_id and objects to each frame
    for frame in visor_annotations:
        frame_path_adjusted = visor_frame_mapping[video_id][frame["image"]["image_path"].split("/")[-1]]
        frame["frame_id"] = int(frame_path_adjusted.split("_")[-1].split(".")[0])
        frame["objects"] = [(k["class_id"], k["name"]) for k in frame["annotations"]]

    max_subsequence_visor = max([int(frame["image"]["subsequence"].split("_")[-1]) for frame in visor_annotations])

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
    sequence_start_frame = None
    sequence_end_time = None
    sequence_end_frame = None
    non_overlapping_count = 1

    os.makedirs(output_dir, exist_ok=True)
    txt_path = f"{output_dir}/active_objects_{video_id}.txt"
    json_path = f"{output_dir}/active_objects_{video_id}.json"

    with open(txt_path, "w") as txt_file:
        narrations = []
        for _, row in narrations_low_level_filtered.iterrows():
            start_timestamp = hhmmss_to_seconds(row["start_timestamp"])
            stop_timestamp = hhmmss_to_seconds(row["stop_timestamp"])
            start_frame = row["start_frame"]
            stop_frame = row["stop_frame"]

            # Write one block per narration
            txt_file.write(f"Start timestamp: {start_timestamp}, Stop timestamp: {stop_timestamp}\n")
            txt_file.write(f"Narration: {row['narration']}\n")
            txt_file.write("------\n")

            narration = f"{row['narration']};{start_timestamp};{stop_timestamp}"

            # Count non-overlapping narrations; each sequence = 3 non-overlapping actions
            if prev_stop is None or start_timestamp >= prev_stop:
                if non_overlapping_count == 1:
                    sequence_start_time = start_timestamp
                    sequence_start_frame = start_frame
                    narrations = []
                non_overlapping_count += 1
                sequence_end_time = stop_timestamp
                sequence_end_frame = stop_frame

                if non_overlapping_count > 3:
                    # Sequence complete: [sequence_start_time, sequence_end_time] with 3 actions
                    txt_file.write("************\n")
                    # Frame filtering based on subsequence index
                    frames_in_sequence = [
                        frame for frame in visor_annotations
                        if int(frame["image"]["subsequence"].split("_")[-1]) == sequence_index
                    ]
                    # ## Frame filtering based on frame_id
                    # frames_in_sequence = [
                    #     frame for frame in visor_annotations
                    #     if sequence_start_frame <= int(frame["frame_id"]) <= sequence_end_frame
                    # ]
                    # Collect (class_id, name) from all frames, then build objects with class_name from noun classes
                    raw_objects = set(sum([f["objects"] for f in frames_in_sequence], []))
                    objects_in_sequence = []
                    for class_id, name in raw_objects:
                        class_name = noun_class_names.get(class_id, "unknown")
                        # Exclude left/right hand (class_id 300, 301 or hand:left, hand:right)
                        if class_id in (300, 301) or class_name in ("hand:left", "hand:right"):
                            continue
                        objects_in_sequence.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "name": name,
                        })
                    objects_in_sequence = sorted(objects_in_sequence, key=lambda x: (x["class_name"], x["name"]))
                    frame_ids_in_sequence = sorted([f["frame_id"] for f in frames_in_sequence])
                    if not all(sequence_start_frame <= frame_id <= sequence_end_frame for frame_id in frame_ids_in_sequence):
                        print(f"{sequence_index}: Frame IDs in sequence are not within the start and stop frame for {video_id}")
                        # import pdb; pdb.set_trace()
                    txt_file.write(f"Frame IDs in sequence: {frame_ids_in_sequence}\n")
                    obj_strs = [f"{o['class_name']} ({o['name']})" for o in objects_in_sequence]
                    txt_file.write(f"Objects in sequence: {obj_strs}\n")
                    txt_file.write(f"Verb-noun pairs in sequence: {narrations}\n")
                    txt_file.write("************\n")

                    sequences.append({
                        "segment_id": f"{video_id}_ActiveUsage_{sequence_index:04d}",
                        "start_time": sequence_start_time,
                        "end_time": sequence_end_time,
                        "start_frame": sequence_start_frame,
                        "end_frame": sequence_end_frame,
                        "objects_in_sequence": objects_in_sequence,
                        "frame_ids": frame_ids_in_sequence,
                        "narrations": narrations,
                    })
                    ## Reset objects_in_sequence and non-overlapping count
                    non_overlapping_count = 1
                    sequence_index += 1
                prev_stop = stop_timestamp

            narrations.append(narration)

    if len(sequences) == 0:
        print(f"No active objects found for {video_id}")
        return

    max_sequence_index = max([int(sequence["segment_id"].split("_")[-1]) for sequence in sequences])

    if max_sequence_index != max_subsequence_visor:
        print(f"Max sequence index {max_sequence_index} does not match max subsequence {max_subsequence_visor} for {video_id}")
        # import pdb; pdb.set_trace()

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

    print("Loading noun class names")
    noun_class_names = load_noun_class_names()

    for row in video_info:
        # print(f"Processing video {row['video_id']}")
        save_active_objects(row["video_id"], video_info, narration_low_level_df, noun_class_names, output_dir="active_objects")
    
    # ## DEBUG
    # save_active_objects("P37_101", video_info, narration_low_level_df, noun_class_names, output_dir="active_objects")


if __name__ == "__main__":
    main()