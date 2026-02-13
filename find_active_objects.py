import csv
import json
import pandas as pd
from math import floor, ceil

VIDEO_ID = "P37_101"
VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
VISOR_ANNOTATIONS_FILE = f"visor_annotations/train/{VIDEO_ID}.json"
NARRATION_SENTENCES_FILES = ["retrieval_annotations/EPIC_100_retrieval_train_sentence.csv", "retrieval_annotations/EPIC_100_retrieval_test_sentence.csv"]
NARRATION_LOW_LEVEL_FILES = ["EPIC_100_train.csv", "EPIC_100_validation.csv"]

def hhmmss_to_seconds(hhmmss):
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def main():
    print("Reading visor annotations")
    with open(VISOR_ANNOTATIONS_FILE) as f:
        visor_data = json.load(f)
    visor_annotations = visor_data["video_annotations"]
    ## Add frame_id and objects to each frame
    for frame in visor_annotations:
        frame["frame_id"] = int(frame["image"]["name"].split("_")[-1].split(".")[0])
        frame["objects"] = [k["name"] for k in frame["annotations"]]

    print("Reading video info")
    with open(VIDEO_INFO_FILE) as f:
        video_info = list(csv.DictReader(f))

    print("Reading narration sentences")
    narration_sentences = []
    for file in NARRATION_SENTENCES_FILES:
        narration_file = pd.read_csv(file)
        narration_sentences.append(narration_file)
    narrations_sentences_df = pd.concat(narration_sentences)

    print("Reading narration low level")
    narration_low_level = []
    for file in NARRATION_LOW_LEVEL_FILES:
        narration_file = pd.read_csv(file)
        narration_low_level.append(narration_file)
    narration_low_level_df = pd.concat(narration_low_level)

    print("Finding active objects")
    fps = float([v for v in video_info if v["video_id"] == VIDEO_ID][0]["fps"])
    narrations_low_level_filtered = narration_low_level_df[narration_low_level_df["video_id"] == VIDEO_ID]
    ## Sort by start_timestamp
    narrations_low_level_filtered = narrations_low_level_filtered.sort_values(by="start_timestamp")

    with open(f"active_objects_{VIDEO_ID}.txt", "w") as f:
        for _, row in narrations_low_level_filtered.iterrows():
            start_timestamp = hhmmss_to_seconds(row["start_timestamp"])
            stop_timestamp = hhmmss_to_seconds(row["stop_timestamp"])
            start_frame = floor(start_timestamp * fps)
            stop_frame = ceil(stop_timestamp * fps)
            f.write(f"Start timestamp: {start_timestamp}, Stop timestamp: {stop_timestamp}\n")
            f.write(f"Start frame: {start_frame}, Stop frame: {stop_frame}\n")
            narration = row["narration"]
            f.write(f"Narration: {narration}\n")

            all_objects_in_range = []
            all_frames_in_range = [frame for frame in visor_annotations if frame["frame_id"] >= start_frame and frame["frame_id"] <= stop_frame]
            f.write(f"Frames in range: {[k['image']['name'] for k in all_frames_in_range]}\n")
            all_objects_in_range = set(sum([frame["objects"] for frame in all_frames_in_range], []))
            f.write(f"{all_objects_in_range}\n")
            f.write("------\n")


if __name__ == "__main__":
    main()