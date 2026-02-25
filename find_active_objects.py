import argparse
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
        return {int(row["id"]): {"key": row["key"], "category": row["category"]} for row in reader}

def hhmmss_to_seconds(hhmmss):
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def bbox_from_segments(segments):
    """Compute axis-aligned bounding box [x, y, w, h] from VISOR polygon segments."""
    all_x, all_y = [], []
    for polygon in segments:
        for pt in polygon:
            all_x.append(pt[0])
            all_y.append(pt[1])
    if not all_x:
        return None
    x = min(all_x)
    y = min(all_y)
    w = max(all_x) - x
    h = max(all_y) - y
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]

def get_crop_for_object(frames_in_sequence, class_id, name):
    """Return (frame_id, bbox, image_name) for first frame containing this object, or (None, None, None)."""
    image_crops = []
    for frame in frames_in_sequence:
        for ann in frame["annotations"]:
            if ann["class_id"] == class_id and ann["name"] == name:
                bbox = bbox_from_segments(ann.get("segments", []))
                if bbox is not None:
                    image_name = frame["frame_path_adjusted"]
                    image_crops.append({"frame_path": image_name, "bbox": bbox})
    return image_crops

def save_active_objects(video_id, video_info, narration_low_level_df, noun_class_names, output_dir="./"):
    visor_annotations_file = f"visor_annotations/train/{video_id}.json"
    split = "train"
    if not os.path.exists(visor_annotations_file):
        visor_annotations_file = f"visor_annotations/val/{video_id}.json"
        split = "val"
        if not os.path.exists(visor_annotations_file):
            # print(f"Visor annotations not found for {video_id}")
            return

    with open(visor_annotations_file) as f:
        visor_data = json.load(f)
    visor_annotations = visor_data["video_annotations"]

    with open(f"visor-frame-mapping.json", "r") as f:
        visor_frame_mapping = json.load(f)
    with open(f"visor-frames_to_timestamps.json", "r") as f:
        visor_frame_to_timestamps = json.load(f)["timestamps"]

    ## Add frame_id and objects to each frame
    for frame in visor_annotations:
        visor_frame_path = frame["image"]["image_path"].split("/")[-1]
        frame_path_adjusted = visor_frame_mapping[video_id][visor_frame_path]
        frame_timestamp = visor_frame_to_timestamps[visor_frame_path]
        ## Frame path mapped to EPIC-KITCHENS-100 dataset
        frame["frame_path_adjusted"] = frame_path_adjusted
        frame["frame_id"] = int(frame_path_adjusted.split("_")[-1].split(".")[0])
        frame["timestamp"] = frame_timestamp
        frame["objects"] = [(k["class_id"], k["name"]) for k in frame["annotations"]]

    max_subsequence_visor = max([int(frame["image"]["subsequence"].split("_")[-1]) for frame in visor_annotations])

    fps = float([v for v in video_info if v["video_id"] == video_id][0]["fps"])
    # Use only narrations from the same split as visor (train/val) so sequence count matches visor
    narrations_low_level_filtered = narration_low_level_df[
        (narration_low_level_df["video_id"] == video_id) & (narration_low_level_df["_source"] == split)
    ]
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
                # Append current (non-overlapping) action before completing sequence so stored narrations has 3 entries
                narrations.append(narration)

                if non_overlapping_count > 3:
                    # Sequence complete: [sequence_start_time, sequence_end_time] with 3 actions
                    txt_file.write("************\n")
                    # Frame filtering based on timestamp
                    frames_in_sequence = [
                        frame for frame in visor_annotations
                        if sequence_start_time <= frame["timestamp"] <= sequence_end_time
                    ]
                    # Collect (class_id, name) from all frames, then build objects with class_name from noun classes
                    raw_objects = set(sum([f["objects"] for f in frames_in_sequence], []))
                    objects_in_sequence = []
                    for class_id, name in raw_objects:
                        object_info = noun_class_names.get(class_id, {})
                        subclass_name = object_info.get("key", "unknown")
                        category = object_info.get("category", "unknown")
                        # Exclude left/right hand (class_id 300, 301 or hand:left, hand:right)
                        if class_id in (11, 300, 301) or name in ("hand:left", "hand:right") or category == "hand":
                            continue
                        image_crops = get_crop_for_object(frames_in_sequence, class_id, name)
                        obj = {
                            "class_id": class_id,
                            "category": category,
                            "subclass_name": subclass_name,
                            "name": name,
                            "image_crops": image_crops,
                        }
                        objects_in_sequence.append(obj)
                    objects_in_sequence = sorted(objects_in_sequence, key=lambda x: (x["category"], x["subclass_name"], x["name"]))

                    ## Check if all frames are within the start and stop frame and timestamp
                    frame_ids_in_sequence = sorted([f["frame_id"] for f in frames_in_sequence])

                    ## Frame id based check
                    if not all(sequence_start_frame <= frame_id <= sequence_end_frame for frame_id in frame_ids_in_sequence):
                        num_out_of_range_frames = sum(
                            not (sequence_start_frame <= frame_id <= sequence_end_frame)
                            for frame_id in frame_ids_in_sequence
                        )
                        print(f"  - [FRAME CHECK] {sequence_index}: {num_out_of_range_frames}/{len(frame_ids_in_sequence)} frame(s) are not within the start and stop frame for {video_id}")

                    txt_file.write(f"Frame IDs in sequence: {frame_ids_in_sequence}\n")
                    obj_strs = [f"{o['category']} ({o['subclass_name']}) ({o['name']})" for o in objects_in_sequence]
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
            # Only non-overlapping actions are appended above (inside the if block)

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


def report_frame_id_mismatch(output_dir="active_objects"):
    """
    For each video_id with both VISOR annotations and active_objects JSON, report:
    - Total unique frame IDs in VISOR
    - Total unique frame IDs in active objects sequences (union across sequences)
    - Mismatch: frames in VISOR not in any sequence, and frames in sequences not in VISOR.
    """
    if not os.path.exists("visor-frame-mapping.json") or not os.path.exists("visor-frames_to_timestamps.json"):
        print("Missing visor-frame-mapping.json or visor-frames_to_timestamps.json")
        return

    with open("visor-frame-mapping.json", "r") as f:
        visor_frame_mapping = json.load(f)
    with open("visor-frames_to_timestamps.json", "r") as f:
        visor_frame_to_timestamps = json.load(f)["timestamps"]

    results = []
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".json") or not fname.startswith("active_objects_"):
            continue
        video_id = fname.replace("active_objects_", "").replace(".json", "")

        visor_annotations_file = f"visor_annotations/train/{video_id}.json"
        if not os.path.exists(visor_annotations_file):
            visor_annotations_file = f"visor_annotations/val/{video_id}.json"
        if not os.path.exists(visor_annotations_file):
            results.append((video_id, None, None, None, None, "no VISOR file"))
            continue

        with open(visor_annotations_file) as f:
            visor_data = json.load(f)
        visor_annotations = visor_data["video_annotations"]

        visor_frame_ids = set()
        for frame in visor_annotations:
            visor_frame_path = frame["image"]["image_path"].split("/")[-1]
            if video_id not in visor_frame_mapping or visor_frame_path not in visor_frame_mapping[video_id]:
                continue
            frame_path_adjusted = visor_frame_mapping[video_id][visor_frame_path]
            frame_id = int(frame_path_adjusted.split("_")[-1].split(".")[0])
            visor_frame_ids.add(frame_id)

        with open(os.path.join(output_dir, fname)) as f:
            sequences = json.load(f)
        active_frame_ids = set()
        for seq in sequences:
            active_frame_ids.update(seq.get("frame_ids", []))

        only_in_visor = visor_frame_ids - active_frame_ids
        only_in_active = active_frame_ids - visor_frame_ids
        results.append((
            video_id,
            len(visor_frame_ids),
            len(active_frame_ids),
            len(only_in_visor),
            len(only_in_active),
            None,
        ))

    # Print table
    print("\nFrame ID mismatch: VISOR total vs active objects sequences (per video_id)")
    print("-" * 90)
    print(f"{'video_id':<12} {'visor_total':>12} {'active_total':>12} {'only_visor':>12} {'only_active':>12}  note")
    print("-" * 90)
    for r in results:
        video_id, v_total, a_total, only_v, only_a, note = r
        if note:
            print(f"{video_id:<12} {'—':>12} {'—':>12} {'—':>12} {'—':>12}  {note}")
        else:
            print(f"{video_id:<12} {v_total:>12} {a_total:>12} {only_v:>12} {only_a:>12}")
    print("-" * 90)
    with_note = [r for r in results if r[5]]
    with_counts = [r for r in results if r[1] is not None]
    if with_counts:
        total_visor = sum(r[1] for r in with_counts)
        total_active = sum(r[2] for r in with_counts)
        total_only_visor = sum(r[3] for r in with_counts)
        total_only_active = sum(r[4] for r in with_counts)
        print(f"Sum (videos with both): visor_total={total_visor}, active_total={total_active}, "
              f"only_in_visor={total_only_visor}, only_in_active={total_only_active}")
    if with_note:
        print(f"Skipped {len(with_note)} video(s) (no VISOR file).")


def main():
    parser = argparse.ArgumentParser(description="Build active objects sequences from VISOR and narrations.")
    parser.add_argument(
        "--report-mismatch",
        action="store_true",
        help="Report mismatch between total frame IDs in VISOR vs active objects sequences per video_id.",
    )
    parser.add_argument("--output-dir", default="active_objects", help="Output directory for active_objects JSON/txt.")
    args = parser.parse_args()

    if args.report_mismatch:
        report_frame_id_mismatch(output_dir=args.output_dir)
        return

    print("Reading video info")
    with open(VIDEO_INFO_FILE) as f:
        video_info = list(csv.DictReader(f))

    print("Reading narration low level")
    narration_low_level = []
    for file in NARRATION_LOW_LEVEL_FILES:
        narration_file = pd.read_csv(file)
        if "train" in file:
            narration_file["_source"] = "train"
        elif "validation" in file:
            narration_file["_source"] = "val"
        else:
            narration_file["_source"] = "test"
        narration_low_level.append(narration_file)
    narration_low_level_df = pd.concat(narration_low_level)

    print("Loading noun class names")
    noun_class_names = load_noun_class_names()

    for row in video_info:
        # print(f"Processing video {row['video_id']}")
        save_active_objects(row["video_id"], video_info, narration_low_level_df, noun_class_names, output_dir=args.output_dir)


if __name__ == "__main__":
    main()