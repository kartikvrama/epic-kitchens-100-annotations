import os
import json
import csv
from object_filtering import CATEGORIES_INCLUDED, SUBCLASSES_EXCLUDED

NOUN_CLASSES_FILE = "EPIC_100_noun_classes_v2.csv"

VIDEO_ID_FILE = "video_paths_updated.csv"
VIDEO_INFO_FILE = "EPIC_100_video_info.csv"

ACTIVE_OBJECTS_DIR = "active_objects"
INACTIVE_SEGMENTS_DIR = "inactive_segments"

VISOR_FRAMES_TO_TIMESTAMPS_FILE = "visor-frames_to_timestamps.json"

MIN_DURATION_INACTIVE_SEGMENT = 6 # seconds


def include_object(cat, subcat, name):
    """Return True if the object (category, subclass, name) should be included in the dataset."""
    if cat not in CATEGORIES_INCLUDED:
        return False
    subcat_prefix = subcat.split(":")[0]
    if subcat_prefix in SUBCLASSES_EXCLUDED:
        return False
    return True


def load_inactive_segments(
    video_id,
    object_exclusion_list=[],
    min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT
):
    """Load inactive segments from disk.
    
    Args:
        video_id: The ID of the video to load inactive segments for.
        object_exclusion_list: A list of object subclasses to exclude from the inactive segments.
        min_duration_inactive_segment: The minimum duration of an inactive segment to be included.
    Returns:
        A dictionary of inactive segments, keyed by (query_object, start_time, end_time).
    """
    path = os.path.join(INACTIVE_SEGMENTS_DIR, f"inactive_segments_{video_id}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for qobj, segs in data.items():
        ## Object exclusion
        if qobj.split("/", maxsplit=2)[1] in object_exclusion_list:
            continue
        for s in segs:
            st, et = s.get("start_time"), s.get("end_time")
            if st is None or et is None:
                continue
            ## Minimum duration exclusion
            if (et - st) < min_duration_inactive_segment:
                continue
            out[(qobj, st, et)] = s
    return out


def load_inactive_annotations(
    video_id,
    annotations_filepath,
    object_exclusion_list=[],
    min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT
):
    """Load VLM generated annotations for inactive segments."""
    inactive = load_inactive_segments(video_id, object_exclusion_list=object_exclusion_list, min_duration_inactive_segment=min_duration_inactive_segment)
    fname = os.path.basename(annotations_filepath)
    if not fname.startswith("inactive_segments_") or not fname.endswith("_labels.jsonl"):
        return []
    with open(annotations_filepath) as f:
        annotations_raw = [json.loads(line) for line in f if line.strip()]

    annotations_filtered = []
    keys_null, keys_missing = set(), set()
    for seg_key, seg in inactive.items():
        object_key, start_time, end_time = seg_key
        matching_annotations = [
            ann for ann in annotations_raw if ann.get("query_object") == object_key and ann.get("start_time") == start_time and ann.get("end_time") == end_time
        ]
        if not matching_annotations:
            keys_missing.add(seg_key)
            continue
        if all(ann.get("is_passive_usage") is None for ann in matching_annotations):
            keys_null.add(seg_key)
            continue
        valid_annotations = [ann for ann in matching_annotations if ann.get("is_passive_usage") is not None]
        if len(valid_annotations) !=1:
            print(f"{len(valid_annotations)} valid annotations for segment {object_key} {start_time}-{end_time}")
            import pdb; pdb.set_trace()
        valid_annotation = valid_annotations[0]
        annotations_filtered.append(valid_annotation)

    # For all segments in the inactive annotation file not present in 'seen', generate a "vlm negative" annotation with is_passive_usage=False
    for seg_key, seg in inactive.items():
        if not isinstance(seg, dict):
            import pdb; pdb.set_trace()
            raise ValueError(f"Inactive segments must be flattened before calling load_inactive_annotations, refer to load_inactive_segments()")
        object_key, start_time, end_time = seg_key
        if seg_key in keys_null or seg_key in keys_missing:
            # Synthesize negative annotation, trying to match VLM fields
            annotations_filtered.append({
                "query_object": object_key,
                "start_time": start_time,
                "end_time": end_time,
                "is_passive_usage": False,
                "synthesized": True
            })
    return annotations_filtered, keys_missing, keys_null



def load_noun_class_names():
    """Load class_id -> class name (key) from EPIC_100_noun_classes_v2.csv."""
    with open(NOUN_CLASSES_FILE) as f:
        reader = csv.DictReader(f)
        return {int(row["id"]): {"key": row["key"], "category": row["category"]} for row in reader}


def hhmmss_to_seconds(hhmmss):
    """Convert a time string in the format "hh:mm:ss" to seconds."""
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


def get_visor_object_appearance_times(video_id, frames_to_timestamps_path=None):
    """Get per-object appearance times for a video from VISOR annotations.

    For the given video_id, loads VISOR annotations, collects all unique objects
    (by object key class_id/subclass/name), and for each object the list of
    timestamps (in seconds) at which it appears. Frame indices are converted to
    timestamps using visor-frames_to_timestamps.json.

    Args:
        video_id: Video ID (e.g. 'P01_01').
        frames_to_timestamps_path: Path to visor-frames_to_timestamps.json.
            Defaults to VISOR_FRAMES_TO_TIMESTAMPS_FILE.

    Returns:
        List of dicts, one per unique object: [{"object_key": str, "timestamps": [float]}, ...].
        Timestamps are in seconds, sorted ascending. Returns [] if VISOR annotations
        or the frames-to-timestamps file are missing.
    """
    frames_to_timestamps_path = frames_to_timestamps_path or VISOR_FRAMES_TO_TIMESTAMPS_FILE
    if not os.path.isfile(frames_to_timestamps_path):
        return []

    visor_path = os.path.join("visor_annotations", "train", f"{video_id}.json")
    if not os.path.isfile(visor_path):
        visor_path = os.path.join("visor_annotations", "val", f"{video_id}.json")
    if not os.path.isfile(visor_path):
        return []

    with open(frames_to_timestamps_path) as f:
        frame_to_ts = json.load(f).get("timestamps") or {}
    with open(visor_path) as f:
        visor_data = json.load(f)
    video_annotations = visor_data.get("video_annotations") or []
    noun_class_names = load_noun_class_names()

    # object_key -> set of timestamps (seconds)
    appearance_times = {}
    for frame in video_annotations:
        image_path = frame.get("image") or {}
        image_path_str = image_path.get("image_path") or ""
        visor_frame_path = image_path_str.split("/")[-1]
        timestamp = frame_to_ts.get(visor_frame_path)
        if timestamp is None:
            continue
        for ann in frame.get("annotations") or []:
            class_id = ann.get("class_id")
            name = ann.get("name")
            if class_id is None or name is None:
                continue
            info = noun_class_names.get(class_id)
            subclass = info.get("key")
            object_key = f"{class_id}/{subclass}/{name}"
            if object_key not in appearance_times:
                appearance_times[object_key] = set()
            appearance_times[object_key].add(float(timestamp))

    return [
        {"object_key": key, "timestamps": sorted(timestamps)}
        for key, timestamps in sorted(appearance_times.items())
    ]

