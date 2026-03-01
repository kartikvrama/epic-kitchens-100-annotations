import os
import json
import csv

NOUN_CLASSES_FILE = "EPIC_100_noun_classes_v2.csv"

VIDEO_ID_FILE = "video_paths_updated.csv"
VIDEO_INFO_FILE = "EPIC_100_video_info.csv"

ACTIVE_OBJECTS_DIR = "active_objects"
INACTIVE_SEGMENTS_DIR = "inactive_segments"

MIN_DURATION_INACTIVE_SEGMENT = 6 # seconds


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


def load_inactive_annotations(filepath, inactive, object_exclusion_list=[], min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT):
    """Load VLM generated annotations for inactive segments."""
    fname = os.path.basename(filepath)
    if not fname.startswith("inactive_segments_") or not fname.endswith("_labels.jsonl"):
        return []
    anns, seen = [], set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "query_object" not in d:
                continue
            if d.get("is_passive_usage") is None:
                continue
            subclass = d["query_object"].split("/", maxsplit=2)[1]
            ## Object exclusion
            if subclass in object_exclusion_list:
                continue
            ## Minimum duration exclusion
            if (float(d["end_time"]) - float(d["start_time"])) < min_duration_inactive_segment:
                continue
            key = (d["query_object"], d["start_time"], d["end_time"])
            if key in seen:
                continue
            seen.add(key)
            anns.append(d)
    # For all segments in the inactive annotation file not present in 'seen', generate a "vlm negative" annotation with is_passive_usage=False
    for seg_key, seg in inactive.items():
        if not isinstance(seg, dict):
            import pdb; pdb.set_trace()
            raise ValueError(f"Inactive segments must be flattened before calling load_inactive_annotations, refer to load_inactive_segments()")
        object_key, start_time, end_time = seg_key
        subclass = object_key.split("/", maxsplit=2)[1]
        ## Object exclusion
        if subclass in object_exclusion_list:
            continue
        ## Minimum duration exclusion
        if (float(seg.get("end_time")) - float(seg.get("start_time"))) < min_duration_inactive_segment:
            continue
        search_key = (object_key, seg.get("start_time"), seg.get("end_time"))
        if search_key in seen:
            continue
        # Synthesize negative annotation, trying to match VLM fields
        anns.append({
            "query_object": object_key,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "is_passive_usage": False,
            "synthesized": True
        })
    return anns



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

