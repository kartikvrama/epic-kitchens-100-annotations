import csv

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

