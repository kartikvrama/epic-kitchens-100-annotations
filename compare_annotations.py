import os
import json
from collections import defaultdict
from utils import load_noun_class_names
from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
import argparse


VLM_ANNOTATIONS_DIR = "vlm_annotations_maxImages1_minSpacing3_20260226"
MANUAL_LABELS_DIR = "manual_labels"
print(f"VLM annotations directory: {VLM_ANNOTATIONS_DIR}")
print(f"Manual labels directory: {MANUAL_LABELS_DIR}")

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_DURATION = 6 # half the length of average active segment

NOUN_CLASS_NAMES = {}


def object_key_to_category_and_name(obj_key):
    """From object key like '0/tap/tap' return (subclass_name, name). Uses last part as name, middle as subclass_name."""
    global NOUN_CLASS_NAMES
    parts = obj_key.split("/", maxsplit=2)
    if len(parts) >= 3:
        class_id = int(parts[0])
        subclass_name = NOUN_CLASS_NAMES[class_id]["key"]
        category = NOUN_CLASS_NAMES[class_id]["category"]
        return category, subclass_name, parts[2]  # category, subclass_name, name
    return "", "", ""


def segment_key(query_object, start_time, end_time):
    """Hashable key for a segment (rounded times for matching)."""
    return (query_object, round(float(start_time), 2), round(float(end_time), 2))


def build_filtered_flat_segments(inactive_segments_data, video_id, objects_skip):
    """Flatten inactive segments and filter by object skip list. Returns [(query_object, start_time, end_time), ...]."""
    items = []
    for obj_key in sorted(inactive_segments_data.keys()):
        parts = obj_key.split("/")
        if len(parts) < 3:
            continue
        subclass_name = parts[1]
        if subclass_name in OBJECTS_TO_EXCLUDE_FROM_VLM:
            continue
        for seg in inactive_segments_data[obj_key]:
            items.append((obj_key, seg["start_time"], seg["end_time"]))
    return items


def load_manual_labels_dict(manual_labels_path):
    """Load manual labels as dict keyed by (query_object, start_time, end_time) -> used (bool). Rounded to 2 decimals."""
    labels = {}
    with open(manual_labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                q = rec.get("query_object")
                st = rec.get("start_time")
                et = rec.get("end_time")
                if q is None or st is None or et is None or "used" not in rec:
                    continue
                key = segment_key(q, st, et)
                labels[key] = bool(rec["used"])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return labels


def load_vlm_annotations_by_segment(vlm_annotations_path):
    """Load VLM annotations and group by segment key. Returns dict key -> list of annotation dicts."""
    by_key = {}
    with open(vlm_annotations_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "date_time" in rec and "query_object" not in rec:
                    continue
                q = rec.get("query_object")
                st = rec.get("start_time")
                et = rec.get("end_time")
                if q is None or st is None or et is None:
                    continue
                key = segment_key(q, st, et)
                by_key.setdefault(key, []).append(rec)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return by_key


def pick_valid_vlm_label(annotations):
    """
    From a list of VLM annotations for the same segment, choose the one with a valid passive usage label.
    Valid = has 'is_passive_usage' and it is a boolean.
    Returns (is_passive_usage: bool, found_valid: bool, reasoning: str). If none valid, returns (False, False, "").
    """
    for a in annotations:
        val = a.get("is_passive_usage")
        if isinstance(val, bool):
            reasoning = a.get("reasoning") or ""
            return (val, True, reasoning)
    return (False, False, "")


def main():
    global NOUN_CLASS_NAMES
    NOUN_CLASS_NAMES = load_noun_class_names()

    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", type=str)
    args = parser.parse_args()
    video_id = args.video_id

    # Load inactive segments
    inactive_segments_file = f"inactive_segments/inactive_segments_{video_id}.json"
    with open(inactive_segments_file, "r") as f:
        inactive_segments_data = json.load(f)

    # Build filtered flat list of segments (object skip list applied)
    flat_segments = build_filtered_flat_segments(
        inactive_segments_data, video_id, OBJECTS_TO_EXCLUDE_FROM_VLM
    )

    # Load manual labels as dict
    manual_labels_file = f"{MANUAL_LABELS_DIR}/inactive_segments_{video_id}.jsonl"
    manual_labels_dict = load_manual_labels_dict(manual_labels_file)

    # Load VLM annotations grouped by segment key
    vlm_annotations_file = f"{VLM_ANNOTATIONS_DIR}/inactive_segments_{video_id}_labels.jsonl"
    vlm_by_key = load_vlm_annotations_by_segment(vlm_annotations_file)

    # Build results: for each filtered segment, manual label, vlm label, and whether vlm was missing
    results = []
    missing_vlm_count = 0
    missing_manual_label_count = 0

    for query_object, start_time, end_time in flat_segments:
        if end_time - start_time < MIN_DURATION:
            continue
        key = segment_key(query_object, start_time, end_time)
        manual_label = manual_labels_dict.get(key)  # None if not manually labeled
        if manual_label is None:
            print(f"No manual label for segment: {key}")
            missing_manual_label_count += 1
            continue

        vlm_list = vlm_by_key.get(key, [])
        vlm_label, has_valid_vlm, vlm_reasoning = pick_valid_vlm_label(vlm_list)
        assert not has_valid_vlm == (len(vlm_list) == 0), f"has_valid_vlm: {has_valid_vlm}, vlm_list: {vlm_list}"
        if not has_valid_vlm or not vlm_list:
            print(f"No valid VLM label for segment: {key}")
            vlm_label = False
            vlm_reasoning = ""
            missing_vlm_count += 1

        results.append({
            "query_object": query_object,
            "start_time": round(float(start_time), 2),
            "end_time": round(float(end_time), 2),
            "manual_label": manual_label,
            "vlm_label": vlm_label,
            "vlm_reasoning": vlm_reasoning,
        })

    # F1: only on segments that have a manual label
    TP = FP = FN = TN = 0
    for r in results:
        if r["manual_label"] is None:
            continue
        gt = r["manual_label"]
        pred = r["vlm_label"]
        if gt and pred:
            TP += 1
        elif not gt and pred:
            FP += 1
        elif gt and not pred:
            FN += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Report
    print(f"Video: {video_id}")
    # Get the number of unique objects in the filtered segments
    unique_objects = set([query_object for query_object, _, _ in flat_segments])
    # print(f"Unique objects: {unique_objects}")
    num_objects = len(unique_objects)
    print(f"Number of unique objects in filtered segments: {num_objects}")
    print(f"Filtered segments (after object skip list): {len(flat_segments)}")
    print(f"Segments with manual label (used in F1): {TP + FP + FN + TN}")

    print()
    print(f"Missing manual labels: {missing_manual_label_count}")
    print(f"Missing VLM annotations (no VLM row for segment): {missing_vlm_count}")

    print()
    print("Confusion (manual = ground truth, vlm = prediction):")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 score:  {f1:.4f}")
    print(f"  FPR (FP / (FP+TN)): {(FP / (FP + TN)):.4f}")

    # Per-object false positive rate: FPR = FP / (FP + TN) among segments where manual_label is False
    per_obj = defaultdict(lambda: {"FP": 0, "TN": 0})
    for r in results:
        if r["manual_label"] is None:
            continue
        gt = r["manual_label"]
        pred = r["vlm_label"]
        category, subclass_name, name = object_key_to_category_and_name(r["query_object"])
        if not gt:  # ground truth negative
            if pred:
                per_obj[category]["FP"] += 1
            else:
                per_obj[category]["TN"] += 1

    print()
    print("Per-object false positive rate (FPR = FP / (FP+TN), among manual negatives):")
    print("-" * 70)
    for obj in sorted(per_obj.keys()):
        fp, tn = per_obj[obj]["FP"], per_obj[obj]["TN"]
        neg = fp + tn
        fpr = fp / neg if neg > 0 else 0.0
        print(f"  {obj}")
        print(f"    FP={fp}, TN={tn} (negatives={neg})  FPR={fpr:.4f}")

    # Optionally save results to JSONL for inspection
    out_path = f"{OUTPUT_DIR}/compare_results_{video_id}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {out_path}")


    def seconds_to_hhmmss(secs):
        """Format seconds (float) as hh:mm:ss (with fractional seconds)."""
        secs = float(secs)
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"


    # False positives: manual_label false and vlm_label true
    false_positives = [
        r for r in results
        if r["manual_label"] is not None and not r["manual_label"] and r["vlm_label"]
    ]

    fp_txt_path = f"{OUTPUT_DIR}/compare_false_positives_{video_id}.txt"
    with open(fp_txt_path, "w") as f:
        f.write(f"False positives ({video_id}) — manual label false and VLM label true\n")
        f.write(f"Total: {len(false_positives)}\n")
        f.write("-" * 60 + "\n")
        for r in false_positives:
            start_ss = seconds_to_hhmmss(r["start_time"])
            end_ss = seconds_to_hhmmss(r["end_time"])
            f.write(f"{r['query_object']}\n")
            f.write(f"  Start: {start_ss}  End: {end_ss}\n")
            reasoning = r.get("vlm_reasoning", "")
            if reasoning:
                f.write(f"  VLM reasoning: {reasoning}\n")
            f.write("\n")
    print(f"False positives list written to {fp_txt_path}")


if __name__ == "__main__":
    main()