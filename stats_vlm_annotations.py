import os
import json
import numpy as np
from collections import defaultdict
from utils import load_inactive_segments, load_inactive_annotations
from utils import MIN_DURATION_INACTIVE_SEGMENT

from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM

INACTIVE_ANNOTATIONS_DIR = "vlm_annotations_maxImages1_minSpacing3_20260226"
LENGTH_BINS = [(0, 10, "0-10s"), (10, 60, "10-60s"), (60, 300, "60-300s"), (300, float("inf"), "300s+")]


def pct_yes(anns):
    if not anns:
        return float("nan")
    return sum(1 for a in anns if a.get("is_passive_usage") is True) / len(anns)


def stats(values):
    clean = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "std": float("nan"), "n": 0}
    a = np.array(clean)
    return {"mean": float(np.mean(a)), 
            "std": float(np.std(a)) if len(a) > 1 else 0.0, 
            "min": float(np.min(a)), 
            "max": float(np.max(a)),
            "n": len(clean)}


def crop_type(seg):
    crops = seg.get("crop_from_previous_active") or []
    if not crops:
        return None
    by_frame = defaultdict(int)
    for c in crops:
        by_frame[c.get("frame_path") or ""] += 1
    m = max(by_frame.values())
    return "one" if m == 1 else "more"


def length_bin(dur):
    for lo, hi, label in LENGTH_BINS:
        if lo <= dur < hi:
            return label
    return "other"


def main():
    vlm_dir = INACTIVE_ANNOTATIONS_DIR
    per_video = []
    for fname in sorted(os.listdir(vlm_dir)):
        if not fname.endswith("_labels.jsonl"):
            continue
        video_id = fname.replace("inactive_segments_", "").replace("_labels.jsonl", "")
        inactive = load_inactive_segments(video_id, object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM, min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT)
        if not video_id or not inactive:
            continue
        anns = load_inactive_annotations(os.path.join(vlm_dir, fname), inactive, object_exclusion_list=OBJECTS_TO_EXCLUDE_FROM_VLM, min_duration_inactive_segment=MIN_DURATION_INACTIVE_SEGMENT)
        if not anns:
            print(f"No annotations found for video {video_id}")
            continue

        sub_name_list = defaultdict(list)
        for a in anns:
            subclass = a["query_object"].split("/", maxsplit=2)[1]
            name = a["query_object"].split("/", maxsplit=2)[2]
            if subclass in OBJECTS_TO_EXCLUDE_FROM_VLM:
                continue
            sub_name_list[subclass].append(name)
        sub_count = {sub: len(names) for sub, names in sub_name_list.items()}
        # import pdb; pdb.set_trace()

        by_sub = {"one_inst": [], "more_inst": []}
        subs_with_more_crops = set()
        by_crop = {"one": [], "more": []}
        by_len = defaultdict(list)
        for a in anns:
            sub = a["query_object"].split("/", maxsplit=2)[1]
            by_sub["one_inst" if sub_count[sub] == 1 else "more_inst"].append(a)
            key = (a["query_object"], a["start_time"], a["end_time"])
            ct = crop_type(inactive.get(key, {}))
            if ct:
                by_crop[ct].append(a)
                if ct == "more":
                    subs_with_more_crops.add(sub)
            by_len[length_bin(a["end_time"] - a["start_time"])].append(a)

        per_video.append({
            "video_id": video_id,
            "n": len(anns),
            "overall": pct_yes(anns),
            "one_inst": pct_yes(by_sub["one_inst"]),
            "more_inst": pct_yes(by_sub["more_inst"]),
            "one_crop": pct_yes(by_crop["one"]),
            "more_crop": pct_yes(by_crop["more"]),
            "by_len": {k: pct_yes(v) for k, v in by_len.items()},
        })

    # Per-video
    print("Per-video: % of responses labeled yes (is_passive_usage=True)")
    print("-" * 60)
    for r in per_video:
        p = lambda x: f"{x:.2f}" if not np.isnan(x) else "—"
        print(f"\n  Video {r['video_id']}  ({r['n']} segments)")
        print(f"    Overall:                              {p(r['overall'])}")
        print(f"    One instance per subclass:              {p(r['one_inst'])}")
        print(f"    More than one instance per subclass:    {p(r['more_inst'])}")
        print(f"    One crop per frame:                    {p(r['one_crop'])}")
        print(f"    More than one crop per frame:         {p(r['more_crop'])}")
        print(f"    Subclasses with more than one crop:    {', '.join(subs_with_more_crops)}")
        print(f"    Subclasses with more than one instance: {', '.join([sub for sub, v in sub_count.items() if v > 1])}")
        print(f"    By segment length:  ", "  ".join(f"{k}={p(v)}" for k, v in sorted(r["by_len"].items())))

    # Summary
    def fmt(s):
        return f"Mean {s['mean']:.2f}   Std {s['std']:.2f}   Min {s['min']:.2f}   Max {s['max']:.2f}   ({s['n']} videos)"

    print("\n" + "=" * 60)
    print("Summary across videos (mean, min, max, std)")
    print("=" * 60)
    print("\n  Overall yes (is_passive_usage=True):                    ", fmt(stats([r["overall"] for r in per_video])))
    print("  One instance per subclass:          ", fmt(stats([r["one_inst"] for r in per_video])))
    print("  More than one instance per subclass:", fmt(stats([r["more_inst"] for r in per_video])))
    print("  One crop per frame:                ", fmt(stats([r["one_crop"] for r in per_video])))
    print("  More than one crop per frame:     ", fmt(stats([r["more_crop"] for r in per_video])))
    print("\n  By segment length:")
    for label in [lb for _, _, lb in LENGTH_BINS] + ["other"]:
        vals = [r["by_len"][label] for r in per_video if label in r["by_len"]]
        vals = [v for v in vals if not np.isnan(v)]
        s = stats(vals)
        if s["n"]:
            print(f"    {label:10}  ", fmt(s))


if __name__ == "__main__":
    main()
