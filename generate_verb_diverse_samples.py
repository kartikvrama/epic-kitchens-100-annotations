"""
For each verb, print 20 diverse samples from object_last_action, with diversity
across noun classes. Each sample shows noun_class id, noun name, and narration.
"""
import json
import os
import random
from collections import defaultdict

OBJECT_LAST_ACTION_DIR = "object_last_action"
NUM_SAMPLES_PER_VERB = 10
SEED = 42


def load_all_entries_with_source(data_dir):
    """
    Load all object_last_action JSONs; return list of
    (source_basename, segment_id, noun_class, narration, noun_name, verb).
    """
    entries = []
    for name in os.listdir(data_dir):
        if not name.endswith(".json") or not name.startswith("object_last_action_"):
            continue
        path = os.path.join(data_dir, name)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            last = entry.get("last_narration") or {}
            verb = (last.get("verb") or "").strip()
            if not verb:
                continue
            narration = (last.get("narration") or "").strip()
            all_noun_classes = last.get("all_noun_classes")
            all_nouns = last.get("all_nouns")
            seg = entry.get("active_segment") or {}
            segment_id = seg.get("segment_id") or ""
            if narration and all_noun_classes is not None and all_nouns is not None and segment_id:
                source = os.path.basename(path)
                entries.append((source, segment_id, all_noun_classes, all_nouns, narration, verb))
    return entries


def sample_diverse_by_noun_class(entries_for_verb, k=20, rng=None):
    """
    From entries for one verb, sample up to k entries with maximum diversity
    across noun classes (round-robin so we get as many distinct noun classes as possible).
    Deduplicates by (narration, noun_class) first.
    Returns list of (source, segment_id, noun_class, narration, noun_name, verb).
    """
    rng = rng or random.Random(SEED)
    # One representative per (narration, noun_class)
    unique = list({(e[3], e[2]): e for e in entries_for_verb}.values())
    if not unique:
        return []
    by_noun = defaultdict(list)
    for e in unique:
        by_noun[e[2]].append(e)
    noun_classes = list(by_noun.keys())
    rng.shuffle(noun_classes)
    indices = {nc: 0 for nc in noun_classes}
    result = []
    for i in range(k):
        nc = noun_classes[i % len(noun_classes)]
        items = by_noun[nc]
        idx = indices[nc] % len(items)
        result.append(items[idx])
        indices[nc] += 1
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Noun class id -> key (name) from EPIC CSV
    try:
        from utils import load_noun_class_names
        noun_info = load_noun_class_names()  # id -> {"key": ..., "category": ...}
    except Exception:
        noun_info = {}

    def noun_display(noun_class):
        info = noun_info.get(noun_class, {})
        key = info.get("key") if isinstance(info, dict) else ""
        return key or str(noun_class)

    all_entries = load_all_entries_with_source(OBJECT_LAST_ACTION_DIR)
    by_verb = defaultdict(list)
    for t in all_entries:
        by_verb[t[5]].append(t)
    verbs = sorted(by_verb, key=lambda v: len(by_verb[v]), reverse=True)

    rng = random.Random(SEED)
    out_lines = []

    for verb in verbs:
        entries = by_verb.get(verb, [])
        total = len(entries)
        samples = sample_diverse_by_noun_class(entries, k=NUM_SAMPLES_PER_VERB, rng=rng)

        header = f"\n{'='*60}\nVerb: {verb}  (total in data: {total}, showing up to {NUM_SAMPLES_PER_VERB} diverse)\n{'='*60}"
        print(header)
        out_lines.append(header)

        for i, t in enumerate(samples, 1):
            source, segment_id, noun_class, narration, noun_name, _ = t
            name_display = noun_name or noun_display(noun_class)
            line = f"  {i:2}. [{noun_class:3}] {name_display:<20} \"{narration}\""
            print(line)
            out_lines.append(line)

    out_path = os.path.join(OBJECT_LAST_ACTION_DIR, "verb_sample_narrations.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
