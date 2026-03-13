"""
For each verb in verb_list.txt, print 20 random diverse action narrations
with their noun classes from object_last_action JSON files.
"""
import json
import os
import random
from collections import defaultdict

VERB_LIST_PATH = "object_last_action/verb_list.txt"
OBJECT_LAST_ACTION_DIR = "object_last_action"
NUM_SAMPLES = 20


def load_verbs(path):
    """Parse verb list: lines like '  put-down: 500' -> ['put-down', ...]."""
    verbs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-") or "Per-verb" in line:
                continue
            if ":" in line:
                verb = line.split(":")[0].strip()
                if verb:
                    verbs.append(verb)
    return verbs


def load_all_narrations_by_verb(data_dir):
    """Load all object_last_action JSONs and group (narration, noun_class, noun_name) by verb."""
    by_verb = defaultdict(list)
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
            noun_class = last.get("noun_class")
            noun_name = (last.get("noun") or "").strip()
            if narration and noun_class is not None:
                by_verb[verb].append((narration, noun_class, noun_name))
    return by_verb


def sample_diverse(items, k, seed=42):
    """Sample up to k items, preferring unique (narration, noun_class)."""
    rng = random.Random(seed)
    # Deduplicate by (narration, noun_class) to get diversity
    unique = list({(narr, nc): (narr, nc, nn) for narr, nc, nn in items}.values())
    rng.shuffle(unique)
    return unique[:k]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    verbs = load_verbs(VERB_LIST_PATH)
    by_verb = load_all_narrations_by_verb(OBJECT_LAST_ACTION_DIR)

    for verb in verbs:
        items = by_verb.get(verb, [])
        samples = sample_diverse(items, NUM_SAMPLES)
        print(f"\n{'='*60}")
        print(f"Verb: {verb}  (total in data: {len(items)}, showing up to {NUM_SAMPLES} diverse)")
        print("=" * 60)
        if not samples:
            print("  (no narrations found for this verb)")
            continue
        for i, (narration, noun_class, noun_name) in enumerate(samples, 1):
            print(f"  {i:2}. [{noun_class:3}] {noun_name:20}  \"{narration}\"")


if __name__ == "__main__":
    main()
