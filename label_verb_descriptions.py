"""
For each EPIC verb class, gather verb instances from EPIC_100_verb_classes.csv and
five training narrations from EPIC_100_train.csv, then ask an LLM whether the action
typically leaves the acted-on object idle or in use.

Outputs JSON with definition, post-condition, object_state label, and explanation.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from ast import literal_eval
from pathlib import Path

import pandas as pd

DEFAULT_VERB_CLASSES = "EPIC_100_verb_classes.csv"
DEFAULT_TRAIN = "EPIC_100_train.csv"
DEFAULT_OUT = "verb_class_object_state_labels.json"


SYSTEM_PROMPT = """You are an expert on everyday actions in kitchen egocentric video datasets (EPIC-Kitchens).
You classify how a verb class affects the state of the primary object (the grammatical object of the action — often the `noun` column): after the action completes, is that object best described as idle (at rest, not actively being operated or consumed in the described action) or in use (actively engaged, held for use, running, being processed, etc.)?

Respond ONLY with valid JSON matching the schema given in the user message."""


def parse_instances(cell: str) -> list[str]:
    """Parse the `instances` column (Python-list-like string)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    v = literal_eval(s)
    if not isinstance(v, list):
        return []
    return [str(x).strip().lower() for x in v if str(x).strip()]


def collect_examples(
    train: pd.DataFrame,
    instances: list[str],
    n: int,
    seed: int,
) -> list[dict]:
    """Up to `n` example rows where `verb` is one of `instances`."""
    if not instances:
        return []
    inst = set(instances)
    sub = train[train["verb"].astype(str).str.lower().isin(inst)]
    if sub.empty:
        return []
    k = min(n, len(sub))
    sampled = sub.sample(n=k, random_state=seed, replace=False)
    out = []
    for _, row in sampled.iterrows():
        out.append(
            {
                "narration": str(row.get("narration", "")),
                "verb": str(row.get("verb", "")),
                "noun": str(row.get("noun", "")),
                "video_id": str(row.get("video_id", "")),
                "narration_id": str(row.get("narration_id", "")),
            }
        )
    return out


def build_user_prompt(
    verb_key: str,
    category: str,
    instances: list[str],
    examples: list[dict],
) -> str:
    inst_lines = "\n".join(f"  - {i}" for i in instances)
    ex_lines = []
    for i, ex in enumerate(examples, start=1):
        ex_lines.append(
            f'{i}. narration: "{ex["narration"]}" | verb: {ex["verb"]} | noun: {ex["noun"]} | video: {ex["video_id"]}'
        )
    examples_block = "\n".join(ex_lines) if ex_lines else "(no training examples matched these instances)"

    return f"""Verb class (canonical name): {verb_key}
EPIC category label: {category}

All low-level verb instances mapped to this class:
{inst_lines}

Five example narrations from EPIC-100 train (same verb class via instance match):
{examples_block}

Tasks:
1. Give a concise definition of what this verb class means in kitchen activities, and state the typical post-condition for the primary object (the main `noun` being acted on).
2. Decide whether, immediately after this action, that primary object is best labeled "idle" or "in_use" for downstream state tracking (idle ≈ not actively engaged in the task; in_use ≈ actively used, held for immediate use, powered on, open as a container in use, being washed/cut, etc.). If genuinely ambiguous, pick the more common case and say so in the explanation.
3. Briefly explain your label.

Return JSON with exactly these keys:
{{
  "definition_and_post_condition": "<string>",
  "object_state": "idle" | "in_use",
  "explanation": "<string>"
}}"""


def chat_completion_openai(
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float = 120.0,
) -> str:
    """POST to OpenAI Chat Completions; returns assistant message content."""
    url = "https://api.openai.com/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {err_body}") from e

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected API response: {data!r}")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise RuntimeError(f"Empty content: {data!r}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verb-classes",
        default=DEFAULT_VERB_CLASSES,
        help="Path to EPIC_100_verb_classes.csv",
    )
    parser.add_argument(
        "--train",
        default=DEFAULT_TRAIN,
        help="Path to EPIC_100_train.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of train narrations to sample per class",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N verb classes (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print one prompt; do not call the API",
    )
    args = parser.parse_args()

    vc_path = Path(args.verb_classes)
    tr_path = Path(args.train)
    if not vc_path.is_file():
        raise SystemExit(f"Not found: {vc_path}")
    if not tr_path.is_file():
        raise SystemExit(f"Not found: {tr_path}")

    verb_df = pd.read_csv(vc_path)
    train = pd.read_csv(tr_path)
    for col in ("verb", "narration", "noun", "video_id", "narration_id"):
        if col not in train.columns:
            raise SystemExit(f"Train CSV missing column: {col}")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit("Set OPENAI_API_KEY or use --dry-run")

    rows_out: list[dict] = []
    n_classes = len(verb_df) if args.limit is None else min(args.limit, len(verb_df))

    for idx in range(n_classes):
        r = verb_df.iloc[idx]
        key = str(r.get("key", "")).strip()
        cat = str(r.get("category", "")).strip()
        vid = r.get("id")
        instances = parse_instances(r.get("instances", ""))
        seed = int(vid) if vid is not None and str(vid).isdigit() else idx
        examples = collect_examples(train, instances, args.examples, seed=seed)

        user_prompt = build_user_prompt(key, cat, instances, examples)
        if args.dry_run and idx == 0:
            print("--- dry-run: first user prompt ---\n")
            print(user_prompt)
            print("\n--- end ---")
            return

        raw = chat_completion_openai(api_key, args.model, SYSTEM_PROMPT, user_prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Model did not return valid JSON for {key!r}: {raw[:500]}... ({e})") from e

        obj_state = parsed.get("object_state")
        if obj_state not in ("idle", "in_use"):
            raise SystemExit(
                f"Bad object_state for {key!r}: {obj_state!r}. Full: {parsed!r}"
            )

        try:
            vcid = int(vid) if pd.notna(vid) else None
        except (TypeError, ValueError):
            vcid = vid
        rows_out.append(
            {
                "verb_class_id": vcid,
                "verb_class_key": key,
                "category": cat,
                "instances": instances,
                "train_examples": examples,
                "definition_and_post_condition": parsed.get("definition_and_post_condition"),
                "object_state": obj_state,
                "explanation": parsed.get("explanation"),
                "model": args.model,
            }
        )
        print(f"[{idx + 1}/{n_classes}] {key} -> {obj_state}", flush=True)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows_out)} entries to {out_path}")


if __name__ == "__main__":
    main()
