#!/usr/bin/env python3
"""
Query a VLM (via Ollama) to label object state after its last action.

For each object entry in object_last_action_{video_id}.json (e.g. under
object_last_action_combined/), sends the VLM:
  - past 3 actions (context narrations preceding the last action)
  - 1 image before the last action
  - up to 3 uniformly sampled images during the last action
  - 1 image after the last action
and classifies the object state as "idle", "tidied", or "in_use".

Each entry uses ``last_active_segment``: the last narration in ``narrations`` supplies
the action text and (when present) ``narration;start_sec;stop_sec`` timestamps; otherwise
segment ``start_time`` and ``end_time`` / ``stop_time`` are used. Frame anchors use
``start_frame`` and ``stop_frame`` when present.

Usage:
    python label_object_state_vlm.py --video-path /path/to/P01_01.MP4 [options]
"""

import csv
import os
import sys
import argparse
import base64
import json
from math import ceil, floor
from ast import literal_eval
from datetime import datetime
import cv2

import ollama

from utils import VIDEO_INFO_FILE
from utils import include_object
from verb_noun_filtering import label_verb_noun
from prompts_object_state import system_prompt
from generate_object_last_action import load_narration_df

MODEL_NAME = "qwen3-vl:30b"
TEMPERATURE = 0.8
MAX_NUM_PREDICT = 2000
NUM_TRIES = 3
MAX_IMAGES_DURING = 3
NUM_CONTEXT_ACTIONS = 3

OLLAMA_MIN_IMAGE_DIM = 33

VALID_OBJECT_STATES = {"idle", "tidied", "in_use"}

# Max |Δstart| and |Δstop| (seconds) when matching a narration line to EPIC CSV rows
NARRATION_TIME_MATCH_TOL = 0.35


def parse_last_action_from_segment(seg):
    """From ``last_active_segment``, return (narration_text, start_sec, stop_sec) for the last line.

    Narration lines are either ``text`` or ``text;start;stop``. If times are missing on the line,
    uses segment ``start_time`` and ``end_time`` or ``stop_time``.
    """
    if not seg:
        return None
    narrations = seg.get("narrations") or []
    if not narrations:
        return None
    last_line = str(narrations[-1]).strip()
    parts = last_line.split(";")
    narration_text = parts[0].strip() if parts else last_line
    if len(parts) >= 3:
        try:
            start_ts = float(parts[1])
            stop_ts = float(parts[2])
            return narration_text, start_ts, stop_ts
        except ValueError:
            pass
    start_ts = seg.get("start_time")
    stop_ts = seg.get("stop_time")
    if stop_ts is None:
        stop_ts = seg.get("end_time")
    try:
        start_ts = float(start_ts)
        stop_ts = float(stop_ts)
    except (TypeError, ValueError):
        return None
    return narration_text, start_ts, stop_ts


def lookup_narration_row(video_narrations, narration_text, start_ts, stop_ts, tol=NARRATION_TIME_MATCH_TOL):
    """Find the EPIC CSV row for this narration text and time span (seconds)."""
    nar_norm = (narration_text or "").strip()
    if not nar_norm:
        return None
    cand = video_narrations[video_narrations["narration"].astype(str).str.strip() == nar_norm]
    if cand.empty:
        return None
    for _, row in cand.iterrows():
        if abs(row["start_sec"] - start_ts) <= tol and abs(row["stop_sec"] - stop_ts) <= tol:
            return row
    if len(cand) == 1:
        return cand.iloc[0]
    return None


def extract_frames_full_res(frame_numbers, video_path=None, cap=None):
    """Extract frames at original resolution, returned as {frame_number: rgb_array}."""
    own_cap = cap is None
    if own_cap:
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
        else:
            raise ValueError("video_path must be provided if cap is not provided")
    results = {}
    for fn in sorted(set(frame_numbers)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        results[fn] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    if own_cap:
        cap.release()
    return results


def load_video_fps(video_info_csv_path, video_id):
    """Load FPS for the given video_id from EPIC_100_video_info.csv."""
    if not video_info_csv_path or not os.path.isfile(video_info_csv_path):
        return None
    with open(video_info_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_id") == video_id:
                try:
                    return float(row["fps"])
                except (ValueError, KeyError):
                    return None
    return None


def ensure_ollama_model_loaded(model_name):
    """Ensure the specified ollama model is loaded."""
    try:
        models = ollama.list().get("models", [])
        loaded_model_names = set()
        for m in models:
            name = m.get("name") or m.get("model") or ""
            if not name:
                continue
            loaded_model_names.add(name.split(":")[0])
        print(f"Loaded model names: {loaded_model_names}")
        if model_name.split(":")[0] not in loaded_model_names:
            print(f"Pulling model {model_name} since it is not loaded...")
            ollama.pull(model_name)
        else:
            print(f"Model {model_name} is already loaded.")
    except Exception as e:
        print(f"Error while checking/loading model '{model_name}': {e}")


def image_to_base64(rgb_array):
    """Encode an RGB image (numpy array) to base64 PNG string for Ollama."""
    if rgb_array is None or rgb_array.size == 0:
        return None
    h, w = rgb_array.shape[:2]
    if h < OLLAMA_MIN_IMAGE_DIM or w < OLLAMA_MIN_IMAGE_DIM:
        scale = OLLAMA_MIN_IMAGE_DIM / float(min(h, w))
        new_w = OLLAMA_MIN_IMAGE_DIM
        new_h = int(round(h * scale))
        rgb_array = cv2.resize(rgb_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", rgb_array)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ollama_response_metadata(response):
    """Extract serializable debugging fields from an Ollama ChatResponse."""
    if response is None:
        return {}
    out = {}

    def get(k, default=None):
        if isinstance(response, dict):
            return response.get(k, default)
        return getattr(response, k, default)

    out["ollama_model"] = get("model")
    out["ollama_created_at"] = get("created_at")
    out["ollama_done"] = get("done")
    out["ollama_done_reason"] = get("done_reason")
    out["ollama_total_duration_ns"] = get("total_duration")
    out["ollama_load_duration_ns"] = get("load_duration")
    out["ollama_prompt_eval_count"] = get("prompt_eval_count")
    out["ollama_prompt_eval_duration_ns"] = get("prompt_eval_duration")
    out["ollama_eval_count"] = get("eval_count")
    out["ollama_eval_duration_ns"] = get("eval_duration")
    msg = get("message")
    if msg is not None:
        msg_get = (lambda k, d=None: msg.get(k, d)) if isinstance(msg, dict) else (lambda k, d=None: getattr(msg, k, d))
        out["ollama_message_content"] = msg_get("content")
        out["ollama_message_thinking"] = msg_get("thinking")
    try:
        raw = response.model_dump() if hasattr(response, "model_dump") else response
        out["llm_response"] = json.loads(json.dumps(raw, default=str)) if isinstance(raw, dict) else raw
    except Exception:
        out["llm_response"] = str(response)
    return out


def get_object_key(entry):
    """Build a unique object key like 'class_id/subclass_name/name' from an entry."""
    obj = entry["object"]
    return f"{obj['class_id']}/{obj['subclass_name']}/{obj['name']}"


def get_past_actions(video_narrations_sorted, before_timestamp, num_actions=NUM_CONTEXT_ACTIONS):
    """Return the `num_actions` narrations immediately preceding `before_timestamp`.

    Args:
        video_narrations_sorted: DataFrame of narrations for this video, sorted by start_sec ascending.
        before_timestamp: The start timestamp of the last action (seconds).
        num_actions: Number of preceding actions to return.

    Returns:
        List of narration strings (oldest first).
    """
    preceding = video_narrations_sorted[video_narrations_sorted["stop_sec"] <= before_timestamp]
    tail = preceding.tail(num_actions)
    return tail["narration"].tolist()


def sample_frames_during_action(video_cap, start_time, stop_time, fps, max_images):
    """Uniformly sample up to `max_images` frames between start_time and stop_time."""
    start_frame = ceil(start_time * fps)
    end_frame = floor(stop_time * fps)
    total_frames = end_frame - start_frame

    if total_frames <= 0:
        frame_ids = [start_frame]
    elif total_frames <= max_images:
        step = max(1, total_frames // (max_images + 1))
        frame_ids = list(range(start_frame + step, end_frame, step))[:max_images]
        if not frame_ids:
            frame_ids = [(start_frame + end_frame) // 2]
    else:
        step = total_frames // (max_images + 1)
        frame_ids = [start_frame + step * (i + 1) for i in range(max_images)]

    frames = extract_frames_full_res(frame_ids, cap=video_cap)
    images_b64 = []
    for fid in sorted(frames.keys()):
        b64 = image_to_base64(frames[fid])
        if b64 is not None:
            images_b64.append(b64)
    return frame_ids, images_b64


def extract_single_frame_b64(video_cap, frame_id):
    """Extract a single frame and return its base64 encoding (or None)."""
    frames = extract_frames_full_res([frame_id], cap=video_cap)
    frame = frames.get(frame_id)
    if frame is not None:
        return image_to_base64(frame)
    return None


def build_object_info_message(obj):
    """Build user message with object metadata."""
    lines = [
        f"Object category: {obj['category']}",
        f"Sub-category: {obj['subclass_name']}",
        f"Object name: {obj['name']}",
    ]
    return "\n".join(lines)


def query_ollama(model, messages, num_tries=NUM_TRIES):
    """Send messages to Ollama with structured JSON output. Returns raw response or None."""
    json_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "object_state": {"type": "string", "enum": ["idle", "tidied", "in_use"]},
        },
        "required": ["reasoning", "object_state"],
    }

    print("=== Messages to be sent to Ollama ===")
    for idx, msg in enumerate(messages):
        content = msg.get("content")
        images = msg.get("images", [])
        images_len = len(images) if images is not None else 0
        print(f"Message {idx}: Content: {repr(content[:120] if content else content)}  Images: {images_len}")
    print("=== End of messages ===")

    last_exception = None
    for attempt in range(num_tries):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                format=json_schema,
                options={"temperature": TEMPERATURE, "num_predict": MAX_NUM_PREDICT, "num_ctx": 150000},
            )
            if response is not None:
                if response.get("message") and response.get("message").get("content"):
                    return response
                else:
                    print(f"query_ollama: got response with no message content (attempt {attempt + 1}/{num_tries})")
            else:
                print(f"query_ollama: got None response (attempt {attempt + 1}/{num_tries})")
        except Exception as e:
            print(f"query_ollama: Exception on attempt {attempt + 1}/{num_tries}: {e}")
            last_exception = e
    if last_exception is not None:
        raise last_exception
    return None


def parse_vlm_response(response):
    """Parse reasoning and object_state from Ollama response. Returns (reasoning, object_state)."""
    reasoning, object_state = None, None
    if response is None:
        return reasoning, object_state
    msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if msg:
        raw_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if raw_content and isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
                reasoning = parsed.get("reasoning")
                object_state = parsed.get("object_state")
            except json.JSONDecodeError:
                pass
    if reasoning is None and object_state is None:
        if isinstance(response, dict):
            reasoning = response.get("reasoning")
            object_state = response.get("object_state")
        else:
            reasoning = getattr(response, "reasoning", None)
            object_state = getattr(response, "object_state", None)
    return reasoning, object_state


def safe_print_str(s):
    """Return a string safe for printing on latin-1/ASCII consoles (e.g. replace em dash)."""
    if s is None:
        return None
    if not isinstance(s, str):
        return str(s)
    return s.encode("ascii", errors="replace").decode("ascii")


def save_debug_visualization(
    obj_key,
    obj,
    before_b64,
    during_b64,
    after_b64,
    during_frame_ids,
    narration_text,
    start_ts,
    stop_ts,
    past_actions,
    output_dir,
    video_path,
    video_id,
    count,
):
    """Create and save a debug image with before/during/after frames and text info."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        ncols = max(2, 2 + max(len(during_frame_ids), 1))
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(3 * (2 + max(len(during_frame_ids), 1)), 6))
        if axes.ndim == 1:
            axes = axes.reshape(2, -1)

        titles = []
        imgs = []

        if before_b64:
            before_img = cv2.imdecode(np.frombuffer(base64.b64decode(before_b64), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(before_img)
            titles.append("Before")
        else:
            imgs.append(None)
            titles.append("Before (None)")

        for i, db64 in enumerate(during_b64):
            if db64:
                dur_img = cv2.imdecode(np.frombuffer(base64.b64decode(db64), np.uint8), cv2.IMREAD_COLOR)
                imgs.append(dur_img)
                titles.append(f"During {i+1}")
            else:
                imgs.append(None)
                titles.append(f"During {i+1} (None)")

        if after_b64:
            after_img = cv2.imdecode(np.frombuffer(base64.b64decode(after_b64), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(after_img)
            titles.append("After")
        else:
            imgs.append(None)
            titles.append("After (None)")

        for ax, img, title in zip(axes[0], imgs, titles):
            ax.axis("off")
            ax.set_title(title)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

        text_lines = [
            f"Object: {obj.get('name', '')}",
            f"Video: {os.path.basename(video_path)}",
            f"Object key: {obj_key}",
            f"Last action:\n  \"{narration_text}\"",
            f"Start: {start_ts} | Stop: {stop_ts}",
        ]
        if past_actions:
            text_lines.append("Past actions:")
            for i, act in enumerate(past_actions):
                text_lines.append(f"  {i + 1}. {act}")
        else:
            text_lines.append("No prior actions")
        text_block = "\n".join(text_lines)
        for ax in axes[1]:
            ax.axis("off")
        axes[1, 0].text(0, 1, text_block, va="top", ha="left", fontsize=10, wrap=True, family="monospace")
        for idx in range(len(imgs), axes.shape[1]):
            axes[0, idx].axis("off")
            axes[1, idx].axis("off")

        fig.tight_layout()
        debug_image_dir = os.path.join(output_dir, "debug_images")
        os.makedirs(debug_image_dir, exist_ok=True)
        debug_image_path = os.path.join(
            debug_image_dir, f"{video_id}_{count:05d}_{obj_key.replace('/', '_')}_viz.png"
        )
        fig.savefig(debug_image_path, dpi=125)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate debug visualization for {obj_key}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Query VLM (Ollama) for object state labels after last action."
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--video-path", required=True, help="Path to the video file (e.g. P01_01.MP4)")
    parser.add_argument("--last-action-dir", default="object_last_action_combined",
                        help="Directory containing object_last_action_{video_id}.json")
    parser.add_argument("--output-dir", default="vlm_object_state_labels",
                        help="Directory to write output JSONL files")
    parser.add_argument("--max-images-during", type=int, default=MAX_IMAGES_DURING,
                        help="Max images to sample during the last action")
    args = parser.parse_args()

    ensure_ollama_model_loaded(args.model)

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    fps = load_video_fps(VIDEO_INFO_FILE, video_id)
    if fps is None:
        print(f"Error: FPS not found for {video_id} in {VIDEO_INFO_FILE}", file=sys.stderr)
        sys.exit(1)
    print(f"Using FPS={fps} for {video_id}")

    # Load object last action entries
    last_action_file = os.path.join(args.last_action_dir, f"object_last_action_{video_id}.json")
    if not os.path.isfile(last_action_file):
        print(f"Error: Last action file not found: {last_action_file}", file=sys.stderr)
        sys.exit(1)
    with open(last_action_file) as f:
        entries = json.load(f)

    # Load narrations for context (past 3 actions) and verb/noun filtering
    print("Loading narration files for context actions...")
    narration_df = load_narration_df()
    video_narrations = narration_df[narration_df["video_id"] == video_id].copy()
    video_narrations["start_sec"] = video_narrations["start_timestamp"].apply(
        lambda ts: _hhmmss_to_seconds(ts) if isinstance(ts, str) else float(ts)
    )
    video_narrations["stop_sec"] = video_narrations["stop_timestamp"].apply(
        lambda ts: _hhmmss_to_seconds(ts) if isinstance(ts, str) else float(ts)
    )
    video_narrations = video_narrations.sort_values("start_sec").reset_index(drop=True)
    print(f"Loaded {len(video_narrations)} narrations for {video_id}")

    filtered_entries = []
    for entry in entries:
        seg = entry.get("last_active_segment")
        parsed = parse_last_action_from_segment(seg) if seg else None
        if not parsed:
            print(f"Warning: skip entry (no parsable last action): {entry.get('object')}", file=sys.stderr)
            continue
        narration_text, start_ts, stop_ts = parsed
        skip_vlm = False
        try:
            row = lookup_narration_row(video_narrations, narration_text, start_ts, stop_ts)
            if row is not None:
                verb = row["verb"]
                noun_classes = literal_eval(str(row["all_noun_classes"]))
                if label_verb_noun(verb=verb, noun_classes=noun_classes) != "unknown":
                    skip_vlm = True
        except Exception as e:
            print(f"Error: {e} for entry object={entry.get('object')}", file=sys.stderr)
        if skip_vlm:
            continue
        filtered_entries.append(
            {"entry": entry, "narration_text": narration_text, "start_ts": start_ts, "stop_ts": stop_ts}
        )
    total_entries = len(filtered_entries)
    print(f"Loaded {len(entries)} rows from {last_action_file}; {total_entries} after verb/noun filter (unknown only)")

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"object_state_{video_id}_labels.jsonl")
    input_prompt_path = os.path.join(args.output_dir, f"input_prompt_{video_id}.jsonl")
    debug_path = os.path.join(args.output_dir, f"debug_vlm_{video_id}.jsonl")

    # Resume support: load previously saved results
    saved_results = []
    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            saved_results = [json.loads(line) for line in f if line.strip()]
        print(f"Resuming: found {len(saved_results)} existing results in {output_path}")

    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for path in [output_path, input_prompt_path, debug_path]:
        with open(path, "a") as fout:
            fout.write(json.dumps({"date_time": date_time_str}) + "\n")

    done = 0

    try:
        video_cap = cv2.VideoCapture(str(args.video_path))
    except Exception as e:
        print(f"Error: Failed to open video file: {args.video_path} - {e}", file=sys.stderr)
        sys.exit(1)

    for count, item in enumerate(filtered_entries):
        entry = item["entry"]
        obj = entry["object"]
        obj_key = get_object_key(entry)
        las = entry.get("last_active_segment") or {}

        if not include_object(obj["category"], obj["subclass_name"], obj["name"]):
            continue

        narration_text = item["narration_text"]
        start_ts = item["start_ts"]
        stop_ts = item["stop_ts"]

        # Check if already processed
        if _is_already_processed(saved_results, obj_key, start_ts, stop_ts):
            print(f"Skipping already processed: {obj_key} ({narration_text})", flush=True)
            continue

        done += 1
        print(f"[{done}/{total_entries}] Processing {obj_key}: \"{narration_text}\" ({start_ts:.2f}-{stop_ts:.2f})", flush=True)

        # --- CONTEXT: past 3 actions ---
        past_actions = get_past_actions(video_narrations, start_ts, NUM_CONTEXT_ACTIONS)

        # --- FRAME EXTRACTION ---
        before_frame_id = max(0, ceil(start_ts * fps) - fps)
        after_frame_id = floor(stop_ts * fps) + fps
        if las.get("start_frame") is not None:
            before_frame_id = max(0, int(las["start_frame"]) - 1)
        if las.get("stop_frame") is not None:
            after_frame_id = int(las["stop_frame"])

        before_b64 = extract_single_frame_b64(video_cap, before_frame_id)
        during_frame_ids, during_b64 = sample_frames_during_action(
            video_cap, start_ts, stop_ts, fps, args.max_images_during
        )
        after_b64 = extract_single_frame_b64(video_cap, after_frame_id)

        # --- BUILD VLM MESSAGES ---
        messages = [{"role": "system", "content": system_prompt}]

        # Message 1: Object info
        messages.append({
            "role": "user",
            "content": build_object_info_message(obj),
        })

        # Message 2: Past 3 actions (context)
        if past_actions:
            context_text = "Recent actions before the last action:\n" + "\n".join(
                f"  {i + 1}. {a}" for i, a in enumerate(past_actions)
            )
        else:
            context_text = "No prior actions recorded for context."
        messages.append({"role": "user", "content": context_text})

        # Message 3: Image before the last action
        if before_b64:
            messages.append({
                "role": "user",
                "content": "Image before the last action:",
                "images": [before_b64],
            })

        # Message 4: Last action narration + images during the action
        action_msg = {"role": "user", "content": f"Last action: \"{narration_text}\""}
        if during_b64:
            action_msg["images"] = during_b64
        messages.append(action_msg)

        # Message 5: Image after the last action
        if after_b64:
            messages.append({
                "role": "user",
                "content": "Image after the last action:",
                "images": [after_b64],
            })

        # --- WRITE INPUT PROMPT ---
        with open(input_prompt_path, "a") as fout:
            fout.write(json.dumps({
                "object_key": obj_key,
                "narration": narration_text,
                "start_timestamp": start_ts,
                "stop_timestamp": stop_ts,
                "past_actions": past_actions,
                "before_frame_id": before_frame_id,
                "during_frame_ids": during_frame_ids,
                "after_frame_id": after_frame_id,
                "video_path": args.video_path,
            }) + "\n")
        
        # --- CREATE DEBUG VISUALIZATION IMAGE ---
        save_debug_visualization(
            obj_key=obj_key,
            obj=obj,
            before_b64=before_b64,
            during_b64=during_b64,
            after_b64=after_b64,
            during_frame_ids=during_frame_ids,
            narration_text=narration_text,
            start_ts=start_ts,
            stop_ts=stop_ts,
            past_actions=past_actions,
            output_dir=args.output_dir,
            video_path=args.video_path,
            video_id=video_id,
            count=count,
        )

        # Skip if no images could be extracted at all
        has_any_image = before_b64 or during_b64 or after_b64
        if not has_any_image:
            print(f"  Warning: No images extracted for {obj_key}, skipping VLM query", flush=True)
            with open(debug_path, "a") as fout:
                fout.write(json.dumps({
                    "object_key": obj_key,
                    "narration": narration_text,
                    "start_timestamp": start_ts,
                    "stop_timestamp": stop_ts,
                    "error": "no images extracted",
                }) + "\n")
            continue

        # --- QUERY OLLAMA ---
        response = query_ollama(args.model, messages)
        reasoning, object_state = parse_vlm_response(response)

        print(f"  Reasoning: {safe_print_str(reasoning)}")
        print(f"  Object state: {safe_print_str(object_state)}")

        # --- WRITE RESULTS ---
        try:
            result = {
                "object_key": obj_key,
                "narration": narration_text,
                "start_timestamp": start_ts,
                "stop_timestamp": stop_ts,
                "reasoning": reasoning,
                "object_state": object_state,
            }
            debug_record = {
                "object_key": obj_key,
                "object_name": obj["name"],
                "narration": narration_text,
                "start_timestamp": start_ts,
                "stop_timestamp": stop_ts,
                "reasoning": reasoning,
                "object_state": object_state,
                "past_actions": past_actions,
                **_ollama_response_metadata(response),
            }
        except Exception as e:
            result = {
                "object_key": obj_key,
                "narration": narration_text,
                "start_timestamp": start_ts,
                "stop_timestamp": stop_ts,
                "error": str(e),
                "parse_error": True,
            }
            debug_record = {
                "object_key": obj_key,
                "narration": narration_text,
                "start_timestamp": start_ts,
                "stop_timestamp": stop_ts,
                "error": str(e),
                **_ollama_response_metadata(response),
            }

        with open(output_path, "a") as fout:
            fout.write(json.dumps(result) + "\n")
        with open(debug_path, "a") as fout:
            fout.write(json.dumps(debug_record) + "\n")

        print(f"  Done: {obj_key}", flush=True)

    video_cap.release()
    print(f"\nFinished. Processed {done} entries for {video_id}.")


def _hhmmss_to_seconds(ts_str):
    """Convert 'HH:MM:SS.ss' to float seconds."""
    parts = ts_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _is_already_processed(saved_results, obj_key, start_ts, stop_ts):
    """Check if a valid result already exists for this object/timestamp pair."""
    for r in saved_results:
        if (r.get("object_key") == obj_key
                and r.get("start_timestamp") == start_ts
                and r.get("stop_timestamp") == stop_ts
                and r.get("reasoning")
                and r.get("object_state") in VALID_OBJECT_STATES):
            return True
    return False


if __name__ == "__main__":
    main()
