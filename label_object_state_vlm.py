#!/usr/bin/env python3
"""
Query a VLM (via Ollama) to label object state after its last action.

For each object entry in object_last_action_{video_id}.json, sends the VLM:
  - past 3 actions (context narrations preceding the last action)
  - 1 image before the last action
  - up to 3 uniformly sampled images during the last action
  - 1 image after the last action
and classifies the object state as "idle", "tidied", or "in_use".

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


def main():
    parser = argparse.ArgumentParser(
        description="Query VLM (Ollama) for object state labels after last action."
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--video-path", required=True, help="Path to the video file (e.g. P01_01.MP4)")
    parser.add_argument("--last-action-dir", default="object_last_action",
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
    filtered_entries = {
        entry for entry in entries if label_verb_noun(entry["last_narration"]["verb"], entry["last_narration"]["noun_categories"]) == "unknown"
    }
    total_entries = len(entries)
    print(f"Loaded {total_entries} object entries from {last_action_file}")

    # Load narrations for context (past 3 actions)
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

    for entry in filtered_entries:
        obj = entry["object"]
        obj_key = get_object_key(entry)
        last_narr = entry["last_narration"]
        active_seg = entry.get("active_segment")

        if not include_object(obj["category"], obj["subclass_name"], obj["name"]):
            continue

        narration_text = last_narr.get("narration", "")
        start_ts = last_narr["start_timestamp"]
        stop_ts = last_narr["stop_timestamp"]

        # Check if already processed
        if _is_already_processed(saved_results, obj_key, start_ts, stop_ts):
            print(f"Skipping already processed: {obj_key} ({narration_text})", flush=True)
            continue

        done += 1
        print(f"[{done}/{total_entries}] Processing {obj_key}: \"{narration_text}\" ({start_ts:.2f}-{stop_ts:.2f})", flush=True)

        # --- CONTEXT: past 3 actions ---
        past_actions = get_past_actions(video_narrations, start_ts, NUM_CONTEXT_ACTIONS)

        # --- FRAME EXTRACTION ---
        before_frame_id = max(0, ceil(start_ts * fps) - 1)
        after_frame_id = floor(stop_ts * fps)
        if active_seg:
            if active_seg.get("start_frame") is not None:
                before_frame_id = max(0, active_seg["start_frame"] - 1)
            if active_seg.get("end_frame") is not None:
                after_frame_id = active_seg["end_frame"]

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

        print(f"  Reasoning: {reasoning}")
        print(f"  Object state: {object_state}")

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
