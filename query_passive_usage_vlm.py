#!/usr/bin/env python3
"""
Query a VLM (via Ollama) to label passive usage for each inactive segment.
Reads inactive segments from disk, extracts object crops from the video,
sends prompt (system + object category/name + sequence of crops + event history)
and saves JSON responses.

Usage:
    python query_passive_usage_vlm.py --video-path /path/to/P01_01.MP4 [options]
"""

import os
import sys
import argparse
import base64
import json

import cv2
import ollama

from label_passive_usage_vlm import system_prompt
from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from visualize_inactive_segments import (
    extract_frames_full_res,
    extract_object_crops,
    parse_crop_specs,
)

MODEL_NAME="qwen3-vl:30b"
TEMPERATURE=0.8
MAX_NUM_PREDICT=2000
NUM_TRIES=3


def ensure_ollama_model_loaded(model_name):
    """Ensure the specified ollama model is loaded."""
    try:
        models = ollama.list().get("models", [])
        loaded_model_names = set()
        for m in models:
            # Some entries may not have a "name" key, so use get with fallback
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
    # OpenCV imencode writes channels as-is; we have RGB, PNG viewers expect RGB
    ok, buf = cv2.imencode(".png", rgb_array)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def object_key_to_category_and_name(obj_key):
    """From object key like '0/tap/tap' return (category, name). Uses last part as name, middle as category."""
    parts = obj_key.split("/")
    if len(parts) >= 3:
        return parts[1], parts[2]  # class_name, name
    if len(parts) == 2:
        return parts[1], parts[1]
    return parts[0], parts[0] if parts else ("", "")


def format_event_history(event_history, video_path):
    """
    Format event_history for the VLM, preserving order.
    - Items starting with "narration:" -> "Action: {narration text}".
    - Items starting with "frame_id:" -> extract that frame from the video, resize to 3x, encode as base64;
      add a placeholder line and append the image to the returned image list.
    Returns (event_lines, event_frame_images_base64).
    """
    event_lines = []
    event_frame_images_base64 = []
    if not event_history:
        return event_lines, event_frame_images_base64
    frame_ids = [int(item.split("frame_id:", 1)[1]) for item in event_history if item.startswith("frame_id:")]
    frames_dict = extract_frames_full_res(video_path, frame_ids) if frame_ids else {}
    for item in event_history:
        if item.startswith("narration:"):
            action = item.split("narration:", 1)[1].strip()
            event_lines.append(f"Action: {action}")
            event_frame_images_base64.append([])
        elif item.startswith("frame_id:"):
            frame_id = int(item.split("frame_id:", 1)[1])
            frame = frames_dict.get(frame_id)
            if frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                frame_3x = cv2.resize(frame, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
                b64 = image_to_base64(frame_3x)
                if b64 is not None:
                    event_frame_images_base64[-1].append(b64)
                else:
                    print("Warning: Failed to encode frame to base64.")
            else:
                print("Warning: Failed to extract frame.")
    assert len(event_lines) == len(event_frame_images_base64)
    return event_lines, event_frame_images_base64


def build_user_message(category, name):
    """Build user message in required order: object category/name, then line about images, then event history (unchanged order)."""
    lines = [
        f"Object category: {category}",
        f"Object name: {name}",
        "",
        "Images of the object",
    ]
    return "\n".join(lines)


def get_crop_images_base64(video_path, segment):
    """Extract sequence of object crops for segment and return list of base64 PNG strings."""
    crop_specs = parse_crop_specs(segment.get("crop_from_previous_active"))
    if not crop_specs:
        return []
    crops = extract_object_crops(video_path, crop_specs)
    return [image_to_base64(c) for c in crops if image_to_base64(c) is not None]


def query_ollama(
    model,
    user_content,
    crop_images_base64,
    event_history_lines,
    event_frame_images_base64,
    num_tries=NUM_TRIES
):
    """
    POST to Ollama /api/chat with system prompt, user message, images; format json. Returns parsed JSON or None.
    Tries up to `num_tries` times on failure (e.g., exception or None response).
    """
    json_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "is_passive_usage": {"type": "boolean"},
        },
        "required": ["reasoning", "is_passive_usage"]
    }

    messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
                "images": crop_images_base64,
            },
        ]
    for i, (content, images_base64) in enumerate(zip(event_history_lines, event_frame_images_base64)):
        if i == 0:
            content = f"Event history:\n{content}"
        if images_base64:
            messages.append({
                "role": "user",
                "content": content,
                "images": images_base64,
            })
        else:
            messages.append({
                "role": "user",
                "content": content,
            })

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
                return response
            else:
                print(f"query_ollama: got None response (attempt {attempt + 1}/{num_tries})")
        except Exception as e:
            print(f"query_ollama: Exception on attempt {attempt + 1}/{num_tries}: {e}")
            last_exception = e
    # All attempts failed
    if last_exception is not None:
        raise last_exception
    return None


def _ollama_response_metadata(response):
    """
    Extract serializable debugging fields from an Ollama ChatResponse.
    Handles both dict and Pydantic-style (get/attribute) response.
    See: https://docs.ollama.com/api/chat and ollama-python _types.ChatResponse
    """
    if response is None:
        return {}
    out = {}
    def get(k, default=None):
        if isinstance(response, dict):
            return response.get(k, default)
        return getattr(response, k, default)
    # Top-level timing and counts (all durations in nanoseconds)
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
    # Message content (raw string; when format=json this is the JSON text)
    msg = get("message")
    if msg is not None:
        msg_get = (lambda k, d=None: msg.get(k, d)) if isinstance(msg, dict) else (lambda k, d=None: getattr(msg, k, d))
        out["ollama_message_content"] = msg_get("content")
        out["ollama_message_thinking"] = msg_get("thinking")
    # Full response for raw debugging (ensure JSON-serializable)
    try:
        raw = response.model_dump() if hasattr(response, "model_dump") else response
        out["llm_response"] = json.loads(json.dumps(raw, default=str)) if isinstance(raw, dict) else raw
    except Exception:
        out["llm_response"] = str(response)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Query VLM (Ollama) for passive usage labels on inactive segments."
    )
    parser.add_argument('--model', type=str, required=False, default=MODEL_NAME,
                    help='Model name to use')
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the video file (e.g. P01_01.MP4); video_id is derived from basename.",
    )
    parser.add_argument(
        "--segments-dir",
        default="inactive_segments",
        help="Directory containing inactive_segments_{video_id}.json (default: inactive_segments)",
    )
    parser.add_argument(
        "--output-dir",
        default="vlm_passive_usage_labels",
        help="Directory to write labels JSON (default: vlm_passive_usage_labels)",
    )
    args = parser.parse_args()

    ensure_ollama_model_loaded(args.model)

    video_path = os.path.abspath(args.video_path)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    segments_file = os.path.join(args.segments_dir, f"inactive_segments_{video_id}.json")
    if not os.path.isfile(segments_file):
        print(f"Error: Segments file not found: {segments_file}", file=sys.stderr)
        sys.exit(1)

    with open(segments_file) as f:
        segments_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"inactive_segments_{video_id}_labels.json")
    debug_jsonl_path = os.path.join(args.output_dir, f"inactive_segments_{video_id}_labels_debug.jsonl")
    results = {}

    total_segments = sum([len(segments_data[k]) for k in segments_data])
    done = 0
    with open(debug_jsonl_path, "w") as debug_jsonl:
        for obj_key in sorted(segments_data.keys()):
            seg_list = segments_data[obj_key]
            category, name = object_key_to_category_and_name(obj_key)
            if category in OBJECTS_TO_EXCLUDE_FROM_VLM or name in OBJECTS_TO_EXCLUDE_FROM_VLM:
                print(f"Skipping excluded object (liquid/fixed): {obj_key} ({category}/{name})", flush=True)
                results[obj_key] = []
                continue
            results[obj_key] = []

            for seg in seg_list:
                done += 1
                start_time = seg.get("start_time")
                end_time = seg.get("end_time")
                print(f"[{done}/{total_segments}] {obj_key} segment {start_time}-{end_time}", flush=True)
                event_history_lines, event_frame_images_base64 = format_event_history(
                    seg.get("event_history", []), video_path
                )
                crop_images_base64 = get_crop_images_base64(video_path, seg)
                user_content = build_user_message(category, name)
                if not crop_images_base64:
                    results[obj_key].append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "error": "no crops extracted",
                        "parse_error": True,
                    })
                    debug_jsonl.write(json.dumps({
                        "query_object": obj_key,
                        "object_name": name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "error": "no crops extracted",
                        "llm_response": None,
                    }) + "\n")
                    continue
                out = query_ollama(args.model, user_content, crop_images_base64, event_history_lines, event_frame_images_base64)
                # Parse message.content when format=json (Ollama returns content in message.content)
                reasoning, is_passive_usage = None, None
                raw_content = None
                if out:
                    msg = out.get("message") if isinstance(out, dict) else getattr(out, "message", None)
                    if msg:
                        raw_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                        if raw_content and isinstance(raw_content, str):
                            try:
                                parsed = json.loads(raw_content)
                                reasoning = parsed.get("reasoning")
                                is_passive_usage = parsed.get("is_passive_usage")
                            except json.JSONDecodeError:
                                pass
                    if reasoning is None and is_passive_usage is None:
                        reasoning = out.get("reasoning") if isinstance(out, dict) else getattr(out, "reasoning", None)
                        is_passive_usage = out.get("is_passive_usage") if isinstance(out, dict) else getattr(out, "is_passive_usage", None)
                try:
                    results[obj_key].append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "reasoning": reasoning,
                        "is_passive_usage": is_passive_usage,
                    })
                    debug_record = {
                        "query_object": obj_key,
                        "object_name": name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "reasoning": reasoning,
                        "is_passive_usage": is_passive_usage,
                        **_ollama_response_metadata(out),
                    }
                    debug_jsonl.write(json.dumps(debug_record) + "\n")
                except Exception as e:
                    results[obj_key].append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "error": str(e),
                        "parse_error": True,
                    })
                    debug_record = {
                        "query_object": obj_key,
                        "object_name": name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "error": str(e),
                        **_ollama_response_metadata(out),
                    }
                    debug_jsonl.write(json.dumps(debug_record) + "\n")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_path}")
    print(f"Wrote debug JSONL {debug_jsonl_path}")


if __name__ == "__main__":
    main()
