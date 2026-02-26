#!/usr/bin/env python3
"""
Query a VLM (via Ollama) to label passive usage for each inactive segment.
Reads inactive segments from disk, extracts object crops from the video,
sends prompt (system + object category/name + sequence of crops + event history)
and saves JSON responses.

Usage:
    python query_passive_usage_vlm.py --video-path /path/to/P01_01.MP4 [options]
"""

import csv
import os
import sys
import argparse
import base64
import json
from math import ceil, floor
import numpy as np
from datetime import datetime
import cv2

import ollama
import pdb

from utils import load_noun_class_names
from prompts_usage_labeling import system_prompt
from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from label_inactive_segments_manual import (
    extract_frames_full_res,
    extract_object_crops,
    parse_crop_specs,
)

MODEL_NAME="qwen3-vl:30b"
TEMPERATURE=0.8
MAX_NUM_PREDICT=2000
NUM_TRIES=3

# Inactive-segment image sampling for VLM
MAX_IMAGES_INACTIVE = 1
MIN_SPACING_SEC = 3.0
MIN_DURATION = 6.0  # Ignore inactive segments shorter than this (seconds)

# Qwen3 VL (and some other VLMs) require image dimensions > 32 (e.g. "factor:32").
# Upscale any smaller image so both height and width exceed this.
OLLAMA_MIN_IMAGE_DIM = 33

NOUN_CLASS_NAMES = {}

def sample_frames(video_cap, start_time, end_time, fps, min_spacing_sec, max_images_inactive):
    """Uniformly sample frames between start time and end time, atleast min_spacing_sec apart.
    """
    start_frame = ceil(start_time * fps)
    end_frame = floor(end_time * fps)
    segment_length_frames = end_frame - start_frame

    if segment_length_frames < MIN_SPACING_SEC * fps:
        frame_ids = [start_frame, end_frame]
    else:
        frame_ids = list(range(start_frame, end_frame, int(min_spacing_sec * fps)))
    if len(frame_ids) > max_images_inactive:
        frame_ids = frame_ids[:max_images_inactive]
    frames = extract_frames_full_res(frame_ids, cap=video_cap)
    return frame_ids, [image_to_base64(frames[fid]) for fid in sorted(frames.keys())]


def load_video_fps(video_info_csv_path, video_id):
    """
    Load FPS for the given video_id from EPIC_100_video_info.csv.
    Returns float FPS or None if not found. CSV columns: video_id, duration, fps, resolution.
    """
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
    """Encode an RGB image (numpy array) to base64 PNG string for Ollama, ensuring dimensions are > OLLAMA_MIN_IMAGE_DIM."""
    if rgb_array is None or rgb_array.size == 0:
        return None
    h, w = rgb_array.shape[:2]
    if h < OLLAMA_MIN_IMAGE_DIM or w < OLLAMA_MIN_IMAGE_DIM:
        ## Upscale to OLLAMA_MIN_IMAGE_DIM
        scale = OLLAMA_MIN_IMAGE_DIM / float(min(h, w))
        new_w = OLLAMA_MIN_IMAGE_DIM
        new_h = int(round(h * scale))
        rgb_array = cv2.resize(rgb_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # OpenCV imencode writes channels as-is; we have RGB, PNG viewers expect RGB
    ok, buf = cv2.imencode(".png", rgb_array)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


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


def format_event_history(event_history, video_cap):
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
    frames_dict = extract_frames_full_res(frame_ids, cap=video_cap) if frame_ids else {}
    for item in event_history:
        if item.startswith("narration:"):
            action = item.split("narration:", 1)[1].strip()
            event_lines.append(f"Action: {action}")
            event_frame_images_base64.append([])
        elif item.startswith("frame_id:"):
            frame_id = int(item.split("frame_id:", 1)[1])
            frame = frames_dict.get(frame_id)
            if frame is not None and frame.size > 0:
                b64 = image_to_base64(frame)
                if b64 is not None:
                    event_frame_images_base64[-1].append(b64)
            else:
                print("Warning: Failed to extract frame.")
    assert len(event_lines) == len(event_frame_images_base64)
    return event_lines, event_frame_images_base64


def build_user_message(category, subclass_name, name):
    """Build user message in required order: object category/name, then line about images, then event history (unchanged order)."""
    lines = [
        f"Object category: {category}",
        f"Sub-category: {subclass_name}",
        f"Description: {name}",
        "",
        "Images of the object",
    ]
    return "\n".join(lines)


def get_crop_images_base64(video_path, segment, video_cap=None):
    """Extract sequence of object crops for segment and return list of base64 PNG strings.
    video_path: path to video file (str). video_cap: optional open cv2.VideoCapture to reuse."""
    crop_specs = parse_crop_specs(segment.get("crop_from_previous_active"))
    if not crop_specs:
        return [], []
    crops = extract_object_crops(video_path, crop_specs, cap=video_cap)
    return [image_to_base64(c) for c in crops if image_to_base64(c) is not None], crop_specs


def query_ollama(
    model,
    user_content,
    crop_images_base64,
    event_history_lines,
    event_frame_images_base64,
    post_active_content=None,
    post_active_images_b64=[],
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

    if post_active_content and post_active_images_b64:
        messages.append({
            "role": "user",
            "content": post_active_content,
            "images": post_active_images_b64,
        })

    # DEBUGGING
    print("=== Messages to be sent to Ollama ===")
    for idx, msg in enumerate(messages):
        content = msg.get("content")
        images = msg.get("images", [])
        images_len = len(images) if images is not None else 0
        print(f"Message {idx}:")
        print(f"  Content: {repr(content)}")
        print(f"  Number of images: {images_len}")
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
    global NOUN_CLASS_NAMES
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
    parser.add_argument("--max-images-inactive", type=int, default=MAX_IMAGES_INACTIVE, help="Maximum number of images to sample per segment (default: 5)")
    parser.add_argument("--min-spacing-sec", type=float, default=MIN_SPACING_SEC, help="Minimum spacing between images in seconds (default: 3.0)")
    parser.add_argument(
        "--output-dir",
        default="vlm_annotations",
        help="Directory to write labels JSON (default: vlm_passive_usage_labels)",
    )
    parser.add_argument(
        "--video-info-csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "EPIC_100_video_info.csv"),
        help="Path to EPIC_100_video_info.csv for video FPS",
    )
    args = parser.parse_args()

    ensure_ollama_model_loaded(args.model)

    NOUN_CLASS_NAMES = load_noun_class_names()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    fps = load_video_fps(args.video_info_csv, video_id)
    if fps is None:
        print(f"Warning: FPS not found for {video_id} in {args.video_info_csv}; will use single frame per segment.", file=sys.stderr)
    else:
        print(f"Using FPS={fps} for {video_id} (from {args.video_info_csv})")
    segments_file = os.path.join(args.segments_dir, f"inactive_segments_{video_id}.json")
    if not os.path.isfile(segments_file):
        print(f"Error: Segments file not found: {segments_file}", file=sys.stderr)
        sys.exit(1)

    with open(segments_file) as f:
        segments_data = json.load(f)
    total_segments = sum([len(segments_data[k]) for k in segments_data])

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"inactive_segments_{video_id}_labels.jsonl")
    input_prompt_jsonl_path = os.path.join(args.output_dir, f"input_prompt_{video_id}.jsonl")
    debug_jsonl_path = os.path.join(args.output_dir, f"debug_vlm_{video_id}.jsonl")

    segments_saved = []
    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            segments_saved = [json.loads(line) for line in f]
        print(f"Resuming from {len(segments_saved)}/{total_segments} segments in {output_path}")

    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(debug_jsonl_path, "a") as debug_jsonl:
        debug_jsonl.write(json.dumps({"date_time": date_time_str}) + "\n")
    with open(output_path, "a") as output_jsonl:
        output_jsonl.write(json.dumps({"date_time": date_time_str}) + "\n")
    with open(input_prompt_jsonl_path, "a") as input_prompt_jsonl:
        input_prompt_jsonl.write(json.dumps({"date_time": date_time_str}) + "\n")

    done = len(segments_saved)

    try:
        video_cap = cv2.VideoCapture(str(args.video_path)) ## Open video file
    except Exception as e:
        print(f"Error: Failed to open video file: {args.video_path} - {e}", file=sys.stderr)
        sys.exit(1)
    for obj_key in sorted(segments_data.keys()):
        seg_list = segments_data[obj_key]
        category, subclass_name, name = object_key_to_category_and_name(obj_key)
        if subclass_name in OBJECTS_TO_EXCLUDE_FROM_VLM:
            print(f"Skipping excluded object (liquid/fixed): {obj_key} ({subclass_name}/{name})", flush=True)
            continue

        print(f"Processing object: {obj_key} ({category}/{subclass_name}/{name})", flush=True)

        ## Get segments saved for this object (from previous runs)
        segments_saved_this_obj = []
        if segments_saved:
            segments_saved_this_obj = [seg_saved for seg_saved in segments_saved if seg_saved.get("query_object", "") == obj_key]

        for seg in seg_list:
            ## Skip segments shorter than MIN_DURATION
            duration_sec = seg.get("duration_sec")
            if duration_sec is None:
                duration_sec = seg.get("end_time", 0.0) - seg.get("start_time", 0.0)
            if duration_sec < MIN_DURATION:
                print(f"Skipping short segment {seg.get('start_time')}-{seg.get('end_time')} for {obj_key} (duration={duration_sec:.2f}s < {MIN_DURATION}s)", flush=True)
                continue
            ## Skip if already processed
            if segments_saved_this_obj:
                matched_segments = [
                    seg_saved for seg_saved in segments_saved_this_obj
                    if seg_saved.get("start_time", 0.0) == seg.get("start_time")
                    and seg_saved.get("end_time", 0.0) == seg.get("end_time")
                    
                ]
                if matched_segments:
                    if any(
                        seg_matched.get("reasoning") and (seg_matched.get("is_passive_usage") in [True, False])
                        for seg_matched in matched_segments
                    ):
                        ## Skip if already processed and has valid reasoning & is_passive_usage.
                        print(f"Skipping already processed segment {seg.get('start_time')}-{seg.get('end_time')} for {obj_key}", flush=True)
                        continue
                    else:
                        print(f"Segment {seg.get('start_time')}-{seg.get('end_time')} for {obj_key} has invalid reasoning & is_passive_usage, reprocessing...", flush=True)
                        for seg_matched in matched_segments:
                            print(seg_matched.get("start_time"), seg_matched.get("end_time"), seg_matched.get("reasoning"), seg_matched.get("is_passive_usage"), type(seg_matched.get("is_passive_usage")))
                        print("--------------------------------")
            ## Process segment
            print(f"Processing segment {seg.get('start_time')}-{seg.get('end_time')} for {obj_key}", flush=True)
            start_time = seg.get("start_time")
            end_time = seg.get("end_time")
            print(f"[{done}/{total_segments}] {obj_key} segment {start_time}-{end_time}", flush=True)

            ## ------- PROMPT CONSTRUCTION -------
            ## Build user content
            user_content = build_user_message(category, subclass_name, name)
            ## Extract crop images
            crop_images_base64, crop_specs = get_crop_images_base64(args.video_path, seg, video_cap=video_cap)
            ## Format event history
            event_history_lines, event_frame_images_base64 = format_event_history(
                seg.get("event_history", []), video_cap
            )
            ## Sample MAX_IMAGES_INACTIVE images between start time and end time, MIN_SPACING_SEC apart
            post_active_content = "Sequence of images following the period of active usage (after the user's last action)"
            post_active_frame_ids, post_active_images_b64 = sample_frames(video_cap, start_time, end_time, fps, args.min_spacing_sec, args.max_images_inactive)

            ## ------- WRITE INPUT PROMPT TO JSONL FILE -------
            with open(input_prompt_jsonl_path, "a") as input_prompt_jsonl:
                input_prompt_jsonl.write(json.dumps({
                    "query_object": obj_key,
                    "start_time": start_time,
                    "end_time": end_time,
                    "user_content": user_content,
                    "video_path": args.video_path,
                    "crop_specs": json.dumps(crop_specs),
                    "event_history": seg.get("event_history", []),
                    "post_active_content": post_active_content,
                    "post_active_frame_ids": post_active_frame_ids
                }) + "\n")
            ## Skip if no crops extracted
            if not crop_images_base64:
                with open(debug_jsonl_path, "a") as debug_jsonl:
                    debug_jsonl.write(json.dumps({
                        "query_object": obj_key,
                        "start_time": start_time,
                        "end_time": end_time,
                        "error": "no crops extracted",
                        "llm_response": None,
                    }) + "\n")
                continue

            ## ------- QUERY OLLAMA -------
            out = query_ollama(args.model, user_content, crop_images_base64, event_history_lines, event_frame_images_base64, post_active_content, post_active_images_b64)
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

            # DEBUGGING
            print(reasoning)
            print(is_passive_usage)

            ## ------- WRITE RESULT TO OUTPUT JSONL FILE -------
            try:
                result = {
                    "query_object": obj_key,
                    "start_time": start_time,
                    "end_time": end_time,
                    "reasoning": reasoning,
                    "is_passive_usage": is_passive_usage,
                }
                debug_record = {
                    "query_object": obj_key,
                    "object_name": name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "reasoning": reasoning,
                    "is_passive_usage": is_passive_usage,
                    **_ollama_response_metadata(out),
                }
            except Exception as e:
                result = {
                    "query_object": obj_key,
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": str(e),
                    "parse_error": True,
                }
                debug_record = {
                    "query_object": obj_key,
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": str(e),
                    **_ollama_response_metadata(out),
                }

            ## ------- WRITE OUTPUT AND DEBUG RECORD TO JSONL FILES -------
            with open(debug_jsonl_path, "a") as debug_jsonl:
                debug_jsonl.write(json.dumps(debug_record) + "\n")
            with open(output_path, "a") as output_jsonl:
                output_jsonl.write(json.dumps(result) + "\n")
            print(f"Finished processing {obj_key} segment {start_time}-{end_time}")
            done += 1

    video_cap.release()


if __name__ == "__main__":
    main()
