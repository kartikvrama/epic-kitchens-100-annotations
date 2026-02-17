import os
import json
import unicodedata
import re
from copy import deepcopy
from utils import seconds_to_minutes_seconds

def format_scene_graph(scene_graph: dict, show_empty: bool = False) -> str:
    """Format a scene graph dictionary into a human-readable string."""
    lines = []
    # Sort nodes for consistent output
    sorted_nodes = sorted(scene_graph.keys())
    for node in sorted_nodes:
        objects = scene_graph[node]
        if not objects and not show_empty:
          ## Skip nodes with no objects if show_empty is False (default)
            continue
        objects_str = ", ".join(sorted(objects))
        lines.append(f"  {node}: [{objects_str}]")
    return "\n".join(lines)


def normalize_text(text):
    """
    Normalize text to avoid encoding errors by replacing problematic Unicode characters.
    Handles em-dash, en-dash, and other common problematic characters.
    """
    if not isinstance(text, str):
        return text
    
    # Normalize Unicode characters (NFD normalization helps with composed characters)
    text = unicodedata.normalize('NFD', text)
    
    # Replace common problematic Unicode characters with ASCII equivalents
    replacements = {
        '\u2011': '-',  # non-breaking hyphen
        '\u2013': '-',  # en-dash
        '\u2014': '-',  # em-dash
        '\u2015': '-',  # horizontal bar
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201C': '"',  # left double quotation mark
        '\u201D': '"',  # right double quotation mark
        '\u2026': '...',  # horizontal ellipsis
        '\u00A0': ' ',  # non-breaking space
        '\u200B': '',   # zero-width space
        '\u200C': '',   # zero-width non-joiner
        '\u200D': '',   # zero-width joiner
        '\uFEFF': '',   # zero-width no-break space (BOM)
    }
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove any remaining non-printable control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    return text


def _extract_event_history(scene_graphs, mask_info_dict, query_object_name, long=False):
    result = {
        "object_name": query_object_name,
        "event_history": [],
    }

    # import pdb; pdb.set_trace()
    for scene_graph in scene_graphs:
        if scene_graph["action"] == "INITIAL":
            continue
        if scene_graph['mask_id'] in mask_info_dict:
            if mask_info_dict[scene_graph['mask_id']]['fixture'] is None:
                fixture_name = "unknown"
            else:
                fixture_name = mask_info_dict[scene_graph['mask_id']]['fixture']
        else:
            fixture_name = "unknown"
        event = {
            "time": scene_graph['time'],
            "time_str": seconds_to_minutes_seconds(scene_graph['time']),
            "high_level_activity": scene_graph['high_level_activity']['high_level_activity_label'],
            "action_narrations": [narration['narration'] for narration in scene_graph['narrations']],
            "action": scene_graph['action'],
            "object": scene_graph['object_name'],
            "fixture": fixture_name.split("_")[1] if "_" in fixture_name else fixture_name,
        }
        ## Get state before the action
        event['objects_in_hand'] = set(scene_graph['scene_graph'].get("Human"))
        if scene_graph['action'] == "PICK":
            ## Pick up: remove object from hand, add to fixture
            event['objects_in_hand'] = event['objects_in_hand'].difference(set([scene_graph['object_name']]))
            event['nearby_objects_fixture'] = set(scene_graph['scene_graph'].get(fixture_name)).union(set([scene_graph['object_name']])) if fixture_name in scene_graph["scene_graph"] else set()
        elif scene_graph['action'] == "DROP":
            ## Drop: add object to hand, remove from fixture
            event['objects_in_hand'] = event['objects_in_hand'].union(set([scene_graph['object_name']]))
            event['nearby_objects_fixture'] = set(scene_graph['scene_graph'].get(fixture_name)).difference(set([scene_graph['object_name']])) if fixture_name in scene_graph["scene_graph"] else set()
        else:
            raise ValueError(f"Invalid action: {scene_graph['action']}")
        event['nearby_objects_fixture'] = list(event['nearby_objects_fixture'])
        event['objects_in_hand'] = list(event['objects_in_hand'])
        
        # Include full scene graph if long flag is set
        if long:
            ## Human and Free Space must come first
            human_and_free_space = {v: scene_graph['scene_graph'][v] for v in ["Human", "Free Space"] if v in scene_graph['scene_graph']}
            fixture_name_dict = {v: v.split("_")[1] if "_" in v else v for v in sorted(scene_graph['scene_graph']) if v not in ["Human", "Free Space"]}
            event['full_scene_graph'] = {**human_and_free_space, **{v_name: scene_graph['scene_graph'][v] for v, v_name in fixture_name_dict.items()}}
        result['event_history'].append(event)
    return result


def generate_prompts_for_video(video_id, max_segment_length=120, long=False):
    """
    Generate prompts for a given video ID.
    
    Args:
        video_id: Video ID to process (e.g., P01-20240202-171220)
        max_segment_length: Maximum segment length in seconds (default: 120)
        long: If True, include full scene graph in prompts (default: False)
    
    Returns:
        List of prompt info dictionaries
    """
    with open("scene-and-object-movements/assoc_info.json", encoding='utf-8') as f:
        object_movement_dict = json.load(f)

    with open("scene-and-object-movements/mask_info.json", encoding='utf-8') as f:
        mask_info_dict = json.load(f)

    with open(f"outputs/scene_graphs/scene_graphs_{video_id}.jsonl", "r") as f:
        scene_graphs = [json.loads(line) for line in f]

    long_suffix = "_long" if long else ""
    output_filename = f"outputs/prompts/prompt_info_{video_id}_max_segment_length_{max_segment_length}{long_suffix}.json"
    if os.path.exists(output_filename):
        ## Delete the file
        os.remove(output_filename)
        print(f"Deleted existing file: {output_filename}")

    prompt_info = []
    for assoc_id, assoc_data in object_movement_dict[video_id].items():
        # import pdb; pdb.set_trace()
        object_name = assoc_data['name']
        if "skipped" in object_name:
            print(f"Skipping object: {object_name}")
            continue
        timesteps = [0] + [scene_graph['time'] for scene_graph in scene_graphs if scene_graph["object_name"] == object_name] + [scene_graphs[-1]["time"]]

        ## Print event history for consecutive timesteps
        segment_categories = ["passive", "active"]
        for i in range(len(timesteps) - 1):
            timestep_1 = timesteps[i]
            timestep_2 = timesteps[i + 1]
            segment_length = timestep_2 - timestep_1

            # Split segment if it's longer than max_segment_length
            if segment_length <= max_segment_length:
                # Segment is short enough, process as is
                scene_graphs_between_timesteps = [scene_graph for scene_graph in scene_graphs if scene_graph['time'] >= timestep_1 and scene_graph['time'] <= timestep_2]
                result = _extract_event_history(scene_graphs_between_timesteps, mask_info_dict[video_id], object_name, long=long)
                result["time_start"] = timestep_1
                result["time_end"] = timestep_2
                result["segment_category"] = segment_categories[i%2]
                # print(result["time_start"], result["time_end"], object_name)
                prompt_info.append(result)
            else:
                # Split segment into chunks of max_segment_length
                num_splits = int(segment_length / max_segment_length) + (1 if segment_length % max_segment_length > 0 else 0)
                merge_last_chunk = False
                for split_idx in range(num_splits):
                    split_start = timestep_1 + split_idx * max_segment_length
                    split_end = min(timestep_1 + (split_idx + 1) * max_segment_length, timestep_2)
                    if timestep_2 - split_end < max_segment_length//2: ## If the last chunk is less than half of the max_segment_length, merge it with the previous chunk
                        split_end = timestep_2
                        merge_last_chunk = True
                    scene_graphs_between_timesteps = [scene_graph for scene_graph in scene_graphs if scene_graph['time'] >= split_start and scene_graph['time'] <= split_end]
                    result = _extract_event_history(scene_graphs_between_timesteps, mask_info_dict[video_id], object_name, long=long)
                    result["time_start"] = split_start
                    result["time_end"] = split_end
                    result["segment_category"] = segment_categories[i%2]
                    # print(result["time_start"], result["time_end"], object_name)
                    prompt_info.append(result)
                    if merge_last_chunk:
                        break

    os.makedirs("outputs/prompts", exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(prompt_info, f, indent=1)
    
    return prompt_info


def _recover_scene_graph_before_action(scene_graph, action, object_name, fixture):
    """
    Recover the scene graph before the action.
    
    Args:
        scene_graph: Scene graph after the action
        action: Action type ("PICK" or "DROP")
        object: Object name
        fixture: Fixture name
    """
    # Make a deep copy of the scene graph so as not to mutate the original
    pre_action_graph = deepcopy(scene_graph)

    if action in ("PICK", "DROP") and object_name is not None and fixture is not None:
        # To get graph before the action, PICK: remove object from Human, add to 'fixture'
        if action == "PICK":
            assert 'Human' in pre_action_graph, f"Human not in scene graph: {pre_action_graph}"
            if object_name not in pre_action_graph['Human']:
                # print(f"Object {object_name} not in Human scene graph: {pre_action_graph}")
                # import pdb; pdb.set_trace()
                assert object_name in pre_action_graph['Human'], f"Object {object_name} not in Human scene graph: {pre_action_graph}"   
            pre_action_graph['Human'].remove(object_name)

            if fixture != "unknown":
                assert fixture in pre_action_graph, f"Fixture {fixture} not in scene graph: {pre_action_graph}"
                assert object_name not in pre_action_graph[fixture], f"Object {object_name} already in fixture {fixture} scene graph: {pre_action_graph}"

            if fixture not in pre_action_graph:
                pre_action_graph[fixture] = []
            if object_name not in pre_action_graph[fixture]:
                pre_action_graph[fixture].append(object_name)

        # To get graph before the action, DROP: remove from 'fixture', add to 'Human'
        elif action == "DROP":
            if fixture != "unknown":
                assert fixture in pre_action_graph, f"Fixture {fixture} not in scene graph: {pre_action_graph}"
                assert object_name in pre_action_graph[fixture], f"Object {object_name} not in fixture {fixture} scene graph: {pre_action_graph}"

            if fixture not in pre_action_graph:
                pre_action_graph[fixture] = []
            if object_name in pre_action_graph[fixture]:
                pre_action_graph[fixture].remove(object_name)

            assert 'Human' in pre_action_graph, f"Human not in scene graph: {pre_action_graph}"

            if object_name in pre_action_graph['Human']:
                # print(f"Object {object_name} already in Human scene graph: {pre_action_graph}")
                # import pdb; pdb.set_trace()
                assert object_name in pre_action_graph['Human'], f"Object {object_name} not in Human scene graph: {pre_action_graph}"
            pre_action_graph['Human'].append(object_name)

    return pre_action_graph


def format_event_history(event_history, show_empty: bool = False):
    """
    Format event history into a readable string for the prompt.
    
    Args:
        event_history: List of event dictionaries
        
    Returns:
        Formatted string representation of event history
    """
    lines = []
    for event in event_history:
        event_lines = []
        event_lines.append(f"Time: {event['time_str']} ({event['time']:.2f}s)")
        event_lines.append(f"High-level task being performed: {event['high_level_activity']}")
        
        if event.get('action_narrations'):
            narrations = event['action_narrations']
            if narrations:
                event_lines.append("Current scene narration:")
                for narration in narrations:
                    event_lines.append(f"  - {narration}")
        
        # import pdb; pdb.set_trace()
        # If full scene graph is available (long mode), use it instead of just fixture-specific objects
        if event.get('full_scene_graph'):
            # INSERT_YOUR_CODE
            # full_scene_graph is the scene graph after the action; recover scene graph before the action
            event_lines.append("Object locations before human action:")
            pre_action_graph = _recover_scene_graph_before_action(event['full_scene_graph'], event['action'], event['object'], event['fixture'])
            formatted_graph_before = format_scene_graph(pre_action_graph, show_empty=show_empty)
            if formatted_graph_before:
                event_lines.append(formatted_graph_before)
            else:
                event_lines.append("  (empty scene graph)")

            # formatted_graph = format_scene_graph(event['full_scene_graph'], show_empty=show_empty)
            # if formatted_graph:
            #     event_lines.append(formatted_graph)
            # else:
            #     event_lines.append("  (empty scene graph)")

        else:
            # Default behavior: show objects in hand and at fixture
            if event.get('objects_in_hand'):
                event_lines.append(f"Objects currently in hand: {', '.join(event['objects_in_hand'])}")
            else:
                event_lines.append("Objects currently in hand: []")
            
            if event.get('nearby_objects_fixture'):
                event_lines.append(f"Objects currently at `{event['fixture']}`: {', '.join(event['nearby_objects_fixture'])}")
            else:
                event_lines.append(f"Objects currently at `{event['fixture']}`: []")

        action = "pick up" if event['action'] == "PICK" else "put down" if event['action'] == "DROP" else event['action']
        action_event = f"{action} `{event['object']}` from `{event['fixture']}`" if event['action'] == "PICK" else f"{action} `{event['object']}` to `{event['fixture']}`"
        event_lines.append(f"Human atomic action: {action_event}")
        
        lines.append("\n".join(event_lines))
        lines.append("")  # Empty line between events
    
    return "\n".join(lines)


def generate_system_prompt():
    """
    Generate the system prompt for object usage labeling.
    
    Returns:
        System prompt string
    """
    system_prompt = """You are an expert in analyzing kitchen activities to determine if an object is being used during a specific time period. Your task is to determine whether a given object is being used during a specified time period, along with a clear explanation of your reasoning.
    
You will be given a history of events that occurred during the time period. The events include:
    - High-level activity
    - Low-level action narrations
    - Current object locations (e.g., in the person's hand, on the countertop, etc.)
    - Atomic actions such as picking up and placing down

An object is considered 'being used' if it is contributing to the high-level activity, either by actively being held by the person or passively performing a function as part of the high-level activity.
Analyze the evidence step-by-step before providing your final answer, and provide a clear explanation of your reasoning. Think through the following questions:
    1. If the object is in the person's hand during this period, is the person using this object to perform the high-level activity? Otherwise, it is not being used.
    2. If the object is not in the person's hand during this period, is the object meaningfully contributing to the task being performed? Otherwise, it is not being used.

Provide your analysis using Chain of Thought reasoning, and respond with the following JSON structure:""" + """
{
  'is_used': true/false,
  'explanation': 'Step-by-step Chain of Thought reasoning explaining your decision...'
}"""
    
    return normalize_text(system_prompt)
