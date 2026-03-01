import os
import csv
import json
import numpy as np
from utils import load_noun_class_names
from utils import ACTIVE_OBJECTS_DIR
from objects_to_exclude_vlm import OBJECTS_TO_EXCLUDE_FROM_VLM
from collections import defaultdict

NOUN_CLASS_NAMES = load_noun_class_names()

fractions_subclasses_with_more_than_one_name = []

with open("video_paths_updated.csv", "r") as f:
    reader = csv.reader(f)
    video_id_list = [row[0] for row in reader][1:]

for file in os.listdir(ACTIVE_OBJECTS_DIR):
    if not (file.startswith("active_objects_") and file.endswith(".json")):
        continue

    video_id = file.split("active_objects_")[1].split(".")[0]
    if video_id not in video_id_list:
        continue


    print(f"Processing {file}")
    sub_name_dict = defaultdict(set)
    with open(os.path.join(ACTIVE_OBJECTS_DIR, file), 'r') as f:
        data = json.load(f)
        for segment in data:
            for object in segment['objects_in_sequence']:
                if object['subclass_name'] in OBJECTS_TO_EXCLUDE_FROM_VLM:
                    continue
                sub_name_dict[object['subclass_name']].add(object['name'])


    # # Get number of unique names per subclass
    # print(f"Video ID: {video_id}")
    # for subclass, names in sub_name_dict.items():
    #     print(f"{subclass}: {names}")

    ## Get number and fraction of subclasses with more than one name
    num_subclasses_with_more_than_one_name = sum(1 for names in sub_name_dict.values() if len(names) > 1)
    fraction_subclasses_with_more_than_one_name = num_subclasses_with_more_than_one_name / len(sub_name_dict)
    fractions_subclasses_with_more_than_one_name.append(fraction_subclasses_with_more_than_one_name)


    print(f"Number of subclasses with more than one name: {num_subclasses_with_more_than_one_name}")
    print(f"Fraction of subclasses with more than one name: {fraction_subclasses_with_more_than_one_name}")
    print()

## Print summary statistics
print(f"Mean fraction of object subclasses with multiple instances: {np.mean(fractions_subclasses_with_more_than_one_name)}")
print(f"Std fraction of object subclasses with multiple instances: {np.std(fractions_subclasses_with_more_than_one_name)}")
print()
print(f"Min fraction of object subclasses with multiple instances: {np.min(fractions_subclasses_with_more_than_one_name)}")
print(f"Max fraction of object subclasses with multiple instances: {np.max(fractions_subclasses_with_more_than_one_name)}")
