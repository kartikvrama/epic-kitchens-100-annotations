import os
import csv
import shutil
import numpy as np
import statistics

import argparse

VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
DATASET_FOLDERS = ["/coc/flash5/kvr6/data/epic_kitchens_100_hf", "/coc/flash5/kvr6/data/epic_kitchens_100_hf_V2"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination_folder", type=str, help="Path to the destination folder")
    args = parser.parse_args()

    video_info = csv.DictReader(open(VIDEO_INFO_FILE))
    video_info = list(video_info)

    ek100_videos_longer_than_15_minutes = [v for v in video_info if float(v["duration"]) >= 900]

    files = [f for f in os.listdir("active_objects") if f.endswith(".json")]
    video_ids_visor = ["_".join(f.split(".")[0].split("_")[2:]) for f in files]
    visor_videos_longer_than_15_minutes = [v for v in ek100_videos_longer_than_15_minutes if v['video_id'] in video_ids_visor]

    ## Print min, max, mean, median, mode of video durations
    print(f"Total number of videos: {len(video_info)}")
    print(f"Number of visor annotated videos longer than 15 minutes: {len(visor_videos_longer_than_15_minutes)}")
    print(f"Min: {min(visor_videos_longer_than_15_minutes, key=lambda x: float(x['duration']))}")
    print(f"Max: {max(visor_videos_longer_than_15_minutes, key=lambda x: float(x['duration']))}")
    print(f"Mean: {np.mean([float(v['duration']) for v in visor_videos_longer_than_15_minutes])}")
    print(f"Median: {statistics.median(float(v['duration']) for v in visor_videos_longer_than_15_minutes)}")

    ## Find the video paths and write to output file
    ## Video path is in the format of /coc/flash5/kvr6/data/epic_kitchens_100_hf/P01/videos/P01_14.MP4
    video_ids_found = []
    video_paths = []
    for dataset_folder in DATASET_FOLDERS:
        video_paths.extend([(v['video_id'], f"{dataset_folder}/{v['video_id'].split('_')[0]}/videos/{v['video_id']}.MP4") for v in visor_videos_longer_than_15_minutes])

    for video_id, path in video_paths:
        print(f"Checking video path: {path}")
        if os.path.exists(path):
            if video_id not in video_ids_found:
                video_ids_found.append(video_id)
            else:
                print(f"Video path repeated: {video_id}")

    videos_ids_not_found = [v['video_id'] for v in visor_videos_longer_than_15_minutes if v['video_id'] not in video_ids_found]

    ## Print the number of video IDs found and not found
    print(f"Number of video IDs found: {len(video_ids_found)}")
    print(f"Number of video IDs not found: {len(videos_ids_not_found)}")

    os.makedirs(args.destination_folder, exist_ok=True)
    with open("video_paths_updated.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "path_updated", "path_original"])
        writer.writeheader()
        data_to_write = []
        for video_id, path_original in video_paths:
            path_destination = os.path.join(args.destination_folder, os.path.basename(path_original))
            try:
                if os.path.exists(path_original):
                    print(f"Copying {path_original} to {path_destination}...")
                    shutil.copy2(path_original, path_destination)
                    data_to_write.append({"video_id": video_id, "path_updated": path_destination, "path_original": path_original})
            except Exception as e:
                print(f"Error copying {path_original} to {path_destination}: {e}")
        writer.writerows(data_to_write)


if __name__ == "__main__":
    main()