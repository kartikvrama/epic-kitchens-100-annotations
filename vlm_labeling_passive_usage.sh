#!/bin/bash
#SBATCH --job-name=query_passive_usage_vlm
#SBATCH --partition=overcap
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=a40:1
#SBATCH --qos=long
#SBATCH --output=logs_slurm/R-%x.%j.out
#SBATCH --error=logs_slurm/R-%x.%j.err
#SBATCH --exclude=xaea-12

cd /coc/flash5/kvr6/repos
OLLAMA_MODELS=/coc/flash5/kvr6/repos/ollama_models ./ollama/bin/ollama serve&
export PYTHONPATH=/coc/flash5/kvr6/containers/envs/llmEnv/bin/python
cd /coc/flash5/kvr6/repos/epic-kitchen-repos/epic-kitchens-100-annotations

# CSV with columns: video_id, path_updated, path_original
VIDEO_PATHS_CSV="${1:-video_paths_updated.csv}"

MODEL_NAME="qwen3-vl:30b"

MAX_IMAGES_INACTIVE=1
MIN_SPACING_SEC=3

# Skip header, use path_updated column (2nd field)
tail -n +2 "$VIDEO_PATHS_CSV" | cut -d',' -f2 | while read -r VIDEO_PATH; do
    # Skip empty lines
    [ -z "$VIDEO_PATH" ] && continue
    echo "Current time: $(date +%H:%M:%S)"
    echo "Processing video: $VIDEO_PATH"
    CMD="$PYTHONPATH -u query_passive_usage_vlm.py \
        --video-path \"$VIDEO_PATH\" \
        --model $MODEL_NAME \
        --segments-dir inactive_segments \
        --output-dir vlm_annotations_maxImages${MAX_IMAGES_INACTIVE}_minSpacing${MIN_SPACING_SEC} \
        --max-images-inactive $MAX_IMAGES_INACTIVE \
        --min-spacing-sec $MIN_SPACING_SEC"
    echo "Running command: $CMD"
    eval $CMD
done