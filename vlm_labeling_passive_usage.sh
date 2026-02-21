#!/bin/bash
#SBATCH --job-name=query_passive_usage_vlm
#SBATCH --partition=rail-lab
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

VIDEO_PATHS_FILE="video_paths.txt"

MODEL_NAME="qwen3-vl:30b"

while read -r VIDEO_PATH; do
    echo "Processing video: $VIDEO_PATH"
    CMD="$PYTHONPATH -u query_passive_usage_vlm.py \
        --video-path $VIDEO_PATH \
        --model $MODEL_NAME \
        --segments-dir inactive_segments \
        --output-dir vlm_annotations"
    echo "Running command: $CMD"
    eval $CMD
done < $VIDEO_PATHS_FILE