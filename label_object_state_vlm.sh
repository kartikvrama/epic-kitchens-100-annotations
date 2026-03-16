#!/bin/bash
#SBATCH --job-name=label_object_state_vlm
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=a40:1
#SBATCH --qos=long
#SBATCH --output=logs_slurm/R-%x.%j.out
#SBATCH --error=logs_slurm/R-%x.%j.err

cd /coc/flash5/kvr6/repos

OLLAMA_LOG_FILE="/coc/flash5/kvr6/repos/ollama.log"
OLLAMA_PID_FILE="/coc/flash5/kvr6/repos/ollama.pid"

# Clean up Ollama on exit (normal, error, or signal)
stop_ollama() {
    if [ -f "$OLLAMA_PID_FILE" ]; then
        PID=$(cat "$OLLAMA_PID_FILE")
        kill "$PID" 2>/dev/null || true
        rm -f "$OLLAMA_PID_FILE"
        echo "Ollama (PID $PID) has been stopped cleanly."
    fi
}
trap stop_ollama EXIT INT TERM

# 1. Set custom ollama models directory and host
export OLLAMA_MODELS=/coc/flash5/kvr6/repos/ollama_models 
export OLLAMA_HOST="127.0.0.1:11450"

# 2. Start the server in the background, redirecting output to a log file
./ollama/bin/ollama serve > "$OLLAMA_LOG_FILE" 2>&1 &

# 3. Save the Process ID ($!) of that specific background job to a file
echo $! > "$OLLAMA_PID_FILE"

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
    CMD="$PYTHONPATH -u label_object_state_vlm.py \
        --video-path \"$VIDEO_PATH\" \
        --model $MODEL_NAME \
        --last-action-dir object_last_action \
        --output-dir vlm_object_state_labels"
    echo "Running command: $CMD"
    eval $CMD
done