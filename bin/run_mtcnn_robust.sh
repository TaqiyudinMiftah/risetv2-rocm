#!/bin/bash
# run_mtcnn_robust.sh
# Robust MTCNN face detection with auto-restart on crash.
# Usage: ./bin/run_mtcnn_robust.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/mtcnn_robust.log"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/detect_faces_mtcnn_batch.py"

echo "========================================"
echo "MTCNN Robust Runner with Auto-Restart"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Log: $LOG_FILE"
echo ""

# Function to get current checkpoint progress
get_progress() {
    local ckpt_file="$PROJECT_DIR/artifacts/caers/mtcnn_checkpoint.json"
    if [ -f "$ckpt_file" ]; then
        local idx=$(cat "$ckpt_file" | grep -o '[0-9]*' | head -1)
        echo "$idx"
    else
        echo "0"
    fi
}

restart_count=0

while true; do
    restart_count=$((restart_count + 1))
    
    progress=$(get_progress)
    echo "[$restart_count] Starting MTCNN detection (checkpoint: $progress/69999)..."
    echo "[$restart_count] $(date) - Starting MTCNN (checkpoint: $progress)" >> "$LOG_FILE"
    
    # Run the Python script
    if uv run python "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1; then
        echo "[$restart_count] MTCNN completed successfully!"
        echo "[$restart_count] $(date) - Completed successfully" >> "$LOG_FILE"
        break
    else
        exit_code=$?
        echo "[$restart_count] MTCNN exited with code $exit_code"
        echo "[$restart_count] $(date) - Exited with code $exit_code" >> "$LOG_FILE"
        
        new_progress=$(get_progress)
        if [ "$new_progress" = "69999" ]; then
            echo "All 69,999 images processed! Done."
            break
        fi
        
        echo "Waiting 5 seconds before restart..."
        sleep 5
    fi
done

echo ""
echo "========================================"
echo "MTCNN processing finished!"
echo "Total restarts: $restart_count"
echo "Final log: $LOG_FILE"
echo "========================================"
