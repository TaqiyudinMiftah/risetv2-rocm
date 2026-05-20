#!/bin/bash
# Wait for MTCNN detection to finish, then run CAER-Net training
# Usage: ./bin/train_caernet_mtcnn.sh

set -e

MTCNN_MANIFEST="artifacts/caers/manifest_caers_mtcnn.jsonl"
LOG_FILE="logs/mtcnn_detect.log"

echo "Waiting for MTCNN face detection to complete..."
echo "Monitoring: $LOG_FILE"

# Poll until manifest exists and has 69999 lines
while true; do
    if [ -f "$MTCNN_MANIFEST" ]; then
        line_count=$(wc -l < "$MTCNN_MANIFEST")
        if [ "$line_count" -ge 69999 ]; then
            echo "MTCNN detection complete ($line_count lines)."
            break
        fi
        echo "MTCNN progress: $line_count/69999 lines..."
    else
        echo "MTCNN still running, manifest not yet created..."
    fi
    sleep 60
done

echo "Starting CAER-Net training with MTCNN bboxes + Places365..."
uv run python scripts/train.py --config configs/caernet.yaml --augment --eval-after-train
