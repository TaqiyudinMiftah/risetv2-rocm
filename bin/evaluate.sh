#!/usr/bin/env bash
set -euo pipefail

# Auto-detect RX 6600 / gfx1032 dan apply workaround ROCm
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo | grep -oP 'Name:\s+\Kgfx[0-9a-z]+' | head -1 || true)
    if [ "$GPU_ARCH" = "gfx1032" ] && [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
        echo "WARNING: Terdeteksi GPU $GPU_ARCH (RX 6600 series). ROCm tidak support natively."
        echo "         Apply workaround: export HSA_OVERRIDE_GFX_VERSION=10.3.0"
        export HSA_OVERRIDE_GFX_VERSION=10.3.0
    fi
fi

# Detect python executable (python3 preferred)
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: python or python3 not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caernet.yaml"
CHECKPOINT=""
SPLIT="test"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --run-name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --offline)
            WANDB_OFFLINE="--wandb-offline"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH  Path to model checkpoint"
            echo "  --split SPLIT      Split to evaluate: test or val (default: test)"
            echo "  --run-name NAME    W&B run name"
            echo "  --offline          Run W&B in offline mode"
            echo "  --config PATH      Custom config file path"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-caers-emotion-recognition}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-}"

# Auto-detect checkpoint from config method if not provided
if [ -z "$CHECKPOINT" ]; then
    METHOD=$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['method'])" 2>/dev/null || echo "caernet")
    CHECKPOINT="${PROJECT_ROOT}/checkpoints/${METHOD}/best_model.pt"
fi

echo "================================"
echo "Evaluating Emotion Recognition"
echo "================================"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "W&B Project: $WANDB_PROJECT"
[ -n "$WANDB_RUN_NAME" ] && echo "W&B Run Name: $WANDB_RUN_NAME"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Train first with: ./bin/train.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

# Build command
CMD=(
    "$PYTHON" scripts/evaluate.py
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --split "$SPLIT"
)

[ -n "$WANDB_API_KEY" ] && CMD+=(--wandb-api-key "$WANDB_API_KEY")
[ -n "$WANDB_PROJECT" ] && CMD+=(--wandb-project "$WANDB_PROJECT")
[ -n "$WANDB_ENTITY" ] && CMD+=(--wandb-entity "$WANDB_ENTITY")
[ -n "$WANDB_RUN_NAME" ] && CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
[ -n "$WANDB_OFFLINE" ] && CMD+=($WANDB_OFFLINE)

# Run evaluation
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "Evaluation complete!"
