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

# Default mode
MODE="train"
CONFIGS_DIR="${PROJECT_ROOT}/configs"
FAILED_MODELS=()
SUCCESS_MODELS=()

# Collect extra arguments to forward to train.sh or evaluate.sh
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --augment)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --offline)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --split)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --resume)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --run-name)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --checkpoint)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --eval-after-train)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run training or evaluation for ALL available models sequentially."
            echo ""
            echo "Models executed: caernet, zhou_cross_attention, yang_ccim, glamor_net, cd_ica_net"
            echo ""
            echo "Options:"
            echo "  --mode MODE           Mode to run: train or evaluate (default: train)"
            echo "  --augment             Enable data augmentation (train mode only)"
            echo "  --eval-after-train    Evaluate test set after training in same W&B run (train only)"
            echo "  --offline             Run W&B in offline mode"
            echo "  --split SPLIT         Split to evaluate: test or val (evaluate mode only)"
            echo "  --resume PATH         Resume from checkpoint (train mode only)"
            echo "  --run-name NAME       W&B run name"
            echo "  --checkpoint PATH     Custom checkpoint (evaluate mode only)"
            echo "  --help, -h            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --mode train"
            echo "  $0 --mode train --augment --offline"
            echo "  $0 --mode train --eval-after-train"
            echo "  $0 --mode evaluate --split test"
            echo "  $0 --mode evaluate --split val --offline"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "train" && "$MODE" != "evaluate" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Use 'train' or 'evaluate'."
    exit 1
fi

# List of configs (order matters)
CONFIGS=(
    "${CONFIGS_DIR}/caernet.yaml"
    "${CONFIGS_DIR}/zhou_cross_attention.yaml"
    "${CONFIGS_DIR}/yang_ccim.yaml"
    "${CONFIGS_DIR}/glamor_net.yaml"
    "${CONFIGS_DIR}/cd_ica_net.yaml"
)

echo "============================================================"
echo "Running $MODE for ALL models"
echo "============================================================"
echo "Total models: ${#CONFIGS[@]}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

# Run each model
for CONFIG in "${CONFIGS[@]}"; do
    MODEL_NAME=$(basename "$CONFIG" .yaml)
    echo "------------------------------------------------------------"
    echo "[$MODE] $MODEL_NAME"
    echo "Config: $CONFIG"
    echo "------------------------------------------------------------"

    # Build command
    CMD=("${SCRIPT_DIR}/${MODE}.sh" --config "$CONFIG")
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    # Run and capture exit code (allow continuation on failure)
    set +e
    "${CMD[@]}"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS: $MODEL_NAME"
        SUCCESS_MODELS+=("$MODEL_NAME")
    else
        echo "❌ FAILED: $MODEL_NAME (exit code $EXIT_CODE)"
        FAILED_MODELS+=("$MODEL_NAME")
    fi
    echo ""
done

# Summary
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Mode: $MODE"
echo "Successful: ${#SUCCESS_MODELS[@]}/${#CONFIGS[@]}"
if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo "  ✅ ${SUCCESS_MODELS[*]}"
fi
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED_MODELS[@]}/${#CONFIGS[@]}"
    echo "  ❌ ${FAILED_MODELS[*]}"
fi
echo "============================================================"

# Exit with error if any model failed
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
fi

exit 0
