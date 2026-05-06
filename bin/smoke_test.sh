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

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH      Custom config file path (default: configs/caernet.yaml)"
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

echo "================================"
echo "Smoke Test Data Pipeline"
echo "================================"
echo "Config: $CONFIG"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

cd "$PROJECT_ROOT"

"$PYTHON" scripts/smoke_data_pipeline.py --config "$CONFIG" --batch-size 4

echo ""
echo "Smoke test complete!"
