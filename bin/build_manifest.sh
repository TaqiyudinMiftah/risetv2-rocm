#!/usr/bin/env bash
set -euo pipefail

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
DETECTION_ARGS=()

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --train-detections)
            DETECTION_ARGS+=("--train-detections" "$2")
            shift 2
            ;;
        --val-detections)
            DETECTION_ARGS+=("--val-detections" "$2")
            shift 2
            ;;
        --test-detections)
            DETECTION_ARGS+=("--test-detections" "$2")
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH              Custom config file path (default: configs/caernet.yaml)"
            echo "  --train-detections PATH    Official train bbox file: rel_path,label,x1,y1,x2,y2"
            echo "  --val-detections PATH      Official val bbox file. Overrides random val split."
            echo "  --test-detections PATH     Official test bbox file: rel_path,label,x1,y1,x2,y2"
            echo "  --help, -h                 Show this help"
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
echo "Building CAER-S Manifest"
echo "================================"
echo "Config: $CONFIG"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

cd "$PROJECT_ROOT"
"$PYTHON" scripts/build_caers_manifest.py --config "$CONFIG" "${DETECTION_ARGS[@]}"

echo ""
echo "Manifest build complete!"
echo "  Manifest: artifacts/caers/manifest_caers.jsonl"
echo "  Diagnostics: artifacts/caers/diagnostics_caers.json"
