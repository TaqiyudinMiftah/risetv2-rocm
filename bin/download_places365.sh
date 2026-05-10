#!/usr/bin/env bash
set -euo pipefail

# Script to download Places365 pretrained ResNet-152 weights
# for confounder extraction in Yang CCIM / CD-ICA-Net.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="$HOME/.cache/cd_ica_net/places365"

echo "============================================================"
echo "Places365 ResNet-152 Weight Downloader"
echo "============================================================"
echo ""
echo "Paper reference (Yang et al. CVPR 2023):"
echo '  "We use ResNet-152 pretrained on Places365 to extract'
echo '   context features for building the confounder dictionary."'
echo ""
echo "Target directory: $CACHE_DIR"
echo ""

mkdir -p "$CACHE_DIR"

# Try multiple sources
URLS=(
    "https://github.com/CSAILVision/places365/raw/master/models/resnet152_places365.pth.tar"
    "http://places2.csail.mit.edu/models_places365/resnet152_places365.pth.tar"
)

SUCCESS=0
for url in "${URLS[@]}"; do
    echo "Trying: $url"
    if wget -q --show-progress -O "$CACHE_DIR/resnet152_places365.pth.tar" "$url" 2>/dev/null; then
        echo "Download successful! Extracting..."
        cd "$CACHE_DIR"
        tar -xf resnet152_places365.pth.tar
        rm resnet152_places365.pth.tar
        SUCCESS=1
        break
    else
        echo "Failed."
    fi
done

if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "ERROR: Could not download Places365 weights automatically."
    echo ""
    echo "Please download manually from one of these sources:"
    echo "  1. https://github.com/CSAILVision/places365/tree/master/models"
    echo "  2. https://drive.google.com/drive/folders/1H6EvH9oMt_5GY-FdW8Er3dF7_Y8n3fU1"
    echo ""
    echo "Then place the file at:"
    echo "  $CACHE_DIR/resnet152_places365.pth"
    echo ""
    exit 1
fi

echo ""
echo "✅ Places365 ResNet-152 weights installed successfully!"
echo "Location: $CACHE_DIR/resnet152_places365.pth"
echo ""
echo "The confounder builder will automatically use these weights"
echo "when confounder_backbone is set to 'resnet152_places365'."
