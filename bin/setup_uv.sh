#!/usr/bin/env bash
set -euo pipefail

echo "================================"
echo "CAER-S Pipeline UV Setup (ROCm)"
echo "================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Auto-detect RX 6600 / gfx1032 dan apply workaround ROCm
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo | grep -oP 'Name:\s+\Kgfx[0-9a-z]+' | head -1 || true)
    if [ "$GPU_ARCH" = "gfx1032" ] && [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
        echo "WARNING: Terdeteksi GPU $GPU_ARCH (RX 6600 series). ROCm tidak support natively."
        echo "         Apply workaround: export HSA_OVERRIDE_GFX_VERSION=10.3.0"
        export HSA_OVERRIDE_GFX_VERSION=10.3.0
    fi
fi

echo "Python version: $(uv python --version 2>/dev/null || echo 'not managed by uv')"
echo ""

# ROCm configuration
ROCM_VERSION="${ROCM_VERSION:-6.2}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

echo "GPU Backend : AMD ROCm ${ROCM_VERSION}"
echo "Index URL   : ${TORCH_INDEX_URL}"
echo ""

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12
else
    echo "Virtual environment already exists."
fi

echo ""
echo "[1/3] Installing PyTorch + TorchVision from ROCm index..."
echo "      (manual install karena uv resolver belum stabil untuk ROCm + triton-rocm)"
uv pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

echo ""
echo "[2/3] Installing triton-rocm (dependency ROCm PyTorch)..."
uv pip install triton-rocm --index-url "${TORCH_INDEX_URL}" || true

echo ""
echo "[3/3] Installing remaining dependencies with uv..."
uv pip install -e ".[dev]"

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Activate environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Verifikasi GPU:"
echo "  python -c \"import torch; print('HIP available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
echo ""
echo "Next steps:"
echo "  1. Update configs/caers_data.yaml dengan path dataset kamu"
echo "  2. Run: ./bin/build_manifest.sh"
echo "  3. Run: ./bin/train.sh"
echo ""
echo "Catatan ROCm:"
echo "  - torch.cuda.* API digunakan juga oleh HIP untuk kompatibilitas"
echo "  - Kalau triton-rocm gagal install otomatis, bisa manual dengan:"
echo "    uv pip install triton-rocm --index-url ${TORCH_INDEX_URL}"
echo "  - GPU RX 6600 (gfx1032): workaround HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "    sudah auto-apply kalau terdeteksi, atau set manual sebelum run."
