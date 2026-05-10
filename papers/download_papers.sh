#!/usr/bin/env bash
set -euo pipefail

# Script to download relevant papers for CAER-S emotion recognition research

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS_DIR="$SCRIPT_DIR"

echo "============================================================"
echo "Paper Downloader for CAER-S Emotion Recognition"
echo "============================================================"
echo ""

# Create papers directory if not exists
mkdir -p "$PAPERS_DIR"

cd "$PAPERS_DIR"

# Paper list with URLs
# Format: "filename|url|description"
PAPERS=(
    "Lee_CAER-Net_ICCV2019.pdf|https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Context-Aware_Emotion_Recognition_Networks_ICCV_2019_paper.pdf|CAER-Net (ICCV 2019)"
    "Le_GLAMOR-Net_arxiv.pdf|https://arxiv.org/pdf/2002.02392|GLAMOR-Net (NCA 2022)"
    "Zhou_CAHFW_arxiv.pdf|https://arxiv.org/pdf/2103.10186|CAHFW-Net (IJERPH 2023)"
    "Yang_CCIM_CVPR2023.pdf|https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Context_De-Confounded_Emotion_Recognition_CVPR_2023_paper.pdf|Yang CCIM (CVPR 2023)"
    "AGCD-Net_ICIAP2025.pdf|https://arxiv.org/pdf/2507.09248|AGCD-Net (ICIAP 2025)"
)

for paper_info in "${PAPERS[@]}"; do
    IFS='|' read -r filename url desc <<< "$paper_info"
    
    if [ -f "$filename" ] && [ -s "$filename" ]; then
        echo "✅ $desc already exists ($filename)"
    else
        echo "📥 Downloading $desc..."
        if wget -q --timeout=60 "$url" -O "$filename" 2>/dev/null; then
            if [ -s "$filename" ]; then
                echo "   ✅ Success"
            else
                echo "   ⚠️ Downloaded file is empty, trying curl..."
                curl -sL --max-time 60 "$url" -o "$filename"
                if [ -s "$filename" ]; then
                    echo "   ✅ Success (via curl)"
                else
                    echo "   ❌ Failed"
                fi
            fi
        else
            echo "   ⚠️ wget failed, trying curl..."
            curl -sL --max-time 60 "$url" -o "$filename"
            if [ -s "$filename" ]; then
                echo "   ✅ Success (via curl)"
            else
                echo "   ❌ Failed"
            fi
        fi
    fi
done

echo ""
echo "============================================================"
echo "Download Summary"
echo "============================================================"
ls -la *.pdf 2>/dev/null | awk '{print $9, "(" $5 " bytes)"}' || echo "No PDFs found"
echo ""
echo "Total papers: $(ls *.pdf 2>/dev/null | wc -l)"
