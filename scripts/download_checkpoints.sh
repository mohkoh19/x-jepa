#!/bin/bash
# Download all X-JEPA model checkpoints from Hugging Face Model Hub.
# Usage: bash scripts/download_checkpoints.sh [output_dir]
# Default output directory: checkpoints/

set -euo pipefail

REPO_ID="mohkoh/x-jepa"
OUTPUT_DIR="${1:-checkpoints}"

mkdir -p "$OUTPUT_DIR"

echo "Downloading X-JEPA checkpoints from https://huggingface.co/$REPO_ID"
echo "Output directory: $OUTPUT_DIR"
echo ""

declare -A FILES=(
    ["clip.ckpt"]="CLIP"
    ["siglip.ckpt"]="SigLIP"
    ["xjepa_p.ckpt"]="X-JEPA [P]"
    ["xjepa_tc.ckpt"]="X-JEPA [TC]"
    ["xjepa_pa_lam003.ckpt"]="X-JEPA [P,A] lambda=0.03"
    ["xjepa_pa_lam01.ckpt"]="X-JEPA [P,A] lambda=0.1"
    ["xjepa_pa_lam03.ckpt"]="X-JEPA [P,A] lambda=0.3"
    ["xjepa_pa_lam10.ckpt"]="X-JEPA [P,A] lambda=1.0"
)

for file in "${!FILES[@]}"; do
    label="${FILES[$file]}"
    dest="$OUTPUT_DIR/$file"
    if [ -f "$dest" ]; then
        echo "✓ Already exists: $dest (skipping)"
        continue
    fi
    echo "Downloading $label -> $dest"
    wget -q --show-progress \
         --continue \
         -O "$dest" \
         "https://huggingface.co/$REPO_ID/resolve/main/$file"
done

echo ""
echo "All checkpoints downloaded to: $OUTPUT_DIR/"
