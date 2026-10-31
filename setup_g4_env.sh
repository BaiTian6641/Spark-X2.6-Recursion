#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Spark-X2.6-Recursion Full-Finetuning Environment (Colab G4 96GB VRAM / 130GB RAM) ==="
python3 -m pip install -q -r requirements.txt
# Install Unsloth / Unsloth Studio kernels without overriding Colab's native Blackwell PyTorch or transformers==4.57.1
python3 -m pip install -q --no-deps unsloth unsloth_zoo || true
echo "=== Environment Ready ==="
