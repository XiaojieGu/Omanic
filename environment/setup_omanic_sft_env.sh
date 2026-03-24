#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx omanic_sft; then
  conda create -n omanic_sft python=3.11 -y
fi

conda activate omanic_sft

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd "${PROJECT_DIR}/LlamaFactory"
pip install -e .
pip install -r requirements/metrics.txt
pip install deepspeed

echo "omanic_sft environment setup completed."
