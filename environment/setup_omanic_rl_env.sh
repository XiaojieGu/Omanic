#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx omanic_rl; then
  conda create -n omanic_rl python=3.11 -y
fi

conda activate omanic_rl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# vLLM installation may require a visible CUDA toolchain on the current node.
if command -v module >/dev/null 2>&1; then
  module load cuda/12.2.2 || true
fi
if [[ -z "${CUDA_HOME:-}" && -n "${CUDA_PATH:-}" ]]; then
  export CUDA_HOME="${CUDA_PATH}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

cd "${PROJECT_DIR}/verl"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install --force-reinstall setuptools==80.10.2 packaging==25.0 wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install --no-build-isolation -U vllm

# Qwen full fine-tuning can save tokenizer_config.json with extra_special_tokens as a list.
# transformers 4.57.x expects additional_special_tokens in this case, otherwise RL startup fails.
python - <<'EOF'
from pathlib import Path
import json

path = Path('../LlamaFactory/saves/qwen3-8b/full/tokenizer_config.json')
if not path.exists():
    print(f'Skip tokenizer patch: {path} does not exist yet.')
else:
    data = json.loads(path.read_text(encoding='utf-8'))
    extra = data.get('extra_special_tokens')
    if isinstance(extra, list):
        if not data.get('additional_special_tokens'):
            data['additional_special_tokens'] = extra
        data.pop('extra_special_tokens', None)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + \"\n\", encoding='utf-8')
        print(f'Patched tokenizer config: {path}')
    else:
        print(f'Tokenizer config already compatible: {path}')
EOF

echo "omanic_rl environment setup completed."
