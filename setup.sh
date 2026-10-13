#!/usr/bin/env bash
# Environment setup for NVIDIA B200 (CUDA 12.8, sm_100).
# Other GPUs need different torch / vLLM / flash-attn builds.
#
# Usage:
#   bash setup.sh
#   source .venv/bin/activate      # the venv does not survive the script
#
# Set WANDB_API_KEY beforehand to log in:
#   WANDB_API_KEY=... bash setup.sh

set -eo pipefail

pip install uv
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -e .
uv pip install wheel
uv pip install flash-attn --no-build-isolation

uv pip install vllm==0.15.1
uv pip install wandb

# vLLM ships its own flash-attn, so pin the version we need afterwards.
uv pip install flash-attn==2.8.1 --no-build-isolation

if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set. Run 'wandb login' before training."
fi

python3 -c "
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 512, 8, 64, dtype=torch.float16, device='cuda')
k, v = torch.randn_like(q), torch.randn_like(q)
out = flash_attn_func(q, k, v, causal=True)
print('Flash Attention OK:', out.shape)
"

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"
