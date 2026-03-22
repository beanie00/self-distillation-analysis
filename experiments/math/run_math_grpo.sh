#!/bin/bash

# Runs a single GRPO training experiment locally (no Slurm).
# Edit the parameters below to choose which configuration to run.

# Load environment configuration
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env.local" ]; then
    set +u  # Allow unset variables in .env.local
    source "${PROJECT_ROOT}/.env.local"
    set -u
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EXPERIMENT="math-GRPO-DeepSeek-Qwen-7B"
# Dataset — pick one
export TASK="data/math"
DATA_PATH="data/math"
export RAY_PORT=6379

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled."
fi

# =============================================================================
# PICK YOUR CONFIGURATION (edit these)
# =============================================================================
CONFIG_NAME="baseline_grpo"

# Model — pick one
# MODEL_PATH="Qwen/Qwen3-8B"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_PATH="allenai/Olmo-3-7B-Instruct"

# Hyperparameters
TRAIN_BATCH_SIZE=256
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=128
LR=1e-6

# GPU settings (adjust to your local machine)
NUM_GPUS=4  # set to number of GPUs you have

# =============================================================================
# SETUP
# =============================================================================
# WORKSPACE_DIR is loaded from .env.local
if [ -z "${WORKSPACE_DIR:-}" ]; then
    echo "Error: WORKSPACE_DIR not set. Make sure .env.local exists."
    exit 1
fi

# Install dependencies (skip if already installed)
if [[ "${SKIP_INSTALL:-false}" != "true" ]]; then
    echo "Installing dependencies..."
    pip install word2number latex2sympy2 "math-verify[antlr4_9_3]==0.8.0"
    pip install -e "$WORKSPACE_DIR"
    pip install --upgrade wandb
fi

export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# =============================================================================
# BUILD EXPERIMENT NAME & ARGS
# =============================================================================
EXP_NAME="${EXPERIMENT}-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}"

export RAY_TMPDIR="/tmp/ray_grpo"
mkdir -p "$RAY_TMPDIR"
ray stop --force --temp-dir="$RAY_TMPDIR" 2>/dev/null || true
mkdir -p "$RAY_TMPDIR"
ray start --head --disable-usage-stats --port=$RAY_PORT --dashboard-port=8265 --temp-dir="$RAY_TMPDIR"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.group_name=GRPO-generalization \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.model.path=$MODEL_PATH \
  algorithm.rollout_correction.rollout_is=token \
  actor_rollout_ref.rollout.val_kwargs.n=1"

# =============================================================================
# RUN
# =============================================================================
CMD="bash $WORKSPACE_DIR/training/verl_training.sh \"$EXP_NAME\" \"$CONFIG_NAME\" \"$DATA_PATH\" $ARGS"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Would run:"
    echo "  EXP_NAME:  $EXP_NAME"
    echo "  DATA_PATH: $DATA_PATH"
    echo "  MODEL:     $MODEL_PATH"
    echo "  CMD:       $CMD"
else
    echo "Running experiment: $EXP_NAME"
    eval "$CMD"
fi