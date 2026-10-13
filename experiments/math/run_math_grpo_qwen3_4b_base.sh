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
export EXPERIMENT="math-GRPO-Qwen3-4B-Base"
# Dataset — pick one
export TASK="data/math"
DATA_PATH="data/math"
export RAY_PORT=${RAY_PORT:-6403}

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
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"

# Hyperparameters
TRAIN_BATCH_SIZE=256
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=128
LR=1e-6

# GPU settings (adjust to your local machine)
NUM_GPUS=${NUM_GPUS:-2}  # set to number of GPUs you have
_VISIBLE_DEVICES=""
for ((_i = 0; _i < NUM_GPUS; _i++)); do _VISIBLE_DEVICES+="${_VISIBLE_DEVICES:+,}$_i"; done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$_VISIBLE_DEVICES}

# =============================================================================
# SETUP
# =============================================================================
# WORKSPACE_DIR is loaded from .env.local
if [ -z "${WORKSPACE_DIR:-}" ]; then
    echo "Error: WORKSPACE_DIR not set. Make sure .env.local exists."
    exit 1
fi

# Install dependencies (skip if already installed)
if [[ "$DRY_RUN" != true && "${SKIP_INSTALL:-false}" != "true" ]]; then
    echo "Installing dependencies..."
    pip install word2number latex2sympy2 "math-verify[antlr4_9_3]==0.8.0"
    pip install -e "$WORKSPACE_DIR"
    pip install --upgrade wandb
fi

export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# =============================================================================
# BUILD EXPERIMENT NAME & ARGS
# =============================================================================
EXP_NAME="${EXPERIMENT}-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH##*/}"

export RAY_TMPDIR="/tmp/ray_grpo"
if [ "$DRY_RUN" != true ]; then
    mkdir -p "$RAY_TMPDIR"
    ray stop --force --temp-dir="$RAY_TMPDIR" 2>/dev/null || true
    mkdir -p "$RAY_TMPDIR"
    ray start --head --disable-usage-stats --port=$RAY_PORT --dashboard-port=8265 --temp-dir="$RAY_TMPDIR"
fi

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.group_name=GRPO-generalization \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.model.path=$MODEL_PATH \
  algorithm.rollout_correction.rollout_is=token \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  trainer.n_gpus_per_node=$NUM_GPUS"

# =============================================================================
# RUN
# =============================================================================
CMD="bash $WORKSPACE_DIR/training/verl_training.sh \"$EXP_NAME\" \"$CONFIG_NAME\" \"$DATA_PATH\" $ARGS"

LOG_FILE="nohup_grpo_$(date +%Y%m%d_%H%M%S).out"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Would run:"
    echo "  EXP_NAME:  $EXP_NAME"
    echo "  DATA_PATH: $DATA_PATH"
    echo "  MODEL:     $MODEL_PATH"
    echo "  CMD:       nohup $CMD > $LOG_FILE 2>&1 &"
else
    echo "Running experiment: $EXP_NAME"
    echo "Logging to: $LOG_FILE"
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo "Started with PID $!"
fi