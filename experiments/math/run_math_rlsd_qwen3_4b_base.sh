#!/bin/bash

# Runs a single RLSD (RLVR with Self-Distillation) training experiment locally (no Slurm).
# The teacher sees the original question + ground-truth answer; the student sees only the question.
# Per-token log-prob ratios modulate GRPO advantage magnitude while keeping reward-based direction.
# λ decays linearly from rlsd_lambda_init to 0 over rlsd_lambda_decay_steps, then falls back to GRPO.
# Reference: https://arxiv.org/abs/2604.03128

# Load environment configuration
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env.local" ]; then
    set +u  # Allow unset variables in .env.local
    source "${PROJECT_ROOT}/.env.local"
    set -u
fi
export EXPERIMENT="math-RLSD-Qwen3-4B-Base"
export TASK="data/math"
DATA_PATH="data/math"
export RAY_PORT=${RAY_PORT:-6385}

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled."
fi

# =============================================================================
# PICK YOUR CONFIGURATION (edit these)
# =============================================================================
CONFIG_NAME="rlsd"

# Model — pick one
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"

# Hyperparameters
TRAIN_BATCH_SIZE=256
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=128
LR=1e-6

# RLSD-specific hyperparameters
EPSILON_W=${EPSILON_W:-0.2}           # Clipping bound for token evidence weights (paper default)
LAMBDA_INIT=${LAMBDA_INIT:-0.5}         # Initial λ (0=pure GRPO, 1=full RLSD reweighting)
LAMBDA_DECAY_STEPS=${LAMBDA_DECAY_STEPS:-50}   # Steps over which λ decays linearly to 0

# GPU settings (adjust to your local machine)
NUM_GPUS=${NUM_GPUS:-2}
_VISIBLE_DEVICES=""
for ((_i = 0; _i < NUM_GPUS; _i++)); do _VISIBLE_DEVICES+="${_VISIBLE_DEVICES:+,}$_i"; done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$_VISIBLE_DEVICES}

# =============================================================================
# SETUP
# =============================================================================
if [ -z "${WORKSPACE_DIR:-}" ]; then
    echo "Error: WORKSPACE_DIR not set. Make sure .env.local exists."
    exit 1
fi

if [[ "$DRY_RUN" != true && "${SKIP_INSTALL:-false}" != "true" ]]; then
    echo "Installing dependencies..."
    pip install word2number latex2sympy2 "math-verify[antlr4_9_3]==0.8.0"
    pip install -e "$WORKSPACE_DIR" --no-deps
    pip install --upgrade wandb
fi

export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# =============================================================================
# BUILD EXPERIMENT NAME & ARGS
# =============================================================================
EXP_NAME="${EXPERIMENT}-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-ew${EPSILON_W}-li${LAMBDA_INIT}-ld${LAMBDA_DECAY_STEPS}-model${MODEL_PATH##*/}"

RAY_TMPDIR="/tmp/ray_rlsd"
if [ "$DRY_RUN" != true ]; then
    mkdir -p "$RAY_TMPDIR"
    ray stop --force --temp-dir="$RAY_TMPDIR" 2>/dev/null || true
    ray start --head --disable-usage-stats --port=$RAY_PORT --dashboard-port=8267 --temp-dir="$RAY_TMPDIR"
fi
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.group_name=RLSD \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.self_distillation.epsilon_w=$EPSILON_W \
  actor_rollout_ref.actor.self_distillation.rlsd_lambda_init=$LAMBDA_INIT \
  actor_rollout_ref.actor.self_distillation.rlsd_lambda_decay_steps=$LAMBDA_DECAY_STEPS \
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=False \
  algorithm.rollout_correction.rollout_is=token \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  trainer.n_gpus_per_node=$NUM_GPUS"

# =============================================================================
# RUN
# =============================================================================
CMD="bash $WORKSPACE_DIR/training/verl_training.sh \"$EXP_NAME\" \"$CONFIG_NAME\" \"$DATA_PATH\" $ARGS"

LOG_FILE="nohup_rlsd_$(date +%Y%m%d_%H%M%S).out"

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
