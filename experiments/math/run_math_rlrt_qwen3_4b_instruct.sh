#!/bin/bash

# Runs RLRT (RLVR with Reversed Teacher) training.
# Token-level reweighting is applied ONLY to correct trajectories; incorrect
# trajectories receive vanilla GRPO penalties with no teacher-based modulation.
#
# Every setting below can be overridden from the environment, e.g.
#   EPSILON_W=0.5 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash <this script>

# Load environment configuration
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env.local" ]; then
    set +u  # Allow unset variables in .env.local
    source "${PROJECT_ROOT}/.env.local"
    set -u
fi

# =============================================================================
# MODEL  (the only block that differs between the RLRT scripts)
# =============================================================================
RUN_TAG="rlrt_4b_instruct"
export EXPERIMENT="math-RLRT-Qwen3-4B-Instruct"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
EPSILON_W=${EPSILON_W:-0.5}            # Clip bound for token evidence weights
LAMBDA_DECAY_STEPS=${LAMBDA_DECAY_STEPS:-30}   # Steps over which lambda decays linearly to 0
ENABLE_THINKING=${ENABLE_THINKING:-}   # empty = leave the chat template alone
export RAY_PORT=${RAY_PORT:-6414}
DASHBOARD_PORT=${DASHBOARD_PORT:-8271}

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================
CONFIG_NAME="rlrt"
export TASK="data/math"
DATA_PATH="data/math"

# Batching / budget
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
LR=${LR:-1e-6}

# RLRT hyperparameters
LAMBDA_INIT=${LAMBDA_INIT:-0.5}   # 0 = pure GRPO, 1 = full RLRT reweighting

# Machine settings — override NUM_GPUS (CUDA_VISIBLE_DEVICES follows it unless set)
NUM_GPUS=${NUM_GPUS:-2}
_VISIBLE_DEVICES=""
for ((_i = 0; _i < NUM_GPUS; _i++)); do _VISIBLE_DEVICES+="${_VISIBLE_DEVICES:+,}$_i"; done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$_VISIBLE_DEVICES}


DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled."
fi

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

RAY_TMPDIR="/tmp/ray_${RUN_TAG}"
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
if [ "$DRY_RUN" != true ]; then
    mkdir -p "$RAY_TMPDIR"
    ray stop --force --temp-dir="$RAY_TMPDIR" 2>/dev/null || true
    ray start --head --disable-usage-stats --port=$RAY_PORT --dashboard-port=$DASHBOARD_PORT --temp-dir="$RAY_TMPDIR"
fi

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.group_name=RLRT \
  trainer.n_gpus_per_node=$NUM_GPUS \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.actor.self_distillation.epsilon_w=$EPSILON_W \
  actor_rollout_ref.actor.self_distillation.rlsd_lambda_init=$LAMBDA_INIT \
  actor_rollout_ref.actor.self_distillation.rlsd_lambda_decay_steps=$LAMBDA_DECAY_STEPS \
  actor_rollout_ref.actor.self_distillation.reweight_correct_only=true \
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=False \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  algorithm.rollout_correction.rollout_is=token"

if [ -n "$ENABLE_THINKING" ]; then
    ARGS="$ARGS data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING"
fi

# =============================================================================
# RUN
# =============================================================================
CMD="bash $WORKSPACE_DIR/training/verl_training.sh \"$EXP_NAME\" \"$CONFIG_NAME\" \"$DATA_PATH\" $ARGS"

LOG_FILE="nohup_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).out"

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
