#!/bin/bash
#SBATCH --job-name=niaf_cont_traj
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00

set -eo pipefail
trap 'echo "ERROR: train_continuous_trajectory_field_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_trajectory_field/configs/phoenix_continuous_trajectory_full.yaml}"
RUN_TAG="${RUN_TAG:-phoenix_continuous_trajectory_full}"
DEVICE="${DEVICE:-auto}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
DISTRIBUTED="${DISTRIBUTED:-ddp}"
DDP_BACKEND="${DDP_BACKEND:-nccl}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-120}"
WANDB="${WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-niaf-continuous-trajectory}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID:-local}}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
DRY_RUN="${DRY_RUN:-0}"

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
RESUME="${RESUME:-}"
WARM_START="${WARM_START:-}"
RESET_LOCAL_BRANCH="${RESET_LOCAL_BRANCH:-0}"
OUT_DIR="${OUT_DIR:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "ERROR: WANDB_API_KEY_FILE does not exist: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  WANDB_KEY_LINE="$(sed -n '/[^[:space:]]/ {s/\r$//; p; q;}' "$WANDB_API_KEY_FILE")"
  WANDB_KEY_LINE="${WANDB_KEY_LINE#export }"
  if [[ "$WANDB_KEY_LINE" == WANDB_API_KEY=* ]]; then
    WANDB_API_KEY="${WANDB_KEY_LINE#WANDB_API_KEY=}"
  else
    WANDB_API_KEY="$WANDB_KEY_LINE"
  fi
  if [[ "$WANDB_API_KEY" == \"*\" || "$WANDB_API_KEY" == \'*\' ]]; then
    WANDB_API_KEY="${WANDB_API_KEY:1:${#WANDB_API_KEY}-2}"
  fi
  unset WANDB_KEY_LINE
  if [[ -z "$WANDB_API_KEY" ]]; then
    echo "ERROR: WANDB_API_KEY_FILE did not contain a key" >&2
    exit 1
  fi
fi
if [[ -z "$WANDB_API_KEY" && -f "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" ]]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY="${WANDB_API_KEY:-\(.*\)}"$/\1/p' "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" | head -n 1)"
fi
export WANDB_MODE
if [[ -n "$WANDB_API_KEY" ]]; then export WANDB_API_KEY; fi
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/cvpr/haomian/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/media/cvpr/haomian/.config/wandb}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  fi
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
  fi
  export MASTER_ADDR MASTER_PORT
  if [[ -n "${SLURM_NTASKS:-}" ]]; then export WORLD_SIZE="$SLURM_NTASKS"; fi
fi

mkdir -p \
  "$PROJECT_DIR/logs/sbatch" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd "$PROJECT_DIR"

TRAIN_CMD=(
  srun --kill-on-bad-exit=1
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field
  --config "$CFG"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)
if [[ -n "$EPOCHS" ]]; then TRAIN_CMD+=(--epochs "$EPOCHS"); fi
if [[ -n "$BATCH_SIZE" ]]; then TRAIN_CMD+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$LIMIT_TRAIN" ]]; then TRAIN_CMD+=(--limit_train "$LIMIT_TRAIN"); fi
if [[ -n "$LIMIT_VAL" ]]; then TRAIN_CMD+=(--limit_val "$LIMIT_VAL"); fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then TRAIN_CMD+=(--max_train_batches "$MAX_TRAIN_BATCHES"); fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then TRAIN_CMD+=(--max_val_batches "$MAX_VAL_BATCHES"); fi
if [[ -n "$RESUME" ]]; then TRAIN_CMD+=(--resume "$RESUME"); fi
if [[ -n "$WARM_START" ]]; then TRAIN_CMD+=(--warm_start "$WARM_START"); fi
if [[ "$RESET_LOCAL_BRANCH" == "1" ]]; then TRAIN_CMD+=(--reset_local_branch); fi
if [[ -n "$OUT_DIR" ]]; then TRAIN_CMD+=(--out_dir "$OUT_DIR"); fi
if [[ "$WANDB" == "1" ]]; then
  TRAIN_CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
  if [[ -n "$WANDB_ID" ]]; then TRAIN_CMD+=(--wandb_id "$WANDB_ID"); fi
  if [[ -n "$WANDB_RESUME" ]]; then TRAIN_CMD+=(--wandb_resume "$WANDB_RESUME"); fi
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodes: ${SLURM_NODELIST:-local} world_size=${WORLD_SIZE:-1}"
echo "Config: $CFG"
echo "Batch override: ${BATCH_SIZE:-config value}"
echo "DDP: $DISTRIBUTED/$DDP_BACKEND master=${MASTER_ADDR:-unset}:${MASTER_PORT:-unset}"
echo "W&B: $WANDB/$WANDB_MODE project=$WANDB_PROJECT run=$WANDB_RUN_NAME"
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi
"${TRAIN_CMD[@]}"
