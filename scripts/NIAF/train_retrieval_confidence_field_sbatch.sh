#!/bin/bash
#SBATCH --job-name=niaf_retrieval_field
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00

set -eo pipefail
trap 'echo "ERROR: train_retrieval_confidence_field_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/retrieval_confidence_field/configs/phoenix_retrieval_confidence_adaptive_trainbalanced.yaml}"
RUN_TAG="${RUN_TAG:-phoenix_retrieval_confidence_adaptive_trainbalanced}"
DEVICE="${DEVICE:-auto}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
WANDB="${WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-niaf-retrieval-adaptive}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID:-local}}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
RESUME="${RESUME:-}"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HOME="${HOME_VALUE:-/media/cvpr/haomian}"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi
if [[ -z "$WANDB_API_KEY" && -f "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" ]]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY="${WANDB_API_KEY:-\(.*\)}"$/\1/p' "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" | head -n 1)"
fi
export WANDB_MODE
if [[ -n "$WANDB_API_KEY" ]]; then export WANDB_API_KEY; fi
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/cvpr/haomian/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/media/cvpr/haomian/.config/wandb}"

mkdir -p "$PROJECT_DIR/logs/sbatch" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd "$PROJECT_DIR"

TRAIN_CMD=(
  srun "$PYTHON_BIN" -m NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field
  --config "$CFG"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
)
if [[ -n "$EPOCHS" ]]; then TRAIN_CMD+=(--epochs "$EPOCHS"); fi
if [[ -n "$BATCH_SIZE" ]]; then TRAIN_CMD+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then TRAIN_CMD+=(--max_train_batches "$MAX_TRAIN_BATCHES"); fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then TRAIN_CMD+=(--max_val_batches "$MAX_VAL_BATCHES"); fi
if [[ -n "$RESUME" ]]; then TRAIN_CMD+=(--resume "$RESUME"); fi
if [[ "$WANDB" == "1" ]]; then
  TRAIN_CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
fi

echo "Launching retrieval-confidence field: config=$CFG device=$DEVICE text_device=$TEXT_DEVICE W&B=$WANDB/$WANDB_MODE"
"${TRAIN_CMD[@]}"
