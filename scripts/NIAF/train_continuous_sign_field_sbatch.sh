#!/bin/bash
#SBATCH --job-name=niaf_csf_train
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00

set -eo pipefail
trap 'echo "ERROR: train_continuous_sign_field_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_sign_field/configs/phoenix_adapter_scaffold_stage2.yaml}"
RUN_TAG="${RUN_TAG:-phoenix_adapter_scaffold_stage2}"
RUN_FK_CACHE="${RUN_FK_CACHE:-0}"
FK_SPLITS="${FK_SPLITS:-train,val}"
OVERWRITE_FK_CACHE="${OVERWRITE_FK_CACHE:-0}"
DEVICE="${DEVICE:-auto}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DDP_BACKEND="${DDP_BACKEND:-auto}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-120}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB="${WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-niaf-continuous-sign-field}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID:-local}}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
DRY_RUN="${DRY_RUN:-0}"

# Optional trainer overrides. Leave empty to use the YAML config values.
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
RESUME="${RESUME:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# Keep runtime state on shared storage; /home is not shared on this cluster.
export HOME="${HOME_VALUE:-/media/cvpr/haomian}"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "ERROR: WANDB_API_KEY_FILE does not exist: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" ]]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY="${WANDB_API_KEY:-\(.*\)}"$/\1/p' "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" | head -n 1)"
fi

export WANDB_MODE
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/cvpr/haomian/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/media/cvpr/haomian/.config/wandb}"
export WANDB_PROJECT
export WANDB_RUN_NAME
export SOKE_RUN_TIME="${SOKE_RUN_TIME:-${RUN_TAG}_${SLURM_JOB_ID:-local}}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  fi
  export MASTER_ADDR
  export MASTER_PORT
  if [[ -n "${SLURM_NTASKS:-}" ]]; then
    export WORLD_SIZE="$SLURM_NTASKS"
  fi
fi

mkdir -p \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
  "$PROJECT_DIR/logs/sbatch"

cd "$PROJECT_DIR"

CACHE_CMD=(
  srun --nodes=1 --ntasks=1 "$PYTHON_BIN" -m NIAF.continuous_sign_field.scripts.cache_fk_joints
  --config "$CFG"
  --splits "$FK_SPLITS"
  --device "$DEVICE"
)

if [[ "$OVERWRITE_FK_CACHE" == "1" ]]; then
  CACHE_CMD+=(--overwrite)
fi

TRAIN_CMD=(
  srun "$PYTHON_BIN" -m NIAF.continuous_sign_field.scripts.train_residual_flow
  --config "$CFG"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)

if [[ -n "$EPOCHS" ]]; then
  TRAIN_CMD+=(--epochs "$EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_CMD+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$LIMIT_TRAIN" ]]; then
  TRAIN_CMD+=(--limit_train "$LIMIT_TRAIN")
fi
if [[ -n "$LIMIT_VAL" ]]; then
  TRAIN_CMD+=(--limit_val "$LIMIT_VAL")
fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  TRAIN_CMD+=(--max_train_batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then
  TRAIN_CMD+=(--max_val_batches "$MAX_VAL_BATCHES")
fi
if [[ -n "$RESUME" ]]; then
  TRAIN_CMD+=(--resume "$RESUME")
fi
if [[ "$WANDB" == "1" ]]; then
  TRAIN_CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
  if [[ -n "$WANDB_ID" ]]; then
    TRAIN_CMD+=(--wandb_id "$WANDB_ID")
  fi
  if [[ -n "$WANDB_RESUME" ]]; then
    TRAIN_CMD+=(--wandb_resume "$WANDB_RESUME")
  fi
fi

WANDB_API_KEY_STATUS="unset"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY_STATUS="set"
fi

echo "========================================"
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : ${SLURM_NODELIST:-local}"
echo "Start time      : $(date)"
echo "Project         : $PROJECT_DIR"
echo "Python          : $PYTHON_BIN"
echo "Config          : $CFG"
echo "Run tag         : $RUN_TAG"
echo "Device          : $DEVICE"
echo "Text device     : $TEXT_DEVICE"
echo "DDP             : distributed=$DISTRIBUTED backend=$DDP_BACKEND timeout_min=$DDP_TIMEOUT_MIN master=${MASTER_ADDR:-unset}:${MASTER_PORT}"
echo "Slurm tasks     : nodes=${SLURM_NNODES:-local} ntasks=${SLURM_NTASKS:-local} tasks_per_node=${SLURM_TASKS_PER_NODE:-local}"
echo "W&B             : enabled=$WANDB mode=$WANDB_MODE project=$WANDB_PROJECT run=$WANDB_RUN_NAME api_key=$WANDB_API_KEY_STATUS"
echo "Run FK cache    : $RUN_FK_CACHE"
echo "FK splits       : $FK_SPLITS"
echo "Overwrite cache : $OVERWRITE_FK_CACHE"
echo "Train command   : ${TRAIN_CMD[*]}"
if [[ "$RUN_FK_CACHE" == "1" ]]; then
  echo "Cache command   : ${CACHE_CMD[*]}"
fi
echo "========================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, commands were not executed."
  exit 0
fi

if [[ "$RUN_FK_CACHE" == "1" ]]; then
  "${CACHE_CMD[@]}"
fi

if [[ "$WANDB" == "1" && "$WANDB_MODE" == "online" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "W&B online authentication: WANDB_API_KEY is set and hidden from logs."
  else
    echo "WARNING: W&B online mode is enabled, but WANDB_API_KEY is empty; this requires an existing wandb login on the node." >&2
  fi
fi

"${TRAIN_CMD[@]}"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
