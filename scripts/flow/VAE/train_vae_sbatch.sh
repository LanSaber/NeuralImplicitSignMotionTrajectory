#!/bin/bash
#SBATCH --job-name=soke_flow_vae
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/VAE/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/VAE/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

set -eo pipefail
trap 'echo "ERROR: train_vae_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/soke}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
PYTHON_SITE_OVERLAY="${PYTHON_SITE_OVERLAY:-/media/cvpr/haomian/python_user_site_wandb_0_27_min}"
HOME_VALUE="${HOME_VALUE:-/media/cvpr/haomian}"
OUT_DIR_EXPLICIT=0
if [[ -n "${OUT_DIR+x}" ]]; then
  OUT_DIR_EXPLICIT=1
fi

DATA_DIR="${DATA_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175}"
RUN_NAME="${RUN_NAME:-chatsign175_temporal_vae_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUT_DIR="${OUT_DIR:-experiments/flow/VAE/$RUN_NAME}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_WITHOUT_OPTIMIZER="${RESUME_WITHOUT_OPTIMIZER:-0}"

LIMIT_TRAIN="${LIMIT_TRAIN:-0}"
LIMIT_VAL="${LIMIT_VAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-1500}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
LATENT_DIM="${LATENT_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
DROPOUT="${DROPOUT:-0.0}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIN_FRAMES="${MIN_FRAMES:-40}"
MAX_FRAMES="${MAX_FRAMES:-400}"
ROTATION_REP="${ROTATION_REP:-axis_angle}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
HAND_WEIGHT="${HAND_WEIGHT:-5.0}"
JAW_WEIGHT="${JAW_WEIGHT:-2.0}"
EXPRESSION_WEIGHT="${EXPRESSION_WEIGHT:-2.0}"
POSE_LOSS_WEIGHT="${POSE_LOSS_WEIGHT:-1.0}"
VELOCITY_LOSS_WEIGHT="${VELOCITY_LOSS_WEIGHT:-1.0}"
ACCEL_LOSS_WEIGHT="${ACCEL_LOSS_WEIGHT:-0.5}"
JERK_LOSS_WEIGHT="${JERK_LOSS_WEIGHT:-0.25}"
KL_WEIGHT="${KL_WEIGHT:-1e-6}"
KL_START_EPOCH="${KL_START_EPOCH:-200}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-300}"
VAL_EVERY="${VAL_EVERY:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-100}"
SAVE_LAST_EVERY="${SAVE_LAST_EVERY:-10}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
USE_GPUS="${USE_GPUS:-auto}"
RANDOM_CROP="${RANDOM_CROP:-0}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DDP_BACKEND="${DDP_BACKEND:-auto}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-60}"
MASTER_PORT="${MASTER_PORT:-29500}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-flow-vae}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
WANDB_DISABLE_STATS="${WANDB_DISABLE_STATS:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/flow/VAE/train_vae_sbatch.sh [options]
  bash scripts/flow/VAE/train_vae_sbatch.sh --dry-run [options]

Options can also be written as --key=value. CLI options override environment variables.

Paths:
  --project-dir PATH
  --python-env PATH
  --python-bin PATH
  --python-site-overlay PATH
  --home-value PATH
  --data-dir PATH
  --run-name NAME
  --out-dir PATH
  --resume-from-checkpoint PATH
  --resume-without-optimizer

Training:
  --limit-train N
  --limit-val N
  --batch-size N
  --epochs N
  --hidden-dim N
  --latent-dim N
  --num-layers N
  --num-heads N
  --dropout VALUE
  --downsample-factor N
  --num-workers N
  --min-frames N
  --max-frames N
  --rotation-rep axis_angle|rot6d
  --lr VALUE
  --weight-decay VALUE
  --grad-clip VALUE
  --hand-weight VALUE
  --jaw-weight VALUE
  --expression-weight VALUE
  --pose-loss-weight VALUE
  --velocity-loss-weight VALUE
  --accel-loss-weight VALUE
  --jerk-loss-weight VALUE
  --kl-weight VALUE
  --kl-start-epoch N
  --kl-warmup-epochs N
  --val-every N
  --sample-every N
  --save-every N
  --save-last-every N
  --save-top-k N
  --seed N
  --device auto|cuda|cpu
  --use-gpus auto|IDS
  --random-crop
  --no-random-crop
  --distributed auto|none|ddp
  --ddp-backend auto|nccl|gloo
  --ddp-timeout-min N
  --master-port N
  --dry-run

W&B:
  --wandb
  --no-wandb
  --wandb-online
  --wandb-offline
  --wandb-project NAME
  --wandb-id ID
  --wandb-resume allow|must|never|auto
  --wandb-api-key KEY
  --wandb-api-key-file PATH
  --wandb-disable-stats
  --wandb-enable-stats

  -h, --help
EOF
}

set_cli_value() {
  local key="$1"
  local value="$2"
  case "$key" in
    project-dir) PROJECT_DIR="$value" ;;
    python-env) PYTHON_ENV="$value" ;;
    python-bin) PYTHON_BIN="$value" ;;
    python-site-overlay) PYTHON_SITE_OVERLAY="$value" ;;
    home-value) HOME_VALUE="$value" ;;
    data-dir) DATA_DIR="$value" ;;
    run-name) RUN_NAME="$value" ;;
    out-dir) OUT_DIR="$value"; OUT_DIR_EXPLICIT=1 ;;
    resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$value" ;;
    limit-train) LIMIT_TRAIN="$value" ;;
    limit-val) LIMIT_VAL="$value" ;;
    batch-size) BATCH_SIZE="$value" ;;
    epochs) EPOCHS="$value" ;;
    hidden-dim) HIDDEN_DIM="$value" ;;
    latent-dim) LATENT_DIM="$value" ;;
    num-layers) NUM_LAYERS="$value" ;;
    num-heads) NUM_HEADS="$value" ;;
    dropout) DROPOUT="$value" ;;
    downsample-factor) DOWNSAMPLE_FACTOR="$value" ;;
    num-workers) NUM_WORKERS="$value" ;;
    min-frames) MIN_FRAMES="$value" ;;
    max-frames) MAX_FRAMES="$value" ;;
    rotation-rep) ROTATION_REP="$value" ;;
    lr) LR="$value" ;;
    weight-decay) WEIGHT_DECAY="$value" ;;
    grad-clip) GRAD_CLIP="$value" ;;
    hand-weight) HAND_WEIGHT="$value" ;;
    jaw-weight) JAW_WEIGHT="$value" ;;
    expression-weight) EXPRESSION_WEIGHT="$value" ;;
    pose-loss-weight) POSE_LOSS_WEIGHT="$value" ;;
    velocity-loss-weight) VELOCITY_LOSS_WEIGHT="$value" ;;
    accel-loss-weight) ACCEL_LOSS_WEIGHT="$value" ;;
    jerk-loss-weight) JERK_LOSS_WEIGHT="$value" ;;
    kl-weight) KL_WEIGHT="$value" ;;
    kl-start-epoch) KL_START_EPOCH="$value" ;;
    kl-warmup-epochs) KL_WARMUP_EPOCHS="$value" ;;
    val-every) VAL_EVERY="$value" ;;
    sample-every) SAMPLE_EVERY="$value" ;;
    save-every) SAVE_EVERY="$value" ;;
    save-last-every) SAVE_LAST_EVERY="$value" ;;
    save-top-k) SAVE_TOP_K="$value" ;;
    seed) SEED="$value" ;;
    device) DEVICE="$value" ;;
    use-gpus) USE_GPUS="$value" ;;
    distributed) DISTRIBUTED="$value" ;;
    ddp-backend) DDP_BACKEND="$value" ;;
    ddp-timeout-min) DDP_TIMEOUT_MIN="$value" ;;
    master-port) MASTER_PORT="$value" ;;
    wandb-project) WANDB_PROJECT="$value" ;;
    wandb-id) WANDB_ID="$value" ;;
    wandb-resume) WANDB_RESUME="$value" ;;
    wandb-api-key) WANDB_API_KEY="$value" ;;
    wandb-api-key-file) WANDB_API_KEY_FILE="$value" ;;
    wandb-disable-stats) WANDB_DISABLE_STATS="$value" ;;
    *)
      echo "ERROR: unknown option --$key" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --wandb)
      WANDB=1
      shift
      ;;
    --wandb-online)
      WANDB=1
      WANDB_MODE=online
      shift
      ;;
    --wandb-offline)
      WANDB=1
      WANDB_MODE=offline
      shift
      ;;
    --wandb-disable-stats)
      WANDB_DISABLE_STATS=1
      shift
      ;;
    --wandb-enable-stats)
      WANDB_DISABLE_STATS=0
      shift
      ;;
    --no-wandb)
      WANDB=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-dry-run)
      DRY_RUN=0
      shift
      ;;
    --resume-without-optimizer)
      RESUME_WITHOUT_OPTIMIZER=1
      shift
      ;;
    --random-crop)
      RANDOM_CROP=1
      shift
      ;;
    --no-random-crop)
      RANDOM_CROP=0
      shift
      ;;
    --*=*)
      option="${1%%=*}"
      option="${option#--}"
      value="${1#*=}"
      set_cli_value "$option" "$value"
      shift
      ;;
    --*)
      option="${1#--}"
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "ERROR: option --$option requires a value." >&2
        exit 2
      fi
      set_cli_value "$option" "$2"
      shift 2
      ;;
    *)
      echo "ERROR: unexpected argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "ERROR: --wandb-api-key-file does not exist: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi
if [[ "$OUT_DIR_EXPLICIT" == "0" ]]; then
  OUT_DIR="experiments/flow/VAE/$RUN_NAME"
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && ! -f "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "ERROR: --resume-from-checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
case "$ROTATION_REP" in
  axis_angle|rot6d)
    ;;
  *)
    echo "ERROR: --rotation-rep must be axis_angle or rot6d; got $ROTATION_REP" >&2
    exit 2
    ;;
esac

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "$PYTHON_SITE_OVERLAY" && -d "$PYTHON_SITE_OVERLAY" ]]; then
  export PYTHONPATH="$PROJECT_DIR:$PYTHON_SITE_OVERLAY:${PYTHONPATH:-}"
else
  export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
fi
export HOME="$HOME_VALUE"
if [[ "$USE_GPUS" != "auto" ]]; then
  export CUDA_VISIBLE_DEVICES="$USE_GPUS"
fi
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
if [[ "$WANDB_DISABLE_STATS" == "1" ]]; then
  export WANDB__DISABLE_STATS=true
fi
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/cvpr/haomian/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/media/cvpr/haomian/.config/wandb}"
export WANDB_PROJECT
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

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
  "$PROJECT_DIR/logs/sbatch/flow/VAE" \
  "$PROJECT_DIR/logs/wandb" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$PROJECT_DIR"

TRAIN_CMD=(
  "$PYTHON_BIN" -m flow.VAE.train_vae
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --limit_train "$LIMIT_TRAIN"
  --limit_val "$LIMIT_VAL"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --hidden_dim "$HIDDEN_DIM"
  --latent_dim "$LATENT_DIM"
  --num_layers "$NUM_LAYERS"
  --num_heads "$NUM_HEADS"
  --dropout "$DROPOUT"
  --downsample_factor "$DOWNSAMPLE_FACTOR"
  --num_workers "$NUM_WORKERS"
  --min_frames "$MIN_FRAMES"
  --max_frames "$MAX_FRAMES"
  --rotation_rep "$ROTATION_REP"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --hand_weight "$HAND_WEIGHT"
  --jaw_weight "$JAW_WEIGHT"
  --expression_weight "$EXPRESSION_WEIGHT"
  --pose_loss_weight "$POSE_LOSS_WEIGHT"
  --velocity_loss_weight "$VELOCITY_LOSS_WEIGHT"
  --accel_loss_weight "$ACCEL_LOSS_WEIGHT"
  --jerk_loss_weight "$JERK_LOSS_WEIGHT"
  --kl_weight "$KL_WEIGHT"
  --kl_start_epoch "$KL_START_EPOCH"
  --kl_warmup_epochs "$KL_WARMUP_EPOCHS"
  --val_every "$VAL_EVERY"
  --sample_every "$SAMPLE_EVERY"
  --save_every "$SAVE_EVERY"
  --save_last_every "$SAVE_LAST_EVERY"
  --save_top_k "$SAVE_TOP_K"
  --seed "$SEED"
  --device "$DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)
if [[ "$RANDOM_CROP" == "1" ]]; then
  TRAIN_CMD+=(--random_crop)
else
  TRAIN_CMD+=(--no_random_crop)
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  TRAIN_CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ "$RESUME_WITHOUT_OPTIMIZER" == "1" ]]; then
  TRAIN_CMD+=(--resume_without_optimizer)
fi
if [[ "$WANDB" == "1" ]]; then
  TRAIN_CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN_NAME")
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
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURM_NODELIST:-local}"
echo "Start time   : $(date)"
echo "Project      : $PROJECT_DIR"
echo "Python       : $PYTHON_BIN"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-slurm/default}"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "Resume       : checkpoint=${RESUME_FROM_CHECKPOINT:-unset} optimizer=$((1 - RESUME_WITHOUT_OPTIMIZER))"
echo "Model        : hidden=$HIDDEN_DIM latent=$LATENT_DIM layers=$NUM_LAYERS heads=$NUM_HEADS downsample=$DOWNSAMPLE_FACTOR dropout=$DROPOUT"
echo "Rotation rep : $ROTATION_REP"
echo "Training     : epochs=$EPOCHS batch_per_rank=$BATCH_SIZE lr=$LR wd=$WEIGHT_DECAY random_crop=$RANDOM_CROP"
echo "Distributed  : mode=$DISTRIBUTED backend=$DDP_BACKEND timeout_min=$DDP_TIMEOUT_MIN master=${MASTER_ADDR:-unset}:${MASTER_PORT} slurm_tasks=${SLURM_NTASKS:-local}"
echo "Loss weights : pose=$POSE_LOSS_WEIGHT vel=$VELOCITY_LOSS_WEIGHT accel=$ACCEL_LOSS_WEIGHT jerk=$JERK_LOSS_WEIGHT hand=$HAND_WEIGHT jaw=$JAW_WEIGHT expr=$EXPRESSION_WEIGHT kl=$KL_WEIGHT kl_start=$KL_START_EPOCH kl_warmup=$KL_WARMUP_EPOCHS"
echo "Checkpoints  : save_every=$SAVE_EVERY save_last_every=$SAVE_LAST_EVERY save_top_k=$SAVE_TOP_K val_every=$VAL_EVERY sample_every=$SAMPLE_EVERY"
echo "W&B          : enabled=$WANDB mode=$WANDB_MODE project=$WANDB_PROJECT run=$WANDB_RUN_NAME api_key=$WANDB_API_KEY_STATUS"
echo "W&B stats    : disabled=$WANDB_DISABLE_STATS"
echo "Train cmd    : ${TRAIN_CMD[*]}"
echo "========================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, command was not executed."
  exit 0
fi

if [[ "$WANDB" == "1" && "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: W&B online mode is enabled, but WANDB_API_KEY is empty; this requires an existing wandb login on the node." >&2
fi

echo "Launching temporal SMPL-X VAE training..."
SRUN_CMD=(srun)
if [[ "$DISTRIBUTED" != "none" && -n "${SLURM_NTASKS:-}" ]]; then
  SRUN_CMD+=(--ntasks "$SLURM_NTASKS")
fi
echo "Srun cmd     : ${SRUN_CMD[*]}"
"${SRUN_CMD[@]}" "${TRAIN_CMD[@]}"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
