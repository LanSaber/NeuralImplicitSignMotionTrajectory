#!/bin/bash
#SBATCH --job-name=soke_flow_overfit
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

set -eo pipefail
trap 'echo "ERROR: train_overfit_unconditional_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PYTHON_BIN_EXPLICIT=0
if [[ -n "${PYTHON_BIN+x}" ]]; then
  PYTHON_BIN_EXPLICIT=1
fi
OUT_DIR_EXPLICIT=0
if [[ -n "${OUT_DIR+x}" ]]; then
  OUT_DIR_EXPLICIT=1
fi

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/soke}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
HOME_VALUE="${HOME_VALUE:-/media/cvpr/haomian}"

PKL_DIR="${PKL_DIR:-/media/cvpr/haomian/data/how2sign_pkls_cropTrue_shapeFalse}"
SOKE_ROOT="${SOKE_ROOT:-/media/cvpr/haomian/data/SOKE/How2Sign}"
DATA_DIR="${DATA_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_smoke}"
PREPARE_DATA="${PREPARE_DATA:-auto}"
PREPARE_LIMIT="${PREPARE_LIMIT:-32}"
PREPARE_OVERWRITE="${PREPARE_OVERWRITE:-0}"
TARGET_FPS="${TARGET_FPS:-20}"

RUN_NAME="${RUN_NAME:-overfit_1clip_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUT_DIR="${OUT_DIR:-experiments/flow/$RUN_NAME}"

LIMIT_TRAIN="${LIMIT_TRAIN:-1}"
LIMIT_VAL="${LIMIT_VAL:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-3000}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MIN_FRAMES="${MIN_FRAMES:-40}"
MAX_FRAMES="${MAX_FRAMES:-400}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HAND_WEIGHT="${HAND_WEIGHT:-3.0}"
POSE_LOSS_WEIGHT="${POSE_LOSS_WEIGHT:-2.0}"
VELOCITY_LOSS_WEIGHT="${VELOCITY_LOSS_WEIGHT:-2.0}"
ACCEL_LOSS_WEIGHT="${ACCEL_LOSS_WEIGHT:-1.0}"
NOISE_SAMPLES="${NOISE_SAMPLES:-8}"
NOISE_SMOOTHING="${NOISE_SMOOTHING:-9}"
SAMPLE_LENGTH="${SAMPLE_LENGTH:-0}"
SAMPLER="${SAMPLER:-heun}"
SAMPLE_EVERY="${SAMPLE_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_LAST_EVERY="${SAVE_LAST_EVERY:-50}"
VAL_EVERY="${VAL_EVERY:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
SEED="${SEED:-42}"
USE_GPUS="${USE_GPUS:-0}"
DEVICE="${DEVICE:-auto}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-flow}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

NO_RANDOM_CROP="${NO_RANDOM_CROP:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/flow/train_overfit_unconditional_sbatch.sh [options]
  bash scripts/flow/train_overfit_unconditional_sbatch.sh --dry-run [options]

Examples:
  sbatch scripts/flow/train_overfit_unconditional_sbatch.sh \
    --run-name overfit_debug --epochs 1000 --lr 1e-4 --noise-smoothing 9

  sbatch scripts/flow/train_overfit_unconditional_sbatch.sh \
    --run-name overfit_online --wandb --wandb-mode online --use-gpus 0

Options can also be written as --key=value. CLI options override environment variables.

Paths and data:
  --project-dir PATH        Project root.
  --python-env PATH         Conda/env prefix. Recomputes PYTHON_BIN unless --python-bin is set.
  --python-bin PATH         Python executable.
  --home-value PATH         HOME value inside the job.
  --pkl-dir PATH            Input SMPL-X pickle directory.
  --soke-root PATH          SOKE How2Sign root.
  --data-dir PATH           Prepared flow dataset directory.
  --prepare-data VALUE      0, 1, or auto.
  --prepare-limit N         Dataset preparation limit.
  --prepare-overwrite       Rebuild prepared .npz files.
  --no-prepare-overwrite    Do not overwrite prepared .npz files.
  --target-fps FPS          Prepared data FPS.

Run/output:
  --run-name NAME           Experiment run name. Recomputes OUT_DIR unless --out-dir is set.
  --out-dir PATH            Experiment output directory.
  --dry-run                 Print commands and exit before srun.
  --no-dry-run              Run normally.

Training:
  --limit-train N
  --limit-val N
  --batch-size N
  --epochs N
  --hidden-dim N
  --num-layers N
  --num-heads N
  --dropout VALUE
  --num-workers N
  --min-frames N
  --max-frames N
  --lr VALUE
  --weight-decay VALUE
  --hand-weight VALUE
  --pose-loss-weight VALUE
  --velocity-loss-weight VALUE
  --accel-loss-weight VALUE
  --noise-samples N
  --noise-smoothing N
  --sample-length N
  --sampler euler|heun
  --sample-every N
  --save-every N
  --save-last-every N
  --val-every N
  --sample-steps N
  --seed N
  --device auto|cuda|cpu
  --use-gpus IDS           Value for CUDA_VISIBLE_DEVICES.
  --no-random-crop         Disable random crop during training. Default for overfit.
  --random-crop            Enable random crop.

W&B:
  --wandb                  Enable wandb.
  --no-wandb               Disable wandb.
  --wandb-online           Enable wandb and set WANDB_MODE=online.
  --wandb-offline          Enable wandb and set WANDB_MODE=offline.
  --wandb-project NAME
  --wandb-mode MODE        Usually offline or online.
  --wandb-api-key KEY      API key passed to WANDB_API_KEY. Prefer env/key file on shared systems.
  --wandb-api-key-file PATH
                            Read API key from the first line of this file.

  -h, --help               Show this help.
EOF
}

set_cli_value() {
  local key="$1"
  local value="$2"
  case "$key" in
    project-dir) PROJECT_DIR="$value" ;;
    python-env) PYTHON_ENV="$value" ;;
    python-bin) PYTHON_BIN="$value"; PYTHON_BIN_EXPLICIT=1 ;;
    home-value) HOME_VALUE="$value" ;;
    pkl-dir) PKL_DIR="$value" ;;
    soke-root) SOKE_ROOT="$value" ;;
    data-dir) DATA_DIR="$value" ;;
    prepare-data) PREPARE_DATA="$value" ;;
    prepare-limit) PREPARE_LIMIT="$value" ;;
    target-fps) TARGET_FPS="$value" ;;
    run-name) RUN_NAME="$value" ;;
    out-dir) OUT_DIR="$value"; OUT_DIR_EXPLICIT=1 ;;
    limit-train) LIMIT_TRAIN="$value" ;;
    limit-val) LIMIT_VAL="$value" ;;
    batch-size) BATCH_SIZE="$value" ;;
    epochs) EPOCHS="$value" ;;
    hidden-dim) HIDDEN_DIM="$value" ;;
    num-layers) NUM_LAYERS="$value" ;;
    num-heads) NUM_HEADS="$value" ;;
    dropout) DROPOUT="$value" ;;
    num-workers) NUM_WORKERS="$value" ;;
    min-frames) MIN_FRAMES="$value" ;;
    max-frames) MAX_FRAMES="$value" ;;
    lr) LR="$value" ;;
    weight-decay) WEIGHT_DECAY="$value" ;;
    hand-weight) HAND_WEIGHT="$value" ;;
    pose-loss-weight) POSE_LOSS_WEIGHT="$value" ;;
    velocity-loss-weight) VELOCITY_LOSS_WEIGHT="$value" ;;
    accel-loss-weight) ACCEL_LOSS_WEIGHT="$value" ;;
    noise-samples) NOISE_SAMPLES="$value" ;;
    noise-smoothing) NOISE_SMOOTHING="$value" ;;
    sample-length) SAMPLE_LENGTH="$value" ;;
    sampler) SAMPLER="$value" ;;
    sample-every) SAMPLE_EVERY="$value" ;;
    save-every) SAVE_EVERY="$value" ;;
    save-last-every) SAVE_LAST_EVERY="$value" ;;
    val-every) VAL_EVERY="$value" ;;
    sample-steps) SAMPLE_STEPS="$value" ;;
    seed) SEED="$value" ;;
    use-gpus) USE_GPUS="$value" ;;
    device) DEVICE="$value" ;;
    wandb-project) WANDB_PROJECT="$value" ;;
    wandb-mode) WANDB_MODE="$value" ;;
    wandb-api-key) WANDB_API_KEY="$value" ;;
    wandb-api-key-file) WANDB_API_KEY_FILE="$value" ;;
    *)
      echo "ERROR: unknown option --$key" >&2
      echo "Run with --help to see supported options." >&2
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
    --prepare-overwrite)
      PREPARE_OVERWRITE=1
      shift
      ;;
    --no-prepare-overwrite)
      PREPARE_OVERWRITE=0
      shift
      ;;
    --no-random-crop)
      NO_RANDOM_CROP=1
      shift
      ;;
    --random-crop)
      NO_RANDOM_CROP=0
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
      echo "Run with --help to see supported options." >&2
      exit 2
      ;;
  esac
done

if [[ "$PYTHON_BIN_EXPLICIT" == "0" ]]; then
  PYTHON_BIN="$PYTHON_ENV/bin/python"
fi
if [[ "$OUT_DIR_EXPLICIT" == "0" ]]; then
  OUT_DIR="experiments/flow/$RUN_NAME"
fi

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "ERROR: --wandb-api-key-file does not exist: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi

WANDB_API_KEY_STATUS="unset"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY_STATUS="set"
fi
WANDB_API_KEY_FILE_STATUS="unset"
if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  WANDB_API_KEY_FILE_STATUS="set"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HOME="$HOME_VALUE"
export CUDA_VISIBLE_DEVICES="$USE_GPUS"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WANDB_MODE
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/cvpr/haomian/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/media/cvpr/haomian/.config/wandb}"
export WANDB_PROJECT
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

mkdir -p \
  "$PROJECT_DIR/logs/sbatch/flow" \
  "$PROJECT_DIR/logs/wandb" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$PROJECT_DIR"

MANIFEST="$DATA_DIR/meta/manifest_train.jsonl"
PREPARE_CMD=(
  "$PYTHON_BIN" -m flow.dataset.prepare_dataset
  --pkl_dir "$PKL_DIR"
  --soke_root "$SOKE_ROOT"
  --out_dir "$DATA_DIR"
  --target_fps "$TARGET_FPS"
  --limit "$PREPARE_LIMIT"
)
if [[ "$PREPARE_OVERWRITE" == "1" ]]; then
  PREPARE_CMD+=(--overwrite)
fi

TRAIN_CMD=(
  "$PYTHON_BIN" -m flow.train_unconditional
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --limit_train "$LIMIT_TRAIN"
  --limit_val "$LIMIT_VAL"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --hidden_dim "$HIDDEN_DIM"
  --num_layers "$NUM_LAYERS"
  --num_heads "$NUM_HEADS"
  --dropout "$DROPOUT"
  --num_workers "$NUM_WORKERS"
  --min_frames "$MIN_FRAMES"
  --max_frames "$MAX_FRAMES"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --hand_weight "$HAND_WEIGHT"
  --pose_loss_weight "$POSE_LOSS_WEIGHT"
  --velocity_loss_weight "$VELOCITY_LOSS_WEIGHT"
  --accel_loss_weight "$ACCEL_LOSS_WEIGHT"
  --noise_samples "$NOISE_SAMPLES"
  --noise_smoothing "$NOISE_SMOOTHING"
  --sample_length "$SAMPLE_LENGTH"
  --sampler "$SAMPLER"
  --sample_every "$SAMPLE_EVERY"
  --save_every "$SAVE_EVERY"
  --save_last_every "$SAVE_LAST_EVERY"
  --val_every "$VAL_EVERY"
  --sample_steps "$SAMPLE_STEPS"
  --seed "$SEED"
  --device "$DEVICE"
)
if [[ "$NO_RANDOM_CROP" == "1" ]]; then
  TRAIN_CMD+=(--no_random_crop)
fi
if [[ "$WANDB" == "1" ]]; then
  TRAIN_CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN_NAME")
fi

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURM_NODELIST:-local}"
echo "Start time   : $(date)"
echo "Project      : $PROJECT_DIR"
echo "Python       : $PYTHON_BIN"
echo "CUDA devices : $CUDA_VISIBLE_DEVICES"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "Prepare mode : $PREPARE_DATA (limit=$PREPARE_LIMIT overwrite=$PREPARE_OVERWRITE)"
echo "Train limit  : train=$LIMIT_TRAIN val=$LIMIT_VAL"
echo "Model        : hidden=$HIDDEN_DIM layers=$NUM_LAYERS heads=$NUM_HEADS dropout=$DROPOUT"
echo "Training     : epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR wd=$WEIGHT_DECAY"
echo "Flow/noise    : noise_samples=$NOISE_SAMPLES noise_smoothing=$NOISE_SMOOTHING sampler=$SAMPLER sample_steps=$SAMPLE_STEPS"
echo "Loss weights : pose=$POSE_LOSS_WEIGHT vel=$VELOCITY_LOSS_WEIGHT accel=$ACCEL_LOSS_WEIGHT hand=$HAND_WEIGHT"
echo "Random crop  : enabled=$((1 - NO_RANDOM_CROP))"
echo "W&B          : enabled=$WANDB mode=$WANDB_MODE project=$WANDB_PROJECT run=$WANDB_RUN_NAME api_key=$WANDB_API_KEY_STATUS api_key_file=$WANDB_API_KEY_FILE_STATUS"
echo "Prepare cmd  : ${PREPARE_CMD[*]}"
echo "Train cmd    : ${TRAIN_CMD[*]}"
echo "========================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, commands were not executed."
  exit 0
fi

if [[ "$WANDB" == "1" && "$WANDB_MODE" == "online" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "W&B online authentication: WANDB_API_KEY is set and hidden from logs."
  else
    echo "WARNING: W&B online mode is enabled, but WANDB_API_KEY is empty; this requires an existing wandb login on the node." >&2
  fi
fi

if [[ "$PREPARE_DATA" == "1" || ( "$PREPARE_DATA" == "auto" && ! -s "$MANIFEST" ) ]]; then
  echo "Preparing flow smoke dataset..."
  srun "${PREPARE_CMD[@]}"
else
  echo "Skipping data preparation; found $MANIFEST"
fi

echo "Launching unconditional overfit training..."
srun "${TRAIN_CMD[@]}"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
