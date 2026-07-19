#!/bin/bash
#SBATCH --job-name=soke_flow_textcond
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

set -eo pipefail
trap 'echo "ERROR: train_text_conditional_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

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
PYTHON_SITE_OVERLAY="${PYTHON_SITE_OVERLAY:-/media/cvpr/haomian/python_user_site_wandb_0_27_min}"
HOME_VALUE="${HOME_VALUE:-/media/cvpr/haomian}"

DATA_DIR="${DATA_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_smoke}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-deps/flan-t5-base}"
MAX_TEXT_TOKENS="${MAX_TEXT_TOKENS:-64}"
TEXT_CONDITIONING="${TEXT_CONDITIONING:-pooled}"
ROTATION_REP="${ROTATION_REP:-axis_angle}"
MOTION_SPACE="${MOTION_SPACE:-smplx}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-experiments/flow/VAE/chatsign175_temporal_vae_b32/checkpoints/best.pt}"
SOURCE_MODE="${SOURCE_MODE:-noise}"
WORD_DATA_DIR="${WORD_DATA_DIR:-}"
WORD_SPLIT="${WORD_SPLIT:-train}"
CONDITION_FIELD="${CONDITION_FIELD:-text}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-}"
RESIDUAL_NOISE_SCALE="${RESIDUAL_NOISE_SCALE:-0.25}"

RUN_NAME="${RUN_NAME:-text_cond_32_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUT_DIR="${OUT_DIR:-experiments/flow/$RUN_NAME}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_WITHOUT_OPTIMIZER="${RESUME_WITHOUT_OPTIMIZER:-0}"

LIMIT_TRAIN="${LIMIT_TRAIN:-32}"
LIMIT_VAL="${LIMIT_VAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3000}"
MODEL_SIZE="${MODEL_SIZE:-custom}"
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
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-1.0}"
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
SAVE_TOP_K="${SAVE_TOP_K:-3}"
VAL_EVERY="${VAL_EVERY:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
SEED="${SEED:-42}"
USE_GPUS="${USE_GPUS:-auto}"
DEVICE="${DEVICE:-auto}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DDP_BACKEND="${DDP_BACKEND:-auto}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-60}"
MASTER_PORT="${MASTER_PORT:-29500}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-flow}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

NO_RANDOM_CROP="${NO_RANDOM_CROP:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/flow/train_text_conditional_sbatch.sh [options]
  bash scripts/flow/train_text_conditional_sbatch.sh --dry-run [options]

Example:
  sbatch scripts/flow/train_text_conditional_sbatch.sh \
    --run-name text_cond_32 --wandb-online --wandb-api-key-file ~/.wandb_api_key

Options can also be written as --key=value. CLI options override environment variables.

Paths:
  --project-dir PATH
  --python-env PATH
  --python-bin PATH
  --python-site-overlay PATH
  --home-value PATH
  --data-dir PATH
  --text-model-path PATH
  --max-text-tokens N
  --text-conditioning pooled|token_prefix
  --rotation-rep axis_angle|rot6d
  --motion-space smplx|latent
  --vae-checkpoint PATH
  --source-mode noise|residual|adapter_residual
  --word-data-dir PATH
  --word-split SPLIT
  --condition-field text|gloss|text_gloss|label_word
  --adapter-checkpoint PATH
  --residual-noise-scale VALUE

Run/output:
  --run-name NAME
  --out-dir PATH
  --resume-from-checkpoint PATH
  --resume-without-optimizer
  --dry-run
  --no-dry-run

Training:
  --limit-train N
  --limit-val N
  --batch-size N
  --epochs N
  --model-size custom|small|base|large|xl
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
  --latent-loss-weight VALUE
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
  --save-top-k N
  --val-every N
  --sample-steps N
  --seed N
  --device auto|cuda|cpu
  --use-gpus auto|IDS
  --distributed auto|none|ddp
  --ddp-backend auto|nccl|gloo
  --ddp-timeout-min N
  --master-port N
  --no-random-crop
  --random-crop

W&B:
  --wandb
  --no-wandb
  --wandb-online
  --wandb-offline
  --wandb-project NAME
  --wandb-mode MODE
  --wandb-id ID
  --wandb-resume allow|must|never|auto
  --wandb-api-key KEY       Prefer env/key file on shared systems.
  --wandb-api-key-file PATH Read API key from the first line of this file.

  -h, --help
EOF
}

set_cli_value() {
  local key="$1"
  local value="$2"
  case "$key" in
    project-dir) PROJECT_DIR="$value" ;;
    python-env) PYTHON_ENV="$value" ;;
    python-bin) PYTHON_BIN="$value"; PYTHON_BIN_EXPLICIT=1 ;;
    python-site-overlay) PYTHON_SITE_OVERLAY="$value" ;;
    home-value) HOME_VALUE="$value" ;;
    data-dir) DATA_DIR="$value" ;;
    text-model-path) TEXT_MODEL_PATH="$value" ;;
    max-text-tokens) MAX_TEXT_TOKENS="$value" ;;
    text-conditioning) TEXT_CONDITIONING="$value" ;;
    rotation-rep) ROTATION_REP="$value" ;;
    motion-space) MOTION_SPACE="$value" ;;
    vae-checkpoint) VAE_CHECKPOINT="$value" ;;
    source-mode) SOURCE_MODE="$value" ;;
    word-data-dir) WORD_DATA_DIR="$value" ;;
    word-split) WORD_SPLIT="$value" ;;
    condition-field) CONDITION_FIELD="$value" ;;
    adapter-checkpoint) ADAPTER_CHECKPOINT="$value" ;;
    residual-noise-scale) RESIDUAL_NOISE_SCALE="$value" ;;
    run-name) RUN_NAME="$value" ;;
    out-dir) OUT_DIR="$value"; OUT_DIR_EXPLICIT=1 ;;
    resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$value" ;;
    limit-train) LIMIT_TRAIN="$value" ;;
    limit-val) LIMIT_VAL="$value" ;;
    batch-size) BATCH_SIZE="$value" ;;
    epochs) EPOCHS="$value" ;;
    model-size) MODEL_SIZE="$value" ;;
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
    latent-loss-weight) LATENT_LOSS_WEIGHT="$value" ;;
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
    save-top-k) SAVE_TOP_K="$value" ;;
    val-every) VAL_EVERY="$value" ;;
    sample-steps) SAMPLE_STEPS="$value" ;;
    seed) SEED="$value" ;;
    use-gpus) USE_GPUS="$value" ;;
    device) DEVICE="$value" ;;
    distributed) DISTRIBUTED="$value" ;;
    ddp-backend) DDP_BACKEND="$value" ;;
    ddp-timeout-min) DDP_TIMEOUT_MIN="$value" ;;
    master-port) MASTER_PORT="$value" ;;
    wandb-project) WANDB_PROJECT="$value" ;;
    wandb-mode) WANDB_MODE="$value" ;;
    wandb-id) WANDB_ID="$value" ;;
    wandb-resume) WANDB_RESUME="$value" ;;
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
    --resume-without-optimizer)
      RESUME_WITHOUT_OPTIMIZER=1
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
MODEL_SIZE="$(echo "$MODEL_SIZE" | tr '[:upper:]' '[:lower:]')"
case "$TEXT_CONDITIONING" in
  pooled|token_prefix)
    ;;
  *)
    echo "ERROR: --text-conditioning must be pooled or token_prefix; got $TEXT_CONDITIONING" >&2
    exit 2
    ;;
esac
case "$ROTATION_REP" in
  axis_angle|rot6d)
    ;;
  *)
    echo "ERROR: --rotation-rep must be axis_angle or rot6d; got $ROTATION_REP" >&2
    exit 2
    ;;
esac
case "$MOTION_SPACE" in
  smplx|latent)
    ;;
  *)
    echo "ERROR: --motion-space must be smplx or latent; got $MOTION_SPACE" >&2
    exit 2
    ;;
esac
if [[ "$MOTION_SPACE" == "latent" && ! -f "$VAE_CHECKPOINT" ]]; then
  echo "ERROR: --motion-space latent requires an existing --vae-checkpoint: $VAE_CHECKPOINT" >&2
  exit 1
fi
case "$SOURCE_MODE" in
  noise|residual|adapter_residual)
    ;;
  *)
    echo "ERROR: --source-mode must be noise, residual, or adapter_residual; got $SOURCE_MODE" >&2
    exit 2
    ;;
esac
if [[ "$SOURCE_MODE" == "residual" && -z "$WORD_DATA_DIR" ]]; then
  echo "ERROR: --source-mode residual requires --word-data-dir" >&2
  exit 2
fi
if [[ "$SOURCE_MODE" == "adapter_residual" ]]; then
  if [[ "$MOTION_SPACE" != "latent" ]]; then
    echo "ERROR: --source-mode adapter_residual requires --motion-space latent" >&2
    exit 2
  fi
  if [[ -z "$ADAPTER_CHECKPOINT" ]]; then
    echo "ERROR: --source-mode adapter_residual requires --adapter-checkpoint" >&2
    exit 2
  fi
  if [[ ! -f "$ADAPTER_CHECKPOINT" ]]; then
    echo "ERROR: --adapter-checkpoint does not exist: $ADAPTER_CHECKPOINT" >&2
    exit 1
  fi
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && ! -f "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "ERROR: --resume-from-checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi
case "$MODEL_SIZE" in
  custom)
    ;;
  small)
    HIDDEN_DIM=256
    NUM_LAYERS=4
    NUM_HEADS=4
    ;;
  base)
    HIDDEN_DIM=512
    NUM_LAYERS=8
    NUM_HEADS=8
    ;;
  large)
    HIDDEN_DIM=768
    NUM_LAYERS=12
    NUM_HEADS=12
    ;;
  xl)
    HIDDEN_DIM=1024
    NUM_LAYERS=16
    NUM_HEADS=16
    ;;
  *)
    echo "ERROR: --model-size must be custom, small, base, large, or xl; got $MODEL_SIZE" >&2
    exit 2
    ;;
esac

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
if [[ -n "$PYTHON_SITE_OVERLAY" && ! -d "$PYTHON_SITE_OVERLAY" ]]; then
  echo "WARNING: python site overlay does not exist, ignoring: $PYTHON_SITE_OVERLAY" >&2
  PYTHON_SITE_OVERLAY=""
fi
if [[ -n "$PYTHON_SITE_OVERLAY" ]]; then
  export PYTHONPATH="$PROJECT_DIR:$PYTHON_SITE_OVERLAY:${PYTHONPATH:-}"
else
  export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
fi
export HOME="$HOME_VALUE"
if [[ "$USE_GPUS" != "auto" ]]; then
  export CUDA_VISIBLE_DEVICES="$USE_GPUS"
fi
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
  "$PROJECT_DIR/logs/sbatch/flow" \
  "$PROJECT_DIR/logs/wandb" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$PROJECT_DIR"

TRAIN_CMD=(
  "$PYTHON_BIN" -m flow.train_text_conditional
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --text_model_path "$TEXT_MODEL_PATH"
  --max_text_tokens "$MAX_TEXT_TOKENS"
  --text_conditioning "$TEXT_CONDITIONING"
  --rotation_rep "$ROTATION_REP"
  --motion_space "$MOTION_SPACE"
  --vae_checkpoint "$VAE_CHECKPOINT"
  --source_mode "$SOURCE_MODE"
  --word_split "$WORD_SPLIT"
  --condition_field "$CONDITION_FIELD"
  --residual_noise_scale "$RESIDUAL_NOISE_SCALE"
  --limit_train "$LIMIT_TRAIN"
  --limit_val "$LIMIT_VAL"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --model_size "$MODEL_SIZE"
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
  --latent_loss_weight "$LATENT_LOSS_WEIGHT"
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
  --save_top_k "$SAVE_TOP_K"
  --val_every "$VAL_EVERY"
  --sample_steps "$SAMPLE_STEPS"
  --seed "$SEED"
  --device "$DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)
if [[ -n "$WORD_DATA_DIR" ]]; then
  TRAIN_CMD+=(--word_data_dir "$WORD_DATA_DIR")
fi
if [[ -n "$ADAPTER_CHECKPOINT" ]]; then
  TRAIN_CMD+=(--adapter_checkpoint "$ADAPTER_CHECKPOINT")
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  TRAIN_CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ "$RESUME_WITHOUT_OPTIMIZER" == "1" ]]; then
  TRAIN_CMD+=(--resume_without_optimizer)
fi
if [[ "$NO_RANDOM_CROP" == "1" ]]; then
  TRAIN_CMD+=(--no_random_crop)
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

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURM_NODELIST:-local}"
echo "Start time   : $(date)"
echo "Project      : $PROJECT_DIR"
echo "Python       : $PYTHON_BIN"
echo "Python overlay: ${PYTHON_SITE_OVERLAY:-unset}"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-slurm/default}"
echo "DDP          : distributed=$DISTRIBUTED backend=$DDP_BACKEND timeout_min=$DDP_TIMEOUT_MIN master=${MASTER_ADDR:-unset}:${MASTER_PORT}"
echo "Slurm tasks  : nodes=${SLURM_NNODES:-local} ntasks=${SLURM_NTASKS:-local} tasks_per_node=${SLURM_TASKS_PER_NODE:-local}"
echo "Data dir     : $DATA_DIR"
echo "Text model   : $TEXT_MODEL_PATH tokens=$MAX_TEXT_TOKENS frozen=1 conditioning=$TEXT_CONDITIONING condition_field=$CONDITION_FIELD"
echo "Rotation rep : $ROTATION_REP"
echo "Motion space : $MOTION_SPACE vae_checkpoint=$VAE_CHECKPOINT"
echo "Flow source  : mode=$SOURCE_MODE word_data_dir=${WORD_DATA_DIR:-unset} word_split=$WORD_SPLIT adapter_checkpoint=${ADAPTER_CHECKPOINT:-unset} residual_noise_scale=$RESIDUAL_NOISE_SCALE"
echo "Output dir   : $OUT_DIR"
echo "Resume       : checkpoint=${RESUME_FROM_CHECKPOINT:-unset} optimizer=$((1 - RESUME_WITHOUT_OPTIMIZER))"
echo "Train limit  : train=$LIMIT_TRAIN val=$LIMIT_VAL"
echo "Model        : size=$MODEL_SIZE hidden=$HIDDEN_DIM layers=$NUM_LAYERS heads=$NUM_HEADS dropout=$DROPOUT"
echo "Training     : epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR wd=$WEIGHT_DECAY"
echo "Flow/noise   : noise_samples=$NOISE_SAMPLES noise_smoothing=$NOISE_SMOOTHING sampler=$SAMPLER sample_steps=$SAMPLE_STEPS"
echo "Checkpoints  : save_every=$SAVE_EVERY save_last_every=$SAVE_LAST_EVERY save_top_k=$SAVE_TOP_K val_every=$VAL_EVERY"
echo "Loss weights : latent=$LATENT_LOSS_WEIGHT pose=$POSE_LOSS_WEIGHT vel=$VELOCITY_LOSS_WEIGHT accel=$ACCEL_LOSS_WEIGHT hand=$HAND_WEIGHT"
echo "Random crop  : enabled=$((1 - NO_RANDOM_CROP))"
echo "W&B          : enabled=$WANDB mode=$WANDB_MODE project=$WANDB_PROJECT run=$WANDB_RUN_NAME id=${WANDB_ID:-unset} resume=${WANDB_RESUME:-unset} api_key=$WANDB_API_KEY_STATUS api_key_file=$WANDB_API_KEY_FILE_STATUS"
echo "Train cmd    : ${TRAIN_CMD[*]}"
echo "========================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, command was not executed."
  exit 0
fi

if [[ "$WANDB" == "1" && "$WANDB_MODE" == "online" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "W&B online authentication: WANDB_API_KEY is set and hidden from logs."
  else
    echo "WARNING: W&B online mode is enabled, but WANDB_API_KEY is empty; this requires an existing wandb login on the node." >&2
  fi
fi

echo "Launching text-conditioned flow training..."
SRUN_CMD=(srun)
if [[ "$DISTRIBUTED" != "none" && -n "${SLURM_NTASKS:-}" ]]; then
  SRUN_CMD+=(--ntasks "$SLURM_NTASKS")
fi
echo "Srun cmd     : ${SRUN_CMD[*]}"
"${SRUN_CMD[@]}" "${TRAIN_CMD[@]}"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
