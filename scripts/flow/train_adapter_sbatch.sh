#!/bin/bash
#SBATCH --job-name=soke_flow_adapter
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/adapter/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/adapter/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

set -eo pipefail
trap 'echo "ERROR: train_adapter_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

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

DATA_DIR="${DATA_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175}"
WORD_DATA_DIR="${WORD_DATA_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word}"
WORD_SPLIT="${WORD_SPLIT:-train}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt}"
STATS_DATA_DIR="${STATS_DATA_DIR:-}"
PRIOR_MODE="${PRIOR_MODE:-concat}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-deps/flan-t5-base}"
MAX_TEXT_TOKENS="${MAX_TEXT_TOKENS:-64}"
CONDITION_FIELD="${CONDITION_FIELD:-text}"

RUN_NAME="${RUN_NAME:-chatsign175_adapter_jointvae_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUT_DIR="${OUT_DIR:-experiments/flow/adapter/$RUN_NAME}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_WITHOUT_OPTIMIZER="${RESUME_WITHOUT_OPTIMIZER:-0}"

LIMIT_TRAIN="${LIMIT_TRAIN:-0}"
LIMIT_VAL="${LIMIT_VAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-1000}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
CONTENT_DIM="${CONTENT_DIM:-256}"
STYLE_DIM="${STYLE_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
DROPOUT="${DROPOUT:-0.0}"
ARRANGER_HIDDEN_DIM="${ARRANGER_HIDDEN_DIM:-512}"
ARRANGER_NUM_HEADS="${ARRANGER_NUM_HEADS:-8}"
ARRANGER_DROPOUT="${ARRANGER_DROPOUT:-0.0}"
MAX_WORD_LATENT_FRAMES="${MAX_WORD_LATENT_FRAMES:-64}"
NUM_WORD_CANDIDATES="${NUM_WORD_CANDIDATES:-32}"
NUM_NEGATIVE_CANDIDATES="${NUM_NEGATIVE_CANDIDATES:-16}"
CANDIDATE_SELECTION="${CANDIDATE_SELECTION:-flat}"
MAX_POSITIVE_VARIANTS_PER_KEY="${MAX_POSITIVE_VARIANTS_PER_KEY:-0}"
SHUFFLE_WORD_CANDIDATES="${SHUFFLE_WORD_CANDIDATES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-16}"
MIN_FRAMES="${MIN_FRAMES:-40}"
MAX_FRAMES="${MAX_FRAMES:-400}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
HAND_WEIGHT="${HAND_WEIGHT:-5.0}"
JAW_WEIGHT="${JAW_WEIGHT:-2.0}"
EXPRESSION_WEIGHT="${EXPRESSION_WEIGHT:-2.0}"
HAND_VALID_FLOOR="${HAND_VALID_FLOOR:-0.2}"
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-1.0}"
POSE_LOSS_WEIGHT="${POSE_LOSS_WEIGHT:-0.5}"
VELOCITY_LOSS_WEIGHT="${VELOCITY_LOSS_WEIGHT:-0.5}"
ACCEL_LOSS_WEIGHT="${ACCEL_LOSS_WEIGHT:-0.25}"
JERK_LOSS_WEIGHT="${JERK_LOSS_WEIGHT:-0.1}"
STYLE_LOSS_WEIGHT="${STYLE_LOSS_WEIGHT:-0.1}"
CONTENT_PAIR_LOSS_WEIGHT="${CONTENT_PAIR_LOSS_WEIGHT:-0.1}"
DELTA_LOSS_WEIGHT="${DELTA_LOSS_WEIGHT:-0.001}"
ORTH_LOSS_WEIGHT="${ORTH_LOSS_WEIGHT:-0.0}"
CONTENT_DOMAIN_CONFUSION_LOSS_WEIGHT="${CONTENT_DOMAIN_CONFUSION_LOSS_WEIGHT:-0.0}"
GRADIENT_REVERSAL_LAMBDA="${GRADIENT_REVERSAL_LAMBDA:-1.0}"
ARRANGER_PRIOR_LOSS_WEIGHT="${ARRANGER_PRIOR_LOSS_WEIGHT:-1.0}"
GATE_BCE_LOSS_WEIGHT="${GATE_BCE_LOSS_WEIGHT:-0.1}"
GATE_SPARSITY_LOSS_WEIGHT="${GATE_SPARSITY_LOSS_WEIGHT:-0.01}"
ATTENTION_SMOOTHNESS_WEIGHT="${ATTENTION_SMOOTHNESS_WEIGHT:-0.0}"
NULL_USAGE_LOSS_WEIGHT="${NULL_USAGE_LOSS_WEIGHT:-0.01}"
GROUP_COVERAGE_LOSS_WEIGHT="${GROUP_COVERAGE_LOSS_WEIGHT:-0.02}"
GROUP_COVERAGE_MASS="${GROUP_COVERAGE_MASS:-0.5}"
GROUP_ENTROPY_PEAK_LOSS_WEIGHT="${GROUP_ENTROPY_PEAK_LOSS_WEIGHT:-0.0}"
GROUP_ENTROPY_PEAK_TARGET="${GROUP_ENTROPY_PEAK_TARGET:-0.6931471805599453}"
ATTENTION_VARIATION_LOSS_WEIGHT="${ATTENTION_VARIATION_LOSS_WEIGHT:-0.01}"
ATTENTION_VARIATION_TARGET="${ATTENTION_VARIATION_TARGET:-0.05}"
PRIOR_VELOCITY_LOSS_WEIGHT="${PRIOR_VELOCITY_LOSS_WEIGHT:-0.25}"
PRIOR_ACCEL_LOSS_WEIGHT="${PRIOR_ACCEL_LOSS_WEIGHT:-0.05}"
PRIOR_VARIANCE_FLOOR_LOSS_WEIGHT="${PRIOR_VARIANCE_FLOOR_LOSS_WEIGHT:-0.05}"
PRIOR_VARIANCE_FLOOR_RATIO="${PRIOR_VARIANCE_FLOOR_RATIO:-0.5}"
NEGATIVE_USAGE_LOSS_WEIGHT="${NEGATIVE_USAGE_LOSS_WEIGHT:-0.02}"
VAL_EVERY="${VAL_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-100}"
SAVE_LAST_EVERY="${SAVE_LAST_EVERY:-10}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
USE_GPUS="${USE_GPUS:-auto}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DDP_BACKEND="${DDP_BACKEND:-auto}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-120}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANDOM_CROP="${RANDOM_CROP:-0}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-flow-adapter}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
WANDB_DISABLE_STATS="${WANDB_DISABLE_STATS:-1}"
DRY_RUN="${DRY_RUN:-0}"
DISABLE_SOFTARRANGER="${DISABLE_SOFTARRANGER:-0}"
DISABLE_ADAPTER="${DISABLE_ADAPTER:-0}"
DISABLE_ARRANGER_CANDIDATE_GATES="${DISABLE_ARRANGER_CANDIDATE_GATES:-0}"
DISABLE_ARRANGER_NULL_MEMORY="${DISABLE_ARRANGER_NULL_MEMORY:-0}"
DISABLE_ARRANGER_WORD_TEXT_FEATURES="${DISABLE_ARRANGER_WORD_TEXT_FEATURES:-0}"
DISABLE_ARRANGER_WORD_MOTION_LATENTS="${DISABLE_ARRANGER_WORD_MOTION_LATENTS:-0}"

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/flow/train_adapter_sbatch.sh [options]
  bash scripts/flow/train_adapter_sbatch.sh --dry-run [options]

Options can also be written as --key=value. CLI options override environment variables.

Paths:
  --project-dir PATH
  --python-env PATH
  --python-bin PATH
  --python-site-overlay PATH
  --home-value PATH
  --data-dir PATH
  --word-data-dir PATH
  --word-split SPLIT
  --vae-checkpoint PATH
  --stats-data-dir PATH
  --prior-mode concat|soft_arranger
  --disable-softarranger
  --disable-adapter
  --enable-adapter
  --text-model-path PATH
  --max-text-tokens N
  --condition-field text|gloss|text_gloss
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
  --content-dim N
  --style-dim N
  --num-layers N
  --num-heads N
  --dropout VALUE
  --arranger-hidden-dim N
  --arranger-num-heads N
  --arranger-dropout VALUE
  --max-word-latent-frames N
  --num-word-candidates N
  --num-negative-candidates N
  --candidate-selection flat|round_robin
  --max-positive-variants-per-key N
  --shuffle-word-candidates
  --no-shuffle-word-candidates
  --num-workers N
  --stats-batch-size N
  --min-frames N
  --max-frames N
  --lr VALUE
  --weight-decay VALUE
  --grad-clip VALUE
  --hand-weight VALUE
  --jaw-weight VALUE
  --expression-weight VALUE
  --hand-valid-floor VALUE
  --latent-loss-weight VALUE
  --pose-loss-weight VALUE
  --velocity-loss-weight VALUE
  --accel-loss-weight VALUE
  --jerk-loss-weight VALUE
  --style-loss-weight VALUE
  --content-pair-loss-weight VALUE
  --delta-loss-weight VALUE
  --orth-loss-weight VALUE
  --content-domain-confusion-loss-weight VALUE
  --gradient-reversal-lambda VALUE
  --arranger-prior-loss-weight VALUE
  --gate-bce-loss-weight VALUE
  --gate-sparsity-loss-weight VALUE
  --attention-smoothness-weight VALUE
  --null-usage-loss-weight VALUE
  --group-coverage-loss-weight VALUE
  --group-coverage-mass VALUE
  --group-entropy-peak-loss-weight VALUE
  --group-entropy-peak-target VALUE
  --attention-variation-loss-weight VALUE
  --attention-variation-target VALUE
  --prior-velocity-loss-weight VALUE
  --prior-accel-loss-weight VALUE
  --prior-variance-floor-loss-weight VALUE
  --prior-variance-floor-ratio VALUE
  --negative-usage-loss-weight VALUE
  --val-every N
  --save-every N
  --save-last-every N
  --save-top-k N
  --seed N
  --device auto|cuda|cpu
  --use-gpus auto|IDS
  --distributed auto|none|ddp
  --ddp-backend auto|nccl|gloo
  --ddp-timeout-min N
  --master-port N
  --random-crop
  --no-random-crop
  --disable-arranger-candidate-gates
  --disable-arranger-null-memory
  --disable-arranger-word-text-features
  --disable-arranger-word-motion-latents
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
    python-bin) PYTHON_BIN="$value"; PYTHON_BIN_EXPLICIT=1 ;;
    python-site-overlay) PYTHON_SITE_OVERLAY="$value" ;;
    home-value) HOME_VALUE="$value" ;;
    data-dir) DATA_DIR="$value" ;;
    word-data-dir) WORD_DATA_DIR="$value" ;;
    word-split) WORD_SPLIT="$value" ;;
    vae-checkpoint) VAE_CHECKPOINT="$value" ;;
    stats-data-dir) STATS_DATA_DIR="$value" ;;
    prior-mode) PRIOR_MODE="$value" ;;
    disable-softarranger) DISABLE_SOFTARRANGER="$value"; if [[ "$value" == "1" || "$value" == "true" ]]; then PRIOR_MODE=concat; fi ;;
    disable-adapter) DISABLE_ADAPTER="$value" ;;
    disable-arranger-candidate-gates) DISABLE_ARRANGER_CANDIDATE_GATES="$value" ;;
    disable-arranger-null-memory) DISABLE_ARRANGER_NULL_MEMORY="$value" ;;
    disable-arranger-word-text-features) DISABLE_ARRANGER_WORD_TEXT_FEATURES="$value" ;;
    disable-arranger-word-motion-latents) DISABLE_ARRANGER_WORD_MOTION_LATENTS="$value" ;;
    text-model-path) TEXT_MODEL_PATH="$value" ;;
    max-text-tokens) MAX_TEXT_TOKENS="$value" ;;
    condition-field) CONDITION_FIELD="$value" ;;
    run-name) RUN_NAME="$value" ;;
    out-dir) OUT_DIR="$value"; OUT_DIR_EXPLICIT=1 ;;
    resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$value" ;;
    limit-train) LIMIT_TRAIN="$value" ;;
    limit-val) LIMIT_VAL="$value" ;;
    batch-size) BATCH_SIZE="$value" ;;
    epochs) EPOCHS="$value" ;;
    hidden-dim) HIDDEN_DIM="$value" ;;
    content-dim) CONTENT_DIM="$value" ;;
    style-dim) STYLE_DIM="$value" ;;
    num-layers) NUM_LAYERS="$value" ;;
    num-heads) NUM_HEADS="$value" ;;
    dropout) DROPOUT="$value" ;;
    arranger-hidden-dim) ARRANGER_HIDDEN_DIM="$value" ;;
    arranger-num-heads) ARRANGER_NUM_HEADS="$value" ;;
    arranger-dropout) ARRANGER_DROPOUT="$value" ;;
    max-word-latent-frames) MAX_WORD_LATENT_FRAMES="$value" ;;
    num-word-candidates) NUM_WORD_CANDIDATES="$value" ;;
    num-negative-candidates) NUM_NEGATIVE_CANDIDATES="$value" ;;
    candidate-selection) CANDIDATE_SELECTION="$value" ;;
    max-positive-variants-per-key) MAX_POSITIVE_VARIANTS_PER_KEY="$value" ;;
    num-workers) NUM_WORKERS="$value" ;;
    stats-batch-size) STATS_BATCH_SIZE="$value" ;;
    min-frames) MIN_FRAMES="$value" ;;
    max-frames) MAX_FRAMES="$value" ;;
    lr) LR="$value" ;;
    weight-decay) WEIGHT_DECAY="$value" ;;
    grad-clip) GRAD_CLIP="$value" ;;
    hand-weight) HAND_WEIGHT="$value" ;;
    jaw-weight) JAW_WEIGHT="$value" ;;
    expression-weight) EXPRESSION_WEIGHT="$value" ;;
    hand-valid-floor) HAND_VALID_FLOOR="$value" ;;
    latent-loss-weight) LATENT_LOSS_WEIGHT="$value" ;;
    pose-loss-weight) POSE_LOSS_WEIGHT="$value" ;;
    velocity-loss-weight) VELOCITY_LOSS_WEIGHT="$value" ;;
    accel-loss-weight) ACCEL_LOSS_WEIGHT="$value" ;;
    jerk-loss-weight) JERK_LOSS_WEIGHT="$value" ;;
    style-loss-weight) STYLE_LOSS_WEIGHT="$value" ;;
    content-pair-loss-weight) CONTENT_PAIR_LOSS_WEIGHT="$value" ;;
    delta-loss-weight) DELTA_LOSS_WEIGHT="$value" ;;
    orth-loss-weight) ORTH_LOSS_WEIGHT="$value" ;;
    content-domain-confusion-loss-weight) CONTENT_DOMAIN_CONFUSION_LOSS_WEIGHT="$value" ;;
    gradient-reversal-lambda) GRADIENT_REVERSAL_LAMBDA="$value" ;;
    arranger-prior-loss-weight) ARRANGER_PRIOR_LOSS_WEIGHT="$value" ;;
    gate-bce-loss-weight) GATE_BCE_LOSS_WEIGHT="$value" ;;
    gate-sparsity-loss-weight) GATE_SPARSITY_LOSS_WEIGHT="$value" ;;
    attention-smoothness-weight) ATTENTION_SMOOTHNESS_WEIGHT="$value" ;;
    null-usage-loss-weight) NULL_USAGE_LOSS_WEIGHT="$value" ;;
    group-coverage-loss-weight) GROUP_COVERAGE_LOSS_WEIGHT="$value" ;;
    group-coverage-mass) GROUP_COVERAGE_MASS="$value" ;;
    group-entropy-peak-loss-weight) GROUP_ENTROPY_PEAK_LOSS_WEIGHT="$value" ;;
    group-entropy-peak-target) GROUP_ENTROPY_PEAK_TARGET="$value" ;;
    attention-variation-loss-weight) ATTENTION_VARIATION_LOSS_WEIGHT="$value" ;;
    attention-variation-target) ATTENTION_VARIATION_TARGET="$value" ;;
    prior-velocity-loss-weight) PRIOR_VELOCITY_LOSS_WEIGHT="$value" ;;
    prior-accel-loss-weight) PRIOR_ACCEL_LOSS_WEIGHT="$value" ;;
    prior-variance-floor-loss-weight) PRIOR_VARIANCE_FLOOR_LOSS_WEIGHT="$value" ;;
    prior-variance-floor-ratio) PRIOR_VARIANCE_FLOOR_RATIO="$value" ;;
    negative-usage-loss-weight) NEGATIVE_USAGE_LOSS_WEIGHT="$value" ;;
    val-every) VAL_EVERY="$value" ;;
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
    --no-wandb)
      WANDB=0
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-dry-run)
      DRY_RUN=0
      shift
      ;;
    --disable-softarranger)
      DISABLE_SOFTARRANGER=1
      PRIOR_MODE=concat
      shift
      ;;
    --disable-adapter)
      DISABLE_ADAPTER=1
      shift
      ;;
    --enable-adapter)
      DISABLE_ADAPTER=0
      shift
      ;;
    --disable-arranger-candidate-gates)
      DISABLE_ARRANGER_CANDIDATE_GATES=1
      shift
      ;;
    --disable-arranger-null-memory)
      DISABLE_ARRANGER_NULL_MEMORY=1
      shift
      ;;
    --disable-arranger-word-text-features)
      DISABLE_ARRANGER_WORD_TEXT_FEATURES=1
      shift
      ;;
    --disable-arranger-word-motion-latents)
      DISABLE_ARRANGER_WORD_MOTION_LATENTS=1
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
    --shuffle-word-candidates)
      SHUFFLE_WORD_CANDIDATES=1
      shift
      ;;
    --no-shuffle-word-candidates)
      SHUFFLE_WORD_CANDIDATES=0
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

if [[ "$PYTHON_BIN_EXPLICIT" == "0" ]]; then
  PYTHON_BIN="$PYTHON_ENV/bin/python"
fi
if [[ "$OUT_DIR_EXPLICIT" == "0" ]]; then
  OUT_DIR="experiments/flow/adapter/$RUN_NAME"
fi
if [[ "$DISABLE_SOFTARRANGER" == "1" || "$DISABLE_SOFTARRANGER" == "true" ]]; then
  PRIOR_MODE=concat
fi
if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "ERROR: --wandb-api-key-file does not exist: $WANDB_API_KEY_FILE" >&2
    exit 1
  fi
  WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" ]]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY="${WANDB_API_KEY:-\(.*\)}"$/\1/p' "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" | head -n 1)"
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && ! -f "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "ERROR: --resume-from-checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

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
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
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
  "$PROJECT_DIR/logs/sbatch/flow/adapter" \
  "$PROJECT_DIR/logs/wandb" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$PROJECT_DIR"

TRAIN_CMD=(
  "$PYTHON_BIN" -m flow.train_adapter
  --data_dir "$DATA_DIR"
  --word_data_dir "$WORD_DATA_DIR"
  --word_split "$WORD_SPLIT"
  --vae_checkpoint "$VAE_CHECKPOINT"
  --prior_mode "$PRIOR_MODE"
  --text_model_path "$TEXT_MODEL_PATH"
  --max_text_tokens "$MAX_TEXT_TOKENS"
  --condition_field "$CONDITION_FIELD"
  --out_dir "$OUT_DIR"
  --limit_train "$LIMIT_TRAIN"
  --limit_val "$LIMIT_VAL"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --hidden_dim "$HIDDEN_DIM"
  --content_dim "$CONTENT_DIM"
  --style_dim "$STYLE_DIM"
  --num_layers "$NUM_LAYERS"
  --num_heads "$NUM_HEADS"
  --dropout "$DROPOUT"
  --arranger_hidden_dim "$ARRANGER_HIDDEN_DIM"
  --arranger_num_heads "$ARRANGER_NUM_HEADS"
  --arranger_dropout "$ARRANGER_DROPOUT"
  --max_word_latent_frames "$MAX_WORD_LATENT_FRAMES"
  --num_word_candidates "$NUM_WORD_CANDIDATES"
  --num_negative_candidates "$NUM_NEGATIVE_CANDIDATES"
  --candidate_selection "$CANDIDATE_SELECTION"
  --max_positive_variants_per_key "$MAX_POSITIVE_VARIANTS_PER_KEY"
  --num_workers "$NUM_WORKERS"
  --stats_batch_size "$STATS_BATCH_SIZE"
  --min_frames "$MIN_FRAMES"
  --max_frames "$MAX_FRAMES"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --hand_weight "$HAND_WEIGHT"
  --jaw_weight "$JAW_WEIGHT"
  --expression_weight "$EXPRESSION_WEIGHT"
  --hand_valid_floor "$HAND_VALID_FLOOR"
  --latent_loss_weight "$LATENT_LOSS_WEIGHT"
  --pose_loss_weight "$POSE_LOSS_WEIGHT"
  --velocity_loss_weight "$VELOCITY_LOSS_WEIGHT"
  --accel_loss_weight "$ACCEL_LOSS_WEIGHT"
  --jerk_loss_weight "$JERK_LOSS_WEIGHT"
  --style_loss_weight "$STYLE_LOSS_WEIGHT"
  --content_pair_loss_weight "$CONTENT_PAIR_LOSS_WEIGHT"
  --delta_loss_weight "$DELTA_LOSS_WEIGHT"
  --orth_loss_weight "$ORTH_LOSS_WEIGHT"
  --content_domain_confusion_loss_weight "$CONTENT_DOMAIN_CONFUSION_LOSS_WEIGHT"
  --gradient_reversal_lambda "$GRADIENT_REVERSAL_LAMBDA"
  --arranger_prior_loss_weight "$ARRANGER_PRIOR_LOSS_WEIGHT"
  --gate_bce_loss_weight "$GATE_BCE_LOSS_WEIGHT"
  --gate_sparsity_loss_weight "$GATE_SPARSITY_LOSS_WEIGHT"
  --attention_smoothness_weight "$ATTENTION_SMOOTHNESS_WEIGHT"
  --null_usage_loss_weight "$NULL_USAGE_LOSS_WEIGHT"
  --group_coverage_loss_weight "$GROUP_COVERAGE_LOSS_WEIGHT"
  --group_coverage_mass "$GROUP_COVERAGE_MASS"
  --group_entropy_peak_loss_weight "$GROUP_ENTROPY_PEAK_LOSS_WEIGHT"
  --group_entropy_peak_target "$GROUP_ENTROPY_PEAK_TARGET"
  --attention_variation_loss_weight "$ATTENTION_VARIATION_LOSS_WEIGHT"
  --attention_variation_target "$ATTENTION_VARIATION_TARGET"
  --prior_velocity_loss_weight "$PRIOR_VELOCITY_LOSS_WEIGHT"
  --prior_accel_loss_weight "$PRIOR_ACCEL_LOSS_WEIGHT"
  --prior_variance_floor_loss_weight "$PRIOR_VARIANCE_FLOOR_LOSS_WEIGHT"
  --prior_variance_floor_ratio "$PRIOR_VARIANCE_FLOOR_RATIO"
  --negative_usage_loss_weight "$NEGATIVE_USAGE_LOSS_WEIGHT"
  --val_every "$VAL_EVERY"
  --save_every "$SAVE_EVERY"
  --save_last_every "$SAVE_LAST_EVERY"
  --save_top_k "$SAVE_TOP_K"
  --seed "$SEED"
  --device "$DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)
if [[ -n "$STATS_DATA_DIR" ]]; then
  TRAIN_CMD+=(--stats_data_dir "$STATS_DATA_DIR")
fi
if [[ "$RANDOM_CROP" == "1" ]]; then
  TRAIN_CMD+=(--random_crop)
else
  TRAIN_CMD+=(--no_random_crop)
fi
if [[ "$SHUFFLE_WORD_CANDIDATES" == "1" ]]; then
  TRAIN_CMD+=(--shuffle_word_candidates)
else
  TRAIN_CMD+=(--no_shuffle_word_candidates)
fi
if [[ "$DISABLE_SOFTARRANGER" == "1" || "$DISABLE_SOFTARRANGER" == "true" ]]; then
  TRAIN_CMD+=(--disable_softarranger)
fi
if [[ "$DISABLE_ADAPTER" == "1" || "$DISABLE_ADAPTER" == "true" ]]; then
  TRAIN_CMD+=(--disable_adapter)
fi
if [[ "$DISABLE_ARRANGER_CANDIDATE_GATES" == "1" || "$DISABLE_ARRANGER_CANDIDATE_GATES" == "true" ]]; then
  TRAIN_CMD+=(--disable_arranger_candidate_gates)
fi
if [[ "$DISABLE_ARRANGER_NULL_MEMORY" == "1" || "$DISABLE_ARRANGER_NULL_MEMORY" == "true" ]]; then
  TRAIN_CMD+=(--disable_arranger_null_memory)
fi
if [[ "$DISABLE_ARRANGER_WORD_TEXT_FEATURES" == "1" || "$DISABLE_ARRANGER_WORD_TEXT_FEATURES" == "true" ]]; then
  TRAIN_CMD+=(--disable_arranger_word_text_features)
fi
if [[ "$DISABLE_ARRANGER_WORD_MOTION_LATENTS" == "1" || "$DISABLE_ARRANGER_WORD_MOTION_LATENTS" == "true" ]]; then
  TRAIN_CMD+=(--disable_arranger_word_motion_latents)
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
echo "DDP          : distributed=$DISTRIBUTED backend=$DDP_BACKEND timeout_min=$DDP_TIMEOUT_MIN master=${MASTER_ADDR:-unset}:${MASTER_PORT}"
echo "Slurm tasks  : nodes=${SLURM_NNODES:-local} ntasks=${SLURM_NTASKS:-local} tasks_per_node=${SLURM_TASKS_PER_NODE:-local}"
echo "Data dir     : $DATA_DIR"
echo "Word dir     : $WORD_DATA_DIR split=$WORD_SPLIT"
echo "VAE          : $VAE_CHECKPOINT"
echo "Prior mode   : $PRIOR_MODE"
echo "Ablation     : disable_softarranger=$DISABLE_SOFTARRANGER disable_adapter=$DISABLE_ADAPTER"
echo "SWA ablation : disable_gates=$DISABLE_ARRANGER_CANDIDATE_GATES disable_null=$DISABLE_ARRANGER_NULL_MEMORY disable_word_text=$DISABLE_ARRANGER_WORD_TEXT_FEATURES disable_word_motion=$DISABLE_ARRANGER_WORD_MOTION_LATENTS"
echo "Text model   : $TEXT_MODEL_PATH max_tokens=$MAX_TEXT_TOKENS condition_field=$CONDITION_FIELD"
echo "Stats dir    : ${STATS_DATA_DIR:-from VAE checkpoint}"
echo "Output dir   : $OUT_DIR"
echo "Resume       : checkpoint=${RESUME_FROM_CHECKPOINT:-unset} optimizer=$((1 - RESUME_WITHOUT_OPTIMIZER))"
echo "Model        : hidden=$HIDDEN_DIM content=$CONTENT_DIM style=$STYLE_DIM layers=$NUM_LAYERS heads=$NUM_HEADS dropout=$DROPOUT"
echo "Arranger     : hidden=$ARRANGER_HIDDEN_DIM heads=$ARRANGER_NUM_HEADS dropout=$ARRANGER_DROPOUT max_word_latent=$MAX_WORD_LATENT_FRAMES candidates=$NUM_WORD_CANDIDATES negatives=$NUM_NEGATIVE_CANDIDATES selection=$CANDIDATE_SELECTION max_pos_variants_per_key=$MAX_POSITIVE_VARIANTS_PER_KEY shuffle=$SHUFFLE_WORD_CANDIDATES"
echo "Training     : epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR wd=$WEIGHT_DECAY random_crop=$RANDOM_CROP"
echo "Loss weights : latent=$LATENT_LOSS_WEIGHT pose=$POSE_LOSS_WEIGHT vel=$VELOCITY_LOSS_WEIGHT accel=$ACCEL_LOSS_WEIGHT jerk=$JERK_LOSS_WEIGHT style=$STYLE_LOSS_WEIGHT content_pair=$CONTENT_PAIR_LOSS_WEIGHT delta=$DELTA_LOSS_WEIGHT orth=$ORTH_LOSS_WEIGHT content_domain_confusion=$CONTENT_DOMAIN_CONFUSION_LOSS_WEIGHT grl=$GRADIENT_REVERSAL_LAMBDA"
echo "Arranger loss: prior=$ARRANGER_PRIOR_LOSS_WEIGHT gate_bce=$GATE_BCE_LOSS_WEIGHT gate_sparsity=$GATE_SPARSITY_LOSS_WEIGHT attn_smooth=$ATTENTION_SMOOTHNESS_WEIGHT null_usage=$NULL_USAGE_LOSS_WEIGHT"
echo "Anti-collapse: group_cov=$GROUP_COVERAGE_LOSS_WEIGHT@$GROUP_COVERAGE_MASS group_entropy_peak=$GROUP_ENTROPY_PEAK_LOSS_WEIGHT@$GROUP_ENTROPY_PEAK_TARGET attn_var=$ATTENTION_VARIATION_LOSS_WEIGHT@$ATTENTION_VARIATION_TARGET prior_vel=$PRIOR_VELOCITY_LOSS_WEIGHT prior_accel=$PRIOR_ACCEL_LOSS_WEIGHT var_floor=$PRIOR_VARIANCE_FLOOR_LOSS_WEIGHT@$PRIOR_VARIANCE_FLOOR_RATIO neg_usage=$NEGATIVE_USAGE_LOSS_WEIGHT"
echo "Checkpoints  : val_every=$VAL_EVERY save_every=$SAVE_EVERY save_last_every=$SAVE_LAST_EVERY save_top_k=$SAVE_TOP_K"
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

echo "Launching content-style adapter training..."
srun "${TRAIN_CMD[@]}"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
