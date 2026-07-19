#!/bin/bash
#SBATCH --job-name=soke_flow_ctc
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/flow/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

set -eo pipefail
trap 'echo "ERROR: train_ctc_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/soke}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
PYTHON_SITE_OVERLAY="${PYTHON_SITE_OVERLAY:-/media/cvpr/haomian/python_user_site_wandb_0_27_min}"
HOME_VALUE="${HOME_VALUE:-/media/cvpr/haomian}"

DATA_DIR="${DATA_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx}"
VOCAB_PATH="${VOCAB_PATH:-}"
RUN_NAME="${RUN_NAME:-phoenix_ctc_b48_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUT_DIR="${OUT_DIR:-experiments/flow/align/$RUN_NAME}"

FEATURES="${FEATURES:-motion}"
APPEND_VALID="${APPEND_VALID:-0}"
GATE_HANDS="${GATE_HANDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-48}"
EPOCHS="${EPOCHS:-150}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DROPOUT="${DROPOUT:-0.2}"
MODEL_DIM="${MODEL_DIM:-256}"
CONV_LAYERS="${CONV_LAYERS:-3}"
CONV_KERNEL="${CONV_KERNEL:-5}"
LSTM_HIDDEN="${LSTM_HIDDEN:-256}"
LSTM_LAYERS="${LSTM_LAYERS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
USE_GPUS="${USE_GPUS:-auto}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DDP_BACKEND="${DDP_BACKEND:-auto}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-60}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-1234}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
LIMIT_TRAIN="${LIMIT_TRAIN:-0}"
LIMIT_VAL="${LIMIT_VAL:-0}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-flow-align}"
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
  sbatch scripts/flow/train_ctc_sbatch.sh [options]
  bash scripts/flow/train_ctc_sbatch.sh --dry-run [options]

Options can also be written as --key=value. CLI options override environment variables.

Paths:
  --project-dir PATH
  --python-env PATH
  --python-bin PATH
  --python-site-overlay PATH
  --home-value PATH
  --data-dir PATH
  --vocab-path PATH
  --run-name NAME
  --out-dir PATH

Training:
  --features motion|motion_velocity
  --append-valid
  --gate-hands
  --batch-size N
  --epochs N
  --lr VALUE
  --weight-decay VALUE
  --warmup-epochs N
  --dropout VALUE
  --model-dim N
  --conv-layers N
  --conv-kernel N
  --lstm-hidden N
  --lstm-layers N
  --num-workers N
  --device auto|cuda|cpu
  --use-gpus auto|IDS
  --distributed auto|none|ddp
  --ddp-backend auto|nccl|gloo
  --ddp-timeout-min N
  --master-port N
  --seed N
  --grad-clip VALUE
  --limit-train N
  --limit-val N
  --wandb
  --no-wandb
  --wandb-online
  --wandb-offline
  --wandb-project NAME
  --wandb-mode online|offline|disabled
  --wandb-id ID
  --wandb-resume allow|must|never|auto
  --wandb-api-key KEY
  --wandb-api-key-file PATH
  --wandb-disable-stats
  --wandb-enable-stats
  --dry-run
  --no-dry-run

  -h, --help
EOF
}

set_cli_value() {
  local key="$1"
  local value="$2"
  case "$key" in
    project-dir) PROJECT_DIR="$value" ;;
    python-env) PYTHON_ENV="$value"; PYTHON_BIN="$PYTHON_ENV/bin/python" ;;
    python-bin) PYTHON_BIN="$value" ;;
    python-site-overlay) PYTHON_SITE_OVERLAY="$value" ;;
    home-value) HOME_VALUE="$value" ;;
    data-dir) DATA_DIR="$value" ;;
    vocab-path) VOCAB_PATH="$value" ;;
    run-name) RUN_NAME="$value"; OUT_DIR="experiments/flow/align/$value" ;;
    out-dir) OUT_DIR="$value" ;;
    features) FEATURES="$value" ;;
    batch-size) BATCH_SIZE="$value" ;;
    epochs) EPOCHS="$value" ;;
    lr) LR="$value" ;;
    weight-decay) WEIGHT_DECAY="$value" ;;
    warmup-epochs) WARMUP_EPOCHS="$value" ;;
    dropout) DROPOUT="$value" ;;
    model-dim) MODEL_DIM="$value" ;;
    conv-layers) CONV_LAYERS="$value" ;;
    conv-kernel) CONV_KERNEL="$value" ;;
    lstm-hidden) LSTM_HIDDEN="$value" ;;
    lstm-layers) LSTM_LAYERS="$value" ;;
    num-workers) NUM_WORKERS="$value" ;;
    device) DEVICE="$value" ;;
    use-gpus) USE_GPUS="$value" ;;
    distributed) DISTRIBUTED="$value" ;;
    ddp-backend) DDP_BACKEND="$value" ;;
    ddp-timeout-min) DDP_TIMEOUT_MIN="$value" ;;
    master-port) MASTER_PORT="$value" ;;
    seed) SEED="$value" ;;
    grad-clip) GRAD_CLIP="$value" ;;
    limit-train) LIMIT_TRAIN="$value" ;;
    limit-val) LIMIT_VAL="$value" ;;
    wandb-project) WANDB_PROJECT="$value" ;;
    wandb-mode) WANDB_MODE="$value" ;;
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
    --append-valid)
      APPEND_VALID=1
      shift
      ;;
    --no-append-valid)
      APPEND_VALID=0
      shift
      ;;
    --gate-hands)
      GATE_HANDS=1
      shift
      ;;
    --no-gate-hands)
      GATE_HANDS=0
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
    --*=*)
      key="${1%%=*}"
      value="${1#*=}"
      set_cli_value "${key#--}" "$value"
      shift
      ;;
    --*)
      key="${1#--}"
      if [[ $# -lt 2 ]]; then
        echo "ERROR: missing value for --$key" >&2
        exit 2
      fi
      set_cli_value "$key" "$2"
      shift 2
      ;;
    *)
      echo "ERROR: unexpected argument $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    echo "ERROR: --wandb-api-key-file does not exist: $WANDB_API_KEY_FILE" >&2
    exit 2
  fi
  WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" ]]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY="${WANDB_API_KEY:-\(.*\)}"$/\1/p' "$PROJECT_DIR/scripts/flow/train_overfit_unconditional_sbatch.sh" | head -n 1)"
fi
WANDB_API_KEY_STATUS="unset"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY_STATUS="set"
fi
WANDB_API_KEY_FILE_STATUS="unset"
if [[ -n "$WANDB_API_KEY_FILE" ]]; then
  WANDB_API_KEY_FILE_STATUS="set"
fi

cd "$PROJECT_DIR"
mkdir -p logs/sbatch/flow "$OUT_DIR"
export HOME="$HOME_VALUE"
if [[ -n "$PYTHON_SITE_OVERLAY" && -d "$PYTHON_SITE_OVERLAY" ]]; then
  export PYTHONPATH="$PROJECT_DIR:$PYTHON_SITE_OVERLAY:${PYTHONPATH:-}"
else
  export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
fi
export MASTER_PORT="$MASTER_PORT"
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
mkdir -p "$PROJECT_DIR/logs/wandb" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
if [[ "$USE_GPUS" != "auto" ]]; then
  export CUDA_VISIBLE_DEVICES="$USE_GPUS"
fi

cmd=(
  "$PYTHON_BIN" -m flow.align.train_ctc
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --features "$FEATURES"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_epochs "$WARMUP_EPOCHS"
  --dropout "$DROPOUT"
  --model_dim "$MODEL_DIM"
  --conv_layers "$CONV_LAYERS"
  --conv_kernel "$CONV_KERNEL"
  --lstm_hidden "$LSTM_HIDDEN"
  --lstm_layers "$LSTM_LAYERS"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
  --seed "$SEED"
  --grad_clip "$GRAD_CLIP"
  --limit_train "$LIMIT_TRAIN"
  --limit_val "$LIMIT_VAL"
)
if [[ -n "$VOCAB_PATH" ]]; then
  cmd+=(--vocab_path "$VOCAB_PATH")
fi
if [[ "$APPEND_VALID" == "1" ]]; then
  cmd+=(--append_valid)
fi
if [[ "$GATE_HANDS" == "1" ]]; then
  cmd+=(--gate_hands)
fi
if [[ "$WANDB" == "1" ]]; then
  cmd+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN_NAME")
  if [[ -n "$WANDB_ID" ]]; then
    cmd+=(--wandb_id "$WANDB_ID")
  fi
  if [[ -n "$WANDB_RESUME" ]]; then
    cmd+=(--wandb_resume "$WANDB_RESUME")
  fi
fi

echo "W&B: enabled=$WANDB mode=$WANDB_MODE project=$WANDB_PROJECT run=$WANDB_RUN_NAME api_key=$WANDB_API_KEY_STATUS api_key_file=$WANDB_API_KEY_FILE_STATUS"
echo "DDP: distributed=$DISTRIBUTED backend=$DDP_BACKEND timeout_min=$DDP_TIMEOUT_MIN master=${MASTER_ADDR:-unset}:$MASTER_PORT slurm_tasks=${SLURM_NTASKS:-local}"
if [[ "$WANDB" == "1" && "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: W&B online mode is enabled, but WANDB_API_KEY is empty; this requires an existing wandb login on the node." >&2
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

srun "${cmd[@]}"
