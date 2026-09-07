#!/bin/bash
#SBATCH --job-name=signtraj_slot_diag
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
trap 'echo "ERROR: diagnose_temporal_slots_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2.yaml}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2/checkpoints/epoch0002.pt}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2/evaluation/epoch0002_validation_slot_diagnostics}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${OUT_DIR}_smoke}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PERTURB_BATCH_SIZE="${PERTURB_BATCH_SIZE:-128}"
FK_BATCH_SIZE="${FK_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
VERIFY_HASHES="${VERIFY_HASHES:-1}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:-${SIGNTRAJ_SOURCE_GIT_HEAD:-}}"
LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID:?Submit this launcher through sbatch}"

for flag_name in RUN_SMOKE RUN_FULL VERIFY_HASHES; do
  flag_value="${!flag_name}"
  case "$flag_value" in
    0|1) ;;
    *) echo "ERROR: $flag_name must be 0 or 1" >&2; exit 1 ;;
  esac
done
if [[ "$RUN_SMOKE" == "0" && "$RUN_FULL" == "0" ]]; then
  echo "ERROR: at least one of RUN_SMOKE or RUN_FULL must be enabled" >&2
  exit 1
fi
if [[ ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ && ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: SOURCE_GIT_HEAD must be the submit-host git rev-parse HEAD value" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: configuration not found: $CFG" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$SENTENCE_MEMORY_DIR" || ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: sentence-memory bank is not ready: $SENTENCE_MEMORY_DIR" >&2
  exit 1
fi

cleanup_sentence_memory() {
  local exit_code=$?
  trap - EXIT
  srun --nodes=1 --ntasks=1 \
    bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
    cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true
  exit "$exit_code"
}
trap cleanup_sentence_memory EXIT

mkdir -p "$PROJECT_DIR/logs/sbatch" "$(dirname "$OUT_DIR")"
export STAGE_ONLY_REQUESTED_NEIGHBORS=1
srun --nodes=1 --ntasks=1 \
  bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
  stage "$LOCAL_SENTENCE_MEMORY_DIR" "$SENTENCE_MEMORY_DIR" \
  "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SIGNTRAJ_SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD,,}"

# This is a local scientific audit, never an externally logged run. Remove any
# inherited credential variable as an additional guard; the Python entry point
# independently rejects online W&B mode.
unset WANDB_API_KEY
export WANDB_MODE=disabled
export WANDB_DISABLED=true

cd "$PROJECT_DIR"

run_stage() {
  local stage="$1"
  local destination="$2"

  local command=(
    srun --kill-on-bad-exit=1
    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.diagnose_temporal_slots
    --config "$CFG"
    --checkpoint "$CHECKPOINT"
    --out_dir "$destination"
    --stage "$stage"
    --batch_size "$BATCH_SIZE"
    --perturb_batch_size "$PERTURB_BATCH_SIZE"
    --fk_batch_size "$FK_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device cuda
    --text_device cpu
    --resume
  )
  if [[ "$VERIFY_HASHES" == "1" ]]; then
    command+=(--verify_hashes)
  fi
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  "${command[@]}"
}

echo "Job ID: $SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown}"
echo "Checkpoint: $CHECKPOINT"
echo "Validation-only output: $OUT_DIR"
echo "Sentence-memory bank: $SENTENCE_MEMORY_DIR (staged to node-local storage)"
echo "W&B: disabled"

if [[ "$RUN_SMOKE" == "1" ]]; then
  run_stage smoke "$SMOKE_OUT_DIR"
fi
if [[ "$RUN_FULL" == "1" ]]; then
  run_stage full "$OUT_DIR"
fi
