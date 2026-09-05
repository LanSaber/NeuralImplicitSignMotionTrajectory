#!/bin/bash
#SBATCH --job-name=signtraj_rag_diag
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
trap 'echo "ERROR: diagnose_sentence_retrieval_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:?Set CFG to the v3 trajectory configuration}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:?Set SENTENCE_MEMORY_DIR to the ready shared bank}"
OUT_JSON="${OUT_JSON:?Set OUT_JSON to the diagnostic summary path}"
DETAILS_JSONL="${DETAILS_JSONL:-${OUT_JSON%.json}_details.jsonl}"
SPLIT="${SPLIT:-val}"
TOP_K="${TOP_K:-8}"
RESAMPLE_POINTS="${RESAMPLE_POINTS:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
VERIFY_HASHES="${VERIFY_HASHES:-1}"
LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID:?Slurm allocation required}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$SENTENCE_MEMORY_DIR" || ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: SENTENCE_MEMORY_DIR is not a ready bank: $SENTENCE_MEMORY_DIR" >&2
  exit 1
fi
case "$VERIFY_HASHES" in
  0|1) ;;
  *) echo "ERROR: VERIFY_HASHES must be 0 or 1" >&2; exit 1 ;;
esac

cleanup_sentence_memory() {
  local exit_code=$?
  trap - EXIT
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
    cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true
  exit "$exit_code"
}
trap cleanup_sentence_memory EXIT

mkdir -p "$PROJECT_DIR/logs/sbatch" "$(dirname "$OUT_JSON")" \
  "$(dirname "$DETAILS_JSONL")"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
  stage "$LOCAL_SENTENCE_MEMORY_DIR" "$SENTENCE_MEMORY_DIR" \
  "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train $SPLIT"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

cd "$PROJECT_DIR"
DIAG_CMD=(
  srun --kill-on-bad-exit=1
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.diagnose_sentence_retrieval
  --config "$CFG"
  --bank_dir "$LOCAL_SENTENCE_MEMORY_DIR"
  --split "$SPLIT"
  --top_k "$TOP_K"
  --resample_points "$RESAMPLE_POINTS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --device "$DEVICE"
  --out_json "$OUT_JSON"
  --details_jsonl "$DETAILS_JSONL"
)
if [[ "$LIMIT" -gt 0 ]]; then DIAG_CMD+=(--limit "$LIMIT"); fi
if [[ "$VERIFY_HASHES" == "1" ]]; then DIAG_CMD+=(--verify_hashes); fi

printf 'Command:'
printf ' %q' "${DIAG_CMD[@]}"
printf '\n'
"${DIAG_CMD[@]}"
