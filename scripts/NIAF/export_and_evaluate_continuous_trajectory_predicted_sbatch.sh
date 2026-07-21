#!/bin/bash
#SBATCH --job-name=niaf_ct_pred_eval
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00

set -eo pipefail
trap 'echo "ERROR: export_and_evaluate_continuous_trajectory_predicted_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:?Set CFG to the continuous trajectory configuration}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a checkpoint or checkpoint directory}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to the predicted-length export directory}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CONTEXT_FPS="${CONTEXT_FPS:-20}"
SAMPLE_FPS="${SAMPLE_FPS:-20}"
DEVICE="${DEVICE:-cuda}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT_DIR="$CHECKPOINT"
  CHECKPOINT=""
  for CANDIDATE in best.pt best_infeasible.pt last.pt; do
    if [[ -f "$CHECKPOINT_DIR/$CANDIDATE" ]]; then
      CHECKPOINT="$CHECKPOINT_DIR/$CANDIDATE"
      break
    fi
  done
  if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no selected or final checkpoint found under $CHECKPOINT_DIR" >&2
    exit 1
  fi
elif [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint does not exist: $CHECKPOINT" >&2
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

mkdir -p "$PROJECT_DIR/logs/sbatch" "$OUT_DIR"
cd "$PROJECT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT num_samples=$NUM_SAMPLES length_mode=predicted"
echo "Context/sample FPS: $CONTEXT_FPS/$SAMPLE_FPS"
echo "Output: $OUT_DIR"

srun --kill-on-bad-exit=1 "$PYTHON_BIN" \
  -m NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
  --config "$CFG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --num_samples "$NUM_SAMPLES" \
  --selection_mode first \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --text_device "$TEXT_DEVICE" \
  --length_mode predicted \
  --context_fps "$CONTEXT_FPS" \
  --sample_fps "$SAMPLE_FPS"

for ALIGNMENT_MODE in default pa; do
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" \
    -m flow.evaluate.dtw_mpjpe_t2m_default \
    --samples_dir "$OUT_DIR" \
    --out_json "$OUT_DIR/dtw_mpjpe_t2m_${ALIGNMENT_MODE}_h2s_betas.json" \
    --out_csv "$OUT_DIR/dtw_mpjpe_t2m_${ALIGNMENT_MODE}_h2s_betas.csv" \
    --sample_key smplx \
    --gt_key smplx \
    --prior_key adapter_context_smplx \
    --device "$DEVICE" \
    --betas_mode h2s_fixed \
    --alignment_mode "$ALIGNMENT_MODE" \
    --parts body lhand rhand wholebody
done
