#!/bin/bash
#SBATCH --job-name=niaf_dtw_pa
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -eo pipefail
trap 'echo "ERROR: evaluate_dtw_pa_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SAMPLES_DIR="${SAMPLES_DIR:?Set SAMPLES_DIR to an existing trajectory export directory}"
OUT_STEM="${OUT_STEM:-dtw_mpjpe_t2m_pa_partwise_same_subset_h2s_betas}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$SAMPLES_DIR" ]]; then
  echo "ERROR: samples directory does not exist: $SAMPLES_DIR" >&2
  exit 1
fi

OUT_JSON="$SAMPLES_DIR/$OUT_STEM.json"
OUT_CSV="$SAMPLES_DIR/$OUT_STEM.csv"
if [[ -e "$OUT_JSON" || -e "$OUT_CSV" ]]; then
  echo "ERROR: versioned PA-DTW output already exists: $OUT_JSON or $OUT_CSV" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$PROJECT_DIR/logs/sbatch"
cd "$PROJECT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Samples: $SAMPLES_DIR"
echo "PA metric preset: t2m_partwise_pa_same_subset"
echo "Output JSON: $OUT_JSON"
echo "Output CSV: $OUT_CSV"

srun --kill-on-bad-exit=1 "$PYTHON_BIN" \
  -m flow.evaluate.dtw_mpjpe_t2m_default \
  --samples_dir "$SAMPLES_DIR" \
  --out_json "$OUT_JSON" \
  --out_csv "$OUT_CSV" \
  --sample_key smplx \
  --gt_key smplx \
  --prior_key adapter_context_smplx \
  --device "$DEVICE" \
  --betas_mode h2s_fixed \
  --alignment_mode pa \
  --parts body lhand rhand wholebody
