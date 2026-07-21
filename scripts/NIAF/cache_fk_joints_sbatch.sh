#!/bin/bash
#SBATCH --job-name=niaf_fk_cache
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00

set -eo pipefail
trap 'echo "ERROR: cache_fk_joints_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_trajectory_field/configs/phoenix_continuous_trajectory_stage2_part_experts_full.yaml}"
SPLITS="${SPLITS:-train,val}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-auto}"
OVERWRITE="${OVERWRITE:-0}"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$PROJECT_DIR/logs/sbatch"
cd "$PROJECT_DIR"

CACHE_CMD=(
  srun "$PYTHON_BIN" -m NIAF.continuous_sign_field.scripts.cache_fk_joints
  --config "$CFG"
  --splits "$SPLITS"
  --limit "$LIMIT"
  --device "$DEVICE"
)
if [[ "$OVERWRITE" == "1" ]]; then CACHE_CMD+=(--overwrite); fi

echo "Caching FK joints: config=$CFG splits=$SPLITS limit=$LIMIT device=$DEVICE overwrite=$OVERWRITE"
"${CACHE_CMD[@]}"
