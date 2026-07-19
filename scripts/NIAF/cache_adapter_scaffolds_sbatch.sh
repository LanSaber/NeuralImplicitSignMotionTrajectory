#!/bin/bash
#SBATCH --job-name=niaf_scaffold_cache
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -eo pipefail
trap 'echo "ERROR: cache_adapter_scaffolds_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_sign_field/configs/phoenix_local_amortized_adapter_text_pilot500_rawgt.yaml}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-auto}"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HOME="${HOME_VALUE:-/media/cvpr/haomian}"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$PROJECT_DIR/logs/sbatch" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd "$PROJECT_DIR"

echo "Caching text-adapter scaffolds with config=$CFG batch=$BATCH_SIZE device=$DEVICE"
srun "$PYTHON_BIN" -m NIAF.continuous_sign_field.scripts.cache_adapter_scaffolds \
  --config "$CFG" \
  --splits train val \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"
