#!/bin/bash
#SBATCH --job-name=csl_v3_splitkv_s1
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
export FACTORIZED_STAGE=stage1
export FACTORIZED_RESUME=0
export FACTORIZED_LAUNCHER="$PROJECT_DIR/scripts/NIAF/train_csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1_sbatch.sh"
exec bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_factorized_memory_stage.sh"
