#!/bin/bash
#SBATCH --job-name=csl_centered_b
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
export CENTERED_STAGE=stage2 CENTERED_RESUME="${CENTERED_RESUME:-0}"
export CENTERED_LAUNCHER="$PROJECT_DIR/scripts/NIAF/train_csl_daily_centered_absolute_binding_stage_b_v1_sbatch.sh"
bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_centered_memory_stage.sh"
