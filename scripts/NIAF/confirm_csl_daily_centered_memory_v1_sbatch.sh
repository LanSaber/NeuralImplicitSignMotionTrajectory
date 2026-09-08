#!/bin/bash
#SBATCH --job-name=csl_centered_confirm
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the locked shared standalone clone}"
export POSTSELECTION_PURPOSE=confirmation
bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_centered_memory_postselection.sh"
