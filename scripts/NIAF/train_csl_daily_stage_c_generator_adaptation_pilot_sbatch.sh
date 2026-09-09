#!/bin/bash
#SBATCH --job-name=csl_stage_c_pair
#SBATCH --partition=spark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# The pair constraint is intentionally supplied by the submission command as
# `sbatch --constraint=pairNN`; Slurm directives do not expand shell variables.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
export STAGE_C_EXECUTION_MODE=pilot
bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_pilot.sh"
