#!/bin/bash
#SBATCH --job-name=csl_v3_splitkv_resume
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00

# Exact continuation of an interrupted Stage 1 or Stage 2. The shared driver
# accepts only the selected stage's existing last.pt and original launch chain.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
FACTORIZED_STAGE="${FACTORIZED_STAGE:?Set FACTORIZED_STAGE to stage1 or stage2}"
export PROJECT_DIR FACTORIZED_STAGE
export FACTORIZED_RESUME=1
export FACTORIZED_LAUNCHER="$PROJECT_DIR/scripts/NIAF/resume_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
exec bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_factorized_memory_stage.sh"
