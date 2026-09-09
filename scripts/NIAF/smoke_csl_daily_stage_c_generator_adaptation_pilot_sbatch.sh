#!/bin/bash
#SBATCH --job-name=csl_stage_c_smoke
#SBATCH --partition=spark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# One optimizer update per matched arm on the exact paired-RoCE path.  As with
# the full pilot, submit with an explicit `--constraint=pairNN`.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
export STAGE_C_EXECUTION_MODE=smoke
bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_pilot.sh"
