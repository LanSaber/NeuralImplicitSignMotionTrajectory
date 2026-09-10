#!/bin/bash
#SBATCH --job-name=csl_stage_c_protocol_v3_run_r4_pilot
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
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the exact pushed source commit}"
[[ "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ ]] || {
  echo "ERROR: SOURCE_GIT_HEAD must be a full Git SHA1" >&2
  exit 1
}
export SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD,,}"
export STAGE_C_EXECUTION_MODE=pilot
bash "$PROJECT_DIR/scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh"
