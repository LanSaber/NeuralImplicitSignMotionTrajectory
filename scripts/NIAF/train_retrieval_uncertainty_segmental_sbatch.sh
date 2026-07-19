#!/bin/bash
#SBATCH --job-name=niaf_segmental
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
export CFG="${CFG:-NIAF/retrieval_confidence_field/configs/phoenix_retrieval_uncertainty_segmental_trainbalanced.yaml}"
export RUN_TAG="${RUN_TAG:-phoenix_retrieval_uncertainty_segmental_trainbalanced}"
export WANDB_PROJECT="${WANDB_PROJECT:-soke-niaf-segmental}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID:-local}}"

exec bash "$PROJECT_DIR/scripts/NIAF/train_retrieval_uncertainty_adaptive_knots_sbatch.sh"
