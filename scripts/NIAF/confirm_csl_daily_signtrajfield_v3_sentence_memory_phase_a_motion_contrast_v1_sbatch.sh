#!/bin/bash
#SBATCH --job-name=csl_v3_motion_confirm
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00

set -eo pipefail
trap 'echo "ERROR: locked validation confirmation failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml"
V2_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the one locked, development-feasible Phase-A checkpoint}"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_CHECKPOINT_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
PARTITION_DIR="${PARTITION_DIR:-${RUN_DIR}.prerequisites/validation_text_partition}"
OUT_ROOT="${OUT_ROOT:-$RUN_DIR/evaluation/locked_validation_confirmation}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
LOCAL_SENTENCE_MEMORY_DIR=""

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: locked checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi
if [[ "$(realpath -- "$CHECKPOINT")" != "$(realpath -- "$RUN_DIR/checkpoints/best.pt")" ]]; then
  echo "ERROR: confirmation accepts only RUN_DIR/checkpoints/best.pt" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/metrics.jsonl" || ! -f "$RUN_DIR/selection_summary.json" ]]; then
  echo "ERROR: full-run terminal selection evidence is unavailable" >&2
  exit 1
fi
if [[ ! -f "$V2_CFG" || ! -f "$V2_CHECKPOINT" ]]; then
  echo "ERROR: pinned v2 config/checkpoint is unavailable" >&2
  exit 1
fi
ACTUAL_V2_CHECKPOINT_SHA256="$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_V2_CHECKPOINT_SHA256" != "$EXPECTED_V2_CHECKPOINT_SHA256" ]]; then
  echo "ERROR: original v2 text-only checkpoint SHA256 mismatch" >&2
  exit 1
fi
if [[ "$(basename -- "$CHECKPOINT")" == "best_infeasible.pt" ]]; then
  echo "ERROR: best_infeasible.pt cannot enter confirmation" >&2
  exit 1
fi
if [[ ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: sentence-memory bank is not READY: $SENTENCE_MEMORY_DIR" >&2
  exit 1
fi
if [[ -e "$OUT_ROOT" ]]; then
  echo "ERROR: refusing to overwrite one-shot confirmation output: $OUT_ROOT" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE=disabled
unset WANDB_API_KEY

cd "$PROJECT_DIR"
echo "Original v2 checkpoint: $V2_CHECKPOINT"
echo "Original v2 checkpoint SHA256: $ACTUAL_V2_CHECKPOINT_SHA256"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.build_validation_text_partition \
  --config "$CFG" \
  --bank_dir "$SENTENCE_MEMORY_DIR" \
  --out_dir "$PARTITION_DIR" \
  --development_text_count 256 \
  --expected_novel_text_count 796 \
  --verify_only

cleanup_sentence_memory() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$LOCAL_SENTENCE_MEMORY_DIR" ]]; then
    srun --nodes=1 --ntasks=1 bash \
      "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
      cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true
  fi
  exit "$exit_code"
}
trap cleanup_sentence_memory EXIT

LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID:?Slurm job required}"
export STAGE_ONLY_REQUESTED_NEIGHBORS=1
srun --nodes=1 --ntasks=1 bash \
  "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
  stage "$LOCAL_SENTENCE_MEMORY_DIR" "$SENTENCE_MEMORY_DIR" \
  "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"

mkdir -p "$OUT_ROOT"
for MODE in off on shuffled motion_shuffled; do
  case "$MODE" in
    off) LABEL="text_only" ;;
    on) LABEL="sentence_memory" ;;
    shuffled) LABEL="shuffled_sentence_memory" ;;
    motion_shuffled) LABEL="motion_shuffled_sentence_memory" ;;
  esac
  MODE_DIR="$OUT_ROOT/$LABEL"
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
    --config "$CFG" \
    --checkpoint "$CHECKPOINT" \
    --split val \
    --manifest "$PARTITION_DIR/manifest_confirmation.jsonl" \
    --num_samples 0 \
    --selection_mode first \
    --out_dir "$MODE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --text_device "$TEXT_DEVICE" \
    --length_mode predicted \
    --word_prior off \
    --sentence_memory "$MODE" \
    --context_fps 20 \
    --sample_fps 20

  for ALIGNMENT in default pa; do
    srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
      flow.evaluate.dtw_mpjpe_t2m_default \
      --samples_dir "$MODE_DIR" \
      --out_json "$MODE_DIR/dtw_mpjpe_t2m_${ALIGNMENT}_h2s_betas.json" \
      --out_csv "$MODE_DIR/dtw_mpjpe_t2m_${ALIGNMENT}_h2s_betas.csv" \
      --sample_key smplx \
      --gt_key smplx \
      --prior_key adapter_context_smplx \
      --device "$DEVICE" \
      --betas_mode h2s_fixed \
      --alignment_mode "$ALIGNMENT" \
      --parts body lhand rhand face wholebody
  done
done

# These two prediction-only integrity controls do not enter the scientific
# four-mode metric family. The all-null path constructs fully masked tensors
# without opening any memory payload. The original v2 export provides direct
# selected-checkpoint text-only parity evidence on every confirmation row.
srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
  --config "$CFG" \
  --checkpoint "$CHECKPOINT" \
  --split val \
  --manifest "$PARTITION_DIR/manifest_confirmation.jsonl" \
  --num_samples 0 \
  --selection_mode first \
  --out_dir "$OUT_ROOT/all_null" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --text_device "$TEXT_DEVICE" \
  --length_mode predicted \
  --word_prior off \
  --sentence_memory all_null \
  --context_fps 20 \
  --sample_fps 20

srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
  --config "$V2_CFG" \
  --checkpoint "$V2_CHECKPOINT" \
  --split val \
  --manifest "$PARTITION_DIR/manifest_confirmation.jsonl" \
  --num_samples 0 \
  --selection_mode first \
  --out_dir "$OUT_ROOT/original_v2_text_only" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --text_device "$TEXT_DEVICE" \
  --length_mode predicted \
  --word_prior off \
  --context_fps 20 \
  --sample_fps 20

# The analyzer writes evidence even when a gate fails, then exits nonzero so an
# afterok Phase-B submission remains blocked.
srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation \
  --checkpoint "$CHECKPOINT" \
  --run_dir "$RUN_DIR" \
  --config "$CFG" \
  --v2_checkpoint "$V2_CHECKPOINT" \
  --v2_config "$V2_CFG" \
  --partition_dir "$PARTITION_DIR" \
  --text_only_dir "$OUT_ROOT/text_only" \
  --sentence_memory_dir "$OUT_ROOT/sentence_memory" \
  --shuffled_sentence_memory_dir "$OUT_ROOT/shuffled_sentence_memory" \
  --motion_shuffled_sentence_memory_dir "$OUT_ROOT/motion_shuffled_sentence_memory" \
  --all_null_dir "$OUT_ROOT/all_null" \
  --v2_text_only_dir "$OUT_ROOT/original_v2_text_only" \
  --out_dir "$OUT_ROOT/analysis" \
  --bootstrap_samples 10000 \
  --seed 1234
