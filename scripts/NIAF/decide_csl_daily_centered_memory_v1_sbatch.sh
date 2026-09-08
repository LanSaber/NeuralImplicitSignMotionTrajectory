#!/bin/bash
#SBATCH --job-name=csl_centered_decide
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# Complete-dev selected-checkpoint replay plus ordered Stage-A''' decision.
set -euo pipefail
trap 'echo "ERROR: centered ordered decision failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the training commit}"
CENTERED_STAGE="${CENTERED_STAGE:?Set CENTERED_STAGE to stage1 or stage2}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
HELPER="NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage"
LAUNCHER="$PROJECT_DIR/scripts/NIAF/decide_csl_daily_centered_memory_v1_sbatch.sh"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
V2_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"
EXPECTED_DEV_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
EXPECTED_PARTITION_ID="2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
CALIBRATION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_v1_retry1"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"

case "$CENTERED_STAGE" in
  stage1) EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1" ;;
  stage2) EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1" ;;
  *) echo "ERROR: invalid CENTERED_STAGE" >&2; exit 1 ;;
esac
CANONICAL_RUN_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$EXPERIMENT"
if [[ -n "${RUN_DIR:-}" && "$RUN_DIR" != "$CANONICAL_RUN_DIR" ]]; then
  echo "ERROR: centered decisions forbid a noncanonical RUN_DIR" >&2
  exit 1
fi
RUN_DIR="$CANONICAL_RUN_DIR"
CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${EXPERIMENT}.yaml"
DECISION_DIR="$RUN_DIR/evaluation/ordered_development_decision"
ATTEMPT_ROOT="$RUN_DIR/evaluation/ordered_decision_attempt_artifacts/${SLURM_JOB_ID:-missing}.${SLURM_RESTART_COUNT:-0}"
STAGE1_RUN="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1"
CANONICAL_STAGE1_AUTHORIZATION="$STAGE1_RUN/evaluation/ordered_development_decision/authorize_stage2.json"
if [[ -n "${STAGE1_AUTHORIZATION:-}" && "$STAGE1_AUTHORIZATION" != "$CANONICAL_STAGE1_AUTHORIZATION" ]]; then
  echo "ERROR: centered decisions forbid a noncanonical Stage-A authorization" >&2
  exit 1
fi
STAGE1_AUTHORIZATION="$CANONICAL_STAGE1_AUTHORIZATION"

[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-0}" == "1" ]] || {
  echo "ERROR: centered decision requires one Slurm node" >&2; exit 1;
}
for required in "$PYTHON_BIN" "$CFG" "$RUN_DIR/selection_summary.json" \
  "$V2_CFG" "$V2_CHECKPOINT" "$PARTITION_DIR/READY" \
  "$PARTITION_DIR/manifest_development.jsonl" "$CALIBRATION_DIR/READY" \
  "$SOURCE_BANK/READY" "$LAUNCHER" "$STAGER"; do
  [[ -e "$required" ]] || { echo "ERROR: decision prerequisite absent: $required" >&2; exit 1; }
done
[[ ! -e "$DECISION_DIR" && ! -e "$GLOBAL_HOLDOUT_SPEND" && ! -e "$ATTEMPT_ROOT" ]] || {
  echo "ERROR: decision is already terminal, holdout spent, or attempt path reused" >&2; exit 1;
}
[[ "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_V2_SHA256" \
   && "$(sha256sum -- "$PARTITION_DIR/READY" | awk '{print $1}')" == "$EXPECTED_PARTITION_READY_SHA256" \
   && "$(sha256sum -- "$PARTITION_DIR/manifest_development.jsonl" | awk '{print $1}')" == "$EXPECTED_DEV_SHA256" \
   && "$(awk 'NF {n += 1} END {print n + 0}' "$PARTITION_DIR/manifest_development.jsonl")" == "347" ]] || {
  echo "ERROR: v2 or sealed development identity changed" >&2; exit 1;
}

cd "$PROJECT_DIR"
[[ -d .git && ! -L .git \
   && "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" == "$(realpath -e "$PROJECT_DIR/.git")" \
   && "$(git rev-parse HEAD)" == "$SOURCE_GIT_HEAD" \
   && -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: decision requires the exact clean shared standalone source" >&2; exit 1;
}
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
[[ -n "$REMOTE_HEAD" && "${REMOTE_HEAD,,}" == "${SOURCE_GIT_HEAD,,}" ]] || {
  echo "ERROR: decision source is not the pushed origin head" >&2; exit 1;
}

export PATH="$PYTHON_ENV/bin:$PATH" PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
export SIGNTRAJ_SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD,,}"
export SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR="$PARTITION_DIR"
export SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256="$EXPECTED_DEV_SHA256"
export SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS=347
export SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY="$EXPECTED_PARTITION_ID"
unset WANDB_API_KEY SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256

"$PYTHON_BIN" - "$CALIBRATION_DIR" "$PROJECT_DIR" "$SOURCE_GIT_HEAD" \
  "origin/$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" <<'PY'
import sys
from NIAF.continuous_trajectory_field.relevance_calibration import (
    validate_relevance_calibration_artifact,
    validate_relevance_calibration_source,
)
value = validate_relevance_calibration_artifact(sys.argv[1])
validate_relevance_calibration_source(
    value, source_root=sys.argv[2], expected_git_head=sys.argv[3],
    expected_remote_ref=sys.argv[4], expected_remote_head=sys.argv[5],
)
PY
if [[ "$CENTERED_STAGE" == "stage2" ]]; then
  "$PYTHON_BIN" -m "$HELPER" verify --purpose stage2 --stage stage1 \
    --authorization "$STAGE1_AUTHORIZATION"
fi

CHECKPOINT_NAME="$($PYTHON_BIN - "$RUN_DIR/selection_summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print("best.pt" if value.get("has_feasible_checkpoint") is True else "best_infeasible.pt")
PY
)"
CHECKPOINT="$RUN_DIR/checkpoints/$CHECKPOINT_NAME"
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: selected checkpoint absent" >&2; exit 1; }

LEASE="$RUN_DIR/evaluation/active_ordered_decision_lease"
LEASE_ATTESTATION="$RUN_DIR/evaluation/ordered_decision_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
"$PYTHON_BIN" -m "$HELPER" acquire-execution-lease \
  --purpose decision --stage "$CENTERED_STAGE" --lease "$LEASE" \
  --out_file "$LEASE_ATTESTATION" --source_git_head "$SOURCE_GIT_HEAD" \
  --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$(sha256sum -- "$CHECKPOINT" | awk '{print $1}')"

LOCAL_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
cleanup() {
  local exit_code=$?
  trap - EXIT
  "$STAGER" cleanup "$LOCAL_BANK" || true
  if [[ "$exit_code" == "0" ]]; then
    "$PYTHON_BIN" -m "$HELPER" release-execution-lease \
      --lease "$LEASE" --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup EXIT
"$STAGER" stage "$LOCAL_BANK" "$SOURCE_BANK" "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"
[[ -f "$LOCAL_BANK/neighbors_train.npz" && -f "$LOCAL_BANK/neighbors_val.npz" \
   && ! -e "$LOCAL_BANK/neighbors_test.npz" ]] || {
  echo "ERROR: decision local staging is not exactly train/val" >&2; exit 1;
}
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_BANK" SENTENCE_MEMORY_DIR=""
mkdir -p -- "$ATTEMPT_ROOT"

AUDIT_JSON="$ATTEMPT_ROOT/selected_checkpoint_auto.json"
srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.evaluate_continuous_trajectory_field \
  --config "$CFG" --checkpoint "$CHECKPOINT" --split val --limit 0 \
  --out_json "$AUDIT_JSON" --batch_size 8 --max_batches 0 \
  --device cuda --text_device cpu --distributed none \
  --word_prior off --sentence_memory auto

run_export() {
  local mode="$1" directory="$2" config_path="${3:-$CFG}" checkpoint_path="${4:-$CHECKPOINT}"
  [[ ! -e "$directory" ]] || { echo "ERROR: duplicate attempt export: $directory" >&2; return 1; }
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
    --config "$config_path" --checkpoint "$checkpoint_path" --split val \
    --manifest "$PARTITION_DIR/manifest_development.jsonl" --num_samples 0 \
    --selection_mode first --out_dir "$directory" --batch_size 8 \
    --device cuda --text_device cpu --length_mode predicted \
    --word_prior off --sentence_memory "$mode" --context_fps 20 --sample_fps 20
}
run_export off "$ATTEMPT_ROOT/text_only"
run_export all_null "$ATTEMPT_ROOT/all_null"
run_export auto "$ATTEMPT_ROOT/original_v2_text_only" "$V2_CFG" "$V2_CHECKPOINT"

DECIDE_ARGS=(
  decide --stage "$CENTERED_STAGE" --run_dir "$RUN_DIR" --config "$CFG"
  --partition_dir "$PARTITION_DIR" --integrity_audit "$AUDIT_JSON"
  --memory_off_dir "$ATTEMPT_ROOT/text_only" --all_null_dir "$ATTEMPT_ROOT/all_null"
  --v2_memory_off_dir "$ATTEMPT_ROOT/original_v2_text_only"
  --v2_config "$V2_CFG" --v2_checkpoint "$V2_CHECKPOINT" --out_dir "$DECISION_DIR"
)
if [[ "$CENTERED_STAGE" == "stage2" ]]; then
  DECIDE_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
fi
srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m "$HELPER" "${DECIDE_ARGS[@]}"
