#!/bin/bash
#SBATCH --job-name=csl_v3_splitkv_decide
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Post-lock development-only integrity audit and ordered scientific decision.
set -euo pipefail
trap 'echo "ERROR: factorized-memory ordered decision failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
FACTORIZED_STAGE="${FACTORIZED_STAGE:?Set FACTORIZED_STAGE to stage1 or stage2}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the training source commit}"
V2_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"

case "$FACTORIZED_STAGE" in
  stage1)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
    ;;
  stage2)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1"
    ;;
  *) echo "ERROR: invalid FACTORIZED_STAGE" >&2; exit 1 ;;
esac

RUN_DIR="${RUN_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$EXPERIMENT}"
CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${EXPERIMENT}.yaml"
SEALED_PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
PARTITION_DIR="${PARTITION_DIR:-$SEALED_PARTITION_DIR}"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"
EXPECTED_DEVELOPMENT_MANIFEST_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
EXPECTED_PARTITION_ARTIFACT_IDENTITY="2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
INTEGRITY_ROOT="${INTEGRITY_ROOT:-$RUN_DIR/evaluation/development_integrity_for_ordered_decision}"
DECISION_DIR="${DECISION_DIR:-$RUN_DIR/evaluation/ordered_development_decision}"
STAGE1_RUN_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
STAGE1_AUTHORIZATION="${STAGE1_AUTHORIZATION:-$STAGE1_RUN_DIR/evaluation/ordered_development_decision/authorize_stage2.json}"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "1" ]]; then
  echo "ERROR: ordered decision requires a one-node Slurm allocation" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$CFG" || ! -f "$RUN_DIR/selection_summary.json" ]]; then
  echo "ERROR: decision prerequisites are missing" >&2
  exit 1
fi
if [[ ! -f "$V2_CFG" || ! -f "$V2_CHECKPOINT" || "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_V2_SHA256" ]]; then
  echo "ERROR: decision requires the pinned original v2 config/checkpoint" >&2
  exit 1
fi
if [[ "$(realpath -e "$PARTITION_DIR")" != "$(realpath -e "$SEALED_PARTITION_DIR")" \
      || "$(sha256sum -- "$PARTITION_DIR/READY" | awk '{print $1}')" != "$EXPECTED_PARTITION_READY_SHA256" \
      || "$(sha256sum -- "$PARTITION_DIR/manifest_development.jsonl" | awk '{print $1}')" != "$EXPECTED_DEVELOPMENT_MANIFEST_SHA256" ]]; then
  echo "ERROR: ordered decision requires the sealed development partition" >&2
  exit 1
fi
cd "$PROJECT_DIR"
if [[ ! -d "$PROJECT_DIR/.git" || -L "$PROJECT_DIR/.git" ]]; then
  echo "ERROR: decision requires the original shared standalone source clone" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: decision source clone must reside on /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/experiments" ]]; then
  echo "ERROR: decision clone must expose the durable experiments directory" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/experiments")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: decision experiments must resolve to /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/deps/mt5-base" ]]; then
  echo "ERROR: decision clone must expose frozen deps/mt5-base" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/deps/mt5-base")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: decision mT5 dependency must resolve to /media/cvpr" >&2; exit 1 ;;
esac
if [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: decision source uses external or node-local git metadata" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$SOURCE_GIT_HEAD" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: decision must use the same clean source commit as training" >&2
  exit 1
fi

CHECKPOINT_NAME="$("$PYTHON_BIN" - "$RUN_DIR/selection_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print("best.pt" if summary.get("has_feasible_checkpoint") is True else "best_infeasible.pt")
PY
)"
CHECKPOINT="$RUN_DIR/checkpoints/$CHECKPOINT_NAME"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: selected development checkpoint is missing: $CHECKPOINT" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY
unset SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256
export SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256="$EXPECTED_DEVELOPMENT_MANIFEST_SHA256"
export SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS=347
export SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY="$EXPECTED_PARTITION_ARTIFACT_IDENTITY"

LEASE_PATH="$RUN_DIR/evaluation/active_ordered_decision_lease"
LEASE_ATTESTATION="$RUN_DIR/evaluation/ordered_decision_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
  acquire-execution-lease \
  --purpose decision --stage "$FACTORIZED_STAGE" \
  --lease "$LEASE_PATH" --out_file "$LEASE_ATTESTATION" \
  --source_git_head "$SOURCE_GIT_HEAD" --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$(sha256sum -- "$CHECKPOINT" | awk '{print $1}')"
LEASE_HELD=1
release_decision_lease_on_success() {
  local exit_code=$?
  trap - EXIT
  if [[ "$exit_code" == "0" && "${LEASE_HELD:-0}" == "1" ]]; then
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      release-execution-lease --lease "$LEASE_PATH" \
      --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap release_decision_lease_on_success EXIT

run_export() {
  local mode="$1"
  local directory="$2"
  local config_path="${3:-$CFG}"
  local checkpoint_path="${4:-$CHECKPOINT}"
  if [[ -f "$directory/export_summary.json" ]]; then
    return 0
  fi
  if [[ -e "$directory" ]]; then
    echo "ERROR: refusing partial development integrity export: $directory" >&2
    return 1
  fi
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
    --config "$config_path" \
    --checkpoint "$checkpoint_path" \
    --split val \
    --manifest "$PARTITION_DIR/manifest_development.jsonl" \
    --num_samples 0 \
    --selection_mode first \
    --out_dir "$directory" \
    --batch_size 8 \
    --device cuda \
    --text_device cpu \
    --length_mode predicted \
    --word_prior off \
    --sentence_memory "$mode" \
    --context_fps 20 \
    --sample_fps 20
}

run_export off "$INTEGRITY_ROOT/text_only"
run_export all_null "$INTEGRITY_ROOT/all_null"
run_export auto "$INTEGRITY_ROOT/original_v2_text_only" "$V2_CFG" "$V2_CHECKPOINT"

DECIDE_ARGS=(
  decide
  --stage "$FACTORIZED_STAGE"
  --run_dir "$RUN_DIR"
  --config "$CFG"
  --partition_dir "$PARTITION_DIR"
  --memory_off_dir "$INTEGRITY_ROOT/text_only"
  --all_null_dir "$INTEGRITY_ROOT/all_null"
  --v2_memory_off_dir "$INTEGRITY_ROOT/original_v2_text_only"
  --v2_config "$V2_CFG"
  --v2_checkpoint "$V2_CHECKPOINT"
  --out_dir "$DECISION_DIR"
)
if [[ "$FACTORIZED_STAGE" == "stage2" ]]; then
  DECIDE_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
fi
srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
  "${DECIDE_ARGS[@]}"
