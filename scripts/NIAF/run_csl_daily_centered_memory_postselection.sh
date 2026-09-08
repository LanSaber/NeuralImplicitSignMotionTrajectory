#!/bin/bash

# Shared, fail-closed post-selection driver for Phase-A''' diagnostics and the
# one-shot confirmation.  Resource requests live in the two sbatch wrappers.
set -euo pipefail
trap 'echo "ERROR: centered post-selection failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the locked shared standalone clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to its clean pushed commit}"
CENTERED_STAGE="${CENTERED_STAGE:?Set CENTERED_STAGE to stage1 or stage2}"
POSTSELECTION_PURPOSE="${POSTSELECTION_PURPOSE:?Set purpose to diagnostic or confirmation}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
DECISION_MODULE="NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
PARTITION_DIR="${PARTITION_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition}"
V2_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
CONTROL_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_ordered_v1_control"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"
BATCH_SIZE="${BATCH_SIZE:-8}"

case "$CENTERED_STAGE" in
  stage1)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1"
    MODES=(off on motion_shuffled_n0 motion_shuffled_n1 motion_shuffled_n2 cross_query_motion full_replacement joint_tuple_permuted uniform_final_mass analytic_prior)
    ;;
  stage2)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1"
    MODES=(off on motion_shuffled_n0 motion_shuffled_n1 motion_shuffled_n2 cross_query_motion full_replacement joint_tuple_permuted uniform_final_mass analytic_prior association_disabled)
    ;;
  *) echo "ERROR: CENTERED_STAGE must be stage1 or stage2" >&2; exit 1 ;;
esac
case "$POSTSELECTION_PURPOSE" in diagnostic|confirmation) ;; *) echo "ERROR: invalid post-selection purpose" >&2; exit 1 ;; esac

CANONICAL_RUN_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$EXPERIMENT"
if [[ -n "${RUN_DIR:-}" && "$RUN_DIR" != "$CANONICAL_RUN_DIR" ]]; then
  echo "ERROR: centered post-selection forbids a noncanonical RUN_DIR" >&2
  exit 1
fi
RUN_DIR="$CANONICAL_RUN_DIR"
CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${EXPERIMENT}.yaml"
CHECKPOINT="$RUN_DIR/checkpoints/best.pt"
DECISION_DIR="$RUN_DIR/evaluation/ordered_development_decision"
DIAGNOSTIC_AUTH="$DECISION_DIR/authorize_diagnostic.json"
CONFIRMATION_AUTH="$DECISION_DIR/authorize_confirmation.json"
DIAGNOSTIC_DIR="$RUN_DIR/evaluation/locked_development_layerwise_diagnostic"
DIAGNOSTIC_EXPORTS="$RUN_DIR/evaluation/locked_development_layerwise_exports"
CONFIRMATION_ROOT="$RUN_DIR/evaluation/locked_validation_confirmation"
DIAGNOSTIC_ALREADY_FINALIZED=0

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "1" ]]; then
  echo "ERROR: post-selection requires a one-node Slurm allocation" >&2
  exit 1
fi
for required in "$PYTHON_BIN" "$CFG" "$CHECKPOINT" "$STAGER" "$SOURCE_BANK/READY" "$PARTITION_DIR/READY"; do
  [[ -e "$required" ]] || { echo "ERROR: missing prerequisite: $required" >&2; exit 1; }
done
[[ "$(basename -- "$CHECKPOINT")" == "best.pt" && ! -L "$CHECKPOINT" ]] || {
  echo "ERROR: post-selection accepts only the real feasible best.pt" >&2
  exit 1
}
[[ "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_V2_SHA256" ]] || {
  echo "ERROR: v2 checkpoint identity changed" >&2
  exit 1
}

cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git ]] || [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e .git)" ]]; then
  echo "ERROR: post-selection requires a standalone shared clone" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in /media/cvpr/*) ;; *) echo "ERROR: source clone is not shared" >&2; exit 1 ;; esac
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "${ACTUAL_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: source is not the locked clean commit" >&2
  exit 1
fi
SOURCE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$REMOTE_HEAD" || "${REMOTE_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" ]]; then
  echo "ERROR: locked source is not the exact pushed branch head" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

if [[ "$POSTSELECTION_PURPOSE" == "diagnostic" ]]; then
  AUTHORIZATION="$DIAGNOSTIC_AUTH"
  QUERY_MANIFEST="$PARTITION_DIR/manifest_development.jsonl"
  EXPORT_ROOT="$DIAGNOSTIC_EXPORTS"
  EXPECTED_EXPORT_ROWS=347
  [[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || { echo "ERROR: diagnostic must precede holdout spend" >&2; exit 1; }
  "$PYTHON_BIN" -m "$DECISION_MODULE" verify --purpose diagnostic --stage "$CENTERED_STAGE" --authorization "$AUTHORIZATION"
  if [[ -e "$CONFIRMATION_AUTH" ]]; then
    [[ -f "$DIAGNOSTIC_DIR/READY" ]] || { echo "ERROR: confirmation auth lacks diagnostic READY" >&2; exit 1; }
    "$PYTHON_BIN" -m "$DECISION_MODULE" verify --purpose confirmation --stage "$CENTERED_STAGE" --authorization "$CONFIRMATION_AUTH"
    DIAGNOSTIC_ALREADY_FINALIZED=1
  fi
  unset SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256
else
  AUTHORIZATION="$CONFIRMATION_AUTH"
  QUERY_MANIFEST="$PARTITION_DIR/manifest_confirmation.jsonl"
  EXPORT_ROOT="$CONFIRMATION_ROOT/exports"
  EXPECTED_EXPORT_ROWS=728
  [[ -f "$DIAGNOSTIC_DIR/READY" ]] || { echo "ERROR: confirmation lacks locked diagnostic READY" >&2; exit 1; }
  "$PYTHON_BIN" -m "$DECISION_MODULE" verify --purpose confirmation --stage "$CENTERED_STAGE" --authorization "$AUTHORIZATION"
fi

LEASE_ROOT="$CONTROL_DIR/active_${POSTSELECTION_PURPOSE}_execution_lease"
LEASE_ATTESTATION="$CONTROL_DIR/${POSTSELECTION_PURPOSE}_execution_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
"$PYTHON_BIN" -m "$DECISION_MODULE" acquire-execution-lease \
  --purpose "$POSTSELECTION_PURPOSE" --stage "$CENTERED_STAGE" \
  --lease "$LEASE_ROOT" --out_file "$LEASE_ATTESTATION" \
  --source_git_head "$SOURCE_GIT_HEAD" --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$(sha256sum -- "$AUTHORIZATION" | awk '{print $1}')"
LEASE_HELD=1
LOCAL_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
POSTSELECTION_COMPLETE=0

cleanup_postselection() {
  local exit_code=$?
  trap - EXIT
  srun --nodes=1 --ntasks=1 bash "$STAGER" cleanup "$LOCAL_BANK" || true
  if [[ "$LEASE_HELD" == "1" && "$POSTSELECTION_COMPLETE" == "1" ]]; then
    "$PYTHON_BIN" -m "$DECISION_MODULE" release-execution-lease \
      --lease "$LEASE_ROOT" --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup_postselection EXIT

if [[ "$POSTSELECTION_PURPOSE" == "diagnostic" && "$DIAGNOSTIC_ALREADY_FINALIZED" == "1" ]]; then
  # A scheduler restart after authorization publication only revalidates the
  # immutable result and releases its newly reconciled lease.
  POSTSELECTION_COMPLETE=1
  exit 0
fi

if [[ "$POSTSELECTION_PURPOSE" == "confirmation" ]]; then
  # This marker is permanent even if any subsequent export or gate fails.
  "$PYTHON_BIN" -m "$DECISION_MODULE" spend-confirmation \
    --stage "$CENTERED_STAGE" --authorization "$AUTHORIZATION" \
    --marker "$GLOBAL_HOLDOUT_SPEND" --allow_matching_existing
  export SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256
  SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256="$($PYTHON_BIN - "$PARTITION_DIR" <<'PY'
import sys
from pathlib import Path
from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import load_confirmation_partition
print(load_confirmation_partition(Path(sys.argv[1]))["manifest_sha256"])
PY
)"
fi

srun --nodes=1 --ntasks=1 bash "$STAGER" stage "$LOCAL_BANK" "$SOURCE_BANK" \
  "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_BANK"
mkdir -p "$EXPORT_ROOT"

rebase_and_publish() {
  local build_dir="$1"
  local final_dir="$2"
  "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.publish_centered_export \
    --build_dir "$build_dir" --final_dir "$final_dir"
}

validate_published_export() {
  local directory="$1"
  local scored="$2"
  "$PYTHON_BIN" - "$directory" "$EXPECTED_EXPORT_ROWS" "$scored" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
scored = bool(int(sys.argv[3]))
summary_path = directory / "export_summary.json"
manifest_summary_path = directory / "sample_manifest_summary.json"
if (
    not directory.is_dir()
    or directory.is_symlink()
    or not summary_path.is_file()
    or summary_path.is_symlink()
    or not manifest_summary_path.is_file()
    or manifest_summary_path.is_symlink()
):
    raise SystemExit("published export is not a complete real directory")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = summary.get("rows")
manifest = summary.get("manifest")
if (
    not isinstance(rows, list)
    or len(rows) != expected
    or summary.get("num_exported") != expected
    or not isinstance(manifest, dict)
    or manifest.get("sample_count") != expected
):
    raise SystemExit("published export row count is incomplete")
seen = set()
for row in rows:
    sample = Path(str(row.get("sample", "")))
    sample = sample if sample.is_absolute() else directory / sample.name
    sample = sample.resolve()
    if sample.parent != directory or not sample.is_file() or sample.is_symlink():
        raise SystemExit("published export references an unsafe/missing sample")
    if sample in seen:
        raise SystemExit("published export repeats a sample path")
    seen.add(sample)
if scored:
    for alignment in ("default", "pa"):
        for suffix in ("json", "csv"):
            path = directory / f"dtw_mpjpe_t2m_{alignment}_h2s_betas.{suffix}"
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise SystemExit("published scored export is incomplete")
PY
}

run_export() {
  local mode="$1"
  local label="$2"
  local scored="$3"
  local config_path="${4:-$CFG}"
  local checkpoint_path="${5:-$CHECKPOINT}"
  local mode_dir="$EXPORT_ROOT/$label"
  if [[ -d "$mode_dir" ]]; then
    validate_published_export "$mode_dir" "$scored"
    return
  fi
  [[ ! -e "$mode_dir" ]] || { echo "ERROR: invalid mode path $mode_dir" >&2; exit 1; }
  local build_dir="$EXPORT_ROOT/.${label}.building.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "$build_dir" ]] || { echo "ERROR: stale mode build exists: $build_dir" >&2; exit 1; }
  local memory_args=()
  [[ -n "$mode" ]] && memory_args=(--sentence_memory "$mode")
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
    --config "$config_path" --checkpoint "$checkpoint_path" --split val \
    --manifest "$QUERY_MANIFEST" --num_samples 0 --selection_mode first \
    --out_dir "$build_dir" --batch_size "$BATCH_SIZE" --device cuda \
    --text_device cpu --length_mode predicted --word_prior off \
    "${memory_args[@]}" --context_fps 20 --sample_fps 20
  if [[ "$scored" == "1" ]]; then
    local alignment
    for alignment in default pa; do
      srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m flow.evaluate.dtw_mpjpe_t2m_default \
        --samples_dir "$build_dir" \
        --out_json "$build_dir/dtw_mpjpe_t2m_${alignment}_h2s_betas.json" \
        --out_csv "$build_dir/dtw_mpjpe_t2m_${alignment}_h2s_betas.csv" \
        --sample_key smplx --gt_key smplx --prior_key adapter_context_smplx \
        --device cuda --betas_mode h2s_fixed --alignment_mode "$alignment" \
        --parts body lhand rhand face wholebody
    done
  fi
  rebase_and_publish "$build_dir" "$mode_dir"
  validate_published_export "$mode_dir" "$scored"
}

SCORED=0
[[ "$POSTSELECTION_PURPOSE" == "confirmation" ]] && SCORED=1
for mode in "${MODES[@]}"; do run_export "$mode" "$mode" "$SCORED"; done
run_export broadcast_complete broadcast_complete "$SCORED"
run_export all_null all_null 0

MODE_ARGS=()
for mode in "${MODES[@]}"; do MODE_ARGS+=("--${mode}_dir" "$EXPORT_ROOT/$mode"); done

if [[ "$POSTSELECTION_PURPOSE" == "diagnostic" ]]; then
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.analyze_centered_memory_diagnostic \
    --authorization "$AUTHORIZATION" --checkpoint "$CHECKPOINT" --config "$CFG" \
    --partition_dir "$PARTITION_DIR" --broadcast_complete_dir "$EXPORT_ROOT/broadcast_complete" \
    --all_null_dir "$EXPORT_ROOT/all_null" --out_dir "$DIAGNOSTIC_DIR" \
    --bootstrap_samples 10000 --seed 1234 "${MODE_ARGS[@]}"
  "$PYTHON_BIN" -m "$DECISION_MODULE" finalize-confirmation-authorization \
    --stage "$CENTERED_STAGE" --diagnostic_authorization "$AUTHORIZATION" \
    --diagnostic_dir "$DIAGNOSTIC_DIR" --out_file "$CONFIRMATION_AUTH"
  POSTSELECTION_COMPLETE=1
else
  run_export "" original_v2_text_only 0 "$V2_CFG" "$V2_CHECKPOINT"
  set +e
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.analyze_centered_memory_confirmation \
    --authorization "$AUTHORIZATION" --holdout_spend "$GLOBAL_HOLDOUT_SPEND" \
    --checkpoint "$CHECKPOINT" --config "$CFG" --partition_dir "$PARTITION_DIR" \
    --development_diagnostic_dir "$DIAGNOSTIC_DIR" \
    --broadcast_complete_dir "$EXPORT_ROOT/broadcast_complete" \
    --all_null_dir "$EXPORT_ROOT/all_null" \
    --v2_text_only_dir "$EXPORT_ROOT/original_v2_text_only" \
    --v2_checkpoint "$V2_CHECKPOINT" --v2_config "$V2_CFG" \
    --out_dir "$CONFIRMATION_ROOT/analysis" --bootstrap_samples 10000 --seed 1234 \
    "${MODE_ARGS[@]}"
  ANALYSIS_STATUS=$?
  set -e
  if [[ "$ANALYSIS_STATUS" != "0" && "$ANALYSIS_STATUS" != "2" ]]; then
    exit "$ANALYSIS_STATUS"
  fi
  [[ -f "$CONFIRMATION_ROOT/analysis/READY" ]] || { echo "ERROR: confirmation analysis is incomplete" >&2; exit 1; }
  POSTSELECTION_COMPLETE=1
  if [[ "$ANALYSIS_STATUS" == "2" ]]; then
    echo "Confirmation completed but failed promotion gates; holdout remains spent." >&2
    exit 2
  fi
fi
