#!/bin/bash

# Prepare, or explicitly submit, one matched Stage-C pilot on a cabled pair.
# Without --submit this is a side-effect-free command preview.
set -euo pipefail

usage() {
  echo "usage: $0 <pair01..pair15> [--submit]" >&2
  exit 2
}

PAIR_CONSTRAINT="${1:-}"
ACTION="${2:-}"
[[ "$PAIR_CONSTRAINT" =~ ^pair(0[1-9]|1[0-5])$ ]] || usage
[[ -z "$ACTION" || "$ACTION" == "--submit" ]] || usage

# The dependency chain is deliberately offline.  Do not let `--export=ALL`
# propagate an ambient W&B credential into any Slurm job environment.
unset WANDB_API_KEY
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the exact pushed source commit}"
[[ "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ ]] || {
  echo "ERROR: SOURCE_GIT_HEAD must be a full Git SHA1" >&2
  exit 1
}
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD,,}"
export SOURCE_GIT_HEAD
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:?Set SOURCE_REMOTE_BRANCH to the exact pushed branch}"
EXPECTED_SOURCE_REMOTE_BRANCH="codex/csl-daily-centered-generator-stage-c-v5-run-r6"
BATCH_SCRIPT="$PROJECT_DIR/scripts/NIAF/train_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh"
SMOKE_SCRIPT="$PROJECT_DIR/scripts/NIAF/smoke_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh"
CALIBRATION_SCRIPT="$PROJECT_DIR/scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh"
CPU_SCRIPT="$PROJECT_DIR/scripts/NIAF/test_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
PREREQUISITE_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_pilot_prerequisites"
PREREQUISITE_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_prerequisites/source_${SOURCE_GIT_HEAD,,}"
CPU_READY="$PREREQUISITE_ROOT/cpu_gate/READY"
CALIBRATION_COMPLETION="$PREREQUISITE_ROOT/calibration/COMPLETE.json"
SMOKE_READY="$PREREQUISITE_ROOT/smoke/PUBLICATION/READY"
PILOT_READY="$PREREQUISITE_ROOT/pilot/PUBLICATION/READY"
CALIBRATION_LEASE="$PREREQUISITE_ROOT/calibration/active_execution_lease"
SMOKE_LEASE="$PREREQUISITE_ROOT/smoke/active_execution_lease"
PILOT_LEASE="$PREREQUISITE_ROOT/pilot/active_execution_lease"
CALIBRATION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_protocol_v5_run_r6"
MEMORY_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v5_run_r6.yaml"
OFF_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v5_run_r6.yaml"
RECOVERY_POLICY="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_decision_policy_v1.json"
RECOVERY_MANIFEST="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_recovery_evidence_v1.json"
RECOVERY_EVIDENCE_ROOT="/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory"
SOURCE_DECISION="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/evaluation/ordered_development_decision/decision.json"
SMOKE_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v5_run_r6_smoke"
PILOT_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v5_run_r6_pilot"
SMOKE_ATTEMPTS_ROOT="$SMOKE_ROOT/sources/source_${SOURCE_GIT_HEAD,,}/attempts"
PILOT_ATTEMPTS_ROOT="$PILOT_ROOT/sources/source_${SOURCE_GIT_HEAD,,}/attempts"

[[ "$SOURCE_REMOTE_BRANCH" == "$EXPECTED_SOURCE_REMOTE_BRANCH" \
   && -f "$BATCH_SCRIPT" && -f "$SMOKE_SCRIPT" && -f "$CALIBRATION_SCRIPT" \
   && -f "$CPU_SCRIPT" && -x "$PYTHON_BIN" ]] || {
  echo "ERROR: invalid Stage-C launch chain or SOURCE_GIT_HEAD" >&2
  exit 1
}

cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "${SOURCE_GIT_HEAD,,}" \
   && -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "ERROR: launcher requires the exact clean Stage-C source" >&2
  exit 1
}
REMOTE_HEAD="$(timeout 120s git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
[[ "$REMOTE_HEAD" == "${SOURCE_GIT_HEAD,,}" ]] || {
  echo "ERROR: launcher source is not the exact pushed remote head" >&2
  exit 1
}
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONNOUSERSITE=1

"$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-protocol-v5-runtime --policy_path "$RECOVERY_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT" >/dev/null

# Refuse to create or preview a second dependency chain while any Stage-C job
# from any run generation is live.  This is diagnostic only: recovery and
# cancellation remain explicit operator actions.
ACTIVE_STAGE_C_JOBS="$(
  squeue -h -u "${USER:?USER is required for the Stage-C job audit}" \
    -o '%i|%j|%T' |
    awk -F'|' '$2 ~ /^csl_stage_c(_|$)/ {print}'
)"
if [[ -n "$ACTIVE_STAGE_C_JOBS" ]]; then
  echo "ERROR: active Stage-C Slurm job(s) already exist; no job submitted:" >&2
  printf '%s\n' "$ACTIVE_STAGE_C_JOBS" >&2
  exit 1
fi

HAVE_CPU=0 HAVE_CALIBRATION=0 HAVE_SMOKE=0 HAVE_PILOT=0
RECOVER_CALIBRATION=0 RECOVER_SMOKE=0 RECOVER_PILOT=0
for partial in \
  "$PREREQUISITE_ROOT/cpu_gate:$CPU_READY" \
  "$PREREQUISITE_ROOT/smoke/PUBLICATION:$SMOKE_READY" \
  "$PREREQUISITE_ROOT/pilot/PUBLICATION:$PILOT_READY"; do
  container="${partial%%:*}"
  marker="${partial#*:}"
  if [[ ( -e "$container" || -L "$container" ) \
        && ! -f "$marker" ]]; then
    echo "ERROR: partial immutable prerequisite requires audited recovery: $container" >&2
    exit 1
  fi
done
if [[ ( -e "$CALIBRATION_COMPLETION" || -L "$CALIBRATION_COMPLETION" ) \
      && ! -f "$CALIBRATION_COMPLETION" ]]; then
  echo "ERROR: calibration COMPLETE is not a regular file" >&2
  exit 1
fi
if [[ -e "$CPU_READY" ]]; then
  "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-protocol-v5-cpu --cpu_gate "$CPU_READY" --source_git_head "$SOURCE_GIT_HEAD" --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD" --policy_path "$RECOVERY_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT" --memory_config "$MEMORY_CFG" --matched_off_config "$OFF_CFG" >/dev/null
  HAVE_CPU=1
fi
if [[ -e "$CALIBRATION_COMPLETION" ]]; then
  [[ "$HAVE_CPU" == "1" ]] || { echo "ERROR: calibration exists without CPU gate" >&2; exit 1; }
  if [[ -d "$CALIBRATION_LEASE" && ! -L "$CALIBRATION_LEASE" ]]; then
    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.stage_c_calibration_control \
      inspect-completed --lease "$CALIBRATION_LEASE" \
      --completion "$CALIBRATION_COMPLETION" \
      --expected_source_git_head "$SOURCE_GIT_HEAD" \
      --expected_source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
      --expected_source_remote_head "$REMOTE_HEAD" >/dev/null
    RECOVER_CALIBRATION=1
  else
    [[ ! -e "$CALIBRATION_LEASE" ]] || { echo "ERROR: calibration lease is malformed" >&2; exit 1; }
    "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-foundation \
      --recovery_policy "$RECOVERY_POLICY" \
      --recovery_manifest "$RECOVERY_MANIFEST" \
      --recovery_evidence_root "$RECOVERY_EVIDENCE_ROOT" \
      --cpu_gate "$CPU_READY" --calibration_completion "$CALIBRATION_COMPLETION" \
      --calibration_dir "$CALIBRATION_DIR" --calibration_launcher "$CALIBRATION_SCRIPT" \
      --stage_c_config "$MEMORY_CFG" --source_terminal_decision "$SOURCE_DECISION" \
      --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD" \
      --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
      --source_remote_head "$REMOTE_HEAD" >/dev/null
    HAVE_CALIBRATION=1
  fi
fi
if [[ ! -e "$CALIBRATION_COMPLETION" \
      && ( -e "$PREREQUISITE_ROOT/calibration" \
           || -L "$PREREQUISITE_ROOT/calibration" \
           || -e "$CALIBRATION_DIR" || -L "$CALIBRATION_DIR" ) ]]; then
  echo "ERROR: protocol-v5/run-r6 partial calibration is terminal; no retry submitted" >&2
  exit 1
fi

# A run-r6 control directory without a publication is never a new-attempt
# opportunity.  The sole recoverable form is one already complete scientific
# execution awaiting immutable publication; submit that stage alone and let
# the runner authenticate every bound file.  Any partial/no-completion state
# is the protocol's terminal stop-no-retry outcome.
for mode_spec in \
  "smoke:$PREREQUISITE_ROOT/smoke:$SMOKE_READY:$SMOKE_ATTEMPTS_ROOT"; do
  mode="${mode_spec%%:*}"
  remainder="${mode_spec#*:}"
  control="${remainder%%:*}"
  remainder="${remainder#*:}"
  marker="${remainder%%:*}"
  attempts="${remainder#*:}"
  if [[ -e "$control" && ! -e "$marker" ]]; then
    mapfile -t publication_recovery_completions < <(
      if [[ -d "$attempts" && ! -L "$attempts" ]]; then
        find "$attempts" -mindepth 4 -maxdepth 4 \
          -type f -name COMPLETE.json -print | sort
      fi
    )
    [[ "${#publication_recovery_completions[@]}" == "1" ]] || {
      echo "ERROR: protocol-v5/run-r6 $mode is terminal without one complete publication-recovery execution" >&2
      exit 1
    }
    [[ "$HAVE_CALIBRATION" == "1" ]] || {
      echo "ERROR: smoke publication recovery lacks exact calibration" >&2
      exit 1
    }
    RECOVER_SMOKE=1
  fi
done
if [[ -e "$SMOKE_READY" && "$RECOVER_CALIBRATION" == "0" ]]; then
  [[ "$HAVE_CALIBRATION" == "1" ]] || { echo "ERROR: smoke exists without calibration" >&2; exit 1; }
  if [[ -d "$SMOKE_LEASE" && ! -L "$SMOKE_LEASE" ]]; then
    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.stage_c_execution_control \
      inspect-published --lease "$SMOKE_LEASE" \
      --completion "${SMOKE_READY%/READY}/COMPLETE.json" --ready "$SMOKE_READY" \
      --expected_source_git_head "$SOURCE_GIT_HEAD" \
      --expected_pair_constraint "$PAIR_CONSTRAINT" >/dev/null
    RECOVER_SMOKE=1
  else
    [[ ! -e "$SMOKE_LEASE" ]] || { echo "ERROR: smoke lease is malformed" >&2; exit 1; }
    "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-smoke \
      --smoke_ready "$SMOKE_READY" --smoke_root "$SMOKE_ROOT" \
      --source_git_head "$SOURCE_GIT_HEAD" --pair_constraint "$PAIR_CONSTRAINT" >/dev/null
    HAVE_SMOKE=1
  fi
fi
if [[ -e "$PREREQUISITE_ROOT/pilot" && ! -e "$PILOT_READY" \
      && "$RECOVER_CALIBRATION" == "0" && "$RECOVER_SMOKE" == "0" ]]; then
  mapfile -t pilot_publication_recovery_completions < <(
    if [[ -d "$PILOT_ATTEMPTS_ROOT" && ! -L "$PILOT_ATTEMPTS_ROOT" ]]; then
      find "$PILOT_ATTEMPTS_ROOT" -mindepth 4 -maxdepth 4 \
        -type f -name COMPLETE.json -print | sort
    fi
  )
  [[ "$HAVE_SMOKE" == "1" \
     && "${#pilot_publication_recovery_completions[@]}" == "1" ]] || {
    echo "ERROR: protocol-v5/run-r6 pilot is terminal without one complete publication-recovery execution" >&2
    exit 1
  }
  RECOVER_PILOT=1
fi
if [[ -e "$PILOT_READY" && "$RECOVER_CALIBRATION" == "0" \
      && "$RECOVER_SMOKE" == "0" ]]; then
  [[ "$HAVE_SMOKE" == "1" ]] || { echo "ERROR: pilot exists without smoke gate" >&2; exit 1; }
  if [[ -d "$PILOT_LEASE" && ! -L "$PILOT_LEASE" ]]; then
    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.stage_c_execution_control \
      inspect-published --lease "$PILOT_LEASE" \
      --completion "${PILOT_READY%/READY}/COMPLETE.json" --ready "$PILOT_READY" \
      --expected_source_git_head "$SOURCE_GIT_HEAD" \
      --expected_pair_constraint "$PAIR_CONSTRAINT" >/dev/null
    RECOVER_PILOT=1
  else
    [[ ! -e "$PILOT_LEASE" ]] || { echo "ERROR: pilot lease is malformed" >&2; exit 1; }
    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.stage_c_execution_control \
      validate-finalized --lease "$PILOT_LEASE" \
      --completion "${PILOT_READY%/READY}/COMPLETE.json" --ready "$PILOT_READY" \
      --expected_source_git_head "$SOURCE_GIT_HEAD" \
      --expected_pair_constraint "$PAIR_CONSTRAINT" >/dev/null
  "$PYTHON_BIN" - "$PILOT_READY" "$SOURCE_GIT_HEAD" "$PAIR_CONSTRAINT" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

ready_path = Path(sys.argv[1]).resolve()
ready = json.loads(ready_path.read_text(encoding="utf-8"))
complete_path = Path(ready.get("complete_path", ""))
complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else {}
decision_path = Path(ready.get("decision_path", ""))
decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else {}
execution_path = Path(complete.get("execution_complete_path", ""))
execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.is_file() else {}
decision_unsigned = {key: value for key, value in decision.items() if key != "decision_identity"}
decision_identity = hashlib.sha256(
    json.dumps(
        decision_unsigned, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
expected_summaries = {
    arm: {
        "epoch": 1, "global_step": 72, "data_limit_train": 0,
        "data_limit_val": 0, "train_max_batches": 0,
        "eval_max_batches": 0, "length_bucketed_batches": True,
        "drop_last": False,
    }
    for arm in ("memory", "matched_off")
}
status = ready.get("decision_status")
next_action = (
    "none_requires_fresh_preregistration_without_pilot_outcome_access"
    if status == "pilot_complete_development_signal" else "none"
)
if (
    set(ready) != {
        "schema_name", "schema_version", "execution_mode", "source_git_head",
        "pair_constraint", "active_lease_claim_identity",
        "execution_lease_claim_identity", "complete_path", "complete_sha256",
        "decision_path", "decision_sha256", "decision_identity",
        "decision_status", "expected_epoch", "expected_global_step_per_arm",
        "one_optimizer_update_per_arm", "one_full_train_epoch_per_arm",
        "development_only", "non_authorizing", "promotion_eligible",
        "authorized_purpose", "confirmation_manifest_opened", "test_data_accessed",
        "prior_one_update_smoke",
    }
    or ready.get("schema_name") != "signtrajfield_stage_c_mode_ready"
    or ready.get("schema_version") != 1
    or ready.get("execution_mode") != "pilot"
    or ready.get("source_git_head") != sys.argv[2].lower()
    or ready.get("pair_constraint") != sys.argv[3]
    or status not in {"pilot_complete_development_signal", "stop"}
    or ready.get("expected_epoch") != 1
    or ready.get("expected_global_step_per_arm") != 72
    or ready.get("one_optimizer_update_per_arm") is not False
    or ready.get("one_full_train_epoch_per_arm") is not True
    or ready.get("development_only") is not True
    or ready.get("non_authorizing") is not True
    or ready.get("promotion_eligible") is not False
    or ready.get("authorized_purpose") is not None
    or ready.get("confirmation_manifest_opened") is not False
    or ready.get("test_data_accessed") is not False
    or complete_path.resolve() != (ready_path.parent / "COMPLETE.json").resolve()
    or ready.get("complete_sha256") != sha(complete_path)
    or complete.get("schema_name") != "signtrajfield_stage_c_mode_complete"
    or complete.get("execution_mode") != "pilot"
    or complete.get("source_git_head") != sys.argv[2].lower()
    or complete.get("pair_constraint") != sys.argv[3]
    or complete.get("expected_epoch") != 1
    or complete.get("expected_global_step_per_arm") != 72
    or complete.get("world_size") != 2
    or complete.get("batch_per_rank") != 64
    or complete.get("accumulation_steps") != 2
    or complete.get("effective_global_batch") != 256
    or complete.get("development_only") is not True
    or complete.get("non_authorizing") is not True
    or complete.get("promotion_eligible") is not False
    or complete.get("authorized_purpose") is not None
    or complete.get("confirmation_manifest_opened") is not False
    or complete.get("test_data_accessed") is not False
    or complete.get("decision_sha256") != sha(decision_path)
    or complete.get("decision_identity") != decision_identity
    or decision.get("decision_identity") != decision_identity
    or decision.get("status") != status
    or decision.get("next_permitted_action") != next_action
    or decision.get("execution_mode") != "pilot"
    or decision.get("source_git_head") != sys.argv[2].lower()
    or decision.get("pair_constraint") != sys.argv[3]
    or decision.get("development_only") is not True
    or decision.get("non_authorizing") is not True
    or decision.get("promotion_eligible") is not False
    or decision.get("authorized_purpose") is not None
    or complete.get("execution_complete_sha256") != sha(execution_path)
    or execution.get("execution_mode") != "pilot"
    or execution.get("source_git_head") != sys.argv[2].lower()
    or execution.get("pair_constraint") != sys.argv[3]
    or execution.get("arm_execution_summaries") != expected_summaries
    or not isinstance(ready.get("prior_one_update_smoke"), dict)
    or complete.get("prior_one_update_smoke")
    != ready.get("prior_one_update_smoke")
    or decision.get("prior_one_update_smoke")
    != ready.get("prior_one_update_smoke")
    or execution.get("prior_one_update_smoke")
    != ready.get("prior_one_update_smoke")
):
    raise SystemExit("ERROR: immutable Stage-C pilot publication is invalid")
PY
    HAVE_PILOT=1
  fi
fi
mapfile -t PAIR_NODES < <(
  sinfo -N -h -p spark -o '%N|%f' |
    awk -F'|' -v pair="$PAIR_CONSTRAINT" '
      {
        count = split($2, features, ",")
        for (feature_index = 1; feature_index <= count; feature_index += 1) {
          if (features[feature_index] == pair) {
            print $1
            break
          }
        }
      }
    ' |
    sort -u
)
[[ "${#PAIR_NODES[@]}" == "2" ]] || {
  echo "ERROR: $PAIR_CONSTRAINT must identify exactly two spark nodes; got ${#PAIR_NODES[@]}" >&2
  exit 1
}

EXPORTS="ALL,PROJECT_DIR=$PROJECT_DIR,SOURCE_GIT_HEAD=$SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH=$SOURCE_REMOTE_BRANCH,STAGE_C_PAIR_CONSTRAINT=$PAIR_CONSTRAINT"
printf 'Validated %s nodes: %s %s\n' "$PAIR_CONSTRAINT" "${PAIR_NODES[0]}" "${PAIR_NODES[1]}"
printf 'Existing exact stages: cpu=%s calibration=%s smoke=%s pilot=%s\n' \
  "$HAVE_CPU" "$HAVE_CALIBRATION" "$HAVE_SMOKE" "$HAVE_PILOT"
if [[ "$RECOVER_CALIBRATION" == "1" || "$RECOVER_SMOKE" == "1" \
      || "$RECOVER_PILOT" == "1" ]]; then
  if [[ "$RECOVER_CALIBRATION" == "1" ]]; then
    RECOVERY_LABEL="calibration" RECOVERY_SCRIPT="$CALIBRATION_SCRIPT"
    RECOVERY_CONSTRAINT=()
  elif [[ "$RECOVER_SMOKE" == "1" ]]; then
    RECOVERY_LABEL="smoke" RECOVERY_SCRIPT="$SMOKE_SCRIPT"
    RECOVERY_CONSTRAINT=(--constraint="$PAIR_CONSTRAINT")
  else
    RECOVERY_LABEL="pilot" RECOVERY_SCRIPT="$BATCH_SCRIPT"
    RECOVERY_CONSTRAINT=(--constraint="$PAIR_CONSTRAINT")
    "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-smoke \
      --smoke_ready "$SMOKE_READY" --smoke_root "$SMOKE_ROOT" \
      --source_git_head "$SOURCE_GIT_HEAD" \
      --pair_constraint "$PAIR_CONSTRAINT" >/dev/null
  fi
  if [[ "$ACTION" == "--submit" ]]; then
    RECOVERY_JOB_ID="$(sbatch --parsable "${RECOVERY_CONSTRAINT[@]}" \
      --export="$EXPORTS" "$RECOVERY_SCRIPT")"
    printf 'RECOVERY_STAGE=%s RECOVERY_JOB_ID=%s\n' \
      "$RECOVERY_LABEL" "$RECOVERY_JOB_ID"
  else
    printf 'NEXT recovery-%s: sbatch --parsable' "$RECOVERY_LABEL"
    if [[ "${#RECOVERY_CONSTRAINT[@]}" -gt 0 ]]; then
      printf ' --constraint=%q' "$PAIR_CONSTRAINT"
    fi
    printf ' --export=%q %q\n' "$EXPORTS" "$RECOVERY_SCRIPT"
    echo "Preview only; recovery is isolated and no downstream job is scheduled."
  fi
  exit 0
fi
if [[ "$ACTION" == "--submit" ]]; then
  [[ "$HAVE_PILOT" == "0" ]] || {
    echo "ERROR: immutable Stage-C pilot already exists; no job submitted" >&2
    exit 1
  }
  PREVIOUS_JOB_ID=""
  if [[ "$HAVE_CPU" == "0" ]]; then
    CPU_JOB_ID="$(sbatch --parsable --export="$EXPORTS" "$CPU_SCRIPT")"
    PREVIOUS_JOB_ID="$CPU_JOB_ID"
    printf 'CPU_GATE_JOB_ID=%s\n' "$CPU_JOB_ID"
  fi
  if [[ "$HAVE_CALIBRATION" == "0" ]]; then
    CALIBRATION_ARGS=(--parsable --export="$EXPORTS")
    [[ -z "$PREVIOUS_JOB_ID" ]] || CALIBRATION_ARGS+=(--dependency="afterok:$PREVIOUS_JOB_ID")
    CALIBRATION_JOB_ID="$(sbatch "${CALIBRATION_ARGS[@]}" "$CALIBRATION_SCRIPT")"
    PREVIOUS_JOB_ID="$CALIBRATION_JOB_ID"
    printf 'CALIBRATION_JOB_ID=%s\n' "$CALIBRATION_JOB_ID"
  fi
  if [[ "$HAVE_SMOKE" == "0" ]]; then
    SMOKE_ARGS=(--parsable --constraint="$PAIR_CONSTRAINT" --export="$EXPORTS")
    [[ -z "$PREVIOUS_JOB_ID" ]] || SMOKE_ARGS+=(--dependency="afterok:$PREVIOUS_JOB_ID")
    SMOKE_JOB_ID="$(sbatch "${SMOKE_ARGS[@]}" "$SMOKE_SCRIPT")"
    PREVIOUS_JOB_ID="$SMOKE_JOB_ID"
    printf 'SMOKE_JOB_ID=%s\n' "$SMOKE_JOB_ID"
  fi
  PILOT_ARGS=(--parsable --constraint="$PAIR_CONSTRAINT" --export="$EXPORTS")
  [[ -z "$PREVIOUS_JOB_ID" ]] || PILOT_ARGS+=(--dependency="afterok:$PREVIOUS_JOB_ID")
  if [[ -z "$PREVIOUS_JOB_ID" ]]; then
    "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-smoke \
      --smoke_ready "$SMOKE_READY" --smoke_root "$SMOKE_ROOT" \
      --source_git_head "$SOURCE_GIT_HEAD" \
      --pair_constraint "$PAIR_CONSTRAINT" >/dev/null
  fi
  PILOT_JOB_ID="$(sbatch "${PILOT_ARGS[@]}" "$BATCH_SCRIPT")"
  printf 'PILOT_JOB_ID=%s\n' "$PILOT_JOB_ID"
else
  PREVIOUS_STAGE=""
  if [[ "$HAVE_CPU" == "0" ]]; then
    printf 'NEXT cpu: sbatch --parsable --export=%q %q\n' "$EXPORTS" "$CPU_SCRIPT"
    PREVIOUS_STAGE="CPU_JOB_ID"
  fi
  if [[ "$HAVE_CALIBRATION" == "0" ]]; then
    printf 'NEXT calibration: sbatch --parsable%s --export=%q %q\n' \
      "${PREVIOUS_STAGE:+ --dependency=afterok:\$$PREVIOUS_STAGE}" \
      "$EXPORTS" "$CALIBRATION_SCRIPT"
    PREVIOUS_STAGE="CALIBRATION_JOB_ID"
  fi
  if [[ "$HAVE_SMOKE" == "0" ]]; then
    printf 'NEXT smoke: sbatch --parsable%s --constraint=%q --export=%q %q\n' \
      "${PREVIOUS_STAGE:+ --dependency=afterok:\$$PREVIOUS_STAGE}" \
      "$PAIR_CONSTRAINT" "$EXPORTS" "$SMOKE_SCRIPT"
    PREVIOUS_STAGE="SMOKE_JOB_ID"
  fi
  if [[ "$HAVE_PILOT" == "0" ]]; then
    printf 'NEXT pilot: sbatch --parsable%s --constraint=%q --export=%q %q\n' \
      "${PREVIOUS_STAGE:+ --dependency=afterok:\$$PREVIOUS_STAGE}" \
      "$PAIR_CONSTRAINT" "$EXPORTS" "$BATCH_SCRIPT"
  else
    echo "No submission: immutable Stage-C pilot publication already exists."
  fi
  echo "Preview only; pass --submit to create only the incomplete dependency suffix."
fi
