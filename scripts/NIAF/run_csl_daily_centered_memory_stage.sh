#!/bin/bash

# Shared fail-closed driver for ordered centered Phase-A''' full/resume jobs.
set -euo pipefail
trap 'echo "ERROR: centered-memory stage failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to its clean pushed commit}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
CENTERED_STAGE="${CENTERED_STAGE:?CENTERED_STAGE must be stage1 or stage2}"
CENTERED_RESUME="${CENTERED_RESUME:-0}"
CENTERED_LAUNCHER="${CENTERED_LAUNCHER:?Wrapper must export its exact path}"
DECISION_MODULE="NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage"
DRIVER="$PROJECT_DIR/scripts/NIAF/run_csl_daily_centered_memory_stage.sh"
TRAIN_WRAPPER="$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
CALIBRATION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_v1_retry2"
PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"
EXPECTED_DEV_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
CONTROL_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_ordered_v1_control"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"

case "$CENTERED_STAGE" in
  stage1)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1"
    ;;
  stage2)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1"
    ;;
  *) echo "ERROR: CENTERED_STAGE must be stage1 or stage2" >&2; exit 1 ;;
esac
case "$CENTERED_RESUME" in
  0|1) ;;
  *) echo "ERROR: CENTERED_RESUME must be 0 or 1" >&2; exit 1 ;;
esac
PRECHECKPOINT_RETRY=0

CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${EXPERIMENT}.yaml"
CANONICAL_OUT_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$EXPERIMENT"
if [[ -n "${OUT_DIR:-}" && "$OUT_DIR" != "$CANONICAL_OUT_DIR" ]]; then
  echo "ERROR: centered stages forbid a noncanonical OUT_DIR" >&2
  exit 1
fi
OUT_DIR="$CANONICAL_OUT_DIR"
STAGE1_RUN="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1"
CANONICAL_STAGE1_AUTHORIZATION="$STAGE1_RUN/evaluation/ordered_development_decision/authorize_stage2.json"
if [[ -n "${STAGE1_AUTHORIZATION:-}" && "$STAGE1_AUTHORIZATION" != "$CANONICAL_STAGE1_AUTHORIZATION" ]]; then
  echo "ERROR: centered stages forbid a noncanonical Stage-A authorization" >&2
  exit 1
fi
STAGE1_AUTHORIZATION="$CANONICAL_STAGE1_AUTHORIZATION"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "4" ]]; then
  echo "ERROR: centered full training requires four Slurm nodes" >&2
  exit 1
fi
if [[ ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: SOURCE_GIT_HEAD must be a full Git SHA1" >&2
  exit 1
fi
for required in "$PYTHON_BIN" "$CFG" "$V2_CHECKPOINT" "$CENTERED_LAUNCHER" "$DRIVER" "$TRAIN_WRAPPER" "$STAGER" "$CALIBRATION_DIR/READY"; do
  [[ -e "$required" ]] || { echo "ERROR: prerequisite is absent: $required" >&2; exit 1; }
done
[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: the single global confirmation holdout was already spent" >&2
  exit 1
}
if [[ "$CENTERED_RESUME" == "0" ]]; then
  if [[ -e "$OUT_DIR" || -e "${OUT_DIR}.prerequisites" ]]; then
    echo "ERROR: fresh stage requires fresh output and prerequisite paths" >&2
    exit 1
  fi
else
  if [[ ! -d "$OUT_DIR" && ! -d "${OUT_DIR}.prerequisites" ]]; then
    echo "ERROR: exact recovery requires this stage's existing evidence" >&2
    exit 1
  fi
  if [[ ! -f "$OUT_DIR/checkpoints/last.pt" ]]; then
    PRECHECKPOINT_RETRY=1
  fi
fi
if [[ -e "$OUT_DIR/evaluation/ordered_development_decision" ]]; then
  echo "ERROR: a decided centered stage cannot resume or relaunch" >&2
  exit 1
fi
for forbidden in WARM_START BASE_CHECKPOINT PHASE_B_GATE_REPORT LIMIT_TRAIN LIMIT_VAL MAX_TRAIN_BATCHES MAX_VAL_BATCHES; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "ERROR: centered full training forbids inherited $forbidden" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git ]]; then
  echo "ERROR: PROJECT_DIR must be a shared standalone clone" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: source must resolve below /media/cvpr" >&2; exit 1 ;;
esac
for durable in "$PROJECT_DIR/experiments" "$PROJECT_DIR/deps/mt5-base"; do
  [[ -d "$durable" ]] || { echo "ERROR: durable dependency absent: $durable" >&2; exit 1; }
  case "$(realpath -e "$durable")" in
    /media/cvpr/*) ;;
    *) echo "ERROR: dependency is not durable shared storage: $durable" >&2; exit 1 ;;
  esac
done
if [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: source uses external worktree metadata" >&2
  exit 1
fi
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "${ACTUAL_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: source must equal the exact clean SOURCE_GIT_HEAD" >&2
  exit 1
fi
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
[[ -n "$SOURCE_REMOTE_BRANCH" ]] || { echo "ERROR: detached HEAD requires SOURCE_REMOTE_BRANCH" >&2; exit 1; }
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$REMOTE_HEAD" || "${REMOTE_HEAD,,}" != "${ACTUAL_HEAD,,}" ]]; then
  echo "ERROR: source is not the exact pushed origin branch head" >&2
  exit 1
fi
if [[ "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_V2_SHA256" ]]; then
  echo "ERROR: frozen v2 checkpoint identity changed" >&2
  exit 1
fi
if [[ ! -f "$PARTITION_DIR/READY" || ! -f "$PARTITION_DIR/manifest_development.jsonl" \
      || "$(sha256sum -- "$PARTITION_DIR/READY" | awk '{print $1}')" != "$EXPECTED_PARTITION_READY_SHA256" \
      || "$(sha256sum -- "$PARTITION_DIR/manifest_development.jsonl" | awk '{print $1}')" != "$EXPECTED_DEV_SHA256" \
      || "$(awk 'NF {n += 1} END {print n + 0}' "$PARTITION_DIR/manifest_development.jsonl")" != "347" ]]; then
  echo "ERROR: sealed 347-row development partition changed" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

# Reopen the seal and bind its source files/commit/remote to this exact clean
# checkout before any model/provider construction.
"$PYTHON_BIN" - "$CALIBRATION_DIR" "$PROJECT_DIR" "$SOURCE_GIT_HEAD" \
  "origin/$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" <<'PY'
import sys
from NIAF.continuous_trajectory_field.relevance_calibration import (
    validate_relevance_calibration_artifact,
    validate_relevance_calibration_source,
)

artifact = validate_relevance_calibration_artifact(sys.argv[1])
validate_relevance_calibration_source(
    artifact,
    source_root=sys.argv[2],
    expected_git_head=sys.argv[3],
    expected_remote_ref=sys.argv[4],
    expected_remote_head=sys.argv[5],
)
PY

# Stage B consumes its exact Stage-A valid-infeasible authorization before any
# memory provider or node-local bank is initialized.
if [[ "$CENTERED_STAGE" == "stage2" ]]; then
  "$PYTHON_BIN" -m "$DECISION_MODULE" verify \
    --purpose stage2 --stage stage1 --authorization "$STAGE1_AUTHORIZATION"
fi

LEASE="${OUT_DIR}.prerequisites/active_training_execution_lease"
LEASE_ATTESTATION="${OUT_DIR}.prerequisites/execution_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
BINDING_IDENTITY="$(sha256sum -- "$CFG" | awk '{print $1}')"
"$PYTHON_BIN" -m "$DECISION_MODULE" acquire-execution-lease \
  --purpose training --stage "$CENTERED_STAGE" \
  --lease "$LEASE" --out_file "$LEASE_ATTESTATION" \
  --source_git_head "$SOURCE_GIT_HEAD" --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$BINDING_IDENTITY"

LOCAL_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
cleanup_and_release() {
  local exit_code=$?
  trap - EXIT
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    bash "$STAGER" cleanup "$LOCAL_BANK" || true
  if [[ "$exit_code" == "0" ]]; then
    "$PYTHON_BIN" -m "$DECISION_MODULE" release-execution-lease \
      --lease "$LEASE" --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup_and_release EXIT

if [[ "$CENTERED_RESUME" == "1" && "$PRECHECKPOINT_RETRY" == "1" ]]; then
  RUN_LAUNCH_FILE="${OUT_DIR}.prerequisites/run_launch_identity.json"
  if [[ ! -f "$RUN_LAUNCH_FILE" ]]; then
    RECOVERY_LAUNCH_ARGS=(
      record-run-launch --stage "$CENTERED_STAGE"
      --out_file "$RUN_LAUNCH_FILE"
      --config "$CFG" --v2_checkpoint "$V2_CHECKPOINT"
      --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD"
      --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD"
      --launcher "$CENTERED_LAUNCHER" --launcher "$DRIVER"
      --launcher "$TRAIN_WRAPPER" --launcher "$STAGER"
    )
    if [[ "$CENTERED_STAGE" == "stage2" ]]; then
      RECOVERY_LAUNCH_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
    fi
    "$PYTHON_BIN" -m "$DECISION_MODULE" "${RECOVERY_LAUNCH_ARGS[@]}"
  fi
  if [[ "$CENTERED_STAGE" == "stage2" \
        && ! -f "${OUT_DIR}.prerequisites/ordered_stage_input.json" ]]; then
    "$PYTHON_BIN" -m "$DECISION_MODULE" record-stage2-input \
      --authorization "$STAGE1_AUTHORIZATION" \
      --out_file "${OUT_DIR}.prerequisites/ordered_stage_input.json" \
      --config "$CFG" --v2_checkpoint "$V2_CHECKPOINT" \
      --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD" \
      --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD"
  fi
  PRECHECKPOINT_ARGS=(
    record-precheckpoint-retry --stage "$CENTERED_STAGE"
    --run_dir "$OUT_DIR"
    --out_file "${OUT_DIR}.prerequisites/precheckpoint_retry_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
    --config "$CFG" --partition_dir "$PARTITION_DIR"
    --lease "$LEASE" --lease_attestation "$LEASE_ATTESTATION"
    --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD"
    --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD"
  )
  if [[ "$CENTERED_STAGE" == "stage2" ]]; then
    PRECHECKPOINT_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
  fi
  "$PYTHON_BIN" -m "$DECISION_MODULE" "${PRECHECKPOINT_ARGS[@]}"
elif [[ "$CENTERED_RESUME" == "0" ]]; then
  LAUNCH_ARGS=(
    record-run-launch --stage "$CENTERED_STAGE"
    --out_file "${OUT_DIR}.prerequisites/run_launch_identity.json"
    --config "$CFG" --v2_checkpoint "$V2_CHECKPOINT"
    --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD"
    --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD"
    --launcher "$CENTERED_LAUNCHER" --launcher "$DRIVER"
    --launcher "$TRAIN_WRAPPER" --launcher "$STAGER"
  )
  if [[ "$CENTERED_STAGE" == "stage2" ]]; then
    LAUNCH_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
  fi
  "$PYTHON_BIN" -m "$DECISION_MODULE" "${LAUNCH_ARGS[@]}"
  if [[ "$CENTERED_STAGE" == "stage2" ]]; then
    "$PYTHON_BIN" -m "$DECISION_MODULE" record-stage2-input \
      --authorization "$STAGE1_AUTHORIZATION" \
      --out_file "${OUT_DIR}.prerequisites/ordered_stage_input.json" \
      --config "$CFG" --v2_checkpoint "$V2_CHECKPOINT" \
      --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD" \
      --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD"
  fi
else
  RESUME_ARGS=(
    record-run-resume --stage "$CENTERED_STAGE" --run_dir "$OUT_DIR"
    --out_file "${OUT_DIR}.prerequisites/resume_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
    --config "$CFG" --partition_dir "$PARTITION_DIR"
    --last_checkpoint "$OUT_DIR/checkpoints/last.pt"
    --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD"
    --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD"
    --launcher "$CENTERED_LAUNCHER" --launcher "$DRIVER"
    --launcher "$TRAIN_WRAPPER" --launcher "$STAGER"
  )
  if [[ "$CENTERED_STAGE" == "stage2" ]]; then
    RESUME_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
  fi
  "$PYTHON_BIN" -m "$DECISION_MODULE" "${RESUME_ARGS[@]}"
  TERMINAL_FINALIZED="$($PYTHON_BIN - "${OUT_DIR}.prerequisites/resume_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json" "$OUT_DIR/selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

attestation = json.load(open(sys.argv[1], encoding="utf-8"))
terminal = attestation.get("terminal_finalization")
if terminal is None:
    print("0")
else:
    summary = Path(sys.argv[2])
    if (
        terminal.get("trainer_reentry") is not False
        or Path(terminal.get("selection_summary_path", "")).resolve()
        != summary.resolve()
        or json.load(summary.open(encoding="utf-8"))
        != terminal.get("selection_summary_payload")
    ):
        raise SystemExit("ERROR: terminal no-op finalization is not attestation-bound")
    print("1")
PY
  )"
  if [[ "$TERMINAL_FINALIZED" == "1" ]]; then
    echo "Exact recovery finalized terminal selection; trainer re-entry is unnecessary."
    exit 0
  fi
fi

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  bash "$STAGER" stage "$LOCAL_BANK" "$SOURCE_BANK" "$PROJECT_DIR" \
  "$PYTHON_BIN" "$CFG" "train val"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  bash -c 'set -euo pipefail; d="$1"; [[ -f "$d/neighbors_train.npz" && -f "$d/neighbors_val.npz" && ! -e "$d/neighbors_test.npz" ]]' \
  bash "$LOCAL_BANK"

export CFG OUT_DIR RUN_TAG="$EXPERIMENT" EPOCHS=6
export SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR="$PARTITION_DIR"
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_BANK"
export SENTENCE_MEMORY_DIR="" STAGE_SENTENCE_MEMORY=0
export DEVICE=cuda TEXT_DEVICE=cpu DISTRIBUTED=ddp DDP_BACKEND=nccl
if [[ "$CENTERED_RESUME" == "1" && "$PRECHECKPOINT_RETRY" == "0" ]]; then
  export RESUME="$OUT_DIR/checkpoints/last.pt"
else
  unset RESUME
fi
bash "$TRAIN_WRAPPER"
