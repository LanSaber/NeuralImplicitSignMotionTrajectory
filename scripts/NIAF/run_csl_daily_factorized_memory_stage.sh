#!/bin/bash

# Shared fail-closed driver for the two full Phase-A'' Slurm launchers.
set -euo pipefail
trap 'echo "ERROR: factorized-memory stage launch failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
FACTORIZED_STAGE="${FACTORIZED_STAGE:?FACTORIZED_STAGE must be stage1 or stage2}"
FACTORIZED_RESUME="${FACTORIZED_RESUME:-0}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the clean pushed commit selected on the submit host}"
FACTORIZED_LAUNCHER="${FACTORIZED_LAUNCHER:?Wrapper must export its exact launcher path}"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"

case "$FACTORIZED_STAGE" in
  stage1)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
    ;;
  stage2)
    EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1"
    ;;
  *)
    echo "ERROR: FACTORIZED_STAGE must be stage1 or stage2" >&2
    exit 1
    ;;
esac

CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${EXPERIMENT}.yaml"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$EXPERIMENT}"
SEALED_PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
PARTITION_DIR="${PARTITION_DIR:-$SEALED_PARTITION_DIR}"
EXPECTED_DEVELOPMENT_MANIFEST_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"
STAGE1_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
STAGE1_RUN_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$STAGE1_EXPERIMENT"
STAGE1_AUTHORIZATION="${STAGE1_AUTHORIZATION:-$STAGE1_RUN_DIR/evaluation/ordered_development_decision/authorize_stage2.json}"
CONTROL_DIR="${CONTROL_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control}"
HOLDOUT_SPEND="$CONTROL_DIR/confirmation_holdout_spent.json"

case "$FACTORIZED_RESUME" in
  0|1) ;;
  *) echo "ERROR: FACTORIZED_RESUME must be 0 or 1" >&2; exit 1 ;;
esac
PRECHECKPOINT_RETRY=0

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "4" ]]; then
  echo "ERROR: full factorized training requires a four-node Slurm allocation" >&2
  exit 1
fi
if [[ ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ && ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: SOURCE_GIT_HEAD is malformed" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$CFG" || ! -f "$V2_CHECKPOINT" ]]; then
  echo "ERROR: Python, full-stage config, or v2 checkpoint is missing" >&2
  exit 1
fi
if [[ ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: sentence-memory bank is not READY" >&2
  exit 1
fi
if [[ "$FACTORIZED_RESUME" == "0" && -e "$OUT_DIR" ]]; then
  echo "ERROR: initial full stage requires a fresh output directory: $OUT_DIR" >&2
  exit 1
fi
if [[ "$FACTORIZED_RESUME" == "1" ]]; then
  if [[ ! -d "$OUT_DIR" && ! -d "${OUT_DIR}.prerequisites" ]]; then
    echo "ERROR: exact recovery requires this stage's existing run evidence" >&2
    exit 1
  fi
  if [[ -e "$OUT_DIR/selection_summary.json" \
        || -e "$OUT_DIR/evaluation/ordered_development_decision" ]]; then
    echo "ERROR: terminal or decided training cannot be resumed" >&2
    exit 1
  fi
  if [[ ! -f "$OUT_DIR/checkpoints/last.pt" ]]; then
    PRECHECKPOINT_RETRY=1
  fi
fi
if [[ -e "$HOLDOUT_SPEND" ]]; then
  echo "ERROR: confirmation is already spent; no adaptive training is permitted" >&2
  exit 1
fi
for forbidden in \
  RESUME WARM_START BASE_CHECKPOINT PHASE_B_GATE_REPORT \
  LIMIT_TRAIN LIMIT_VAL MAX_TRAIN_BATCHES MAX_VAL_BATCHES; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "ERROR: fresh Phase-A'' stage forbids inherited $forbidden" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR"
if [[ ! -d "$PROJECT_DIR/.git" || -L "$PROJECT_DIR/.git" ]]; then
  echo "ERROR: PROJECT_DIR must be a shared standalone clone with a real .git directory; Codex worktree gitdir pointers are not cluster-visible" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: factorized training source clone must reside on /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/experiments" ]]; then
  echo "ERROR: shared source clone must expose the durable experiments directory" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/experiments")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: experiments must resolve to durable shared /media/cvpr storage" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/deps/mt5-base" ]]; then
  echo "ERROR: shared source clone must expose the frozen deps/mt5-base model" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/deps/mt5-base")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: frozen mT5 dependency must resolve to /media/cvpr" >&2; exit 1 ;;
esac
GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
if [[ "$(realpath -e "$GIT_COMMON_DIR")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: PROJECT_DIR uses external git metadata instead of a standalone clone" >&2
  exit 1
fi
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "${ACTUAL_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" ]]; then
  echo "ERROR: shared worktree HEAD differs from SOURCE_GIT_HEAD" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: full training requires a clean reproducible worktree" >&2
  exit 1
fi
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
if [[ -z "$SOURCE_REMOTE_BRANCH" ]]; then
  echo "ERROR: detached HEAD requires explicit SOURCE_REMOTE_BRANCH" >&2
  exit 1
fi
SOURCE_REMOTE_REF="origin/$SOURCE_REMOTE_BRANCH"
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$REMOTE_HEAD" || "${REMOTE_HEAD,,}" != "${ACTUAL_HEAD,,}" ]]; then
  echo "ERROR: live origin branch $SOURCE_REMOTE_BRANCH does not equal SOURCE_GIT_HEAD" >&2
  exit 1
fi
ACTUAL_V2_SHA256="$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_V2_SHA256" != "$EXPECTED_V2_SHA256" ]]; then
  echo "ERROR: v2 checkpoint SHA256 mismatch" >&2
  exit 1
fi
if [[ ! -f "$PARTITION_DIR/READY" \
      || ! -f "$PARTITION_DIR/manifest_development.jsonl" ]]; then
  echo "ERROR: full training requires the pre-existing sealed development partition" >&2
  exit 1
fi
if [[ "$(realpath -e "$PARTITION_DIR")" != "$(realpath -e "$SEALED_PARTITION_DIR")" \
      || "$(sha256sum -- "$PARTITION_DIR/READY" | awk '{print $1}')" != "$EXPECTED_PARTITION_READY_SHA256" \
      || "$(sha256sum -- "$PARTITION_DIR/manifest_development.jsonl" | awk '{print $1}')" != "$EXPECTED_DEVELOPMENT_MANIFEST_SHA256" \
      || "$(awk 'NF {count += 1} END {print count + 0}' "$PARTITION_DIR/manifest_development.jsonl")" != "347" ]]; then
  echo "ERROR: development-only partition identity/count changed" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

"$PYTHON_BIN" - "$CFG" "$V2_CHECKPOINT" "$FACTORIZED_STAGE" <<'PY'
import sys
from pathlib import Path
from NIAF.continuous_sign_field.config import load_config

cfg = load_config(Path(sys.argv[1]))
base = Path(str(cfg.get("train", {}).get("base_checkpoint", "")))
base = base if base.is_absolute() else Path.cwd() / base
if base.resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("ERROR: train.base_checkpoint is not the pinned v2 checkpoint")
expected_prior = "none" if sys.argv[3] == "stage1" else "gaussian"
memory = cfg.get("sentence_memory", {})
if memory.get("key_value_mode") != "factorized_metadata_motion_v1":
    raise SystemExit("ERROR: full stage is not factorized_metadata_motion_v1")
if memory.get("temporal_prior_mode") != expected_prior:
    raise SystemExit("ERROR: full stage has the wrong temporal prior")
if cfg.get("eval", {}).get("sentence_memory_modes") != [
    "off", "on", "motion_shuffled", "shuffled", "analytic_prior"
]:
    raise SystemExit("ERROR: full stage does not use the five fixed eval modes")
partition = cfg.get("validation_text_partition", {})
if partition.get("expected_partition_artifact_identity") != (
    "2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
):
    raise SystemExit("ERROR: full stage changed the sealed partition identity")
if partition.get("expected_development_manifest_sha256") != (
    "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
):
    raise SystemExit("ERROR: full stage changed the development manifest identity")
PY

LEASE_PATH="${OUT_DIR}.prerequisites/active_training_execution_lease"
LEASE_ATTESTATION="${OUT_DIR}.prerequisites/execution_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
  acquire-execution-lease \
  --purpose training --stage "$FACTORIZED_STAGE" \
  --lease "$LEASE_PATH" --out_file "$LEASE_ATTESTATION" \
  --source_git_head "$SOURCE_GIT_HEAD" --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$(sha256sum -- "$CFG" | awk '{print $1}')"
LEASE_HELD=1
release_training_lease_on_success() {
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
trap release_training_lease_on_success EXIT

if [[ "$FACTORIZED_RESUME" == "1" && "$PRECHECKPOINT_RETRY" == "1" ]]; then
  RUN_LAUNCH_FILE="${OUT_DIR}.prerequisites/run_launch_identity.json"
  if [[ ! -f "$RUN_LAUNCH_FILE" ]]; then
    RECOVERY_LAUNCH_ARGS=(
      record-run-launch
      --stage "$FACTORIZED_STAGE"
      --out_file "$RUN_LAUNCH_FILE"
      --config "$CFG"
      --v2_checkpoint "$V2_CHECKPOINT"
      --source_root "$PROJECT_DIR"
      --source_git_head "$SOURCE_GIT_HEAD"
      --source_remote_ref "$SOURCE_REMOTE_REF"
      --source_remote_head "$REMOTE_HEAD"
      --launcher "$FACTORIZED_LAUNCHER"
      --launcher "$PROJECT_DIR/scripts/NIAF/run_csl_daily_factorized_memory_stage.sh"
      --launcher "$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
    )
    if [[ "$FACTORIZED_STAGE" == "stage2" ]]; then
      RECOVERY_LAUNCH_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
    fi
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      "${RECOVERY_LAUNCH_ARGS[@]}"
  fi
  if [[ "$FACTORIZED_STAGE" == "stage2" \
        && ! -f "${OUT_DIR}.prerequisites/ordered_stage_input.json" ]]; then
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      record-stage2-input \
      --authorization "$STAGE1_AUTHORIZATION" \
      --out_file "${OUT_DIR}.prerequisites/ordered_stage_input.json" \
      --config "$CFG" --v2_checkpoint "$V2_CHECKPOINT" \
      --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD" \
      --source_remote_ref "$SOURCE_REMOTE_REF" --source_remote_head "$REMOTE_HEAD"
  fi
  PRECHECKPOINT_ATTESTATION="${OUT_DIR}.prerequisites/precheckpoint_retry_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
  PRECHECKPOINT_ARGS=(
    record-precheckpoint-retry
    --stage "$FACTORIZED_STAGE"
    --run_dir "$OUT_DIR"
    --out_file "$PRECHECKPOINT_ATTESTATION"
    --config "$CFG"
    --partition_dir "$PARTITION_DIR"
    --lease "$LEASE_PATH"
    --lease_attestation "$LEASE_ATTESTATION"
    --source_root "$PROJECT_DIR"
    --source_git_head "$SOURCE_GIT_HEAD"
    --source_remote_ref "$SOURCE_REMOTE_REF"
    --source_remote_head "$REMOTE_HEAD"
  )
  if [[ "$FACTORIZED_STAGE" == "stage2" ]]; then
    PRECHECKPOINT_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
  fi
  "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
    "${PRECHECKPOINT_ARGS[@]}"
elif [[ "$FACTORIZED_RESUME" == "0" ]]; then
  RECORD_LAUNCH_ARGS=(
    record-run-launch
    --stage "$FACTORIZED_STAGE"
    --out_file "${OUT_DIR}.prerequisites/run_launch_identity.json"
    --config "$CFG"
    --v2_checkpoint "$V2_CHECKPOINT"
    --source_root "$PROJECT_DIR"
    --source_git_head "$SOURCE_GIT_HEAD"
    --source_remote_ref "$SOURCE_REMOTE_REF"
    --source_remote_head "$REMOTE_HEAD"
    --launcher "$FACTORIZED_LAUNCHER"
    --launcher "$PROJECT_DIR/scripts/NIAF/run_csl_daily_factorized_memory_stage.sh"
    --launcher "$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
  )
  if [[ "$FACTORIZED_STAGE" == "stage2" ]]; then
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      verify --purpose stage2 --stage stage1 \
      --authorization "$STAGE1_AUTHORIZATION"
    RECORD_LAUNCH_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
  fi
  "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
    "${RECORD_LAUNCH_ARGS[@]}"

  if [[ "$FACTORIZED_STAGE" == "stage2" ]]; then
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      record-stage2-input \
      --authorization "$STAGE1_AUTHORIZATION" \
      --out_file "${OUT_DIR}.prerequisites/ordered_stage_input.json" \
      --config "$CFG" \
      --v2_checkpoint "$V2_CHECKPOINT" \
      --source_root "$PROJECT_DIR" \
      --source_git_head "$SOURCE_GIT_HEAD" \
      --source_remote_ref "$SOURCE_REMOTE_REF" \
      --source_remote_head "$REMOTE_HEAD"
  fi

else
  RESUME_ATTESTATION="${OUT_DIR}.prerequisites/resume_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
  RESUME_ARGS=(
    record-run-resume
    --stage "$FACTORIZED_STAGE"
    --run_dir "$OUT_DIR"
    --out_file "$RESUME_ATTESTATION"
    --config "$CFG"
    --partition_dir "$PARTITION_DIR"
    --last_checkpoint "$OUT_DIR/checkpoints/last.pt"
    --source_root "$PROJECT_DIR"
    --source_git_head "$SOURCE_GIT_HEAD"
    --source_remote_ref "$SOURCE_REMOTE_REF"
    --source_remote_head "$REMOTE_HEAD"
    --launcher "$FACTORIZED_LAUNCHER"
    --launcher "$PROJECT_DIR/scripts/NIAF/run_csl_daily_factorized_memory_stage.sh"
    --launcher "$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
  )
  if [[ "$FACTORIZED_STAGE" == "stage2" ]]; then
    RESUME_ARGS+=(--stage1_authorization "$STAGE1_AUTHORIZATION")
  fi
  "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
    "${RESUME_ARGS[@]}"
  TERMINAL_FINALIZED="$($PYTHON_BIN - "$RESUME_ATTESTATION" "$OUT_DIR/selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

attestation = json.load(open(sys.argv[1], encoding="utf-8"))
terminal = attestation.get("terminal_finalization")
if terminal is None:
    print("0")
else:
    summary_path = Path(sys.argv[2])
    if (terminal.get("trainer_reentry") is not False
            or Path(terminal.get("selection_summary_path", "")).resolve()
            != summary_path.resolve()
            or json.load(summary_path.open(encoding="utf-8"))
            != terminal.get("selection_summary_payload")):
        raise SystemExit("ERROR: terminal no-op finalization is not attestation-bound")
    print("1")
PY
  )"
  if [[ "$TERMINAL_FINALIZED" == "1" ]]; then
    echo "Exact recovery finalized the terminal selection summary; trainer re-entry is unnecessary."
    exit 0
  fi
fi

export CFG SENTENCE_MEMORY_DIR OUT_DIR
export SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR="$PARTITION_DIR"
export RUN_TAG="$EXPERIMENT"
export EPOCHS=4
export STAGE_SENTENCE_MEMORY=1 STAGE_ONLY_REQUESTED_NEIGHBORS=1
export SPLITS="train val"
export DEVICE=cuda TEXT_DEVICE=cpu DISTRIBUTED=ddp DDP_BACKEND=nccl

if [[ "$FACTORIZED_RESUME" == "1" && "$PRECHECKPOINT_RETRY" == "0" ]]; then
  export RESUME="$OUT_DIR/checkpoints/last.pt"
fi

bash "$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
