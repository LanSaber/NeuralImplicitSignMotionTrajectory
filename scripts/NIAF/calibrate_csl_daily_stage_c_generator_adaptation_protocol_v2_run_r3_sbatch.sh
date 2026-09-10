#!/bin/bash
#SBATCH --job-name=csl_stage_c_protocol_v2_run_r3_calibration
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# Fresh train-only relevance calibration bound to the immutable Stage-C source.
set -euo pipefail
trap 'echo "ERROR: Stage-C relevance calibration failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the exact pushed source commit}"
[[ "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ ]] || {
  echo "ERROR: SOURCE_GIT_HEAD must be a full Git SHA1" >&2
  exit 1
}
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD,,}"
export SOURCE_GIT_HEAD
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:?Set SOURCE_REMOTE_BRANCH to the exact pushed branch}"
EXPECTED_SOURCE_REMOTE_BRANCH="codex/csl-daily-centered-generator-stage-c-v2-run-r3"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
AUDIT_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1.yaml"
LAUNCHER="$PROJECT_DIR/scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
PREREQUISITE_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_pilot_prerequisites"
CALIBRATION_CONTROL_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_calibration_control"
TRANSITION_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_calibration_transition"
PREREQUISITE_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_pilot_prerequisites.py"
CALIBRATION_CONTROL_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_calibration_control.py"
TRANSITION_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_calibration_transition.py"
STAGE_C_CONFIG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v2_run_r3.yaml"
STAGE_C_OFF_CONFIG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v2_run_r3.yaml"
RECOVERY_POLICY="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_decision_policy_v1.json"
RECOVERY_MANIFEST="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_recovery_evidence_v1.json"
RECOVERY_EVIDENCE_ROOT="/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory"
SOURCE_DECISION="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/evaluation/ordered_development_decision/decision.json"
CALIBRATION_OUT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_protocol_v2_run_r3"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"
PREREQUISITE_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_prerequisites/source_${SOURCE_GIT_HEAD,,}"
CPU_GATE_READY="$PREREQUISITE_ROOT/cpu_gate/READY"
CONTROL_DIR="$PREREQUISITE_ROOT/calibration"
LEASE="$CONTROL_DIR/active_execution_lease"
COMPLETION="$CONTROL_DIR/COMPLETE.json"
LEASE_ATTESTATION="$CONTROL_DIR/lease_attestations/${SLURM_JOB_ID:-absent}.${SLURM_RESTART_COUNT:-0}.json"
TRANSITION_AUDIT="$CONTROL_DIR/calibration_transition.json"
EXPECTED_AUDIT_CONFIG_SHA256="99984578cf7ea8ed537b589fa576c3b639fe3f8b19c4ea3831686ccfb93e36de"
EXPECTED_BANK_ID="a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
EXPECTED_BANK_JSON_SHA256="c75120cf3022d1b965963f303ae4264f2c0b1f411597f13c822d98d3ce499d30"
EXPECTED_BANK_READY_SHA256="ac4445de5a798b52ca4bf80f5227aec694929007fe77bf1a0bd57a3dc7687793"
EXPECTED_TRAIN_SHA256="d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852"
EXPECTED_STAGER_SHA256="8cb3b6fb765af73c8d0fa9db074c267fcdbd4f8943b602b8caef3421e2693323"

[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-0}" == "1" ]] || {
  echo "ERROR: Stage-C calibration requires one Slurm node" >&2; exit 1;
}
[[ "$SOURCE_REMOTE_BRANCH" == "$EXPECTED_SOURCE_REMOTE_BRANCH" \
   && -f "$CPU_GATE_READY" ]] || {
  echo "ERROR: exact source identity or completed CPU gate is absent" >&2; exit 1;
}
[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: confirmation holdout is already spent" >&2; exit 1;
}
for required in "$PYTHON_BIN" "$AUDIT_CFG" "$LAUNCHER" "$STAGER" \
  "$PREREQUISITE_HELPER" "$CALIBRATION_CONTROL_HELPER" "$TRANSITION_HELPER" \
  "$STAGE_C_CONFIG" "$STAGE_C_OFF_CONFIG" \
  "$RECOVERY_POLICY" "$RECOVERY_MANIFEST" \
  "$SOURCE_DECISION" \
  "$SOURCE_BANK/bank.json" "$SOURCE_BANK/build_summary.json" \
  "$SOURCE_BANK/READY" "$SOURCE_BANK/neighbors_train.npz"; do
  [[ -e "$required" ]] || { echo "ERROR: missing calibration prerequisite: $required" >&2; exit 1; }
done
[[ "$(sha256sum -- "$AUDIT_CFG" | awk '{print $1}')" == "$EXPECTED_AUDIT_CONFIG_SHA256" \
   && "$(sha256sum -- "$SOURCE_BANK/bank.json" | awk '{print $1}')" == "$EXPECTED_BANK_JSON_SHA256" \
   && "$(sha256sum -- "$SOURCE_BANK/READY" | awk '{print $1}')" == "$EXPECTED_BANK_READY_SHA256" \
   && "$(sha256sum -- "$SOURCE_BANK/neighbors_train.npz" | awk '{print $1}')" == "$EXPECTED_TRAIN_SHA256" \
   && "$(sha256sum -- "$STAGER" | awk '{print $1}')" == "$EXPECTED_STAGER_SHA256" ]] || {
  echo "ERROR: calibration config, bank, train table, or stager identity changed" >&2; exit 1;
}

cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git \
      || "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" \
      || "$(git rev-parse HEAD)" != "${SOURCE_GIT_HEAD,,}" \
      || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: calibration requires the exact clean standalone source" >&2; exit 1;
fi
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
[[ -n "$REMOTE_HEAD" && "$REMOTE_HEAD" == "${SOURCE_GIT_HEAD,,}" ]] || {
  echo "ERROR: calibration source is not the exact pushed branch head" >&2; exit 1;
}
"$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-cpu \
  --cpu_gate "$CPU_GATE_READY" \
  --source_git_head "$SOURCE_GIT_HEAD" \
  --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
  --source_remote_head "$REMOTE_HEAD"

export PATH="$PYTHON_ENV/bin:$PATH" PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=""
unset WANDB_API_KEY
mkdir -p -- "$CONTROL_DIR/execution_attempts"

"$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-recovery \
  --policy_path "$RECOVERY_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" \
  --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT"
"$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-protocol-v2-configs \
  --policy_path "$RECOVERY_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" \
  --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT" \
  --memory_config "$STAGE_C_CONFIG" --matched_off_config "$STAGE_C_OFF_CONFIG"

validate_completed_calibration() {
  "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-foundation \
    --recovery_policy "$RECOVERY_POLICY" \
    --recovery_manifest "$RECOVERY_MANIFEST" \
    --recovery_evidence_root "$RECOVERY_EVIDENCE_ROOT" \
    --cpu_gate "$CPU_GATE_READY" --calibration_completion "$COMPLETION" \
    --calibration_dir "$CALIBRATION_OUT" --calibration_launcher "$LAUNCHER" \
    --stage_c_config "$STAGE_C_CONFIG" \
    --source_terminal_decision "$SOURCE_DECISION" \
    --source_root "$PROJECT_DIR" --source_git_head "$SOURCE_GIT_HEAD" \
    --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
    --source_remote_head "$REMOTE_HEAD"
}

if [[ -f "$COMPLETION" && -d "$LEASE" ]]; then
  "$PYTHON_BIN" -m "$CALIBRATION_CONTROL_MODULE" reconcile-completed \
    --lease "$LEASE" --completion "$COMPLETION" \
    --current_job_id "$SLURM_JOB_ID" \
    --current_restart "${SLURM_RESTART_COUNT:-0}"
  validate_completed_calibration
  echo "STAGE_C_CALIBRATION_RECOVERED=$COMPLETION"
  exit 0
elif [[ -f "$COMPLETION" && ! -e "$LEASE" ]]; then
  validate_completed_calibration
  echo "STAGE_C_CALIBRATION_ALREADY_COMPLETE=$COMPLETION"
  exit 0
fi

# This protocol generation has no calibration retry/resume path.  A same-job
# scheduler requeue is allowed only while its authenticated active claim has
# produced no artifact or transition.  A failed/history claim, an orphaned
# attestation, or any partial calibration output is terminal and preserved.
if [[ -e "$CALIBRATION_OUT" || -L "$CALIBRATION_OUT" \
      || -e "$TRANSITION_AUDIT" || -L "$TRANSITION_AUDIT" ]]; then
  echo "ERROR: protocol-v2/run-r3 forbids calibration resume after partial output" >&2
  exit 1
fi
if [[ -d "${LEASE}.history" \
      && -n "$(find "${LEASE}.history" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: protocol-v2/run-r3 calibration lease history forbids retry" >&2
  exit 1
fi
if [[ -d "$LEASE" && ! -L "$LEASE" ]]; then
  ACTIVE_OWNER_JOB="$($PYTHON_BIN - "$LEASE/owner.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("slurm_job_id", ""))
PY
)"
  [[ "$ACTIVE_OWNER_JOB" == "$SLURM_JOB_ID" && ! -e "$LEASE/TERMINAL.json" ]] || {
    echo "ERROR: protocol-v2/run-r3 forbids replacing a calibration owner" >&2
    exit 1
  }
elif [[ -e "$LEASE" || -L "$LEASE" ]]; then
  echo "ERROR: protocol-v2/run-r3 calibration lease is malformed" >&2
  exit 1
elif [[ -d "$CONTROL_DIR/lease_attestations" \
        && -n "$(find "$CONTROL_DIR/lease_attestations" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: protocol-v2/run-r3 orphaned calibration claim forbids retry" >&2
  exit 1
fi

"$PYTHON_BIN" -m "$CALIBRATION_CONTROL_MODULE" acquire \
  --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
  --slurm_job_id "$SLURM_JOB_ID" \
  --slurm_restart_count "${SLURM_RESTART_COUNT:-0}" \
  --source_git_head "$SOURCE_GIT_HEAD" \
  --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
  --source_remote_head "$REMOTE_HEAD" \
  --source_file_profile stage_c_generator_adaptation_protocol_v2_run_r3 \
  --launcher "$LAUNCHER" --cpu_gate "$CPU_GATE_READY" \
  --audit_config "$AUDIT_CFG" --bank_manifest "$SOURCE_BANK/bank.json" \
  --bank_ready "$SOURCE_BANK/READY" \
  --train_neighbors "$SOURCE_BANK/neighbors_train.npz"
LEASE_ACTIVE=1

STAGED_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
cleanup() {
  local exit_code=$?
  trap - EXIT
  "$STAGER" cleanup "$STAGED_BANK" || true
  if [[ "${LEASE_ACTIVE:-0}" == "1" && -d "$LEASE" ]]; then
    if [[ -f "$COMPLETION" ]]; then
      "$PYTHON_BIN" -m "$CALIBRATION_CONTROL_MODULE" release \
        --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
        --completion "$COMPLETION" || exit_code=$?
    else
      "$PYTHON_BIN" -m "$CALIBRATION_CONTROL_MODULE" fail \
        --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
        --exit_code "$exit_code" || exit_code=$?
    fi
  fi
  exit "$exit_code"
}
trap cleanup EXIT
if [[ -f "$COMPLETION" ]]; then
  "$PYTHON_BIN" -m "$CALIBRATION_CONTROL_MODULE" release \
    --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
    --completion "$COMPLETION"
  LEASE_ACTIVE=0
  validate_completed_calibration
  echo "STAGE_C_CALIBRATION_RECOVERED=$COMPLETION"
  exit 0
fi
if [[ ! -e "$COMPLETION" ]]; then
  if [[ ! -e "$CALIBRATION_OUT" ]]; then
    "$STAGER" stage "$STAGED_BANK" "$SOURCE_BANK" "$PROJECT_DIR" \
      "$PYTHON_BIN" "$AUDIT_CFG" "train"
    [[ -f "$STAGED_BANK/neighbors_train.npz" \
       && ! -e "$STAGED_BANK/neighbors_val.npz" \
       && ! -e "$STAGED_BANK/neighbors_test.npz" \
       && "$(sha256sum -- "$STAGED_BANK/neighbors_train.npz" | awk '{print $1}')" == "$EXPECTED_TRAIN_SHA256" ]] || {
      echo "ERROR: calibration staging is not exactly the pinned train table" >&2; exit 1;
    }

    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.calibrate_sentence_memory_relevance \
      --bank_dir "$STAGED_BANK" --train_neighbors "$STAGED_BANK/neighbors_train.npz" \
      --out_dir "$CALIBRATION_OUT" --source_root "$PROJECT_DIR" \
      --source_git_head "$SOURCE_GIT_HEAD" \
      --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" --source_remote_head "$REMOTE_HEAD" \
      --source_file_profile stage_c_generator_adaptation_protocol_v2_run_r3 \
      --launcher "$LAUNCHER" --seed 1234 --duration_weight 0.05 \
      --minimum_auroc 0.75 --minimum_probability_gap 0.20 \
      --expected_bank_id "$EXPECTED_BANK_ID" \
      --expected_train_neighbor_sha256 "$EXPECTED_TRAIN_SHA256"
  fi
  if [[ -f "$TRANSITION_AUDIT" ]]; then
    "$PYTHON_BIN" -m "$TRANSITION_MODULE" verify \
      --source_terminal_decision "$SOURCE_DECISION" \
      --stage_c_config "$STAGE_C_CONFIG" --audit "$TRANSITION_AUDIT"
  else
    "$PYTHON_BIN" -m "$TRANSITION_MODULE" create \
      --source_terminal_decision "$SOURCE_DECISION" \
      --stage_c_config "$STAGE_C_CONFIG" --out_file "$TRANSITION_AUDIT"
  fi
fi

"$PYTHON_BIN" - "$CALIBRATION_OUT" "$PROJECT_DIR" "$SOURCE_GIT_HEAD" \
  "origin/$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" "$COMPLETION" \
  "$CPU_GATE_READY" "$LAUNCHER" "$TRANSITION_AUDIT" \
  "$LEASE_ATTESTATION" <<'PY'
import hashlib, json, sys
from pathlib import Path
from NIAF.continuous_trajectory_field.relevance_calibration import (
    validate_relevance_calibration_artifact,
    validate_relevance_calibration_source,
)
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    publish_bytes_no_replace,
)
artifact = validate_relevance_calibration_artifact(sys.argv[1])
validate_relevance_calibration_source(
    artifact, source_root=sys.argv[2], expected_git_head=sys.argv[3],
    expected_remote_ref=sys.argv[4], expected_remote_head=sys.argv[5],
    expected_source_file_profile="stage_c_generator_adaptation_protocol_v2_run_r3",
)
root = Path(sys.argv[1])
transition_path = Path(sys.argv[9])
attestation_path = Path(sys.argv[10])
transition = json.loads(transition_path.read_text(encoding="utf-8"))
attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
if (
    transition.get("schema_name")
    != "signtrajfield_stage_c_calibration_transition_audit"
    or transition.get("semantic_equivalence", {}).get(
        "all_non_provenance_fields_exact"
    ) is not True
    or transition.get("train_only") is not True
    or transition.get("development_only") is not True
):
    raise SystemExit("ERROR: calibration transition audit is incomplete")
payload = {
    "schema_name": "signtrajfield_stage_c_calibration_completion",
    "schema_version": 1,
    "source_git_head": sys.argv[3].lower(),
    "source_remote_ref": sys.argv[4],
    "source_remote_head": sys.argv[5].lower(),
    "artifact_path": str(root.resolve()),
    "artifact_identity": artifact["identity"],
    "calibration_sha256": hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest(),
    "map_sha256": hashlib.sha256((root / "calibration_map.npz").read_bytes()).hexdigest(),
    "ready_sha256": hashlib.sha256((root / "READY").read_bytes()).hexdigest(),
    "cpu_gate_sha256": hashlib.sha256(Path(sys.argv[7]).read_bytes()).hexdigest(),
    "launcher_sha256": hashlib.sha256(Path(sys.argv[8]).read_bytes()).hexdigest(),
    "calibration_transition_path": str(transition_path.resolve()),
    "calibration_transition_sha256": hashlib.sha256(
        transition_path.read_bytes()
    ).hexdigest(),
    "calibration_transition_identity": transition["audit_identity"],
    "active_lease_claim_identity": attestation["claim"]["claim_identity"],
    "lease_attestation_path": str(attestation_path.resolve()),
    "lease_attestation_sha256": hashlib.sha256(
        attestation_path.read_bytes()
    ).hexdigest(),
    "lease_attestation_identity": attestation["attestation_identity"],
    "train_only": True,
    "development_only": True,
    "confirmation_manifest_opened": False,
    "test_data_accessed": False,
}
path = Path(sys.argv[6])
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
publish_bytes_no_replace(path, encoded)
PY
"$PYTHON_BIN" -m "$CALIBRATION_CONTROL_MODULE" release \
  --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
  --completion "$COMPLETION"
LEASE_ACTIVE=0
validate_completed_calibration
echo "STAGE_C_CALIBRATION_COMPLETE=$COMPLETION"
