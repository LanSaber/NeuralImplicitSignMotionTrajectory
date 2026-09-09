#!/bin/bash

# Fail-closed driver for the matched, exploratory Stage-C generator pilot.
# Both arms run sequentially on the same two-node pair and independently load
# the exact Stage-B best-infeasible checkpoint.  This workflow is development
# only and cannot authorize confirmation/test access or promotion.
set -euo pipefail
trap 'echo "ERROR: Stage-C paired pilot failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the exact pushed source commit}"
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:?Set SOURCE_REMOTE_BRANCH to the exact pushed branch}"
PAIR_CONSTRAINT="${STAGE_C_PAIR_CONSTRAINT:?Set STAGE_C_PAIR_CONSTRAINT to pair01..pair15}"
EXECUTION_MODE="${STAGE_C_EXECUTION_MODE:-pilot}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"

MEMORY_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_pilot_run_r2"
OFF_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_pilot_run_r2"
MEMORY_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${MEMORY_EXPERIMENT}.yaml"
OFF_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${OFF_EXPERIMENT}.yaml"
TRAINER_MODULE="NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field"
ROCE_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_paired_roce"
PREREQUISITE_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_pilot_prerequisites"
SMOKE_AUDIT_MODULE="NIAF.continuous_trajectory_field.scripts.audit_stage_c_smoke_checkpoint"
EXECUTION_CONTROL_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_execution_control"
DECISION_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_pilot_decision"
ROCE_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_paired_roce.py"
PREREQUISITE_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_pilot_prerequisites.py"
SMOKE_AUDIT_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/audit_stage_c_smoke_checkpoint.py"
EXECUTION_CONTROL_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_execution_control.py"
DECISION_HELPER="$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_pilot_decision.py"
DECISION_POLICY="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
RECOVERY_MANIFEST="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_retry2_recovery_evidence_v1.json"
RECOVERY_EVIDENCE_ROOT="/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
CALIBRATION_LAUNCHER="$PROJECT_DIR/scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh"

STAGE_B_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1"
STAGE_B_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${STAGE_B_EXPERIMENT}.yaml"
STAGE_B_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$STAGE_B_EXPERIMENT/checkpoints/best_infeasible.pt"
STAGE_B_DECISION="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$STAGE_B_EXPERIMENT/evaluation/ordered_development_decision/decision.json"
V2_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
CALIBRATION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_retry2_v1"
PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
DEV_MANIFEST="$PARTITION_DIR/manifest_development.jsonl"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"
PREREQUISITE_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_prerequisites/source_${SOURCE_GIT_HEAD,,}"
CPU_GATE_READY="$PREREQUISITE_ROOT/cpu_gate/READY"
CALIBRATION_COMPLETION="$PREREQUISITE_ROOT/calibration/COMPLETE.json"
SMOKE_READY="$PREREQUISITE_ROOT/smoke/PUBLICATION/READY"
PILOT_READY="$PREREQUISITE_ROOT/pilot/PUBLICATION/READY"

EXPECTED_STAGE_B_CHECKPOINT_SHA256="b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202"
EXPECTED_STAGE_B_CONFIG_SHA256="7741da46d37a4b77f481663f25fa580281f6d29de6c2616baebcc30dac59b85e"
EXPECTED_STAGE_B_DECISION_SHA256="8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5"
EXPECTED_STAGE_B_DECISION_IDENTITY="7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69"
EXPECTED_STAGE_B_ARCHITECTURE_IDENTITY="bc69fd35ac58e10bc894460c35175f13356b79416df40e8c57156b1929614236"
EXPECTED_V2_CHECKPOINT_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
EXPECTED_V2_CONFIG_SHA256="87b39c456c93354a34eafef2dde1f03b753ad0321ec3cf24ae20a57586d4af82"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"
EXPECTED_DEV_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
EXPECTED_PARTITION_IDENTITY="2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
EXPECTED_BANK_ID="a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
EXPECTED_BANK_JSON_SHA256="c75120cf3022d1b965963f303ae4264f2c0b1f411597f13c822d98d3ce499d30"
EXPECTED_BANK_BUILD_SHA256="6e47e5a685faf739402706d9b2f7d650f24b3450e4708423276267fd2db90d77"
EXPECTED_BANK_READY_SHA256="ac4445de5a798b52ca4bf80f5227aec694929007fe77bf1a0bd57a3dc7687793"
EXPECTED_STAGER_SHA256="8cb3b6fb765af73c8d0fa9db074c267fcdbd4f8943b602b8caef3421e2693323"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "2" || "${SLURM_NTASKS:-0}" != "2" ]]; then
  echo "ERROR: Stage C requires exactly two Slurm nodes and two ranks" >&2
  exit 1
fi
[[ "$PAIR_CONSTRAINT" =~ ^pair(0[1-9]|1[0-5])$ ]] || {
  echo "ERROR: STAGE_C_PAIR_CONSTRAINT must be pair01..pair15" >&2
  exit 1
}
case "$EXECUTION_MODE" in
  smoke|pilot) ;;
  *) echo "ERROR: STAGE_C_EXECUTION_MODE must be smoke or pilot" >&2; exit 1 ;;
esac
export STAGE_C_EXECUTION_MODE="$EXECUTION_MODE"
if [[ "$EXECUTION_MODE" == "smoke" ]]; then
  MODE_READY="$SMOKE_READY"
else
  MODE_READY="$PILOT_READY"
fi
MODE_CONTROL_DIR="$PREREQUISITE_ROOT/$EXECUTION_MODE"
MODE_PUBLICATION="$MODE_CONTROL_DIR/PUBLICATION"
MODE_COMPLETE="$MODE_PUBLICATION/COMPLETE.json"
LEASE="$MODE_CONTROL_DIR/active_execution_lease"
LEASE_ATTESTATION="$MODE_CONTROL_DIR/lease_attestations/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
[[ "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ ]] || {
  echo "ERROR: SOURCE_GIT_HEAD must be a full Git SHA1" >&2
  exit 1
}

# Nothing inherited may turn this into a resume, alternate-data, online-log,
# or trainer-override run.  The two arm configs are the complete authority.
for forbidden in CFG OUT_DIR RESUME WARM_START STAGE_C_WARM_START BASE_CHECKPOINT \
  PHASE_B_GATE_REPORT LIMIT_TRAIN LIMIT_VAL MAX_TRAIN_BATCHES MAX_VAL_BATCHES \
  BATCH_SIZE EPOCHS SENTENCE_MEMORY_K WANDB_API_KEY; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "ERROR: Stage-C paired pilot forbids inherited $forbidden" >&2
    exit 1
  fi
done

for required in "$PYTHON_BIN" "$MEMORY_CFG" "$OFF_CFG" "$STAGE_B_CFG" \
  "$STAGE_B_CHECKPOINT" "$STAGE_B_DECISION" "$V2_CFG" "$V2_CHECKPOINT" \
  "$CALIBRATION_DIR/READY" "$CALIBRATION_COMPLETION" "$CPU_GATE_READY" \
  "$PARTITION_DIR/READY" "$DEV_MANIFEST" \
  "$SOURCE_BANK/bank.json" "$SOURCE_BANK/build_summary.json" \
  "$SOURCE_BANK/READY" "$STAGER" "$ROCE_HELPER" "$PREREQUISITE_HELPER" \
  "$SMOKE_AUDIT_HELPER" "$EXECUTION_CONTROL_HELPER" \
  "$DECISION_HELPER" "$DECISION_POLICY" "$RECOVERY_MANIFEST" \
  "$CALIBRATION_LAUNCHER"; do
  [[ -e "$required" ]] || { echo "ERROR: missing Stage-C prerequisite: $required" >&2; exit 1; }
done
[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: global confirmation-spend marker exists; Stage C is development-only" >&2
  exit 1
}
if [[ "$EXECUTION_MODE" == "pilot" && ! -f "$SMOKE_READY" ]]; then
  echo "ERROR: full pilot requires the completed paired one-update smoke gate" >&2
  exit 1
fi

cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git ]]; then
  echo "ERROR: PROJECT_DIR must be a shared standalone clone" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: source must resolve below durable /media/cvpr storage" >&2; exit 1 ;;
esac
if [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: Stage C forbids external worktree metadata" >&2
  exit 1
fi
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "${ACTUAL_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: source must be clean and equal the exact SOURCE_GIT_HEAD" >&2
  exit 1
fi
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$REMOTE_HEAD" || "${REMOTE_HEAD,,}" != "${ACTUAL_HEAD,,}" ]]; then
  echo "ERROR: source is not the exact pushed origin branch head" >&2
  exit 1
fi

require_sha256() {
  local path="$1" expected="$2" label="$3"
  local observed
  observed="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "ERROR: $label identity changed: $observed != $expected" >&2
    return 1
  }
}
require_sha256 "$STAGE_B_CHECKPOINT" "$EXPECTED_STAGE_B_CHECKPOINT_SHA256" "Stage-B checkpoint"
require_sha256 "$STAGE_B_CFG" "$EXPECTED_STAGE_B_CONFIG_SHA256" "Stage-B config"
require_sha256 "$STAGE_B_DECISION" "$EXPECTED_STAGE_B_DECISION_SHA256" "Stage-B terminal decision"
require_sha256 "$V2_CHECKPOINT" "$EXPECTED_V2_CHECKPOINT_SHA256" "frozen-v2 checkpoint"
require_sha256 "$V2_CFG" "$EXPECTED_V2_CONFIG_SHA256" "frozen-v2 config"
require_sha256 "$PARTITION_DIR/READY" "$EXPECTED_PARTITION_READY_SHA256" "development partition seal"
require_sha256 "$DEV_MANIFEST" "$EXPECTED_DEV_SHA256" "development manifest"
require_sha256 "$SOURCE_BANK/bank.json" "$EXPECTED_BANK_JSON_SHA256" "sentence bank manifest"
require_sha256 "$SOURCE_BANK/build_summary.json" "$EXPECTED_BANK_BUILD_SHA256" "sentence bank summary"
require_sha256 "$SOURCE_BANK/READY" "$EXPECTED_BANK_READY_SHA256" "sentence bank seal"
require_sha256 "$STAGER" "$EXPECTED_STAGER_SHA256" "strict train/val stager"
[[ "$(awk 'NF {n += 1} END {print n + 0}' "$DEV_MANIFEST")" == "347" ]] || {
  echo "ERROR: development manifest is not the exact 347-row artifact" >&2
  exit 1
}

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

"$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-recovery \
  --policy_path "$DECISION_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" \
  --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT"

# Validate both complete configs together.  Apart from arm/name/output and the
# training-time memory switch, the resolved arms must be exactly identical.
"$PYTHON_BIN" - "$MEMORY_CFG" "$OFF_CFG" "$PROJECT_DIR" \
  "$EXPECTED_STAGE_B_CHECKPOINT_SHA256" "$EXPECTED_STAGE_B_DECISION_SHA256" \
  "$EXPECTED_STAGE_B_DECISION_IDENTITY" "$EXPECTED_STAGE_B_CONFIG_SHA256" \
  "$EXPECTED_STAGE_B_ARCHITECTURE_IDENTITY" "$EXPECTED_V2_CHECKPOINT_SHA256" <<'PY'
import copy
import hashlib
import json
import sys
from pathlib import Path

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.scripts.stage_c_pilot_prerequisites import (
    validate_source_terminal_decision,
)

memory_path, off_path, project = map(Path, sys.argv[1:4])
expected = {
    "checkpoint": sys.argv[4],
    "decision": sys.argv[5],
    "decision_identity": sys.argv[6],
    "config": sys.argv[7],
    "architecture": sys.argv[8],
    "v2": sys.argv[9],
}
memory = load_config(memory_path)
off = load_config(off_path)
for cfg, arm, mode, probability in (
    (memory, "memory", "dropout", 0.25),
    (off, "matched_off", "off", 1.0),
):
    stage = cfg["sentence_memory_safety"]["stage_c"]
    assert stage == {
        "schema_name": "signtrajfield_centered_stage_c_generator_adaptation",
        "schema_version": 1,
        "enabled": True,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "arm": arm,
        "huber_beta": 0.10,
        "text_only_max_relative_degradation": 0.005,
        "source_checkpoint": {
            "path": "experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/checkpoints/best_infeasible.pt",
            "sha256": expected["checkpoint"],
            "selection_status": "best_infeasible",
            "epoch": 5,
            "global_step": 360,
        },
        "source_terminal_decision": {
            "path": "experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/evaluation/ordered_development_decision/decision.json",
            "sha256": expected["decision"],
            "decision_identity": expected["decision_identity"],
            "status": "valid_infeasible",
            "authorized_purpose": None,
        },
        "frozen_v2_teacher": {
            "path": "experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt",
            "sha256": expected["v2"],
        },
        "source_stage_b": {
            "config_path": "NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1.yaml",
            "config_sha256": expected["config"],
            "architecture_identity": expected["architecture"],
        },
        "active_stage_c": {
            "calibration_artifact_dir": "experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_retry2_v1",
            "calibration_schema_name": "signtrajfield_sentence_memory_relevance_calibration",
            "calibration_schema_version": 1,
        },
    }
    assert cfg["sentence_memory"]["relevance_calibration"]["artifact_dir"] == stage["active_stage_c"]["calibration_artifact_dir"]
    assert cfg["sentence_memory"]["relevance_calibration"]["artifact_identity"] is None
    assert cfg["conditioning"]["word_prior_train_mode"] == "off"
    assert cfg["conditioning"]["sentence_memory_train_mode"] == mode
    assert cfg["conditioning"]["sentence_memory_dropout_probability"] == probability
    assert cfg["sentence_memory_safety"]["phase_b"]["enabled"] is False
    assert cfg["sentence_memory_safety"]["paired_corruption"]["enabled"] is False
    assert cfg["train"]["base_checkpoint"] is None
    assert cfg["train"]["freeze_base"] is True
    assert cfg["train"]["freeze_sentence_memory"] is True
    assert cfg["train"]["unfreeze_base_prefixes"] == [
        "hypernetwork.global_context.",
        "hypernetwork.coarse_head.",
        "hypernetwork.residual_head.",
        "hypernetwork.gate_head.",
        "hypernetwork.local_context.",
        "hypernetwork.local_head.",
        "hypernetwork.local_gate_head.",
    ]
    assert cfg["train"]["epochs"] == 1
    assert cfg["train"]["batch_size"] == 64
    assert cfg["train"]["accumulation_steps"] == 2
    assert cfg["train"].get("max_train_batches", 0) == 0
    assert cfg["eval"].get("max_batches", 0) == 0
    assert cfg["data"].get("limit_train", 0) == 0
    assert cfg["data"].get("limit_val", 0) == 0
    assert cfg["train"]["length_bucketed_batches"] is True
    assert cfg["train"]["drop_last"] is False
    assert cfg["selection"]["require_feasible"] is False
    assert cfg["eval"]["sentence_memory_modes"] == [
        "off", "on", "motion_shuffled_n0", "motion_shuffled_n1",
        "motion_shuffled_n2", "cross_query_motion", "full_replacement",
        "joint_tuple_permuted", "uniform_final_mass", "analytic_prior",
        "association_disabled",
    ]

normalized = []
for cfg in (memory, off):
    value = copy.deepcopy(cfg)
    value.pop("experiment_name")
    value["output"].pop("out_dir")
    value["conditioning"].pop("sentence_memory_train_mode")
    value["conditioning"].pop("sentence_memory_dropout_probability")
    value["sentence_memory_safety"]["stage_c"].pop("arm")
    normalized.append(value)
assert normalized[0] == normalized[1], "Stage-C arms differ outside the approved training-memory switch"

decision_path = project / memory["sentence_memory_safety"]["stage_c"]["source_terminal_decision"]["path"]
validate_source_terminal_decision(
    decision_path=decision_path,
    expected_sha256=expected["decision"],
    expected_identity=expected["decision_identity"],
)
print(json.dumps({
    "memory_config_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
    "matched_off_config_sha256": hashlib.sha256(off_path.read_bytes()).hexdigest(),
    "effective_global_batch": 64 * 2 * 2,
}, sort_keys=True))
PY

# Reopen the complete CPU/calibration evidence chain and require exact schemas,
# immutable hashes, no-access flags, and binding to this clean pushed source.
"$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-foundation \
  --recovery_policy "$DECISION_POLICY" \
  --recovery_manifest "$RECOVERY_MANIFEST" \
  --recovery_evidence_root "$RECOVERY_EVIDENCE_ROOT" \
  --cpu_gate "$CPU_GATE_READY" \
  --calibration_completion "$CALIBRATION_COMPLETION" \
  --calibration_dir "$CALIBRATION_DIR" \
  --calibration_launcher "$CALIBRATION_LAUNCHER" \
  --stage_c_config "$MEMORY_CFG" \
  --source_terminal_decision "$STAGE_B_DECISION" \
  --source_root "$PROJECT_DIR" \
  --source_git_head "$SOURCE_GIT_HEAD" \
  --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
  --source_remote_head "$REMOTE_HEAD"
if [[ "$EXECUTION_MODE" == "pilot" ]]; then
  "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-smoke \
    --smoke_ready "$SMOKE_READY" \
    --smoke_root "$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_smoke" \
    --source_git_head "$SOURCE_GIT_HEAD" \
    --pair_constraint "$PAIR_CONSTRAINT"
fi
export STAGE_C_SMOKE_READY="$SMOKE_READY"

# Require that Slurm actually scheduled the exact requested feature and that
# both allocated nodes advertise it.  Merely landing on two adjacent nodes is
# not accepted as evidence of a cabled pair.
JOB_FEATURES="$(scontrol show job -o "$SLURM_JOB_ID" | awk '{for(i=1;i<=NF;i++) if($i ~ /^Features=/){sub(/^Features=/,"",$i); print $i; exit}}')"
[[ "$JOB_FEATURES" == "$PAIR_CONSTRAINT" ]] || {
  echo "ERROR: Slurm job did not request exact --constraint=$PAIR_CONSTRAINT (Features=$JOB_FEATURES)" >&2
  exit 1
}
mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_NODELIST")
[[ "${#ALLOCATED_NODES[@]}" == "2" && "${ALLOCATED_NODES[0]}" != "${ALLOCATED_NODES[1]}" ]] || {
  echo "ERROR: allocation does not contain exactly two distinct nodes" >&2
  exit 1
}
for node in "${ALLOCATED_NODES[@]}"; do
  available="$(scontrol show node -o "$node" | awk '{for(i=1;i<=NF;i++) if($i ~ /^AvailableFeatures=/){sub(/^AvailableFeatures=/,"",$i); print $i; exit}}')"
  [[ ",$available," == *",$PAIR_CONSTRAINT,"* ]] || {
    echo "ERROR: allocated node $node does not advertise $PAIR_CONSTRAINT" >&2
    exit 1
  }
done

if [[ "$EXECUTION_MODE" == "smoke" ]]; then
  PILOT_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_smoke"
else
  PILOT_ROOT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_pilot"
fi
ATTEMPTS_ROOT="$PILOT_ROOT/sources/source_${SOURCE_GIT_HEAD,,}/attempts"

# A published mode is globally singleton across all pairNN choices.  If its
# owner crashed before releasing, only scheduler-terminal reconciliation may
# archive that exact claim; a new scientific attempt is never created.
if [[ -e "$MODE_PUBLICATION" ]]; then
  [[ -d "$MODE_PUBLICATION" && ! -L "$MODE_PUBLICATION" ]] || {
    echo "ERROR: Stage-C mode publication is not a regular directory" >&2
    exit 1
  }
  if [[ ! -d "$LEASE" ]]; then
    [[ ! -e "$LEASE" ]] || {
      echo "ERROR: finalized Stage-C lease path is malformed" >&2
      exit 1
    }
    "$PYTHON_BIN" -m "$EXECUTION_CONTROL_MODULE" validate-finalized \
      --lease "$LEASE" --completion "$MODE_COMPLETE" --ready "$MODE_READY" \
      --expected_source_git_head "$SOURCE_GIT_HEAD" \
      --expected_pair_constraint "$PAIR_CONSTRAINT"
    if [[ "$EXECUTION_MODE" == "smoke" ]]; then
      "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-smoke \
        --smoke_ready "$MODE_READY" --smoke_root "$PILOT_ROOT" \
        --source_git_head "$SOURCE_GIT_HEAD" \
        --pair_constraint "$PAIR_CONSTRAINT"
    else
      mapfile -t FINALIZED_PATHS < <(
        "$PYTHON_BIN" - "$MODE_COMPLETE" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["execution_complete_path"])
print(value["decision_path"])
print(value["lease_attestation_path"])
PY
      )
      [[ "${#FINALIZED_PATHS[@]}" == "3" ]] || {
        echo "ERROR: finalized pilot publication paths are incomplete" >&2
        exit 1
      }
      FINAL_EXECUTION="${FINALIZED_PATHS[0]}"
      FINAL_DECISION="${FINALIZED_PATHS[1]}"
      FINAL_ATTESTATION="${FINALIZED_PATHS[2]}"
      FINAL_ATTEMPT="$(dirname "$FINAL_EXECUTION")"
      "$PYTHON_BIN" -m "$DECISION_MODULE" verify \
        --mode pilot --execution_complete "$FINAL_EXECUTION" \
        --source_checkpoint "$STAGE_B_CHECKPOINT" \
        --memory_checkpoint "$FINAL_ATTEMPT/memory/checkpoints/last.pt" \
        --matched_off_checkpoint "$FINAL_ATTEMPT/matched_off/checkpoints/last.pt" \
        --policy "$DECISION_POLICY" \
        --active_lease_attestation "$FINAL_ATTESTATION" \
        --expected_source_git_head "$SOURCE_GIT_HEAD" \
        --expected_source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
        --expected_source_remote_head "$REMOTE_HEAD" \
        --expected_pair_constraint "$PAIR_CONSTRAINT" \
        --decision "$FINAL_DECISION"
    fi
    FINALIZED_STATUS="$($PYTHON_BIN - "$MODE_READY" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision_status"])
PY
)"
    echo "STAGE_C_VALIDATED_FINALIZED_${EXECUTION_MODE^^}=$MODE_PUBLICATION"
    if [[ "$EXECUTION_MODE:$FINALIZED_STATUS" == "smoke:smoke_ready" \
          || "$EXECUTION_MODE:$FINALIZED_STATUS" == "pilot:pilot_complete_development_signal" ]]; then
      exit 0
    fi
    exit 2
  fi
  "$PYTHON_BIN" -m "$EXECUTION_CONTROL_MODULE" reconcile-published \
    --lease "$LEASE" --completion "$MODE_COMPLETE" --ready "$MODE_READY" \
    --current_slurm_job_id "$SLURM_JOB_ID" \
    --current_slurm_restart_count "${SLURM_RESTART_COUNT:-0}"
  RECOVERED_STATUS="$($PYTHON_BIN - "$MODE_READY" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision_status"])
PY
)"
  echo "STAGE_C_RECOVERED_IMMUTABLE_${EXECUTION_MODE^^}=$MODE_PUBLICATION"
  if [[ "$EXECUTION_MODE:$RECOVERED_STATUS" == "smoke:smoke_ready" \
        || "$EXECUTION_MODE:$RECOVERED_STATUS" == "pilot:pilot_complete_development_signal" ]]; then
    exit 0
  fi
  exit 2
fi

# Claim the singleton mode/pair execution before creating an attempt, touching
# network evidence, staging data, or entering either trainer.  A requeued job
# keeps the original authenticated claim only after Slurm proves a newer
# restart generation; another job can replace it only after terminal proof.
"$PYTHON_BIN" -m "$EXECUTION_CONTROL_MODULE" acquire \
  --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
  --slurm_job_id "$SLURM_JOB_ID" \
  --slurm_restart_count "${SLURM_RESTART_COUNT:-0}" \
  --mode "$EXECUTION_MODE" \
  --source_git_head "$SOURCE_GIT_HEAD" \
  --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
  --source_remote_head "$REMOTE_HEAD" \
  --pair_constraint "$PAIR_CONSTRAINT" \
  --memory_config "$MEMORY_CFG" --matched_off_config "$OFF_CFG" \
  --cpu_gate "$CPU_GATE_READY" \
  --calibration_completion "$CALIBRATION_COMPLETION"
LEASE_ACTIVE=1
LEASE_CLAIM_IDENTITY="$($PYTHON_BIN - "$LEASE_ATTESTATION" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["claim"]["claim_identity"])
PY
)"
export STAGE_C_LEASE_CLAIM_IDENTITY="$LEASE_CLAIM_IDENTITY"
export STAGE_C_LEASE_ATTESTATION="$LEASE_ATTESTATION"

LOCAL_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
cleanup() {
  local exit_code=$?
  trap - EXIT
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    bash "$STAGER" cleanup "$LOCAL_BANK" || true
  if [[ "${LEASE_ACTIVE:-0}" == "1" && -d "$LEASE" ]]; then
    "$PYTHON_BIN" -m "$EXECUTION_CONTROL_MODULE" fail \
      --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
      --exit_code "$exit_code" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup EXIT

if [[ -d "$MODE_PUBLICATION" ]]; then
  "$PYTHON_BIN" -m "$EXECUTION_CONTROL_MODULE" release \
    --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
    --completion "$MODE_COMPLETE" --ready "$MODE_READY"
  LEASE_ACTIVE=0
  RECOVERED_STATUS="$($PYTHON_BIN - "$MODE_READY" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision_status"])
PY
)"
  echo "STAGE_C_RECOVERED_IMMUTABLE_${EXECUTION_MODE^^}=$MODE_PUBLICATION"
  if [[ "$EXECUTION_MODE" == "smoke" ]]; then
    [[ "$RECOVERED_STATUS" == "smoke_ready" ]]
  else
    [[ "$RECOVERED_STATUS" == "pilot_complete_development_signal" ]]
  fi
  exit
fi

mkdir -p -- "$ATTEMPTS_ROOT"
mapfile -t COMPLETED_EXECUTIONS < <(
  find "$ATTEMPTS_ROOT" -mindepth 4 -maxdepth 4 \
    -type f -name COMPLETE.json -print | sort
)
[[ "${#COMPLETED_EXECUTIONS[@]}" -le 1 ]] || {
  echo "ERROR: multiple selectable execution completions exist for one logical attempt" >&2
  exit 1
}
mapfile -t PRIOR_SCIENTIFIC_FILES < <(
  find "$ATTEMPTS_ROOT" -mindepth 1 -type f \
    ! -name '.*.building.*' -print | sort
)
if [[ "${#COMPLETED_EXECUTIONS[@]}" == "1" ]]; then
  ATTEMPT="$(dirname "${COMPLETED_EXECUTIONS[0]}")"
  for prior_file in "${PRIOR_SCIENTIFIC_FILES[@]}"; do
    [[ "$prior_file" == "$ATTEMPT/"* ]] || {
      echo "ERROR: a partial result exists outside the unique completed logical execution" >&2
      exit 1
    }
  done
  RECOVER_EXECUTION=1
else
  [[ "${#PRIOR_SCIENTIFIC_FILES[@]}" == "0" ]] || {
    echo "ERROR: predeclared retry policy forbids retraining after partial scientific output" >&2
    exit 1
  }
  LOGICAL_ATTEMPT="$ATTEMPTS_ROOT/$SLURM_JOB_ID"
  mkdir -p -- "$LOGICAL_ATTEMPT/executions"
  ATTEMPT="$LOGICAL_ATTEMPT/executions/restart_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "$ATTEMPT" ]] || {
    echo "ERROR: Stage-C execution generation already exists without COMPLETE: $ATTEMPT" >&2
    exit 1
  }
  mkdir -- "$ATTEMPT"
  RECOVER_EXECUTION=0
fi
NETWORK_DIR="$ATTEMPT/network_preflight"
SOURCE_BINDING_MANIFEST="$NETWORK_DIR/SOURCE_BINDING.json"
SOURCE_BINDING_FILES=(
  "$MEMORY_CFG" "$OFF_CFG" "$STAGE_B_CFG" "$STAGE_B_CHECKPOINT"
  "$STAGE_B_DECISION" "$V2_CFG" "$V2_CHECKPOINT" "$CALIBRATION_DIR/READY"
  "$CALIBRATION_COMPLETION" "$CPU_GATE_READY" "$PARTITION_DIR/READY"
  "$DEV_MANIFEST" "$SOURCE_BANK/bank.json" "$SOURCE_BANK/build_summary.json"
  "$SOURCE_BANK/READY" "$STAGER" "$ROCE_HELPER" "$PREREQUISITE_HELPER"
  "$SMOKE_AUDIT_HELPER" "$EXECUTION_CONTROL_HELPER" "$DECISION_HELPER"
  "$DECISION_POLICY" "$RECOVERY_MANIFEST" "$CALIBRATION_LAUNCHER"
  "$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/stage_c_atomic.py"
  "$PROJECT_DIR/NIAF/continuous_trajectory_field/scripts/train_continuous_trajectory_field.py"
  "$PROJECT_DIR/scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_pilot.sh"
)

revalidate_source_binding() {
  [[ "$(git rev-parse HEAD)" == "${SOURCE_GIT_HEAD,,}" \
     && -z "$(git status --porcelain --untracked-files=all)" \
     && ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
    echo "ERROR: Stage-C source/scope changed during paired execution" >&2
    return 1
  }
  local current_remote
  current_remote="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
  [[ "$current_remote" == "${REMOTE_HEAD,,}" ]] || {
    echo "ERROR: Stage-C remote source head changed during paired execution" >&2
    return 1
  }
  "$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-recovery \
    --policy_path "$DECISION_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" \
    --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT" \
    >/dev/null
  "$PYTHON_BIN" - "$SOURCE_BINDING_MANIFEST" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_name") != "signtrajfield_stage_c_source_binding" or value.get("schema_version") != 1:
    raise SystemExit("ERROR: Stage-C source binding manifest schema changed")
if len(value.get("files", {})) != 27:
    raise SystemExit("ERROR: Stage-C source binding file set is not exact")
for path_text, expected in value.get("files", {}).items():
    path = Path(path_text)
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"ERROR: bound Stage-C source file is not regular: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise SystemExit(f"ERROR: bound Stage-C source file changed: {path}")
PY
}

if [[ "$RECOVER_EXECUTION" == "0" ]]; then
mkdir -- "$NETWORK_DIR"
"$PYTHON_BIN" - "$SOURCE_BINDING_MANIFEST" "$SOURCE_GIT_HEAD" \
  "origin/$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" "${SOURCE_BINDING_FILES[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import publish_bytes_no_replace

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

path = Path(sys.argv[1])
files = [Path(value).resolve() for value in sys.argv[5:]]
if len(files) != 27 or len(files) != len(set(files)) or any(not value.is_file() or value.is_symlink() for value in files):
    raise SystemExit("ERROR: Stage-C source binding inputs are not exact regular files")
payload = {
    "schema_name": "signtrajfield_stage_c_source_binding",
    "schema_version": 1,
    "source_git_head": sys.argv[2].lower(),
    "source_remote_ref": sys.argv[3],
    "source_remote_head": sys.argv[4].lower(),
    "files": {
        str(value): sha256_file(value)
        for value in files
    },
    "development_only": True,
    "non_authorizing": True,
}
publish_bytes_no_replace(
    path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
)
PY
revalidate_source_binding

# Each node must expose both active ConnectX-7 f1 rails, a RoCE-v2 IPv4 GID at
# index 5, 200-Gb/s carrier, jumbo MTU, and the pair-specific two /30 networks.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -m "$ROCE_MODULE" audit-node --out_dir "$NETWORK_DIR"
"$PYTHON_BIN" -m "$ROCE_MODULE" validate-pair --manifest_dir "$NETWORK_DIR" \
  --pair "$PAIR_CONSTRAINT" --nodes "${ALLOCATED_NODES[@]}"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -m "$ROCE_MODULE" verify-peer --manifest_dir "$NETWORK_DIR"
[[ "$(find "$NETWORK_DIR" -maxdepth 1 -type f -name '*.connectivity.json' | wc -l)" == "2" ]] || {
  echo "ERROR: both direct RoCE rails were not connectivity-validated on both nodes" >&2
  exit 1
}
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -m "$ROCE_MODULE" snapshot-counters \
  --out_dir "$NETWORK_DIR" --phase pre

MASTER_ADDR="$($PYTHON_BIN - "$NETWORK_DIR/PAIR.json" "${ALLOCATED_NODES[0]}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["rails"][0]["endpoints"][sys.argv[2]])
PY
)"
MASTER_PORT="$((24000 + SLURM_JOB_ID % 16000))"
export MASTER_ADDR MASTER_PORT
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=5
export NCCL_SOCKET_IFNAME="enp1s0f1np1,enP2p1s0f1np1"
export GLOO_SOCKET_IFNAME="enp1s0f1np1"
export NCCL_CROSS_NIC=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET

# Exercise each f1 rail independently and then both together.  Every profile
# runs exact-sum warmups and timed all-reduces for the audited 4,437,580-byte
# Stage-C gradient plus 64- and 256-MiB payloads.  NCCL_NET=IB makes any socket
# data-transport fallback fatal instead of silently accepting 1-GbE.
run_nccl_profile() {
  local profile="$1" hcas="$2"
  export NCCL_IB_HCA="$hcas"
  export NCCL_DEBUG_FILE="$NETWORK_DIR/nccl.${profile}.%h.%p.log"
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m "$ROCE_MODULE" distributed-smoke --profile "$profile" \
    --out_file "$NETWORK_DIR/NCCL_BENCHMARK_${profile}.json"
  "$PYTHON_BIN" -m "$ROCE_MODULE" verify-logs \
    --manifest_dir "$NETWORK_DIR" --prefix "nccl.${profile}." \
    --profile "$profile" --out_file "$NETWORK_DIR/NCCL_LOGS_${profile}.json"
}
run_nccl_profile single_primary "rocep1s0f1:1"
run_nccl_profile single_secondary "roceP2p1s0f1:1"
run_nccl_profile dual "rocep1s0f1:1,roceP2p1s0f1:1"
"$PYTHON_BIN" -m "$ROCE_MODULE" compare-benchmarks --manifest_dir "$NETWORK_DIR"
for evidence in NCCL_BENCHMARK_single_primary.json \
  NCCL_BENCHMARK_single_secondary.json NCCL_BENCHMARK_dual.json \
  NCCL_DUAL_VS_SINGLE.json; do
  [[ -s "$NETWORK_DIR/$evidence" ]] || {
    echo "ERROR: forced-IB NCCL benchmark evidence is absent: $evidence" >&2
    exit 1
  }
done
# Use both rails when its gradient-payload median is within 5% of the fastest
# single rail. Otherwise retain the fastest healthy 200-Gb/s rail; the other
# rail remains positively tested, but is not imposed on training.
read -r TRAINING_PROFILE TRAINING_HCAS < <(
  "$PYTHON_BIN" - "$NETWORK_DIR/NCCL_DUAL_VS_SINGLE.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["training_profile"], value["training_hcas"])
PY
)
case "$TRAINING_PROFILE:$TRAINING_HCAS" in
  single_primary:rocep1s0f1:1|single_secondary:roceP2p1s0f1:1|dual:rocep1s0f1:1,roceP2p1s0f1:1) ;;
  *) echo "ERROR: benchmark selected an unsupported training rail profile" >&2; exit 1 ;;
esac
export TRAINING_PROFILE
export NCCL_IB_HCA="$TRAINING_HCAS"

# The bank is copied independently to each node, with only train and val
# neighbor tables.  The strict stager rejects any attempt to request test.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash "$STAGER" stage "$LOCAL_BANK" "$SOURCE_BANK" "$PROJECT_DIR" \
  "$PYTHON_BIN" "$MEMORY_CFG" "train val"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c 'set -euo pipefail; d="$1"; [[ -f "$d/neighbors_train.npz" && -f "$d/neighbors_val.npz" && ! -e "$d/neighbors_test.npz" ]]' \
  bash "$LOCAL_BANK"

export SIGNTRAJ_SOURCE_GIT_HEAD="$SOURCE_GIT_HEAD"
export SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR="$PARTITION_DIR"
export SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256="$EXPECTED_DEV_SHA256"
export SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS=347
export SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY="$EXPECTED_PARTITION_IDENTITY"
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_BANK"
unset SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256

write_launch_evidence() {
  "$PYTHON_BIN" - "$ATTEMPT" "$SOURCE_GIT_HEAD" "$SOURCE_REMOTE_BRANCH" \
    "$REMOTE_HEAD" "$PAIR_CONSTRAINT" "$MASTER_ADDR" "$MASTER_PORT" \
    "$MEMORY_CFG" "$OFF_CFG" "$STAGE_B_CHECKPOINT" "$STAGE_B_DECISION" \
    "$V2_CHECKPOINT" "$CALIBRATION_DIR/READY" "$DEV_MANIFEST" \
    "$SOURCE_BANK/READY" "$NETWORK_DIR/PAIR.json" \
    "$NETWORK_DIR/NCCL_BENCHMARK_single_primary.json" \
    "$NETWORK_DIR/NCCL_BENCHMARK_single_secondary.json" \
    "$NETWORK_DIR/NCCL_BENCHMARK_dual.json" \
    "$NETWORK_DIR/NCCL_DUAL_VS_SINGLE.json" "$CPU_GATE_READY" \
    "$CALIBRATION_COMPLETION" "$LEASE_ATTESTATION" "$DECISION_POLICY" \
    "$SOURCE_BINDING_MANIFEST" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    publish_bytes_no_replace,
)

def identity(path):
    value = Path(path)
    return {"path": str(value.resolve()), "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}

attempt = Path(sys.argv[1])
payload = {
    "schema_name": "signtrajfield_centered_stage_c_paired_pilot_launch",
    "schema_version": 1,
    "development_only": True,
    "non_authorizing": True,
    "promotion_eligible": False,
    "execution_mode": os.environ["STAGE_C_EXECUTION_MODE"],
    "confirmation_or_test_access_permitted": False,
    "source_git_head": sys.argv[2],
    "source_remote_ref": f"origin/{sys.argv[3]}",
    "source_remote_head": sys.argv[4],
    "pair_constraint": sys.argv[5],
    "nodes": os.environ["SLURM_NODELIST"],
    "world_size": 2,
    "batch_per_rank": 64,
    "accumulation_steps": 2,
    "effective_global_batch": 256,
    "master_address": sys.argv[6],
    "master_port": int(sys.argv[7]),
    "training_network_profile": os.environ["TRAINING_PROFILE"],
    "training_nccl_ib_hca": os.environ["NCCL_IB_HCA"],
    "active_lease_claim_identity": os.environ["STAGE_C_LEASE_CLAIM_IDENTITY"],
    "lease_attestation": identity(sys.argv[23]),
    "arms_run_order": ["memory", "matched_off"],
    "artifacts": {
        "memory_config": identity(sys.argv[8]),
        "matched_off_config": identity(sys.argv[9]),
        "stage_b_checkpoint": identity(sys.argv[10]),
        "stage_b_terminal_decision": identity(sys.argv[11]),
        "frozen_v2_teacher": identity(sys.argv[12]),
        "calibration_seal": identity(sys.argv[13]),
        "development_manifest": identity(sys.argv[14]),
        "sentence_bank_seal": identity(sys.argv[15]),
        "paired_roce": identity(sys.argv[16]),
        "forced_ib_single_primary": identity(sys.argv[17]),
        "forced_ib_single_secondary": identity(sys.argv[18]),
        "forced_ib_dual": identity(sys.argv[19]),
        "dual_vs_single_comparison": identity(sys.argv[20]),
        "cpu_gate": identity(sys.argv[21]),
        "calibration_completion": identity(sys.argv[22]),
        "decision_policy": identity(sys.argv[24]),
        "source_binding_manifest": identity(sys.argv[25]),
    },
}
smoke_ready = Path(os.environ["STAGE_C_SMOKE_READY"])
if smoke_ready.is_file():
    payload["artifacts"]["prior_one_update_smoke"] = identity(smoke_ready)
path = attempt / "LAUNCH.json"
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
publish_bytes_no_replace(path, encoded)
PY
}
write_launch_evidence

run_arm() {
  local arm="$1" config="$2" out_dir="$ATTEMPT/$1"
  local -a bounded_args=()
  revalidate_source_binding
  if [[ "$EXECUTION_MODE" == "smoke" ]]; then
    # Two accumulated logical batches produce exactly one optimizer update.
    bounded_args=(--max_train_batches 2 --max_val_batches 1)
  fi
  [[ ! -e "$out_dir" ]] || { echo "ERROR: no-replace arm output exists: $out_dir" >&2; return 1; }
  echo "BEGIN_STAGE_C_ARM=$arm"
  export NCCL_DEBUG_FILE="$NETWORK_DIR/nccl.train.${arm}.${TRAINING_PROFILE}.%h.%p.log"
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$PYTHON_BIN" -m "$TRAINER_MODULE" \
    --config "$config" --device cuda --text_device cpu \
    --distributed ddp --ddp_backend nccl --ddp_timeout_min 30 \
    --epochs 1 --batch_size 64 \
    "${bounded_args[@]}" \
    --stage_c_warm_start "$STAGE_B_CHECKPOINT" --out_dir "$out_dir"
  "$PYTHON_BIN" - "$out_dir" "$arm" "$EXECUTION_MODE" <<'PY'
import json, sys
from pathlib import Path

import torch

directory, arm, execution_mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
required = [
    directory / "config.resolved.json",
    directory / "metrics.jsonl",
    directory / "selection_summary.json",
    directory / "checkpoints/last.pt",
    directory / "checkpoints/epoch0001.pt",
]
if any(not path.is_file() or path.is_symlink() for path in required):
    raise SystemExit(f"ERROR: Stage-C arm {arm} did not publish its complete one-epoch result")
rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
expected_step = 1 if execution_mode == "smoke" else 72
if (
    len(rows) != 1
    or rows[0].get("epoch") != 1
    or rows[0].get("global_step") != expected_step
):
    raise SystemExit(f"ERROR: Stage-C arm {arm} did not run exactly one epoch")
for checkpoint_name in ("last.pt", "epoch0001.pt"):
    checkpoint = torch.load(
        directory / "checkpoints" / checkpoint_name,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("epoch") != 1 or checkpoint.get("global_step") != expected_step:
        raise SystemExit(
            f"ERROR: Stage-C {execution_mode} arm {arm}/{checkpoint_name} has "
            f"epoch/step={checkpoint.get('epoch')}/{checkpoint.get('global_step')}, "
            f"expected 1/{expected_step}"
        )
resolved = json.loads((directory / "config.resolved.json").read_text(encoding="utf-8"))
stage = resolved.get("sentence_memory_safety", {}).get("stage_c", {})
if stage.get("arm") != arm or not stage.get("development_only") or not stage.get("non_authorizing") or stage.get("promotion_eligible"):
    raise SystemExit(f"ERROR: Stage-C arm {arm} lost its non-authorizing contract")
if execution_mode == "smoke":
    if resolved.get("train", {}).get("max_train_batches") != 2 or resolved.get("eval", {}).get("max_batches") != 1:
        raise SystemExit(f"ERROR: Stage-C smoke arm {arm} lost its bounded-batch contract")
else:
    if resolved.get("train", {}).get("max_train_batches", 0) != 0 or resolved.get("eval", {}).get("max_batches", 0) != 0:
        raise SystemExit(f"ERROR: Stage-C pilot arm {arm} is unexpectedly batch-bounded")
if resolved.get("train", {}).get("length_bucketed_batches") is not True or resolved.get("train", {}).get("drop_last") is not False:
    raise SystemExit(f"ERROR: Stage-C arm {arm} lost its exact batch-construction contract")
if resolved.get("data", {}).get("limit_train", 0) != 0 or resolved.get("data", {}).get("limit_val", 0) != 0:
    raise SystemExit(f"ERROR: Stage-C arm {arm} is unexpectedly data-limited")
PY
  "$PYTHON_BIN" -m "$ROCE_MODULE" verify-logs \
    --manifest_dir "$NETWORK_DIR" \
    --prefix "nccl.train.${arm}.${TRAINING_PROFILE}." \
    --profile "$TRAINING_PROFILE" \
    --out_file "$NETWORK_DIR/NCCL_LOGS_train_${arm}.json"
  echo "COMPLETE_STAGE_C_ARM=$arm"
}

# Separate trainer processes ensure each arm resets RNG, epoch, optimizer, and
# selection state, while --stage_c_warm_start reloads the same source weights.
run_arm memory "$MEMORY_CFG"
run_arm matched_off "$OFF_CFG"
revalidate_source_binding

# Publication is impossible until every serialized tensor/state transition has
# independently proved the exact 233-total, 20-trainable/213-frozen split.
if [[ "$EXECUTION_MODE" == "smoke" ]]; then
  CHECKPOINT_AUDIT="$NETWORK_DIR/SMOKE_CHECKPOINT_AUDIT.json"
else
  CHECKPOINT_AUDIT="$NETWORK_DIR/PILOT_CHECKPOINT_AUDIT.json"
fi
"$PYTHON_BIN" -m "$SMOKE_AUDIT_MODULE" \
  --source_checkpoint "$STAGE_B_CHECKPOINT" \
  --memory_dir "$ATTEMPT/memory" \
  --matched_off_dir "$ATTEMPT/matched_off" \
  --execution_mode "$EXECUTION_MODE" --out_file "$CHECKPOINT_AUDIT"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -m "$ROCE_MODULE" snapshot-counters \
  --out_dir "$NETWORK_DIR" --phase post
"$PYTHON_BIN" -m "$ROCE_MODULE" compare-counters \
  --manifest_dir "$NETWORK_DIR" --nodes "${ALLOCATED_NODES[@]}"

[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: confirmation-spend marker appeared during development-only Stage C" >&2
  exit 1
}
"$PYTHON_BIN" - "$ATTEMPT" "$LEASE_ATTESTATION" "$DECISION_POLICY" \
  "origin/$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    publish_bytes_no_replace,
)

attempt = Path(sys.argv[1])
artifacts = {}
arm_execution_summaries = {}
mode = os.environ["STAGE_C_EXECUTION_MODE"]
expected_step = 1 if mode == "smoke" else 72
for arm in ("memory", "matched_off"):
    directory = attempt / arm
    artifacts[arm] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "config.resolved.json", "metrics.jsonl", "selection_summary.json",
            "checkpoints/last.pt", "checkpoints/epoch0001.pt",
        )
    }
    row = json.loads(
        next(
            line for line in (directory / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        )
    )
    resolved = json.loads(
        (directory / "config.resolved.json").read_text(encoding="utf-8")
    )
    if row.get("epoch") != 1 or row.get("global_step") != expected_step:
        raise SystemExit(f"ERROR: Stage-C COMPLETE cannot bind {arm} epoch/step")
    arm_execution_summaries[arm] = {
        "epoch": 1,
        "global_step": expected_step,
        "data_limit_train": resolved.get("data", {}).get("limit_train", 0),
        "data_limit_val": resolved.get("data", {}).get("limit_val", 0),
        "train_max_batches": resolved.get("train", {}).get("max_train_batches", 0),
        "eval_max_batches": resolved.get("eval", {}).get("max_batches", 0),
        "length_bucketed_batches": resolved.get("train", {}).get(
            "length_bucketed_batches"
        ),
        "drop_last": resolved.get("train", {}).get("drop_last"),
    }
def identity(path):
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise SystemExit(f"ERROR: incomplete Stage-C execution evidence: {value}")
    return hashlib.sha256(value.read_bytes()).hexdigest()

network = attempt / "network_preflight"
execution_artifacts = {
    "launch": identity(attempt / "LAUNCH.json"),
    "source_binding": identity(network / "SOURCE_BINDING.json"),
    "pair": identity(network / "PAIR.json"),
    "counter_health": identity(network / "COUNTER_HEALTH.json"),
    "dual_vs_single": identity(network / "NCCL_DUAL_VS_SINGLE.json"),
    "train_memory_logs": identity(network / "NCCL_LOGS_train_memory.json"),
    "train_matched_off_logs": identity(network / "NCCL_LOGS_train_matched_off.json"),
}
pair_evidence = json.loads((network / "PAIR.json").read_text(encoding="utf-8"))
pair_nodes = pair_evidence.get("nodes")
if (
    not isinstance(pair_nodes, list)
    or len(pair_nodes) != 2
    or len(set(pair_nodes)) != 2
):
    raise SystemExit("ERROR: Stage-C COMPLETE cannot bind malformed pair nodes")
for index, node in enumerate(pair_nodes):
    execution_artifacts[f"node_audit_{index}"] = identity(
        network / f"{node}.node.json"
    )
    execution_artifacts[f"connectivity_audit_{index}"] = identity(
        network / f"{node}.connectivity.json"
    )
    for phase in ("pre", "post"):
        execution_artifacts[f"counter_{phase}_{index}"] = identity(
            network / f"{node}.counters_{phase}.json"
        )
for profile in ("single_primary", "single_secondary", "dual"):
    execution_artifacts[f"benchmark_{profile}"] = identity(
        network / f"NCCL_BENCHMARK_{profile}.json"
    )
    execution_artifacts[f"benchmark_logs_{profile}"] = identity(
        network / f"NCCL_LOGS_{profile}.json"
    )
if os.environ["STAGE_C_EXECUTION_MODE"] == "smoke":
    execution_artifacts["smoke_checkpoint_audit"] = identity(
        network / "SMOKE_CHECKPOINT_AUDIT.json"
    )
else:
    execution_artifacts["pilot_checkpoint_audit"] = identity(
        network / "PILOT_CHECKPOINT_AUDIT.json"
    )
payload = {
    "schema_name": "signtrajfield_centered_stage_c_paired_execution_complete",
    "schema_version": 1,
    "development_only": True,
    "non_authorizing": True,
    "promotion_eligible": False,
    "execution_mode": mode,
    "source_git_head": os.environ["SIGNTRAJ_SOURCE_GIT_HEAD"].lower(),
    "source_remote_ref": sys.argv[4],
    "source_remote_head": sys.argv[5].lower(),
    "pair_constraint": os.environ["STAGE_C_PAIR_CONSTRAINT"],
    "world_size": 2,
    "batch_per_rank": 64,
    "accumulation_steps": 2,
    "effective_global_batch": 256,
    "confirmation_manifest_opened": False,
    "test_data_accessed": False,
    "active_lease_claim_identity": os.environ["STAGE_C_LEASE_CLAIM_IDENTITY"],
    "lease_attestation_path": str(Path(sys.argv[2]).resolve()),
    "lease_attestation_sha256": identity(Path(sys.argv[2])),
    "lease_attestation_identity": json.loads(
        Path(sys.argv[2]).read_text(encoding="utf-8")
    )["attestation_identity"],
    "decision_policy": {
        "path": str(Path(sys.argv[3]).resolve()),
        "sha256": identity(Path(sys.argv[3])),
    },
    "arms": artifacts,
    "arm_execution_summaries": arm_execution_summaries,
    "execution_artifacts": execution_artifacts,
}
path = attempt / "COMPLETE.json"
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
publish_bytes_no_replace(path, encoded)
PY
fi

EXECUTION_COMPLETE="$ATTEMPT/COMPLETE.json"
DECISION="$ATTEMPT/DECISION.json"
if [[ -f "$DECISION" ]]; then
  "$PYTHON_BIN" -m "$DECISION_MODULE" verify \
    --mode "$EXECUTION_MODE" --execution_complete "$EXECUTION_COMPLETE" \
    --source_checkpoint "$STAGE_B_CHECKPOINT" \
    --memory_checkpoint "$ATTEMPT/memory/checkpoints/last.pt" \
    --matched_off_checkpoint "$ATTEMPT/matched_off/checkpoints/last.pt" \
    --policy "$DECISION_POLICY" \
    --active_lease_attestation "$LEASE_ATTESTATION" \
    --expected_source_git_head "$SOURCE_GIT_HEAD" \
    --expected_source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
    --expected_source_remote_head "$REMOTE_HEAD" \
    --expected_pair_constraint "$PAIR_CONSTRAINT" --decision "$DECISION"
else
  "$PYTHON_BIN" -m "$DECISION_MODULE" create \
    --mode "$EXECUTION_MODE" --execution_complete "$EXECUTION_COMPLETE" \
    --source_checkpoint "$STAGE_B_CHECKPOINT" \
    --memory_checkpoint "$ATTEMPT/memory/checkpoints/last.pt" \
    --matched_off_checkpoint "$ATTEMPT/matched_off/checkpoints/last.pt" \
    --policy "$DECISION_POLICY" \
    --active_lease_attestation "$LEASE_ATTESTATION" \
    --expected_source_git_head "$SOURCE_GIT_HEAD" \
    --expected_source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
    --expected_source_remote_head "$REMOTE_HEAD" \
    --expected_pair_constraint "$PAIR_CONSTRAINT" --out_file "$DECISION"
fi

# Publish COMPLETE and READY together by one same-directory rename.  Thus a
# requeue can observe either no selectable result or the whole immutable pair,
# never a half-published gate.
"$PYTHON_BIN" - "$MODE_CONTROL_DIR" "$MODE_PUBLICATION" "$MODE_COMPLETE" \
  "$MODE_READY" "$EXECUTION_COMPLETE" "$DECISION" "$LEASE_ATTESTATION" \
  "$DECISION_POLICY" "$SOURCE_GIT_HEAD" "$PAIR_CONSTRAINT" \
  "$EXECUTION_MODE" "origin/$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" <<'PY'
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    rename_no_replace,
)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

control, publication, final_complete, final_ready = map(Path, sys.argv[1:5])
execution, decision_path, attestation_path, policy_path = map(Path, sys.argv[5:9])
head, pair, mode, remote_ref, remote_head = sys.argv[9:14]
decision = json.loads(decision_path.read_text(encoding="utf-8"))
attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
execution_complete = json.loads(execution.read_text(encoding="utf-8"))
claim = os.environ["STAGE_C_LEASE_CLAIM_IDENTITY"]
execution_claim = execution_complete["active_lease_claim_identity"]
expected_step = 1 if mode == "smoke" else 72
expected_success_status = (
    "smoke_ready" if mode == "smoke" else "pilot_complete_development_signal"
)
active_claim = attestation["claim"]
active_binding = active_claim["execution_binding"]
replaced = active_claim.get("replaced_stale_owner") or {}
launch = json.loads((execution.parent / "LAUNCH.json").read_text(encoding="utf-8"))
launch_artifacts = launch.get("artifacts") or {}
if (
    decision.get("active_lease_claim_identity") != execution_claim
    or decision.get("status") not in {expected_success_status, "stop"}
    or decision.get("execution_mode") != mode
    or active_claim.get("claim_identity") != claim
    or active_binding.get("source_git_head") != head.lower()
    or active_binding.get("source_remote_ref") != remote_ref
    or active_binding.get("source_remote_head") != remote_head.lower()
    or active_binding.get("pair_constraint") != pair
    or execution_complete.get("source_git_head") != head.lower()
    or execution_complete.get("source_remote_ref") != remote_ref
    or execution_complete.get("source_remote_head") != remote_head.lower()
    or execution_complete.get("pair_constraint") != pair
    or decision.get("source_git_head") != head.lower()
    or decision.get("pair_constraint") != pair
    or (launch_artifacts.get("memory_config") or {}).get("sha256")
    != active_binding.get("memory_config_sha256")
    or (launch_artifacts.get("matched_off_config") or {}).get("sha256")
    != active_binding.get("matched_off_config_sha256")
    or (launch_artifacts.get("cpu_gate") or {}).get("sha256")
    != active_binding.get("cpu_gate_sha256")
    or (launch_artifacts.get("calibration_completion") or {}).get("sha256")
    != active_binding.get("calibration_completion_sha256")
    or not (
        execution_claim == claim
        or replaced.get("claim_identity") == execution_claim
        or execution_claim in active_claim.get("prior_claim_identities", [])
    )
):
    raise SystemExit("ERROR: Stage-C decision cannot bind the active execution claim")
complete = {
    "schema_name": "signtrajfield_stage_c_mode_complete",
    "schema_version": 1,
    "execution_mode": mode,
    "source_git_head": head.lower(),
    "pair_constraint": pair,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or "0"),
    "active_lease_claim_identity": claim,
    "execution_lease_claim_identity": execution_claim,
    "lease_attestation_path": str(attestation_path.resolve()),
    "lease_attestation_sha256": sha(attestation_path),
    "lease_attestation_identity": attestation["attestation_identity"],
    "execution_complete_path": str(execution.resolve()),
    "execution_complete_sha256": sha(execution),
    "decision_path": str(decision_path.resolve()),
    "decision_sha256": sha(decision_path),
    "decision_identity": decision["decision_identity"],
    "decision_status": decision["status"],
    "decision_policy_path": str(policy_path.resolve()),
    "decision_policy_sha256": sha(policy_path),
    "expected_epoch": 1,
    "expected_global_step_per_arm": expected_step,
    "world_size": 2,
    "batch_per_rank": 64,
    "accumulation_steps": 2,
    "effective_global_batch": 256,
    "development_only": True,
    "non_authorizing": True,
    "promotion_eligible": False,
    "authorized_purpose": None,
    "confirmation_manifest_opened": False,
    "test_data_accessed": False,
}
complete_bytes = encode(complete)
ready = {
    "schema_name": "signtrajfield_stage_c_mode_ready",
    "schema_version": 1,
    "execution_mode": mode,
    "source_git_head": head.lower(),
    "pair_constraint": pair,
    "active_lease_claim_identity": claim,
    "execution_lease_claim_identity": execution_claim,
    "complete_path": str(final_complete.resolve()),
    "complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
    "decision_path": str(decision_path.resolve()),
    "decision_sha256": sha(decision_path),
    "decision_identity": decision["decision_identity"],
    "decision_status": decision["status"],
    "expected_epoch": 1,
    "expected_global_step_per_arm": expected_step,
    "one_optimizer_update_per_arm": mode == "smoke",
    "one_full_train_epoch_per_arm": mode == "pilot",
    "development_only": True,
    "non_authorizing": True,
    "promotion_eligible": False,
    "authorized_purpose": None,
    "confirmation_manifest_opened": False,
    "test_data_accessed": False,
}
control.mkdir(parents=True, exist_ok=True)
building = control / f".PUBLICATION.building.{claim}.{uuid.uuid4().hex}"
if publication.exists():
    raise SystemExit("ERROR: immutable Stage-C mode publication already exists")
building.mkdir()
try:
    for name, content in (("COMPLETE.json", complete_bytes), ("READY", encode(ready))):
        descriptor = os.open(building / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
    rename_no_replace(building, publication)
finally:
    if building.exists():
        shutil.rmtree(building)
PY

"$PYTHON_BIN" -m "$EXECUTION_CONTROL_MODULE" release \
  --lease "$LEASE" --attestation "$LEASE_ATTESTATION" \
  --completion "$MODE_COMPLETE" --ready "$MODE_READY"
LEASE_ACTIVE=0
DECISION_STATUS="$($PYTHON_BIN - "$MODE_READY" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision_status"])
PY
)"
echo "STAGE_C_PAIRED_${EXECUTION_MODE^^}_COMPLETE=$MODE_PUBLICATION"
if [[ "$EXECUTION_MODE" == "smoke" ]]; then
  EXPECTED_SUCCESS_STATUS="smoke_ready"
else
  EXPECTED_SUCCESS_STATUS="pilot_complete_development_signal"
fi
if [[ "$DECISION_STATUS" != "$EXPECTED_SUCCESS_STATUS" ]]; then
  echo "ERROR: immutable Stage-C $EXECUTION_MODE decision is stop" >&2
  exit 2
fi
