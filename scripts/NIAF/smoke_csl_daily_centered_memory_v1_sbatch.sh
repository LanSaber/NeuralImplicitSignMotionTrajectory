#!/bin/bash
#SBATCH --job-name=csl_centered_smoke
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# Sequential one-optimizer-step implementation coverage for both centered arms.
set -euo pipefail
trap 'echo "ERROR: centered two-arm smoke failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the clean pushed commit}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
TRAIN_WRAPPER="$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"
PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
CALIBRATION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_v1_retry3"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"

[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-0}" == "1" ]] || {
  echo "ERROR: centered smoke requires one Slurm node" >&2; exit 1;
}
for required in "$PYTHON_BIN" "$STAGER" "$TRAIN_WRAPPER" "$PARTITION_DIR/READY" "$CALIBRATION_DIR/READY"; do
  [[ -e "$required" ]] || { echo "ERROR: smoke prerequisite absent: $required" >&2; exit 1; }
done
[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: the global confirmation holdout was already spent" >&2; exit 1;
}
if [[ "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_V2_SHA256" ]]; then
  echo "ERROR: smoke v2 checkpoint identity changed" >&2; exit 1
fi
for forbidden in RESUME WARM_START BASE_CHECKPOINT PHASE_B_GATE_REPORT; do
  [[ -z "${!forbidden:-}" ]] || { echo "ERROR: smoke forbids $forbidden" >&2; exit 1; }
done

cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git \
      || "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" \
      || "$(git rev-parse HEAD)" != "$SOURCE_GIT_HEAD" \
      || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: smoke requires the exact clean shared standalone source" >&2
  exit 1
fi
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
[[ -n "$REMOTE_HEAD" && "${REMOTE_HEAD,,}" == "${SOURCE_GIT_HEAD,,}" ]] || {
  echo "ERROR: smoke source is not the pushed branch head" >&2; exit 1;
}

export PATH="$PYTHON_ENV/bin:$PATH" PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

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

LOCAL_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
cleanup() {
  local exit_code=$?
  trap - EXIT
  "$STAGER" cleanup "$LOCAL_BANK" || true
  exit "$exit_code"
}
trap cleanup EXIT
STAGE_A_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1_smoke_retry3.yaml"
"$STAGER" stage "$LOCAL_BANK" "$SOURCE_BANK" "$PROJECT_DIR" "$PYTHON_BIN" "$STAGE_A_CFG" "train val"
[[ -f "$LOCAL_BANK/neighbors_train.npz" && -f "$LOCAL_BANK/neighbors_val.npz" && ! -e "$LOCAL_BANK/neighbors_test.npz" ]] || {
  echo "ERROR: smoke local staging violated train/val-only visibility" >&2; exit 1;
}

run_arm() {
  local experiment="$1"
  local cfg="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${experiment}.yaml"
  local out="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$experiment"
  [[ -f "$cfg" && ! -e "$out" ]] || {
    echo "ERROR: smoke config absent or output non-fresh: $experiment" >&2
    return 1
  }
  CFG="$cfg" OUT_DIR="$out" RUN_TAG="$experiment" EPOCHS=1 \
  LIMIT_TRAIN=64 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 \
  SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR="$PARTITION_DIR" \
  SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_BANK" SENTENCE_MEMORY_DIR="" \
  STAGE_SENTENCE_MEMORY=0 WANDB=0 WANDB_MODE=disabled \
  DEVICE=cuda TEXT_DEVICE=cpu DISTRIBUTED=none DDP_BACKEND=auto \
    bash "$TRAIN_WRAPPER"

  "$PYTHON_BIN" - "$out/checkpoints/last.pt" "$experiment" "$V2_CHECKPOINT" <<'PY'
import math
import sys
from pathlib import Path

import torch

from NIAF.continuous_trajectory_field.scripts import (
    train_continuous_trajectory_field as trainer,
)
from NIAF.continuous_trajectory_field.models.hierarchical_field import (
    build_continuous_trajectory_field,
)

checkpoint = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
if int(checkpoint.get("epoch", -1)) != 1 or int(checkpoint.get("global_step", -1)) != 1:
    raise SystemExit("ERROR: smoke did not execute exactly one optimizer step")
cfg = checkpoint.get("config", {})
if cfg.get("experiment_name") != sys.argv[2]:
    raise SystemExit("ERROR: smoke experiment identity changed")
resolved = trainer.resolve_sentence_memory_relevance_calibration(cfg)
if resolved != cfg.get("sentence_memory", {}).get(
    "resolved_relevance_calibration_identity"
):
    raise SystemExit("ERROR: smoke calibration identity does not revalidate")
for name, function in (
    ("architecture", trainer.sentence_memory_architecture_identity),
    ("behavior", trainer.sentence_memory_behavior_identity),
    ("objective", trainer.sentence_memory_objective_identity),
    ("evaluation_control", trainer.sentence_memory_evaluation_control_identity),
    ("selection_aggregation", trainer.sentence_memory_selection_aggregation_identity),
):
    if checkpoint.get(f"sentence_memory_{name}_identity") != function(cfg):
        raise SystemExit(f"ERROR: smoke {name} identity does not reload")
parity = checkpoint.get("v2_to_v3_text_only_parity", {})
if (
    parity.get("passed") is not True
    or float(parity.get("prediction_max_abs", math.inf)) > 1e-7
    or float(parity.get("duration_max_abs", math.inf)) > 1e-7
):
    raise SystemExit("ERROR: smoke lacks strict v2 memory-off parity")
if not isinstance(checkpoint.get("optimizer"), dict):
    raise SystemExit("ERROR: smoke optimizer state is absent")
metrics = checkpoint.get("metrics", {})
expected_modes = [
    "text_only", "sentence_memory", "motion_shuffled_n0_sentence_memory",
    "motion_shuffled_n1_sentence_memory", "motion_shuffled_n2_sentence_memory",
    "cross_query_motion_sentence_memory", "full_replacement_sentence_memory",
    "joint_tuple_permuted_sentence_memory", "uniform_final_mass_sentence_memory",
    "analytic_prior_sentence_memory",
]
if cfg.get("sentence_memory", {}).get("association_mode") != "none":
    expected_modes.append("association_disabled_sentence_memory")
for namespace in expected_modes:
    if not any(str(key).startswith(f"val_{namespace}/") for key in metrics):
        raise SystemExit(f"ERROR: smoke lacks validation namespace {namespace}")
exact = (
    "uniform_final_vs_off_prediction_max_abs",
    "uniform_final_vs_off_duration_max_abs",
    "broadcast_complete_vs_off_prediction_max_abs",
    "broadcast_complete_vs_off_duration_max_abs",
    "all_null_vs_off_prediction_max_abs",
    "all_null_vs_off_duration_max_abs",
    "all_null_gate_max_abs",
    "all_null_candidate_mass_max_abs",
    "all_null_one_minus_null_mass_max_abs",
)
for name in exact:
    if float(metrics.get(f"val_paired_sentence_memory/{name}", math.nan)) != 0.0:
        raise SystemExit(f"ERROR: smoke exact integrity audit failed: {name}")
for name in ("joint_tuple_prediction_max_abs", "joint_tuple_duration_max_abs"):
    value = float(metrics.get(f"val_paired_sentence_memory/{name}", math.nan))
    if value > 1e-7:
        raise SystemExit(f"ERROR: smoke joint tuple audit failed: {name}")
motion_fraction = float(
    metrics.get("train_sentence_memory_motion_corrupt_fraction", math.nan)
)
full_fraction = float(
    metrics.get("train_sentence_memory_full_shuffle_fraction", math.nan)
)
if not (0.0 < motion_fraction < 1.0 and 0.0 < full_fraction < 1.0):
    raise SystemExit("ERROR: smoke did not exercise both paired corruption arms")

def finite_tree(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items(): finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value): finite_tree(child, f"{path}[{index}]")
    elif torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise SystemExit(f"ERROR: non-finite tensor: {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"ERROR: non-finite scalar: {path}")

finite_tree(metrics, "metrics")
finite_tree(checkpoint["optimizer"], "optimizer")
state = checkpoint.get("model", {})
for name, value in state.items():
    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        raise SystemExit(f"ERROR: non-finite smoke tensor: {name}")
v2 = torch.load(Path(sys.argv[3]), map_location="cpu", weights_only=False)
v2_state = v2.get("model", {})
base_names = {name for name in state if not trainer.is_sentence_memory_parameter(name)}
if base_names != set(v2_state):
    raise SystemExit("ERROR: smoke base/planner state keys differ from pinned v2")
for name in base_names:
    if not torch.equal(state[name], v2_state[name]):
        raise SystemExit(f"ERROR: frozen base/planner tensor changed: {name}")

model = build_continuous_trajectory_field(cfg, text_dim=768)
model.load_state_dict(state, strict=True)
trainable = trainer.configure_sentence_memory_trainable_parameters(model, cfg)
expected_trainable = {
    name for name, parameter in model.named_parameters()
    if trainer.is_sentence_memory_parameter(name)
}
if set(trainable.get("trainable", ())) != expected_trainable or any(
    parameter.requires_grad != (name in expected_trainable)
    for name, parameter in model.named_parameters()
):
    raise SystemExit("ERROR: smoke trainable tensor set is not exactly sentence memory")
optimizer = trainer.build_optimizer(model, cfg)
optimizer.load_state_dict(checkpoint["optimizer"])
name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
if [group.get("group_name") for group in optimizer.param_groups] != ["sentence_memory"]:
    raise SystemExit("ERROR: smoke optimizer includes a non-memory group")
association_parameters = []
for group in optimizer.param_groups:
    for parameter in group["params"]:
        name = name_by_parameter[id(parameter)]
        state_row = optimizer.state.get(parameter)
        if not state_row:
            raise SystemExit(f"ERROR: optimizer state absent for {name}")
        finite_tree(state_row, f"optimizer.{name}")
        if "association" in name:
            association_parameters.append((name, state_row))
if cfg.get("sentence_memory", {}).get("association_mode") != "none":
    if not association_parameters:
        raise SystemExit("ERROR: Stage-B association parameters are absent")
    for name, state_row in association_parameters:
        moment = state_row.get("exp_avg")
        if not torch.is_tensor(moment) or float(moment.abs().max()) == 0.0:
            raise SystemExit(f"ERROR: Stage-B association gradient was zero: {name}")
PY
}

run_arm csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1_smoke_retry3
run_arm csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1_smoke_retry3
