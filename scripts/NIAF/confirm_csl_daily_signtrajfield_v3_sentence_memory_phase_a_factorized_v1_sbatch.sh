#!/bin/bash
#SBATCH --job-name=csl_v3_splitkv_confirm
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00

# One-shot confirmation for exactly the first development-feasible stage.
set -euo pipefail
trap 'echo "ERROR: factorized-memory confirmation failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
FACTORIZED_STAGE="${FACTORIZED_STAGE:?Set FACTORIZED_STAGE to the first feasible stage1 or stage2}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the locked training source commit}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
V2_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
STAGE1_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
STAGE2_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1"

case "$FACTORIZED_STAGE" in
  stage1) EXPERIMENT="$STAGE1_EXPERIMENT" ;;
  stage2) EXPERIMENT="$STAGE2_EXPERIMENT" ;;
  *) echo "ERROR: invalid FACTORIZED_STAGE" >&2; exit 1 ;;
esac

RUN_DIR="${RUN_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$EXPERIMENT}"
CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${EXPERIMENT}.yaml"
CHECKPOINT="$RUN_DIR/checkpoints/best.pt"
SEALED_PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
PARTITION_DIR="${PARTITION_DIR:-$SEALED_PARTITION_DIR}"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"
EXPECTED_DEVELOPMENT_MANIFEST_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
AUTHORIZATION="${AUTHORIZATION:-$RUN_DIR/evaluation/ordered_development_decision/authorize_confirmation.json}"
DEVELOPMENT_DIAGNOSTIC_DIR="${DEVELOPMENT_DIAGNOSTIC_DIR:-$RUN_DIR/evaluation/locked_development_slot_diagnostics}"
OUT_ROOT="${OUT_ROOT:-$RUN_DIR/evaluation/locked_validation_confirmation}"
CONTROL_DIR="${CONTROL_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control}"
HOLDOUT_SPEND="$CONTROL_DIR/confirmation_holdout_spent.json"
STAGE1_RUN_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$STAGE1_EXPERIMENT"
STAGE2_RUN_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$STAGE2_EXPERIMENT"
STAGE1_STAGE2_AUTH="$STAGE1_RUN_DIR/evaluation/ordered_development_decision/authorize_stage2.json"
LOCAL_SENTENCE_MEMORY_DIR=""
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "1" ]]; then
  echo "ERROR: confirmation requires a one-node Slurm allocation" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$CFG" || ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: confirmation Python/config/best.pt prerequisite is missing" >&2
  exit 1
fi
if [[ "$(basename -- "$CHECKPOINT")" == "best_infeasible.pt" ]]; then
  echo "ERROR: best_infeasible.pt can never enter confirmation" >&2
  exit 1
fi
if [[ ! -f "$AUTHORIZATION" || ! -f "$DEVELOPMENT_DIAGNOSTIC_DIR/READY" ]]; then
  echo "ERROR: confirmation requires authorization and locked dev diagnostic" >&2
  exit 1
fi
if [[ "$(realpath -e "$PARTITION_DIR")" != "$(realpath -e "$SEALED_PARTITION_DIR")" \
      || "$(sha256sum -- "$PARTITION_DIR/READY" | awk '{print $1}')" != "$EXPECTED_PARTITION_READY_SHA256" \
      || "$(sha256sum -- "$PARTITION_DIR/manifest_development.jsonl" | awk '{print $1}')" != "$EXPECTED_DEVELOPMENT_MANIFEST_SHA256" ]]; then
  echo "ERROR: confirmation partition is not the predeclared sealed artifact" >&2
  exit 1
fi
if [[ ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: sentence-memory bank is not READY" >&2
  exit 1
fi
if [[ "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_V2_SHA256" ]]; then
  echo "ERROR: original v2 checkpoint SHA256 mismatch" >&2
  exit 1
fi
if [[ -e "$OUT_ROOT" && ! -d "$OUT_ROOT" ]]; then
  echo "ERROR: confirmation output root exists but is not a directory" >&2
  exit 1
fi
if [[ -e "$HOLDOUT_SPEND" && ! -f "$HOLDOUT_SPEND" ]]; then
  echo "ERROR: confirmation-spend path exists but is not a regular file" >&2
  exit 1
fi
if [[ -d "$OUT_ROOT" && ! -f "$HOLDOUT_SPEND" ]]; then
  echo "ERROR: confirmation outputs exist without the immutable spend marker" >&2
  exit 1
fi

cd "$PROJECT_DIR"
if [[ ! -d "$PROJECT_DIR/.git" || -L "$PROJECT_DIR/.git" ]]; then
  echo "ERROR: confirmation requires the authorized shared standalone source clone" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: confirmation source clone must reside on /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/experiments" ]]; then
  echo "ERROR: confirmation clone must expose the durable experiments directory" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/experiments")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: confirmation experiments must resolve to /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/deps/mt5-base" ]]; then
  echo "ERROR: confirmation clone must expose frozen deps/mt5-base" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/deps/mt5-base")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: confirmation mT5 dependency must resolve to /media/cvpr" >&2; exit 1 ;;
esac
if [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: confirmation source uses external or node-local git metadata" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$SOURCE_GIT_HEAD" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: confirmation must use the locked clean source commit" >&2
  exit 1
fi
if [[ "$FACTORIZED_STAGE" == "stage1" ]]; then
  if [[ -e "$STAGE1_STAGE2_AUTH" || -e "$STAGE2_RUN_DIR" || -e "${STAGE2_RUN_DIR}.prerequisites" ]]; then
    echo "ERROR: Stage 1 confirmation is not first-feasible after Stage 2 activity" >&2
    exit 1
  fi
else
  if [[ ! -f "$STAGE1_STAGE2_AUTH" ]]; then
    echo "ERROR: Stage 2 lacks its Stage-1 valid-infeasible authorization" >&2
    exit 1
  fi
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY
unset SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256
unset SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS
unset SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY

"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
  verify --purpose confirmation --stage "$FACTORIZED_STAGE" \
  --authorization "$AUTHORIZATION"

readarray -t AUTH_SOURCE < <("$PYTHON_BIN" - "$AUTHORIZATION" <<'PY'
import json
import sys

authorization = json.load(open(sys.argv[1], encoding="utf-8"))
source = authorization["run_launch_identity"]["source"]
print(source["git_head"])
print(source["remote_ref"])
print(source["remote_head"])
print(source["repository_root"])
print("1" if source.get("standalone_shared_clone_checked") is True else "0")
PY
)
AUTH_GIT_HEAD="${AUTH_SOURCE[0]:-}"
AUTH_REMOTE_REF="${AUTH_SOURCE[1]:-}"
AUTH_REMOTE_HEAD="${AUTH_SOURCE[2]:-}"
AUTH_SOURCE_ROOT="${AUTH_SOURCE[3]:-}"
if [[ "${#AUTH_SOURCE[@]}" != "5" \
      || "${AUTH_GIT_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" \
      || "${AUTH_REMOTE_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" \
      || "$AUTH_REMOTE_REF" != origin/* \
      || "$(realpath -e "$AUTH_SOURCE_ROOT")" != "$(realpath -e "$PROJECT_DIR")" \
      || "${AUTH_SOURCE[4]:-}" != "1" ]]; then
  echo "ERROR: confirmation source does not match the authorized pushed source" >&2
  exit 1
fi
AUTH_REMOTE_BRANCH="${AUTH_REMOTE_REF#origin/}"
LIVE_REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$AUTH_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$LIVE_REMOTE_HEAD" || "${LIVE_REMOTE_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" ]]; then
  echo "ERROR: live authorized origin branch no longer resolves to the locked source" >&2
  exit 1
fi

# Validate every development-only prerequisite before irreversibly opening the
# confirmation holdout. This imports no confirmation manifest.
"$PYTHON_BIN" - "$AUTHORIZATION" "$CHECKPOINT" "$CFG" \
  "$DEVELOPMENT_DIAGNOSTIC_DIR" <<'PY'
import sys
from pathlib import Path

from NIAF.continuous_trajectory_field.scripts.analyze_factorized_memory_confirmation import (
    _authorization_evidence,
    _load_development_diagnostic,
)

authorization_path = Path(sys.argv[1])
checkpoint = Path(sys.argv[2]).resolve()
config = Path(sys.argv[3]).resolve()
_stage, authorization = _authorization_evidence(
    authorization_path,
    checkpoint_path=checkpoint,
    config_path=config,
)
_load_development_diagnostic(
    Path(sys.argv[4]),
    checkpoint_path=checkpoint,
    checkpoint_sha256=str(authorization["checkpoint"]["sha256"]),
    config_path=config,
    authorization_path=authorization_path,
    authorization=authorization,
)
print("validated locked development diagnostic before confirmation spend")
PY

LEASE_PATH="$CONTROL_DIR/active_confirmation_execution_lease"
LEASE_ATTESTATION="$CONTROL_DIR/confirmation_execution_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
  acquire-execution-lease \
  --purpose confirmation --stage "$FACTORIZED_STAGE" \
  --lease "$LEASE_PATH" --out_file "$LEASE_ATTESTATION" \
  --source_git_head "$SOURCE_GIT_HEAD" --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$(sha256sum -- "$AUTHORIZATION" | awk '{print $1}')"
LEASE_HELD=1

cleanup_confirmation() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$LOCAL_SENTENCE_MEMORY_DIR" ]]; then
    srun --nodes=1 --ntasks=1 bash \
      "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
      cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true
  fi
  if [[ "$exit_code" == "0" && "${LEASE_HELD:-0}" == "1" ]]; then
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      release-execution-lease --lease "$LEASE_PATH" \
      --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup_confirmation EXIT

# This exclusive marker is written before manifest_confirmation.jsonl is ever
# opened. It is intentionally retained on every later success or failure.
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
  spend-confirmation --stage "$FACTORIZED_STAGE" \
  --authorization "$AUTHORIZATION" \
  --marker "$HOLDOUT_SPEND" \
  --allow_matching_existing

# Resolve the authorized confirmation-manifest identity only after the
# irreversible spend marker exists. Factorized exports require this explicit
# identity and refuse arbitrary subset or online-retrieval fallback.
SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256="$($PYTHON_BIN - "$PARTITION_DIR" <<'PY'
import sys
from pathlib import Path

from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    load_confirmation_partition,
)

print(load_confirmation_partition(Path(sys.argv[1]))["manifest_sha256"])
PY
)"
if [[ ! "$SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: confirmation query-manifest identity is invalid" >&2
  exit 1
fi
export SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256

LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
export STAGE_ONLY_REQUESTED_NEIGHBORS=1
srun --nodes=1 --ntasks=1 bash \
  "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
  stage "$LOCAL_SENTENCE_MEMORY_DIR" "$SENTENCE_MEMORY_DIR" \
  "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"

mkdir -p "$OUT_ROOT"

validate_scored_mode() {
  local mode_name="$1"
  local mode_dir="$2"
  "$PYTHON_BIN" - "$mode_name" "$mode_dir" "$PARTITION_DIR" \
    "$CHECKPOINT" "$CFG" <<'PY'
import sys
from pathlib import Path

from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    load_confirmation_partition,
    load_mode_export,
)
from NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage import (
    EXPECTED_EVALUATION_CORRUPTION,
)

mode = sys.argv[1]
directory = Path(sys.argv[2])
partition = load_confirmation_partition(Path(sys.argv[3]))
checkpoint = Path(sys.argv[4]).resolve()
config = Path(sys.argv[5]).resolve()
expected_modes = {
    "text_only": "off",
    "sentence_memory": "on",
    "motion_shuffled_sentence_memory": "motion_shuffled",
    "shuffled_sentence_memory": "shuffled",
    "analytic_prior_sentence_memory": "analytic_prior",
}
load_mode_export(
    mode,
    directory,
    partition=partition,
    checkpoint_path=checkpoint,
    config_path=config,
    checkpoint_epoch=None,
    motion_shuffle_seed=1234,
    expected_memory_mode=expected_modes[mode],
    expected_evaluation_corruption=EXPECTED_EVALUATION_CORRUPTION,
)
print(f"validated complete confirmation unit: {mode}")
PY
}

validate_integrity_mode() {
  local label="$1"
  local mode_dir="$2"
  local checkpoint_path="$3"
  local config_path="$4"
  "$PYTHON_BIN" - "$label" "$mode_dir" "$PARTITION_DIR" \
    "$checkpoint_path" "$config_path" <<'PY'
import sys
from pathlib import Path

from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    load_confirmation_partition,
    load_integrity_export,
)

label = sys.argv[1]
partition = load_confirmation_partition(Path(sys.argv[3]))
load_integrity_export(
    label,
    Path(sys.argv[2]),
    partition=partition,
    checkpoint_path=Path(sys.argv[4]).resolve(),
    config_path=Path(sys.argv[5]).resolve(),
)
print(f"validated complete confirmation integrity unit: {label}")
PY
}

rebase_unit_paths() {
  local build_dir="$1"
  local final_dir="$2"
  shift 2
  "$PYTHON_BIN" - "$build_dir" "$final_dir" "$@" <<'PY'
import os
import sys
from pathlib import Path

build = Path(sys.argv[1]).resolve()
final = Path(sys.argv[2]).resolve()
old = str(build)
new = str(final)
for filename in sys.argv[3:]:
    path = build / filename
    if not path.is_file():
        raise SystemExit(f"missing unit artifact before finalization: {path}")
    text = path.read_text(encoding="utf-8")
    replaced = text.replace(old, new)
    temporary = path.with_name(f".{path.name}.rebasing.{os.getpid()}")
    temporary.write_text(replaced, encoding="utf-8")
    os.replace(temporary, path)
PY
}

publish_unit_no_replace() {
  local build_dir="$1"
  local final_dir="$2"
  "$PYTHON_BIN" - "$build_dir" "$final_dir" <<'PY'
import os
import sys
from pathlib import Path

building = Path(sys.argv[1])
final = Path(sys.argv[2])
if not building.is_dir() or final.exists():
    raise SystemExit("refusing to clobber or nest a confirmation unit")
os.rename(building, final)
PY
}

run_scored_mode() {
  local mode="$1"
  local label="$2"
  local mode_dir="$OUT_ROOT/$label"
  if [[ -e "$mode_dir" && ! -d "$mode_dir" ]]; then
    echo "ERROR: completed-mode path is not a directory: $mode_dir" >&2
    exit 1
  fi
  if [[ -d "$mode_dir" ]]; then
    validate_scored_mode "$label" "$mode_dir"
    echo "Reusing provenance-complete confirmation unit: $label"
    return
  fi

  local build_dir="$OUT_ROOT/.${label}.building.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
  if [[ -e "$build_dir" ]]; then
    echo "ERROR: this job's mode build directory already exists: $build_dir" >&2
    exit 1
  fi
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
    --config "$CFG" \
    --checkpoint "$CHECKPOINT" \
    --split val \
    --manifest "$PARTITION_DIR/manifest_confirmation.jsonl" \
    --num_samples 0 \
    --selection_mode first \
    --out_dir "$build_dir" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --text_device cpu \
    --length_mode predicted \
    --word_prior off \
    --sentence_memory "$mode" \
    --context_fps 20 \
    --sample_fps 20

  local alignment
  for alignment in default pa; do
    srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
      flow.evaluate.dtw_mpjpe_t2m_default \
      --samples_dir "$build_dir" \
      --out_json "$build_dir/dtw_mpjpe_t2m_${alignment}_h2s_betas.json" \
      --out_csv "$build_dir/dtw_mpjpe_t2m_${alignment}_h2s_betas.csv" \
      --sample_key smplx \
      --gt_key smplx \
      --prior_key adapter_context_smplx \
      --device cuda \
      --betas_mode h2s_fixed \
      --alignment_mode "$alignment" \
      --parts body lhand rhand face wholebody
  done
  validate_scored_mode "$label" "$build_dir"
  rebase_unit_paths "$build_dir" "$mode_dir" \
    export_summary.json \
    sample_manifest_summary.json \
    dtw_mpjpe_t2m_default_h2s_betas.json \
    dtw_mpjpe_t2m_default_h2s_betas.csv \
    dtw_mpjpe_t2m_pa_h2s_betas.json \
    dtw_mpjpe_t2m_pa_h2s_betas.csv
  publish_unit_no_replace "$build_dir" "$mode_dir"
  validate_scored_mode "$label" "$mode_dir"
}

run_integrity_mode() {
  local label="$1"
  local memory_mode="$2"
  local config_path="$3"
  local checkpoint_path="$4"
  local directory_label="${5:-$label}"
  local mode_dir="$OUT_ROOT/$directory_label"
  if [[ -e "$mode_dir" && ! -d "$mode_dir" ]]; then
    echo "ERROR: integrity-mode path is not a directory: $mode_dir" >&2
    exit 1
  fi
  if [[ -d "$mode_dir" ]]; then
    validate_integrity_mode "$label" "$mode_dir" "$checkpoint_path" "$config_path"
    echo "Reusing provenance-complete confirmation unit: $label"
    return
  fi

  local build_dir="$OUT_ROOT/.${directory_label}.building.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
  if [[ -e "$build_dir" ]]; then
    echo "ERROR: this job's integrity build directory already exists: $build_dir" >&2
    exit 1
  fi
  local memory_args=()
  if [[ -n "$memory_mode" ]]; then
    memory_args=(--sentence_memory "$memory_mode")
  fi
  srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
    --config "$config_path" \
    --checkpoint "$checkpoint_path" \
    --split val \
    --manifest "$PARTITION_DIR/manifest_confirmation.jsonl" \
    --num_samples 0 \
    --selection_mode first \
    --out_dir "$build_dir" \
    --batch_size "$BATCH_SIZE" \
    --device cuda --text_device cpu --length_mode predicted \
    --word_prior off "${memory_args[@]}" \
    --context_fps 20 --sample_fps 20
  validate_integrity_mode "$label" "$build_dir" "$checkpoint_path" "$config_path"
  rebase_unit_paths "$build_dir" "$mode_dir" \
    export_summary.json sample_manifest_summary.json
  publish_unit_no_replace "$build_dir" "$mode_dir"
  validate_integrity_mode "$label" "$mode_dir" "$checkpoint_path" "$config_path"
}

run_scored_mode off text_only
run_scored_mode on sentence_memory
run_scored_mode motion_shuffled motion_shuffled_sentence_memory
run_scored_mode shuffled shuffled_sentence_memory
run_scored_mode analytic_prior analytic_prior_sentence_memory
run_integrity_mode all_null all_null "$CFG" "$CHECKPOINT"
run_integrity_mode v2_text_only "" "$V2_CFG" "$V2_CHECKPOINT" original_v2_text_only

srun --kill-on-bad-exit=1 "$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.analyze_factorized_memory_confirmation \
  --authorization "$AUTHORIZATION" \
  --holdout_spend "$HOLDOUT_SPEND" \
  --checkpoint "$CHECKPOINT" \
  --config "$CFG" \
  --v2_checkpoint "$V2_CHECKPOINT" \
  --v2_config "$V2_CFG" \
  --partition_dir "$PARTITION_DIR" \
  --development_diagnostic_dir "$DEVELOPMENT_DIAGNOSTIC_DIR" \
  --text_only_dir "$OUT_ROOT/text_only" \
  --sentence_memory_dir "$OUT_ROOT/sentence_memory" \
  --motion_shuffled_sentence_memory_dir "$OUT_ROOT/motion_shuffled_sentence_memory" \
  --shuffled_sentence_memory_dir "$OUT_ROOT/shuffled_sentence_memory" \
  --analytic_prior_sentence_memory_dir "$OUT_ROOT/analytic_prior_sentence_memory" \
  --all_null_dir "$OUT_ROOT/all_null" \
  --v2_text_only_dir "$OUT_ROOT/original_v2_text_only" \
  --out_dir "$OUT_ROOT/analysis" \
  --bootstrap_samples 10000 \
  --seed 1234
