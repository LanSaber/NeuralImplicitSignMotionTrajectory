#!/bin/bash
#SBATCH --job-name=signtraj_slot_diag
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
trap 'echo "ERROR: diagnose_temporal_slots_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR_WAS_EXPLICIT="${PROJECT_DIR+x}"
PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
DIAGNOSTIC_PROFILE="${DIAGNOSTIC_PROFILE:-dw005_epoch2}"
CONFIRMATION_AUTHORIZATION="${CONFIRMATION_AUTHORIZATION:-}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
PROFILE_FACTORIZED_STAGE=""
case "$DIAGNOSTIC_PROFILE" in
  dw005_epoch2)
    PROFILE_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2"
    PROFILE_CHECKPOINT="epoch0002.pt"
    PROFILE_EVALUATION="epoch0002_validation_slot_diagnostics"
    ;;
  phase_a_motion_contrast_v1)
    PROFILE_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1"
    PROFILE_CHECKPOINT="best.pt"
    PROFILE_EVALUATION="locked_validation_slot_diagnostics"
    ;;
  factorized_stage1_v1)
    PROFILE_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
    PROFILE_CHECKPOINT="best.pt"
    PROFILE_EVALUATION="locked_development_slot_diagnostics"
    PROFILE_FACTORIZED_STAGE="stage1"
    ;;
  factorized_stage2_v1)
    PROFILE_EXPERIMENT="csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1"
    PROFILE_CHECKPOINT="best.pt"
    PROFILE_EVALUATION="locked_development_slot_diagnostics"
    PROFILE_FACTORIZED_STAGE="stage2"
    ;;
  *)
    echo "ERROR: unsupported DIAGNOSTIC_PROFILE=$DIAGNOSTIC_PROFILE" >&2
    exit 1
    ;;
esac
CFG="${CFG:-$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${PROFILE_EXPERIMENT}.yaml}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/${PROFILE_EXPERIMENT}/checkpoints/${PROFILE_CHECKPOINT}}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/${PROFILE_EXPERIMENT}/evaluation/${PROFILE_EVALUATION}}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${OUT_DIR}_smoke}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PERTURB_BATCH_SIZE="${PERTURB_BATCH_SIZE:-128}"
FK_BATCH_SIZE="${FK_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
VERIFY_HASHES="${VERIFY_HASHES:-1}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:-${SIGNTRAJ_SOURCE_GIT_HEAD:-}}"
LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID:?Submit this launcher through sbatch}"

for flag_name in RUN_SMOKE RUN_FULL VERIFY_HASHES; do
  flag_value="${!flag_name}"
  case "$flag_value" in
    0|1) ;;
    *) echo "ERROR: $flag_name must be 0 or 1" >&2; exit 1 ;;
  esac
done
if [[ "$RUN_SMOKE" == "0" && "$RUN_FULL" == "0" ]]; then
  echo "ERROR: at least one of RUN_SMOKE or RUN_FULL must be enabled" >&2
  exit 1
fi
if [[ ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ && ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: SOURCE_GIT_HEAD must be the submit-host git rev-parse HEAD value" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: configuration not found: $CFG" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ -n "$PROFILE_FACTORIZED_STAGE" ]]; then
  if [[ "$PROJECT_DIR_WAS_EXPLICIT" != "x" ]]; then
    echo "ERROR: factorized diagnostics require an explicit PROJECT_DIR shared clone" >&2
    exit 1
  fi
  EXPECTED_AUTHORIZATION="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$PROFILE_EXPERIMENT/evaluation/ordered_development_decision/authorize_confirmation.json"
  if [[ -z "$CONFIRMATION_AUTHORIZATION" ]]; then
    echo "ERROR: factorized diagnostics require explicit CONFIRMATION_AUTHORIZATION" >&2
    exit 1
  fi
  if [[ ! -f "$CONFIRMATION_AUTHORIZATION" ]]; then
    echo "ERROR: confirmation authorization is missing: $CONFIRMATION_AUTHORIZATION" >&2
    exit 1
  fi
  if [[ "$(realpath "$CONFIRMATION_AUTHORIZATION")" != "$(realpath "$EXPECTED_AUTHORIZATION")" ]]; then
    echo "ERROR: confirmation authorization must be exactly $EXPECTED_AUTHORIZATION" >&2
    exit 1
  fi
  if [[ "$(basename "$CHECKPOINT")" != "best.pt" || -L "$CHECKPOINT" ]]; then
    echo "ERROR: factorized diagnostic refuses best_infeasible.pt and checkpoint aliases" >&2
    exit 1
  fi
  cd "$PROJECT_DIR"
  if [[ ! -d "$PROJECT_DIR/.git" || -L "$PROJECT_DIR/.git" ]]; then
    echo "ERROR: factorized diagnostics require the authorized shared standalone source clone" >&2
    exit 1
  fi
  case "$(realpath -e "$PROJECT_DIR")" in
    /media/cvpr/*) ;;
    *) echo "ERROR: factorized diagnostic source clone must reside on /media/cvpr" >&2; exit 1 ;;
  esac
  if [[ ! -d "$PROJECT_DIR/experiments" ]]; then
    echo "ERROR: factorized diagnostic clone must expose durable experiments" >&2
    exit 1
  fi
  case "$(realpath -e "$PROJECT_DIR/experiments")" in
    /media/cvpr/*) ;;
    *) echo "ERROR: factorized diagnostic experiments must resolve to /media/cvpr" >&2; exit 1 ;;
  esac
  if [[ ! -d "$PROJECT_DIR/deps/mt5-base" ]]; then
    echo "ERROR: factorized diagnostic clone must expose frozen deps/mt5-base" >&2
    exit 1
  fi
  case "$(realpath -e "$PROJECT_DIR/deps/mt5-base")" in
    /media/cvpr/*) ;;
    *) echo "ERROR: factorized diagnostic mT5 dependency must resolve to /media/cvpr" >&2; exit 1 ;;
  esac
  if [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
    echo "ERROR: factorized diagnostic source uses external or node-local git metadata" >&2
    exit 1
  fi
  ACTUAL_SOURCE_HEAD="$(git rev-parse HEAD)"
  if [[ "${ACTUAL_SOURCE_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: factorized diagnostic source is not the requested clean commit" >&2
    exit 1
  fi
  readarray -t AUTH_SOURCE < <("$PYTHON_BIN" - "$CONFIRMATION_AUTHORIZATION" <<'PY'
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
    echo "ERROR: factorized diagnostic source differs from its authorization" >&2
    exit 1
  fi
  AUTH_REMOTE_BRANCH="${AUTH_REMOTE_REF#origin/}"
  LIVE_REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$AUTH_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
  if [[ -z "$LIVE_REMOTE_HEAD" || "${LIVE_REMOTE_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" ]]; then
    echo "ERROR: live authorized origin branch differs from diagnostic source" >&2
    exit 1
  fi
  PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
    verify \
    --purpose confirmation \
    --stage "$PROFILE_FACTORIZED_STAGE" \
    --authorization "$CONFIRMATION_AUTHORIZATION"
fi
if [[ ! -d "$SENTENCE_MEMORY_DIR" || ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: sentence-memory bank is not ready: $SENTENCE_MEMORY_DIR" >&2
  exit 1
fi

LEASE_HELD=0
if [[ -n "$PROFILE_FACTORIZED_STAGE" ]]; then
  LEASE_PATH="${OUT_DIR}.active_diagnostic_execution_lease"
  LEASE_ATTESTATION="${OUT_DIR}.diagnostic_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
  PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
    acquire-execution-lease \
    --purpose diagnostic --stage "$PROFILE_FACTORIZED_STAGE" \
    --lease "$LEASE_PATH" --out_file "$LEASE_ATTESTATION" \
    --source_git_head "$SOURCE_GIT_HEAD" --slurm_job_id "$SLURM_JOB_ID" \
    --binding_identity "$(sha256sum -- "$CONFIRMATION_AUTHORIZATION" | awk '{print $1}')"
  LEASE_HELD=1
fi

cleanup_sentence_memory() {
  local exit_code=$?
  trap - EXIT
  srun --nodes=1 --ntasks=1 \
    bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
    cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true
  if [[ "$exit_code" == "0" && "$LEASE_HELD" == "1" ]]; then
    PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage \
      release-execution-lease --lease "$LEASE_PATH" \
      --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup_sentence_memory EXIT

mkdir -p "$PROJECT_DIR/logs/sbatch" "$(dirname "$OUT_DIR")"
export STAGE_ONLY_REQUESTED_NEIGHBORS=1
srun --nodes=1 --ntasks=1 \
  bash "$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_node.sh" \
  stage "$LOCAL_SENTENCE_MEMORY_DIR" "$SENTENCE_MEMORY_DIR" \
  "$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SIGNTRAJ_SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD,,}"

# This is a local scientific audit, never an externally logged run. Remove any
# inherited credential variable as an additional guard; the Python entry point
# independently rejects online W&B mode.
unset WANDB_API_KEY
export WANDB_MODE=disabled
export WANDB_DISABLED=true

cd "$PROJECT_DIR"

run_stage() {
  local stage="$1"
  local destination="$2"

  local command=(
    srun --kill-on-bad-exit=1
    "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.diagnose_temporal_slots
    --profile "$DIAGNOSTIC_PROFILE"
    --config "$CFG"
    --checkpoint "$CHECKPOINT"
    --out_dir "$destination"
    --stage "$stage"
    --batch_size "$BATCH_SIZE"
    --perturb_batch_size "$PERTURB_BATCH_SIZE"
    --fk_batch_size "$FK_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device cuda
    --text_device cpu
    --resume
  )
  if [[ -n "$PROFILE_FACTORIZED_STAGE" ]]; then
    command+=(--authorization "$CONFIRMATION_AUTHORIZATION")
  fi
  if [[ "$VERIFY_HASHES" == "1" ]]; then
    command+=(--verify_hashes)
  fi
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  "${command[@]}"
}

echo "Job ID: $SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown}"
echo "Diagnostic profile: $DIAGNOSTIC_PROFILE"
echo "Checkpoint: $CHECKPOINT"
if [[ -n "$PROFILE_FACTORIZED_STAGE" ]]; then
  echo "Authorized factorized stage: $PROFILE_FACTORIZED_STAGE"
  echo "Development-only authorization: $CONFIRMATION_AUTHORIZATION"
fi
echo "Validation-only output: $OUT_DIR"
echo "Sentence-memory bank: $SENTENCE_MEMORY_DIR (staged to node-local storage)"
echo "W&B: disabled"

if [[ "$RUN_SMOKE" == "1" ]]; then
  run_stage smoke "$SMOKE_OUT_DIR"
fi
if [[ "$RUN_FULL" == "1" ]]; then
  run_stage full "$OUT_DIR"
fi
