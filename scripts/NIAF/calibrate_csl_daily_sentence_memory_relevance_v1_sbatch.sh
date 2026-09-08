#!/bin/bash
#SBATCH --job-name=csl-rel-cal-v1
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# Train-only relevance calibration prerequisite for ordered Phase-A'''.
set -euo pipefail
trap 'echo "ERROR: relevance calibration failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the clean pushed commit}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_BANK="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
CALIBRATION_OUT="${CALIBRATION_OUT:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_v1}"
CONTROL_DIR="${CONTROL_DIR:-$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_ordered_v1_control}"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"
AUDIT_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1.yaml"
LAUNCHER="$PROJECT_DIR/scripts/NIAF/calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh"
STAGER="$PROJECT_DIR/scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
EXPECTED_BANK_ID="a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
EXPECTED_TRAIN_SHA256="d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "1" ]]; then
  echo "ERROR: calibration requires one Slurm node" >&2
  exit 1
fi
if [[ ! "$SOURCE_GIT_HEAD" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: SOURCE_GIT_HEAD must be a full Git SHA1" >&2
  exit 1
fi
for required in "$PYTHON_BIN" "$AUDIT_CFG" "$LAUNCHER" "$STAGER"; do
  [[ -e "$required" ]] || { echo "ERROR: prerequisite is absent: $required" >&2; exit 1; }
done
EXISTING_ACCEPTED=0
if [[ -e "$CALIBRATION_OUT" ]]; then
  if [[ -f "$CALIBRATION_OUT/READY" && ! -e "$CALIBRATION_OUT/REJECTED" ]]; then
    EXISTING_ACCEPTED=1
  else
    echo "ERROR: existing rejected/incomplete calibration is terminal and immutable" >&2
    exit 1
  fi
fi
[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: global confirmation holdout is already spent" >&2
  exit 1
}

cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git ]]; then
  echo "ERROR: PROJECT_DIR must be a shared standalone clone" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: PROJECT_DIR must resolve below /media/cvpr" >&2; exit 1 ;;
esac
GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
if [[ "$(realpath -e "$GIT_COMMON_DIR")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: PROJECT_DIR uses external worktree metadata" >&2
  exit 1
fi
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "${ACTUAL_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" \
      || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: calibration requires the exact clean SOURCE_GIT_HEAD" >&2
  exit 1
fi
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
[[ -n "$SOURCE_REMOTE_BRANCH" ]] || {
  echo "ERROR: detached HEAD requires SOURCE_REMOTE_BRANCH" >&2
  exit 1
}
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$REMOTE_HEAD" || "${REMOTE_HEAD,,}" != "${ACTUAL_HEAD,,}" ]]; then
  echo "ERROR: source commit is not the live pushed origin branch head" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

mkdir -p -- "$CONTROL_DIR/calibration_execution_attempts"
LEASE="$CONTROL_DIR/active_calibration_execution_lease"
LEASE_ATTESTATION="$CONTROL_DIR/calibration_execution_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
COMPLETION_ATTESTATION="$CONTROL_DIR/calibration_execution_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.completion.json"
BINDING_IDENTITY="$(sha256sum -- "$LAUNCHER" | awk '{print $1}')"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage \
  acquire-execution-lease \
  --purpose calibration --stage calibration \
  --lease "$LEASE" --out_file "$LEASE_ATTESTATION" \
  --source_git_head "$SOURCE_GIT_HEAD" \
  --slurm_job_id "$SLURM_JOB_ID" \
  --binding_identity "$BINDING_IDENTITY"

STAGED_BANK="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
cleanup_on_success() {
  local exit_code=$?
  trap - EXIT
  "$STAGER" cleanup "$STAGED_BANK" || true
  if [[ "$exit_code" == "0" ]]; then
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage \
      release-execution-lease \
      --lease "$LEASE" --attestation "$LEASE_ATTESTATION" || exit_code=$?
  fi
  exit "$exit_code"
}
trap cleanup_on_success EXIT

COMPLETION_ARGS=(
  record-calibration-completion
  --artifact_dir "$CALIBRATION_OUT"
  --source_root "$PROJECT_DIR"
  --source_git_head "$SOURCE_GIT_HEAD"
  --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH"
  --source_remote_head "$REMOTE_HEAD"
  --lease "$LEASE"
  --lease_attestation "$LEASE_ATTESTATION"
  --out_file "$COMPLETION_ATTESTATION"
)
if [[ "$EXISTING_ACCEPTED" == "1" ]]; then
  # This also closes the publish-before-release crash window: acquisition above
  # can archive only a scheduler-proven stale claim, while completion validates
  # the immutable artifact and records recovery outside the artifact directory.
  "$PYTHON_BIN" -m \
    NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage \
    "${COMPLETION_ARGS[@]}" --publication_existing
  exit 0
fi

"$STAGER" stage "$STAGED_BANK" "$SOURCE_BANK" "$PROJECT_DIR" \
  "$PYTHON_BIN" "$AUDIT_CFG" "train"

# Prove the exact visibility contract again immediately before the CLI.  These
# probes address only the node-local target and never synthesize a shared path.
[[ -f "$STAGED_BANK/neighbors_train.npz" ]] || {
  echo "ERROR: local train neighbors are absent" >&2
  exit 1
}
[[ ! -e "$STAGED_BANK/neighbors_test.npz" ]] || {
  echo "ERROR: local test neighbors are forbidden" >&2
  exit 1
}
if [[ "$(sha256sum -- "$STAGED_BANK/neighbors_train.npz" | awk '{print $1}')" != "$EXPECTED_TRAIN_SHA256" ]]; then
  echo "ERROR: local train-neighbor identity changed" >&2
  exit 1
fi

"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.calibrate_sentence_memory_relevance \
  --bank_dir "$STAGED_BANK" \
  --train_neighbors "$STAGED_BANK/neighbors_train.npz" \
  --out_dir "$CALIBRATION_OUT" \
  --source_root "$PROJECT_DIR" \
  --source_git_head "$SOURCE_GIT_HEAD" \
  --source_remote_ref "origin/$SOURCE_REMOTE_BRANCH" \
  --source_remote_head "$REMOTE_HEAD" \
  --launcher "$LAUNCHER" \
  --seed 1234 --duration_weight 0.05 \
  --minimum_auroc 0.75 --minimum_probability_gap 0.20 \
  --expected_bank_id "$EXPECTED_BANK_ID" \
  --expected_train_neighbor_sha256 "$EXPECTED_TRAIN_SHA256"

"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage \
  "${COMPLETION_ARGS[@]}"
