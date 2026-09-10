#!/bin/bash
#SBATCH --job-name=csl_stage_c_protocol_v5_run_r6_cpu
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# Repository-wide CPU/ruff gate for the exact immutable Stage-C source.
set -euo pipefail
trap 'echo "ERROR: Stage-C CPU gate failed at line $LINENO with exit code $?" >&2' ERR

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
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
UV_BIN="${UV_BIN:-/media/cvpr/haomian/python_envs/slt/bin/uv}"
STAGE_C_LINT_BASE="97f785e3bc6bf48f3c2c686e3365215f4e9c5cb0"
GLOBAL_HOLDOUT_SPEND="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json"
GATE_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_prerequisites/source_${SOURCE_GIT_HEAD,,}/cpu_gate"
PREREQUISITE_MODULE="NIAF.continuous_trajectory_field.scripts.stage_c_pilot_prerequisites"
RECOVERY_POLICY="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_decision_policy_v1.json"
RECOVERY_MANIFEST="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_recovery_evidence_v1.json"
RECOVERY_EVIDENCE_ROOT="/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory"
INCIDENT_ARCHIVE="$RECOVERY_EVIDENCE_ROOT/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v4_run_r5_smoke.invalid_attempts/source_36d121dd361df2c08616031bd7c2076d85da78d0_smoke143609_pilot143610/ARCHIVE.json"
MEMORY_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v5_run_r6.yaml"
OFF_CFG="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v5_run_r6.yaml"

[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-0}" == "1" ]] || {
  echo "ERROR: Stage-C CPU gate requires one Slurm node" >&2; exit 1;
}
[[ "$SOURCE_REMOTE_BRANCH" == "$EXPECTED_SOURCE_REMOTE_BRANCH" \
   && ! -e "$GATE_DIR" ]] || {
  echo "ERROR: invalid source identity or immutable CPU-gate path already exists" >&2; exit 1;
}
[[ ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: confirmation holdout is already spent" >&2; exit 1;
}
cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git \
      || "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" \
      || "$(git rev-parse HEAD)" != "${SOURCE_GIT_HEAD,,}" \
      || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: CPU gate requires the exact clean standalone source" >&2
  exit 1
fi
REMOTE_HEAD="$(timeout 120s git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
[[ -n "$REMOTE_HEAD" && "$REMOTE_HEAD" == "${SOURCE_GIT_HEAD,,}" ]] || {
  echo "ERROR: CPU-gate source is not the exact pushed branch head" >&2; exit 1;
}

for required in "$RECOVERY_POLICY" "$RECOVERY_MANIFEST" "$INCIDENT_ARCHIVE"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    echo "ERROR: missing Stage-C protocol-v5/run-r6 recovery evidence: $required" >&2
    exit 1
  }
done

export PATH="$PYTHON_ENV/bin:$PATH" PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
export UV_CACHE_DIR="/tmp/signtraj_stage_c_cpu_uv_${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
unset WANDB_API_KEY
CONFIG_AUDIT="$("$PYTHON_BIN" -m "$PREREQUISITE_MODULE" validate-protocol-v5-configs --policy_path "$RECOVERY_POLICY" --recovery_manifest "$RECOVERY_MANIFEST" --source_root "$PROJECT_DIR" --evidence_root "$RECOVERY_EVIDENCE_ROOT" --memory_config "$MEMORY_CFG" --matched_off_config "$OFF_CFG")"
[[ -x "$PYTHON_BIN" && -x "$UV_BIN" && "$($UV_BIN --version)" == "uv 0.10.0" ]] || {
  echo "ERROR: pinned Python/uv CPU-gate tools are unavailable" >&2; exit 1;
}
TEST_PYTHON=("$UV_BIN" run --python "$PYTHON_BIN" --no-project --with pytest==8.4.2 python)
"${TEST_PYTHON[@]}" - <<'PY'
import pytest
import torch
if torch.cuda.is_available():
    raise SystemExit("ERROR: Stage-C CPU gate unexpectedly sees CUDA")
print(f"pytest={pytest.__version__} torch={torch.__version__}")
PY
"${TEST_PYTHON[@]}" -m compileall -q NIAF flow tests
git cat-file -e "${STAGE_C_LINT_BASE}^{commit}"
git merge-base --is-ancestor "$STAGE_C_LINT_BASE" "$SOURCE_GIT_HEAD"
mapfile -t STAGE_C_RUFF_FILES < <(
  git diff --name-only --diff-filter=ACM "$STAGE_C_LINT_BASE" "$SOURCE_GIT_HEAD" -- '*.py'
)
[[ "${#STAGE_C_RUFF_FILES[@]}" -gt 0 ]] || {
  echo "ERROR: Stage-C lint scope is unexpectedly empty" >&2; exit 1;
}
for file in "${STAGE_C_RUFF_FILES[@]}"; do
  [[ -f "$file" && ! -L "$file" ]] || {
    echo "ERROR: invalid Stage-C Ruff input: $file" >&2; exit 1;
  }
done
"$UV_BIN" tool run --from ruff==0.12.0 ruff check "${STAGE_C_RUFF_FILES[@]}"
"${TEST_PYTHON[@]}" -m pytest -q tests/test_*.py
[[ -z "$(git status --porcelain --untracked-files=all)" && ! -e "$GLOBAL_HOLDOUT_SPEND" ]] || {
  echo "ERROR: source or confirmation state changed during CPU gate" >&2; exit 1;
}

mkdir -p -- "$(dirname "$GATE_DIR")"
GATE_BUILDING="${GATE_DIR}.building.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
[[ ! -e "$GATE_BUILDING" ]] || {
  echo "ERROR: current CPU-gate publication generation already exists" >&2
  exit 1
}
mkdir -- "$GATE_BUILDING"
"$PYTHON_BIN" - "$GATE_BUILDING/READY" "$SOURCE_GIT_HEAD" \
  "$SOURCE_REMOTE_BRANCH" "$REMOTE_HEAD" "$SLURM_JOB_ID" "$CONFIG_AUDIT" \
  "$RECOVERY_POLICY" "$RECOVERY_MANIFEST" "$INCIDENT_ARCHIVE" \
  "$MEMORY_CFG" "$OFF_CFG" <<'PY'
import hashlib
import json, os, sys
from pathlib import Path
config_audit = json.loads(sys.argv[6])
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema_name": "signtrajfield_stage_c_cpu_gate",
    "schema_version": 2,
    "source_git_head": sys.argv[2].lower(),
    "source_remote_ref": f"origin/{sys.argv[3]}",
    "source_remote_head": sys.argv[4].lower(),
    "slurm_job_id": sys.argv[5],
    "compileall": True,
    "ruff_version": "0.12.0",
    "pytest_version": "8.4.2",
    "complete_repository_test_glob": "tests/test_*.py",
    "development_only": True,
    "confirmation_manifest_opened": False,
    "test_data_accessed": False,
    "passed": True,
    "recovery_audit_identity": config_audit["recovery_audit_identity"],
    "config_audit_identity": config_audit["audit_identity"],
    "incident_archive_sha256": sha256(sys.argv[9]),
    "decision_policy_sha256": sha256(sys.argv[7]),
    "recovery_manifest_sha256": sha256(sys.argv[8]),
    "memory_config_sha256": sha256(sys.argv[10]),
    "matched_off_config_sha256": sha256(sys.argv[11]),
    "historical_clone_metadata_checked_in_cpu_gate": True,
    "historical_clone_local_git_metadata_timeout_seconds": 600,
    "historical_remote_ref_timeout_seconds": 120,
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
"$PYTHON_BIN" - "$GATE_BUILDING" "$GATE_DIR" <<'PY'
import sys
from pathlib import Path
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    AtomicPublishError,
    rename_no_replace,
)
try:
    rename_no_replace(Path(sys.argv[1]), Path(sys.argv[2]))
except (AtomicPublishError, OSError) as error:
    raise SystemExit(f"ERROR: atomic CPU-gate publication failed: {error}") from error
PY
echo "STAGE_C_CPU_GATE_COMPLETE=$GATE_DIR/READY"
