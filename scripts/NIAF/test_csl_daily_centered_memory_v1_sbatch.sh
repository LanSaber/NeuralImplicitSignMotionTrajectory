#!/bin/bash
#SBATCH --job-name=csl_centered_cpu
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err

# Read-only static checks and the complete repository CPU test suite.
set -euo pipefail
trap 'echo "ERROR: centered static/CPU gate failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the shared standalone source clone}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the clean pushed commit}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
# Compute-node home directories are node-local.  Use the shared ARM64 uv
# binary so scheduler-selected nodes all execute the same disposable overlay.
UV_BIN="${UV_BIN:-/media/cvpr/haomian/python_envs/slt/bin/uv}"

[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-0}" == "1" ]] || {
  echo "ERROR: CPU gate requires one Slurm node" >&2; exit 1;
}
cd "$PROJECT_DIR"
if [[ ! -d .git || -L .git \
      || "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" \
      || "$(git rev-parse HEAD)" != "$SOURCE_GIT_HEAD" \
      || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: CPU gate requires the exact clean standalone source" >&2
  exit 1
fi
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
[[ -n "$REMOTE_HEAD" && "${REMOTE_HEAD,,}" == "${SOURCE_GIT_HEAD,,}" ]] || {
  echo "ERROR: CPU-gate source is not the pushed branch head" >&2; exit 1;
}

export PATH="$PYTHON_ENV/bin:$PATH" PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

TEST_PYTHON=("$PYTHON_BIN")
if ! "$PYTHON_BIN" -c 'import pytest, ruff' >/dev/null 2>&1; then
  [[ -x "$UV_BIN" ]] || {
    echo "ERROR: SOKE lacks test tools and no executable UV_BIN was provided" >&2
    exit 1
  }
  [[ "$($UV_BIN --version)" == "uv 0.10.0" ]] || {
    echo "ERROR: CPU-gate uv version changed" >&2
    exit 1
  }
  # The overlay is disposable and leaves the shared production environment
  # untouched. Both test tools are version-pinned in the launch evidence.
  TEST_PYTHON=(
    "$UV_BIN" run --python "$PYTHON_BIN" --no-project
    --with pytest==8.4.2 --with ruff==0.12.0 python
  )
fi

"${TEST_PYTHON[@]}" - <<'PY'
import pytest
import torch

if torch.cuda.is_available():
    raise SystemExit("ERROR: CPU gate unexpectedly sees CUDA")
print(f"pytest={pytest.__version__} torch={torch.__version__}")
PY
"${TEST_PYTHON[@]}" -m compileall -q NIAF flow tests
"${TEST_PYTHON[@]}" -m ruff check NIAF flow tests
# The explicit shell expansion is intentional: it proves tests/test_*.py, not
# an accidentally narrower historical test_niaf_* subset, was requested.
"${TEST_PYTHON[@]}" -m pytest -q tests/test_*.py
