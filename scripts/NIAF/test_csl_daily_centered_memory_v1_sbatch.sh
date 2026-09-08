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
CENTERED_LINT_BASE="96fc62aa12e1c6b690a564dd7f97bbd9670006d0"

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
export UV_CACHE_DIR="/tmp/signtraj_centered_cpu_uv_${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
unset WANDB_API_KEY

TEST_PYTHON=("$PYTHON_BIN")
[[ -x "$UV_BIN" ]] || {
  echo "ERROR: no executable shared UV_BIN was provided" >&2
  exit 1
}
[[ "$($UV_BIN --version)" == "uv 0.10.0" ]] || {
  echo "ERROR: CPU-gate uv version changed" >&2
  exit 1
}
if ! "$PYTHON_BIN" -c 'import pytest' >/dev/null 2>&1; then
  # The overlay is disposable and leaves the shared production environment
  # untouched. Pytest is version-pinned in the launch evidence.
  TEST_PYTHON=(
    "$UV_BIN" run --python "$PYTHON_BIN" --no-project
    --with pytest==8.4.2 python
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
# Invoke Ruff as its own pinned uv tool. `python -m ruff` from an ephemeral
# multi-package overlay can retain a console-script path after uv removes the
# temporary build directory, which caused invalid gate attempt 143292.
# Lint the complete experiment diff against its immutable pre-implementation
# base. Repository-wide lint is not a clean baseline (invalid attempt 143293
# found 23 pre-existing F401/F403/F405 findings in untouched legacy modules),
# while compileall and the complete test suite below remain repository-wide.
git cat-file -e "${CENTERED_LINT_BASE}^{commit}"
git merge-base --is-ancestor "$CENTERED_LINT_BASE" "$SOURCE_GIT_HEAD"
mapfile -t CENTERED_RUFF_FILES < <(
  git diff --name-only --diff-filter=ACM "$CENTERED_LINT_BASE" "$SOURCE_GIT_HEAD" -- '*.py'
)
[[ "${#CENTERED_RUFF_FILES[@]}" -gt 0 ]] || {
  echo "ERROR: centered implementation lint scope is unexpectedly empty" >&2
  exit 1
}
for file in "${CENTERED_RUFF_FILES[@]}"; do
  [[ -f "$file" && ! -L "$file" ]] || {
    echo "ERROR: centered Ruff input is missing or a symlink: $file" >&2
    exit 1
  }
done
"$UV_BIN" tool run --from ruff==0.12.0 ruff check "${CENTERED_RUFF_FILES[@]}"
# The explicit shell expansion is intentional: it proves tests/test_*.py, not
# an accidentally narrower historical test_niaf_* subset, was requested.
"${TEST_PYTHON[@]}" -m pytest -q tests/test_*.py
