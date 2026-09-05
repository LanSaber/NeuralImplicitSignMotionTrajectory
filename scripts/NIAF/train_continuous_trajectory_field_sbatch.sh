#!/bin/bash
#SBATCH --job-name=niaf_cont_traj
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00

set -eo pipefail
trap 'echo "ERROR: train_continuous_trajectory_field_sbatch.sh failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:-/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
CFG="${CFG:-NIAF/continuous_trajectory_field/configs/phoenix_continuous_trajectory_full.yaml}"
RUN_TAG="${RUN_TAG:-phoenix_continuous_trajectory_full}"
DEVICE="${DEVICE:-auto}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
DISTRIBUTED="${DISTRIBUTED:-ddp}"
DDP_BACKEND="${DDP_BACKEND:-nccl}"
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-120}"
WANDB="${WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-soke-niaf-continuous-trajectory}"
WANDB_ENTITY="${WANDB_ENTITY:-hh3443-new-york-university}"
WANDB_USERNAME="${WANDB_USERNAME:-hh3443}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID:-local}}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

# Online jobs must authenticate exclusively through the private, node-local
# netrc on the batch/rank-0 host. Remember an inherited inline key only so the
# online preflight can reject it; never pass it to Python or the training SDK.
INHERITED_WANDB_API_KEY_PRESENT=0
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  INHERITED_WANDB_API_KEY_PRESENT=1
fi
unset WANDB_API_KEY

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SENTENCE_MEMORY_K="${SENTENCE_MEMORY_K:-}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
RESUME="${RESUME:-}"
WARM_START="${WARM_START:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
PHASE_B_GATE_REPORT="${PHASE_B_GATE_REPORT:-}"
RESET_LOCAL_BRANCH="${RESET_LOCAL_BRANCH:-0}"
OUT_DIR="${OUT_DIR:-}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-}"
STAGE_SENTENCE_MEMORY="${STAGE_SENTENCE_MEMORY:-0}"
LOCAL_SENTENCE_MEMORY_DIR=""

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/media/cvpr/haomian/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export WANDB_MODE WANDB_ENTITY WANDB_USERNAME WANDB_BASE_URL
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/cvpr/haomian/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/media/cvpr/haomian/.config/wandb}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  fi
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
  fi
  export MASTER_ADDR MASTER_PORT
  if [[ -n "${SLURM_NTASKS:-}" ]]; then export WORLD_SIZE="$SLURM_NTASKS"; fi
fi

validate_wandb_online_auth() {
  if [[ "$WANDB" != "1" || "${WANDB_MODE,,}" != "online" ]]; then
    return 0
  fi
  if [[ "$INHERITED_WANDB_API_KEY_PRESENT" == "1" ]]; then
    echo "ERROR: refusing WANDB_API_KEY for an online run; use the batch host's private \$HOME/.netrc" >&2
    return 1
  fi
  if [[ "${WANDB_BASE_URL%/}" != "https://api.wandb.ai" ]]; then
    echo "ERROR: online W&B runs must use https://api.wandb.ai; got $WANDB_BASE_URL" >&2
    return 1
  fi

  local wandb_netrc="${HOME:?HOME must be set}/.netrc"
  if [[ -L "$wandb_netrc" || ! -f "$wandb_netrc" ]]; then
    echo "ERROR: online W&B requires a regular, non-symlink credential file at \$HOME/.netrc" >&2
    return 1
  fi

  local expected_uid actual_uid actual_mode
  expected_uid="$(id -u)"
  actual_uid="$(stat -c '%u' -- "$wandb_netrc")"
  actual_mode="$(stat -c '%a' -- "$wandb_netrc")"
  if [[ "$actual_uid" != "$expected_uid" ]]; then
    echo "ERROR: \$HOME/.netrc must be owned by the current uid ($expected_uid); got $actual_uid" >&2
    return 1
  fi
  if [[ "$actual_mode" != "600" ]]; then
    echo "ERROR: \$HOME/.netrc must have mode 0600; got 0$actual_mode" >&2
    return 1
  fi

  # Parse the credential with Python's netrc implementation and use that value
  # only in-process for a read-only viewer query. Nothing secret is printed,
  # exported, written to shared storage, or placed in a command argument.
  "$PYTHON_BIN" - "$wandb_netrc" "$WANDB_BASE_URL" "$WANDB_USERNAME" <<'PY'
import netrc
import os
import stat
import sys
from urllib.parse import urlparse

netrc_path, base_url, expected_username = sys.argv[1:]
metadata = os.stat(netrc_path, follow_symlinks=False)
if metadata.st_uid != os.getuid():
    raise SystemExit("ERROR: W&B netrc ownership changed during preflight")
if stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit("ERROR: W&B netrc permissions changed during preflight")

host = urlparse(base_url).hostname
if host != "api.wandb.ai":
    raise SystemExit("ERROR: W&B preflight endpoint is not api.wandb.ai")
credentials = netrc.netrc(netrc_path).authenticators(host)
if credentials is None or not credentials[0] or not credentials[2]:
    raise SystemExit("ERROR: $HOME/.netrc has no complete api.wandb.ai credentials")

import wandb

api = wandb.Api(
    overrides={"base_url": base_url},
    api_key=credentials[2],
    timeout=30,
)
viewer = api.viewer
username = str(getattr(viewer, "username", "") or "").strip()
if not username:
    raise SystemExit("ERROR: W&B viewer/auth check returned no authenticated username")
if username != expected_username:
    raise SystemExit(
        "ERROR: W&B viewer/auth account differs from the configured WANDB_USERNAME"
    )
print(f"W&B online preflight passed for authenticated viewer {username!r}.", flush=True)
PY
}

validate_wandb_online_auth

cleanup_sentence_memory() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$LOCAL_SENTENCE_MEMORY_DIR" && "$DRY_RUN" != "1" ]]; then
    local expected="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID:-}"
    if [[ -z "${SLURM_JOB_ID:-}" || "$LOCAL_SENTENCE_MEMORY_DIR" != "$expected" ]]; then
      echo "ERROR: refusing to clean unexpected staging path: $LOCAL_SENTENCE_MEMORY_DIR" >&2
      exit "$exit_code"
    fi
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
      bash -c 'set -eu; target="$1"; expected="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"; [[ "$target" == "$expected" ]]; rm -rf -- "$target"' \
      bash "$LOCAL_SENTENCE_MEMORY_DIR" || true
  fi
  exit "$exit_code"
}
trap cleanup_sentence_memory EXIT

case "$STAGE_SENTENCE_MEMORY" in
  0|1) ;;
  *)
    echo "ERROR: STAGE_SENTENCE_MEMORY must be 0 or 1; got $STAGE_SENTENCE_MEMORY" >&2
    exit 1
    ;;
esac
if [[ "$STAGE_SENTENCE_MEMORY" == "1" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: sentence-memory staging requires a Slurm allocation" >&2
    exit 1
  fi
  if [[ -z "$SENTENCE_MEMORY_DIR" || ! -d "$SENTENCE_MEMORY_DIR" ]]; then
    echo "ERROR: set SENTENCE_MEMORY_DIR to an existing exported bank" >&2
    exit 1
  fi
  LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
  if [[ "$DRY_RUN" != "1" ]]; then
    export SENTENCE_MEMORY_DIR LOCAL_SENTENCE_MEMORY_DIR PROJECT_DIR PYTHON_BIN CFG
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
      bash -c 'set -euo pipefail
        source_dir="$SENTENCE_MEMORY_DIR"
        target_dir="$LOCAL_SENTENCE_MEMORY_DIR"
        expected="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
        [[ "$target_dir" == "$expected" ]]
        if [[ -e "$target_dir" ]]; then rm -rf -- "$target_dir"; fi
        mkdir -p -- "$target_dir"
        cp -a -- "$source_dir"/. "$target_dir"/
        [[ -f "$target_dir/READY" ]] || {
          echo "ERROR: staged sentence bank has no READY marker: $target_dir" >&2
          exit 1
        }
        cd "$PROJECT_DIR"
        "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.audit_sentence_memory \
          --config "$CFG" --bank_dir "$target_dir" --splits train val --verify_hashes'
  fi
  export SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"
elif [[ -n "$SENTENCE_MEMORY_DIR" ]]; then
  if [[ ! -d "$SENTENCE_MEMORY_DIR" ]]; then
    echo "ERROR: SENTENCE_MEMORY_DIR does not exist: $SENTENCE_MEMORY_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
    echo "ERROR: SENTENCE_MEMORY_DIR has no READY marker: $SENTENCE_MEMORY_DIR" >&2
    exit 1
  fi
  export SIGNTRAJ_SENTENCE_MEMORY_DIR="$SENTENCE_MEMORY_DIR"
elif [[ -n "${SIGNTRAJ_SENTENCE_MEMORY_DIR:-}" ]]; then
  if [[ ! -d "$SIGNTRAJ_SENTENCE_MEMORY_DIR" || ! -f "$SIGNTRAJ_SENTENCE_MEMORY_DIR/READY" ]]; then
    echo "ERROR: SIGNTRAJ_SENTENCE_MEMORY_DIR must name a ready sentence bank: $SIGNTRAJ_SENTENCE_MEMORY_DIR" >&2
    exit 1
  fi
else
  CONFIG_SENTENCE_MEMORY_DIR="$(
    cd "$PROJECT_DIR"
    "$PYTHON_BIN" - "$CFG" <<'PY'
import sys
from pathlib import Path

from NIAF.continuous_sign_field.config import load_config

cfg = load_config(Path(sys.argv[1]))
if (
    cfg.get("model", {}).get("type")
    == "sentence_memory_continuous_trajectory_field"
    and bool(cfg.get("sentence_memory", {}).get("enabled", True))
):
    value = cfg.get("sentence_memory", {}).get("bank_dir")
    if not value:
        raise SystemExit(
            "ERROR: v3 config has no sentence_memory.bank_dir; set "
            "SENTENCE_MEMORY_DIR or SIGNTRAJ_SENTENCE_MEMORY_DIR"
        )
    path = Path(value)
    print(path if path.is_absolute() else Path.cwd() / path)
PY
  )"
  if [[ -n "$CONFIG_SENTENCE_MEMORY_DIR" ]]; then
    if [[ ! -d "$CONFIG_SENTENCE_MEMORY_DIR" || ! -f "$CONFIG_SENTENCE_MEMORY_DIR/READY" ]]; then
      echo "ERROR: configured sentence-memory bank is not ready: $CONFIG_SENTENCE_MEMORY_DIR" >&2
      echo "Build/export the bank first, or set SENTENCE_MEMORY_DIR to a ready bank." >&2
      exit 1
    fi
  fi
fi

mkdir -p \
  "$PROJECT_DIR/logs/sbatch" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
cd "$PROJECT_DIR"

TRAIN_CMD=(
  srun --kill-on-bad-exit=1
  "$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field
  --config "$CFG"
  --device "$DEVICE"
  --text_device "$TEXT_DEVICE"
  --distributed "$DISTRIBUTED"
  --ddp_backend "$DDP_BACKEND"
  --ddp_timeout_min "$DDP_TIMEOUT_MIN"
)
if [[ -n "$EPOCHS" ]]; then TRAIN_CMD+=(--epochs "$EPOCHS"); fi
if [[ -n "$BATCH_SIZE" ]]; then TRAIN_CMD+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$SENTENCE_MEMORY_K" ]]; then
  TRAIN_CMD+=(--sentence_memory_k "$SENTENCE_MEMORY_K")
fi
if [[ -n "$LIMIT_TRAIN" ]]; then TRAIN_CMD+=(--limit_train "$LIMIT_TRAIN"); fi
if [[ -n "$LIMIT_VAL" ]]; then TRAIN_CMD+=(--limit_val "$LIMIT_VAL"); fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then TRAIN_CMD+=(--max_train_batches "$MAX_TRAIN_BATCHES"); fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then TRAIN_CMD+=(--max_val_batches "$MAX_VAL_BATCHES"); fi
if [[ -n "$RESUME" ]]; then TRAIN_CMD+=(--resume "$RESUME"); fi
if [[ -n "$WARM_START" ]]; then TRAIN_CMD+=(--warm_start "$WARM_START"); fi
if [[ -n "$BASE_CHECKPOINT" ]]; then TRAIN_CMD+=(--base_checkpoint "$BASE_CHECKPOINT"); fi
if [[ -n "$PHASE_B_GATE_REPORT" ]]; then TRAIN_CMD+=(--phase_b_gate_report "$PHASE_B_GATE_REPORT"); fi
if [[ "$RESET_LOCAL_BRANCH" == "1" ]]; then TRAIN_CMD+=(--reset_local_branch); fi
if [[ -n "$OUT_DIR" ]]; then TRAIN_CMD+=(--out_dir "$OUT_DIR"); fi
if [[ "$WANDB" == "1" ]]; then
  TRAIN_CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
  if [[ -n "$WANDB_ID" ]]; then TRAIN_CMD+=(--wandb_id "$WANDB_ID"); fi
  if [[ -n "$WANDB_RESUME" ]]; then TRAIN_CMD+=(--wandb_resume "$WANDB_RESUME"); fi
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodes: ${SLURM_NODELIST:-local} world_size=${WORLD_SIZE:-1}"
echo "Config: $CFG"
echo "Batch override: ${BATCH_SIZE:-config value}"
echo "Sentence-memory K override: ${SENTENCE_MEMORY_K:-config value}"
echo "DDP: $DISTRIBUTED/$DDP_BACKEND master=${MASTER_ADDR:-unset}:${MASTER_PORT:-unset}"
echo "W&B: $WANDB/$WANDB_MODE username=$WANDB_USERNAME entity=$WANDB_ENTITY project=$WANDB_PROJECT run=$WANDB_RUN_NAME"
echo "Sentence memory: ${SIGNTRAJ_SENTENCE_MEMORY_DIR:-config value} staged=$STAGE_SENTENCE_MEMORY"
echo "Phase-B gate report: ${PHASE_B_GATE_REPORT:-not supplied}"
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi
"${TRAIN_CMD[@]}"
