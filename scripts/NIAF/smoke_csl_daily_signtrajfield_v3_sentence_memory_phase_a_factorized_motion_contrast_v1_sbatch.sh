#!/bin/bash
#SBATCH --job-name=csl_v3_splitkv_smoke
#SBATCH --output=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.out
#SBATCH --error=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/logs/sbatch/%x_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Mandatory implementation smoke: Stage 1 then Stage 2, exactly one optimizer
# step apiece. This is architecture coverage, not adaptive model selection.
set -euo pipefail
trap 'echo "ERROR: factorized-memory two-arm smoke failed at line $LINENO with exit code $?" >&2' ERR

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the explicit shared standalone source clone}"
PYTHON_ENV="${PYTHON_ENV:-/media/cvpr/haomian/python_envs/SOKE}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SOURCE_GIT_HEAD="${SOURCE_GIT_HEAD:?Set SOURCE_GIT_HEAD to the clean pushed commit selected on the submit host}"
SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1}"
V2_CHECKPOINT="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
SEALED_PARTITION_DIR="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition"
EXPECTED_DEVELOPMENT_MANIFEST_SHA256="9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
EXPECTED_PARTITION_READY_SHA256="e12b8d2db7989efacf15b5b4abe297ee1cd18a58c9a1519e197b8e8b9ce86df0"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_NNODES:-0}" != "1" ]]; then
  echo "ERROR: smoke requires one Slurm node" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$SENTENCE_MEMORY_DIR/READY" ]]; then
  echo "ERROR: smoke Python or READY sentence bank is missing" >&2
  exit 1
fi
if [[ "$(sha256sum -- "$V2_CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_V2_SHA256" ]]; then
  echo "ERROR: smoke v2 checkpoint SHA256 mismatch" >&2
  exit 1
fi
for forbidden in RESUME WARM_START BASE_CHECKPOINT PHASE_B_GATE_REPORT; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "ERROR: fresh smoke forbids inherited $forbidden" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR"
if [[ ! -d "$PROJECT_DIR/.git" || -L "$PROJECT_DIR/.git" ]]; then
  echo "ERROR: smoke requires an explicit shared standalone clone; Codex worktree gitdir pointers are not cluster-visible" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: smoke source clone must reside on /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/experiments" ]]; then
  echo "ERROR: smoke clone must expose the durable experiments directory" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/experiments")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: smoke experiments must resolve to /media/cvpr" >&2; exit 1 ;;
esac
if [[ ! -d "$PROJECT_DIR/deps/mt5-base" ]]; then
  echo "ERROR: smoke clone must expose frozen deps/mt5-base" >&2
  exit 1
fi
case "$(realpath -e "$PROJECT_DIR/deps/mt5-base")" in
  /media/cvpr/*) ;;
  *) echo "ERROR: smoke mT5 dependency must resolve to /media/cvpr" >&2; exit 1 ;;
esac
if [[ "$(realpath -e "$(git rev-parse --path-format=absolute --git-common-dir)")" != "$(realpath -e "$PROJECT_DIR/.git")" ]]; then
  echo "ERROR: smoke source uses external or node-local git metadata" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$SOURCE_GIT_HEAD" ]]; then
  echo "ERROR: smoke worktree HEAD differs from SOURCE_GIT_HEAD" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: smoke requires a clean worktree" >&2
  exit 1
fi
SOURCE_REMOTE_BRANCH="${SOURCE_REMOTE_BRANCH:-$(git branch --show-current)}"
if [[ -z "$SOURCE_REMOTE_BRANCH" ]]; then
  echo "ERROR: detached smoke source requires SOURCE_REMOTE_BRANCH" >&2
  exit 1
fi
REMOTE_HEAD="$(git ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR == 1 {print $1}')"
if [[ -z "$REMOTE_HEAD" || "${REMOTE_HEAD,,}" != "${SOURCE_GIT_HEAD,,}" ]]; then
  echo "ERROR: live smoke origin branch does not equal SOURCE_GIT_HEAD" >&2
  exit 1
fi
if [[ ! -f "$SEALED_PARTITION_DIR/READY" \
      || "$(sha256sum -- "$SEALED_PARTITION_DIR/READY" | awk '{print $1}')" != "$EXPECTED_PARTITION_READY_SHA256" \
      || "$(sha256sum -- "$SEALED_PARTITION_DIR/manifest_development.jsonl" | awk '{print $1}')" != "$EXPECTED_DEVELOPMENT_MANIFEST_SHA256" \
      || "$(awk 'NF {count += 1} END {print count + 0}' "$SEALED_PARTITION_DIR/manifest_development.jsonl")" != "347" ]]; then
  echo "ERROR: smoke requires the sealed development-only partition" >&2
  exit 1
fi

export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

run_arm() {
  local experiment="$1"
  local cfg="$PROJECT_DIR/NIAF/continuous_trajectory_field/configs/${experiment}.yaml"
  local out_dir="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/$experiment"
  if [[ ! -f "$cfg" || -e "$out_dir" ]]; then
    echo "ERROR: missing smoke config or non-fresh output: $experiment" >&2
    return 1
  fi
  CFG="$cfg" \
  OUT_DIR="$out_dir" \
  RUN_TAG="$experiment" \
  SENTENCE_MEMORY_DIR="$SENTENCE_MEMORY_DIR" \
  SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR="$SEALED_PARTITION_DIR" \
  EPOCHS=1 LIMIT_TRAIN=64 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 \
  WANDB=0 WANDB_MODE=disabled \
  STAGE_SENTENCE_MEMORY=1 STAGE_ONLY_REQUESTED_NEIGHBORS=1 \
  SPLITS="train val" DEVICE=cuda TEXT_DEVICE=cpu \
  DISTRIBUTED=none DDP_BACKEND=auto \
    bash "$PROJECT_DIR/scripts/NIAF/train_continuous_trajectory_field_sbatch.sh"

  local expected_prior="none"
  if [[ "$experiment" == *temporal_bias* ]]; then
    expected_prior="gaussian"
  fi
  "$PYTHON_BIN" - "$out_dir/checkpoints/last.pt" "$experiment" "$expected_prior" <<'PY'
import math
import sys
from pathlib import Path
import torch

from NIAF.continuous_trajectory_field.scripts import (
    train_continuous_trajectory_field as trainer,
)

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
if int(checkpoint.get("epoch", -1)) != 1 or int(checkpoint.get("global_step", -1)) != 1:
    raise SystemExit("ERROR: smoke did not perform exactly one optimizer step")
cfg = checkpoint.get("config", {})
if cfg.get("experiment_name") != sys.argv[2]:
    raise SystemExit("ERROR: smoke checkpoint experiment identity changed")
memory = cfg.get("sentence_memory", {})
if (memory.get("key_value_mode") != "factorized_metadata_motion_v1"
        or memory.get("temporal_prior_mode") != sys.argv[3]):
    raise SystemExit("ERROR: smoke checkpoint has the wrong factorized arm")
if not cfg.get("train", {}).get("freeze_base") or cfg.get("train", {}).get("unfreeze_base_prefixes"):
    raise SystemExit("ERROR: smoke did not preserve frozen-base Phase A")
if not cfg.get("sentence_memory_safety", {}).get("paired_corruption", {}).get("enabled"):
    raise SystemExit("ERROR: smoke did not enable paired corruption")
for name, compute in (
    ("architecture", trainer.sentence_memory_architecture_identity),
    ("behavior", trainer.sentence_memory_behavior_identity),
    ("objective", trainer.sentence_memory_objective_identity),
    ("evaluation_control", trainer.sentence_memory_evaluation_control_identity),
    ("selection_aggregation", trainer.sentence_memory_selection_aggregation_identity),
):
    if checkpoint.get(f"sentence_memory_{name}_identity") != compute(cfg):
        raise SystemExit(f"ERROR: smoke {name} identity does not reload")
parity = checkpoint.get("v2_to_v3_text_only_parity", {})
if (parity.get("passed") is not True
        or float(parity.get("prediction_max_abs", math.inf)) > 1e-7
        or float(parity.get("duration_max_abs", math.inf)) > 1e-7):
    raise SystemExit("ERROR: smoke lacks exact v2 memory-off parity")
metrics = checkpoint.get("metrics", {})
for prefix in (
    "val_text_only/", "val_sentence_memory/",
    "val_motion_shuffled_sentence_memory/", "val_shuffled_sentence_memory/",
    "val_analytic_prior_sentence_memory/",
):
    if not any(str(key).startswith(prefix) for key in metrics):
        raise SystemExit(f"ERROR: smoke lacks {prefix} metrics")
for key, value in metrics.items():
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise SystemExit(f"ERROR: smoke metric is non-finite: {key}")
motion_fraction = float(metrics.get("sentence_memory_motion_corrupt_fraction", 0.0))
full_fraction = float(metrics.get("sentence_memory_full_shuffle_fraction", 0.0))
if not (0.0 <= motion_fraction <= 1.0 and 0.0 <= full_fraction <= 1.0
        and motion_fraction + full_fraction > 0.0):
    raise SystemExit("ERROR: smoke did not exercise paired corrupt outputs")
if not isinstance(checkpoint.get("optimizer"), dict):
    raise SystemExit("ERROR: smoke checkpoint cannot reload optimizer state")
for name, value in checkpoint.get("model", {}).items():
    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        raise SystemExit(f"ERROR: smoke model tensor is non-finite: {name}")
PY
}

run_arm csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1_smoke
run_arm csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1_smoke
