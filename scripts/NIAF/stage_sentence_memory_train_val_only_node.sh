#!/bin/bash

# Strict node-local sentence-memory staging for Phase-A'''.  Unlike the legacy
# compatibility helper, this script never enumerates the shared bank directory:
# it reads the explicit core-artifact names from bank.json and opens only the
# neighbor splits named by the caller.

set -euo pipefail
trap 'echo "ERROR: strict sentence-memory staging failed at line $LINENO with exit code $?" >&2' ERR

ACTION="${1:?usage: $0 <stage|cleanup> <target> [source project python config splits]}"
TARGET="${2:?missing target directory}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: strict node staging requires SLURM_JOB_ID" >&2
  exit 1
fi
EXPECTED_TARGET="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
if [[ "$TARGET" != "$EXPECTED_TARGET" ]]; then
  echo "ERROR: refusing unexpected staging target: $TARGET" >&2
  exit 1
fi

cleanup_target() {
  if [[ -e "$TARGET" ]]; then
    rm -rf -- "$TARGET"
  fi
}

case "$ACTION" in
  cleanup)
    cleanup_target
    exit 0
    ;;
  stage) ;;
  *)
    echo "ERROR: action must be stage or cleanup; got $ACTION" >&2
    exit 1
    ;;
esac

SOURCE="${3:?missing source bank directory}"
PROJECT_DIR="${4:?missing project directory}"
PYTHON_BIN="${5:?missing Python executable}"
CFG="${6:?missing trajectory configuration}"
SPLITS="${7:-train val}"

for required in "$SOURCE/bank.json" "$SOURCE/build_summary.json" "$SOURCE/READY"; do
  [[ -f "$required" ]] || {
    echo "ERROR: required source-bank file is absent: $required" >&2
    exit 1
  }
done
[[ -x "$PYTHON_BIN" && -f "$CFG" ]] || {
  echo "ERROR: Python executable or trajectory config is absent" >&2
  exit 1
}

# The manifest parser rejects paths rather than normalizing them.  Its output is
# the complete explicit core allowlist; no shell glob or directory listing is
# performed against SOURCE.
mapfile -t CORE_ARTIFACTS < <(
  "$PYTHON_BIN" - "$SOURCE/bank.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
artifacts = manifest.get("artifacts")
if not isinstance(artifacts, dict) or not artifacts:
    raise SystemExit("ERROR: sentence-memory manifest has no artifacts")
seen = set()
filenames = []
for logical_name in sorted(artifacts):
    identity = artifacts[logical_name]
    filename = identity.get("file") if isinstance(identity, dict) else None
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {"READY", "bank.json", "build_summary.json"}
        or filename.startswith("neighbors_")
        or filename in seen
    ):
        raise SystemExit(f"ERROR: unsafe/duplicate core artifact: {logical_name}")
    seen.add(filename)
    filenames.append(filename)
for filename in filenames:
    print(filename)
PY
)
if [[ "${#CORE_ARTIFACTS[@]}" -eq 0 ]]; then
  echo "ERROR: empty core-artifact allowlist" >&2
  exit 1
fi

REQUESTED_SPLITS=()
for split in $SPLITS; do
  case "$split" in
    train|val) ;;
    *)
      echo "ERROR: strict Phase-A''' staging permits only train and val" >&2
      exit 1
      ;;
  esac
  REQUESTED_SPLITS+=("$split")
done
if [[ "${#REQUESTED_SPLITS[@]}" -eq 0 ]]; then
  echo "ERROR: at least one explicit train/val split is required" >&2
  exit 1
fi

cleanup_target
mkdir -p -- "$TARGET"
failure_cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ "$exit_code" != "0" ]]; then cleanup_target; fi
  exit "$exit_code"
}
trap failure_cleanup EXIT

cp -a -- "$SOURCE/bank.json" "$TARGET/bank.json"
cp -a -- "$SOURCE/build_summary.json" "$TARGET/build_summary.json"
for filename in "${CORE_ARTIFACTS[@]}"; do
  [[ -f "$SOURCE/$filename" ]] || {
    echo "ERROR: manifest-listed source artifact is absent: $filename" >&2
    exit 1
  }
  cp -a -- "$SOURCE/$filename" "$TARGET/$filename"
done
for split in "${REQUESTED_SPLITS[@]}"; do
  neighbor="neighbors_${split}.npz"
  [[ -f "$SOURCE/$neighbor" ]] || {
    echo "ERROR: explicitly requested neighbor table is absent: $neighbor" >&2
    exit 1
  }
  cp -a -- "$SOURCE/$neighbor" "$TARGET/$neighbor"
done
cp -a -- "$SOURCE/READY" "$TARGET/READY"

[[ -f "$TARGET/neighbors_train.npz" || ! " ${REQUESTED_SPLITS[*]} " =~ " train " ]] || {
  echo "ERROR: requested train table was not staged" >&2
  exit 1
}
[[ ! -e "$TARGET/neighbors_test.npz" ]] || {
  echo "ERROR: forbidden test neighbor table reached local staging" >&2
  exit 1
}

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m NIAF.continuous_trajectory_field.scripts.audit_sentence_memory \
  --config "$CFG" --bank_dir "$TARGET" --verify_hashes \
  --splits "${REQUESTED_SPLITS[@]}"

trap - EXIT
