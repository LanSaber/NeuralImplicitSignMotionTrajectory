#!/bin/bash

# Stage or remove one verified sentence-memory bank on the current Slurm node.
# The caller must invoke this script with one task per allocated node.

set -euo pipefail

ACTION="${1:?usage: $0 <stage|cleanup> <target> [source project python config splits]}"
TARGET="${2:?missing target directory}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: sentence-memory node staging requires SLURM_JOB_ID" >&2
  exit 1
fi
EXPECTED="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"
if [[ "$TARGET" != "$EXPECTED" ]]; then
  echo "ERROR: refusing unexpected sentence-memory staging target: $TARGET" >&2
  exit 1
fi

case "$ACTION" in
  stage)
    SOURCE="${3:?missing source bank directory}"
    PROJECT_DIR="${4:?missing project directory}"
    PYTHON_BIN="${5:?missing Python executable}"
    CFG="${6:?missing trajectory configuration}"
    SPLITS="${7:-train val}"
    if [[ ! -d "$SOURCE" || ! -f "$SOURCE/READY" ]]; then
      echo "ERROR: source sentence-memory bank is not ready: $SOURCE" >&2
      exit 1
    fi
    if [[ -e "$TARGET" ]]; then rm -rf -- "$TARGET"; fi
    mkdir -p -- "$TARGET"
    STAGE_ONLY_REQUESTED_NEIGHBORS="${STAGE_ONLY_REQUESTED_NEIGHBORS:-0}"
    case "$STAGE_ONLY_REQUESTED_NEIGHBORS" in
      0|1) ;;
      *)
        echo "ERROR: STAGE_ONLY_REQUESTED_NEIGHBORS must be 0 or 1" >&2
        exit 1
        ;;
    esac
    if [[ "$STAGE_ONLY_REQUESTED_NEIGHBORS" == "0" ]]; then
      # Compatibility mode for training/evaluation launchers whose persisted
      # checkpoint identity includes every table in the source bank.
      cp -a -- "$SOURCE"/. "$TARGET"/
    else
      # Validation-only diagnostics must not copy, open, or hash an unrelated
      # test neighbor table.  Copy all core bank artifacts, then only the
      # explicitly requested neighbor tables.  READY is copied last.
      for source_path in "$SOURCE"/*; do
        base_name="$(basename -- "$source_path")"
        case "$base_name" in
          READY|neighbors_*.npz) continue ;;
        esac
        cp -a -- "$source_path" "$TARGET"/
      done
      for split in $SPLITS; do
        case "$split" in
          train|val|test) ;;
          *) echo "ERROR: invalid requested neighbor split: $split" >&2; exit 1 ;;
        esac
        neighbor_path="$SOURCE/neighbors_${split}.npz"
        [[ -f "$neighbor_path" ]] || {
          echo "ERROR: requested neighbor table is missing: $neighbor_path" >&2
          exit 1
        }
        cp -a -- "$neighbor_path" "$TARGET"/
      done
      cp -a -- "$SOURCE/READY" "$TARGET/READY"
    fi
    [[ -f "$TARGET/READY" ]] || {
      echo "ERROR: staged sentence-memory bank has no READY marker: $TARGET" >&2
      exit 1
    }
    cd "$PROJECT_DIR"
    # shellcheck disable=SC2086 # SPLITS is an intentional list of split names.
    "$PYTHON_BIN" -m \
      NIAF.continuous_trajectory_field.scripts.audit_sentence_memory \
      --config "$CFG" --bank_dir "$TARGET" --verify_hashes --splits $SPLITS
    ;;
  cleanup)
    if [[ -e "$TARGET" ]]; then rm -rf -- "$TARGET"; fi
    ;;
  *)
    echo "ERROR: action must be stage or cleanup; got $ACTION" >&2
    exit 1
    ;;
esac
