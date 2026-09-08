"""Fit and seal the CSL-Daily train-only sentence-relevance calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from NIAF.continuous_trajectory_field.relevance_calibration import (
    ALTERNATIVE_NONCE,
    DEFAULT_SEED,
    RelevanceCalibrationError,
    SPLIT_NONCE,
    build_calibration_map,
    calibrate_from_map,
    load_train_inputs,
    publish_calibration,
    sha256_file,
    validate_relevance_calibration_artifact,
)


EXPECTED_BANK_ID = "a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
EXPECTED_TRAIN_NEIGHBOR_SHA256 = (
    "d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_dir", type=Path, required=True)
    parser.add_argument("--train_neighbors", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--source_git_head", required=True)
    parser.add_argument("--source_remote_ref", required=True)
    parser.add_argument("--source_remote_head", required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration_weight", type=float, default=0.05)
    parser.add_argument("--minimum_auroc", type=float, default=0.75)
    parser.add_argument("--minimum_probability_gap", type=float, default=0.20)
    parser.add_argument("--split_nonce", default=SPLIT_NONCE)
    parser.add_argument(
        "--alternative_nonce",
        default=ALTERNATIVE_NONCE,
    )
    parser.add_argument("--expected_bank_id", default=EXPECTED_BANK_ID)
    parser.add_argument(
        "--expected_train_neighbor_sha256",
        default=EXPECTED_TRAIN_NEIGHBOR_SHA256,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != DEFAULT_SEED or args.duration_weight != 0.05:
        raise RelevanceCalibrationError(
            "The v1 calibration requires seed=1234 and duration_weight=0.05"
        )
    if args.minimum_auroc != 0.75 or args.minimum_probability_gap != 0.20:
        raise RelevanceCalibrationError("The v1 acceptance thresholds are immutable")
    if args.split_nonce != SPLIT_NONCE or args.alternative_nonce != ALTERNATIVE_NONCE:
        raise RelevanceCalibrationError("The v1 calibration map nonces are immutable")
    source_root = args.source_root.resolve()
    source_git_head = str(args.source_git_head).lower()
    source_remote_head = str(args.source_remote_head).lower()
    if source_remote_head != source_git_head:
        raise RelevanceCalibrationError(
            "Calibration source and pushed remote heads differ"
        )
    try:
        actual_head = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
        actual_remote = (
            subprocess.run(
                ["git", "rev-parse", str(args.source_remote_ref)],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RelevanceCalibrationError(
            "Could not prove the calibration source checkout"
        ) from error
    if actual_head != source_git_head or actual_remote != source_git_head or dirty:
        raise RelevanceCalibrationError(
            "Calibration requires the exact clean pushed source checkout"
        )
    launcher = args.launcher.resolve()
    helper = source_root / "NIAF/continuous_trajectory_field/relevance_calibration.py"
    decision_helper = source_root / (
        "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py"
    )
    stager = source_root / "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh"
    script = Path(__file__).resolve()
    for path in (launcher, helper, decision_helper, stager, script):
        if not path.is_file() or not path.is_relative_to(source_root):
            raise RelevanceCalibrationError(
                f"Calibration source file is absent/outside source root: {path}"
            )
    inputs = load_train_inputs(
        args.bank_dir,
        args.train_neighbors,
        expected_bank_id=args.expected_bank_id,
        expected_neighbor_sha256=args.expected_train_neighbor_sha256,
    )
    arrays = build_calibration_map(
        inputs,
        duration_weight=args.duration_weight,
        candidate_count=8,
        seed=args.seed,
        split_nonce=args.split_nonce,
        alternative_nonce=args.alternative_nonce,
    )
    result = calibrate_from_map(
        arrays,
        minimum_auroc=args.minimum_auroc,
        minimum_probability_gap=args.minimum_probability_gap,
    )
    source_identity = {
        "git_head": source_git_head,
        "remote_head": source_remote_head,
        "remote_ref": str(args.source_remote_ref),
        "repository_root": str(source_root),
        "source_files": {
            str(path.relative_to(source_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (helper, script, decision_helper, stager, launcher)
        },
    }
    calibration = publish_calibration(
        output_dir=args.out_dir,
        arrays=arrays,
        result=result,
        input_identity={
            "bank": inputs["bank_identity"],
            "train_neighbor_table": inputs["neighbor_identity"],
        },
        source_identity=source_identity,
        seed=args.seed,
        duration_weight=args.duration_weight,
        split_nonce=args.split_nonce,
        alternative_nonce=args.alternative_nonce,
    )
    if not result["accepted"]:
        print(json.dumps(calibration, indent=2, ensure_ascii=False))
        raise SystemExit(2)
    validated = validate_relevance_calibration_artifact(
        args.out_dir,
        expected_identity=calibration["identity"],
        minimum_auroc=args.minimum_auroc,
        minimum_probability_gap=args.minimum_probability_gap,
    )
    print(json.dumps(validated, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
