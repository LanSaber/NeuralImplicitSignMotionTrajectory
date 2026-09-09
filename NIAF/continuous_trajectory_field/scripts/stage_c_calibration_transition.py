"""Prove the Stage-C calibration is a fresh provenance-only transition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    publish_bytes_no_replace,
)


SOURCE_DECISION_SHA256 = (
    "8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5"
)
SOURCE_DECISION_IDENTITY = (
    "7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69"
)
SOURCE_ARCHITECTURE_IDENTITY = (
    "bc69fd35ac58e10bc894460c35175f13356b79416df40e8c57156b1929614236"
)
SCHEMA_NAME = "signtrajfield_stage_c_calibration_transition_audit"
SCHEMA_VERSION = 1


class CalibrationTransitionError(RuntimeError):
    """The fresh calibration differs from Stage B beyond source provenance."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise CalibrationTransitionError(f"{label} is not a regular file: {path}")
    return path


def _named(value: Any, label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, dict) or "digest" not in value:
        raise CalibrationTransitionError(f"{label} is not a named identity")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}
    digest = digest_json(payload)
    if value["digest"] != digest:
        raise CalibrationTransitionError(f"{label} digest changed")
    return payload, digest


def build_transition_audit(
    *, source_terminal_decision: Path, stage_c_config: Path
) -> dict[str, Any]:
    _regular(source_terminal_decision, "Stage-B terminal decision")
    _regular(stage_c_config, "Stage-C memory config")
    if sha256_file(source_terminal_decision) != SOURCE_DECISION_SHA256:
        raise CalibrationTransitionError("Stage-B terminal decision SHA256 changed")
    decision = json.loads(source_terminal_decision.read_text(encoding="utf-8"))
    unsigned_decision = {
        key: value for key, value in decision.items() if key != "decision_identity"
    }
    if (
        decision.get("schema_name")
        != "signtrajfield_centered_memory_ordered_decision"
        or decision.get("schema_version") != 1
        or decision.get("decision_identity") != SOURCE_DECISION_IDENTITY
        or decision.get("decision_identity") != digest_json(unsigned_decision)
        or decision.get("status") != "valid_infeasible"
        or decision.get("confirmation_manifest_opened") is not False
        or decision.get("test_data_accessed") is not False
        or decision.get("checkpoint", {}).get("architecture_identity")
        not in {None, SOURCE_ARCHITECTURE_IDENTITY}
        or decision.get("checkpoint", {}).get("identities", {}).get(
            "architecture", {}
        ).get("digest")
        != SOURCE_ARCHITECTURE_IDENTITY
    ):
        raise CalibrationTransitionError("Stage-B terminal decision binding changed")
    source_architecture_wrapper = decision["checkpoint"]["identities"]["architecture"]
    source_architecture = source_architecture_wrapper.get("value")
    if (
        not isinstance(source_architecture, dict)
        or source_architecture.get("digest") != SOURCE_ARCHITECTURE_IDENTITY
    ):
        raise CalibrationTransitionError("source architecture payload is malformed")
    source_calibration = source_architecture.get("relevance_gate", {}).get(
        "calibration_identity"
    )
    source_payload, source_digest = _named(source_calibration, "source calibration")

    config = load_config(stage_c_config)
    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    active_calibration = trainer.resolve_sentence_memory_relevance_calibration(config)
    active_payload, active_digest = _named(active_calibration, "active calibration")
    provenance_fields = {"artifact_identity", "calibration_sha256"}
    source_semantics = {
        key: value for key, value in source_payload.items() if key not in provenance_fields
    }
    active_semantics = {
        key: value for key, value in active_payload.items() if key not in provenance_fields
    }
    if source_semantics != active_semantics:
        changed = sorted(
            key
            for key in set(source_semantics) | set(active_semantics)
            if source_semantics.get(key) != active_semantics.get(key)
        )
        raise CalibrationTransitionError(
            f"fresh calibration changed semantic fields: {changed}"
        )
    if active_digest == source_digest:
        raise CalibrationTransitionError("Stage-C calibration identity is not fresh")
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source_terminal_decision": {
            "path": str(source_terminal_decision.resolve()),
            "sha256": SOURCE_DECISION_SHA256,
            "decision_identity": SOURCE_DECISION_IDENTITY,
            "status": "valid_infeasible",
        },
        "stage_c_config": {
            "path": str(stage_c_config.resolve()),
            "sha256": sha256_file(stage_c_config),
        },
        "source_architecture_identity": SOURCE_ARCHITECTURE_IDENTITY,
        "source_calibration_identity": source_digest,
        "active_calibration_identity": active_digest,
        "source_artifact_identity": source_payload["artifact_identity"],
        "active_artifact_identity": active_payload["artifact_identity"],
        "source_calibration_sha256": source_payload["calibration_sha256"],
        "active_calibration_sha256": active_payload["calibration_sha256"],
        "semantic_equivalence": {
            "coefficients_exact": True,
            "map_sha256_exact": True,
            "map_content_digest_exact": True,
            "heldout_metrics_exact": True,
            "minimum_gates_exact": True,
            "all_non_provenance_fields_exact": True,
            "source_and_active_identities_distinct": True,
        },
        "train_only": True,
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    return {**payload, "audit_identity": digest_json(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--source_terminal_decision", type=Path, required=True)
        command.add_argument("--stage_c_config", type=Path, required=True)
        if name == "create":
            command.add_argument("--out_file", type=Path, required=True)
        else:
            command.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = build_transition_audit(
        source_terminal_decision=args.source_terminal_decision,
        stage_c_config=args.stage_c_config,
    )
    if args.command == "verify":
        _regular(args.audit, "stored Stage-C calibration transition audit")
        observed = json.loads(args.audit.read_text(encoding="utf-8"))
        if observed != value:
            raise CalibrationTransitionError(
                "stored calibration transition differs from exact recomputation"
            )
    else:
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        publish_bytes_no_replace(args.out_file, encoded)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
