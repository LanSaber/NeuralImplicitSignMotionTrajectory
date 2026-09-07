from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import (
    read_jsonl,
    sentence_text_hash,
    sha256_file,
)
from NIAF.continuous_trajectory_field.validation_text_partitions import (
    CONFIRMATION,
    DEVELOPMENT,
    EXACT_SEEN,
    partition_validation_text_clusters,
)


OUTPUT_FILES = {
    DEVELOPMENT: "manifest_development.jsonl",
    CONFIRMATION: "manifest_confirmation.jsonl",
    EXACT_SEEN: "manifest_exact_seen.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic development/confirmation partition of the "
            "canonical validation text clusters. This tool cannot access test data."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--development_text_count", type=int, default=256)
    parser.add_argument("--expected_novel_text_count", type=int, default=796)
    parser.add_argument("--verify_only", action="store_true")
    return parser.parse_args()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    os.replace(temporary, path)


def _resolve_validation_manifest(cfg: dict[str, Any]) -> Path:
    data_cfg = dict(cfg.get("data", {}) or {})
    configured = data_cfg.get("val_manifest_path")
    if configured:
        path = Path(configured)
    else:
        data_dir = data_cfg.get("data_dir")
        if not data_dir:
            raise ValueError("data.data_dir is required")
        path = Path(data_dir) / "meta" / "manifest_val.jsonl"
    if path.name != "manifest_val.jsonl":
        raise ValueError(
            "Validation partition requires the canonical manifest_val.jsonl; "
            f"got {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Validation manifest does not exist: {path}")
    return path.resolve()


def _resolve_bank_dir(cfg: dict[str, Any], requested: Path | None) -> Path:
    configured = dict(cfg.get("sentence_memory", {}) or {}).get("bank_dir")
    value = requested if requested is not None else configured
    if not value:
        raise ValueError("--bank_dir or sentence_memory.bank_dir is required")
    path = Path(value).resolve()
    if not (path / "READY").is_file() or not (path / "bank.json").is_file():
        raise ValueError(f"Sentence bank is not READY: {path}")
    return path


def _source_split(row: dict[str, Any]) -> str:
    return str(row.get("source_split", "val"))


def build_partition_payload(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    bank_dir: Path,
    validation_manifest: Path,
    development_text_count: int,
    expected_novel_text_count: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    bank = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    source = dict(bank.get("source", {}) or {})
    if source.get("split") != "train":
        raise ValueError(
            "Validation partition requires a physically train-only sentence bank"
        )
    bank_id = str(bank.get("bank_id", ""))
    if not bank_id:
        raise ValueError("Sentence bank has no immutable bank_id")

    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    if not bool(partition_cfg.get("enabled", False)):
        raise ValueError("validation_text_partition.enabled must be true")
    if str(partition_cfg.get("split", "val")) != "val":
        raise ValueError("validation_text_partition.split must be 'val'")
    for name, actual in (
        ("development_text_count", int(development_text_count)),
        ("expected_novel_text_count", int(expected_novel_text_count)),
    ):
        configured = partition_cfg.get(name)
        if configured is not None and int(configured) != actual:
            raise ValueError(
                f"validation_text_partition.{name}={configured} conflicts with "
                f"the requested value {actual}"
            )
    expected_bank_id = partition_cfg.get("expected_bank_id")
    if expected_bank_id and bank_id != str(expected_bank_id):
        raise ValueError(
            f"Sentence bank ID mismatch: {bank_id} != {expected_bank_id}"
        )

    validation_rows = read_jsonl(validation_manifest)
    if not validation_rows:
        raise ValueError("Canonical validation manifest is empty")
    wrong_split = [
        index
        for index, row in enumerate(validation_rows)
        if _source_split(row) != "val"
    ]
    if wrong_split:
        raise ValueError(
            "Canonical validation manifest contains non-val source rows; first "
            f"offending index={wrong_split[0]}"
        )
    expected_rows = partition_cfg.get("expected_validation_rows")
    if expected_rows is not None and len(validation_rows) != int(expected_rows):
        raise ValueError(
            "Validation row count mismatch: "
            f"{len(validation_rows)} != {int(expected_rows)}"
        )

    validation_sha256 = sha256_file(validation_manifest)
    expected_manifest_sha256 = partition_cfg.get("expected_validation_manifest_sha256")
    if expected_manifest_sha256 and validation_sha256 != str(
        expected_manifest_sha256
    ):
        raise ValueError(
            "Validation manifest hash mismatch: "
            f"{validation_sha256} != {expected_manifest_sha256}"
        )

    metadata_name = str(
        dict(bank.get("artifacts", {}) or {})
        .get("metadata", {})
        .get("file", "items.jsonl")
    )
    metadata_path = bank_dir / metadata_name
    train_text_hashes = {
        str(row.get("text_hash", "")) for row in read_jsonl(metadata_path)
    }
    train_text_hashes.discard("")
    if not train_text_hashes:
        raise ValueError("Sentence-bank metadata has no train text hashes")

    texts = [str(row.get("text", "")) for row in validation_rows]
    exact_seen = [sentence_text_hash(text) in train_text_hashes for text in texts]
    partition = partition_validation_text_clusters(
        texts,
        exact_seen=exact_seen,
        development_text_count=int(development_text_count),
        expected_novel_text_count=int(expected_novel_text_count),
    )
    expected_partition_digest = partition_cfg.get("expected_partition_digest")
    if (
        expected_partition_digest
        and partition.partition_digest != str(expected_partition_digest)
    ):
        raise ValueError(
            "Validation partition digest mismatch: "
            f"{partition.partition_digest} != {expected_partition_digest}"
        )
    for count_name, actual_count in (
        (
            "expected_development_rows",
            sum(label == DEVELOPMENT for label in partition.labels),
        ),
        (
            "expected_confirmation_rows",
            sum(label == CONFIRMATION for label in partition.labels),
        ),
    ):
        configured_count = partition_cfg.get(count_name)
        if configured_count is not None and int(configured_count) != int(actual_count):
            raise ValueError(
                f"validation_text_partition.{count_name}={configured_count} "
                f"does not match {actual_count} rows"
            )
    rows_by_label = {
        label: [
            row
            for row, row_label in zip(validation_rows, partition.labels)
            if row_label == label
        ]
        for label in OUTPUT_FILES
    }
    payload = {
        **partition.artifact_payload,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "bank": {
            "path": str(bank_dir),
            "bank_id": bank_id,
            "bank_json_sha256": sha256_file(bank_dir / "bank.json"),
            "metadata_sha256": sha256_file(metadata_path),
            "source_split": "train",
        },
        "canonical_validation_manifest": {
            "path": str(validation_manifest),
            "sha256": validation_sha256,
            "row_count": len(validation_rows),
        },
    }
    payload["artifact_identity"] = _digest_json(payload)
    return payload, rows_by_label


def verify_partition_dir(
    out_dir: Path, expected_payload: dict[str, Any]
) -> dict[str, Any]:
    ready_path = out_dir / "READY"
    payload_path = out_dir / "partition.json"
    if not ready_path.is_file() or not payload_path.is_file():
        raise ValueError(f"Validation partition is not READY: {out_dir}")
    actual = json.loads(payload_path.read_text(encoding="utf-8"))
    if actual != expected_payload:
        raise ValueError("Existing validation partition identity does not match inputs")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("artifact_identity") != actual.get("artifact_identity"):
        raise ValueError("READY artifact identity does not match partition.json")
    for label, filename in OUTPUT_FILES.items():
        path = out_dir / filename
        if not path.is_file():
            raise ValueError(f"Validation partition is missing {filename}")
        expected = dict(actual.get("manifest_artifacts", {}) or {}).get(label)
        if not isinstance(expected, dict):
            raise ValueError(f"partition.json is missing {label} manifest identity")
        if sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"Validation partition hash mismatch for {filename}")
        if len(read_jsonl(path)) != int(expected.get("row_count", -1)):
            raise ValueError(f"Validation partition row-count mismatch for {filename}")
    return actual


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validation_manifest = _resolve_validation_manifest(cfg)
    bank_dir = _resolve_bank_dir(cfg, args.bank_dir)
    base_payload, rows_by_label = build_partition_payload(
        cfg=cfg,
        config_path=args.config,
        bank_dir=bank_dir,
        validation_manifest=validation_manifest,
        development_text_count=int(args.development_text_count),
        expected_novel_text_count=int(args.expected_novel_text_count),
    )
    out_dir = args.out_dir.resolve()
    expected_payload = dict(base_payload)
    expected_payload["manifest_artifacts"] = {
        label: {
            "file": filename,
            "row_count": len(rows_by_label[label]),
            "sha256": hashlib.sha256(
                "".join(_canonical_json(row) + "\n" for row in rows_by_label[label]).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        for label, filename in OUTPUT_FILES.items()
    }
    expected_payload["artifact_identity"] = _digest_json(
        {key: value for key, value in expected_payload.items() if key != "artifact_identity"}
    )

    if args.verify_only or out_dir.exists():
        verified = verify_partition_dir(out_dir, expected_payload)
        print(
            json.dumps(
                {
                    "valid": True,
                    "out_dir": str(out_dir),
                    "artifact_identity": verified["artifact_identity"],
                    "counts": verified["counts"],
                },
                indent=2,
            )
        )
        return

    building = out_dir.with_name(f".{out_dir.name}.building")
    if building.exists():
        raise ValueError(f"Refusing to overwrite incomplete build directory: {building}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        for label, filename in OUTPUT_FILES.items():
            _write_jsonl(building / filename, rows_by_label[label])
        _write_json(building / "partition.json", expected_payload)
        verify_payload = dict(expected_payload)
        # READY is written last. The final rename exposes a complete directory.
        _write_json(
            building / "READY",
            {
                "schema_name": expected_payload["schema_name"],
                "schema_version": expected_payload["schema_version"],
                "artifact_identity": expected_payload["artifact_identity"],
            },
        )
        verify_partition_dir(building, verify_payload)
        os.replace(building, out_dir)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise
    print(
        json.dumps(
            {
                "valid": True,
                "out_dir": str(out_dir),
                "artifact_identity": expected_payload["artifact_identity"],
                "counts": expected_payload["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
