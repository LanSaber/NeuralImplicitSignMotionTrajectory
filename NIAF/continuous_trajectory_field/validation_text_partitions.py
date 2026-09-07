from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from NIAF.continuous_trajectory_field.sentence_memory import (
    normalize_sentence_text,
)


SCHEMA_NAME = "signtrajfield_validation_text_cluster_partition"
SCHEMA_VERSION = 1
DEFAULT_SEED = 1234
DEFAULT_DEVELOPMENT_TEXT_COUNT = 256

DEVELOPMENT = "development"
CONFIRMATION = "confirmation"
EXACT_SEEN = "exact_seen"
LABELS = (DEVELOPMENT, CONFIRMATION, EXACT_SEEN)


class NormalizedTextClusterBatchSampler:
    """Assign complete normalized-text clusters to ranks without padding.

    The sampler emits one logical text cluster per loader batch.  Cluster
    ownership is a stable round-robin partition of the SHA256 text order, so a
    repeated signer realization can never migrate to another rank and no row
    is duplicated to equalize shard sizes.  ``set_epoch`` is intentionally a
    no-op: validation order and ownership are checkpoint-independent.
    """

    def __init__(
        self,
        normalized_texts: Sequence[str],
        *,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )

        clusters: dict[str, list[int]] = {}
        for row_index, text in enumerate(normalized_texts):
            normalized = normalize_sentence_text(text)
            if not normalized:
                raise ValueError(
                    "Cluster-equal validation requires non-empty normalized text"
                )
            clusters.setdefault(normalized, []).append(int(row_index))
        ordered_texts = sorted(clusters, key=validation_text_order_key)
        owned_texts = ordered_texts[self.rank :: self.num_replicas]
        self._batches = tuple(tuple(clusters[text]) for text in owned_texts)
        self.cluster_texts = tuple(owned_texts)
        self.row_count = sum(len(batch) for batch in self._batches)
        self.epoch = 0

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        # DataLoader/trainer compatibility only.  Scientific validation maps
        # and ordering must not depend on the training epoch.
        self.epoch = int(epoch)


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


def validation_text_order_key(normalized_text: str) -> tuple[str, str]:
    """Return the stable, platform-independent order key for one text cluster."""

    normalized = normalize_sentence_text(normalized_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest, normalized


@dataclass(frozen=True)
class ValidationTextClusterPartition:
    """A deterministic row-aligned partition of validation text clusters."""

    labels: tuple[str, ...]
    normalized_texts: tuple[str, ...]
    development_texts: tuple[str, ...]
    confirmation_texts: tuple[str, ...]
    exact_seen_texts: tuple[str, ...]
    partition_digest: str
    artifact_payload: dict[str, Any]

    def row_mask(self, label: str) -> tuple[bool, ...]:
        if label not in LABELS:
            raise ValueError(f"Unknown validation partition label: {label!r}")
        return tuple(value == label for value in self.labels)


def partition_validation_text_clusters(
    texts: Sequence[str],
    *,
    exact_seen: Sequence[bool],
    seed: int = DEFAULT_SEED,
    development_text_count: int = DEFAULT_DEVELOPMENT_TEXT_COUNT,
    expected_novel_text_count: int | None = None,
) -> ValidationTextClusterPartition:
    """Split validation rows by normalized text without separating signers.

    Exact train-bank texts are excluded from both scientific partitions. Novel
    normalized texts are ordered by ``SHA256(normalized)`` with the normalized
    text as a deterministic collision tiebreak. The retained ``seed`` argument
    is compatibility-only and cannot affect membership. The first
    ``development_text_count`` clusters form the development set and all
    remaining novel clusters form the post-lock confirmation set.
    """

    text_values = tuple(str(value) for value in texts)
    seen_values = tuple(bool(value) for value in exact_seen)
    if len(text_values) != len(seen_values):
        raise ValueError(
            "texts and exact_seen must have identical row counts: "
            f"{len(text_values)} != {len(seen_values)}"
        )
    if not text_values:
        raise ValueError("Validation partition requires at least one row")
    if int(development_text_count) < 0:
        raise ValueError("development_text_count must be non-negative")
    # Kept for API compatibility with already shared trainer call sites. The
    # scientific partition is deliberately seed-independent.
    _ = seed

    normalized_rows = tuple(normalize_sentence_text(value) for value in text_values)
    if any(not value for value in normalized_rows):
        raise ValueError("Validation texts must be non-empty after normalization")

    cluster_seen: dict[str, bool] = {}
    cluster_row_count: dict[str, int] = {}
    for normalized, seen in zip(normalized_rows, seen_values):
        previous = cluster_seen.setdefault(normalized, seen)
        if previous != seen:
            raise ValueError(
                "Repeated normalized validation text has inconsistent exact-seen "
                f"labels: {normalized!r}"
            )
        cluster_row_count[normalized] = cluster_row_count.get(normalized, 0) + 1

    novel_texts = sorted(
        (text for text, seen in cluster_seen.items() if not seen),
        key=validation_text_order_key,
    )
    exact_seen_texts = sorted(
        (text for text, seen in cluster_seen.items() if seen),
        key=validation_text_order_key,
    )
    if expected_novel_text_count is not None and len(novel_texts) != int(
        expected_novel_text_count
    ):
        raise ValueError(
            "Unexpected number of unique novel validation texts: "
            f"{len(novel_texts)} != {int(expected_novel_text_count)}"
        )
    if int(development_text_count) > len(novel_texts):
        raise ValueError(
            "development_text_count exceeds the number of unique novel texts: "
            f"{int(development_text_count)} > {len(novel_texts)}"
        )

    development = tuple(novel_texts[: int(development_text_count)])
    confirmation = tuple(novel_texts[int(development_text_count) :])
    exact = tuple(exact_seen_texts)
    label_by_text = {
        **{value: DEVELOPMENT for value in development},
        **{value: CONFIRMATION for value in confirmation},
        **{value: EXACT_SEEN for value in exact},
    }
    row_labels = tuple(label_by_text[value] for value in normalized_rows)

    assignments = []
    for normalized in (*development, *confirmation, *exact):
        order_digest, _ = validation_text_order_key(normalized)
        assignments.append(
            {
                "normalized_text": normalized,
                "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "order_sha256": order_digest,
                "label": label_by_text[normalized],
                "row_count": int(cluster_row_count[normalized]),
            }
        )

    payload_without_digest: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": "val",
        "validation_only": True,
        "development_text_count_requested": int(development_text_count),
        "ordering": "SHA256(normalized_text), normalized-text tiebreak",
        "normalization": "NFKC, casefold, whitespace collapse",
        "counts": {
            "rows": len(text_values),
            "unique_texts": len(cluster_seen),
            "novel_unique_texts": len(novel_texts),
            "development_unique_texts": len(development),
            "confirmation_unique_texts": len(confirmation),
            "exact_seen_unique_texts": len(exact),
            "development_rows": sum(label == DEVELOPMENT for label in row_labels),
            "confirmation_rows": sum(label == CONFIRMATION for label in row_labels),
            "exact_seen_rows": sum(label == EXACT_SEEN for label in row_labels),
        },
        "assignments": assignments,
        "row_labels": list(row_labels),
    }
    partition_digest = _digest_json(payload_without_digest)
    payload = {**payload_without_digest, "partition_digest": partition_digest}
    return ValidationTextClusterPartition(
        labels=row_labels,
        normalized_texts=normalized_rows,
        development_texts=development,
        confirmation_texts=confirmation,
        exact_seen_texts=exact,
        partition_digest=partition_digest,
        artifact_payload=payload,
    )
