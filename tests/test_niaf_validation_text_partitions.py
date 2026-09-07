from __future__ import annotations

import hashlib

import pytest

from NIAF.continuous_trajectory_field.validation_text_partitions import (
    CONFIRMATION,
    DEVELOPMENT,
    EXACT_SEEN,
    partition_validation_text_clusters,
)


def test_validation_partition_is_deterministic_and_keeps_signers_together():
    novel = [f"Novel sentence {index}" for index in range(796)]
    texts = novel + [novel[7].upper(), f"  {novel[19]}  ", "seen sentence"]
    exact_seen = [False] * 798 + [True]

    first = partition_validation_text_clusters(
        texts,
        exact_seen=exact_seen,
        seed=1234,
        development_text_count=256,
        expected_novel_text_count=796,
    )
    second = partition_validation_text_clusters(
        texts,
        exact_seen=exact_seen,
        seed=1234,
        development_text_count=256,
        expected_novel_text_count=796,
    )
    different_seed = partition_validation_text_clusters(
        texts,
        exact_seen=exact_seen,
        seed=9999,
        development_text_count=256,
        expected_novel_text_count=796,
    )

    assert first.partition_digest == second.partition_digest
    assert first.partition_digest == different_seed.partition_digest
    assert first.labels == different_seed.labels
    assert len(first.development_texts) == 256
    assert len(first.confirmation_texts) == 540
    assert len(first.exact_seen_texts) == 1
    assert first.labels[7] == first.labels[796]
    assert first.labels[19] == first.labels[797]
    assert first.labels[-1] == EXACT_SEEN
    assert set(first.development_texts).isdisjoint(first.confirmation_texts)
    assert set(first.development_texts).isdisjoint(first.exact_seen_texts)
    expected_first = min(
        first.development_texts + first.confirmation_texts,
        key=lambda text: (
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text,
        ),
    )
    assert first.development_texts[0] == expected_first
    counts = first.artifact_payload["counts"]
    assert first.artifact_payload["ordering"] == (
        "SHA256(normalized_text), normalized-text tiebreak"
    )
    assert "seed" not in first.artifact_payload
    assert counts["development_unique_texts"] == 256
    assert counts["confirmation_unique_texts"] == 540
    assert sum(first.row_mask(DEVELOPMENT)) + sum(
        first.row_mask(CONFIRMATION)
    ) + sum(first.row_mask(EXACT_SEEN)) == len(texts)


def test_validation_partition_rejects_inconsistent_seen_label_within_text_cluster():
    with pytest.raises(ValueError, match="inconsistent exact-seen"):
        partition_validation_text_clusters(
            ["Same text", " same   TEXT "],
            exact_seen=[False, True],
            development_text_count=1,
        )


def test_validation_partition_rejects_wrong_novel_population():
    with pytest.raises(ValueError, match="Unexpected number"):
        partition_validation_text_clusters(
            ["one", "two"],
            exact_seen=[False, False],
            development_text_count=1,
            expected_novel_text_count=796,
        )
