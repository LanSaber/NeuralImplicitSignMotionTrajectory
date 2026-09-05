from __future__ import annotations

import numpy as np
import pytest

from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryValidationError,
    exclusion_policy,
    sentence_text_hash,
)
from NIAF.continuous_trajectory_field.scripts.diagnose_sentence_retrieval import (
    deterministic_random_group,
    diagnose_query,
    normalized_time_latent_rmse,
    prepare_query_row,
    summarize_records,
    validate_neighbor_ranking,
)


def _diagnostic_fixture():
    texts = ("forbidden", "direct", "oracle", "random")
    groups = [
        {"semantic_group_index": index, "semantic_group_id": sentence_text_hash(text)}
        for index, text in enumerate(texts)
    ]
    metadata = [
        {
            "name": "same-source",
            "motion_path": "same.npz",
            "text": texts[0],
            "text_hash": sentence_text_hash(texts[0]),
            "semantic_group_id": sentence_text_hash(texts[0]),
            "source_id": "query-source",
            "source_group_id": "sentence_id:query-source",
        },
        {
            "name": "direct",
            "motion_path": "direct.npz",
            "text": texts[1],
            "text_hash": sentence_text_hash(texts[1]),
            "semantic_group_id": sentence_text_hash(texts[1]),
            "source_id": "source-1",
            "source_group_id": "sentence_id:source-1",
        },
        {
            "name": "oracle-low-validity",
            "motion_path": "oracle-low.npz",
            "text": texts[2],
            "text_hash": sentence_text_hash(texts[2]),
            "semantic_group_id": sentence_text_hash(texts[2]),
            "source_id": "source-2a",
            "source_group_id": "sentence_id:source-2a",
        },
        {
            "name": "oracle-high-validity",
            "motion_path": "oracle-high.npz",
            "text": texts[2],
            "text_hash": sentence_text_hash(texts[2]),
            "semantic_group_id": sentence_text_hash(texts[2]),
            "source_id": "source-2b",
            "source_group_id": "sentence_id:source-2b",
        },
        {
            "name": "random",
            "motion_path": "random.npz",
            "text": texts[3],
            "text_hash": sentence_text_hash(texts[3]),
            "semantic_group_id": sentence_text_hash(texts[3]),
            "source_id": "source-3",
            "source_group_id": "sentence_id:source-3",
        },
    ]
    lengths = np.asarray([2, 3, 4, 2, 5], dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    item_values = (3.0, 1.0, 0.1, 0.5, 2.0)
    motion = np.concatenate(
        [
            np.full((length, 2), value, dtype=np.float16)
            for length, value in zip(lengths, item_values)
        ]
    )
    hand_values = (255, 180, 20, 255, 128)
    hand_valid = np.concatenate(
        [
            np.full((length, 2), value, dtype=np.uint8)
            for length, value in zip(lengths, hand_values)
        ]
    )
    group_keys = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.0, 1.0]], dtype=np.float32
    )
    group_keys /= np.linalg.norm(group_keys, axis=1, keepdims=True)
    return {
        "metadata": metadata,
        "groups": groups,
        "group_keys": group_keys,
        "group_offsets": np.asarray([0, 1, 2, 4, 5], dtype=np.int64),
        "group_item_ids": np.arange(5, dtype=np.int32),
        "motion": motion,
        "offsets": offsets,
        "hand_valid": hand_valid,
        "durations": np.asarray([2.0, 4.0, 5.0, 6.0, 8.0], dtype=np.float32),
        "policy": exclusion_policy(),
    }


def _query(text="novel query"):
    return {
        "name": "query",
        "motion_path": "query.npz",
        "text": text,
        "source_name": "query-source",
        "sentence_id": "query-source",
    }


def _diagnose(fixture, query, neighbor_ids=(1, 2, -1), neighbor_scores=(0.9, 0.8, 0.0)):
    return diagnose_query(
        query=query,
        query_motion=np.zeros((3, 2), dtype=np.float32),
        query_duration=4.0,
        neighbor_ids=np.asarray(neighbor_ids, dtype=np.int64),
        neighbor_scores=np.asarray(neighbor_scores, dtype=np.float32),
        top_k=2,
        seed=17,
        training=False,
        latent_mean=np.zeros(2, dtype=np.float32),
        latent_std=np.ones(2, dtype=np.float32),
        resample_points=8,
        **fixture,
    )


def test_normalized_time_distance_ignores_sampling_rate_not_latent_error():
    sparse = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    dense = np.asarray([[0.0], [0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
    shifted = dense + 0.25
    kwargs = {
        "latent_mean": np.zeros(1, dtype=np.float32),
        "latent_std": np.ones(1, dtype=np.float32),
        "points": 17,
    }

    assert normalized_time_latent_rmse(sparse, dense, **kwargs) == pytest.approx(0.0)
    assert normalized_time_latent_rmse(sparse, shifted, **kwargs) == pytest.approx(0.25)


def test_diagnostic_reports_direct_random_oracle_and_seen_novel_subsets():
    fixture = _diagnostic_fixture()
    novel = _diagnose(fixture, _query())
    seen = _diagnose(fixture, _query("direct"))

    assert novel["subset"] == "novel_text"
    assert seen["subset"] == "exact_seen_text"
    assert novel["top1"]["group_id"] == 1
    assert novel["top1"]["latent_ntrmse"] == pytest.approx(1.0)
    # Normal retrieval selects the high-validity recording from group 2, but
    # the diagnostic oracle may inspect target motion and selects item 2.
    assert novel["oracle_top_k"]["group_id"] == 2
    assert novel["oracle_top_k"]["item_id"] == 2
    assert novel["oracle_top_k"]["latent_ntrmse"] == pytest.approx(0.1, abs=3e-5)
    assert novel["deterministic_random"] is not None
    assert novel["deterministic_random"]["group_id"] != 0
    assert novel["candidate_set"]["unique_semantic_group_fraction"] == 1.0
    assert novel["candidate_set"]["mean_abs_log_duration_gap"] is not None

    summary = summarize_records([novel, seen])
    assert summary["queries"] == 2
    assert summary["usable_queries"] == 2
    assert summary["latent_motion_distance"]["direct_top1"]["count"] == 2
    assert summary["latent_motion_distance"]["diagnostic_oracle_top_k"]["mean"] < 1.0


def test_forbidden_neighbor_fails_closed_and_random_choice_is_reproducible():
    fixture = _diagnostic_fixture()
    query = _query()
    with pytest.raises(SentenceMemoryValidationError, match="forbidden neighbor group"):
        _diagnose(fixture, query, neighbor_ids=(0, 1), neighbor_scores=(1.0, 0.9))

    prepared = prepare_query_row(query, fixture["policy"])

    def eligible(group_id):
        return [] if group_id == 0 else [group_id]

    first = deterministic_random_group(
        prepared, group_count=4, seed=29, eligible_items_for_group=eligible
    )
    second = deterministic_random_group(
        prepared, group_count=4, seed=29, eligible_items_for_group=eligible
    )
    assert first == second
    assert first is not None and first[0] != 0


def test_neighbor_ranking_requires_cosine_descending_scores_and_trailing_padding():
    valid = {
        "top_m": 3,
        "ids": np.asarray([[2, 1, -1]], dtype=np.int64),
        "scores": np.asarray([[0.9, 0.8, 0.0]], dtype=np.float32),
        "scoring": {"metric": "cosine"},
    }
    validate_neighbor_ranking(valid)

    with pytest.raises(SentenceMemoryValidationError, match="not descending"):
        validate_neighbor_ranking(
            {**valid, "scores": np.asarray([[0.8, 0.9, 0.0]], dtype=np.float32)}
        )
    with pytest.raises(SentenceMemoryValidationError, match="after null padding"):
        validate_neighbor_ranking(
            {
                **valid,
                "ids": np.asarray([[2, -1, 1]], dtype=np.int64),
                "scores": np.asarray([[0.9, 0.0, 0.8]], dtype=np.float32),
            }
        )
