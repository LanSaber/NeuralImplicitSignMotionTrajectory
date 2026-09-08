from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    configure_sentence_memory_trainable_parameters,
    is_sentence_memory_parameter,
)
from NIAF.continuous_trajectory_field.scripts.build_sentence_neighbors import (
    exact_chunked_cosine_topk,
)
from NIAF.continuous_trajectory_field.sentence_memory import (
    DEFAULT_ARTIFACT_FILES,
    SentenceMemoryBatch,
    SentenceMemoryProvider,
    SentenceMemoryValidationError,
    artifact_record,
    candidate_is_allowed,
    complete_text_encoder_identity,
    exclusion_policy,
    load_bank_manifest,
    motion_only_shuffle_sentence_memory_batch,
    sentence_text_hash,
    sentence_memory_preprocessing_contract,
    sha256_file,
    validate_sentence_memory_bank,
    write_bank_manifest,
    write_build_summary,
    write_neighbor_table,
    write_ready_marker,
    _validate_finite_array_chunked,
)


def _motion_shuffle_batch(
    candidate_mask: torch.Tensor,
    *,
    token_lengths: torch.Tensor | None = None,
) -> SentenceMemoryBatch:
    batch_size, candidate_count = candidate_mask.shape
    token_count = 4
    if token_lengths is None:
        token_lengths = candidate_mask.long() * token_count
    token_index = torch.arange(token_count).view(1, 1, token_count)
    token_mask = token_index < token_lengths.unsqueeze(-1)
    rank_value = (
        torch.arange(batch_size).view(batch_size, 1, 1) * 100
        + torch.arange(candidate_count).view(1, candidate_count, 1) * 10
        + token_index
    )
    tokens = torch.stack((rank_value, -rank_value), dim=-1).float()
    token_tau = rank_value.float() / 100.0
    part_validity = torch.stack(
        tuple(token_mask.float() * float(index + 1) / 4.0 for index in range(4)),
        dim=-1,
    )
    candidate_keys = torch.arange(
        batch_size * candidate_count * 3, dtype=torch.float32
    ).reshape(batch_size, candidate_count, 3)
    scores = torch.arange(batch_size * candidate_count, dtype=torch.float32).reshape(
        batch_size, candidate_count
    )
    ids = torch.arange(batch_size * candidate_count, dtype=torch.int64).reshape(
        batch_size, candidate_count
    )
    ids = torch.where(candidate_mask, ids, torch.full_like(ids, -1))
    return SentenceMemoryBatch(
        tokens=tokens,
        token_mask=token_mask,
        token_tau=token_tau,
        candidate_mask=candidate_mask.clone(),
        candidate_keys=candidate_keys,
        scores=scores,
        durations=scores + 1.0,
        duration_log_gap=scores / 10.0,
        part_validity=part_validity,
        ids=ids,
        available=candidate_mask.any(dim=-1),
        provenance={
            "mode": "on",
            "candidate_names": [
                [f"row-{row}-rank-{rank}" for rank in range(candidate_count)]
                for row in range(batch_size)
            ],
            "nested": {"unchanged": [1, 2, 3]},
        },
    )


def _index_memory_batch(
    memory: SentenceMemoryBatch, order: torch.Tensor
) -> SentenceMemoryBatch:
    updates = {}
    for name in (
        "tokens",
        "token_mask",
        "token_tau",
        "candidate_mask",
        "candidate_keys",
        "scores",
        "durations",
        "duration_log_gap",
        "part_validity",
        "ids",
        "available",
    ):
        updates[name] = getattr(memory, name).index_select(0, order)
    return replace(memory, **updates)


def test_motion_only_shuffle_deranges_only_motion_payload_and_records_sources():
    candidate_mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
        ]
    )
    token_lengths = torch.tensor(
        [
            [4, 0, 2, 1, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    memory = _motion_shuffle_batch(candidate_mask, token_lengths=token_lengths)
    original_provenance = json.loads(json.dumps(memory.provenance))

    corrupted, permutation, informative = (
        motion_only_shuffle_sentence_memory_batch(
            memory,
            query_ids=("three-valid", "one-valid", "zero-valid"),
            epoch=4,
            seed=1234,
        )
    )

    assert informative.tolist() == [True, False, False]
    valid_ranks = torch.tensor([0, 2, 3])
    assert torch.all(permutation[0, valid_ranks] != valid_ranks)
    assert permutation[0, [1, 4]].tolist() == [1, 4]
    assert torch.equal(permutation[1], torch.arange(5))
    assert torch.equal(permutation[2], torch.arange(5))

    for name in ("tokens", "token_mask", "token_tau", "part_validity"):
        original = getattr(memory, name)
        actual = getattr(corrupted, name)
        for destination, source in enumerate(permutation[0].tolist()):
            assert torch.equal(actual[0, destination], original[0, source])
        assert torch.equal(actual[1:], original[1:])

    for name in (
        "candidate_mask",
        "candidate_keys",
        "scores",
        "durations",
        "duration_log_gap",
        "ids",
        "available",
    ):
        assert getattr(corrupted, name) is getattr(memory, name)
        assert torch.equal(getattr(corrupted, name), getattr(memory, name))
    assert memory.provenance == original_provenance
    for key, value in original_provenance.items():
        if key != "mode":
            assert corrupted.provenance[key] == value
    assert corrupted.provenance["mode"] == "motion_only_shuffle"
    assert corrupted.provenance["motion_only_shuffle_source_mode"] == "on"
    assert corrupted.provenance["motion_only_shuffle_epoch"] == 4
    assert corrupted.provenance["motion_only_shuffle_seed"] == 1234
    assert corrupted.provenance["motion_candidate_permutation"] == permutation.tolist()
    assert corrupted.provenance["motion_only_shuffle_informative"] == [
        True,
        False,
        False,
    ]
    expected_source_ids = torch.gather(memory.ids, 1, permutation)
    assert corrupted.provenance["motion_source_ids"] == expected_source_ids.tolist()


def test_motion_only_shuffle_is_independent_of_batch_order_rank_and_runtime_rng(
    monkeypatch,
):
    memory = _motion_shuffle_batch(torch.ones(4, 5, dtype=torch.bool))
    query_ids = ("query-a", "query-b", "query-c", "query-d")
    monkeypatch.setenv("RANK", "0")
    first, first_permutation, first_informative = (
        motion_only_shuffle_sentence_memory_batch(
            memory,
            query_ids=query_ids,
            epoch=7,
            seed=81,
        )
    )
    torch.manual_seed(987654)
    _ = torch.randn(29)
    monkeypatch.setenv("RANK", "17")
    resumed, resumed_permutation, resumed_informative = (
        motion_only_shuffle_sentence_memory_batch(
            memory,
            query_ids=query_ids,
            epoch=7,
            seed=81,
        )
    )
    assert torch.equal(resumed_permutation, first_permutation)
    assert torch.equal(resumed_informative, first_informative)
    assert torch.equal(resumed.tokens, first.tokens)

    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    reordered, reordered_permutation, reordered_informative = (
        motion_only_shuffle_sentence_memory_batch(
            _index_memory_batch(memory, order),
            query_ids=tuple(query_ids[index] for index in order.tolist()),
            epoch=7,
            seed=81,
        )
    )
    assert torch.equal(reordered_permutation.index_select(0, inverse), first_permutation)
    assert torch.equal(reordered_informative.index_select(0, inverse), first_informative)
    assert torch.equal(reordered.tokens.index_select(0, inverse), first.tokens)

    epoch_permutations = {
        tuple(
            motion_only_shuffle_sentence_memory_batch(
                memory,
                query_ids=query_ids,
                epoch=epoch,
                seed=81,
            )[1][0].tolist()
        )
        for epoch in range(8)
    }
    assert len(epoch_permutations) >= 2


def test_motion_only_shuffle_handles_padding_and_performs_no_file_io(monkeypatch):
    candidate_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    memory = _motion_shuffle_batch(
        candidate_mask,
        token_lengths=torch.tensor([[3, 0, 0], [1, 0, 0], [0, 0, 0]]),
    )

    def reject_io(*_args, **_kwargs):
        raise AssertionError("motion-only corruption must not touch the bank")

    monkeypatch.setattr(Path, "open", reject_io)
    corrupted, permutation, informative = (
        motion_only_shuffle_sentence_memory_batch(
            memory,
            query_ids=("empty-second", "one", "zero"),
            epoch=2,
            seed=3,
        )
    )
    assert not informative.any()
    assert torch.equal(permutation, torch.arange(3).expand(3, -1))
    for name in ("tokens", "token_mask", "token_tau", "part_validity"):
        assert torch.equal(getattr(corrupted, name), getattr(memory, name))


def test_motion_only_shuffle_rejects_inconsistent_batch_geometry():
    memory = _motion_shuffle_batch(torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="one stable identity"):
        motion_only_shuffle_sentence_memory_batch(
            memory,
            query_ids=("only-one",),
            epoch=1,
            seed=1,
        )
    malformed = replace(memory, token_tau=memory.token_tau[:, :, :-1])
    with pytest.raises(ValueError, match="token_tau"):
        motion_only_shuffle_sentence_memory_batch(
            malformed,
            query_ids=("first", "second"),
            epoch=1,
            seed=1,
        )


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unit_rows():
    return [
        {
            "name": "query-self",
            "motion_path": "motion/self.npz",
            "text": "Same   Prompt",
            "source_name": "source-q",
            "sentence_id": "source-q",
            "duration": 5.0,
        },
        {
            "name": "same-prompt-other-source",
            "motion_path": "motion/same-other.npz",
            "text": "same prompt",
            "source_name": "source-1",
            "sentence_id": "source-1",
            "duration": 4.0,
        },
        {
            "name": "different-prompt-same-source",
            "motion_path": "motion/same-source.npz",
            "text": "close source",
            "source_name": "source-q",
            "sentence_id": "source-q",
            "duration": 4.0,
        },
        {
            "name": "allowed-a",
            "motion_path": "motion/a.npz",
            "text": "allowed a",
            "source_name": "source-3",
            "sentence_id": "source-3",
            "duration": 4.0,
        },
        {
            "name": "allowed-b",
            "motion_path": "motion/b.npz",
            "text": "allowed b",
            "source_name": "source-4",
            "sentence_id": "source-4",
            "duration": 8.0,
        },
        {
            "name": "allowed-c-1",
            "motion_path": "motion/c1.npz",
            "text": "allowed c",
            "source_name": "source-5",
            "sentence_id": "source-5",
            "duration": 6.0,
        },
        {
            "name": "allowed-c-2",
            "motion_path": "motion/c2.npz",
            "text": "ALLOWED C",
            "source_name": "source-6",
            "sentence_id": "source-6",
            "duration": 6.0,
        },
    ]


def _make_bank(tmp_path: Path):
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    text_dir = tmp_path / "tiny-text-encoder"
    text_dir.mkdir()
    (text_dir / "config.json").write_text(
        json.dumps({"model_type": "t5", "d_model": 3}), encoding="utf-8"
    )
    (text_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 32}), encoding="utf-8"
    )
    (text_dir / "spiece.model").write_bytes(b"unit-test-tokenizer")
    (text_dir / "pytorch_model.bin").write_bytes(b"unit-test-weights")
    text_identity = {
        "model_type": "t5",
        "model_path": str(text_dir),
        "text_dim": 3,
        "max_length": 32,
    }
    complete_identity = complete_text_encoder_identity(text_identity)

    rows = _unit_rows()
    train_manifest = tmp_path / "manifest_train.jsonl"
    _write_jsonl(train_manifest, rows)
    vae_checkpoint = tmp_path / "vae.pt"
    vae_checkpoint.write_bytes(b"unit-test-vae")
    data_dir = tmp_path / "motion-data"
    stats_dir = data_dir / "meta"
    stats_dir.mkdir(parents=True)
    mean_path = stats_dir / "mean_rot6d.npy"
    std_path = stats_dir / "std_rot6d.npy"
    np.save(mean_path, np.asarray([0.0, 0.5], dtype=np.float32))
    np.save(std_path, np.asarray([1.0, 2.0], dtype=np.float32))

    # Groups 0 and 4 deliberately contain two recording realizations of one
    # normalized sentence; retrieval keys are group-level, payloads item-level.
    item_group_ids = np.asarray([0, 0, 1, 2, 3, 4, 4], dtype=np.int32)
    group_offsets = np.asarray([0, 2, 3, 4, 5, 7], dtype=np.int64)
    group_item_ids = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    group_keys = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9949372, 0.1004987, 0.0],
            [0.9, 0.4358899, 0.0],
            [0.8, 0.6, 0.0],
            [0.7, 0.71414286, 0.0],
        ],
        dtype=np.float32,
    )
    group_keys /= np.linalg.norm(group_keys, axis=1, keepdims=True)

    latent_lengths = np.asarray([2, 3, 2, 4, 3, 2, 5], dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(latent_lengths)
    motion = np.arange(int(offsets[-1]) * 2, dtype=np.float16).reshape(-1, 2) / 10
    hand_valid_float = np.stack(
        [
            np.linspace(0.1, 0.9, len(motion), dtype=np.float16),
            np.linspace(0.9, 0.1, len(motion), dtype=np.float16),
        ],
        axis=1,
    )
    hand_valid_float[14:16] = 0.2
    hand_valid_float[16:21] = 0.9
    hand_valid = np.clip(
        np.rint(hand_valid_float.astype(np.float32) * 255.0), 0, 255
    ).astype(np.uint8)
    durations = np.asarray([row["duration"] for row in rows], dtype=np.float32)

    arrays = {
        "group_keys": group_keys,
        "group_offsets": group_offsets,
        "group_item_ids": group_item_ids,
        "item_group_ids": item_group_ids,
        "motion_mu": motion,
        "offsets": offsets,
        "item_motion_lengths": latent_lengths.astype(np.int16),
        "hand_valid": hand_valid,
        "durations": durations,
    }
    for name, array in arrays.items():
        np.save(bank_dir / DEFAULT_ARTIFACT_FILES[name], array, allow_pickle=False)

    metadata = []
    for item_index, row in enumerate(rows):
        group_index = int(item_group_ids[item_index])
        metadata.append(
            {
                **row,
                "bank_index": item_index,
                "encoded_frame_length": int(latent_lengths[item_index] * 4),
                "latent_length": int(latent_lengths[item_index]),
                "text_hash": sentence_text_hash(row["text"]),
                "semantic_group_id": sentence_text_hash(row["text"]),
                "semantic_group_index": group_index,
                "source_id": row["source_name"],
                "source_group_id": f"sentence_id:{row['sentence_id']}",
            }
        )
    group_rows = []
    for group_index in range(len(group_keys)):
        member_ids = group_item_ids[
            group_offsets[group_index] : group_offsets[group_index + 1]
        ]
        first = metadata[int(member_ids[0])]
        group_rows.append(
            {
                "semantic_group_index": group_index,
                "semantic_group_id": first["semantic_group_id"],
                "text_hash": first["text_hash"],
                "text": first["text"],
                "member_count": len(member_ids),
            }
        )
    _write_jsonl(bank_dir / DEFAULT_ARTIFACT_FILES["metadata"], metadata)
    _write_jsonl(bank_dir / DEFAULT_ARTIFACT_FILES["groups"], group_rows)

    artifacts = {
        name: artifact_record(bank_dir / filename)
        for name, filename in DEFAULT_ARTIFACT_FILES.items()
    }
    policy = exclusion_policy()
    write_bank_manifest(
        bank_dir,
        {
            "source": {
                "split": "train",
                "manifest_path": str(train_manifest),
                "manifest_sha256": sha256_file(train_manifest),
                "row_count": len(rows),
                "semantic_group_count": len(group_keys),
            },
            "text_encoder": complete_identity,
            "codec": {
                "latent_dim": 2,
                "downsample_factor": 4,
                "checkpoint_path": str(vae_checkpoint),
                "checkpoint_sha256": sha256_file(vae_checkpoint),
                "mean_sha256": sha256_file(mean_path),
                "std_sha256": sha256_file(std_path),
            },
            "preprocessing": {
                "rotation_rep": "rot6d",
                "min_frames": 40,
                "max_frames": 400,
                "length_multiple": 4,
                "random_crop": False,
            },
            "filter_policy": policy,
            "artifacts": artifacts,
        },
    )
    write_build_summary(
        bank_dir,
        {
            "row_count": len(rows),
            "semantic_group_count": len(group_keys),
            "latent_tokens": len(motion),
        },
    )
    write_ready_marker(bank_dir)
    return {
        "bank_dir": bank_dir,
        "text_dir": text_dir,
        "text_identity": text_identity,
        "train_manifest": train_manifest,
        "vae_checkpoint": vae_checkpoint,
        "data_dir": data_dir,
        "mean_path": mean_path,
        "std_path": std_path,
        "rows": rows,
        "keys": group_keys,
        "hand_valid": hand_valid,
    }


def _query_dataset(tmp_path: Path, row, *, split="test"):
    manifest = tmp_path / f"manifest_{split}.jsonl"
    _write_jsonl(manifest, [row])
    return SimpleNamespace(
        split=split,
        base=SimpleNamespace(items=[dict(row)], manifest_path=manifest),
    )


def _batch(row):
    return {
        "index": torch.tensor([0]),
        "name": [row["name"]],
        "motion_path": [row["motion_path"]],
        "text": [row["text"]],
    }


def _provider(bank, dataset, **memory_overrides):
    memory_cfg = {
        "bank_dir": str(bank["bank_dir"]),
        "k": 2,
        "top_m": 4,
        "duration_weight": 0.0,
        "sampling": "topk",
        "candidate_dropout_probability": 0.0,
        **memory_overrides,
    }
    return SentenceMemoryProvider(
        {"seed": 7, "sentence_memory": memory_cfg},
        text_encoder_identity=bank["text_identity"],
        dataset=dataset,
    )


def test_bank_validation_checks_grouped_layout_and_semantic_artifacts(tmp_path):
    bank = _make_bank(tmp_path)
    manifest = validate_sentence_memory_bank(
        bank["bank_dir"],
        expected_manifest_path=bank["train_manifest"],
        text_encoder_identity=bank["text_identity"],
        expected_vae_checkpoint=bank["vae_checkpoint"],
        expected_preprocessing=sentence_memory_preprocessing_contract(
            {"data": {}}
        ),
        expected_mean_path=bank["mean_path"],
        expected_std_path=bank["std_path"],
        verify_hashes=True,
        verify_contents=True,
    )

    assert manifest["source"]["row_count"] == 7
    assert manifest["source"]["semantic_group_count"] == 5
    assert manifest["bank_id"] == load_bank_manifest(bank["bank_dir"])["bank_id"]


@pytest.mark.parametrize(
    ("artifact", "index", "value", "message"),
    (
        ("group_keys", (3, 1), np.nan, "Non-finite values in group_keys"),
        ("motion_mu", (7, 0), np.inf, "Non-finite values in motion_mu"),
        ("durations", (4,), np.nan, "Non-finite values in durations"),
    ),
)
def test_runtime_provider_rejects_nonfinite_payload_without_hash_verification(
    tmp_path, artifact, index, value, message
):
    bank = _make_bank(tmp_path)
    path = bank["bank_dir"] / DEFAULT_ARTIFACT_FILES[artifact]
    array = np.load(path, mmap_mode="r+", allow_pickle=False)
    array[index] = value
    array.flush()
    del array

    query = bank["rows"][0]
    with pytest.raises(SentenceMemoryValidationError, match=message):
        _provider(
            bank,
            _query_dataset(tmp_path, query),
            verify_artifact_hashes=False,
            content_scan_chunk_bytes=16,
        )


def test_finite_scan_never_requests_the_complete_payload():
    class ChunkGuard:
        def __init__(self):
            self.values = np.ones((11, 4), dtype=np.float32)
            self.shape = self.values.shape
            self.dtype = self.values.dtype
            self.max_rows = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            assert isinstance(index, slice)
            rows = int(index.stop) - int(index.start)
            self.max_rows = max(self.max_rows, rows)
            assert rows <= 2
            return self.values[index]

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("the complete payload must not be materialized")

    guarded = ChunkGuard()
    _validate_finite_array_chunked(
        guarded,
        name="guarded",
        max_chunk_bytes=2 * 4 * np.dtype(np.float32).itemsize,
    )
    assert guarded.max_rows == 2


def test_chunked_exact_neighbors_match_full_cosine_ranking_and_exclusions():
    keys = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    allowed = np.asarray(
        [
            [False, True, True, True, False, True],
            [True, True, False, False, True, True],
        ],
        dtype=np.bool_,
    )
    normalized_keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    normalized_query = query.numpy() / np.linalg.norm(
        query.numpy(), axis=1, keepdims=True
    )
    full_scores = normalized_query @ normalized_keys.T
    full_scores[~allowed] = -np.inf
    group_ids = np.arange(len(keys))
    expected_ids = np.stack(
        [np.lexsort((group_ids, -row))[:4] for row in full_scores]
    )
    expected_scores = np.take_along_axis(full_scores, expected_ids, axis=1)

    for chunk_size in (1, 2, 5, len(keys)):
        scores, ids = exact_chunked_cosine_topk(
            query,
            keys,
            allowed,
            top_m=4,
            search_device=torch.device("cpu"),
            key_chunk_size=chunk_size,
        )
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(scores, expected_scores, rtol=0.0, atol=1e-7)

    # The excluded exact match (group 0) must not leak into query 0, and equal
    # scores use ascending semantic-group ID for deterministic ordering.
    assert expected_ids[0, 0] == 1
    assert expected_ids[1, :2].tolist() == [0, 1]


def test_provider_identity_binds_neighbor_table_content_without_physical_path(
    tmp_path,
):
    bank = _make_bank(tmp_path)
    manifest = load_bank_manifest(bank["bank_dir"])
    neighbor_path = bank["bank_dir"] / "neighbors_val.npz"

    def write(scores):
        write_neighbor_table(
            neighbor_path,
            bank_id=manifest["bank_id"],
            query_split="val",
            query_manifest_sha256="synthetic-query-manifest",
            query_names=["query"],
            ids=np.asarray([[3, 4]], dtype=np.int32),
            scores=np.asarray([scores], dtype=np.float32),
            policy=manifest["filter_policy"],
            scoring={"metric": "cosine", "key_space": "semantic_groups"},
        )

    write([0.9, 0.8])
    with pytest.raises(
        SentenceMemoryValidationError, match="width differs from sentence_memory.top_m"
    ):
        _provider(
            bank,
            _query_dataset(tmp_path, bank["rows"][0], split="test"),
        )
    first = _provider(
        bank,
        _query_dataset(tmp_path, bank["rows"][0], split="test"),
        top_m=2,
    ).identity["neighbor_tables"]
    assert set(first) == {"val"}
    assert first["val"]["query_count"] == 1
    assert first["val"]["top_m"] == 2
    assert "path" not in json.dumps(first)

    write([0.91, 0.8])
    second = _provider(
        bank,
        _query_dataset(tmp_path, bank["rows"][0], split="test"),
        top_m=2,
    ).identity["neighbor_tables"]
    assert first["val"]["sha256"] != second["val"]["sha256"]


def test_text_encoder_identity_detects_changed_tokenizer(tmp_path):
    bank = _make_bank(tmp_path)
    (bank["text_dir"] / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 99}), encoding="utf-8"
    )

    with pytest.raises(SentenceMemoryValidationError, match="artifacts differ"):
        validate_sentence_memory_bank(
            bank["bank_dir"], text_encoder_identity=bank["text_identity"]
        )


def test_ready_marker_and_configured_train_manifest_are_mandatory(tmp_path):
    bank = _make_bank(tmp_path)
    (bank["bank_dir"] / "READY").unlink()
    with pytest.raises(SentenceMemoryValidationError, match="READY is missing"):
        validate_sentence_memory_bank(bank["bank_dir"])
    write_ready_marker(bank["bank_dir"])

    wrong_manifest = tmp_path / "different_train.jsonl"
    _write_jsonl(wrong_manifest, [{"name": "different"}])
    cfg = {
        "data": {
            "train_split": "train",
            "train_manifest_path": str(wrong_manifest),
        },
        "sentence_memory": {"bank_dir": str(bank["bank_dir"])},
    }
    with pytest.raises(SentenceMemoryValidationError, match="Training manifest differs"):
        SentenceMemoryProvider(cfg, text_encoder_identity=bank["text_identity"])


def test_runtime_rejects_changed_preprocessing_stats_and_nonliteral_train_split(
    tmp_path,
):
    bank = _make_bank(tmp_path)
    base_cfg = {
        "data": {
            "data_dir": str(bank["data_dir"]),
            "train_split": "train",
            "train_manifest_path": str(bank["train_manifest"]),
            "min_frames": 40,
            "max_frames": 400,
            "length_multiple": 4,
        },
        "sentence_memory": {"bank_dir": str(bank["bank_dir"])},
    }

    SentenceMemoryProvider(base_cfg, text_encoder_identity=bank["text_identity"])

    original_mean = bank["mean_path"].read_bytes()
    np.save(bank["mean_path"], np.asarray([9.0, 9.0], dtype=np.float32))
    with pytest.raises(SentenceMemoryValidationError, match="mean statistics differ"):
        SentenceMemoryProvider(base_cfg, text_encoder_identity=bank["text_identity"])
    bank["mean_path"].write_bytes(original_mean)

    changed_preprocessing = {
        **base_cfg,
        "data": {**base_cfg["data"], "max_frames": 404},
    }
    with pytest.raises(SentenceMemoryValidationError, match="preprocessing differs"):
        SentenceMemoryProvider(
            changed_preprocessing, text_encoder_identity=bank["text_identity"]
        )

    wrong_split = {
        **base_cfg,
        "data": {
            **base_cfg["data"],
            "train_split": "val",
            "val_manifest_path": str(bank["train_manifest"]),
        },
    }
    with pytest.raises(SentenceMemoryValidationError, match="literally 'train'"):
        SentenceMemoryProvider(wrong_split, text_encoder_identity=bank["text_identity"])


def test_same_group_filter_uses_persisted_canonical_source_group_id():
    policy = exclusion_policy(
        {
            "exclude_self": False,
            "exclude_same_source": False,
            "exclude_same_group": True,
            "exclude_exact_text_train": False,
            "exclude_exact_text_eval": False,
            "group_fields": ["recording_id"],
        }
    )
    query = {
        "name": "query",
        "text": "first prompt",
        "source_group_id": "recording_id:shared-recording",
    }
    # Bank metadata intentionally has no raw recording_id; only the canonical
    # value survives the lightweight metadata projection.
    candidate = {
        "name": "candidate",
        "text": "different prompt",
        "source_group_id": "recording_id:shared-recording",
    }

    assert not candidate_is_allowed(
        query, candidate, training=True, policy=policy
    )


def test_training_and_eval_apply_semantic_and_source_filters(tmp_path):
    bank = _make_bank(tmp_path)
    query = bank["rows"][0]
    dataset = _query_dataset(tmp_path, query)
    provider = _provider(bank, dataset)

    training = provider.retrieve(
        _batch(query),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([4.0]),
        torch.tensor([True]),
        True,
        "cpu",
        "on",
    )
    evaluation = provider.retrieve(
        _batch(query),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([4.0]),
        torch.tensor([True]),
        False,
        "cpu",
        "on",
    )

    # Training excludes semantic group 0 and same-source group 1. Evaluation
    # retains the exact semantic match but selects its eligible other source.
    assert training.provenance["candidate_group_ids"][0] == [2, 3]
    assert training.ids.tolist() == [[3, 4]]
    assert evaluation.provenance["candidate_group_ids"][0][0] == 0
    assert evaluation.ids[0, 0].item() == 1

    assert evaluation.scores[0, 0].item() == pytest.approx(1.0)
    assert evaluation.duration_log_gap[0, 0].item() == pytest.approx(0.0)
    assert evaluation.tokens.dtype == torch.float16
    assert evaluation.tokens.shape == (1, 2, 4, 2)
    assert evaluation.part_validity[0, 0, :3, 0].eq(1).all()
    assert evaluation.part_validity[0, 0, :3, 3].eq(1).all()
    assert torch.allclose(
        evaluation.part_validity[0, 0, :3, 1:3],
        torch.from_numpy(bank["hand_valid"][2:5]).float() / 255.0,
        atol=5e-3,
    )
    assert set(evaluation.as_model_kwargs()) == {
        "sentence_motion_tokens",
        "sentence_motion_mask",
        "sentence_motion_tau",
        "sentence_candidate_mask",
        "sentence_text_keys",
        "sentence_scores",
        "sentence_durations",
        "sentence_part_validity",
            "sentence_memory_available",
            "sentence_candidate_ids",
        }


def test_off_mode_avoids_candidates_and_returns_shape_safe_batch(
    tmp_path, monkeypatch
):
    bank = _make_bank(tmp_path)
    query = bank["rows"][0]
    provider = _provider(bank, _query_dataset(tmp_path, query))

    def forbidden_lookup(*_args, **_kwargs):
        raise AssertionError("off mode must not search or read memory candidates")

    monkeypatch.setattr(provider, "lookup_candidate_ids", forbidden_lookup)

    output = provider.retrieve(
        _batch(query),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([4.0], dtype=np.float32),
        np.asarray([True]),
        False,
        "cpu",
        "off",
    )

    assert output.tokens.shape == (1, 2, 1, 2)
    assert not output.candidate_mask.any()
    assert not output.available.any()
    assert output.ids.tolist() == [[-1, -1]]


def test_shuffled_mode_replaces_candidate_groups_not_their_order(tmp_path):
    bank = _make_bank(tmp_path)
    query = bank["rows"][0]
    provider = _provider(bank, _query_dataset(tmp_path, query))
    arguments = (
        _batch(query),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([4.0]),
        torch.tensor([True]),
        True,
        "cpu",
    )
    normal = provider.retrieve(*arguments, "on")
    shuffled = provider.retrieve(*arguments, "shuffled")

    normal_groups = {
        group for group in normal.provenance["candidate_group_ids"][0] if group >= 0
    }
    shuffled_groups = {
        group for group in shuffled.provenance["candidate_group_ids"][0] if group >= 0
    }
    assert shuffled_groups
    assert shuffled_groups.isdisjoint(normal_groups)
    assert shuffled_groups.isdisjoint({0, 1})
    assert shuffled.provenance["candidate_source_query_rows"] == [-1]

    query_row = provider._query_rows(_batch(query))[0]
    companion = {
        "name": "external",
        "motion_path": "motion/external.npz",
        "text": "unseen query",
        "source_name": "source-external",
        "sentence_id": "source-external",
        "query_index": 1,
        "semantic_group_id": sentence_text_hash("unseen query"),
        "text_hash": sentence_text_hash("unseen query"),
        "source_id": "source-external",
        "source_group_id": "sentence_id:source-external",
    }
    solo_ids, _, _, _ = provider._shuffle_candidate_groups(
        np.asarray([[2, 3]]),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [query_row],
        training=True,
    )
    paired_ids, _, _, _ = provider._shuffle_candidate_groups(
        np.asarray([[0, 1], [2, 3]]),
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        [companion, query_row],
        training=True,
    )
    assert paired_ids[1].tolist() == solo_ids[0].tolist()

    moved_query_row = {**query_row, "query_index": 999_999}
    moved_ids, _, _, _ = provider._shuffle_candidate_groups(
        np.asarray([[2, 3]]),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [moved_query_row],
        training=True,
    )
    assert moved_ids[0].tolist() == solo_ids[0].tolist()


def test_candidate_dropout_is_deterministic_and_can_select_exact_null(tmp_path):
    bank = _make_bank(tmp_path)
    query = bank["rows"][0]
    provider = _provider(
        bank,
        _query_dataset(tmp_path, query),
        k=3,
        top_m=5,
        candidate_dropout_probability=1.0,
    )
    args = (
        _batch(query),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([4.0]),
        torch.tensor([True]),
        True,
        "cpu",
        "on",
    )

    first = provider.retrieve(*args)
    second = provider.retrieve(*args)

    assert first.ids.tolist() == second.ids.tolist() == [[-1, -1, -1]]
    assert first.candidate_mask.tolist() == [[False, False, False]]
    assert first.provenance["candidate_dropout_ranks"] == [[0, 1, 2]]


def test_group_realizations_sample_by_epoch_in_train_and_rank_stably_in_eval(tmp_path):
    bank = _make_bank(tmp_path)
    external = {
        "name": "external",
        "motion_path": "motion/external.npz",
        "text": "unseen query",
        "source_name": "source-external",
        "sentence_id": "source-external",
        "query_index": 0,
    }
    provider = _provider(bank, _query_dataset(tmp_path, external))

    selections = []
    for epoch in range(8):
        provider.set_epoch(epoch)
        first = provider._select_group_item(external, 4, 6.0, training=True)
        second = provider._select_group_item(external, 4, 6.0, training=True)
        assert first == second
        selections.append(first)
    assert set(selections) == {5, 6}

    # Equal-duration evaluation candidates prefer the realization with better
    # cached hand validity, then would fall back to stable item ID on a tie.
    assert provider._select_group_item(external, 4, 6.0, training=False) == 6


def test_duration_adjusts_selection_but_returned_score_remains_raw_cosine(tmp_path):
    bank = _make_bank(tmp_path)
    external = {
        "name": "external",
        "motion_path": "motion/external.npz",
        "text": "unseen query",
        "source_name": "source-external",
        "sentence_id": "source-external",
    }
    provider = _provider(
        bank,
        _query_dataset(tmp_path, external),
        k=1,
        top_m=5,
        duration_weight=1.0,
    )
    output = provider.retrieve(
        _batch(external),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([8.0]),
        torch.tensor([True]),
        False,
        "cpu",
    )

    assert output.provenance["candidate_group_ids"] == [[3]]
    assert output.durations.item() == pytest.approx(8.0)
    assert output.scores.item() == pytest.approx(0.8)
    assert output.duration_log_gap.item() == pytest.approx(0.0)


def test_precomputed_neighbors_are_bound_to_query_manifest(tmp_path):
    bank = _make_bank(tmp_path)
    query = bank["rows"][0]
    dataset = _query_dataset(tmp_path, query)
    manifest = load_bank_manifest(bank["bank_dir"])
    write_neighbor_table(
        bank["bank_dir"] / "neighbors_test.npz",
        bank_id=manifest["bank_id"],
        query_split="test",
        query_manifest_sha256=sha256_file(dataset.base.manifest_path),
        query_names=[query["name"]],
        ids=np.asarray([[4, 3, 2, 0]], dtype=np.int32),
        scores=np.asarray([[0.99, 0.9, 0.8, 0.7]], dtype=np.float32),
        policy=manifest["filter_policy"],
    )
    provider = _provider(bank, dataset)
    from_table = provider.retrieve(
        _batch(query),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([4.0]),
        torch.tensor([True]),
        False,
        "cpu",
        "on",
    )
    assert from_table.provenance["used_precomputed_neighbors"] == [True]
    assert from_table.provenance["candidate_group_ids"][0] == [4, 3]
    # Scores stay raw values from the table; duration is never subtracted twice.
    assert from_table.scores.tolist()[0] == pytest.approx([0.99, 0.9])

    changed_query = {**query, "text": "manifest changed"}
    changed_dataset = _query_dataset(tmp_path, changed_query, split="test-changed")
    provider.memory_cfg["neighbor_files"] = {
        "test-changed": str(bank["bank_dir"] / "neighbors_test.npz")
    }
    provider.set_dataset(changed_dataset)
    assert provider.neighbor_table is None


@pytest.mark.parametrize(
    ("ids", "scores", "message"),
    (
        ([[4, 4, -1]], [[0.9, 0.8, 0.0]], "duplicate semantic groups"),
        ([[4, -1, 3]], [[0.9, 0.0, 0.8]], "valid ID after padding"),
        ([[4, 3, -1]], [[0.8, 0.9, 0.0]], "not sorted by cosine score"),
    ),
)
def test_neighbor_writer_rejects_noncanonical_group_rows(
    tmp_path, ids, scores, message
):
    bank = _make_bank(tmp_path)
    manifest = load_bank_manifest(bank["bank_dir"])
    with pytest.raises(ValueError, match=message):
        write_neighbor_table(
            bank["bank_dir"] / "neighbors_test.npz",
            bank_id=manifest["bank_id"],
            query_split="test",
            query_manifest_sha256="query-hash",
            query_names=["query"],
            ids=np.asarray(ids, dtype=np.int32),
            scores=np.asarray(scores, dtype=np.float32),
            policy=manifest["filter_policy"],
        )


def test_required_neighbor_table_must_cover_full_manifest_in_order(tmp_path):
    bank = _make_bank(tmp_path)
    rows = [bank["rows"][0], bank["rows"][3]]
    query_manifest = tmp_path / "manifest_test.jsonl"
    _write_jsonl(query_manifest, rows)
    dataset = SimpleNamespace(
        split="test",
        base=SimpleNamespace(items=[dict(row) for row in rows], manifest_path=query_manifest),
    )
    manifest = load_bank_manifest(bank["bank_dir"])
    write_neighbor_table(
        bank["bank_dir"] / "neighbors_test.npz",
        bank_id=manifest["bank_id"],
        query_split="test",
        query_manifest_sha256=sha256_file(query_manifest),
        query_names=[rows[0]["name"]],
        ids=np.asarray([[4, 3, 2, 1]], dtype=np.int32),
        scores=np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        policy=manifest["filter_policy"],
        scoring={"metric": "cosine", "query_limit": 1},
    )

    provider = _provider(bank, dataset)
    assert provider.neighbor_table is None
    with pytest.raises(
        SentenceMemoryValidationError, match="complete query manifest"
    ):
        provider.set_dataset(dataset, require_neighbors=True)


def _gloo_sentence_memory_worker(rank, world_size, init_file, bank):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        query = bank["rows"][0]
        dataset = SimpleNamespace(
            split="test",
            base=SimpleNamespace(
                items=[dict(query)], manifest_path=bank["train_manifest"]
            ),
        )
        provider = _provider(
            bank,
            dataset,
            sampling="weighted",
            candidate_dropout_probability=0.0,
        )
        provider.set_epoch(3)
        retrieval_args = (
            _batch(query),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([4.0]),
            torch.tensor([True]),
            True,
            "cpu",
        )
        memory = provider.retrieve(*retrieval_args, "on")
        shuffled = provider.retrieve(*retrieval_args, "shuffled")
        candidate_ids = torch.cat([memory.ids, shuffled.ids], dim=1).long()
        gathered_ids = [torch.empty_like(candidate_ids) for _ in range(world_size)]
        dist.all_gather(gathered_ids, candidate_ids)
        assert all(
            torch.equal(candidate_ids, other_ids) for other_ids in gathered_ids
        )

        cfg = {
            "model": {
                "type": "sentence_memory_continuous_trajectory_field",
                "context_hidden_dim": 8,
                "context_layers": 1,
                "field_hidden_dim": 4,
                "field_depth": 1,
                "max_local_fields": 1,
                "frames_per_local_field": 8,
                "dropout": 0.0,
            },
            "conditioning": {
                "temporal_slot_count": 2,
                "temporal_slot_layers": 1,
                "temporal_slot_heads": 2,
                "context_fps": 20.0,
            },
            "sentence_memory": {
                "motion_dim": 2,
                "key_dim": 3,
                "attention_layers": 1,
                "attention_heads": 2,
                "score_temperature": 0.1,
                "duration_weight": 0.1,
            },
            "train": {
                "freeze_base": True,
                "unfreeze_base_prefixes": [],
            },
        }
        torch.manual_seed(101)
        model = build_continuous_trajectory_field(cfg, text_dim=3)
        trainable = configure_sentence_memory_trainable_parameters(model, cfg)
        assert trainable["trainable"]
        assert all(
            is_sentence_memory_parameter(name) for name in trainable["trainable"]
        )
        frozen_name, frozen_parameter = next(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        )
        frozen_before = frozen_parameter.detach().clone()
        fusion_before = (
            model.hypernetwork.sentence_memory_fusion.weight.detach().clone()
        )

        distributed_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.05,
        )
        optimizer.zero_grad(set_to_none=True)
        output = distributed_model(
            text_tokens=torch.tensor(
                [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]], dtype=torch.float32
            ),
            text_mask=torch.ones(1, 2, dtype=torch.bool),
            query_times=torch.linspace(-1.0, 1.0, 3).view(1, -1),
            **memory.as_model_kwargs(),
        )
        loss = (output["prediction"] - 0.25).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            if not is_sentence_memory_parameter(name):
                assert parameter.grad is None, name
        fusion_gradient = model.hypernetwork.sentence_memory_fusion.weight.grad
        assert fusion_gradient is not None
        assert torch.isfinite(fusion_gradient).all()
        assert fusion_gradient.abs().sum() > 0
        optimizer.step()

        assert torch.equal(frozen_parameter, frozen_before), frozen_name
        fusion_after = model.hypernetwork.sentence_memory_fusion.weight.detach()
        assert not torch.equal(fusion_after, fusion_before)
        gathered_fusion = [torch.empty_like(fusion_after) for _ in range(world_size)]
        dist.all_gather(gathered_fusion, fusion_after)
        assert all(
            torch.allclose(fusion_after, other, atol=0.0, rtol=0.0)
            for other in gathered_fusion
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_sentence_memory_phase_a_optimizer_smoke(tmp_path):
    bank = _make_bank(tmp_path)
    init_file = (tmp_path / "sentence_memory_gloo_init").resolve()
    mp.spawn(
        _gloo_sentence_memory_worker,
        args=(2, str(init_file), bank),
        nprocs=2,
        join=True,
    )
