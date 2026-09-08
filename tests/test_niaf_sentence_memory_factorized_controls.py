from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from NIAF.continuous_trajectory_field.scripts import (
    evaluate_continuous_trajectory_field as public_evaluator,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    CLUSTER_EQUAL_SELECTION_AGGREGATION,
    cluster_mean_paired_usage_moments,
    configured_sentence_memory_eval_modes,
    distributed_validation_metrics,
    evaluate,
    isolated_development_validation_config,
    load_isolated_development_validation_runtime,
    prepare_field_batch,
    save_checkpoint,
    sentence_memory_architecture_identity,
    sentence_memory_evaluation_control_identity,
    sentence_memory_objective_identity,
    sentence_memory_resume_identity,
    sentence_memory_selection_aggregation_identity,
    sentence_memory_validation_corruption_map_identity,
    validate_sentence_memory_architecture_identity,
    validate_sentence_memory_evaluation_control_identity,
    validate_sentence_memory_selection_aggregation_identity,
    validate_sentence_memory_validation_corruption_map_identity,
    write_json_atomic,
)
from NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory import (
    bind_factorized_export_neighbor_subset,
    prepare_export_manifest,
)
from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryBatch,
    SentenceMemoryProvider,
    canonical_json,
    motion_only_shuffle_sentence_memory_batch,
    sha256_file,
    write_neighbor_table,
)


def _memory(batch_size: int = 2, candidate_count: int = 4) -> SentenceMemoryBatch:
    token_count = 2
    tokens = torch.arange(
        batch_size * candidate_count * token_count * 3, dtype=torch.float32
    ).reshape(batch_size, candidate_count, token_count, 3)
    return SentenceMemoryBatch(
        tokens=tokens,
        token_mask=torch.ones(
            batch_size, candidate_count, token_count, dtype=torch.bool
        ),
        token_tau=torch.linspace(0.0, 1.0, token_count).expand(
            batch_size, candidate_count, -1
        ).clone(),
        candidate_mask=torch.ones(batch_size, candidate_count, dtype=torch.bool),
        candidate_keys=torch.randn(batch_size, candidate_count, 5),
        scores=torch.randn(batch_size, candidate_count),
        durations=torch.ones(batch_size, candidate_count),
        duration_log_gap=torch.zeros(batch_size, candidate_count),
        part_validity=torch.ones(
            batch_size, candidate_count, token_count, 4
        ),
        ids=torch.arange(batch_size * candidate_count).reshape(
            batch_size, candidate_count
        ),
        available=torch.ones(batch_size, dtype=torch.bool),
        provenance={"mode": "on"},
    )


def _reorder_memory(memory: SentenceMemoryBatch, order: torch.Tensor):
    return replace(
        memory,
        **{
            field: getattr(memory, field)[order]
            for field in (
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
            )
        },
    )


def test_fixed_motion_corruption_ignores_epoch_batch_order_and_rank():
    memory = _memory(batch_size=3)
    query_ids = ["query-c", "query-a", "query-b"]
    first, first_permutation, _ = motion_only_shuffle_sentence_memory_batch(
        memory,
        query_ids=query_ids,
        epoch=1,
        seed=1234,
        corruption_nonce="validation-v1",
        corruption_condition="motion_shuffled",
    )
    later, later_permutation, _ = motion_only_shuffle_sentence_memory_batch(
        memory,
        query_ids=query_ids,
        epoch=91,
        seed=1234,
        corruption_nonce="validation-v1",
        corruption_condition="motion_shuffled",
    )
    assert torch.equal(first_permutation, later_permutation)
    assert torch.equal(first.tokens, later.tokens)

    order = torch.tensor([2, 0, 1])
    reordered, reordered_permutation, _ = motion_only_shuffle_sentence_memory_batch(
        _reorder_memory(memory, order),
        query_ids=[query_ids[index] for index in order.tolist()],
        epoch=7,
        seed=1234,
        corruption_nonce="validation-v1",
        corruption_condition="motion_shuffled",
    )
    assert torch.equal(reordered_permutation, first_permutation[order])
    assert torch.equal(reordered.tokens, first.tokens[order])


def test_training_motion_corruption_preserves_original_epoch_hash_formula():
    memory = _memory(batch_size=1, candidate_count=5)
    query_id = "stable-training-query"
    epoch = 3
    _output, permutation, _ = motion_only_shuffle_sentence_memory_batch(
        memory, query_ids=[query_id], epoch=epoch, seed=1234
    )
    identity = canonical_json(
        {"epoch": epoch, "query_id": query_id, "seed": 1234}
    )
    draw = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    shift = 1 + draw % 4
    expected = torch.roll(torch.arange(5), shifts=shift)
    assert torch.equal(permutation[0], expected)


def test_fixed_full_shuffle_alternatives_ignore_provider_epoch_and_query_order():
    provider = SentenceMemoryProvider.__new__(SentenceMemoryProvider)
    provider.seed = 1234
    provider.epoch = 1
    provider.group_count = 31
    provider._group_allowed = lambda _query, _group, _training: True
    queries = [
        {"name": "a", "motion_path": "a.npy", "semantic_group_id": "ga"},
        {"name": "b", "motion_path": "b.npy", "semantic_group_id": "gb"},
    ]

    def maps(rows):
        return {
            row["name"]: provider._random_alternatives(
                row,
                {0, 1},
                8,
                training=False,
                salt="shuffle",
                corruption_nonce="validation-v1",
                corruption_seed=1234,
                corruption_condition="shuffled",
            )
            for row in rows
        }

    expected = maps(queries)
    provider.epoch = 999
    assert maps(list(reversed(queries))) == expected


def _factorized_cfg():
    return {
        "seed": 1234,
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {
            "enabled": True,
            "key_value_mode": "factorized_metadata_motion_v1",
            "temporal_prior_mode": "none",
            "temporal_prior_sigma": 0.25,
            "temporal_prior_scale": 1.0,
            "duration_weight": 0.05,
            "score_temperature": 0.10,
            "retrieval_prior_scale": 1.0,
        },
        "eval": {
            "sentence_memory_modes": [
                "off",
                "on",
                "motion_shuffled",
                "shuffled",
                "analytic_prior",
            ],
            "evaluation_corruption": {
                "mode": "fixed_query_condition_v1",
                "seed": 1234,
                "nonce": "csl_daily_validation_corruption_v1",
            },
        },
        "selection": {"aggregation": CLUSTER_EQUAL_SELECTION_AGGREGATION},
    }


def _isolated_validation_cfg(*, development_sha256, artifact_identity):
    cfg = _factorized_cfg()
    cfg.update(
        {
            "data": {
                "val_split": "val",
                "val_manifest_path": "/forbidden/canonical/manifest_val.jsonl",
                "limit_val": 0,
            },
            "sentence_memory_safety": {
                "paired_corruption": {"enabled": True}
            },
            "validation_text_partition": {
                "enabled": True,
                "split": "val",
                "development_text_count": 2,
                "expected_novel_text_count": 3,
                "expected_validation_rows": 5,
                "expected_development_rows": 3,
                "expected_confirmation_rows": 2,
                "expected_partition_digest": "1" * 64,
                "expected_partition_artifact_identity": artifact_identity,
                "expected_validation_manifest_sha256": "2" * 64,
                "expected_development_manifest_sha256": development_sha256,
                "expected_bank_id": "3" * 64,
            },
        }
    )
    return cfg


def test_factorized_training_reads_only_sealed_development_manifest(tmp_path):
    partition_dir = tmp_path / "sealed_partition"
    partition_dir.mkdir()
    artifact_identity = "a" * 64
    (partition_dir / "READY").write_text(
        json.dumps(
            {
                "schema_name": "signtrajfield_validation_text_cluster_partition",
                "schema_version": 1,
                "artifact_identity": artifact_identity,
            }
        ),
        encoding="utf-8",
    )
    development_rows = [
        {"name": "dev-a-1", "text": "Alpha", "source_split": "val"},
        {"name": "dev-a-2", "text": " alpha ", "source_split": "val"},
        {"name": "dev-b-1", "text": "Beta", "source_split": "val"},
    ]
    development_path = partition_dir / "manifest_development.jsonl"
    development_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in development_rows
        ),
        encoding="utf-8",
    )
    forbidden_names = {
        "partition.json",
        "manifest_confirmation.jsonl",
        "manifest_exact_seen.jsonl",
        "manifest_val.jsonl",
    }
    for name in forbidden_names - {"manifest_val.jsonl"}:
        (partition_dir / name).write_text("holdout sentinel\n", encoding="utf-8")
    cfg = _isolated_validation_cfg(
        development_sha256=sha256_file(development_path),
        artifact_identity=artifact_identity,
    )

    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        if Path(path).name in forbidden_names:
            raise AssertionError(f"training opened holdout artifact {path}")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", guarded_open):
        runtime = load_isolated_development_validation_runtime(
            cfg, partition_dir=partition_dir
        )
        loader_cfg = isolated_development_validation_config(cfg, runtime)

    assert runtime.development_row_count == 3
    assert runtime.development_text_count == 2
    assert runtime.confirmation_text_count == 1
    assert loader_cfg["data"]["val_manifest_path"] == str(development_path)
    assert cfg["data"]["val_manifest_path"].endswith("manifest_val.jsonl")
    assert set(opened) == {"READY", "manifest_development.jsonl"}
    serialized_cfg = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    assert "resolved_artifact" not in serialized_cfg
    assert "Alpha" not in serialized_cfg
    assert "Beta" not in serialized_cfg
    assert "manifest_confirmation.jsonl" not in serialized_cfg


def test_factorized_public_evaluator_uses_only_sealed_development_manifest(
    tmp_path, monkeypatch,
):
    partition_dir = tmp_path / "sealed_partition"
    partition_dir.mkdir()
    artifact_identity = "a" * 64
    (partition_dir / "READY").write_text(
        json.dumps(
            {
                "schema_name": "signtrajfield_validation_text_cluster_partition",
                "schema_version": 1,
                "artifact_identity": artifact_identity,
            }
        ),
        encoding="utf-8",
    )
    development_rows = [
        {"name": "dev-a-1", "text": "Alpha", "source_split": "val"},
        {"name": "dev-a-2", "text": " alpha ", "source_split": "val"},
        {"name": "dev-b-1", "text": "Beta", "source_split": "val"},
    ]
    development_path = partition_dir / "manifest_development.jsonl"
    development_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in development_rows
        ),
        encoding="utf-8",
    )
    forbidden_names = {
        "partition.json",
        "manifest_confirmation.jsonl",
        "manifest_exact_seen.jsonl",
        "manifest_val.jsonl",
        "manifest_test.jsonl",
    }
    for name in forbidden_names - {"manifest_val.jsonl", "manifest_test.jsonl"}:
        (partition_dir / name).write_text("holdout sentinel\n", encoding="utf-8")
    cfg = _isolated_validation_cfg(
        development_sha256=sha256_file(development_path),
        artifact_identity=artifact_identity,
    )
    cfg["eval"]["num_workers"] = 0
    monkeypatch.setenv(
        "SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR", str(partition_dir)
    )

    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        if Path(path).name in forbidden_names:
            raise AssertionError(f"public evaluator opened holdout artifact {path}")
        return original_open(path, *args, **kwargs)

    def fake_make_loader(loader_cfg, split, **kwargs):
        del kwargs
        manifest_path = Path(loader_cfg["data"][f"{split}_manifest_path"])
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset = SimpleNamespace(
            split=split,
            base=SimpleNamespace(manifest_path=manifest_path, items=rows),
            estimated_lengths=tuple(1 for _ in rows),
        )
        return dataset, SimpleNamespace(batch_size=1, collate_fn=None), None

    dist_info = {"enabled": False, "world_size": 1, "rank": 0}
    with (
        patch.object(Path, "open", guarded_open),
        patch.object(public_evaluator, "make_loader", fake_make_loader),
    ):
        dataset, loader, _sampler, runtime = (
            public_evaluator.build_public_evaluation_loader(
                cfg, split="val", limit=0, dist_info=dist_info
            )
        )

    assert len(dataset.base.items) == 3
    assert loader.batch_size == 1
    assert runtime.manifest_path == development_path.resolve()
    assert not forbidden_names.intersection(opened)
    assert set(opened) == {"READY", "manifest_development.jsonl"}

    bound_loader = object()
    bound_sampler = object()
    binding = {"neighbor_lookup_mode": "exact_name_indexed_parent_subset_v1"}
    with (
        patch.object(
            public_evaluator,
            "bind_sentence_memory_validation_dataset",
            return_value=binding,
        ) as bind_mock,
        patch.object(
            public_evaluator,
            "build_isolated_development_validation_loader",
            return_value=(bound_loader, bound_sampler, runtime),
        ) as loader_mock,
    ):
        actual_loader, actual_sampler, actual_binding = (
            public_evaluator.bind_public_evaluation_development_loader(
                cfg, dataset, loader, object(), dist_info, runtime
            )
        )
    assert (actual_loader, actual_sampler, actual_binding) == (
        bound_loader,
        bound_sampler,
        binding,
    )
    bind_mock.assert_called_once()
    loader_mock.assert_called_once()

    opened.clear()
    with (
        patch.object(Path, "open", guarded_open),
        patch.object(
            public_evaluator,
            "make_loader",
            side_effect=AssertionError("test loader must not be constructed"),
        ),
        pytest.raises(ValueError, match="restricted to the sealed development"),
    ):
        public_evaluator.build_public_evaluation_loader(
            cfg, split="test", limit=0, dist_info=dist_info
        )
    assert opened == []


def test_factorized_export_never_reads_canonical_or_holdout_before_dev_binding(
    tmp_path, monkeypatch,
):
    partition_dir = tmp_path / "sealed_partition"
    partition_dir.mkdir()
    artifact_identity = "a" * 64
    (partition_dir / "READY").write_text(
        json.dumps(
            {
                "schema_name": "signtrajfield_validation_text_cluster_partition",
                "schema_version": 1,
                "artifact_identity": artifact_identity,
            }
        ),
        encoding="utf-8",
    )
    development_path = partition_dir / "manifest_development.jsonl"
    development_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {"name": "dev-a-1", "text": "Alpha"},
                {"name": "dev-a-2", "text": " alpha "},
                {"name": "dev-b-1", "text": "Beta"},
            )
        ),
        encoding="utf-8",
    )
    confirmation_path = partition_dir / "manifest_confirmation.jsonl"
    confirmation_path.write_text("holdout sentinel\n", encoding="utf-8")
    partition_payload_path = partition_dir / "partition.json"
    partition_payload_path.write_text("holdout sentinel\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    (data_dir / "meta").mkdir(parents=True)
    canonical_path = data_dir / "meta" / "manifest_val.jsonl"
    canonical_path.write_text("full validation sentinel\n", encoding="utf-8")
    cfg = _isolated_validation_cfg(
        development_sha256=sha256_file(development_path),
        artifact_identity=artifact_identity,
    )
    cfg["data"]["data_dir"] = str(data_dir)
    cfg["data"].pop("val_manifest_path")
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    forbidden = {
        canonical_path.resolve(),
        confirmation_path.resolve(),
        partition_payload_path.resolve(),
    }
    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        resolved = Path(path).resolve()
        opened.append(resolved)
        if resolved in forbidden:
            raise AssertionError(f"factorized dev export opened {resolved}")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", guarded_open):
        dataset_manifest, selected_rows, summary = prepare_export_manifest(
            cfg,
            split="val",
            out_dir=out_dir,
            num_samples=0,
            seed=1234,
            manifest=development_path,
            selection_mode="first",
        )

    assert dataset_manifest == development_path.resolve()
    assert len(selected_rows) == 3
    assert Path(summary["output_manifest"]).resolve() == development_path.resolve()
    selected_copy = Path(summary["selected_manifest_copy"]).resolve()
    assert selected_copy.parent == out_dir.resolve()
    assert summary["selected_manifest_copy_sha256"] == sha256_file(selected_copy)
    assert json.loads((out_dir / "sample_manifest_summary.json").read_text()) == summary
    assert summary["canonical_manifest_inspection"] == (
        "forbidden_isolated_explicit_manifest_v1"
    )
    assert not forbidden.intersection(opened)

    monkeypatch.setenv(
        "SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256",
        sha256_file(development_path),
    )
    monkeypatch.setenv("SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS", "3")
    monkeypatch.setenv(
        "SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY",
        artifact_identity,
    )
    v2_cfg = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "data": {"data_dir": str(data_dir)},
    }
    v2_out_dir = tmp_path / "v2_export"
    v2_out_dir.mkdir()
    opened.clear()
    with patch.object(Path, "open", guarded_open):
        v2_manifest, v2_rows, v2_summary = prepare_export_manifest(
            v2_cfg,
            split="val",
            out_dir=v2_out_dir,
            num_samples=0,
            seed=1234,
            manifest=development_path,
            selection_mode="first",
        )
    assert v2_manifest == development_path.resolve()
    assert len(v2_rows) == 3
    assert Path(v2_summary["output_manifest"]).resolve() == development_path.resolve()
    v2_selected_copy = Path(v2_summary["selected_manifest_copy"]).resolve()
    assert v2_summary["selected_manifest_copy_sha256"] == sha256_file(
        v2_selected_copy
    )
    assert json.loads(
        (v2_out_dir / "sample_manifest_summary.json").read_text()
    ) == v2_summary
    assert v2_summary["query_manifest_authority"] == (
        "explicit_isolated_development_environment_v1"
    )
    assert not forbidden.intersection(opened)

    for name in (
        "SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256",
        "SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS",
        "SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY",
    ):
        monkeypatch.delenv(name)
    post_spend_manifest = tmp_path / "post_spend_queries.jsonl"
    post_spend_manifest.write_text(
        '{  "text" : "Gamma", "name" : "confirmation-a" }\n',
        encoding="utf-8",
    )
    canonical_path.write_text(
        '{"name":"confirmation-a","text":"Gamma"}\n', encoding="utf-8"
    )
    legacy_out_dir = tmp_path / "legacy_post_spend_export"
    legacy_out_dir.mkdir()
    legacy_manifest, legacy_rows, legacy_summary = prepare_export_manifest(
        v2_cfg,
        split="val",
        out_dir=legacy_out_dir,
        num_samples=0,
        seed=1234,
        manifest=post_spend_manifest,
        selection_mode="first",
    )
    assert legacy_manifest == post_spend_manifest.resolve()
    assert len(legacy_rows) == 1
    assert Path(legacy_summary["output_manifest"]).resolve() == (
        post_spend_manifest.resolve()
    )
    assert legacy_summary["dataset_manifest_sha256"] == sha256_file(
        post_spend_manifest
    )
    assert legacy_summary["selected_manifest_copy_sha256"] != (
        legacy_summary["dataset_manifest_sha256"]
    )

    class Provider:
        def __init__(self):
            self.identity = {
                "bank_id": "3" * 64,
                "neighbor_tables": {
                    "val": {
                        "sha256": "4" * 64,
                        "query_manifest_sha256": "2" * 64,
                        "query_order_sha256": "5" * 64,
                        "query_count": 5,
                    }
                },
            }
            self.neighbor_table = None
            self.call = None

        def set_dataset_with_name_indexed_neighbor_subset(self, dataset, **kwargs):
            self.call = (dataset, kwargs)
            self.neighbor_table = {
                "lookup_mode": "exact_name_indexed_parent_subset_v1",
                "parent_query_manifest_sha256": kwargs[
                    "parent_manifest_sha256"
                ],
                "parent_query_count": 5,
            }

    provider = Provider()
    dataset = _NamedDataset(
        development_path, [row["name"] for row in selected_rows]
    )
    binding = bind_factorized_export_neighbor_subset(cfg, dataset, provider)
    assert provider.call[0] is dataset
    assert provider.call[1] == {
        "parent_manifest_sha256": "2" * 64,
        "expected_subset_manifest_sha256": sha256_file(development_path),
    }
    assert binding["authority"] == "config_pinned_development_manifest_v1"
    assert binding["online_fallback_allowed"] is False
    assert binding["lookup_mode"] == "exact_name_indexed_parent_subset_v1"


class _NamedDataset:
    split = "val"

    def __init__(self, manifest_path, names):
        self.base = SimpleNamespace(
            manifest_path=manifest_path,
            items=[{"name": name} for name in names],
        )

    def __len__(self):
        return len(self.base.items)


def test_name_indexed_neighbor_subset_preserves_canonical_rows(tmp_path):
    parent_manifest_sha256 = "b" * 64
    table_path = tmp_path / "neighbors_val.npz"
    ids = torch.tensor([[0, 1], [2, 3], [4, -1]]).numpy()
    scores = torch.tensor([[0.9, 0.8], [0.7, 0.6], [0.5, 0.0]]).numpy()
    write_neighbor_table(
        table_path,
        bank_id="bank",
        query_split="val",
        query_manifest_sha256=parent_manifest_sha256,
        query_names=["full-a", "full-b", "full-c"],
        ids=ids,
        scores=scores,
        policy={},
    )
    development_path = tmp_path / "manifest_development.jsonl"
    development_path.write_text(
        '{"name":"full-c"}\n{"name":"full-a"}\n', encoding="utf-8"
    )
    provider = SentenceMemoryProvider.__new__(SentenceMemoryProvider)
    provider.memory_cfg = {"neighbor_files": {"val": str(table_path)}}
    provider.bank_id = "bank"
    provider.group_count = 5
    provider.policy = {}
    provider.top_m = 2
    provider.k = 2
    provider.key_dim = 3
    provider.duration_weight = 0.0
    provider._neighbor_tables_identity = {
        "val": {"sha256": sha256_file(table_path)}
    }
    dataset = _NamedDataset(development_path, ["full-c", "full-a"])

    provider.set_dataset_with_name_indexed_neighbor_subset(
        dataset,
        parent_manifest_sha256=parent_manifest_sha256,
        expected_subset_manifest_sha256=sha256_file(development_path),
    )

    assert provider.neighbor_table["lookup_mode"] == (
        "exact_name_indexed_parent_subset_v1"
    )
    assert provider.neighbor_table["parent_query_rows"].tolist() == [2, 0]
    assert provider.neighbor_table["query_names"] == ["full-c", "full-a"]
    assert provider.neighbor_table["ids"].tolist() == [[4, -1], [0, 1]]
    provider._group_allowed = lambda *_args, **_kwargs: True
    selected_ids, _selected_scores = provider._table_candidates(
        {"query_index": 0, "name": "full-c"}, training=False
    )
    assert selected_ids.tolist() == [4]
    with patch.object(
        provider,
        "_online_candidates",
        side_effect=AssertionError("name-indexed subset fell back online"),
    ):
        output_ids, _output_scores, used_table = provider.lookup_candidate_ids(
            torch.tensor([[1.0, 0.0, 0.0]]),
            [{"query_index": 0, "name": "full-c"}],
            predicted_duration=None,
            training=False,
        )
    assert output_ids.tolist() == [[4, -1]]
    assert used_table == [True]


def test_factorized_control_identities_are_strict_but_legacy_is_compatible():
    cfg = _factorized_cfg()
    checkpoint = {
        "sentence_memory_architecture_identity": (
            sentence_memory_architecture_identity(cfg)
        ),
        "sentence_memory_evaluation_control_identity": (
            sentence_memory_evaluation_control_identity(cfg)
        ),
        "sentence_memory_selection_aggregation_identity": (
            sentence_memory_selection_aggregation_identity(cfg)
        ),
    }
    validate_sentence_memory_architecture_identity(checkpoint, cfg)
    validate_sentence_memory_evaluation_control_identity(checkpoint, cfg)
    validate_sentence_memory_selection_aggregation_identity(checkpoint, cfg)

    with pytest.raises(RuntimeError, match="no persisted"):
        validate_sentence_memory_architecture_identity({}, cfg)
    with pytest.raises(RuntimeError, match="no persisted"):
        validate_sentence_memory_evaluation_control_identity({}, cfg)
    with pytest.raises(RuntimeError, match="no persisted"):
        validate_sentence_memory_selection_aggregation_identity({}, cfg)

    changed = copy.deepcopy(cfg)
    changed["eval"]["evaluation_corruption"]["nonce"] = "new-version"
    with pytest.raises(RuntimeError, match="differs from the active config"):
        validate_sentence_memory_evaluation_control_identity(checkpoint, changed)
    assert sentence_memory_resume_identity(changed)["digest"] != (
        sentence_memory_resume_identity(cfg)["digest"]
    )

    changed_aggregation = copy.deepcopy(cfg)
    changed_aggregation["selection"]["aggregation"] = (
        "sample_weighted_rows_v1"
    )
    assert sentence_memory_resume_identity(changed_aggregation)["digest"] != (
        sentence_memory_resume_identity(cfg)["digest"]
    )

    legacy = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {},
    }
    validate_sentence_memory_architecture_identity({}, legacy)
    validate_sentence_memory_evaluation_control_identity({}, legacy)
    validate_sentence_memory_selection_aggregation_identity({}, legacy)


def test_factorization_and_eval_controls_do_not_change_training_objective_identity():
    base = {
        "sentence_memory_safety": {
            "paired_corruption": {
                "enabled": True,
                "full_shuffle_probability": 0.10,
                "benefit_margin_relative": 0.001,
                "ranking_margin_relative": 0.005,
                "detach_corrupt_ranking": True,
                "fallback_huber_beta": 0.10,
            }
        },
        "objective": {"lambda_sentence_sparsity": 1e-4},
    }
    changed = copy.deepcopy(base)
    changed.update(_factorized_cfg())
    assert sentence_memory_objective_identity(base) == (
        sentence_memory_objective_identity(changed)
    )


def test_checkpoint_persists_architecture_selection_and_corruption_identities(
    tmp_path,
):
    cfg = _factorized_cfg()
    cfg["validation_text_partition"] = {
        "evaluation_corruption_map_identity": (
            sentence_memory_validation_corruption_map_identity(
                cfg, partition_digest="partition", bank_id="bank"
            )
        )
    }
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 1, 2, cfg, {"finite": 1.0})
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["sentence_memory_architecture_identity"] == (
        sentence_memory_architecture_identity(cfg)
    )
    assert checkpoint["sentence_memory_evaluation_control_identity"] == (
        sentence_memory_evaluation_control_identity(cfg)
    )
    assert checkpoint["sentence_memory_selection_aggregation_identity"] == (
        sentence_memory_selection_aggregation_identity(cfg)
    )
    assert checkpoint["sentence_memory_validation_corruption_map_identity"] == (
        cfg["validation_text_partition"]["evaluation_corruption_map_identity"]
    )
    validate_sentence_memory_validation_corruption_map_identity(
        checkpoint, cfg
    )


def test_checkpoint_failed_atomic_write_preserves_previous_target(tmp_path):
    cfg = _factorized_cfg()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "last.pt"
    torch.save({"sentinel": "previous-valid-checkpoint"}, path)
    previous_bytes = path.read_bytes()

    def fail_after_partial_write(_payload, handle):
        handle.write(b"partial replacement")
        handle.flush()
        raise OSError("injected checkpoint serialization failure")

    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.torch.save",
        side_effect=fail_after_partial_write,
    ):
        with pytest.raises(OSError, match="injected checkpoint"):
            save_checkpoint(path, model, optimizer, 1, 2, cfg, {"finite": 1.0})

    assert path.read_bytes() == previous_bytes
    assert list(tmp_path.glob(".last.pt.*.tmp")) == []


def test_selection_summary_failed_atomic_publish_preserves_previous_target(tmp_path):
    path = tmp_path / "selection_summary.json"
    path.write_text('{"sentinel": "previous-valid-summary"}\n', encoding="utf-8")
    previous_bytes = path.read_bytes()

    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.os.replace",
        side_effect=OSError("injected summary publish failure"),
    ):
        with pytest.raises(OSError, match="injected summary"):
            write_json_atomic(path, {"has_feasible_checkpoint": True})

    assert path.read_bytes() == previous_bytes
    assert list(tmp_path.glob(".selection_summary.json.*.tmp")) == []


def test_analytic_prior_mode_uses_correct_payload_and_model_attention_flag():
    cfg = _factorized_cfg()
    cfg["conditioning"] = {
        "word_prior_train_mode": "off",
        "sentence_memory_train_mode": "on",
    }
    target = torch.zeros(1, 2, 256)
    text_tokens = torch.zeros(1, 3, 5)
    text_mask = torch.ones(1, 3, dtype=torch.bool)
    batch = {
        "name": ["query"],
        "motion_path": ["query.npy"],
        "text": ["query text"],
        "length": torch.tensor([2]),
        "mask": torch.ones(1, 2, dtype=torch.bool),
    }

    class Model:
        def __init__(self):
            self.forward_kwargs = None

        def predict_duration(self, tokens, text_mask=None):
            duration = torch.ones(tokens.shape[0])
            return duration.log(), duration

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            return {"prediction": target.clone()}

    class Provider:
        def __init__(self):
            self.mode = None

        def retrieve(self, *, mode, **_kwargs):
            self.mode = mode
            return _memory(batch_size=1)

    model = Model()
    provider = Provider()
    with (
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "train_continuous_trajectory_field.prepare_motion",
            return_value=target,
        ),
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "train_continuous_trajectory_field.encode_batch_text",
            return_value=(text_tokens, text_mask),
        ),
    ):
        prepare_field_batch(
            model,
            None,
            None,
            batch,
            SimpleNamespace(),
            cfg,
            torch.device("cpu"),
            word_prior_mode="off",
            sentence_memory_provider=provider,
            sentence_memory_mode="analytic_prior",
            training=False,
            epoch=99,
        )
    assert provider.mode == "on"
    assert model.forward_kwargs["sentence_memory_attention_mode"] == (
        "analytic_prior"
    )
    assert configured_sentence_memory_eval_modes(cfg)[-1] == "analytic_prior"


def test_full_shuffle_eval_records_effective_corruption_fraction():
    cfg = _factorized_cfg()
    cfg["conditioning"] = {
        "word_prior_train_mode": "off",
        "sentence_memory_train_mode": "on",
    }
    target = torch.zeros(2, 2, 256)
    text_tokens = torch.zeros(2, 3, 5)
    text_mask = torch.ones(2, 3, dtype=torch.bool)
    batch = {
        "name": ["query-a", "query-b"],
        "motion_path": ["query-a.npy", "query-b.npy"],
        "text": ["query text a", "query text b"],
        "length": torch.tensor([2, 2]),
        "mask": torch.ones(2, 2, dtype=torch.bool),
    }

    class Model:
        def predict_duration(self, tokens, text_mask=None):
            duration = torch.ones(tokens.shape[0])
            return duration.log(), duration

        def __call__(self, **_kwargs):
            return {"prediction": target.clone()}

    class Provider:
        def __init__(self):
            self.mode = None

        def retrieve(self, *, mode, **_kwargs):
            self.mode = mode
            memory = _memory(batch_size=2)
            memory.available[1] = False
            return memory

    provider = Provider()
    with (
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "train_continuous_trajectory_field.prepare_motion",
            return_value=target,
        ),
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "train_continuous_trajectory_field.encode_batch_text",
            return_value=(text_tokens, text_mask),
        ),
    ):
        prepared = prepare_field_batch(
            Model(),
            None,
            None,
            batch,
            SimpleNamespace(),
            cfg,
            torch.device("cpu"),
            word_prior_mode="off",
            sentence_memory_provider=provider,
            sentence_memory_mode="shuffled",
            training=False,
            epoch=99,
        )

    expected = torch.tensor([True, False])
    assert provider.mode == "shuffled"
    assert torch.equal(prepared["sentence_memory_shuffled_mask"], expected)
    assert torch.equal(prepared["sentence_memory_full_shuffle_mask"], expected)
    assert prepared["sentence_memory_full_shuffle_mask"].float().mean() == 0.5


class _ClusterLoader:
    evaluation_unit = "normalized_text_cluster"
    evaluation_unit_count = 2

    def __init__(self):
        self.batches = [
            {
                "name": ["a-1", "a-2"],
                "text": ["Same text", " same   TEXT "],
                "value": torch.tensor([0.0, 2.0]),
            },
            {
                "name": ["b-1"],
                "text": ["different"],
                "value": torch.tensor([10.0]),
            },
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def test_cluster_equal_evaluation_means_signers_then_equal_weights_texts():
    model = SimpleNamespace(eval=lambda: None)

    def fake_evaluate_microbatch(*args, **_kwargs):
        batch = args[4]
        assert len(batch["name"]) == 1
        return {"metric": float(batch["value"].item())}

    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.evaluate_microbatch",
        side_effect=fake_evaluate_microbatch,
    ):
        result = evaluate(
            model,
            None,
            None,
            None,
            _ClusterLoader(),
            None,
            {"eval": {"empty_cache_between_batches": False}},
            torch.device("cpu"),
            show_progress=False,
        )
    # mean(mean(0,2), mean(10)) = 5.5; a row-weighted mean would be 4.0.
    assert result["metric"] == pytest.approx(5.5)


def test_cluster_rmotion_averages_row_mse_before_equal_text_weighting():
    def row(motion_sum, motion_count, off_sum, off_count):
        output = {}
        for part in ("all", "body", "left_hand", "right_hand", "face"):
            output[f"{part}/motion_square_sum"] = motion_sum
            output[f"{part}/motion_element_count"] = motion_count
            output[f"{part}/off_square_sum"] = off_sum
            output[f"{part}/off_element_count"] = off_count
        return output

    cluster_a = cluster_mean_paired_usage_moments(
        [row(2.0, 2.0, 8.0, 2.0), row(90.0, 10.0, 160.0, 10.0)]
    )
    cluster_b = cluster_mean_paired_usage_moments(
        [row(12.0, 3.0, 12.0, 3.0)]
    )
    raw = {
        f"_paired_usage_raw/{name}": cluster_a[name] + cluster_b[name]
        for name in cluster_a
    }
    reduced = distributed_validation_metrics(
        raw,
        local_sample_count=2,
        device=torch.device("cpu"),
        dist_info={"enabled": False, "world_size": 1},
    )
    # Cluster A is mean(row MSEs): motion=5, off=10; cluster B=4,4.
    assert reduced["paired_sentence_memory/Rmotion"] == pytest.approx(
        ((4.5 / 7.0) ** 0.5)
    )


def _distributed_cluster_worker(rank, init_path, output_queue):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        _shuffled, permutation, _informative = (
            motion_only_shuffle_sentence_memory_batch(
                _memory(batch_size=1),
                query_ids=["same-query-on-every-rank"],
                epoch=rank + 1,
                seed=1234,
                corruption_nonce="validation-v1",
                corruption_condition="motion_shuffled",
            )
        )
        gathered_permutations = [
            torch.empty_like(permutation) for _ in range(2)
        ]
        dist.all_gather(gathered_permutations, permutation)
        assert torch.equal(gathered_permutations[0], gathered_permutations[1])

        local_count = 1 if rank == 0 else 2
        ordinary = 1.0 if rank == 0 else 5.0
        motion_sum = 1.0 if rank == 0 else 13.0
        off_sum = 4.0 if rank == 0 else 20.0
        values = {"metric": ordinary}
        for part in ("all", "body", "left_hand", "right_hand", "face"):
            values[f"_paired_usage_raw/{part}/motion_square_sum"] = motion_sum
            values[f"_paired_usage_raw/{part}/motion_element_count"] = local_count
            values[f"_paired_usage_raw/{part}/off_square_sum"] = off_sum
            values[f"_paired_usage_raw/{part}/off_element_count"] = local_count
        reduced = distributed_validation_metrics(
            values,
            local_count,
            torch.device("cpu"),
            {"enabled": True, "world_size": 2},
        )
        if rank == 0:
            output_queue.put(reduced)
    finally:
        dist.destroy_process_group()


def test_cluster_equal_ddp_reduction_uses_cluster_counts(tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    init_path = Path(tmp_path) / "gloo_init"
    processes = [
        context.Process(
            target=_distributed_cluster_worker,
            args=(rank, init_path, queue),
        )
        for rank in range(2)
    ]
    # A cold spawned ARM64 worker imports PyTorch and the trajectory package
    # from shared CIFS.  That startup can legitimately exceed 30 seconds on a
    # scheduler-selected node even though the Gloo reduction itself is fast.
    deadline = time.monotonic() + 210.0
    result = None
    exitcodes = []
    started_processes = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        result = queue.get(timeout=180)
        for process in started_processes:
            process.join(timeout=max(deadline - time.monotonic(), 0.0))
        exitcodes = [process.exitcode for process in started_processes]
    finally:
        for process in started_processes:
            if process.is_alive():
                process.terminate()
        terminate_deadline = time.monotonic() + 10.0
        for process in started_processes:
            process.join(timeout=max(terminate_deadline - time.monotonic(), 0.0))
        for process in started_processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        queue.close()
        queue.join_thread()
    assert exitcodes == [0, 0]
    assert result is not None
    assert result["metric"] == pytest.approx(11.0 / 3.0)
    assert result["paired_sentence_memory/Rmotion"] == pytest.approx(
        (7.0 / 12.0) ** 0.5
    )
