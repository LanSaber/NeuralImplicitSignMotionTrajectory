from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import sha256_file
from NIAF.continuous_trajectory_field.scripts.analyze_factorized_memory_confirmation import (
    ConfirmationInputError,
    _attention_row_metrics,
    _load_development_diagnostic,
    _paired_attention_distance_rows,
    _validate_analytic_pair,
)
from NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage import (
    HOLDOUT_SPEND_SCHEMA_NAME,
    INTEGRITY_INVALID,
    STAGE1,
    OrderedDecisionError,
    _digest_json,
    _atomic_json_record,
    _recover_truncated_terminal_metric,
    _precheckpoint_output_evidence,
    _replay_selection_history,
    _require_stage2_source_match,
    _resume_progress_evidence,
    _selection_inputs_from_persisted_row,
    _terminal_selection_summary_payload,
    _validate_replayed_winner,
    _validate_selected_v2_parity,
    _write_decision,
    _validate_resolved_config,
    acquire_execution_lease,
    decide_stage,
    independently_feasible,
    release_execution_lease,
    record_precheckpoint_retry,
    spend_confirmation,
    validate_factorized_export_query_binding,
)
from NIAF.continuous_trajectory_field.scripts import (
    train_continuous_trajectory_field as trainer,
)
from NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory import (
    FactorizedAttentionCapture,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "NIAF/continuous_trajectory_field/configs"


def test_factorized_full_and_smoke_configs_encode_ordered_protocol():
    stage1_name = (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
    )
    stage2_name = (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1"
    )
    stage1 = load_config(CONFIG_DIR / f"{stage1_name}.yaml")
    stage2_path = CONFIG_DIR / f"{stage2_name}.yaml"
    stage2 = load_config(stage2_path)
    for cfg, temporal_mode in ((stage1, "none"), (stage2, "gaussian")):
        assert cfg["sentence_memory"]["key_value_mode"] == (
            "factorized_metadata_motion_v1"
        )
        assert cfg["sentence_memory"]["temporal_prior_mode"] == temporal_mode
        assert cfg["sentence_memory"]["temporal_prior_sigma"] == pytest.approx(0.25)
        assert cfg["train"]["epochs"] == 4
        assert cfg["train"]["batch_size"] == 32
        assert cfg["train"]["accumulation_steps"] == 2
        assert cfg["train"]["max_samples_per_memory_batch"] == 8
        assert cfg["train"]["max_frames_per_memory_batch"] == 2048
        assert cfg["eval"]["sentence_memory_modes"] == [
            "off",
            "on",
            "motion_shuffled",
            "shuffled",
            "analytic_prior",
        ]
        assert cfg["eval"]["evaluation_corruption"] == {
            "mode": "fixed_query_condition_v1",
            "seed": 1234,
            "nonce": "csl_daily_validation_corruption_v1",
        }
        assert cfg["selection"]["aggregation"] == (
            "normalized_text_cluster_equal_v1"
        )
        assert cfg["validation_text_partition"][
            "expected_partition_artifact_identity"
        ] == "2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
        assert cfg["validation_text_partition"][
            "expected_development_manifest_sha256"
        ] == "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
    assert stage2["train"]["base_checkpoint"] == stage1["train"]["base_checkpoint"]
    stage2_raw = yaml.safe_load(stage2_path.read_text(encoding="utf-8"))
    assert stage2_raw["eval"]["sentence_memory_modes"] == list(
        stage1["eval"]["sentence_memory_modes"]
    )
    assert stage2_raw["eval"]["evaluation_corruption"] == stage1["eval"][
        "evaluation_corruption"
    ]
    assert stage2_raw["selection"]["aggregation"] == stage1["selection"][
        "aggregation"
    ]
    for name in (stage1_name, stage2_name):
        smoke = load_config(CONFIG_DIR / f"{name}_smoke.yaml")
        assert smoke["train"]["epochs"] == 1
        assert smoke["selection"]["require_feasible"] is False


def _selection_metrics(*, correct=0.98, motion=1.0, full=1.01, off=1.02, rmotion=0.1):
    return {
        "selection_sentence_memory_score": correct,
        "selection_text_only_score": off,
        "selection_motion_shuffled_sentence_memory_score": motion,
        "selection_shuffled_sentence_memory_score": full,
        "selection_Rmotion": rmotion,
        "val_text_only/pred_loss_path_lhand": 1.0,
        "val_sentence_memory/pred_loss_path_lhand": 1.01,
        "val_text_only/pred_loss_path_rhand": 2.0,
        "val_sentence_memory/pred_loss_path_rhand": 2.02,
    }


def test_independent_stage_feasibility_enforces_motion_and_hand_gates():
    feasible, evidence = independently_feasible(_selection_metrics())
    assert feasible is True
    assert evidence["relative_improvements"]["motion_shuffled"] >= 0.001

    feasible, _ = independently_feasible(_selection_metrics(rmotion=0.049))
    assert feasible is False


def test_factorized_export_query_binding_is_digest_and_authority_bound():
    payload = {
        "schema_name": "factorized_sentence_memory_export_query_binding",
        "schema_version": 1,
        "authority": "config_pinned_development_manifest_v1",
        "partition_artifact_identity": (
            "2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
        ),
        "query_manifest": {
            "file": "manifest_development.jsonl",
            "sha256": (
                "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
            ),
            "row_count": 347,
        },
        "canonical_neighbor_table": {
            "sha256": (
                "b5f5d0914e55ba9e9c79963bacb955b97d62f99f16f117381187e72c345eb199"
            ),
            "query_manifest_sha256": (
                "2d443adf2cd489709e19d3e55b74128e7f4470b3413692e587e9dd2ed521dabe"
            ),
            "query_order_sha256": "a" * 64,
            "query_count": 1077,
        },
        "bank_id": (
            "a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
        ),
        "lookup_mode": "exact_name_indexed_parent_subset_v1",
        "online_fallback_allowed": False,
    }
    binding = {**payload, "digest": _digest_json(payload)}
    assert validate_factorized_export_query_binding(
        binding,
        expected_authority="config_pinned_development_manifest_v1",
        expected_manifest_file="manifest_development.jsonl",
        expected_manifest_sha256=payload["query_manifest"]["sha256"],
        expected_query_rows=347,
    ) == binding

    changed = copy.deepcopy(binding)
    changed["online_fallback_allowed"] = True
    changed["digest"] = _digest_json(
        {key: value for key, value in changed.items() if key != "digest"}
    )
    with pytest.raises(OrderedDecisionError, match="binding is invalid"):
        validate_factorized_export_query_binding(
            changed,
            expected_authority="config_pinned_development_manifest_v1",
            expected_manifest_file="manifest_development.jsonl",
            expected_manifest_sha256=payload["query_manifest"]["sha256"],
            expected_query_rows=347,
        )


def _synthetic_resolved_config(approved):
    resolved = copy.deepcopy(approved)
    resolved["device"] = "cuda"
    resolved["output"]["out_dir"] = str(
        ROOT / approved["output"]["out_dir"]
    )
    resolved["sentence_memory"]["resolved_behavior_identity"] = (
        trainer.sentence_memory_behavior_identity(approved)
    )
    resolved["sentence_memory"]["resolved_identity"] = {
        "bank_id": (
            "a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
        )
    }
    resolved["sentence_memory_safety"]["v2_to_v3_text_only_parity"] = {
        "passed": True,
        "prediction_max_abs": 0.0,
        "duration_max_abs": 0.0,
        "tolerance": 1e-7,
    }
    partition = resolved.setdefault("validation_text_partition", {})
    development_manifest_identity = {
        "file": "manifest_development.jsonl",
        "sha256": (
            "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
        ),
        "row_count": 347,
        "unique_text_count": 256,
    }
    runtime_payload = {
        "schema_name": "signtrajfield_development_validation_runtime",
        "schema_version": 1,
        "validation_only": True,
        "partition": {
            "digest": (
                "80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e"
            ),
            "artifact_identity": (
                "2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
            ),
        },
        "source_validation_manifest": {
            "sha256": (
                "2d443adf2cd489709e19d3e55b74128e7f4470b3413692e587e9dd2ed521dabe"
            ),
            "row_count": 1077,
        },
        "development_manifest": development_manifest_identity,
        "confirmation_counts": {"row_count": 728, "unique_text_count": 540},
        "bank_id": (
            "a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
        ),
        "retrieval_query_mode": "exact_name_indexed_full_val_table_subset_v1",
        "holdout_access": "development_manifest_only_v1",
    }
    partition.update(
        {
            "partition_digest": (
                "80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e"
            ),
            "development_row_count": 347,
            "confirmation_row_count": 728,
            "confirmation_text_count": 540,
            "partition_artifact_identity": (
                "2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
            ),
            "development_manifest_identity": development_manifest_identity,
            "development_runtime_identity": {
                "schema_name": "signtrajfield_development_validation_runtime",
                "schema_version": 1,
                "digest": _digest_json(runtime_payload),
            },
            "retrieval_query_mode": "exact_name_indexed_full_val_table_subset_v1",
            "confirmation_evaluated_during_training": False,
            "exact_seen_evaluated_during_training": False,
            "selection_aggregation_identity": (
                trainer.sentence_memory_selection_aggregation_identity(approved)
            ),
            "evaluation_corruption_map_identity": (
                trainer.sentence_memory_validation_corruption_map_identity(
                    approved,
                    partition_digest=(
                        "80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e"
                    ),
                    bank_id=(
                        "a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
                    ),
                )
            ),
        }
    )
    return resolved


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("sentence_memory", "k", 7),
        ("train", "max_frames_per_memory_batch", 1024),
        ("objective", "lambda_sentence_motion_rank", 0.5),
    ),
)
def test_resolved_runtime_config_rejects_behavior_overrides(section, key, value):
    approved = load_config(
        CONFIG_DIR
        / "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1.yaml"
    )
    resolved = _synthetic_resolved_config(approved)
    _validate_resolved_config(stage="stage1", resolved=resolved, approved=approved)
    changed = copy.deepcopy(resolved)
    changed[section][key] = value
    with pytest.raises(OrderedDecisionError, match="Resolved runtime config differs"):
        _validate_resolved_config(
            stage="stage1", resolved=changed, approved=approved
        )
    metrics = _selection_metrics()
    metrics["val_sentence_memory/pred_loss_path_lhand"] = 1.021
    feasible, _ = independently_feasible(metrics)
    assert feasible is False


def test_integrity_failure_is_persisted_without_authorization(tmp_path):
    out_dir = tmp_path / "decision"
    with patch(
        "NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage._decide_stage_validated",
        side_effect=OrderedDecisionError("synthetic integrity failure"),
    ):
        result = decide_stage(
            stage=STAGE1,
            run_dir=tmp_path,
            config_path=tmp_path,
            partition_dir=tmp_path,
            memory_off_dir=tmp_path,
            all_null_dir=tmp_path,
            v2_memory_off_dir=tmp_path,
            v2_config_path=tmp_path,
            v2_checkpoint_path=tmp_path,
            out_dir=out_dir,
        )
    assert result["status"] == INTEGRITY_INVALID
    assert not out_dir.exists()
    attempts = list(
        (tmp_path / "decision.integrity_invalid_attempts").glob("*/READY")
    )
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["authorized_purpose"] is None

    valid_payload = {
        "schema_name": "signtrajfield_factorized_memory_ordered_decision",
        "schema_version": 1,
        "stage": STAGE1,
        "status": "development_feasible",
        "integrity_valid": True,
    }
    valid = {**valid_payload, "decision_identity": _digest_json(valid_payload)}
    _write_decision(
        out_dir,
        valid,
        {"purpose": "confirmation", "authorization_identity": "d" * 64},
    )
    assert (out_dir / "READY").is_file()
    assert len(attempts) == 1


def test_confirmation_spend_marker_is_exclusive_and_bound(tmp_path):
    authorization_path = tmp_path / "authorize_confirmation.json"
    authorization_path.write_text("{}", encoding="utf-8")
    authorization = {
        "authorization_identity": "a" * 64,
        "checkpoint": {"sha256": "b" * 64},
        "partition": {"partition_digest": "c" * 64},
    }
    marker = tmp_path / "control/confirmation_holdout_spent.json"
    with patch(
        "NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage.verify_authorization",
        return_value=authorization,
    ):
        with patch(
            "NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage.os.link",
            side_effect=OSError("synthetic spend publication interruption"),
        ):
            with pytest.raises(OSError, match="spend publication interruption"):
                spend_confirmation(
                    authorization_path=authorization_path,
                    stage=STAGE1,
                    marker_path=marker,
                )
        assert not marker.exists()
        payload = spend_confirmation(
            authorization_path=authorization_path,
            stage=STAGE1,
            marker_path=marker,
        )
        assert payload["schema_name"] == HOLDOUT_SPEND_SCHEMA_NAME
        resumed = spend_confirmation(
            authorization_path=authorization_path,
            stage=STAGE1,
            marker_path=marker,
            allow_matching_existing=True,
        )
        assert resumed == payload
        with pytest.raises(OrderedDecisionError, match="already spent"):
            spend_confirmation(
                authorization_path=authorization_path,
                stage=STAGE1,
                marker_path=marker,
            )


def test_atomic_immutable_json_publication_leaves_no_partial_final(tmp_path):
    target = tmp_path / "resume_attempts/42.0.json"
    with patch(
        "NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage.os.link",
        side_effect=OSError("synthetic publication interruption"),
    ):
        with pytest.raises(OSError, match="synthetic publication interruption"):
            _atomic_json_record(target, {"complete": True})
    assert not target.exists()
    assert list(target.parent.glob(f".{target.name}.building.*")) == []
    assert _atomic_json_record(target, {"complete": True}) == {"complete": True}
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}


def _attention_export(tmp_path: Path, name: str, *, analytic: bool = False):
    directory = tmp_path / name
    directory.mkdir()
    slots, parts, candidates, tokens = 3, 4, 2, 2
    null = np.full((slots, parts), 0.2, dtype=np.float32)
    candidate = np.zeros((slots, parts, candidates), dtype=np.float32)
    candidate[..., 0] = 0.6 if not analytic else 0.4
    candidate[..., 1] = 0.2 if not analytic else 0.4
    token_mass = np.repeat(candidate[..., None] / tokens, tokens, axis=-1)
    token_tau = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    token_mask = np.ones((candidates, tokens), dtype=bool)
    validity = np.ones((candidates, tokens, parts), dtype=np.float32)
    ids = np.asarray([11, 12], dtype=np.int64)
    row = {
        "name": "sample",
        "text": "hello",
        "sample": str(directory / "sample_0000.npz"),
    }
    np.savez(
        row["sample"],
        sentence_memory_part_null_mass=null,
        sentence_memory_part_candidate_mass=candidate,
        sentence_memory_part_token_mass=token_mass,
        sentence_memory_token_tau=token_tau,
        sentence_memory_token_mask=token_mask,
        sentence_memory_part_validity=validity,
        trajectory_sentence_memory_gates=np.full((slots, parts), 0.3),
        sentence_memory_ids=ids,
        sentence_memory_scores=np.asarray([0.9, 0.8]),
        sentence_memory_durations=np.asarray([2.0, 2.2]),
        sentence_memory_duration_log_gap=np.asarray([0.0, 0.1]),
        sentence_memory_candidate_mask=np.ones(candidates, dtype=bool),
        sentence_memory_group_ids=np.asarray([21, 22]),
        sentence_memory_exact_text=np.zeros(candidates, dtype=bool),
        sentence_memory_query_seen_text=np.asarray(False),
        sentence_memory_attention_mode=np.asarray(
            "analytic_prior" if analytic else "learned"
        ),
    )
    return {
        "dir": directory,
        "rows": [row],
        "summary": {
            "factorized_attention_artifacts": {
                "schema": "final_layer_part_attention_mass_v1",
                "layer": "final",
                "head_reduction": "mean",
                "part_order": ["body", "lhand", "rhand", "face"],
            }
        },
    }


def test_factorized_attention_metrics_and_analytic_input_integrity(tmp_path):
    learned = _attention_export(tmp_path, "learned")
    analytic = _attention_export(tmp_path, "analytic", analytic=True)
    values = _attention_row_metrics(learned)
    assert values["body/effective_k"].shape == (1,)
    assert values["body/maximum_candidate_share"][0] == pytest.approx(0.75)
    assert np.isfinite(values["body/source_time_correlation"]).all()
    assert _validate_analytic_pair(learned, analytic)["passed"] is True
    distances = _paired_attention_distance_rows(learned, analytic)
    assert distances["conditional_candidate_total_variation"][0] > 0.0
    assert distances["conditional_token_total_variation"][0] > 0.0

    analytic_sample = Path(analytic["rows"][0]["sample"])
    with np.load(analytic_sample, allow_pickle=False) as payload:
        changed = {name: payload[name] for name in payload.files}
    changed["sentence_memory_ids"] = np.asarray([12, 11])
    np.savez(analytic_sample, **changed)
    with pytest.raises(ConfirmationInputError, match="metadata"):
        _validate_analytic_pair(learned, analytic)


def test_locked_development_diagnostic_rejects_source_detachment(
    tmp_path, monkeypatch
):
    from NIAF.continuous_trajectory_field.scripts import (
        diagnose_temporal_slots as diagnostic,
    )

    directory = tmp_path / "diagnostic"
    directory.mkdir()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "stage.yaml"
    config.write_text("experiment_name: stage\n", encoding="utf-8")
    authorization_path = tmp_path / "authorize_confirmation.json"
    source_head = "a" * 40
    source = {
        "git_head": source_head,
        "remote_ref": "origin/immutable-run",
        "remote_head": source_head,
        "standalone_shared_clone_checked": True,
        "worktree_clean_checked": True,
        "remote_ref_exact_match_checked": True,
        "repository_root": "/media/cvpr/source",
        "git_directory": "/media/cvpr/source/.git",
        "durable_experiments_root": "/media/cvpr/experiments",
        "frozen_text_model_root": "/media/cvpr/deps/mt5-base",
    }
    authorization = {
        "authorization_identity": "b" * 64,
        "decision_identity": "c" * 64,
        "run_launch_identity": {"source": source},
        "checkpoint": {
            "neighbor_table_sha256": {
                "train": "d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852",
                "val": "b5f5d0914e55ba9e9c79963bacb955b97d62f99f16f117381187e72c345eb199",
            }
        },
        "partition": {
            "development_manifest_sha256": (
                "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
            )
        },
    }
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    source_hashes = {"diagnose_temporal_slots.py": "d" * 64}
    monkeypatch.setattr(diagnostic, "_diagnostic_source_hashes", lambda: source_hashes)
    summary = {"stage": "full", "validation_only": True}
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    artifact_manifest = {
        "schema": "signtrajfield_temporal_slot_artifacts",
        "version": 1,
        "artifacts": {
            "summary.json": {
                "bytes": summary_path.stat().st_size,
                "sha256": sha256_file(summary_path),
            }
        },
    }
    manifest_path = directory / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")
    locked = {
        "source_git_head": source_head,
        "run_launch_source": source,
        "source_checkout": {
            "git_head": source_head,
            "standalone_shared_clone_checked": True,
            "worktree_clean_checked": True,
            "repository_root": source["repository_root"],
            "git_directory": source["git_directory"],
            "durable_experiments_root": source["durable_experiments_root"],
            "frozen_text_model_root": source["frozen_text_model_root"],
        },
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "authorization_identity": authorization["authorization_identity"],
        "decision_identity": authorization["decision_identity"],
    }
    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "git_head": source_head,
        "git": {"commit": source_head, "tracked_worktree_clean": True},
        "source_sha256": source_hashes,
        "locked_run_evidence": locked,
        "retrieval_lookup_evidence": {
            "schema_name": (
                "signtrajfield_factorized_development_neighbor_lookup"
            ),
            "schema_version": 1,
            "lookup_mode": "exact_name_indexed_parent_subset_v1",
            "online_retrieval_fallback": "forbidden",
            "canonical_table": {
                "split": "val",
                "sha256": (
                    "b5f5d0914e55ba9e9c79963bacb955b97d62f99f16f117381187e72c345eb199"
                ),
                "parent_query_manifest_sha256": (
                    "2d443adf2cd489709e19d3e55b74128e7f4470b3413692e587e9dd2ed521dabe"
                ),
                "parent_query_order_sha256": "e" * 64,
                "parent_query_count": 1077,
            },
            "development_subset": {
                "query_manifest_sha256": (
                    "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
                ),
                "query_order_sha256": "f" * 64,
                "query_count": 347,
                "parent_query_rows_sha256": "1" * 64,
                "parent_query_rows_unique": True,
                "parent_query_row_min": 0,
                "parent_query_row_max": 1000,
            },
        },
        "query_counts": {"passive_rows": 347, "causal_rows": 128},
        "artifact_manifest_sha256": sha256_file(manifest_path),
    }
    provenance_path = directory / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    ready = {
        "schema_name": "signtrajfield_temporal_slot_diagnostics",
        "provenance_sha256": sha256_file(provenance_path),
    }
    ready_path = directory / "READY"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    _load_development_diagnostic(
        directory,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        config_path=config,
        authorization_path=authorization_path,
        authorization=authorization,
    )

    provenance["git_head"] = "e" * 40
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    ready["provenance_sha256"] = sha256_file(provenance_path)
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(ConfirmationInputError, match="detached"):
        _load_development_diagnostic(
            directory,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            config_path=config,
            authorization_path=authorization_path,
            authorization=authorization,
        )

    provenance["git_head"] = source_head
    provenance["retrieval_lookup_evidence"]["lookup_mode"] = "online"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    ready["provenance_sha256"] = sha256_file(provenance_path)
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(ConfirmationInputError, match="exact-name"):
        _load_development_diagnostic(
            directory,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            config_path=config,
            authorization_path=authorization_path,
            authorization=authorization,
        )


def test_factorized_attention_capture_cleans_up_hook():
    layer = torch.nn.Identity()
    model = SimpleNamespace(
        hypernetwork=SimpleNamespace(
            temporal_slot_count=2,
            sentence_memory_encoder=SimpleNamespace(
                layers=torch.nn.ModuleList([layer])
            ),
        )
    )
    capture = FactorizedAttentionCapture(
        model,
        {"sentence_memory": {"key_value_mode": "factorized_metadata_motion_v1"}},
    )
    state = torch.zeros(1, 8, 3)
    null = torch.full((1, 8), 0.2)
    candidate = torch.full((1, 8, 2), 0.4)
    token = torch.full((1, 8, 2, 2), 0.2)
    capture._hook(layer, (), (state, null, candidate, token))
    assert capture.row(0)["candidate_mass"].shape == (2, 4, 2)
    capture.close()
    assert capture.handle is None


def _decorate_selection_rows(rows):
    early = trainer.initial_early_stopping_state()
    decorated = []
    for epoch, row in enumerate(rows, 1):
        value = {**row, "epoch": epoch, "validation_pending": 0.0}
        feasible, _ = independently_feasible(value)
        early, improved, stopped = trainer.update_early_stopping_state(
            early,
            feasible=feasible,
            normalized_constraint_violation=value["selection_constraint_violation"],
            selection_score_value=value["selection_score"],
            epoch=epoch,
            patience=2,
            minimum_epoch=2,
        )
        value.update(
            {
                "selection_feasible": float(feasible),
                "early_stopping_improved": float(improved),
                "early_stopping_bad_validation_count": float(
                    early["bad_validation_count"]
                ),
                "early_stopping_validation_count": float(early["validation_count"]),
                "early_stopping_requested": float(stopped),
            }
        )
        decorated.append(value)
    return decorated


def test_selection_replay_uses_strict_lexicographic_winner_and_patience():
    rows = _decorate_selection_rows(
        [
            {
                **_selection_metrics(correct=1.01, motion=1.0, full=1.0, off=1.0),
                "selection_score": 1.1,
                "selection_constraint_violation": 0.2,
            },
            {
                **_selection_metrics(correct=0.98),
                "selection_score": 0.9,
                "selection_constraint_violation": 0.0,
            },
            {
                **_selection_metrics(correct=0.98),
                "selection_score": 0.9,
                "selection_constraint_violation": 0.0,
            },
        ]
    )
    replay = _replay_selection_history(rows)
    assert replay["best_feasible_row"]["epoch"] == 2
    assert replay["best_feasible_score"] == pytest.approx(0.9)
    assert replay["best_infeasible_key"] == [0.2, 1.1]
    assert replay["early_stopping"]["bad_validation_count"] == 1
    assert replay["early_stopping"]["stopped"] is False
    winner = replay["best_feasible_row"]
    checkpoint = {
        "epoch": winner["epoch"],
        "metrics": winner,
        "selection_state": replay["selection_states"][winner["epoch"]],
    }
    assert _validate_replayed_winner(
        checkpoint, replay, has_feasible=True
    ) == winner
    stale = copy.deepcopy(checkpoint)
    stale["epoch"] = rows[0]["epoch"]
    stale["metrics"] = rows[0]
    stale["selection_state"] = replay["selection_states"][rows[0]["epoch"]]
    with pytest.raises(OrderedDecisionError, match="lexicographic winner"):
        _validate_replayed_winner(stale, replay, has_feasible=True)

    changed = copy.deepcopy(rows)
    changed[1]["early_stopping_requested"] = 1.0
    with pytest.raises(OrderedDecisionError, match="early-stop diagnostics"):
        _replay_selection_history(changed)


def test_persisted_validation_metrics_restore_selection_namespaces():
    persisted = {
        "epoch": 2,
        "val_text_only/pred_loss_endpoint": 1.0,
        "val_sentence_memory/pred_loss_endpoint": 0.9,
        "val_motion_shuffled_sentence_memory/pred_loss_endpoint": 1.1,
        "val_shuffled_sentence_memory/pred_loss_endpoint": 1.2,
        "val_analytic_prior_sentence_memory/pred_loss_endpoint": 0.95,
        "selection_score": 0.9,
    }
    restored = _selection_inputs_from_persisted_row(persisted)
    assert set(restored) == {
        "text_only/pred_loss_endpoint",
        "sentence_memory/pred_loss_endpoint",
        "motion_shuffled_sentence_memory/pred_loss_endpoint",
        "shuffled_sentence_memory/pred_loss_endpoint",
        "analytic_prior_sentence_memory/pred_loss_endpoint",
    }
    assert not any(name.startswith("val_") for name in restored)


def test_execution_lease_rejects_live_owner_and_recovers_proven_stale(tmp_path):
    lease = tmp_path / "active_execution_lease"
    first_attestation = tmp_path / "attempts/100.json"
    first = acquire_execution_lease(
        lease_path=lease,
        attestation_path=first_attestation,
        purpose="training",
        stage="stage1",
        source_git_head="a" * 40,
        slurm_job_id="100",
        binding_identity="b" * 64,
    )
    assert lease.is_dir()
    assert first["claim"]["slurm_job_id"] == "100"

    live_error = OrderedDecisionError(
        "Execution lease is owned by live Slurm job 100 (RUNNING)"
    )
    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "decide_factorized_memory_stage._slurm_job_terminal_evidence",
        side_effect=live_error,
    ):
        with pytest.raises(OrderedDecisionError, match="live Slurm job"):
            acquire_execution_lease(
                lease_path=lease,
                attestation_path=tmp_path / "attempts/101-live.json",
                purpose="training",
                stage="stage1",
                source_git_head="a" * 40,
                slurm_job_id="101",
                binding_identity="b" * 64,
            )
    assert json.loads((lease / "owner.json").read_text())["slurm_job_id"] == "100"

    terminal = {
        "job_id": "100",
        "classification": "terminal",
        "state": "TIMEOUT",
        "scontrol_returncode": 0,
    }
    second_attestation = tmp_path / "attempts/101.json"
    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "decide_factorized_memory_stage._slurm_job_terminal_evidence",
        return_value=terminal,
    ):
        second = acquire_execution_lease(
            lease_path=lease,
            attestation_path=second_attestation,
            purpose="training",
            stage="stage1",
            source_git_head="a" * 40,
            slurm_job_id="101",
            binding_identity="b" * 64,
        )
    assert second["claim"]["slurm_job_id"] == "101"
    assert second["claim"]["replaced_stale_owner"]["scheduler_evidence"] == terminal
    assert any(lease.with_name(f"{lease.name}.history").glob("100.*.stale"))
    with pytest.raises(OrderedDecisionError, match="active attempt"):
        release_execution_lease(
            lease_path=lease, attestation_path=first_attestation
        )
    released = release_execution_lease(
        lease_path=lease, attestation_path=second_attestation
    )
    assert released["released"] is True
    assert not lease.exists()


def test_stage2_source_must_equal_stage1_authorized_source():
    source = {
        "git_head": "a" * 40,
        "remote_ref": "origin/immutable-factorized-run",
        "remote_head": "a" * 40,
        "repository_root": "/media/cvpr/factorized-source",
        "git_directory": "/media/cvpr/factorized-source/.git",
        "durable_experiments_root": "/media/cvpr/experiments",
        "frozen_text_model_root": "/media/cvpr/deps/mt5-base",
        "standalone_shared_clone_checked": True,
        "worktree_clean_checked": True,
        "remote_ref_exact_match_checked": True,
    }
    authorization = {"run_launch_identity": {"source": source}}
    assert _require_stage2_source_match(authorization, source) == source
    changed = copy.deepcopy(source)
    changed["git_head"] = "b" * 40
    changed["remote_head"] = "b" * 40
    with pytest.raises(OrderedDecisionError, match="predeclared Stage 1 source"):
        _require_stage2_source_match(authorization, changed)


def test_precheckpoint_retry_preserves_repeated_initialization_outputs(tmp_path):
    run_dir = tmp_path / (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
    )
    prerequisites = Path(f"{run_dir}.prerequisites")
    prerequisites.mkdir()
    launch_path = prerequisites / "run_launch_identity.json"
    launch_path.write_text("{}", encoding="utf-8")
    config = tmp_path / "stage1.yaml"
    config.write_text("experiment_name: stage1\n", encoding="utf-8")
    development = tmp_path / "manifest_development.jsonl"
    development.write_text("{}\n", encoding="utf-8")
    source = {"git_head": "a" * 40}
    launch = {
        "source": source,
        "slurm": {"job_id": "10"},
        "launch_identity": "c" * 64,
    }

    def claim(job_id, prior):
        return {
            "purpose": "training",
            "stage": "stage1",
            "source_git_head": "a" * 40,
            "binding_identity": sha256_file(config),
            "slurm_job_id": str(job_id),
            "claim_identity": str(job_id).zfill(64),
            "replaced_stale_owner": {
                "claim_identity": str(prior).zfill(64),
                "slurm_job_id": str(prior),
                "scheduler_evidence": {
                    "classification": "terminal",
                    "state": "TIMEOUT",
                },
            },
        }

    common_patches = (
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "decide_factorized_memory_stage._validate_config",
            return_value={},
        ),
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "decide_factorized_memory_stage._validate_partition",
            return_value={
                "development_manifest": development,
                "development_manifest_sha256": sha256_file(development),
            },
        ),
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "decide_factorized_memory_stage._validate_run_launch",
            return_value=launch,
        ),
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "decide_factorized_memory_stage._validate_shared_source_checkout",
            return_value=source,
        ),
    )
    for context in common_patches:
        context.start()
    try:
        for attempt, prior in ((20, 10), (30, 20)):
            run_dir.mkdir()
            (run_dir / "config.resolved.json").write_text(
                json.dumps(
                    {
                        "experiment_name": run_dir.name,
                        "validation_text_partition": {},
                    }
                ),
                encoding="utf-8",
            )
            lease_claim = claim(attempt, prior)
            with patch(
                "NIAF.continuous_trajectory_field.scripts."
                "decide_factorized_memory_stage."
                "_validate_execution_lease_attestation",
                return_value=({}, lease_claim),
            ):
                result = record_precheckpoint_retry(
                    stage="stage1",
                    run_dir=run_dir,
                    out_file=prerequisites
                    / f"precheckpoint_retry_attempts/{attempt}.json",
                    config_path=config,
                    partition_dir=tmp_path,
                    lease_path=tmp_path / "lease",
                    lease_attestation_path=tmp_path / "lease.json",
                    source_root=tmp_path,
                    source_git_head="a" * 40,
                    source_remote_ref="origin/run",
                    source_remote_head="a" * 40,
                )
            assert len(result["quarantined_outputs"]) == (1 if attempt == 20 else 2)
            assert not run_dir.exists()

        # A later crash before recreating OUT_DIR is also recoverable; all
        # earlier safe trees remain bound rather than deleted.
        lease_claim = claim(40, 30)
        with patch(
            "NIAF.continuous_trajectory_field.scripts."
            "decide_factorized_memory_stage._validate_execution_lease_attestation",
            return_value=({}, lease_claim),
        ):
            result = record_precheckpoint_retry(
                stage="stage1",
                run_dir=run_dir,
                out_file=prerequisites / "precheckpoint_retry_attempts/40.json",
                config_path=config,
                partition_dir=tmp_path,
                lease_path=tmp_path / "lease",
                lease_attestation_path=tmp_path / "lease.json",
                source_root=tmp_path,
                source_git_head="a" * 40,
                source_remote_ref="origin/run",
                source_remote_head="a" * 40,
            )
        assert len(result["quarantined_outputs"]) == 2
    finally:
        for context in reversed(common_patches):
            context.stop()


def test_precheckpoint_evidence_accepts_only_unpublished_atomic_temps(tmp_path):
    run_dir = tmp_path / (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1"
    )
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    partition_tmp = run_dir / "validation_text_partition.json.tmp"
    partition_tmp.write_text('{"schema_name":', encoding="utf-8")
    checkpoint_tmp = checkpoints / ".last.pt.AbC_1234.tmp"
    checkpoint_tmp.write_bytes(b"partial atomic checkpoint")
    evidence = _precheckpoint_output_evidence(run_dir, stage="stage1")
    assert evidence["published_checkpoints_absent"] is True
    assert evidence["artifacts"][partition_tmp.name]["json_complete"] is False
    assert evidence["unpublished_checkpoint_temporaries"] == [
        {
            "file": f"checkpoints/{checkpoint_tmp.name}",
            "bytes": checkpoint_tmp.stat().st_size,
            "sha256": sha256_file(checkpoint_tmp),
            "published": False,
        }
    ]

    (checkpoints / "best.pt").write_bytes(b"material")
    with pytest.raises(OrderedDecisionError, match="published or unknown"):
        _precheckpoint_output_evidence(run_dir, stage="stage1")


def _resume_rng_state():
    return {
        "world_size": 4,
        "rank_states": [
            {
                "rank": rank,
                "state": {
                    "python": (3, (rank, 1, 2), None),
                    "numpy": {
                        "bit_generator": "MT19937",
                        "state": [rank, 1, 2],
                        "position": 0,
                        "has_gauss": 0,
                        "cached_gaussian": 0.0,
                    },
                    "torch_cpu": torch.tensor([rank, 1], dtype=torch.uint8),
                    "torch_cuda": torch.tensor([rank, 2], dtype=torch.uint8),
                },
            }
            for rank in range(4)
        ],
    }


def test_pending_validation_resume_accepts_epoch_one_schema_two():
    early = trainer.initial_early_stopping_state()
    selection = trainer.checkpoint_selection_state(
        float("inf"),
        float("inf"),
        early_stopping_state=early,
        best_infeasible_key=None,
    )
    assert selection["schema_version"] == 2
    pending_row = {"epoch": 1, "global_step": 10, "validation_pending": 1.0}
    checkpoint = {
        "epoch": 1,
        "metrics": pending_row,
        "selection_state": selection,
        "rng_state": _resume_rng_state(),
    }
    evidence = _resume_progress_evidence(checkpoint, [])
    assert evidence["mode"] == "validation_pending"
    assert evidence["completed_validation_epochs"] == []


def test_truncated_final_metrics_recovery_is_prefix_bound(tmp_path):
    previous = {"epoch": 1, "validation_pending": 0.0, "selection_score": 2.0}
    current = {"epoch": 2, "validation_pending": 0.0, "selection_score": 1.0}
    path = tmp_path / "metrics.jsonl"
    current_json = json.dumps(current, sort_keys=True)
    path.write_text(
        json.dumps(previous, sort_keys=True) + "\n" + current_json[:17],
        encoding="utf-8",
    )
    rows, evidence = _recover_truncated_terminal_metric(path, current, 2)
    assert rows == [previous, current]
    assert evidence["kind"] == "truncated_final_append"

    path.write_text(json.dumps(previous) + "\n" + "{wrong", encoding="utf-8")
    with pytest.raises(OrderedDecisionError, match="not a truncated"):
        _recover_truncated_terminal_metric(path, current, 2)


def test_terminal_resume_summary_matches_trainer_schema():
    early = trainer.initial_early_stopping_state()
    early.update(
        {
            "best_key": [1, 0.1, 1.0],
            "last_key": [1, 0.2, 1.1],
            "bad_validation_count": 2,
            "validation_count": 3,
            "last_validation_epoch": 3,
            "stopped": True,
            "stop_epoch": 3,
        }
    )
    selection = trainer.checkpoint_selection_state(
        float("inf"),
        1.0,
        early_stopping_state=early,
        best_infeasible_key=(0.1, 1.0),
    )
    summary = _terminal_selection_summary_payload(selection)
    assert summary["has_feasible_checkpoint"] is False
    assert summary["best_infeasible_key"] == [0.1, 1.0]
    assert summary["early_stopping"]["stop_epoch"] == 3


def test_selected_checkpoint_v2_development_parity_gate(tmp_path):
    def export(name, offset):
        directory = tmp_path / name
        directory.mkdir()
        sample = directory / "row.npz"
        np.savez(sample, rot6d=np.full((2, 55, 6), offset, dtype=np.float32))
        return {
            "dir": directory,
            "rows": [{"sample": str(sample)}],
            "durations": np.asarray([2.0 + offset], dtype=np.float64),
        }

    selected = export("selected", 0.0)
    baseline = export("baseline", 0.0)
    assert _validate_selected_v2_parity(selected, baseline)["passed"] is True
    changed = export("changed", 2e-7)
    with pytest.raises(OrderedDecisionError, match="differs from pinned v2"):
        _validate_selected_v2_parity(selected, changed)


def test_factorized_launchers_are_validation_only_and_ordered():
    scripts = ROOT / "scripts/NIAF"
    stage_driver = (scripts / "run_csl_daily_factorized_memory_stage.sh").read_text()
    decision = (
        scripts
        / "decide_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
    ).read_text()
    confirmation = (
        scripts
        / "confirm_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
    ).read_text()
    smoke = (
        scripts
        / "smoke_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_motion_contrast_v1_sbatch.sh"
    ).read_text()
    resume = (
        scripts
        / "resume_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
    ).read_text()
    assert "git ls-remote --heads origin" in stage_driver
    assert 'PROJECT_DIR="${PROJECT_DIR:?' in stage_driver
    assert '[[ ! -d "$PROJECT_DIR/.git"' in stage_driver
    assert 'realpath -e "$PROJECT_DIR/experiments"' in stage_driver
    assert 'realpath -e "$PROJECT_DIR/deps/mt5-base"' in stage_driver
    assert "--source_remote_head" in stage_driver
    assert "--source_root" in stage_driver
    assert "record-run-launch" in stage_driver
    assert "record-stage2-input" in stage_driver
    assert "record-run-resume" in stage_driver
    assert (
        "resume_attempts/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}.json"
        in stage_driver
    )
    assert "record-precheckpoint-retry" in stage_driver
    assert "precheckpoint_outputs" not in stage_driver
    assert 'if [[ ! -f "$RUN_LAUNCH_FILE" ]]' in stage_driver
    assert (
        '&& ! -f "${OUT_DIR}.prerequisites/ordered_stage_input.json"'
        in stage_driver
    )
    assert "acquire-execution-lease" in stage_driver
    assert "release-execution-lease" in stage_driver
    assert "SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR" in stage_driver
    assert "build_validation_text_partition" not in stage_driver
    assert "selection_summary.json" in stage_driver
    assert 'SPLITS="train val"' in stage_driver
    assert "neighbors_test" not in stage_driver
    assert "manifest_confirmation.jsonl" not in stage_driver
    assert "partition.json" not in stage_driver
    assert 'FACTORIZED_RESUME=1' in resume
    assert 'PROJECT_DIR="${PROJECT_DIR:?' in resume
    assert "manifest_confirmation.jsonl" not in decision
    assert "partition.json" not in decision
    assert "SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256" in decision
    assert "SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS=347" in decision
    assert (
        "SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY"
        in decision
    )
    assert 'SENTENCE_MEMORY_DIR="${SENTENCE_MEMORY_DIR:-' in decision
    assert 'LOCAL_SENTENCE_MEMORY_DIR="/tmp/signtraj_sentence_memory_${SLURM_JOB_ID}"' in decision
    assert "STAGE_ONLY_REQUESTED_NEIGHBORS=1" in decision
    assert "stage_sentence_memory_node.sh" in decision
    assert '"$PROJECT_DIR" "$PYTHON_BIN" "$CFG" "train val"' in decision
    assert 'neighbors_test.npz" ]]' in decision
    assert 'SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"' in decision
    assert 'SIGNTRAJ_SENTENCE_MEMORY_DIR="$SENTENCE_MEMORY_DIR"' not in decision
    assert 'cleanup "$LOCAL_SENTENCE_MEMORY_DIR" || true' in decision
    assert decision.index("STAGE_ONLY_REQUESTED_NEIGHBORS=1") < decision.index(
        "run_export off"
    )
    assert decision.index(
        'SIGNTRAJ_SENTENCE_MEMORY_DIR="$LOCAL_SENTENCE_MEMORY_DIR"'
    ) < decision.index("run_export off")
    assert "acquire-execution-lease" in decision
    assert "original_v2_text_only" in decision
    assert "--v2_memory_off_dir" in decision
    assert "spend-confirmation" in confirmation
    assert confirmation.index("spend-confirmation --stage") < confirmation.index(
        '--manifest "$PARTITION_DIR/manifest_confirmation.jsonl"'
    )
    assert confirmation.index("spend-confirmation --stage") < confirmation.index(
        "SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256="
    )
    assert "--allow_matching_existing" in confirmation
    assert "active_confirmation_execution_lease" in confirmation
    assert "publish_unit_no_replace" in confirmation
    assert 'mv -- "$build_dir" "$mode_dir"' not in confirmation
    assert (
        ".building.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
        in confirmation
    )
    assert "SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256" in confirmation
    assert "run_scored_mode analytic_prior analytic_prior_sentence_memory" in confirmation
    assert 'PROJECT_DIR="${PROJECT_DIR:?' in smoke
    assert "git ls-remote --heads origin" in smoke
    assert "build_validation_text_partition" not in smoke
    assert "SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR" in smoke
    assert "train_sentence_memory_motion_corrupt_fraction" in smoke
    assert "train_sentence_memory_full_shuffle_fraction" in smoke
    assert "manifest_confirmation.jsonl" not in smoke
    assert "partition.json" not in smoke
