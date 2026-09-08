import copy
import hashlib
import json
import math

import pytest
import torch

from NIAF.continuous_trajectory_field.scripts.evaluate_continuous_trajectory_field import (
    _write_json_atomic,
    validate_centered_public_evaluation_scope,
)
from NIAF.continuous_trajectory_field.scripts import (
    decide_centered_memory_stage as centered_decision,
)

from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryBatch,
    broadcast_sentence_memory_motion_payload,
    joint_tuple_permute_sentence_memory_batch,
    motion_only_shuffle_sentence_memory_batch,
    replace_sentence_memory_motion_payload,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    balanced_absolute_association_bce,
    checkpoint_selection_diagnostics,
    configured_sentence_memory_eval_modes,
    resolve_sentence_memory_relevance_calibration,
    sentence_memory_architecture_identity,
    sentence_memory_objective_identity,
    sentence_memory_association_losses,
    symmetric_masked_multi_positive_infonce,
    validate_sentence_memory_architecture_identity,
)


def _relevance_cfg(artifact_dir):
    cfg = _paired_cfg()
    cfg["sentence_memory"].update(
        {
            "duration_weight": 0.05,
            "relevance_gate_mode": "frozen_absolute_adjusted_score_v1",
            "relevance_calibration": {
                "artifact_dir": str(artifact_dir),
                "schema_name": (
                    "signtrajfield_sentence_memory_relevance_calibration"
                ),
                "schema_version": 1,
                "minimum_heldout_auroc": 0.75,
                "minimum_heldout_probability_gap": 0.20,
            },
        }
    )
    return cfg


def _memory(batch=2, candidates=3, tokens=2, *, offset=0):
    ids = torch.arange(batch * candidates).reshape(batch, candidates) + offset
    groups = torch.tensor(
        [[10, 11, 12], [10, 13, 14]], dtype=torch.long
    )[:batch, :candidates]
    rank = ids[..., None, None].float()
    motion = rank.expand(batch, candidates, tokens, 2).clone()
    token_mask = torch.ones(batch, candidates, tokens, dtype=torch.bool)
    return SentenceMemoryBatch(
        tokens=motion,
        token_mask=token_mask,
        token_tau=motion[..., 0],
        candidate_mask=torch.ones(batch, candidates, dtype=torch.bool),
        candidate_keys=torch.stack((ids.float(), -ids.float()), dim=-1),
        scores=ids.float() / 10,
        durations=ids.float() + 1,
        duration_log_gap=ids.float() / 100,
        part_validity=torch.ones(batch, candidates, tokens, 4),
        ids=ids,
        available=torch.ones(batch, dtype=torch.bool),
        provenance={
            "mode": "on",
            "candidate_group_ids": groups.tolist(),
            "candidate_names": [
                [f"r{row}k{rank}" for rank in range(candidates)]
                for row in range(batch)
            ],
        },
        group_ids=groups,
        source_group_ids=ids + 1000,
        motion_source_ids=ids.clone(),
        motion_source_group_ids=groups.clone(),
        motion_source_source_group_ids=ids + 1000,
    )


def test_fixed_motion_and_joint_controls_keep_identity_sidecars_aligned():
    memory = _memory()
    shuffled, permutation, informative = motion_only_shuffle_sentence_memory_batch(
        memory,
        query_ids=("q0", "q1"),
        seed=1234,
        corruption_nonce="csl_daily_pair_derangement_v1_n0",
        corruption_condition="motion_shuffled_n0",
    )
    assert informative.tolist() == [True, True]
    assert torch.equal(shuffled.ids, memory.ids)
    assert torch.equal(shuffled.group_ids, memory.group_ids)
    assert torch.equal(shuffled.source_group_ids, memory.source_group_ids)
    assert shuffled.provenance["evaluation_corruption_mode"] == (
        "fixed_evidence_controls_v1"
    )
    assert shuffled.provenance["motion_source_ids"] == torch.gather(
        memory.ids, 1, permutation
    ).tolist()
    assert shuffled.provenance["motion_source_group_ids"] == torch.gather(
        memory.group_ids, 1, permutation
    ).tolist()
    assert torch.equal(
        shuffled.motion_source_ids, torch.gather(memory.ids, 1, permutation)
    )

    joint, joint_permutation, joint_informative = (
        joint_tuple_permute_sentence_memory_batch(
            memory,
            query_ids=("q0", "q1"),
            seed=1234,
            corruption_nonce="csl_daily_joint_tuple_permutation_v1",
        )
    )
    assert joint_informative.all()
    assert torch.equal(joint.ids, torch.gather(memory.ids, 1, joint_permutation))
    assert torch.equal(
        joint.group_ids, torch.gather(memory.group_ids, 1, joint_permutation)
    )
    assert torch.equal(
        joint.source_group_ids,
        torch.gather(memory.source_group_ids, 1, joint_permutation),
    )
    assert joint.provenance["candidate_group_ids"] == joint.group_ids.tolist()


def test_cross_and_broadcast_controls_move_only_complete_motion_payloads():
    destination = _memory(offset=0)
    source = _memory(offset=100)
    cross, informative = replace_sentence_memory_motion_payload(
        destination,
        source,
        corruption_nonce="csl_daily_cross_query_motion_v1",
    )
    assert informative.all()
    assert torch.equal(cross.tokens, source.tokens)
    assert torch.equal(cross.ids, destination.ids)
    assert torch.equal(cross.group_ids, destination.group_ids)
    assert torch.equal(cross.source_group_ids, destination.source_group_ids)
    assert cross.provenance["motion_source_ids"] == source.ids.tolist()

    broadcast, source_rank, informative = broadcast_sentence_memory_motion_payload(
        destination, query_ids=("q0", "q1")
    )
    assert informative.all()
    assert torch.equal(broadcast.ids, destination.ids)
    assert torch.equal(broadcast.group_ids, destination.group_ids)
    assert torch.equal(broadcast.source_group_ids, destination.source_group_ids)
    for row, rank in enumerate(source_rank.tolist()):
        assert torch.equal(
            broadcast.tokens[row],
            destination.tokens[row, rank][None].expand_as(broadcast.tokens[row]),
        )


def test_balanced_absolute_bce_is_class_balanced_and_has_safe_gradients():
    positives = torch.tensor([[0.0, 0.0, 9.0]], requires_grad=True)
    negatives = torch.tensor([[0.0, -9.0]], requires_grad=True)
    loss, diagnostics = balanced_absolute_association_bce(
        positives,
        torch.tensor([[True, True, False]]),
        negatives,
        torch.tensor([[True, False]]),
    )
    assert loss.item() == pytest.approx(math.log(2.0))
    assert diagnostics["positive_count"].item() == 2
    assert diagnostics["negative_count"].item() == 1
    loss.backward()
    assert positives.grad[0, :2].lt(0).all()
    assert negatives.grad[0, 0] > 0
    assert positives.grad[0, 2] == 0
    assert negatives.grad[0, 1] == 0


def test_infonce_uses_repeated_groups_across_rows_as_multi_positives():
    keys = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [-1.0, 0.0]]],
        requires_grad=True,
    )
    motions = keys.detach().clone().requires_grad_(True)
    mask = torch.ones(2, 2, dtype=torch.bool)
    groups = torch.tensor([[10, 11], [10, 12]])
    items = torch.tensor([[0, 1], [2, 3]])
    loss, diagnostics = symmetric_masked_multi_positive_infonce(
        keys,
        motions,
        mask,
        groups,
        items,
        temperature=0.10,
        source_group_ids=torch.tensor([[100, 101], [102, 103]]),
    )
    assert torch.isfinite(loss)
    # Group 10 contributes the two self pairs and both cross-row variants.
    assert diagnostics["positive_pair_count"].item() == 6
    assert diagnostics["negative_pair_count"].item() == 10
    assert diagnostics["key_anchor_count"].item() == 4
    assert diagnostics["motion_anchor_count"].item() == 4
    loss.backward()
    assert torch.isfinite(keys.grad).all()
    assert torch.isfinite(motions.grad).all()


def test_infonce_source_sign_variants_are_never_false_negatives():
    descriptors = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    groups = torch.tensor([[10, 11, 12]])
    items = torch.tensor([[0, 1, 2]])
    _, baseline = symmetric_masked_multi_positive_infonce(
        descriptors,
        descriptors,
        mask,
        groups,
        items,
        temperature=0.10,
        source_group_ids=torch.tensor([[100, 101, 102]]),
    )
    _, variants = symmetric_masked_multi_positive_infonce(
        descriptors,
        descriptors,
        mask,
        groups,
        items,
        temperature=0.10,
        # Different semantic labels but the same known source/sign group form
        # two ordered multi-positive pairs, never false negatives.
        source_group_ids=torch.tensor([[100, 100, 102]]),
    )
    assert baseline["negative_pair_count"].item() == 6
    assert variants["negative_pair_count"].item() == 4
    assert baseline["positive_pair_count"].item() == 3
    assert variants["positive_pair_count"].item() == 5


def test_motion_derangement_equivalent_sources_are_bce_positives_not_negatives():
    correct = _memory(batch=1, candidates=2)
    correct.source_group_ids.fill_(100)
    correct.motion_source_source_group_ids.fill_(100)
    corrupt, _permutation, informative = motion_only_shuffle_sentence_memory_batch(
        correct,
        query_ids=("q0",),
        epoch=1,
        seed=1234,
    )
    assert informative.item()
    descriptor = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    outputs = {
        "sentence_memory_association_logit": torch.zeros(1, 2),
        "sentence_memory_association_mask": torch.ones(1, 2, dtype=torch.bool),
        "sentence_memory_association_key_descriptor": descriptor,
        "sentence_memory_association_motion_descriptor": descriptor,
        "sentence_memory_association_threshold": torch.tensor(0.0),
    }
    _losses, diagnostics = sentence_memory_association_losses(
        correct_outputs=outputs,
        corrupt_outputs=outputs,
        correct_memory=correct,
        corrupt_memory=corrupt,
        motion_row_mask=torch.tensor([True]),
        full_row_mask=torch.tensor([False]),
        cfg=_paired_cfg(association=True),
    )
    assert diagnostics["bce"]["positive_count"].item() == 4
    assert diagnostics["bce"]["negative_count"].item() == 0


def _paired_cfg(*, association=False):
    modes = [
        "off",
        "on",
        "motion_shuffled_n0",
        "motion_shuffled_n1",
        "motion_shuffled_n2",
        "cross_query_motion",
        "full_replacement",
        "joint_tuple_permuted",
        "uniform_final_mass",
    ]
    modes.append("analytic_prior")
    if association:
        modes.append("association_disabled")
    return {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {
            "enabled": True,
            "candidate_value_mode": "centered_candidate_covariance_v1",
            "association_mode": (
                "absolute_text_motion_v1" if association else "none"
            ),
        },
        "sentence_memory_safety": {
            "paired_corruption": {"enabled": True},
            "v2_to_v3_text_only_parity": {
                "passed": True,
                "prediction_max_abs": 0.0,
                "duration_max_abs": 0.0,
                "tolerance": 1e-7,
            },
        },
        "objective": {
            "lambda_sentence_association_bce": 0.10,
            "lambda_sentence_association_infonce": 0.05,
            "association_temperature": 0.10,
        },
        "eval": {"sentence_memory_modes": modes},
        "selection": {"aggregation": "normalized_text_cluster_equal_v1"},
    }


def _passing_metrics(*, association=False):
    scores = {
        "text_only": 10.0,
        "sentence_memory": 9.8,
        "motion_shuffled_n0_sentence_memory": 9.9,
        "motion_shuffled_n1_sentence_memory": 9.91,
        "motion_shuffled_n2_sentence_memory": 9.89,
        "cross_query_motion_sentence_memory": 9.9,
        "full_replacement_sentence_memory": 9.9,
    }
    metrics = {}
    for namespace, score in scores.items():
        metrics[f"{namespace}/pred_loss_endpoint"] = score
        metrics[f"{namespace}/pred_loss_path_lhand"] = (
            1.0 if namespace in {"text_only", "sentence_memory"} else 1.1
        )
        metrics[f"{namespace}/pred_loss_path_rhand"] = (
            1.0 if namespace in {"text_only", "sentence_memory"} else 1.1
        )
    metrics["paired_sentence_memory/Rpair"] = 0.2
    for index in range(3):
        metrics[f"paired_sentence_memory/Rpair_n{index}"] = 0.15
    for name in (
        "joint_tuple_prediction_max_abs",
        "joint_tuple_duration_max_abs",
        "uniform_final_vs_off_prediction_max_abs",
        "uniform_final_vs_off_duration_max_abs",
        "broadcast_complete_vs_off_prediction_max_abs",
        "broadcast_complete_vs_off_duration_max_abs",
        "all_null_vs_off_prediction_max_abs",
        "all_null_vs_off_duration_max_abs",
        "all_null_gate_max_abs",
        "all_null_candidate_mass_max_abs",
        "all_null_one_minus_null_mass_max_abs",
    ):
        metrics[f"paired_sentence_memory/{name}"] = 0.0
    if association:
        metrics["sentence_memory/association_matching_top1"] = 0.5
        metrics["sentence_memory/association_matching_margin"] = 0.1
    return metrics


def test_centered_mode_order_objective_identity_and_selection_gates():
    stage_a = _paired_cfg(association=False)
    assert configured_sentence_memory_eval_modes(stage_a)[0:2] == ("off", "on")
    assert sentence_memory_objective_identity(stage_a)["schema_version"] == 2
    stage_a_changed = copy.deepcopy(stage_a)
    stage_a_changed["objective"]["lambda_sentence_association_bce"] = 999.0
    assert sentence_memory_objective_identity(stage_a_changed) == (
        sentence_memory_objective_identity(stage_a)
    )
    assert checkpoint_selection_diagnostics(
        _passing_metrics(), stage_a
    )[2] is True

    stage_b = _paired_cfg(association=True)
    identity = sentence_memory_objective_identity(stage_b)
    assert identity["schema_version"] == 3
    assert identity["weights"]["lambda_sentence_association_bce"] == 0.10
    assert identity["weights"]["lambda_sentence_association_infonce"] == 0.05
    assert checkpoint_selection_diagnostics(
        _passing_metrics(association=True), stage_b
    )[2] is True

    failed = _passing_metrics(association=True)
    failed["sentence_memory/association_matching_top1"] = 0.249
    assert checkpoint_selection_diagnostics(failed, stage_b)[2] is False

    invalid_utility = _passing_metrics()
    invalid_utility["text_only/pred_loss_endpoint"] = 9.7
    _, _, feasible, details = checkpoint_selection_diagnostics(
        invalid_utility, stage_a, return_details=True
    )
    diagnostic = details["dual_mode"]
    assert feasible is False
    assert diagnostic["identity_utility_denominator_valid"] is False
    assert diagnostic["identity_utility_fraction"] == 0.0
    assert math.isfinite(diagnostic["identity_utility_denominator"])


def test_ordered_replay_requires_trainer_injected_v2_parity():
    runtime_cfg = _paired_cfg()
    metrics = _passing_metrics()
    score, violation, feasible, _details = checkpoint_selection_diagnostics(
        metrics, runtime_cfg, return_details=True
    )
    row = {
        "epoch": 1,
        "validation_pending": 0.0,
        "selection_score": score,
        "selection_constraint_violation": violation,
        "selection_feasible": float(feasible),
        "early_stopping_improved": 1.0,
        "early_stopping_bad_validation_count": 0.0,
        "early_stopping_validation_count": 1.0,
        "early_stopping_requested": 0.0,
        **{f"val_{name}": value for name, value in metrics.items()},
    }
    replay = centered_decision._replay_selection([row], runtime_cfg)
    assert replay["best_feasible_row"] == row
    assert replay["details_by_epoch"][1]["dual_mode"]["v2_text_only_parity"] == {
        "prediction_max_abs": 0.0,
        "duration_max_abs": 0.0,
    }

    raw_cfg = copy.deepcopy(runtime_cfg)
    raw_cfg["sentence_memory_safety"].pop("v2_to_v3_text_only_parity")
    _score, raw_violation, raw_feasible, raw_details = checkpoint_selection_diagnostics(
        metrics, raw_cfg, return_details=True
    )
    assert raw_feasible is False
    assert raw_violation == violation + 2.0
    assert all(
        math.isnan(value)
        for value in raw_details["dual_mode"]["v2_text_only_parity"].values()
    )
    with pytest.raises(
        centered_decision.OrderedDecisionError, match="exact v2 parity proof"
    ):
        centered_decision._replay_selection([row], raw_cfg)


def test_stage_b_architecture_identity_seals_exact_association_descriptors():
    cfg = _paired_cfg(association=True)
    cfg["sentence_memory"].update(
        {
            "key_value_mode": "factorized_metadata_motion_v1",
            "temporal_prior_mode": "none",
            "relevance_gate_mode": "frozen_absolute_adjusted_score_v1",
            "relevance_slope": 2.0,
            "relevance_intercept": -0.5,
            "resolved_relevance_calibration_identity": {"digest": "c" * 64},
        }
    )
    identity = sentence_memory_architecture_identity(cfg)
    assert identity["schema_version"] == 3
    centered = identity["candidate_value"]
    assert centered["candidate_ids_enter_learned_scores"] is False
    assert centered["structural_token_distribution"]["q"] == (
        "q[s,p,k,u]=1/count_u(support[s,p,k,:]) on support; otherwise 0"
    )
    assert centered["rho"] == "rho[l,h,s,p]=sum_k(a[l,h,s,p,k])"
    assert "minimum stable item ID" in centered["reference"]
    assert centered["residual"] == (
        "r[l,h,s,p]=sum_{k in A[s,p]}((a[l,h,s,p,k]-"
        "rho[l,h,s,p]*u[s,p,k])*(m[l,h,s,p,k]-m_ref[l,h,s,p]))"
    )
    assert "N[s,p]<=1" in centered["zero_or_one_supported_candidate"]
    assert "same exact structural u" in centered["uniform_control"]
    assert "exact zero centered residual" in centered["broadcast_control"]
    evidence = centered["evidence_application"]
    assert evidence["final_mass"] == (
        "a[l,h,s,p,k]=b[l,h,s,p,k]*g_rel[k]*g_assoc[k]"
    )
    assert "after base null+real softmax" in evidence["placement"]
    assert evidence["rejection"] == "all rejected real mass is routed to null"

    formula = identity["association"]["descriptor_formula"]
    assert formula["eK"] == (
        "l2_normalize(P_K(candidate_mt5_mean_key)); P_K_is_bias_free"
    )
    assert formula["eV"] == (
        "l2_normalize(P_V(mask_normalized_mean_u(LN_0(vae_motion_mu[k,u])))); "
        "LN_0_is_non_affine; P_V_is_bias_free"
    )
    assert formula["motion_pool_mask"] == (
        "binary_sentence_motion_mask_after_candidate_mask"
    )
    assert formula["excluded_from_descriptors"] == [
        "target_text_slots",
        "query_duration",
        "retrieval_scores",
        "candidate_durations",
        "motion_tau",
        "part_validity_magnitude",
        "candidate_ids",
    ]

    tampered = copy.deepcopy(identity)
    tampered["candidate_value"]["rho"] = "tampered"
    tampered_payload = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered_digest = hashlib.sha256(
        json.dumps(
            tampered_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert tampered_digest != identity["digest"]
    with pytest.raises(RuntimeError, match="invalid sentence-memory architecture"):
        validate_sentence_memory_architecture_identity(
            {"sentence_memory_architecture_identity": tampered},
            cfg,
            source="tampered",
        )
    tampered["digest"] = tampered_digest
    with pytest.raises(RuntimeError, match="differs from the active config"):
        validate_sentence_memory_architecture_identity(
            {"sentence_memory_architecture_identity": tampered},
            cfg,
            source="tampered",
        )


def test_centered_public_evaluator_rejects_partial_replays():
    centered = _paired_cfg()
    validate_centered_public_evaluation_scope(centered, limit=0, max_batches=0)
    with pytest.raises(ValueError, match="complete sealed 347-row"):
        validate_centered_public_evaluation_scope(
            centered, limit=0, max_batches=1
        )
    with pytest.raises(ValueError, match="complete sealed 347-row"):
        validate_centered_public_evaluation_scope(
            centered, limit=1, max_batches=0
        )
    with pytest.raises(ValueError, match="exact configured control suite"):
        validate_centered_public_evaluation_scope(
            centered,
            limit=0,
            max_batches=0,
            sentence_memory_mode="on",
        )
    validate_centered_public_evaluation_scope({}, limit=1, max_batches=2)


def test_centered_evaluator_atomic_json_preserves_prior_on_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "evaluation.json"
    _write_json_atomic(target, {"version": 1})

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(
        "NIAF.continuous_trajectory_field.scripts."
        "evaluate_continuous_trajectory_field.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="injected"):
        _write_json_atomic(target, {"version": 2})
    assert target.read_text(encoding="utf-8") == '{\n  "version": 1\n}\n'
    assert not list(tmp_path.glob(".evaluation.json.*.tmp"))


def test_resolved_relevance_identity_seals_equivalent_parameterizations(
    tmp_path, monkeypatch
):
    artifact_dir = tmp_path / "calibration"
    artifact_dir.mkdir()
    (artifact_dir / "calibration.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "READY").write_text(
        json.dumps({"map_sha256": "1" * 64}), encoding="utf-8"
    )
    calibration = {
        "schema_name": "signtrajfield_sentence_memory_relevance_calibration",
        "schema_version": 1,
        "identity": "2" * 64,
        "feature": {
            "mode": "absolute_adjusted_score_v1",
            "duration_weight": 0.05,
        },
        "map": {"content_digest": "3" * 64},
        "holdout": {"auroc": 0.8, "probability_gap": 0.3},
        "coefficients": {
            "a": 2.0,
            "b": 0.25,
            "formula": "sigmoid(a*(adjusted_score-b))",
            "intercept": -0.5,
            "intercept_derivation": "float64(-a*b)",
            "slope": 2.0,
        },
    }
    monkeypatch.setattr(
        "NIAF.continuous_trajectory_field.relevance_calibration."
        "validate_relevance_calibration_artifact",
        lambda *_args, **_kwargs: calibration,
    )

    cfg = _relevance_cfg(artifact_dir)
    resolved = resolve_sentence_memory_relevance_calibration(cfg)
    assert resolved["coefficients"] == calibration["coefficients"]
    assert cfg["sentence_memory"]["relevance_slope"] == 2.0
    assert cfg["sentence_memory"]["relevance_intercept"] == -0.5

    alternate = copy.deepcopy(calibration)
    alternate["coefficients"].update({"b": 0.5, "intercept": -1.0})
    monkeypatch.setattr(
        "NIAF.continuous_trajectory_field.relevance_calibration."
        "validate_relevance_calibration_artifact",
        lambda *_args, **_kwargs: alternate,
    )
    alternate_resolved = resolve_sentence_memory_relevance_calibration(
        _relevance_cfg(artifact_dir)
    )
    assert alternate_resolved["digest"] != resolved["digest"]

    for field, value in (
        ("a", 2.1),
        ("intercept", -0.499),
        ("formula", "sigmoid(intercept+slope*adjusted_score)"),
        ("intercept_derivation", "approximate"),
    ):
        tampered = copy.deepcopy(calibration)
        tampered["coefficients"][field] = value
        monkeypatch.setattr(
            "NIAF.continuous_trajectory_field.relevance_calibration."
            "validate_relevance_calibration_artifact",
            lambda *_args, _tampered=tampered, **_kwargs: _tampered,
        )
        with pytest.raises(RuntimeError, match="not exactly equivalent"):
            resolve_sentence_memory_relevance_calibration(
                _relevance_cfg(artifact_dir)
            )
