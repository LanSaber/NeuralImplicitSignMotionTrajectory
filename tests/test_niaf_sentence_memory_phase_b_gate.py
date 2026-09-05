import json
from pathlib import Path

import pytest
import torch

from NIAF.continuous_trajectory_field.scripts.analyze_sentence_memory_phase_b_gate import (
    PhaseBGateInputError,
    analyze_phase_b_gate,
)


MODES = ("text_only", "sentence_memory", "shuffled_sentence_memory")
ALIGNMENTS = ("default", "pa")
PARTS = ("body", "lhand", "rhand", "wholebody")
METRICS = ("dtw", "ndtw", "ndtw_ref")


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _make_export(
    path,
    *,
    mode,
    checkpoint,
    duration_offset=0.0,
    drop_last=False,
    unavailable_subset=False,
    subsets=None,
    split="val",
    include_diagnostics=False,
):
    rows = []
    for index in range(4 - int(drop_last)):
        row = {
            "index": f"{index:04d}",
            "name": f"sample-{index}",
            "text": f"sentence {index}",
            "predicted_duration_seconds": 2.0 + index + duration_offset,
            "sentence_memory_text_subset": (
                "not_available"
                if unavailable_subset
                else (
                    subsets[index]
                    if subsets is not None
                    else ("novel_text" if index % 2 == 0 else "exact_seen_text")
                )
            ),
        }
        if include_diagnostics:
            gate_base = {
                "off": 0.0,
                "on": 0.2,
                "shuffled": 0.1,
            }[mode]
            row["sentence_memory_diagnostics"] = {
                "available": mode != "off",
                "gate_mean_by_part": {
                    part: gate_base + 0.01 * part_index
                    for part_index, part in enumerate(
                        ("body", "lhand", "rhand", "face")
                    )
                },
                "null_mass_mean": {
                    "off": 1.0,
                    "on": 0.3,
                    "shuffled": 0.6,
                }[mode],
                "candidate_mass_mean": {
                    "off": 0.0,
                    "on": 0.7,
                    "shuffled": 0.4,
                }[mode],
            }
        rows.append(row)
    return _write_json(
        path,
        {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": 3,
            "split": split,
            "length_mode": "predicted",
            "model_type": "sentence_memory_continuous_trajectory_field",
            "trajectory_contract_version": 3,
            "sentence_memory_mode": mode,
            "sentence_memory": {"bank_id": "synthetic-bank"},
            "num_exported": len(rows),
            "manifest": {
                "canonical_sample_count": len(rows),
                "is_complete_canonical_manifest": True,
            },
            "rows": rows,
        },
    )


def _make_dtw(
    path,
    *,
    mode,
    alignment,
    memory_offset=-2.0,
    memory_offsets_by_index=None,
    memory_hand_path_by_index=None,
    drop_last=False,
):
    mode_offset = {
        "text_only": 0.0,
        "sentence_memory": memory_offset,
        "shuffled_sentence_memory": -1.0,
    }[mode]
    rows = []
    for index in range(4 - int(drop_last)):
        item_offset = mode_offset
        if mode == "sentence_memory" and memory_offsets_by_index is not None:
            item_offset = float(memory_offsets_by_index.get(index, mode_offset))
        hand_path_error = {
            "text_only": 0.10,
            "sentence_memory": 0.10 * (1.01 if memory_offset < 0.0 else 1.03),
            "shuffled_sentence_memory": 0.12,
        }[mode]
        if mode == "sentence_memory" and memory_hand_path_by_index is not None:
            hand_path_error = float(
                memory_hand_path_by_index.get(index, hand_path_error)
            )
        for part_index, part in enumerate(PARTS):
            base = 10.0 + index + part_index
            score = base + item_offset
            rows.append(
                {
                    "index": f"{index:04d}",
                    "comparison": "flow",
                    "part": part,
                    "gt_len": 20,
                    "pred_len": 18,
                    "dtw": score * 20.0,
                    "path_len": 20,
                    "ndtw": score,
                    "ndtw_ref": score * 0.9,
                    **(
                        {
                            "motion_path_error": hand_path_error,
                            "motion_path_ratio": 1.0 + hand_path_error,
                            "jerk_magnitude_ratio": {
                                "text_only": 1.0,
                                "sentence_memory": 0.95,
                                "shuffled_sentence_memory": 1.1,
                            }[mode],
                        }
                        if part in {"lhand", "rhand"}
                        else {}
                    ),
                }
            )
    return _write_json(
        path,
        {
            "alignment_mode": alignment,
            "metric_preset": (
                "t2m_partwise_pa_same_subset"
                if alignment == "pa"
                else "mgpt_t2m_default_align_idx_0"
            ),
            "betas_mode": "h2s_fixed",
            "parts": list(PARTS),
            "num_pairs": 4 - int(drop_last),
            "rows": rows,
        },
    )


def _make_checkpoint(path, *, memory_hand_path=1.01, parity_error=0.0):
    torch.save(
        {
            "model_type": "sentence_memory_continuous_trajectory_field",
            "trajectory_contract_version": 3,
            "epoch": 3,
            "global_step": 12,
            "sentence_memory_identity": {"bank_id": "synthetic-bank"},
            "v2_to_v3_text_only_parity": {
                "prediction_max_abs": parity_error,
                "duration_max_abs": parity_error,
                "passed": parity_error <= 1e-7,
            },
            "config": {
                "sentence_memory_safety": {"phase_b": {"enabled": False}},
                "train": {
                    "freeze_base": True,
                    "unfreeze_base_prefixes": [],
                    "base_checkpoint": "source-v2.pt",
                },
            },
            "metrics": {
                "val_text_only/pred_loss_path_lhand": 1.0,
                "val_sentence_memory/pred_loss_path_lhand": memory_hand_path,
                "val_text_only/pred_loss_path_rhand": 2.0,
                "val_sentence_memory/pred_loss_path_rhand": 2.0 * memory_hand_path,
            },
        },
        path,
    )
    return path


def _evidence(
    tmp_path,
    *,
    duration_offset=0.0,
    memory_offset=-2.0,
    memory_hand_path=1.01,
    parity_error=0.0,
    mismatched_dtw=False,
    subsets=None,
    memory_offsets_by_index=None,
    memory_hand_path_by_index=None,
    split="val",
    include_diagnostics=False,
):
    checkpoint = _make_checkpoint(
        tmp_path / "phase_a.pt",
        memory_hand_path=memory_hand_path,
        parity_error=parity_error,
    )
    export_paths = {
        "text_only": _make_export(
            tmp_path / "text_export.json",
            mode="off",
            checkpoint=checkpoint,
            unavailable_subset=True,
            split=split,
            include_diagnostics=include_diagnostics,
        ),
        "sentence_memory": _make_export(
            tmp_path / "memory_export.json",
            mode="on",
            checkpoint=checkpoint,
            duration_offset=duration_offset,
            subsets=subsets,
            split=split,
            include_diagnostics=include_diagnostics,
        ),
        "shuffled_sentence_memory": _make_export(
            tmp_path / "shuffled_export.json",
            mode="shuffled",
            checkpoint=checkpoint,
            subsets=subsets,
            split=split,
            include_diagnostics=include_diagnostics,
        ),
    }
    dtw_paths = {}
    for mode in MODES:
        dtw_paths[mode] = {}
        for alignment in ALIGNMENTS:
            dtw_paths[mode][alignment] = _make_dtw(
                tmp_path / f"{mode}_{alignment}.json",
                mode=mode,
                alignment=alignment,
                memory_offset=memory_offset,
                memory_offsets_by_index=memory_offsets_by_index,
                memory_hand_path_by_index=memory_hand_path_by_index,
                drop_last=(
                    mismatched_dtw and mode == "sentence_memory" and alignment == "pa"
                ),
            )
    return checkpoint, export_paths, dtw_paths


def _analyze(checkpoint, export_paths, dtw_paths, *, purpose="phase_b_gate"):
    return analyze_phase_b_gate(
        export_paths=export_paths,
        dtw_paths=dtw_paths,
        phase_a_checkpoint=checkpoint,
        purpose=purpose,
        bootstrap_samples=500,
        bootstrap_seed=77,
    )


def test_phase_b_gate_accepts_paired_improvement_and_is_deterministic(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(tmp_path)

    first = _analyze(checkpoint, export_paths, dtw_paths)
    second = _analyze(checkpoint, export_paths, dtw_paths)

    assert first["accepted"] is True
    assert first["subset_counts"] == {
        "overall": 4,
        "novel_text": 2,
        "exact_seen_text": 2,
    }
    assert first["gate_identity"] == second["gate_identity"]
    assert first["settings"]["primary_subset"] == "novel_text"
    assert first["decision"]["primary_subset"] == "novel_text"
    assert (
        first["checks"]["novel_text_pa_wholebody_memory_vs_text_ci_below_zero"]["upper"]
        < 0.0
    )
    assert first["checks"]["duration_invariance"]["passed"] is True
    assert first["checks"]["checkpoint_overall_hand_path_diagnostic"]["passed"] is True
    assert (
        first["checks"]["checkpoint_overall_hand_path_diagnostic"][
            "required_for_acceptance"
        ]
        is False
    )
    assert (
        first["checks"]["novel_text_predicted_hand_path_nonregression"]["passed"]
        is True
    )
    assert first["checks"]["stored_v2_parity"]["passed"] is True
    assert first["checks"]["export_gate_null_diagnostics_available"]["passed"] is False
    assert first["sentence_memory_diagnostics"]["warning"] is not None
    for subset in ("overall", "novel_text", "exact_seen_text"):
        for alignment in ALIGNMENTS:
            for part in PARTS:
                assert set(first["metrics"][subset][alignment][part]) == set(METRICS)


def test_gate_aggregates_optional_export_gate_and_null_diagnostics(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        include_diagnostics=True,
    )

    result = _analyze(checkpoint, export_paths, dtw_paths)

    assert result["accepted"] is True
    assert result["checks"]["export_gate_null_diagnostics_available"]["passed"] is True
    diagnostics = result["sentence_memory_diagnostics"]
    assert diagnostics["warning"] is None
    assert diagnostics["modes"]["sentence_memory"]["novel_text"]["gate_mean_by_part"][
        "lhand"
    ]["mean"] == pytest.approx(0.21)
    assert diagnostics["modes"]["sentence_memory"]["overall"]["null_mass_mean"][
        "mean"
    ] == pytest.approx(0.3)


def test_test_split_is_report_only_and_never_authorizes_phase_b(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        split="test",
        include_diagnostics=True,
    )

    report = _analyze(
        checkpoint,
        export_paths,
        dtw_paths,
        purpose="test_report",
    )

    assert report["scientific_criteria_passed"] is True
    assert report["accepted"] is False
    assert report["decision"]["phase_b_authorizing"] is False
    assert "report-only" in report["decision"]["reason"]
    assert report["provenance"]["split"] == "test"
    assert report["subset_counts"] == {
        "overall": 4,
        "novel_text": 2,
        "exact_seen_text": 2,
    }

    with pytest.raises(PhaseBGateInputError, match="requires split='val'"):
        _analyze(checkpoint, export_paths, dtw_paths, purpose="phase_b_gate")


def test_test_report_rejects_validation_evidence(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(tmp_path)

    with pytest.raises(PhaseBGateInputError, match="requires split='test'"):
        _analyze(checkpoint, export_paths, dtw_paths, purpose="test_report")


def test_phase_b_gate_rejects_metric_duration_hand_and_parity_failures(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        duration_offset=1e-3,
        memory_offset=1.0,
        memory_hand_path=1.03,
        parity_error=1e-4,
    )

    result = _analyze(checkpoint, export_paths, dtw_paths)

    assert result["accepted"] is False
    assert (
        result["checks"]["novel_text_pa_wholebody_memory_better_than_text"]["passed"]
        is False
    )
    assert (
        result["checks"]["novel_text_pa_wholebody_memory_vs_text_ci_below_zero"][
            "passed"
        ]
        is False
    )
    assert (
        result["checks"]["novel_text_pa_wholebody_memory_better_than_shuffled"][
            "passed"
        ]
        is False
    )
    assert result["checks"]["duration_invariance"]["passed"] is False
    assert (
        result["checks"]["checkpoint_overall_hand_path_diagnostic"]["passed"] is False
    )
    assert (
        result["checks"]["novel_text_predicted_hand_path_nonregression"]["passed"]
        is False
    )
    assert result["checks"]["stored_v2_parity"]["passed"] is False


def test_phase_b_gate_rejects_novel_regression_despite_overall_improvement(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        # Novel rows 0/2 regress by one; exact-seen rows 1/3 improve by five.
        # The overall mean therefore improves even though the primary subset fails.
        memory_offsets_by_index={0: 1.0, 1: -5.0, 2: 1.0, 3: -5.0},
    )

    result = _analyze(checkpoint, export_paths, dtw_paths)

    overall = result["metrics"]["overall"]["pa"]["wholebody"]["ndtw"]
    novel = result["metrics"]["novel_text"]["pa"]["wholebody"]["ndtw"]
    assert overall["paired_differences"]["memory_minus_text"]["mean_difference"] < 0.0
    assert novel["paired_differences"]["memory_minus_text"]["mean_difference"] > 0.0
    assert result["accepted"] is False
    assert (
        result["checks"]["novel_text_pa_wholebody_memory_better_than_text"]["passed"]
        is False
    )


def test_phase_b_gate_accepts_novel_gain_despite_overall_metric_regression(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        # Novel rows improve by three and exact-seen rows regress by four.
        memory_offsets_by_index={0: -3.0, 1: 4.0, 2: -3.0, 3: 4.0},
    )

    result = _analyze(checkpoint, export_paths, dtw_paths)

    overall = result["metrics"]["overall"]["pa"]["wholebody"]["ndtw"]
    novel = result["metrics"]["novel_text"]["pa"]["wholebody"]["ndtw"]
    assert overall["paired_differences"]["memory_minus_text"]["mean_difference"] > 0.0
    assert novel["paired_differences"]["memory_minus_text"]["mean_difference"] < 0.0
    assert result["accepted"] is True


def test_phase_b_gate_requires_enough_novel_pairs(tmp_path):
    subsets = [
        "novel_text",
        "exact_seen_text",
        "exact_seen_text",
        "exact_seen_text",
    ]
    checkpoint, export_paths, dtw_paths = _evidence(tmp_path, subsets=subsets)

    result = _analyze(checkpoint, export_paths, dtw_paths)

    check = result["checks"]["minimum_novel_text_paired_samples"]
    assert check["subset"] == "novel_text"
    assert check["count"] == 1
    assert check["minimum"] == 2
    assert check["passed"] is False
    assert result["accepted"] is False


def test_phase_b_gate_uses_novel_hand_path_and_keeps_checkpoint_overall_diagnostic(
    tmp_path,
):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        memory_hand_path=1.03,
        # Novel rows regress by 3%; exact rows are much better, so the overall
        # predicted-duration hand-path aggregate would pass the 2% bound.
        memory_hand_path_by_index={0: 0.103, 1: 0.05, 2: 0.103, 3: 0.05},
    )

    result = _analyze(checkpoint, export_paths, dtw_paths)

    overall_hand = result["metrics"]["overall"]["unwarped"]["lhand"][
        "motion_path_error"
    ]
    assert overall_hand["modes"]["sentence_memory"]["mean"] < 0.102
    assert (
        result["checks"]["novel_text_predicted_hand_path_nonregression"]["passed"]
        is False
    )
    checkpoint_check = result["checks"]["checkpoint_overall_hand_path_diagnostic"]
    assert checkpoint_check["passed"] is False
    assert checkpoint_check["required_for_acceptance"] is False
    assert result["accepted"] is False


def test_checkpoint_overall_hand_path_diagnostic_does_not_override_novel_gate(
    tmp_path,
):
    checkpoint, export_paths, dtw_paths = _evidence(
        tmp_path,
        memory_hand_path=1.03,
    )

    result = _analyze(checkpoint, export_paths, dtw_paths)

    assert (
        result["checks"]["checkpoint_overall_hand_path_diagnostic"]["passed"] is False
    )
    assert (
        result["checks"]["novel_text_predicted_hand_path_nonregression"]["passed"]
        is True
    )
    assert result["accepted"] is True


def test_phase_b_gate_rejects_unpaired_sample_sets(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(tmp_path, mismatched_dtw=True)

    with pytest.raises(PhaseBGateInputError, match="pair exactly"):
        _analyze(checkpoint, export_paths, dtw_paths)


def test_phase_b_gate_rejects_partial_split_exports(tmp_path):
    checkpoint, export_paths, dtw_paths = _evidence(tmp_path)
    path = export_paths["sentence_memory"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest"]["is_complete_canonical_manifest"] = False
    _write_json(path, payload)

    with pytest.raises(PhaseBGateInputError, match="complete configured val manifest"):
        _analyze(checkpoint, export_paths, dtw_paths)


def test_phase_b_gate_can_report_core_gate_without_checkpoint(tmp_path):
    _checkpoint, export_paths, dtw_paths = _evidence(tmp_path)

    result = analyze_phase_b_gate(
        export_paths=export_paths,
        dtw_paths=dtw_paths,
        phase_a_checkpoint=None,
        bootstrap_samples=100,
    )

    assert result["accepted"] is True
    assert result["provenance"]["checkpoint"] == {"supplied": False}
    assert (
        result["checks"]["checkpoint_overall_hand_path_diagnostic"]["evaluated"]
        is False
    )
    assert result["checks"]["stored_v2_parity"]["passed"] is None
