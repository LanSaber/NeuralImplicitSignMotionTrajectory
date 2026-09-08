import importlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from NIAF.continuous_sign_field.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "NIAF/continuous_trajectory_field/configs"
SCRIPT_DIR = ROOT / "scripts/NIAF"
STAGE_A = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "relevance_motion_contrast_v1"
)
STAGE_B = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "absolute_binding_motion_contrast_v1"
)
GLOBAL_SPEND = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/"
    "confirmation_holdout_spent.json"
)
CALIBRATION_RETRY = "csl_daily_sentence_memory_relevance_calibration_v1_retry1"
STAGE_A_MODES = [
    "off",
    "on",
    "motion_shuffled_n0",
    "motion_shuffled_n1",
    "motion_shuffled_n2",
    "cross_query_motion",
    "full_replacement",
    "joint_tuple_permuted",
    "uniform_final_mass",
    "analytic_prior",
]


def _source(name):
    return (SCRIPT_DIR / name).read_text(encoding="utf-8")


def test_centered_full_configs_are_ordered_and_stage_specific():
    stage_a = load_config(CONFIG_DIR / f"{STAGE_A}.yaml")
    stage_b = load_config(CONFIG_DIR / f"{STAGE_B}.yaml")
    for name, cfg in ((STAGE_A, stage_a), (STAGE_B, stage_b)):
        assert cfg["experiment_name"] == name
        assert cfg["train"]["epochs"] == 6
        assert cfg["train"]["early_stopping_min_epochs"] == 3
        assert cfg["train"]["early_stopping_patience"] == 2
        assert cfg["sentence_memory"]["key_value_mode"] == (
            "factorized_metadata_motion_v1"
        )
        assert cfg["sentence_memory"]["candidate_value_mode"] == (
            "centered_candidate_covariance_v1"
        )
        assert cfg["sentence_memory"]["relevance_gate_mode"] == (
            "frozen_absolute_adjusted_score_v1"
        )
        assert cfg["eval"]["export_sentence_memory_mode"] == "on"
        assert cfg["sentence_memory"]["temporal_prior_mode"] == "none"
        calibration = cfg["sentence_memory"]["relevance_calibration"]
        assert calibration["schema_version"] == 1
        assert calibration["minimum_heldout_auroc"] == 0.75
        assert calibration["minimum_heldout_probability_gap"] == 0.20
        assert cfg["eval"]["evaluation_corruption"] == {
            "mode": "fixed_evidence_controls_v1",
            "seed": 1234,
            "nonce": None,
        }
    association_fields = {
        "lambda_sentence_association_bce",
        "lambda_sentence_association_infonce",
        "association_temperature",
    }
    assert stage_a["sentence_memory"]["association_mode"] == "none"
    assert not association_fields.intersection(stage_a["objective"])
    assert stage_a["eval"]["sentence_memory_modes"] == STAGE_A_MODES
    assert stage_b["sentence_memory"]["association_mode"] == ("absolute_text_motion_v1")
    assert {name: stage_b["objective"][name] for name in association_fields} == {
        "lambda_sentence_association_bce": 0.10,
        "lambda_sentence_association_infonce": 0.05,
        "association_temperature": 0.10,
    }
    assert stage_b["eval"]["sentence_memory_modes"] == [
        *STAGE_A_MODES,
        "association_disabled",
    ]


def test_smoke_configs_are_one_epoch_nonselecting_leafs():
    for name in (STAGE_A, STAGE_B):
        cfg = load_config(CONFIG_DIR / f"{name}_smoke.yaml")
        assert cfg["experiment_name"] == f"{name}_smoke"
        assert cfg["train"]["epochs"] == 1
        assert cfg["train"]["early_stopping_min_epochs"] == 1
        assert cfg["train"]["early_stopping_patience"] == 0
        assert cfg["selection"]["require_feasible"] is False


def test_strict_stager_never_enumerates_shared_neighbor_directory():
    source = _source("stage_sentence_memory_train_val_only_node.sh")
    assert '"$SOURCE"/*' not in source
    assert "glob(" not in source
    assert "strict Phase-A''' staging permits only train and val" in source
    assert 'neighbor="neighbors_${split}.npz"' in source
    assert '"$SOURCE/$neighbor"' in source
    assert '[[ ! -e "$TARGET/neighbors_test.npz" ]]' in source
    assert 'cp -a -- "$SOURCE/READY" "$TARGET/READY"' in source


def test_calibration_launcher_is_train_only_attested_and_durable():
    source = _source("calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh")
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --mem=32G" in source
    assert "/logs/sbatch/%x_%j.out" in source
    assert '"$STAGER" stage' in source and '"train"' in source
    assert '[[ ! -e "$STAGED_BANK/neighbors_test.npz" ]]' in source
    assert "--purpose calibration --stage calibration" in source
    assert "release-execution-lease" in source
    assert "existing rejected/incomplete calibration is terminal" in source
    assert "record-calibration-completion" in source
    assert "--publication_existing" in source
    # A publish-before-release crash is recovered only after a new attested
    # acquisition/stale-owner audit; accepted reuse cannot bypass the lease.
    assert source.index("acquire-execution-lease") < source.index(
        'if [[ "$EXISTING_ACCEPTED" == "1" ]]'
    )
    assert source.index("record-calibration-completion") < source.index(
        'if [[ "$EXISTING_ACCEPTED" == "1" ]]'
    )
    assert "--source_remote_ref" in source and "--source_remote_head" in source
    assert CALIBRATION_RETRY in source
    assert "calibration forbids a noncanonical output path" in source
    assert "calibration forbids a noncanonical control path" in source


def test_recovery_calibration_path_is_consistent_and_source_bound():
    stage_a = load_config(CONFIG_DIR / f"{STAGE_A}.yaml")
    stage_b = load_config(CONFIG_DIR / f"{STAGE_B}.yaml")
    for cfg in (stage_a, stage_b):
        artifact = cfg["sentence_memory"]["relevance_calibration"]["artifact_dir"]
        assert Path(artifact).name == CALIBRATION_RETRY
    for script in (
        "calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh",
        "smoke_csl_daily_centered_memory_v1_sbatch.sh",
        "run_csl_daily_centered_memory_stage.sh",
        "decide_csl_daily_centered_memory_v1_sbatch.sh",
    ):
        assert CALIBRATION_RETRY in _source(script)
    helper = (
        ROOT
        / "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py"
    ).read_text(encoding="utf-8")
    assert helper.count(CALIBRATION_RETRY) == 2


def test_full_driver_enforces_order_fresh_v2_and_single_holdout_lock():
    source = _source("run_csl_daily_centered_memory_stage.sh")
    assert GLOBAL_SPEND in source
    assert (
        'EXPECTED_V2_SHA256="06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"'
        in source
    )
    assert "--purpose stage2 --stage stage1" in source
    assert "record-stage2-input" in source
    assert '--v2_checkpoint "$V2_CHECKPOINT"' in source
    assert "EPOCHS=6" in source
    assert '"train val"' in source
    assert "neighbors_test.npz" in source
    assert "STAGE_SENTENCE_MEMORY=0" in source
    assert "WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true" in source
    assert source.index("--purpose stage2 --stage stage1") < source.index(
        'bash "$STAGER" stage "$LOCAL_BANK"'
    )
    assert "PRECHECKPOINT_RETRY=1" in source
    assert "record-precheckpoint-retry" in source
    assert "precheckpoint_retry_attempts/" in source
    assert "record-run-resume" in source
    assert '--partition_dir "$PARTITION_DIR"' in source
    assert "terminal no-op finalization is not attestation-bound" in source
    assert "centered stages forbid a noncanonical OUT_DIR" in source
    assert "centered stages forbid a noncanonical Stage-A authorization" in source


def test_centered_helper_ports_full_recovery_and_is_import_isolated():
    source = (
        ROOT
        / "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py"
    ).read_text(encoding="utf-8")
    assert "def _centered_resume_progress_evidence(" in source
    assert "validation_pending" in source
    assert "_recover_truncated_terminal_metric" in source
    assert "_atomic_append_reconciled_metric" in source
    assert "missing_complete_row" in source
    assert "terminal_complete" in source
    assert "_atomic_write_terminal_selection_summary" in source
    assert "def record_precheckpoint_retry(" in source
    assert "def decide_stage(" in source
    assert "def _centered_legacy_profile():" not in source
    assert "_install_legacy_process_profile" not in source
    assert re.search(r"legacy\.[A-Za-z_]+\s*=", source) is None
    assert "legacy.decide_stage(" not in source
    assert "legacy.record_precheckpoint_retry(" not in source
    assert "legacy.spend_confirmation(" not in source


def test_decision_requires_standalone_replay_then_diagnostic_authorization():
    helper = (
        ROOT
        / "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py"
    ).read_text(encoding="utf-8")
    launcher = _source("decide_csl_daily_centered_memory_v1_sbatch.sh")
    assert "--integrity_audit" in launcher
    assert "--sentence_memory auto" in launcher
    assert "SIGNTRAJ_SOURCE_GIT_HEAD" in launcher
    assert '"train val"' in launcher and "neighbors_test.npz" in launcher
    assert "_EXACT_ZERO_AUDITS" in helper
    assert "joint_tuple_duration_max_abs" in helper
    assert "authorize_diagnostic" not in helper  # derived from purpose, not a bypass
    assert 'purpose = "diagnostic"' in helper
    assert "finalize-confirmation-authorization" in helper
    assert "authorize_confirmation.json" in helper
    assert "signtrajfield_centered_locked_development_diagnostic_ready" in helper
    assert "centered decisions forbid a noncanonical RUN_DIR" in launcher
    assert "centered decisions forbid a noncanonical Stage-A authorization" in launcher


def test_postselection_is_bound_to_the_single_canonical_run_tree():
    source = _source("run_csl_daily_centered_memory_postselection.sh")
    assert "centered post-selection forbids a noncanonical RUN_DIR" in source
    assert 'RUN_DIR="$CANONICAL_RUN_DIR"' in source


def test_smoke_audits_full_modes_frozen_base_optimizer_and_corruptions():
    source = _source("smoke_csl_daily_centered_memory_v1_sbatch.sh")
    assert "expected_modes" in source
    assert "broadcast_complete_vs_off_prediction_max_abs" in source
    assert "torch.equal(state[name], v2_state[name])" in source
    assert "configure_sentence_memory_trainable_parameters" in source
    assert "optimizer.load_state_dict" in source
    assert "Stage-B association gradient was zero" in source
    assert "motion_fraction" in source and "full_fraction" in source
    assert "validate_relevance_calibration_source" in source


def test_full_and_smoke_slurm_resources_are_explicit():
    for script in (
        "train_csl_daily_centered_relevance_stage_a_v1_sbatch.sh",
        "train_csl_daily_centered_absolute_binding_stage_b_v1_sbatch.sh",
    ):
        source = _source(script)
        assert "#SBATCH --nodes=4" in source
        assert "#SBATCH --cpus-per-task=16" in source
        assert "#SBATCH --gres=gpu:1" in source
        assert "#SBATCH --mem=100G" in source
        assert "#SBATCH --time=06:00:00" in source
        assert "/logs/sbatch/%x_%j.out" in source
    smoke = _source("smoke_csl_daily_centered_memory_v1_sbatch.sh")
    assert "#SBATCH --nodes=1" in smoke
    assert "#SBATCH --cpus-per-task=8" in smoke
    assert "#SBATCH --gres=gpu:1" in smoke
    assert "#SBATCH --mem=64G" in smoke
    assert "MAX_TRAIN_BATCHES=2" in smoke
    assert 'global_step", -1)) != 1' in smoke
    assert smoke.count("run_arm csl_daily_signtrajfield_v3") == 2
    assert smoke.count("_smoke_retry1") == 3
    assert "invalid smoke attempt 143300" in (
        CONFIG_DIR
        / f"{STAGE_A}_smoke_retry1.yaml"
    ).read_text(encoding="utf-8")
    cpu = _source("test_csl_daily_centered_memory_v1_sbatch.sh")
    assert "pytest==8.4.2" in cpu and "ruff==0.12.0" in cpu
    assert "/media/cvpr/haomian/python_envs/slt/bin/uv" in cpu
    assert "/home/cvpr" not in cpu
    assert '== "uv 0.10.0"' in cpu
    assert 'UV_CACHE_DIR="/tmp/signtraj_centered_cpu_uv_' in cpu
    assert '--python "$PYTHON_BIN" --no-project' in cpu
    assert 'CENTERED_LINT_BASE="96fc62aa12e1c6b690a564dd7f97bbd9670006d0"' in cpu
    assert 'git merge-base --is-ancestor "$CENTERED_LINT_BASE" "$SOURCE_GIT_HEAD"' in cpu
    assert 'git diff --name-only --diff-filter=ACM "$CENTERED_LINT_BASE"' in cpu
    assert 'tool run --from ruff==0.12.0 ruff check "${CENTERED_RUFF_FILES[@]}"' in cpu
    assert "invalid attempt 143293" in cpu
    assert '"${TEST_PYTHON[@]}" -m ruff' not in cpu
    assert '"${TEST_PYTHON[@]}" -m pytest -q tests/test_*.py' in cpu


def test_new_helper_is_separate_and_legacy_protocol_constants_remain():
    centered = (
        ROOT
        / "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py"
    ).read_text(encoding="utf-8")
    legacy = (
        ROOT
        / "NIAF/continuous_trajectory_field/scripts/decide_factorized_memory_stage.py"
    ).read_text(encoding="utf-8")
    assert "MAXIMUM_EPOCHS = 6" in centered
    assert "MINIMUM_EPOCHS = 3" in centered
    assert "factorized_ordered_v1_control/" in centered
    assert '"confirmation_holdout_spent.json"' in centered
    assert 'int(train.get("epochs", -1)) != 4' in legacy
    assert "range(1, 5)" in legacy
    assert "centered_relevance_motion_contrast" not in legacy


def test_runbook_gates_cpu_before_calibration_smoke_and_stage_a():
    runbook = (
        ROOT / "docs/NIAF/continuous_trajectory_field/phase_a_centered_relevance_v1.md"
    ).read_text(encoding="utf-8")
    cpu = "test_csl_daily_centered_memory_v1_sbatch.sh"
    calibration = "calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh"
    smoke = "smoke_csl_daily_centered_memory_v1_sbatch.sh"
    stage_a = "train_csl_daily_centered_relevance_stage_a_v1_sbatch.sh"
    assert "GIT_CONFIG_KEY_0=safe.directory" in runbook
    assert 'GIT_CONFIG_VALUE_0="$PROJECT_DIR"' in runbook
    assert runbook.index(cpu) < runbook.index(calibration)
    assert runbook.index(calibration) < runbook.index(smoke)
    assert runbook.index(smoke) < runbook.index(stage_a)


def _load_ordered_isolated():
    """Load the ops helper without importing the GPU model package."""

    prefix = "NIAF.continuous_trajectory_field"
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "torch" or name == prefix or name.startswith(f"{prefix}.")
    }
    for name in tuple(saved):
        sys.modules.pop(name, None)
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: (lambda function: function)
    sys.modules["torch"] = fake_torch
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT / "NIAF/continuous_trajectory_field")]
    scripts_name = f"{prefix}.scripts"
    scripts = types.ModuleType(scripts_name)
    scripts.__path__ = [str(ROOT / "NIAF/continuous_trajectory_field/scripts")]
    sys.modules[prefix] = package
    sys.modules[scripts_name] = scripts

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        load(
            f"{prefix}.relevance_calibration",
            "NIAF/continuous_trajectory_field/relevance_calibration.py",
        )
        legacy = load(
            f"{scripts_name}.decide_factorized_memory_stage",
            "NIAF/continuous_trajectory_field/scripts/decide_factorized_memory_stage.py",
        )
        setattr(scripts, "decide_factorized_memory_stage", legacy)
        names = (
            "SCHEMA_NAME",
            "SCHEMA_VERSION",
            "AUTHORIZATION_SCHEMA_NAME",
            "EXPERIMENTS",
            "verify_authorization",
            "decide_stage",
        )
        before = {name: getattr(legacy, name) for name in names}
        ordered = load(
            f"{scripts_name}.decide_centered_memory_stage",
            "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py",
        )
        after = {name: getattr(legacy, name) for name in names}
        return ordered, before, after
    finally:
        for name in tuple(sys.modules):
            if name == "torch" or name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def _ordered_helper():
    return _load_ordered_isolated()[0]


def test_importing_centered_helper_never_mutates_legacy_module():
    _ordered, before, after = _load_ordered_isolated()
    assert after == before


def test_local_decision_mints_only_diagnostic_for_feasible_stage(tmp_path):
    ordered = _ordered_helper()
    launch_path = tmp_path / "run.prerequisites/run_launch_identity.json"
    launch_path.parent.mkdir()
    launch_path.write_text("{}\n", encoding="utf-8")
    config = tmp_path / f"{ordered.EXPERIMENTS[ordered.STAGE1]}.yaml"
    config.write_text("x\n", encoding="utf-8")
    v2_config = tmp_path / "v2.yaml"
    v2_config.write_text("x\n", encoding="utf-8")
    v2 = tmp_path / "v2.pt"
    v2.write_bytes(b"v2")
    summaries = []
    for name in ("off", "null", "v2"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        summaries.append(path)
    partition = {
        "dir": tmp_path / "partition",
        "development_manifest": tmp_path / "manifest.jsonl",
        "development_manifest_sha256": "d" * 64,
    }
    checkpoint_evidence = {
        "path": str(tmp_path / "best.pt"),
        "has_feasible_checkpoint": True,
        "resume_attempts": [],
        "selected_checkpoint_integrity_audit": {"passed": True},
    }
    launch = {
        "launch_identity": "l" * 64,
        "source": {"git_head": "a" * 40},
        "slurm": {"nodes": 4},
    }
    exports = iter(
        {
            "summary_path": path,
            "rows": [],
            "durations": np.asarray([], dtype=np.float64),
        }
        for path in summaries
    )
    written = {}

    def fake_sha(path):
        return ordered.legacy.EXPECTED_V2_SHA256 if Path(path) == v2 else "f" * 64

    with (
        patch.object(ordered.legacy, "_validate_partition", return_value=partition),
        patch.object(
            ordered,
            "_validate_run",
            return_value=({}, checkpoint_evidence, {}),
        ),
        patch.object(ordered, "_validate_run_launch", return_value=launch),
        patch.object(
            ordered.legacy,
            "_validate_shared_source_checkout",
            return_value=launch["source"],
        ),
        patch.object(
            ordered, "_load_dev_export", side_effect=lambda *a, **k: next(exports)
        ),
        patch.object(
            ordered.legacy,
            "_validate_development_all_null",
            return_value={"passed": True},
        ),
        patch.object(
            ordered.legacy,
            "_validate_selected_v2_parity",
            return_value={"passed": True},
        ),
        patch.object(ordered, "sha256_file", side_effect=fake_sha),
        patch.object(
            ordered,
            "_write_centered_decision",
            side_effect=lambda out, decision, authorization: written.update(
                decision=decision, authorization=authorization
            ),
        ),
    ):
        result = ordered._decide_stage_validated(
            stage=ordered.STAGE1,
            run_dir=tmp_path / "run",
            config_path=config,
            partition_dir=tmp_path / "partition",
            memory_off_dir=tmp_path / "off",
            all_null_dir=tmp_path / "null",
            v2_memory_off_dir=tmp_path / "v2_off",
            v2_config_path=v2_config,
            v2_checkpoint_path=v2,
            out_dir=tmp_path / "decision",
        )
    assert result["status"] == ordered.legacy.DEVELOPMENT_FEASIBLE
    assert written["authorization"]["purpose"] == "diagnostic"
    assert written["authorization"]["checkpoint"] == checkpoint_evidence


def test_resume_progress_and_precheckpoint_recovery_are_fail_closed(tmp_path):
    ordered = _ordered_helper()
    checkpoint = {
        "epoch": 1,
        "metrics": {"epoch": 1, "validation_pending": 1.0},
        "selection_state": {
            "early_stopping": {
                "validation_count": 0,
                "last_validation_epoch": None,
                "stopped": False,
                "stop_epoch": None,
            }
        },
    }
    checkpoint["selection_state"] = {
        "schema_version": 2,
        "best_feasible_score": None,
        "best_infeasible_score": None,
        "early_stopping": {
            "schema_version": 1,
            "best_key": None,
            "last_key": None,
            "bad_validation_count": 0,
            "validation_count": 0,
            "last_validation_epoch": None,
            "stopped": False,
            "stop_epoch": None,
        },
    }
    with patch.object(
        ordered.legacy,
        "_rng_resume_evidence",
        return_value={"world_size": 4, "rank_states": [{"rank": i} for i in range(4)]},
    ):
        evidence = ordered._centered_resume_progress_evidence(checkpoint, [], {})
    assert evidence["mode"] == "validation_pending"
    assert evidence["completed_validation_epochs"] == []

    run = tmp_path / ordered.EXPERIMENTS[ordered.STAGE1]
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    temporary = checkpoints / ".last.pt.retry_01.tmp"
    temporary.write_bytes(b"partial")
    safe = ordered._precheckpoint_output_evidence(run, stage=ordered.STAGE1)
    assert safe["published_checkpoints_absent"] is True
    assert safe["unpublished_checkpoint_temporaries"][0]["sha256"]
    (run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ordered.OrderedDecisionError, match="material progress"):
        ordered._precheckpoint_output_evidence(run, stage=ordered.STAGE1)


def _diagnostic_summary(ordered, query_binding):
    modes = (*ordered.EVAL_MODES[ordered.STAGE1], "broadcast_complete", "all_null")
    controls = {
        mode: {"passed": True}
        for mode in ordered.EVAL_MODES[ordered.STAGE1]
        if mode not in {"off", "on"}
    }
    controls["broadcast_complete"] = {"passed": True}
    exact = {
        "prediction_max_abs": 0.0,
        "duration_max_abs": 0.0,
        "prediction_arrays_equal": True,
        "duration_values_equal": True,
        "requires_bitwise_array_equality": True,
        "passed": True,
    }
    return {
        "stage": ordered.STAGE1,
        "experiment_name": ordered.EXPERIMENTS[ordered.STAGE1],
        "scientific_split": "val_development_novel_text_only",
        "partition_digest": ordered.legacy.EXPECTED_PARTITION_DIGEST,
        "development_manifest_sha256": (
            ordered.legacy.EXPECTED_DEVELOPMENT_MANIFEST_SHA256
        ),
        "development_rows": ordered.legacy.EXPECTED_DEVELOPMENT_ROWS,
        "development_text_clusters": ordered.legacy.EXPECTED_DEVELOPMENT_TEXTS,
        "bootstrap": {"samples": 10_000, "seed": 1234},
        "explanatory_only": True,
        "scientific_gate": False,
        "changes_checkpoint_selection": False,
        "query_bindings": {mode: query_binding for mode in modes},
        "input_export_summary_sha256": {mode: "e" * 64 for mode in modes},
        "control_provenance": controls,
        "exact_invariants": {
            "joint_tuple_vs_correct": {
                **exact,
                "requires_bitwise_array_equality": False,
            },
            "uniform_final_vs_off": dict(exact),
            "broadcast_complete_vs_off": dict(exact),
            "all_null_vs_off": dict(exact),
            "all_null": {
                "payload_reads": 0,
                "gate_max_abs": 0.0,
                "candidate_mass_max_abs": 0.0,
                "null_mass_max_abs_error": 0.0,
                "passed": True,
            },
        },
    }


def test_diagnostic_contract_and_finalizer_reject_incomplete_evidence(tmp_path):
    ordered = _ordered_helper()
    summary = _diagnostic_summary(ordered, {"bound": True})
    with patch.object(
        ordered.legacy, "validate_factorized_export_query_binding", return_value={}
    ):
        ordered._validate_diagnostic_summary_contract(summary, stage=ordered.STAGE1)
        broken = dict(summary)
        broken["development_rows"] = 346
        with pytest.raises(ordered.OrderedDecisionError, match="completeness"):
            ordered._validate_diagnostic_summary_contract(broken, stage=ordered.STAGE1)

    layer_row = {
        "mode": "on",
        "layer": 0,
        "part": "body",
        "metric": "effective_k",
        "mean": 2.0,
        "ci95_low": 1.5,
        "ci95_high": 2.5,
    }
    summary["layerwise_metrics_file"] = "layerwise_metrics.csv"
    summary["layerwise_rows_identity"] = ordered.legacy._digest_json([layer_row])
    layerwise = tmp_path / "layerwise_metrics.csv"
    layerwise.write_text(
        "mode,layer,part,metric,mean,ci95_low,ci95_high\n"
        "on,0,body,effective_k,2.0,1.5,2.5\n",
        encoding="utf-8",
    )
    evidence = ordered._validate_diagnostic_layerwise_csv(tmp_path, summary)
    assert evidence["rows"] == 1 and evidence["finite"] is True
    layerwise.write_text(
        "mode,layer,part,metric,mean,ci95_low,ci95_high\n"
        "on,0,body,effective_k,nan,1.5,2.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ordered.OrderedDecisionError, match="Non-finite"):
        ordered._validate_diagnostic_layerwise_csv(tmp_path, summary)

    decision_dir = tmp_path / "decision"
    decision_dir.mkdir()
    decision = {
        "status": ordered.legacy.DEVELOPMENT_FEASIBLE,
        "development_feasible": True,
        "integrity_valid": True,
        "decision_identity": "d" * 64,
        "checkpoint": {"path": "/locked/best.pt", "sha256": "c" * 64},
        "partition": {"partition_digest": ordered.legacy.EXPECTED_PARTITION_DIGEST},
        "run_launch_identity": {"source": {"git_head": "a" * 40}},
        "predecessor_authorization": None,
    }
    (decision_dir / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    diagnostic_auth = decision_dir / "authorize_diagnostic.json"
    diagnostic_auth.write_text("{}\n", encoding="utf-8")
    diagnostic_evidence = {"diagnostic_identity": "z" * 64}
    verified = {
        "authorization_identity": "a" * 64,
        "checkpoint": {
            **decision["checkpoint"],
            "relevance_calibration": {"artifact_identity": "r" * 64},
        },
    }
    out = decision_dir / "authorize_confirmation.json"
    with (
        patch.object(ordered, "verify_authorization", return_value=verified),
        patch.object(
            ordered,
            "_validate_diagnostic_ready",
            return_value=diagnostic_evidence,
        ),
        patch.object(ordered, "SOURCE_ROOT", tmp_path),
    ):
        value = ordered.finalize_confirmation_authorization(
            stage=ordered.STAGE1,
            diagnostic_authorization_path=diagnostic_auth,
            diagnostic_dir=tmp_path / "diagnostic",
            out_file=out,
        )
    assert value["purpose"] == "confirmation"
    assert value["development_diagnostic"] == diagnostic_evidence


def test_authorization_source_failure_happens_before_global_spend(tmp_path):
    ordered = _ordered_helper()
    decision_dir = tmp_path / "decision"
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    decision_dir.mkdir()
    files = {
        "checkpoint": checkpoints / "best.pt",
        "metrics": run / "metrics.jsonl",
        "summary": run / "selection_summary.json",
        "resolved": run / "config.resolved.json",
        "config": tmp_path / f"{ordered.EXPERIMENTS[ordered.STAGE1]}.yaml",
        "audit": tmp_path / "audit.json",
    }
    for path in files.values():
        path.write_bytes(b"x")
    calibration = {"artifact_identity": "r" * 64}
    exact_zero = {name: 0.0 for name in ordered._EXACT_ZERO_AUDITS}
    checkpoint = {
        "path": str(files["checkpoint"]),
        "sha256": ordered.sha256_file(files["checkpoint"]),
        "metrics_jsonl_sha256": ordered.sha256_file(files["metrics"]),
        "selection_summary_sha256": ordered.sha256_file(files["summary"]),
        "resolved_config_sha256": ordered.sha256_file(files["resolved"]),
        "config": {
            "path": str(files["config"]),
            "sha256": ordered.sha256_file(files["config"]),
        },
        "relevance_calibration": calibration,
        "selected_checkpoint_integrity_audit": {
            "path": str(files["audit"]),
            "sha256": ordered.sha256_file(files["audit"]),
            "exact_zero": exact_zero,
            "joint_tuple": {
                "joint_tuple_prediction_max_abs": 0.0,
                "joint_tuple_duration_max_abs": 0.0,
            },
            "v2_text_only_parity": {
                "prediction_max_abs": 0.0,
                "duration_max_abs": 0.0,
            },
            "test_data_accessed": False,
            "confirmation_manifest_opened": False,
        },
    }
    source = {
        "repository_root": str(tmp_path),
        "git_head": "a" * 40,
        "remote_ref": "origin/run",
        "remote_head": "a" * 40,
    }
    decision_without = {
        "schema_name": ordered.SCHEMA_NAME,
        "schema_version": ordered.SCHEMA_VERSION,
        "stage": ordered.STAGE1,
        "experiment_name": ordered.EXPERIMENTS[ordered.STAGE1],
        "status": ordered.legacy.DEVELOPMENT_FEASIBLE,
        "integrity_valid": True,
        "development_feasible": True,
        "checkpoint": checkpoint,
        "partition": {"partition_digest": ordered.legacy.EXPECTED_PARTITION_DIGEST},
        "run_launch_identity": {"source": source},
        "predecessor_authorization": None,
    }
    decision = {
        **decision_without,
        "decision_identity": ordered.legacy._digest_json(decision_without),
    }
    authorization_without = ordered._centered_authorization_payload(
        purpose="confirmation", stage=ordered.STAGE1, decision=decision
    )
    authorization = dict(authorization_without)
    authorization["development_diagnostic"] = {"directory": "/sealed"}
    authorization.pop("authorization_identity")
    authorization["authorization_identity"] = ordered.legacy._digest_json(authorization)
    auth = decision_dir / "authorize_confirmation.json"
    auth.write_text(json.dumps(authorization), encoding="utf-8")
    (decision_dir / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    (decision_dir / "READY").write_text(
        json.dumps(
            {
                "schema_name": ordered.SCHEMA_NAME,
                "schema_version": ordered.SCHEMA_VERSION,
                "decision_identity": decision["decision_identity"],
                "status": ordered.legacy.DEVELOPMENT_FEASIBLE,
                "authorized_purpose": "diagnostic",
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / ordered.GLOBAL_HOLDOUT_SPEND_RELATIVE
    with (
        patch.object(
            ordered,
            "validate_config",
            return_value={
                "sentence_memory": {
                    "resolved_relevance_calibration_identity": calibration
                }
            },
        ),
        patch.object(
            ordered.legacy,
            "_validate_shared_source_checkout",
            side_effect=ordered.OrderedDecisionError("wrong checkout"),
        ),
        patch.object(ordered, "SOURCE_ROOT", tmp_path),
    ):
        with pytest.raises(ordered.OrderedDecisionError, match="wrong checkout"):
            ordered.spend_confirmation(
                authorization_path=auth,
                stage=ordered.STAGE1,
                marker_path=marker,
                allow_matching_existing=False,
            )
    assert not marker.exists()


def test_calibration_accepted_reuse_can_take_over_proven_stale_lease(tmp_path):
    ordered = _ordered_helper()
    lease = tmp_path / "active_calibration_execution_lease"
    first_attestation = tmp_path / "attempts/100.json"
    second_attestation = tmp_path / "attempts/101.json"
    first = ordered.acquire_execution_lease(
        lease_path=lease,
        attestation_path=first_attestation,
        purpose="calibration",
        stage=ordered.CALIBRATION_STAGE,
        source_git_head="a" * 40,
        slurm_job_id="100",
        binding_identity="b" * 64,
    )
    with patch.object(
        ordered.legacy,
        "_slurm_job_terminal_evidence",
        return_value={"classification": "terminal", "state": "PREEMPTED"},
    ):
        second = ordered.acquire_execution_lease(
            lease_path=lease,
            attestation_path=second_attestation,
            purpose="calibration",
            stage=ordered.CALIBRATION_STAGE,
            source_git_head="a" * 40,
            slurm_job_id="101",
            binding_identity="b" * 64,
        )
    assert first["claim"]["claim_identity"] != second["claim"]["claim_identity"]
    assert (
        second["claim"]["replaced_stale_owner"]["claim_identity"]
        == first["claim"]["claim_identity"]
    )
    stale = lease.with_name(f"{lease.name}.history") / (
        f"100.{first['claim']['claim_identity']}.stale"
    )
    assert stale.is_dir()
    released = ordered.release_execution_lease(
        lease_path=lease, attestation_path=second_attestation
    )
    assert released["claim_identity"] == second["claim"]["claim_identity"]
    assert not lease.exists()
