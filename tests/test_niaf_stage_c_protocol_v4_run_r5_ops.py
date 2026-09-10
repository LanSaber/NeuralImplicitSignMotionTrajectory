"""Regression coverage for the r5 standalone-clone evidence-root binding."""

import json
import shutil
from pathlib import Path

import pytest

from NIAF.continuous_trajectory_field import relevance_calibration
from NIAF.continuous_trajectory_field.scripts import stage_c_calibration_control
from NIAF.continuous_trajectory_field.scripts import stage_c_execution_control
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_decision
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_prerequisites


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "NIAF" / "continuous_trajectory_field" / "configs"
POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_"
    "decision_policy_v1.json"
)
RECOVERY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_"
    "recovery_evidence_v1.json"
)


def test_protocol_v4_generic_recovery_dispatch_is_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "validate_protocol_v4_run_r5_recovery_evidence",
        lambda **kwargs: calls.append(kwargs) or {"generation": "run_r5"},
    )
    common = {
        "recovery_manifest": tmp_path / "recovery.json",
        "source_root": tmp_path / "source",
        "evidence_root": tmp_path / "evidence",
    }

    assert stage_c_pilot_prerequisites.validate_recovery_evidence(
        policy_path=tmp_path / POLICY_NAME, **common
    ) == {"generation": "run_r5"}
    assert calls == [{"policy_path": tmp_path / POLICY_NAME, **common}]


def test_protocol_v4_standalone_clone_uses_manifest_canonical_evidence_root(
    tmp_path: Path,
) -> None:
    """A standalone source clone must not replace the pinned evidence root."""

    standalone_root = tmp_path / "standalone_source"
    standalone_configs = standalone_root / "NIAF" / "continuous_trajectory_field" / "configs"
    standalone_configs.mkdir(parents=True)
    for name in (POLICY_NAME, RECOVERY_NAME):
        shutil.copy2(CONFIGS / name, standalone_configs / name)
    policy = standalone_configs / POLICY_NAME
    recovery = standalone_configs / RECOVERY_NAME
    canonical_evidence_root = Path(
        json.loads(recovery.read_text(encoding="utf-8"))["canonical_evidence_root"]
    )

    assert standalone_root != canonical_evidence_root
    audit = stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_runtime_bindings(
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=standalone_root,
        evidence_root=canonical_evidence_root,
    )

    assert audit["historical_clone_metadata_checked_in_cpu_gate"] is False
    assert audit["incident_archive_sha256"] == (
        "a23682b584bcc05ccee67ccec7526d33efc924104a0ea34209740bb4131e5f71"
    )


def test_protocol_v4_policy_and_configs_are_exact_and_cpu_runtime_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CONFIGS / POLICY_NAME
    recovery = CONFIGS / RECOVERY_NAME
    memory = CONFIGS / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_memory_protocol_v4_run_r5.yaml"
    )
    matched_off = CONFIGS / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_matched_off_protocol_v4_run_r5.yaml"
    )
    canonical_evidence_root = Path(
        json.loads(recovery.read_text(encoding="utf-8"))["canonical_evidence_root"]
    )
    contract = json.loads(policy.read_text(encoding="utf-8"))["recovery_contract"]
    assert contract["protocol_generation"] == "protocol_v4"
    assert contract["run_generation"] == "run_r5"
    assert contract["incident_archive"]["sha256"] == (
        "a23682b584bcc05ccee67ccec7526d33efc924104a0ea34209740bb4131e5f71"
    )
    with pytest.raises(stage_c_pilot_prerequisites.PrerequisiteError):
        stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_runtime_bindings(
            policy_path=policy,
            recovery_manifest=recovery,
            source_root=ROOT,
            evidence_root=ROOT / "wrong-evidence-root",
        )
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_validate_standalone_source_clone",
        lambda **_kwargs: pytest.fail("runtime must not scan the archived r4 clone"),
    )
    runtime = stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_runtime_bindings(
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=ROOT,
        evidence_root=canonical_evidence_root,
    )
    assert runtime["historical_clone_metadata_checked_in_cpu_gate"] is False
    observed: list[dict] = []
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_validate_standalone_source_clone",
        lambda **kwargs: observed.append(kwargs),
    )
    config_audit = stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_configs(
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=ROOT,
        evidence_root=canonical_evidence_root,
        memory_config=memory,
        matched_off_config=matched_off,
    )
    assert observed and observed[0]["local_metadata_timeout_seconds"] == 600
    assert config_audit["scientific_settings_equal_protocol_v3_run_r4"] is True


def test_protocol_v4_cpu_ready_validation_skips_archived_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = CONFIGS / POLICY_NAME
    recovery = CONFIGS / RECOVERY_NAME
    memory = CONFIGS / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_memory_protocol_v4_run_r5.yaml"
    )
    matched_off = CONFIGS / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_matched_off_protocol_v4_run_r5.yaml"
    )
    evidence_root = Path(
        json.loads(recovery.read_text(encoding="utf-8"))["canonical_evidence_root"]
    )
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_validate_standalone_source_clone",
        lambda **_kwargs: pytest.fail("CPU READY validation must be runtime-only"),
    )
    recovery_audit = stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_runtime_bindings(
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=ROOT,
        evidence_root=evidence_root,
    )
    config_audit = stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_runtime_configs(
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=ROOT,
        evidence_root=evidence_root,
        memory_config=memory,
        matched_off_config=matched_off,
    )
    head = "1" * 40
    ready = {
        "schema_name": "signtrajfield_stage_c_cpu_gate",
        "schema_version": 2,
        "source_git_head": head,
        "source_remote_ref": "origin/codex/r5",
        "source_remote_head": head,
        "slurm_job_id": "1",
        "compileall": True,
        "ruff_version": "0.12.0",
        "pytest_version": "8.4.2",
        "complete_repository_test_glob": "tests/test_*.py",
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "passed": True,
        "recovery_audit_identity": recovery_audit["audit_identity"],
        "config_audit_identity": config_audit["audit_identity"],
        "incident_archive_sha256": recovery_audit["incident_archive_sha256"],
        "decision_policy_sha256": recovery_audit["decision_policy_sha256"],
        "recovery_manifest_sha256": recovery_audit["recovery_manifest_sha256"],
        "memory_config_sha256": config_audit["memory_config_sha256"],
        "matched_off_config_sha256": config_audit["matched_off_config_sha256"],
        "historical_clone_metadata_checked_in_cpu_gate": True,
        "historical_clone_local_git_metadata_timeout_seconds": 600,
        "historical_remote_ref_timeout_seconds": 120,
    }
    ready_path = tmp_path / "READY"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    assert stage_c_pilot_prerequisites.validate_protocol_v4_run_r5_cpu_gate(
        cpu_gate=ready_path,
        source_git_head=head,
        source_remote_ref="origin/codex/r5",
        source_remote_head=head,
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=ROOT,
        evidence_root=evidence_root,
        memory_config=memory,
        matched_off_config=matched_off,
    ) == ready


def test_protocol_v4_scripts_have_no_stale_execution_generation_markers() -> None:
    scripts = sorted((ROOT / "scripts" / "NIAF").glob("*protocol_v4_run_r5*"))
    assert len(scripts) == 6
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "codex/csl-daily-centered-generator-stage-c-v4-run-r5" in text or (
            script.name.startswith(("smoke_", "train_"))
        )
        text = "\n".join(
            line for line in text.splitlines() if "INCIDENT_ARCHIVE=" not in line
        )
        assert "protocol-v3/run-r4" not in text
        assert "Protocol v3/run-r4" not in text
        assert "run-r4 control" not in text


def test_protocol_v4_controls_and_source_binding_are_registered(tmp_path: Path) -> None:
    profile = "stage_c_generator_adaptation_protocol_v4_run_r5"
    expected_files = {
        "NIAF/continuous_trajectory_field/relevance_calibration.py",
        "NIAF/continuous_trajectory_field/scripts/calibrate_sentence_memory_relevance.py",
        "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py",
        "scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_sbatch.sh",
        "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh",
    }
    assert relevance_calibration.SOURCE_FILE_PROFILES[profile] == expected_files
    assert profile in stage_c_calibration_control.SOURCE_FILE_PROFILES
    assert POLICY_NAME in stage_c_execution_control.SMOKE_PREREQUISITE_POLICY_NAMES
    policy = stage_c_pilot_decision.validate_policy(CONFIGS / POLICY_NAME)
    assert policy["recovery_contract"]["run_generation"] == "run_r5"

    policy_path = tmp_path / POLICY_NAME
    policy_sha = "a" * 64
    marker_names = {
        POLICY_NAME,
        RECOVERY_NAME,
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v2_run_r3.yaml",
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v2_run_r3.yaml",
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v4_run_r5.yaml",
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v4_run_r5.yaml",
        "run_csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5.sh",
        "calibrate_csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_sbatch.sh",
        "ARCHIVE.json",
    }
    bound = {
        str((tmp_path / name).resolve()): (policy_sha if name == POLICY_NAME else "b" * 64)
        for name in marker_names
    }
    bound.update(
        {
            str((tmp_path / f"r5_source_{index}.py").resolve()): "c" * 64
            for index in range(21)
        }
    )
    assert len(bound) == 30
    assert stage_c_pilot_decision._validate_source_binding_generation(
        bound_files=bound,
        decision_policy={"path": str(policy_path.resolve()), "sha256": policy_sha},
        mode="smoke",
    ) == 30
    missing = dict(bound)
    missing.pop(
        str(
            (
                tmp_path
                / "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_memory_protocol_v2_run_r3.yaml"
            ).resolve()
        )
    )
    missing[str((tmp_path / "replacement.py").resolve())] = "d" * 64
    with pytest.raises(stage_c_pilot_decision.StageCDecisionError):
        stage_c_pilot_decision._validate_source_binding_generation(
            bound_files=missing,
            decision_policy={"path": str(policy_path.resolve()), "sha256": policy_sha},
            mode="smoke",
        )
