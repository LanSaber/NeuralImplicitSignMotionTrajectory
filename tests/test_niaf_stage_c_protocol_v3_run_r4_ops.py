import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_decision
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_prerequisites


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "NIAF/continuous_trajectory_field/configs"
SCRIPTS = ROOT / "scripts/NIAF"
POLICY = CONFIGS / (
    "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
    "decision_policy_v1.json"
)
RECOVERY = CONFIGS / (
    "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
    "recovery_evidence_v1.json"
)
V3 = {
    arm: CONFIGS
    / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        f"adaptation_{arm}_protocol_v3_run_r4.yaml"
    )
    for arm in ("memory", "matched_off")
}
R3 = {
    arm: CONFIGS
    / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        f"adaptation_{arm}_protocol_v2_run_r3.yaml"
    )
    for arm in ("memory", "matched_off")
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("experiment_name")
    value["output"].pop("out_dir")
    value["sentence_memory"]["relevance_calibration"].pop("artifact_dir")
    value["sentence_memory_safety"]["stage_c"]["active_stage_c"].pop(
        "calibration_artifact_dir"
    )
    return value


def test_protocol_v3_policy_preserves_r3_science_and_binds_zero_science_archive():
    v3 = stage_c_pilot_decision.validate_policy(POLICY)
    r3 = stage_c_pilot_decision.validate_policy(
        CONFIGS
        / "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
        "decision_policy_v1.json"
    )
    for field in (
        "authorization",
        "execution",
        "invariant_metric_keys",
        "metric_keys",
        "pilot_thresholds",
        "retry_policy",
        "source_checkpoint",
        "status_values",
    ):
        assert v3[field] == r3[field]
    recovery = v3["recovery_contract"]
    assert recovery["prior_scientific_output_observed"] is False
    assert recovery["zero_science_archive_required"] is True
    assert recovery["incident_archive"]["sha256"] == (
        "66368adf6a2310732a4a2bcca87625d45c8dd51c0c4912b16ceba9b4c12b8dc2"
    )
    assert "validation_preserving_execution_safety" in recovery[
        "downstream_proof_delegation"
    ]


def test_protocol_v3_configs_are_scientifically_identical_to_r3_with_fresh_roots():
    for arm in ("memory", "matched_off"):
        v3 = load_config(V3[arm])
        r3 = load_config(R3[arm])
        assert _normalized(v3) == _normalized(r3)
        assert v3["output"]["out_dir"].endswith(f"{arm}_protocol_v3_run_r4")
        assert (
            v3["sentence_memory"]["relevance_calibration"]["artifact_dir"].endswith(
                "protocol_v3_run_r4"
            )
        )
    assert _sha256(R3["memory"]) == (
        "77e61c12a68f5f5bd60cc23355ca935fa5bf0a1ed777a9da440203417e3750cc"
    )
    assert _sha256(R3["matched_off"]) == (
        "c36f5617d20d3ddc7386f8ec8d4811f58417fe716d86a5dc249c38a71258f880"
    )


def test_protocol_v3_manifest_binds_exact_zero_science_absences():
    manifest = json.loads(RECOVERY.read_text(encoding="utf-8"))
    archive = manifest["incident_archive"]
    assert archive == {
        "path": stage_c_pilot_prerequisites.PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_PATH,
        "sha256": stage_c_pilot_prerequisites.PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_SHA256,
        "bytes": 11_442,
        "schema_name": "signtrajfield_stage_c_generator_adaptation_terminal_incident_archive",
        "schema_version": 3,
        "required_absent_path_count": 13,
    }
    absent = manifest["zero_science_runtime_absences"]
    assert len(absent["arm_roots"]) == 2
    assert len(absent["prerequisite_calibration_smoke_pilot_roots"]) == 4
    assert len(absent["six_runtime_logs"]) == 6
    assert absent["confirmation_holdout_spend_marker"].endswith(
        "confirmation_holdout_spent.json"
    )


def test_protocol_v3_archived_clone_timeout_is_split_and_runtime_skips_it(
    tmp_path, monkeypatch
):
    clone = tmp_path / "archived"
    (clone / ".git").mkdir(parents=True)
    observed = []

    def fake_run(command, **kwargs):
        observed.append((tuple(command[-4:]), kwargs["timeout"]))
        arguments = tuple(command)
        if arguments[-1] == "--show-toplevel":
            stdout = str(clone)
        elif arguments[-1] == "HEAD":
            stdout = "a" * 40
        elif "status" in arguments:
            stdout = ""
        else:
            stdout = "a" * 40 + " refs/heads/demo\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(stage_c_pilot_prerequisites.subprocess, "run", fake_run)
    stage_c_pilot_prerequisites._validate_standalone_source_clone(
        clone=clone,
        expected_head="a" * 40,
        expected_remote_ref="origin/demo",
        local_metadata_timeout_seconds=600,
        remote_ref_timeout_seconds=120,
    )
    assert [timeout for _, timeout in observed] == [600, 600, 600, 120]

    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_validate_standalone_source_clone",
        lambda **_kwargs: pytest.fail("runtime must not scan an archived clone"),
    )
    runtime_audit = stage_c_pilot_prerequisites.validate_protocol_v3_run_r4_runtime_bindings(
        policy_path=POLICY,
        recovery_manifest=RECOVERY,
        source_root=ROOT,
        evidence_root=ROOT,
    )
    assert runtime_audit["historical_clone_metadata_checked_in_cpu_gate"] is False
    assert runtime_audit["audit_identity"]


@pytest.mark.parametrize(
    "name",
    [
        "launch_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh",
        "test_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh",
        "calibrate_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh",
        "smoke_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh",
        "train_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh",
        "run_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh",
    ],
)
def test_protocol_v3_shell_surfaces_are_bounded_and_parse(name):
    path = SCRIPTS / name
    text = path.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", path], check=True)
    assert "protocol_v3_run_r4" in text
    assert "git ls-remote" not in text or "timeout 120s git ls-remote" in text
    assert "scancel" not in text
    if name.startswith(("calibrate_", "run_", "launch_")):
        assert "validate-recovery" not in text
        assert "validate-protocol-v3-runtime" in text or "validate-protocol-v3-cpu" in text


def test_protocol_v3_cpu_gate_is_schema_v2_and_only_cpu_runs_historical_proof():
    cpu = (
        SCRIPTS
        / "test_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh"
    ).read_text(encoding="utf-8")
    runner = (
        SCRIPTS / "run_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh"
    ).read_text(encoding="utf-8")
    assert '"schema_version": 2' in cpu
    assert "validate-protocol-v3-configs" in cpu
    assert "historical_clone_local_git_metadata_timeout_seconds" in cpu
    assert "validate-protocol-v3-runtime" in runner
    assert "validate-recovery" not in runner
