import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field import relevance_calibration
from NIAF.continuous_trajectory_field.scripts import stage_c_calibration_control
from NIAF.continuous_trajectory_field.scripts import stage_c_execution_control
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_decision
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_prerequisites


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "NIAF/continuous_trajectory_field/configs"
SCRIPT_DIR = ROOT / "scripts/NIAF"
PROTOCOL = "protocol_v2_run_r3"
MEMORY_CONFIG = CONFIG_DIR / (
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
    f"memory_{PROTOCOL}.yaml"
)
OFF_CONFIG = CONFIG_DIR / (
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
    f"matched_off_{PROTOCOL}.yaml"
)
R2_MEMORY_CONFIG = CONFIG_DIR / (
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
    "memory_pilot_run_r2.yaml"
)
R2_OFF_CONFIG = CONFIG_DIR / (
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
    "matched_off_pilot_run_r2.yaml"
)
POLICY = CONFIG_DIR / (
    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
    "decision_policy_v1.json"
)
RECOVERY = CONFIG_DIR / (
    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
    "recovery_evidence_v1.json"
)
CALIBRATION_ROOT = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_sentence_memory_relevance_calibration_stage_c_"
    "generator_adaptation_protocol_v2_run_r3"
)
SOURCE_CHECKPOINT = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_"
    "binding_motion_contrast_v1/checkpoints/best_infeasible.pt"
)
SOURCE_SHA256 = (
    "b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202"
)


R2_IMMUTABLE_HASHES = {
    R2_MEMORY_CONFIG: "62a6b955594580377638ee1d7a2f47ae4e62ae1ba7258f1cbedd474cf998612d",
    R2_OFF_CONFIG: "dc3722d838c45cb4480552f1db224b9c63f1fdc04cc363219f4d01f625f3b7a4",
    CONFIG_DIR
    / "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json": (
        "8eacff17c161127573ad784a77542a089ebc7eca468b7f6bc4ac9263c6a66edc"
    ),
    CONFIG_DIR
    / "csl_daily_stage_c_generator_adaptation_retry2_recovery_evidence_v1.json": (
        "ccf47b72390c4be28775b207c67f7372d3cbc599fcddb3545df8295acc88b39b"
    ),
    SCRIPT_DIR / "launch_csl_daily_stage_c_generator_adaptation_pilot.sh": (
        "adf78e1d0bff0a62ed735a4ae9500820f9645ff55239abdfd3f578ed4efd0fcf"
    ),
    SCRIPT_DIR / "test_csl_daily_stage_c_generator_adaptation_sbatch.sh": (
        "8cbcd0506f5f007209636cd399be26aec6aedce9800662b0dea72533cde33a9d"
    ),
    SCRIPT_DIR / "calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh": (
        "4092fa600f01a505a74b2faee0946625385aba5af900797a24c110c804d24b89"
    ),
    SCRIPT_DIR
    / "smoke_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh": (
        "313b69643b5d9bb0a1cfbff087d93935ff80ff9687d6e6aa71129cac44c52bf5"
    ),
    SCRIPT_DIR
    / "train_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh": (
        "0ad594485b365c25826ea22b818499029aaf938ee95651ea78e4051dfb320bf9"
    ),
    SCRIPT_DIR / "run_csl_daily_stage_c_generator_adaptation_pilot.sh": (
        "eed608347646dbfb835228d4c7d87f7b7067c9b10c028d65a53576ea98d5f250"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_normalized(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("experiment_name")
    value["output"].pop("out_dir")
    value["sentence_memory"]["relevance_calibration"].pop("artifact_dir")
    value["sentence_memory_safety"]["stage_c"]["active_stage_c"].pop(
        "calibration_artifact_dir"
    )
    return value


def _arm_normalized(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("experiment_name")
    value["output"].pop("out_dir")
    value["conditioning"].pop("sentence_memory_train_mode")
    value["conditioning"].pop("sentence_memory_dropout_probability")
    value["sentence_memory_safety"]["stage_c"].pop("arm")
    return value


def _protocol_v2_smoke_binding() -> tuple[dict, dict, dict]:
    head = "a1" * 20
    pair = "pair02"
    policy_sha = _sha256(POLICY)
    smoke_root = ROOT / (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_protocol_v2_run_r3_smoke"
    )
    ready = ROOT / (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
        f"prerequisites/source_{head}/smoke/PUBLICATION/READY"
    )
    binding = {
        "schema_name": (
            "signtrajfield_stage_c_prior_one_update_smoke_prerequisite"
        ),
        "schema_version": 1,
        "smoke_root_path": str(smoke_root.resolve()),
        "ready_path": str(ready.resolve()),
        "ready_sha256": "1" * 64,
        "decision_policy_sha256": policy_sha,
        "source_git_head": head,
        "source_remote_ref": (
            "origin/codex/csl-daily-centered-generator-stage-c-v2-run-r3"
        ),
        "source_remote_head": head,
        "pair_constraint": pair,
        "memory_config_sha256": "2" * 64,
        "matched_off_config_sha256": "3" * 64,
        "cpu_gate_sha256": "4" * 64,
        "calibration_completion_sha256": "5" * 64,
    }
    complete = {
        "execution_mode": "pilot",
        "source_git_head": head,
        "source_remote_ref": binding["source_remote_ref"],
        "source_remote_head": head,
        "pair_constraint": pair,
        "decision_policy": {
            "path": str(POLICY.resolve()),
            "sha256": policy_sha,
        },
        "prior_one_update_smoke": binding,
    }
    launch = {
        "prior_one_update_smoke": binding,
        "artifacts": {
            "memory_config": {"sha256": binding["memory_config_sha256"]},
            "matched_off_config": {
                "sha256": binding["matched_off_config_sha256"]
            },
            "cpu_gate": {"sha256": binding["cpu_gate_sha256"]},
            "calibration_completion": {
                "sha256": binding["calibration_completion_sha256"]
            },
        },
    }
    return binding, complete, launch


def test_protocol_v2_preserves_every_r2_policy_config_and_script_byte():
    assert {path: _sha256(path) for path in R2_IMMUTABLE_HASHES} == (
        R2_IMMUTABLE_HASHES
    )


def test_protocol_v2_configs_are_exact_scientific_clones_with_fresh_paths():
    memory = load_config(MEMORY_CONFIG)
    off = load_config(OFF_CONFIG)
    r2_memory = load_config(R2_MEMORY_CONFIG)
    r2_off = load_config(R2_OFF_CONFIG)

    assert _generation_normalized(memory) == _generation_normalized(r2_memory)
    assert _generation_normalized(off) == _generation_normalized(r2_off)
    assert _arm_normalized(memory) == _arm_normalized(off)

    for config, arm, mode, dropout in (
        (memory, "memory", "dropout", 0.25),
        (off, "matched_off", "off", 1.0),
    ):
        stage = config["sentence_memory_safety"]["stage_c"]
        assert config["experiment_name"].endswith(f"{arm}_{PROTOCOL}")
        assert config["output"]["out_dir"].endswith(f"{arm}_{PROTOCOL}")
        assert stage["arm"] == arm
        assert config["conditioning"]["sentence_memory_train_mode"] == mode
        assert config["conditioning"]["sentence_memory_dropout_probability"] == (
            dropout
        )
        assert stage["source_checkpoint"]["path"] == SOURCE_CHECKPOINT
        assert stage["source_checkpoint"]["sha256"] == SOURCE_SHA256
        assert stage["source_checkpoint"]["selection_status"] == "best_infeasible"
        assert "run_r2" not in stage["source_checkpoint"]["path"]
        assert "last.pt" not in stage["source_checkpoint"]["path"]
        assert (
            config["sentence_memory"]["relevance_calibration"]["artifact_dir"]
            == CALIBRATION_ROOT
        )
        assert (
            stage["active_stage_c"]["calibration_artifact_dir"]
            == CALIBRATION_ROOT
        )
        assert config["sentence_memory"]["relevance_calibration"][
            "artifact_identity"
        ] is None


def test_protocol_v2_policy_is_exact_non_authorizing_and_no_retry():
    policy = stage_c_pilot_decision.validate_policy(POLICY)
    recovery = policy["recovery_contract"]
    assert recovery["prior_scientific_output_observed"] is True
    assert recovery["prior_memory_optimizer_updates_observed"] == 1
    assert recovery["matched_off_arm_started"] is False
    assert recovery["same_protocol_retry_authorized"] is False
    assert recovery["resume_authorized"] is False
    assert policy["retry_policy"] == {
        "malformed_completion": "stop_no_replace",
        "partial_scientific_output": "stop_no_retry",
        "unique_complete_execution": "reuse_only_for_publication_recovery",
        "zero_scientific_output": "stop_no_retry_within_protocol_generation",
    }
    assert policy["authorization"]["authorized_purpose"] is None
    assert policy["authorization"]["development_only"] is True
    assert policy["authorization"]["non_authorizing"] is True
    assert policy["authorization"]["promotion_eligible"] is False
    assert policy["execution"][
        "pilot_requires_prior_one_update_smoke_ready"
    ] is True
    assert policy["execution"][
        "smoke_requires_prior_one_update_smoke_ready"
    ] is False


def test_pilot_smoke_prerequisite_reopens_exact_binding_and_detects_race(
    monkeypatch,
):
    binding, complete, launch = _protocol_v2_smoke_binding()
    observed_calls = []

    def validate_smoke(**kwargs):
        observed_calls.append(kwargs)
        return copy.deepcopy(binding)

    monkeypatch.setattr(stage_c_pilot_prerequisites, "validate_smoke", validate_smoke)
    assert stage_c_pilot_decision._validate_pilot_smoke_prerequisite(
        value=binding,
        complete=complete,
        launch=launch,
    ) == binding
    assert observed_calls[-1]["source_git_head"] == binding["source_git_head"]
    assert observed_calls[-1]["pair_constraint"] == binding["pair_constraint"]

    raced = copy.deepcopy(binding)
    raced["ready_sha256"] = "f" * 64
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "validate_smoke",
        lambda **_kwargs: raced,
    )
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="differs from active science",
    ):
        stage_c_pilot_decision._validate_pilot_smoke_prerequisite(
            value=binding,
            complete=complete,
            launch=launch,
        )
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="lacks the exact one-update smoke prerequisite",
    ):
        stage_c_pilot_decision._validate_pilot_smoke_prerequisite(
            value=None,
            complete=complete,
            launch=launch,
        )


def test_protocol_v2_publication_requires_same_smoke_binding_everywhere(
    monkeypatch,
):
    binding, complete, launch = _protocol_v2_smoke_binding()
    records = {
        "completion": {"prior_one_update_smoke": binding},
        "ready": {"prior_one_update_smoke": binding},
        "execution_complete": complete,
        "decision": {"prior_one_update_smoke": binding},
        "launch": launch,
    }
    calls = []
    monkeypatch.setattr(
        stage_c_pilot_decision,
        "_validate_pilot_smoke_prerequisite",
        lambda **kwargs: calls.append(kwargs) or binding,
    )
    stage_c_execution_control._validate_protocol_v2_smoke_prerequisite(
        mode="pilot", **records
    )
    assert calls == [
        {"value": binding, "complete": complete, "launch": launch}
    ]

    removed = copy.deepcopy(records)
    removed["ready"].pop("prior_one_update_smoke")
    with pytest.raises(
        stage_c_execution_control.StageCExecutionControlError,
        match="differs across publication evidence",
    ):
        stage_c_execution_control._validate_protocol_v2_smoke_prerequisite(
            mode="pilot", **removed
        )

    smoke_records = {
        name: {} for name in records
    }
    stage_c_execution_control._validate_protocol_v2_smoke_prerequisite(
        mode="smoke", **smoke_records
    )
    smoke_records["launch"]["prior_one_update_smoke"] = binding
    with pytest.raises(
        stage_c_execution_control.StageCExecutionControlError,
        match="forbidden self-prerequisite",
    ):
        stage_c_execution_control._validate_protocol_v2_smoke_prerequisite(
            mode="smoke", **smoke_records
        )


def test_reused_pilot_completion_revalidates_bound_smoke_prerequisite(
    tmp_path, monkeypatch
):
    prior_smoke, complete, launch = _protocol_v2_smoke_binding()
    inputs = {}
    for name in ("memory_config", "matched_off_config", "cpu_gate", "calibration"):
        path = tmp_path / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        inputs[name] = path
    head = complete["source_git_head"]
    binding = stage_c_execution_control.execution_binding(
        mode="pilot",
        source_git_head=head,
        source_remote_ref=complete["source_remote_ref"],
        source_remote_head=head,
        pair_constraint=complete["pair_constraint"],
        memory_config=inputs["memory_config"],
        matched_off_config=inputs["matched_off_config"],
        cpu_gate=inputs["cpu_gate"],
        calibration_completion=inputs["calibration"],
    )
    launch["artifacts"] = {
        "memory_config": {"sha256": binding["memory_config_sha256"]},
        "matched_off_config": {
            "sha256": binding["matched_off_config_sha256"]
        },
        "cpu_gate": {"sha256": binding["cpu_gate_sha256"]},
        "calibration_completion": {
            "sha256": binding["calibration_completion_sha256"]
        },
    }
    execution_dir = tmp_path / "attempt"
    execution_dir.mkdir()
    execution_complete = execution_dir / "COMPLETE.json"
    execution_complete.write_text("{}\n", encoding="utf-8")
    (execution_dir / "LAUNCH.json").write_text(
        json.dumps(launch), encoding="utf-8"
    )
    lease = tmp_path / "control/active_execution_lease"
    attestation = tmp_path / "control/lease_attestations/55.0.json"
    stage_c_execution_control.acquire_execution_lease(
        lease_path=lease,
        attestation_path=attestation,
        slurm_job_id="55",
        slurm_restart_count=0,
        binding=binding,
    )
    claim = json.loads(attestation.read_text(encoding="utf-8"))["claim"]
    complete.update(
        {
            "active_lease_claim_identity": claim["claim_identity"],
            "prior_one_update_smoke": prior_smoke,
        }
    )
    calls = []
    monkeypatch.setattr(
        stage_c_pilot_decision,
        "_validate_pilot_smoke_prerequisite",
        lambda **kwargs: calls.append(kwargs) or prior_smoke,
    )
    stage_c_pilot_decision._validate_current_execution_scope(
        complete=complete,
        execution_complete=execution_complete,
        active_lease_attestation=attestation,
        expected_source_git_head=head,
        expected_source_remote_ref=complete["source_remote_ref"],
        expected_source_remote_head=head,
        expected_pair_constraint=complete["pair_constraint"],
    )
    assert calls[-1]["value"] == prior_smoke

    complete.pop("prior_one_update_smoke")
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="changed its one-update smoke prerequisite",
    ):
        stage_c_pilot_decision._validate_current_execution_scope(
            complete=complete,
            execution_complete=execution_complete,
            active_lease_attestation=attestation,
            expected_source_git_head=head,
            expected_source_remote_ref=complete["source_remote_ref"],
            expected_source_remote_head=head,
            expected_pair_constraint=complete["pair_constraint"],
        )


def test_protocol_v2_rejects_old_policy_filename_before_evidence_access(tmp_path):
    with pytest.raises(
        stage_c_pilot_prerequisites.PrerequisiteError,
        match="policy path changed",
    ):
        stage_c_pilot_prerequisites.validate_protocol_v2_run_r3_recovery_evidence(
            policy_path=(
                CONFIG_DIR
                / "csl_daily_stage_c_generator_adaptation_"
                "decision_policy_retry2_v1.json"
            ),
            recovery_manifest=tmp_path / "not-opened.json",
            source_root=ROOT,
            evidence_root=ROOT,
        )


def test_protocol_v2_inventory_reopen_rejects_tamper_and_absence(tmp_path):
    tree = tmp_path / "evidence/tree"
    tree.mkdir(parents=True)
    first = tree / "first.log"
    second = tree / "nested/second.json"
    second.parent.mkdir()
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    files = {
        path.relative_to(tree).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (first, second)
    }
    inventory = {
        "root": "evidence/tree",
        "file_count": 2,
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "tree_digest_algorithm": stage_c_pilot_prerequisites.TREE_DIGEST_ALGORITHM,
        "tree_digest": stage_c_pilot_prerequisites._tree_digest(files),
        "files": files,
    }
    fields = set(inventory)
    assert (
        stage_c_pilot_prerequisites._reopen_tree_inventory(
            evidence_root=tmp_path,
            inventory=inventory,
            label="protocol-v2 incident",
            expected_fields=fields,
        )
        == tree
    )

    first.write_bytes(b"tampered\n")
    with pytest.raises(
        stage_c_pilot_prerequisites.PrerequisiteError, match="file changed"
    ):
        stage_c_pilot_prerequisites._reopen_tree_inventory(
            evidence_root=tmp_path,
            inventory=inventory,
            label="protocol-v2 incident",
            expected_fields=fields,
        )
    first.write_bytes(b"first\n")
    second.unlink()
    with pytest.raises(
        stage_c_pilot_prerequisites.PrerequisiteError,
        match="regular non-symlink file",
    ):
        stage_c_pilot_prerequisites._reopen_tree_inventory(
            evidence_root=tmp_path,
            inventory=inventory,
            label="protocol-v2 incident",
            expected_fields=fields,
        )


def test_protocol_v2_calibration_profile_is_exact_and_old_profiles_remain():
    profile = "stage_c_generator_adaptation_protocol_v2_run_r3"
    expected = {
        "NIAF/continuous_trajectory_field/relevance_calibration.py",
        "NIAF/continuous_trajectory_field/scripts/"
        "calibrate_sentence_memory_relevance.py",
        "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py",
        "scripts/NIAF/"
        "calibrate_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
        "sbatch.sh",
        "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh",
    }
    assert relevance_calibration.SOURCE_FILE_PROFILES[profile] == expected
    assert profile in stage_c_calibration_control.SOURCE_FILE_PROFILES
    assert "stage_c_generator_adaptation_retry2_v1" in (
        stage_c_calibration_control.SOURCE_FILE_PROFILES
    )


def test_protocol_v2_runner_warm_starts_both_arms_only_from_original_stage_b():
    runner = (
        SCRIPT_DIR
        / "run_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh"
    ).read_text(encoding="utf-8")
    assert 'run_arm memory "$MEMORY_CFG"' in runner
    assert 'run_arm matched_off "$OFF_CFG"' in runner
    assert '--stage_c_warm_start "$STAGE_B_CHECKPOINT"' in runner
    assert "best_infeasible.pt" in runner
    assert "run_r2/checkpoints/last.pt" not in runner
    assert "pilot_run_r2/checkpoints/last.pt" not in runner
    assert "protocol-v2/run-r3 forbids retry/resume after partial execution" in (
        runner
    )
    assert "protocol-v2/run-r3 forbids replacing another logical job" in runner
    assert "EXPECTED_SOURCE_BINDING_FILE_COUNT=29" in runner
    assert 'SOURCE_BINDING_FILES+=("$SMOKE_READY")' in runner
    assert "EXPECTED_SOURCE_BINDING_FILE_COUNT=30" in runner
    assert runner.count(
        '--expected_pair_constraint "$PAIR_CONSTRAINT" --out_file "$DECISION"'
    ) == 1
    assert runner.count("revalidate_source_binding") >= 7
    assert runner.count("require_holdout_unspent") >= 10
    assert (
        '"source_git_head": sys.argv[2].lower(),' in runner
        and '"source_git_head": os.environ["SIGNTRAJ_SOURCE_GIT_HEAD"].lower(),'
        in runner
        and '"source_git_head": head.lower(),' in runner
    )
    assert (
        'require_holdout_unspent\nrevalidate_source_binding\n"$PYTHON_BIN" - '
        '"$MODE_CONTROL_DIR"' in runner
    )
    assert runner.count(
        'require_holdout_unspent\n  "$PYTHON_BIN" -m '
        '"$EXECUTION_CONTROL_MODULE" release'
    ) >= 1
    assert runner.count("LEASE_ACTIVE=0\n  require_holdout_unspent") >= 1
    assert 'payload["prior_one_update_smoke"] = prior_smoke' in runner
    assert 'complete["prior_one_update_smoke"] = prior_smoke' in runner
    assert 'ready["prior_one_update_smoke"] = prior_smoke' in runner


@pytest.mark.parametrize(
    "name",
    [
        "launch_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh",
        "test_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh",
        "calibrate_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh",
        "smoke_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh",
        "train_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh",
        "run_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh",
    ],
)
def test_protocol_v2_shell_surfaces_are_distinct_and_parse(name):
    path = SCRIPT_DIR / name
    text = path.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", path], check=True)
    assert "protocol_v2_run_r3" in text
    assert "WANDB_MODE=disabled" in text or name.startswith(("smoke_", "train_"))
    assert "scancel" not in text
    validation_offset = text.index("^[0-9a-fA-F]{40}$")
    lowercase_offset = text.index("${SOURCE_GIT_HEAD,,}")
    namespace_offset = text.find("source_${SOURCE_GIT_HEAD")
    assert validation_offset < lowercase_offset
    assert namespace_offset == -1 or lowercase_offset < namespace_offset


def test_confirmation_holdout_absence_guard_rejects_file_and_broken_symlink(
    tmp_path,
):
    marker = tmp_path / "confirmation_holdout_spent.json"
    result = stage_c_pilot_prerequisites.validate_unspent_confirmation_holdout(
        spend_marker=marker
    )
    assert result["spend_marker_absent"] is True

    marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        stage_c_pilot_prerequisites.PrerequisiteError,
        match="spend marker exists",
    ):
        stage_c_pilot_prerequisites.validate_unspent_confirmation_holdout(
            spend_marker=marker
        )
    marker.unlink()
    marker.symlink_to(tmp_path / "missing-target")
    with pytest.raises(
        stage_c_pilot_prerequisites.PrerequisiteError,
        match="spend marker exists",
    ):
        stage_c_pilot_prerequisites.validate_unspent_confirmation_holdout(
            spend_marker=marker
        )


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_protocol_v2_launcher_executes_four_stage_dry_preview_without_sbatch(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    head = "a1" * 20
    uppercase_head = head.upper()
    _executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        "case \"$1 $2\" in\n"
        f"  'rev-parse HEAD') echo {head};;\n"
        "  'status --porcelain') exit 0;;\n"
        f"  'ls-remote --heads') echo '{head} refs/heads/'\"$4\";;\n"
        "  *) exit 1;;\n"
        "esac\n",
    )
    _executable(fake_bin / "squeue", "#!/bin/sh\nexit 0\n")
    _executable(
        fake_bin / "sinfo",
        "#!/bin/sh\nprintf 'node-a|gpu,pair02\\nnode-b|pair02,gpu\\n'\n",
    )
    _executable(
        fake_bin / "python",
        "#!/bin/sh\n# Prerequisite validation is isolated in its own unit tests.\nexit 0\n",
    )
    sbatch_marker = tmp_path / "sbatch-called"
    _executable(
        fake_bin / "sbatch",
        f"#!/bin/sh\ntouch '{sbatch_marker}'\nexit 99\n",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "PROJECT_DIR": str(ROOT),
        "SOURCE_GIT_HEAD": uppercase_head,
        "SOURCE_REMOTE_BRANCH": (
            "codex/csl-daily-centered-generator-stage-c-v2-run-r3"
        ),
        "PYTHON_BIN": str(fake_bin / "python"),
        "USER": "stage-c-preview",
    }
    result = subprocess.run(
        [
            "bash",
            SCRIPT_DIR
            / "launch_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh",
            "pair02",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not sbatch_marker.exists()
    assert "Validated pair02 nodes: node-a node-b" in result.stdout
    assert "NEXT cpu: sbatch --parsable" in result.stdout
    assert "NEXT calibration: sbatch --parsable --dependency=afterok:$CPU_JOB_ID" in (
        result.stdout
    )
    assert "NEXT smoke: sbatch --parsable --dependency=afterok:$CALIBRATION_JOB_ID" in (
        result.stdout
    )
    assert "NEXT pilot: sbatch --parsable --dependency=afterok:$SMOKE_JOB_ID" in (
        result.stdout
    )
    assert "Preview only" in result.stdout
    assert f"SOURCE_GIT_HEAD={head}" in result.stdout
    assert f"SOURCE_GIT_HEAD={uppercase_head}" not in result.stdout
