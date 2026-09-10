import copy
import json
from pathlib import Path

import pytest

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.scripts import stage_c_paired_roce as roce
from NIAF.continuous_trajectory_field.scripts import (
    stage_c_atomic,
)
from NIAF.continuous_trajectory_field.scripts import (
    stage_c_calibration_control as calibration_control,
)
from NIAF.continuous_trajectory_field.scripts import (
    stage_c_execution_control as execution_control,
)
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_decision
from NIAF.continuous_trajectory_field.scripts import (
    stage_c_pilot_prerequisites as prerequisites,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "NIAF/continuous_trajectory_field/configs"
SCRIPT_DIR = ROOT / "scripts/NIAF"
RECOVERY_EVIDENCE_ROOT = Path(
    "/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory"
)
MEMORY = (
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
    "memory_pilot_run_r2"
)
MATCHED_OFF = (
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_"
    "generator_adaptation_matched_off_pilot_run_r2"
)
CALIBRATION = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_sentence_memory_relevance_calibration_"
    "stage_c_generator_adaptation_retry2_v1"
)
SOURCE_TERMINAL_DECISION = (
    ROOT
    / "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "absolute_binding_motion_contrast_v1/evaluation/"
    "ordered_development_decision/decision.json"
)
MODES = [
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
    "association_disabled",
]


def _script(name):
    return (SCRIPT_DIR / name).read_text(encoding="utf-8")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_stage_c_atomic_publication_uses_no_replace(tmp_path):
    target = tmp_path / "immutable.json"
    stage_c_atomic.publish_bytes_no_replace(target, b"first\n")
    with pytest.raises(stage_c_atomic.AtomicPublishError):
        stage_c_atomic.publish_bytes_no_replace(target, b"second\n")
    assert target.read_bytes() == b"first\n"


def test_stage_c_configs_are_exact_matched_non_authorizing_pair():
    memory = load_config(CONFIG_DIR / f"{MEMORY}.yaml")
    off = load_config(CONFIG_DIR / f"{MATCHED_OFF}.yaml")
    for cfg, arm, train_mode, probability in (
        (memory, "memory", "dropout", 0.25),
        (off, "matched_off", "off", 1.0),
    ):
        stage = cfg["sentence_memory_safety"]["stage_c"]
        assert stage["schema_name"] == (
            "signtrajfield_centered_stage_c_generator_adaptation"
        )
        assert stage["schema_version"] == 1
        assert stage["development_only"] is True
        assert stage["non_authorizing"] is True
        assert stage["promotion_eligible"] is False
        assert stage["arm"] == arm
        assert stage["source_checkpoint"] == {
            "path": (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
                "absolute_binding_motion_contrast_v1/checkpoints/best_infeasible.pt"
            ),
            "sha256": (
                "b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202"
            ),
            "selection_status": "best_infeasible",
            "epoch": 5,
            "global_step": 360,
        }
        assert stage["active_stage_c"] == {
            "calibration_artifact_dir": CALIBRATION,
            "calibration_schema_name": (
                "signtrajfield_sentence_memory_relevance_calibration"
            ),
            "calibration_schema_version": 1,
        }
        calibration = cfg["sentence_memory"]["relevance_calibration"]
        assert calibration["artifact_dir"] == CALIBRATION
        assert calibration["artifact_identity"] is None
        assert cfg["conditioning"]["word_prior_train_mode"] == "off"
        assert cfg["conditioning"]["sentence_memory_train_mode"] == train_mode
        assert cfg["conditioning"]["sentence_memory_dropout_probability"] == probability
        assert cfg["sentence_memory_safety"]["phase_b"]["enabled"] is False
        assert cfg["sentence_memory_safety"]["paired_corruption"]["enabled"] is False
        assert cfg["train"]["batch_size"] == 64
        assert cfg["train"]["accumulation_steps"] == 2
        assert cfg["train"]["epochs"] == 1
        assert cfg["train"].get("max_train_batches", 0) == 0
        assert cfg["eval"].get("max_batches", 0) == 0
        assert cfg["data"].get("limit_train", 0) == 0
        assert cfg["data"].get("limit_val", 0) == 0
        assert cfg["train"]["length_bucketed_batches"] is True
        assert cfg["train"]["drop_last"] is False
        assert cfg["train"]["freeze_sentence_memory"] is True
        assert cfg["selection"]["require_feasible"] is False
        assert cfg["eval"]["sentence_memory_modes"] == MODES
    assert (
        CONFIG_DIR
        / "csl_daily_signtrajfield_v3_sentence_memory_stage_c_"
        "generator_adaptation_memory_pilot.yaml"
    ).is_file()
    assert (
        CONFIG_DIR
        / "csl_daily_signtrajfield_v3_sentence_memory_stage_c_"
        "generator_adaptation_matched_off_pilot.yaml"
    ).is_file()
    runner = _script("run_csl_daily_stage_c_generator_adaptation_pilot.sh")
    assert "generator_adaptation_smoke\"" in runner
    assert "generator_adaptation_pilot\"" in runner
    assert "generator_adaptation_smoke_run_r2" not in runner
    assert "generator_adaptation_pilot_run_r2\"" not in runner


def test_stage_c_accepts_exact_legacy_source_decision_and_rejects_added_authority(
    tmp_path,
):
    decision = prerequisites.validate_source_terminal_decision(
        decision_path=SOURCE_TERMINAL_DECISION,
        expected_sha256=prerequisites.SOURCE_TERMINAL_DECISION_SHA256,
        expected_identity=prerequisites.SOURCE_TERMINAL_DECISION_IDENTITY,
    )
    assert set(decision) == prerequisites.SOURCE_TERMINAL_DECISION_FIELDS
    assert "authorized_purpose" not in decision
    assert decision["status"] == "valid_infeasible"
    assert decision["integrity_valid"] is True
    assert decision["development_feasible"] is False
    assert decision["confirmation_manifest_opened"] is False
    assert decision["test_data_accessed"] is False

    for value, name in ((None, "null"), ("stage_c", "non_null")):
        changed = dict(decision)
        changed["authorized_purpose"] = value
        path = tmp_path / f"decision_{name}.json"
        _write_json(path, changed)
        with pytest.raises(prerequisites.PrerequisiteError, match="fields differ"):
            prerequisites.validate_source_terminal_decision(
                decision_path=path,
                expected_sha256=prerequisites.sha256_file(path),
                expected_identity=decision["decision_identity"],
            )


def test_retry2_recovery_evidence_reopens_run_r1_and_rejects_false_file_claim(
    tmp_path, monkeypatch
):
    policy_path = (
        CONFIG_DIR
        / "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
    )
    manifest_path = (
        CONFIG_DIR
        / "csl_daily_stage_c_generator_adaptation_retry2_recovery_evidence_v1.json"
    )
    audit = prerequisites.validate_retry2_recovery_evidence(
        policy_path=policy_path,
        recovery_manifest=manifest_path,
        source_root=ROOT,
        evidence_root=RECOVERY_EVIDENCE_ROOT,
    )
    assert audit["run_r1_source_git_head"] == "a883baf4a0a95b4ebb2007564837c4d9b4f817cc"
    assert audit["failed_smoke_job_id"] == "143525"
    assert audit["prior_execution_lease_created"] is False
    assert audit["prior_scientific_output_observed"] is False
    assert audit["retry_authorized"] is True
    assert audit["retry_authorization_reason"] == (
        "attested_pre_claim_pre_science_implementation_failure"
    )
    assert audit["calibration_artifact_identity"] == (
        "8999d9e81b2fccab6ed368d374e99a9f65f0dcdb4c8203f2d41d5e02f469cf49"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forbidden = RECOVERY_EVIDENCE_ROOT / manifest["required_absent_paths"][0]
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True if self == forbidden else original_exists(self),
    )
    with pytest.raises(
        prerequisites.PrerequisiteError,
        match="scientific output unexpectedly exists",
    ):
        prerequisites.validate_retry2_recovery_evidence(
            policy_path=policy_path,
            recovery_manifest=manifest_path,
            source_root=ROOT,
            evidence_root=RECOVERY_EVIDENCE_ROOT,
        )
    monkeypatch.setattr(Path, "exists", original_exists)

    relative = Path(
        "NIAF/continuous_trajectory_field/configs/"
        "csl_daily_stage_c_generator_adaptation_retry2_recovery_evidence_v1.json"
    )
    changed_path = tmp_path / relative
    changed = manifest
    changed["required_files"]["failed_smoke_stdout"]["sha256"] = "0" * 64
    _write_json(changed_path, changed)
    monkeypatch.setattr(
        prerequisites,
        "validate_policy",
        lambda _path: {
            "recovery_contract": {
                "evidence_manifest": {
                    "path": str(relative),
                    "sha256": prerequisites.sha256_file(changed_path),
                }
            }
        },
    )
    with pytest.raises(prerequisites.PrerequisiteError, match="incident file changed"):
        prerequisites.validate_retry2_recovery_evidence(
            policy_path=tmp_path / "policy.json",
            recovery_manifest=changed_path,
            source_root=tmp_path,
            evidence_root=RECOVERY_EVIDENCE_ROOT,
        )


def test_execution_lease_recovers_failed_terminal_write_before_archive(
    tmp_path, monkeypatch
):
    inputs = {}
    for name in (
        "memory_config",
        "matched_off_config",
        "cpu_gate",
        "calibration_completion",
    ):
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        inputs[name] = path
    binding = execution_control.execution_binding(
        mode="smoke",
        source_git_head="a" * 40,
        source_remote_ref="origin/stage-c",
        source_remote_head="a" * 40,
        pair_constraint="pair04",
        **inputs,
    )
    lease = tmp_path / "control/active_execution_lease"
    first_attestation = tmp_path / "control/attestations/91.0.json"
    first = execution_control.acquire_execution_lease(
        lease_path=lease,
        attestation_path=first_attestation,
        slurm_job_id="91",
        slurm_restart_count=0,
        binding=binding,
    )
    terminal_unsigned = {
        "schema_name": "signtrajfield_stage_c_execution_lease_terminal",
        "schema_version": 1,
        "outcome": "failed",
        "claim_identity": first["claim"]["claim_identity"],
        "current_slurm_job_id": "91",
        "current_slurm_restart_count": 0,
        "attestation_sha256": execution_control.sha256_file(first_attestation),
        "self_reported_exit_code": 1,
    }
    _write_json(
        lease / "TERMINAL.json",
        {
            **terminal_unsigned,
            "terminal_identity": execution_control.digest_json(terminal_unsigned),
        },
    )
    monkeypatch.setattr(
        execution_control,
        "_scheduler_job",
        lambda _job: ("RUNNING", 1, "JobState=RUNNING Restarts=1"),
    )
    second = execution_control.acquire_execution_lease(
        lease_path=lease,
        attestation_path=tmp_path / "control/attestations/91.1.json",
        slurm_job_id="91",
        slurm_restart_count=1,
        binding=binding,
    )
    assert second["claim"]["claim_identity"] != first["claim"]["claim_identity"]
    assert first["claim"]["claim_identity"] in second["claim"][
        "prior_claim_identities"
    ]
    assert (
        tmp_path
        / "control/active_execution_lease.history"
        / f"{first['claim']['claim_identity']}.failed/TERMINAL.json"
    ).is_file()


def test_calibration_lease_recovers_failed_terminal_write_before_archive(
    tmp_path, monkeypatch
):
    inputs = {}
    for name in (
        "launcher",
        "cpu_gate",
        "audit_config",
        "bank_manifest",
        "bank_ready",
        "train_neighbors",
    ):
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        inputs[name] = path
    binding = calibration_control.calibration_binding(
        source_git_head="b" * 40,
        source_remote_ref="origin/stage-c",
        source_remote_head="b" * 40,
        **inputs,
    )
    assert binding["source_file_profile"] == (
        "stage_c_generator_adaptation_retry2_v1"
    )
    legacy_binding = dict(binding)
    legacy_binding["source_file_profile"] = "stage_c_generator_adaptation_v1"
    legacy_binding["digest"] = calibration_control.digest_json(
        {key: value for key, value in legacy_binding.items() if key != "digest"}
    )
    calibration_control._validate_binding(legacy_binding)
    protocol_binding = calibration_control.calibration_binding(
        source_git_head="b" * 40,
        source_remote_ref="origin/stage-c",
        source_remote_head="b" * 40,
        source_file_profile="stage_c_generator_adaptation_protocol_v2_run_r3",
        **inputs,
    )
    assert protocol_binding["source_file_profile"] == (
        "stage_c_generator_adaptation_protocol_v2_run_r3"
    )
    calibration_control._validate_binding(protocol_binding)
    unknown_binding = dict(binding)
    unknown_binding["source_file_profile"] = "unapproved"
    unknown_binding["digest"] = calibration_control.digest_json(
        {key: value for key, value in unknown_binding.items() if key != "digest"}
    )
    with pytest.raises(calibration_control.CalibrationLeaseError, match="not exact"):
        calibration_control._validate_binding(unknown_binding)
    lease = tmp_path / "calibration/active_execution_lease"
    first_attestation = tmp_path / "calibration/attestations/92.0.json"
    first = calibration_control.acquire(
        lease=lease,
        attestation=first_attestation,
        job_id="92",
        restart=0,
        binding=binding,
    )
    terminal_unsigned = {
        "schema_name": "signtrajfield_stage_c_calibration_lease_terminal",
        "schema_version": 1,
        "outcome": "failed",
        "claim_identity": first["claim"]["claim_identity"],
        "current_slurm_job_id": "92",
        "current_slurm_restart_count": 0,
        "attestation_sha256": calibration_control.sha256_file(first_attestation),
        "self_reported_exit_code": 1,
    }
    _write_json(
        lease / "TERMINAL.json",
        {
            **terminal_unsigned,
            "terminal_identity": calibration_control.digest_json(
                terminal_unsigned
            ),
        },
    )
    monkeypatch.setattr(
        calibration_control,
        "_scheduler",
        lambda _job: ("RUNNING", 1),
    )
    second = calibration_control.acquire(
        lease=lease,
        attestation=tmp_path / "calibration/attestations/92.1.json",
        job_id="92",
        restart=1,
        binding=binding,
    )
    assert second["claim"]["claim_identity"] != first["claim"]["claim_identity"]
    assert (
        tmp_path
        / "calibration/active_execution_lease.history"
        / f"{first['claim']['claim_identity']}.failed/TERMINAL.json"
    ).is_file()


def test_stage_c_launch_chain_orders_cpu_calibration_smoke_then_pilot():
    launcher = _script("launch_csl_daily_stage_c_generator_adaptation_pilot.sh")
    assert "HAVE_CPU=0 HAVE_CALIBRATION=0 HAVE_SMOKE=0 HAVE_PILOT=0" in launcher
    assert 'if [[ "$HAVE_CPU" == "0" ]]' in launcher
    assert 'if [[ "$HAVE_CALIBRATION" == "0" ]]' in launcher
    assert 'if [[ "$HAVE_SMOKE" == "0" ]]' in launcher
    assert 'if [[ "$HAVE_PILOT" == "0" ]]' in launcher
    assert "create only the incomplete dependency suffix" in launcher
    assert 'SMOKE_ARGS=(--parsable --constraint="$PAIR_CONSTRAINT"' in launcher
    assert 'PILOT_ARGS=(--parsable --constraint="$PAIR_CONSTRAINT"' in launcher
    assert "sinfo -N -h -p spark -o '%N|%f'" in launcher
    assert 'features[feature_index] == pair' in launcher
    assert "sinfo -C" not in launcher
    assert "unset WANDB_API_KEY" in launcher
    assert "WANDB_MODE=disabled" in launcher
    assert "squeue -h" in launcher
    assert "-t PENDING" not in launcher
    assert "'%i|%j|%T'" in launcher
    assert "active Stage-C Slurm job(s) already exist" in launcher
    assert "scancel" not in launcher
    assert "Preview only" in launcher and '"$ACTION" == "--submit"' in launcher
    assert "validate-foundation" in launcher and "validate-smoke" in launcher
    assert "/smoke/PUBLICATION/READY" in launcher
    assert "/pilot/PUBLICATION/READY" in launcher
    for name in (
        "test_csl_daily_stage_c_generator_adaptation_sbatch.sh",
        "calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh",
        "smoke_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh",
        "train_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh",
    ):
        assert name in launcher


def test_stage_c_launcher_isolates_all_publish_before_release_recovery():
    launcher = _script("launch_csl_daily_stage_c_generator_adaptation_pilot.sh")
    for mode in ("CALIBRATION", "SMOKE", "PILOT"):
        assert f"RECOVER_{mode}=1" in launcher
    assert "inspect-completed" in launcher
    assert launcher.count("inspect-published") == 2
    assert "validate-finalized" in launcher
    assert 'if [[ "$RECOVER_CALIBRATION" == "1" || "$RECOVER_SMOKE" == "1"' in launcher
    assert 'RECOVERY_LABEL="calibration" RECOVERY_SCRIPT="$CALIBRATION_SCRIPT"' in launcher
    assert 'RECOVERY_LABEL="smoke" RECOVERY_SCRIPT="$SMOKE_SCRIPT"' in launcher
    assert 'RECOVERY_LABEL="pilot" RECOVERY_SCRIPT="$BATCH_SCRIPT"' in launcher
    assert "recovery is isolated and no downstream job is scheduled" in launcher
    assert launcher.index("inspect-completed") < launcher.index("validate-foundation")
    assert launcher.index("inspect-published") < launcher.index("validate-smoke")


def test_stage_c_policy_predeclares_exact_non_authorizing_progression(tmp_path):
    policy_path = (
        CONFIG_DIR
        / "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
    )
    policy = stage_c_pilot_decision.validate_policy(policy_path)
    assert policy["execution"] == {
        "arms": ["memory", "matched_off"],
        "accumulation_steps": 2,
        "batch_per_rank": 64,
        "effective_global_batch": 256,
        "expected_epoch": 1,
        "expected_global_step": {"pilot": 72, "smoke": 1},
        "full_pilot_data_limit_train": 0,
        "full_pilot_data_limit_val": 0,
        "full_pilot_drop_last": False,
        "full_pilot_eval_max_batches": 0,
        "full_pilot_length_bucketed_batches": True,
        "full_pilot_train_max_batches": 0,
        "smoke_eval_max_batches": 1,
        "smoke_train_max_batches": 2,
        "world_size": 2,
    }
    assert policy["status_values"]["pilot_success"] == {
        "next_permitted_action": (
            "none_requires_fresh_preregistration_without_pilot_outcome_access"
        ),
        "status": "pilot_complete_development_signal",
    }
    assert policy["authorization"]["authorized_purpose"] is None
    assert policy["authorization"]["non_authorizing"] is True
    assert policy["recovery_contract"] == {
        "evidence_manifest": {
            "path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_retry2_"
                "recovery_evidence_v1.json"
            ),
            "sha256": (
                "ccf47b72390c4be28775b207c67f7372d3cbc599fcddb3545df8295acc88b39b"
            ),
        },
        "failed_smoke_job_id": "143525",
        "failure_class": (
            "pre_science_source_terminal_decision_schema_compatibility"
        ),
        "prior_evidence_must_be_preserved": True,
        "prior_execution_lease_created": False,
        "prior_scientific_output_observed": False,
        "retry_authorization_reason": (
            "attested_pre_claim_pre_science_implementation_failure"
        ),
        "retry_authorized": True,
        "run_generation": "run_r2",
        "supersedes_run_generation": "run_r1",
    }
    stage_c_pilot_decision.validate_policy(
        CONFIG_DIR / "csl_daily_stage_c_generator_adaptation_decision_policy_v1.json"
    )
    tampered = json.loads(json.dumps(policy))
    tampered["recovery_contract"]["prior_scientific_output_observed"] = True
    tampered_path = (
        tmp_path
        / "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
    )
    _write_json(tampered_path, tampered)
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="recovery contract changed",
    ):
        stage_c_pilot_decision.validate_policy(tampered_path)
    tampered["recovery_contract"] = None
    _write_json(tampered_path, tampered)
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="recovery contract changed",
    ):
        stage_c_pilot_decision.validate_policy(tampered_path)


def test_protocol_v2_policy_is_exact_and_forbids_same_generation_retry(tmp_path):
    policy_path = (
        CONFIG_DIR
        / "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
        "decision_policy_v1.json"
    )
    policy = stage_c_pilot_decision.validate_policy(policy_path)
    assert policy["recovery_contract"]["prior_scientific_output_observed"] is True
    assert policy["recovery_contract"]["prior_execution_lease_created"] is True
    assert policy["recovery_contract"]["prior_memory_optimizer_updates_observed"] == 1
    assert policy["recovery_contract"]["matched_off_arm_started"] is False
    assert policy["recovery_contract"]["same_protocol_retry_authorized"] is False
    assert policy["recovery_contract"]["resume_authorized"] is False
    assert policy["retry_policy"] == {
        "malformed_completion": "stop_no_replace",
        "partial_scientific_output": "stop_no_retry",
        "unique_complete_execution": "reuse_only_for_publication_recovery",
        "zero_scientific_output": "stop_no_retry_within_protocol_generation",
    }

    changed = copy.deepcopy(policy)
    changed["retry_policy"]["zero_scientific_output"] = (
        "new_attempt_permitted_after_terminal_lease"
    )
    changed_path = tmp_path / policy_path.name
    _write_json(changed_path, changed)
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="policy content changed",
    ):
        stage_c_pilot_decision.validate_policy(changed_path)


def test_recovery_dispatch_and_source_binding_generation_are_exact(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        prerequisites,
        "validate_retry2_recovery_evidence",
        lambda **kwargs: calls.append(("r2", kwargs)) or {"generation": "r2"},
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_protocol_v2_run_r3_recovery_evidence",
        lambda **kwargs: calls.append(("r3", kwargs)) or {"generation": "r3"},
    )
    common = {
        "recovery_manifest": tmp_path / "manifest.json",
        "source_root": tmp_path,
        "evidence_root": tmp_path,
    }
    r2_policy = tmp_path / prerequisites.RETRY2_POLICY_NAME
    r3_policy = tmp_path / prerequisites.PROTOCOL_V2_RUN_R3_POLICY_NAME
    assert prerequisites.validate_recovery_evidence(
        policy_path=r2_policy, **common
    ) == {"generation": "r2"}
    assert prerequisites.validate_recovery_evidence(
        policy_path=r3_policy, **common
    ) == {"generation": "r3"}
    assert [item[0] for item in calls] == ["r2", "r3"]
    with pytest.raises(prerequisites.PrerequisiteError, match="not registered"):
        prerequisites.validate_recovery_evidence(
            policy_path=tmp_path / "unknown.json", **common
        )

    policy_sha = "a" * 64
    recovery_name = (
        "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
        "recovery_evidence_v1.json"
    )
    r2_memory_name = (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
        "memory_pilot_run_r2.yaml"
    )
    r2_off_name = (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
        "matched_off_pilot_run_r2.yaml"
    )
    r3_memory_name = (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
        "memory_protocol_v2_run_r3.yaml"
    )
    r3_off_name = (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
        "matched_off_protocol_v2_run_r3.yaml"
    )
    r3_bound = {
        str(r3_policy.resolve()): policy_sha,
        str((tmp_path / recovery_name).resolve()): "b" * 64,
        str((tmp_path / r2_memory_name).resolve()): "c" * 64,
        str((tmp_path / r2_off_name).resolve()): "d" * 64,
        str((tmp_path / r3_memory_name).resolve()): "1" * 64,
        str((tmp_path / r3_off_name).resolve()): "2" * 64,
        str(
            (
                tmp_path
                / "run_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh"
            ).resolve()
        ): "3" * 64,
        str(
            (
                tmp_path
                / "calibrate_csl_daily_stage_c_generator_adaptation_"
                "protocol_v2_run_r3_sbatch.sh"
            ).resolve()
        ): "4" * 64,
        **{
            str((tmp_path / f"r3_source_{index}.py").resolve()): "e" * 64
            for index in range(21)
        },
    }
    assert stage_c_pilot_decision._validate_source_binding_generation(
        bound_files=r3_bound,
        decision_policy={"path": str(r3_policy.resolve()), "sha256": policy_sha},
        mode="smoke",
    ) == 29
    smoke_ready = tmp_path / "prerequisites/source_abc/smoke/PUBLICATION/READY"
    smoke_sha = "9" * 64
    pilot_bound = {**r3_bound, str(smoke_ready.resolve()): smoke_sha}
    prior_smoke = {
        "ready_path": str(smoke_ready.resolve()),
        "ready_sha256": smoke_sha,
    }
    assert stage_c_pilot_decision._validate_source_binding_generation(
        bound_files=pilot_bound,
        decision_policy={"path": str(r3_policy.resolve()), "sha256": policy_sha},
        mode="pilot",
        prior_one_update_smoke=prior_smoke,
    ) == 30
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="does not hash-bind smoke READY",
    ):
        stage_c_pilot_decision._validate_source_binding_generation(
            bound_files=pilot_bound,
            decision_policy={
                "path": str(r3_policy.resolve()),
                "sha256": policy_sha,
            },
            mode="pilot",
            prior_one_update_smoke={**prior_smoke, "ready_sha256": "0" * 64},
        )
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="lacks the one-update smoke prerequisite",
    ):
        stage_c_pilot_decision._validate_source_binding_generation(
            bound_files=pilot_bound,
            decision_policy={
                "path": str(r3_policy.resolve()),
                "sha256": policy_sha,
            },
            mode="pilot",
        )
    missing_marker = dict(r3_bound)
    missing_marker.pop(str((tmp_path / recovery_name).resolve()))
    missing_marker[str((tmp_path / "replacement.py").resolve())] = "f" * 64
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError,
        match="protocol-v2 source binding file set",
    ):
        stage_c_pilot_decision._validate_source_binding_generation(
            bound_files=missing_marker,
            decision_policy={"path": str(r3_policy.resolve()), "sha256": policy_sha},
            mode="smoke",
        )


def test_protocol_v2_configs_deep_equal_run_r2_and_reject_scientific_drift(
    tmp_path, monkeypatch
):
    names = {
        "r2_memory": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "memory_pilot_run_r2.yaml"
        ),
        "r2_off": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "matched_off_pilot_run_r2.yaml"
        ),
        "r3_memory": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "memory_protocol_v2_run_r3.yaml"
        ),
        "r3_off": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "matched_off_protocol_v2_run_r3.yaml"
        ),
    }
    source_paths = {
        "r2_memory": CONFIG_DIR / names["r2_memory"],
        "r2_off": CONFIG_DIR / names["r2_off"],
        "r3_memory": CONFIG_DIR / names["r3_memory"],
        "r3_off": CONFIG_DIR / names["r3_off"],
    }
    config_root = tmp_path / "NIAF/continuous_trajectory_field/configs"
    paths = {name: config_root / filename for name, filename in names.items()}
    for name, source in source_paths.items():
        _write_json(paths[name], load_config(source))
    archive_path = tmp_path / "ARCHIVE.json"
    _write_json(
        archive_path,
        {
            "retained_evidence": {
                "run_source_configs": {
                    str(paths["r2_memory"]): prerequisites.sha256_file(
                        paths["r2_memory"]
                    ),
                    str(paths["r2_off"]): prerequisites.sha256_file(paths["r2_off"]),
                }
            },
            "recovery": {
                "new_calibration_root": (
                    "experiments/NIAF/continuous_trajectory_field/"
                    "csl_daily_sentence_memory_relevance_calibration_stage_c_"
                    "generator_adaptation_protocol_v2_run_r3"
                ),
                "new_prerequisite_root": (
                    "experiments/NIAF/continuous_trajectory_field/"
                    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                    "prerequisites"
                ),
                "new_smoke_root": (
                    "experiments/NIAF/continuous_trajectory_field/"
                    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                    "adaptation_protocol_v2_run_r3_smoke"
                ),
                "new_pilot_root": (
                    "experiments/NIAF/continuous_trajectory_field/"
                    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                    "adaptation_protocol_v2_run_r3_pilot"
                ),
            },
        },
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_protocol_v2_run_r3_recovery_evidence",
        lambda **_kwargs: {
            "audit_identity": "1" * 64,
            "incident_archive_path": str(archive_path),
        },
    )
    kwargs = {
        "policy_path": tmp_path / prerequisites.PROTOCOL_V2_RUN_R3_POLICY_NAME,
        "recovery_manifest": tmp_path / "recovery.json",
        "source_root": tmp_path,
        "evidence_root": tmp_path,
        "memory_config": paths["r3_memory"],
        "matched_off_config": paths["r3_off"],
    }
    audit = prerequisites.validate_protocol_v2_run_r3_configs(**kwargs)
    assert audit["scientific_settings_equal_run_r2"] is True
    assert audit["same_settings_except_arm_switch"] is True
    assert audit["original_stage_b_warm_start_pinned"] is True

    changed = load_config(paths["r3_memory"])
    changed["train"]["epochs"] = 2
    _write_json(paths["r3_memory"], changed)
    with pytest.raises(prerequisites.PrerequisiteError, match="differ"):
        prerequisites.validate_protocol_v2_run_r3_configs(**kwargs)

    changed = load_config(source_paths["r3_memory"])
    changed["sentence_memory"]["relevance_calibration"]["artifact_dir"] = (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_sentence_memory_relevance_calibration_stage_c_"
        "generator_adaptation_retry2_v1"
    )
    changed["sentence_memory_safety"]["stage_c"]["active_stage_c"][
        "calibration_artifact_dir"
    ] = changed["sentence_memory"]["relevance_calibration"]["artifact_dir"]
    _write_json(paths["r3_memory"], changed)
    with pytest.raises(
        prerequisites.PrerequisiteError,
        match="scientific settings differ|fresh namespace changed",
    ):
        prerequisites.validate_protocol_v2_run_r3_configs(**kwargs)


def test_stage_c_calibration_is_fresh_source_bound_and_train_only():
    calibration = _script(
        "calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh"
    )
    runner = _script("run_csl_daily_stage_c_generator_adaptation_pilot.sh")
    config = (CONFIG_DIR / f"{MEMORY}.yaml").read_text(encoding="utf-8")
    combined = calibration + runner + config
    assert "calibration_stage_c_generator_adaptation_retry2_v1" in combined
    assert "calibration_v1_retry3" not in combined
    assert '"$STAGER" stage' in calibration and '"train"' in calibration
    assert '! -e "$STAGED_BANK/neighbors_val.npz"' in calibration
    assert '! -e "$STAGED_BANK/neighbors_test.npz"' in calibration
    assert "validate_relevance_calibration_source" in calibration
    assert "signtrajfield_stage_c_calibration_completion" in calibration
    assert "stage_c_generator_adaptation_retry2_v1" in calibration
    assert "stage_c_calibration_control" in calibration
    assert "stage_c_calibration_transition" in calibration
    assert "calibration_transition_identity" in calibration
    assert "reconcile-completed" in calibration
    assert "CPU_GATE_READY" in calibration and "CPU_GATE_READY" in runner
    assert "CALIBRATION_COMPLETION" in runner
    assert "stage_c_pilot_prerequisites" in runner
    for script_name in (
        "test_csl_daily_stage_c_generator_adaptation_sbatch.sh",
        "calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh",
        "run_csl_daily_stage_c_generator_adaptation_pilot.sh",
        "launch_csl_daily_stage_c_generator_adaptation_pilot.sh",
    ):
        script = _script(script_name)
        assert "validate-recovery" in script
        assert "retry2_recovery_evidence_v1.json" in script
    assert "--recovery_policy" in calibration and "--recovery_policy" in runner


def test_stage_c_runner_pins_two_rank_effective_batch_and_roce_only():
    runner = _script("run_csl_daily_stage_c_generator_adaptation_pilot.sh")
    assert '"${SLURM_NNODES:-0}" != "2"' in runner
    assert '"${SLURM_NTASKS:-0}" != "2"' in runner
    assert "--batch_size 64" in runner
    assert 'cfg["train"]["accumulation_steps"] == 2' in runner
    assert '"effective_global_batch": 64 * 2 * 2' in runner
    assert "export NCCL_NET=IB" in runner
    assert "export NCCL_IB_DISABLE=0" in runner
    assert 'export NCCL_IB_GID_INDEX=5' in runner
    assert 'NCCL_SOCKET_IFNAME="enp1s0f1np1,enP2p1s0f1np1"' in runner
    assert 'GLOO_SOCKET_IFNAME="enp1s0f1np1"' in runner
    assert "enP7s7" not in runner
    assert "run_nccl_profile()" in runner
    assert runner.count("run_nccl_profile ") == 3
    assert "verify-logs" in runner
    assert "snapshot-counters" in runner and "compare-counters" in runner
    assert '"train val"' in runner
    assert '! -e "$d/neighbors_test.npz"' in runner
    assert "manifest_confirmation" not in runner
    assert "--stage_c_warm_start" in runner
    assert runner.index('run_arm memory "$MEMORY_CFG"') < runner.index(
        'run_arm matched_off "$OFF_CFG"'
    )
    assert "--max_train_batches 2 --max_val_batches 1" in runner
    assert "full pilot requires the completed paired one-update smoke gate" in runner
    assert "expected_step = 1 if execution_mode == \"smoke\" else 72" in runner
    assert 'rows[0].get("global_step") != expected_step' in runner
    assert '"active_lease_claim_identity"' in runner
    assert "stage_c_pilot_decision" in runner
    assert "active_execution_lease.history" not in runner
    assert 'MODE_CONTROL_DIR="$PREREQUISITE_ROOT/$EXECUTION_MODE"' in runner
    assert 'find "$ATTEMPTS_ROOT"' in runner
    assert "partial scientific output" in runner
    assert "os.link(" not in runner
    assert "PILOT_CHECKPOINT_AUDIT.json" in runner
    assert '--execution_mode "$EXECUTION_MODE"' in runner


def _node_value(node, primary_ip, secondary_ip):
    rails = {}
    for (hca, netdev), address in zip(
        roce.RAILS, (primary_ip, secondary_ip), strict=True
    ):
        rails[hca] = {
            "port": 1,
            "state": "4: ACTIVE",
            "physical_state": "5: LinkUp",
            "link_layer": "Ethernet",
            "rate": "200 Gb/sec (4X HDR)",
            "netdev": netdev,
            "operstate": "up",
            "speed_mbit": 200_000,
            "mtu": 9_000,
            "ipv4_interface": f"{address}/30",
            "gid_index": 5,
            "gid_type": "RoCE v2",
            "gid": f"::ffff:{address}",
        }
    return {
        "schema_name": roce.SCHEMA,
        "schema_version": roce.SCHEMA_VERSION,
        "host": node,
        "rails": rails,
    }


def test_pair_validation_binds_both_direct_pair_networks(tmp_path):
    nodes = ["node-a", "node-b"]
    values = (
        _node_value(nodes[0], "10.250.4.1", "10.250.104.1"),
        _node_value(nodes[1], "10.250.4.2", "10.250.104.2"),
    )
    for node, value in zip(nodes, values, strict=True):
        (tmp_path / f"{node}.node.json").write_text(json.dumps(value))
    result = roce.validate_pair(tmp_path, nodes, "pair04")
    assert result["pair_constraint"] == "pair04"
    assert [row["network"] for row in result["rails"]] == [
        "10.250.4.0/30",
        "10.250.104.0/30",
    ]
    with pytest.raises(roce.PreflightError):
        roce.validate_pair(tmp_path / "other", nodes, "pair05")


def _benchmark(profile, gradient_seconds):
    rows = []
    for index, (label, elements, warmups, timed) in enumerate(roce.PAYLOADS):
        seconds = gradient_seconds * (index + 1)
        samples = [seconds * (1.0 + 0.01 * offset) for offset in range(timed)]
        rows.append(
            {
                "label": label,
                "elements": elements,
                "dtype": "float32",
                "logical_payload_bytes": elements * 4,
                "logical_payload_mib": elements * 4 / 1024**2,
                "warmup_iterations": warmups,
                "timed_iterations": timed,
                "maximum_rank_seconds": samples,
                "median_maximum_rank_seconds": samples[len(samples) // 2],
                "algorithmic_bandwidth_gbit_s": elements * 32 / seconds / 1e9,
                "exact_sum_expected": 3.0,
                "exact_sum_passed": True,
            }
        )
    return {
        "schema_name": roce.SCHEMA,
        "schema_version": roce.SCHEMA_VERSION,
        "profile": profile,
        "backend": "nccl",
        "world_size": 2,
        "hosts": ["node-a", "node-b"],
        "ethernet_fallback_permitted": False,
        "environment": {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": roce.NCCL_PROFILES[profile],
            "NCCL_IB_GID_INDEX": "5",
            "NCCL_SOCKET_IFNAME": "enp1s0f1np1,enP2p1s0f1np1",
            "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
        },
        "benchmarks": rows,
    }


@pytest.mark.parametrize(
    ("dual_time", "expected_profile"),
    ((0.0104, "dual"), (0.0110, "single_primary")),
)
def test_timed_payload_comparison_selects_dual_only_within_five_percent(
    tmp_path, dual_time, expected_profile
):
    timings = {
        "single_primary": 0.0100,
        "single_secondary": 0.0102,
        "dual": dual_time,
    }
    for profile, seconds in timings.items():
        (tmp_path / f"NCCL_BENCHMARK_{profile}.json").write_text(
            json.dumps(_benchmark(profile, seconds)), encoding="utf-8"
        )
    result = roce.compare_benchmarks(tmp_path)
    assert roce.PAYLOADS[0][1] == 1_109_395
    assert result["comparisons"][0]["logical_payload_bytes"] == 4_437_580
    assert {row["logical_payload_bytes"] for row in result["comparisons"]} == {
        4_437_580,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    }
    assert result["training_profile"] == expected_profile
    assert result["training_hcas"] == roce.NCCL_PROFILES[expected_profile]


def _counter_snapshot(node, phase, traffic_increment=0, harmful_increment=0):
    rails = {}
    for hca, netdev in roce.RAILS:
        network = {name: 0 for name in roce.NETWORK_COUNTERS}
        verbs = {name: 0 for name in roce.HCA_COUNTERS}
        ethtool = {name: 0 for name in roce.ETHTOOL_HARMFUL_COUNTERS}
        if phase == "post":
            network["rx_bytes"] = traffic_increment
            network["rx_errors"] = harmful_increment
        rails[hca] = {
            "netdev": netdev,
            "network": network,
            "verbs": verbs,
            "ethtool": ethtool,
        }
    return {
        "schema_name": roce.SCHEMA,
        "schema_version": roce.SCHEMA_VERSION,
        "host": node,
        "phase": phase,
        "rails": rails,
    }


def test_counter_comparison_allows_traffic_and_rejects_harmful_increment(tmp_path):
    nodes = ["node-a", "node-b"]
    for node in nodes:
        for phase in ("pre", "post"):
            value = _counter_snapshot(
                node, phase, traffic_increment=100 if phase == "post" else 0
            )
            (tmp_path / f"{node}.counters_{phase}.json").write_text(json.dumps(value))
    result = roce.compare_counters(tmp_path, nodes)
    assert result["harmful_counter_increment_count"] == 0
    assert any(row["counter"] == "rx_bytes" and row["delta"] == 100 for row in result["traffic_counters"])

    broken = tmp_path / "broken"
    broken.mkdir()
    for node in nodes:
        for phase in ("pre", "post"):
            value = _counter_snapshot(
                node, phase, harmful_increment=1 if phase == "post" else 0
            )
            (broken / f"{node}.counters_{phase}.json").write_text(json.dumps(value))
    with pytest.raises(roce.PreflightError, match="harmful RoCE counter"):
        roce.compare_counters(broken, nodes)


def test_nccl_log_audit_requires_positive_ib_evidence_in_each_rank(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    for process_id, host in enumerate(("node-a", "node-b"), start=100):
        (good / f"rank.{host}.{process_id}.log").write_text(
            "NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE\n",
            encoding="utf-8",
        )
    result = roce.verify_nccl_logs(
        good, "rank.", "single_primary", good / "AUDIT.json"
    )
    assert len(result["per_log_evidence"]) == 2
    assert all(row["net_ib_line_count"] == 1 for row in result["per_log_evidence"])

    empty_rank = tmp_path / "empty_rank"
    empty_rank.mkdir()
    (empty_rank / "rank.node-a.100.log").write_text(
        "NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE\n",
        encoding="utf-8",
    )
    (empty_rank / "rank.node-b.101.log").write_text("", encoding="utf-8")
    with pytest.raises(roce.PreflightError, match="no positive NET/IB"):
        roce.verify_nccl_logs(
            empty_rank, "rank.", "single_primary", empty_rank / "AUDIT.json"
        )

    missing_rank = tmp_path / "missing_rank"
    missing_rank.mkdir()
    (missing_rank / "rank.node-a.100.log").write_text(
        "NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE\n",
        encoding="utf-8",
    )
    with pytest.raises(roce.PreflightError, match="exactly two regular NCCL logs"):
        roce.verify_nccl_logs(
            missing_rank, "rank.", "single_primary", missing_rank / "AUDIT.json"
        )


def _foundation_gate(tmp_path, monkeypatch):
    head = "a" * 40
    remote_ref = "origin/stage-c"
    cpu_gate = tmp_path / "cpu" / "READY"
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    for name, content in (
        ("calibration.json", b"calibration"),
        ("calibration_map.npz", b"map"),
        ("READY", b"ready"),
    ):
        (calibration_dir / name).write_bytes(content)
    launcher = tmp_path / "calibrate.sh"
    launcher.write_text("#!/bin/bash\n", encoding="utf-8")
    source_decision = tmp_path / "source_decision.json"
    source_decision.write_text("{}\n", encoding="utf-8")
    stage_c_config = tmp_path / "stage_c.yaml"
    stage_c_config.write_text("stage_c: true\n", encoding="utf-8")
    cpu = {
        "schema_name": "signtrajfield_stage_c_cpu_gate",
        "schema_version": 1,
        "source_git_head": head,
        "source_remote_ref": remote_ref,
        "source_remote_head": head,
        "slurm_job_id": "101",
        "compileall": True,
        "ruff_version": "0.12.0",
        "pytest_version": "8.4.2",
        "complete_repository_test_glob": "tests/test_*.py",
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "passed": True,
    }
    _write_json(cpu_gate, cpu)
    monkeypatch.setattr(
        prerequisites,
        "validate_relevance_calibration_artifact",
        lambda _path: {"identity": "b" * 64},
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_relevance_calibration_source",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_source_terminal_decision",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_recovery_evidence",
        lambda **_kwargs: {
            "audit_identity": "e" * 64,
            "recovery_manifest_sha256": "f" * 64,
        },
    )
    transition = {
        "schema_name": "signtrajfield_stage_c_calibration_transition_audit",
        "schema_version": 1,
        "semantic_equivalence": {"all_non_provenance_fields_exact": True},
        "train_only": True,
        "development_only": True,
        "audit_identity": "d" * 64,
    }
    monkeypatch.setattr(
        prerequisites,
        "build_transition_audit",
        lambda **_kwargs: transition,
    )
    transition_path = tmp_path / "calibration_transition.json"
    _write_json(transition_path, transition)
    binding_inputs = {}
    for name in ("audit_config", "bank_manifest", "bank_ready", "train_neighbors"):
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        binding_inputs[name] = path
    binding = calibration_control.calibration_binding(
        source_git_head=head,
        source_remote_ref=remote_ref,
        source_remote_head=head,
        launcher=launcher,
        cpu_gate=cpu_gate,
        **binding_inputs,
    )
    claim_unsigned = {
        "schema_name": "signtrajfield_stage_c_calibration_lease",
        "schema_version": 1,
        "slurm_job_id": "101",
        "slurm_restart_count": 0,
        "binding": binding,
        "claim_nonce": "1" * 32,
        "created_unix_ns": 1,
    }
    claim = {
        **claim_unsigned,
        "claim_identity": prerequisites.digest_json(claim_unsigned),
    }
    completion_path = tmp_path / "calibration" / "COMPLETE.json"
    lease_path = completion_path.parent / "active_execution_lease"
    attestation_unsigned = {
        "schema_name": "signtrajfield_stage_c_calibration_lease_attestation",
        "schema_version": 1,
        "lease_path": str(lease_path.resolve()),
        "current_slurm_job_id": "101",
        "current_slurm_restart_count": 0,
        "claim": claim,
        "restart_reconciliation": None,
    }
    attestation = {
        **attestation_unsigned,
        "attestation_identity": prerequisites.digest_json(attestation_unsigned),
    }
    attestation_path = completion_path.parent / "lease_attestations/101.0.json"
    _write_json(attestation_path, attestation)
    completion = {
        "schema_name": "signtrajfield_stage_c_calibration_completion",
        "schema_version": 1,
        "source_git_head": head,
        "source_remote_ref": remote_ref,
        "source_remote_head": head,
        "artifact_path": str(calibration_dir.resolve()),
        "artifact_identity": "b" * 64,
        "calibration_sha256": prerequisites.sha256_file(
            calibration_dir / "calibration.json"
        ),
        "map_sha256": prerequisites.sha256_file(
            calibration_dir / "calibration_map.npz"
        ),
        "ready_sha256": prerequisites.sha256_file(calibration_dir / "READY"),
        "cpu_gate_sha256": prerequisites.sha256_file(cpu_gate),
        "launcher_sha256": prerequisites.sha256_file(launcher),
        "calibration_transition_path": str(transition_path.resolve()),
        "calibration_transition_sha256": prerequisites.sha256_file(transition_path),
        "calibration_transition_identity": transition["audit_identity"],
        "active_lease_claim_identity": claim["claim_identity"],
        "lease_attestation_path": str(attestation_path.resolve()),
        "lease_attestation_sha256": prerequisites.sha256_file(attestation_path),
        "lease_attestation_identity": attestation["attestation_identity"],
        "train_only": True,
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    _write_json(completion_path, completion)
    terminal_unsigned = {
        "schema_name": "signtrajfield_stage_c_calibration_lease_terminal",
        "schema_version": 1,
        "outcome": "released",
        "claim_identity": claim["claim_identity"],
        "completion_path": str(completion_path.resolve()),
        "completion_sha256": prerequisites.sha256_file(completion_path),
    }
    terminal = {
        **terminal_unsigned,
        "terminal_identity": prerequisites.digest_json(terminal_unsigned),
    }
    terminal_dir = completion_path.parent / (
        f"active_execution_lease.history/{claim['claim_identity']}.released"
    )
    _write_json(terminal_dir / "TERMINAL.json", terminal)
    kwargs = {
        "recovery_policy": tmp_path / "retry2_policy.json",
        "recovery_manifest": tmp_path / "retry2_evidence.json",
        "recovery_evidence_root": tmp_path,
        "cpu_gate": cpu_gate,
        "calibration_completion": completion_path,
        "calibration_dir": calibration_dir,
        "calibration_launcher": launcher,
        "stage_c_config": stage_c_config,
        "source_terminal_decision": source_decision,
        "source_root": tmp_path,
        "source_git_head": head,
        "source_remote_ref": remote_ref,
        "source_remote_head": head,
    }
    return kwargs, cpu, completion


def _restore_active_calibration_claim(kwargs, completion):
    completion_path = Path(kwargs["calibration_completion"])
    attestation_path = Path(completion["lease_attestation_path"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    claim = attestation["claim"]
    lease = completion_path.parent / "active_execution_lease"
    archived = completion_path.parent / (
        "active_execution_lease.history/"
        f"{claim['claim_identity']}.released"
    )
    _write_json(archived / "owner.json", claim)
    _write_json(
        archived / "READY",
        {
            "schema_name": "signtrajfield_stage_c_calibration_lease",
            "schema_version": 1,
            "claim_identity": claim["claim_identity"],
        },
    )
    archived.rename(lease)
    return lease, attestation_path, claim


def test_foundation_gate_binds_all_evidence_and_rejects_tampering(
    tmp_path, monkeypatch
):
    kwargs, cpu, completion = _foundation_gate(tmp_path, monkeypatch)
    validated = prerequisites.validate_foundation(**kwargs)
    assert validated["calibration_identity"] == "b" * 64
    assert validated["recovery_evidence_audit_identity"] == "e" * 64
    assert validated["recovery_evidence_manifest_sha256"] == "f" * 64

    cpu["test_data_accessed"] = True
    _write_json(kwargs["cpu_gate"], cpu)
    with pytest.raises(prerequisites.PrerequisiteError, match="CPU gate"):
        prerequisites.validate_foundation(**kwargs)

    cpu["test_data_accessed"] = False
    _write_json(kwargs["cpu_gate"], cpu)
    completion["cpu_gate_sha256"] = prerequisites.sha256_file(kwargs["cpu_gate"])
    completion["map_sha256"] = "0" * 64
    _write_json(kwargs["calibration_completion"], completion)
    with pytest.raises(prerequisites.PrerequisiteError, match="binding"):
        prerequisites.validate_foundation(**kwargs)


def test_calibration_same_job_requeue_finalizes_released_terminal(
    tmp_path, monkeypatch
):
    kwargs, _cpu, completion = _foundation_gate(tmp_path, monkeypatch)
    lease, _attestation, claim = _restore_active_calibration_claim(
        kwargs, completion
    )
    monkeypatch.setattr(
        calibration_control,
        "_scheduler",
        lambda job_id: ("RUNNING", 1) if job_id == "101" else ("FAILED", 0),
    )
    result = calibration_control.reconcile_completed(
        lease=lease,
        completion=Path(kwargs["calibration_completion"]),
        current_job_id="101",
        current_restart=1,
    )
    assert result["terminal"]["outcome"] == "released"
    assert not lease.exists()
    assert (
        lease.with_name(f"{lease.name}.history")
        / f"{claim['claim_identity']}.released/TERMINAL.json"
    ).is_file()


def _smoke_gate(tmp_path):
    head = "c" * 40
    pair = "pair04"
    smoke_root = tmp_path / "smoke"
    attempt = smoke_root / f"sources/source_{head}/attempts/202/executions/restart_0"
    network = attempt / "network_preflight"
    network.mkdir(parents=True)
    binding_files = {}
    for name in (
        "memory_config",
        "matched_off_config",
        "cpu_gate",
        "calibration_completion",
    ):
        path = tmp_path / "binding" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        binding_files[name] = path
    binding = execution_control.execution_binding(
        mode="smoke",
        source_git_head=head,
        source_remote_ref="origin/stage-c",
        source_remote_head=head,
        pair_constraint=pair,
        **binding_files,
    )
    claim_unsigned = {
        "schema_name": "signtrajfield_stage_c_execution_lease",
        "schema_version": 1,
        "mode": "smoke",
        "slurm_job_id": "202",
        "slurm_restart_count": 0,
        "execution_binding": binding,
        "claim_nonce": "2" * 32,
        "created_unix_ns": 2,
        "replaced_stale_owner": None,
        "prior_claim_identities": [],
    }
    claim = {
        **claim_unsigned,
        "claim_identity": prerequisites.digest_json(claim_unsigned),
    }
    control = tmp_path / "prerequisites/smoke"
    lease_path = control / "active_execution_lease"
    attestation_unsigned = {
        "schema_name": "signtrajfield_stage_c_execution_lease_attestation",
        "schema_version": 1,
        "lease_path": str(lease_path.resolve()),
        "current_slurm_job_id": "202",
        "current_slurm_restart_count": 0,
        "claim": claim,
        "restart_reconciliation": None,
    }
    attestation = {
        **attestation_unsigned,
        "attestation_identity": prerequisites.digest_json(attestation_unsigned),
    }
    attestation_path = control / "lease_attestations/202.0.json"
    _write_json(attestation_path, attestation)
    policy_path = (
        CONFIG_DIR
        / "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
    )
    bound_files = {**binding_files, "decision_policy": policy_path}
    for index in range(22):
        path = tmp_path / "bound" / f"source_{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# source {index}\n", encoding="utf-8")
        bound_files[f"source_{index}"] = path
    source_binding = {
        "schema_name": "signtrajfield_stage_c_source_binding",
        "schema_version": 1,
        "source_git_head": head,
        "source_remote_ref": "origin/stage-c",
        "source_remote_head": head,
        "files": {
            str(path.resolve()): prerequisites.sha256_file(path)
            for path in bound_files.values()
        },
        "development_only": True,
        "non_authorizing": True,
    }
    _write_json(network / "SOURCE_BINDING.json", source_binding)

    arm_names = (
        "config.resolved.json",
        "metrics.jsonl",
        "selection_summary.json",
        "checkpoints/last.pt",
        "checkpoints/epoch0001.pt",
    )
    arms = {}
    for arm in ("memory", "matched_off"):
        directory = attempt / arm
        for name in arm_names:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{arm}:{name}".encode())
        arms[arm] = {
            name: prerequisites.sha256_file(directory / name) for name in arm_names
        }

    launch = {
        "schema_name": "signtrajfield_centered_stage_c_paired_pilot_launch",
        "schema_version": 1,
        "source_git_head": head,
        "source_remote_ref": "origin/stage-c",
        "source_remote_head": head,
        "pair_constraint": pair,
        "nodes": "node-a,node-b",
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
        "execution_mode": "smoke",
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "confirmation_or_test_access_permitted": False,
        "training_network_profile": "dual",
        "training_nccl_ib_hca": roce.NCCL_PROFILES["dual"],
        "active_lease_claim_identity": claim["claim_identity"],
        "lease_attestation": {
            "path": str(attestation_path.resolve()),
            "sha256": prerequisites.sha256_file(attestation_path),
        },
        "artifacts": {
            "memory_config": {
                "path": str(binding_files["memory_config"].resolve()),
                "sha256": binding["memory_config_sha256"],
            },
            "matched_off_config": {
                "path": str(binding_files["matched_off_config"].resolve()),
                "sha256": binding["matched_off_config_sha256"],
            },
            "cpu_gate": {
                "path": str(binding_files["cpu_gate"].resolve()),
                "sha256": binding["cpu_gate_sha256"],
            },
            "calibration_completion": {
                "path": str(binding_files["calibration_completion"].resolve()),
                "sha256": binding["calibration_completion_sha256"],
            },
            "source_binding_manifest": {
                "path": str((network / "SOURCE_BINDING.json").resolve()),
                "sha256": prerequisites.sha256_file(
                    network / "SOURCE_BINDING.json"
                ),
            },
        },
    }
    _write_json(attempt / "LAUNCH.json", launch)
    nodes = ["node-a", "node-b"]
    node_values = (
        _node_value(nodes[0], "10.250.4.1", "10.250.104.1"),
        _node_value(nodes[1], "10.250.4.2", "10.250.104.2"),
    )
    for node, value in zip(nodes, node_values, strict=True):
        _write_json(network / f"{node}.node.json", value)
    pair_evidence = roce.validate_pair(network, nodes, pair)
    for host in nodes:
        peer = next(node for node in nodes if node != host)
        _write_json(
            network / f"{host}.connectivity.json",
            {
                "schema_name": roce.SCHEMA,
                "schema_version": roce.SCHEMA_VERSION,
                "host": host,
                "pair_constraint": pair,
                "peer_connectivity": [
                    {
                        "netdev": rail["netdev"],
                        "peer": peer,
                        "peer_ipv4": rail["endpoints"][peer],
                        "passed": True,
                    }
                    for rail in pair_evidence["rails"]
                ],
            },
        )
        for phase in ("pre", "post"):
            _write_json(
                network / f"{host}.counters_{phase}.json",
                _counter_snapshot(host, phase),
            )
    roce.compare_counters(network, nodes)
    _write_json(
        network / "SMOKE_CHECKPOINT_AUDIT.json",
        {
            "schema_name": "signtrajfield_stage_c_one_update_checkpoint_audit",
            "schema_version": 1,
            "execution_mode": "smoke",
            "expected_global_step": 1,
            "exact_model_tensor_count": 233,
            "exact_trainable_tensor_count": 20,
            "exact_changed_trainable_tensor_count": 20,
            "exact_trainable_parameter_count": 1_109_395,
            "exact_frozen_model_tensor_count": 213,
            "exact_frozen_nonmemory_base_tensor_count": 151,
            "exact_frozen_sentence_memory_tensor_count": 62,
            "all_model_and_optimizer_tensors_finite": True,
            "all_sentence_memory_and_nonapproved_tensors_bitwise_frozen": True,
            "all_trainables_have_nonzero_finite_one_step_moments": True,
            "all_trainables_have_nonzero_finite_optimizer_moments": True,
            "same_source_provenance_except_arm": True,
            "same_calibration_distribution_architecture_memory": True,
            "same_objective_except_training_memory_availability": True,
            "canonical_best_checkpoint_published": False,
            "development_only": True,
            "non_authorizing": True,
        },
    )
    log_profiles = {
        "train_memory": ("dual", "nccl.train.memory.dual."),
        "train_matched_off": ("dual", "nccl.train.matched_off.dual."),
        "single_primary": ("single_primary", "nccl.single_primary."),
        "single_secondary": ("single_secondary", "nccl.single_secondary."),
        "dual": ("dual", "nccl.dual."),
    }
    for name, (profile, prefix) in log_profiles.items():
        expected_hcas = [
            value.split(":", 1)[0]
            for value in roce.NCCL_PROFILES[profile].split(",")
        ]
        for process_id, host in enumerate(nodes, start=100):
            (network / f"{prefix}{host}.{process_id}.log").write_text(
                f"NCCL INFO NET/IB : Using {' '.join(expected_hcas)}\n",
                encoding="utf-8",
            )
        roce.verify_nccl_logs(
            network,
            prefix,
            profile,
            network / f"NCCL_LOGS_{name}.json",
        )
    for profile, seconds in (
        ("single_primary", 0.0100),
        ("single_secondary", 0.0102),
        ("dual", 0.0104),
    ):
        _write_json(
            network / f"NCCL_BENCHMARK_{profile}.json",
            _benchmark(profile, seconds),
        )
    roce.compare_benchmarks(network)

    network_files = {
        "launch": "LAUNCH.json",
        "source_binding": "network_preflight/SOURCE_BINDING.json",
        "pair": "network_preflight/PAIR.json",
        "counter_health": "network_preflight/COUNTER_HEALTH.json",
        "dual_vs_single": "network_preflight/NCCL_DUAL_VS_SINGLE.json",
        "train_memory_logs": "network_preflight/NCCL_LOGS_train_memory.json",
        "train_matched_off_logs": (
            "network_preflight/NCCL_LOGS_train_matched_off.json"
        ),
        "smoke_checkpoint_audit": (
            "network_preflight/SMOKE_CHECKPOINT_AUDIT.json"
        ),
    }
    for profile in ("single_primary", "single_secondary", "dual"):
        network_files[f"benchmark_{profile}"] = (
            f"network_preflight/NCCL_BENCHMARK_{profile}.json"
        )
        network_files[f"benchmark_logs_{profile}"] = (
            f"network_preflight/NCCL_LOGS_{profile}.json"
        )
    for index, node in enumerate(nodes):
        network_files[f"node_audit_{index}"] = (
            f"network_preflight/{node}.node.json"
        )
        network_files[f"connectivity_audit_{index}"] = (
            f"network_preflight/{node}.connectivity.json"
        )
        for phase in ("pre", "post"):
            network_files[f"counter_{phase}_{index}"] = (
                f"network_preflight/{node}.counters_{phase}.json"
            )
    complete = {
        "schema_name": "signtrajfield_centered_stage_c_paired_execution_complete",
        "schema_version": 1,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "execution_mode": "smoke",
        "source_git_head": head,
        "source_remote_ref": "origin/stage-c",
        "source_remote_head": head,
        "pair_constraint": pair,
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "active_lease_claim_identity": claim["claim_identity"],
        "lease_attestation_path": str(attestation_path.resolve()),
        "lease_attestation_sha256": prerequisites.sha256_file(attestation_path),
        "lease_attestation_identity": attestation["attestation_identity"],
        "decision_policy": {
            "path": str(policy_path.resolve()),
            "sha256": prerequisites.sha256_file(policy_path),
        },
        "arms": arms,
        "arm_execution_summaries": {
            arm: {
                "epoch": 1,
                "global_step": 1,
                "data_limit_train": 0,
                "data_limit_val": 0,
                "train_max_batches": 2,
                "eval_max_batches": 1,
                "length_bucketed_batches": True,
                "drop_last": False,
            }
            for arm in ("memory", "matched_off")
        },
        "execution_artifacts": {
            key: prerequisites.sha256_file(attempt / path)
            for key, path in network_files.items()
        },
    }
    complete_path = attempt / "COMPLETE.json"
    _write_json(complete_path, complete)
    decision_unsigned = {
        "schema_name": "signtrajfield_stage_c_generator_adaptation_decision",
        "schema_version": 1,
        "execution_mode": "smoke",
        "status": "smoke_ready",
        "next_permitted_action": "run_one_epoch_development_pilot",
        "source_git_head": head,
        "pair_constraint": pair,
        "active_lease_claim_identity": claim["claim_identity"],
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "authorized_purpose": None,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    decision = {
        **decision_unsigned,
        "decision_identity": prerequisites.digest_json(decision_unsigned),
    }
    decision_path = attempt / "DECISION.json"
    _write_json(decision_path, decision)
    publication = control / "PUBLICATION"
    mode_complete = {
        "schema_name": "signtrajfield_stage_c_mode_complete",
        "schema_version": 1,
        "execution_mode": "smoke",
        "source_git_head": head,
        "pair_constraint": pair,
        "slurm_job_id": "202",
        "slurm_restart_count": 0,
        "active_lease_claim_identity": claim["claim_identity"],
        "execution_lease_claim_identity": claim["claim_identity"],
        "lease_attestation_path": str(attestation_path.resolve()),
        "lease_attestation_sha256": prerequisites.sha256_file(attestation_path),
        "lease_attestation_identity": attestation["attestation_identity"],
        "execution_complete_path": str(complete_path.resolve()),
        "execution_complete_sha256": prerequisites.sha256_file(complete_path),
        "decision_path": str(decision_path.resolve()),
        "decision_sha256": prerequisites.sha256_file(decision_path),
        "decision_identity": decision["decision_identity"],
        "decision_status": "smoke_ready",
        "decision_policy_path": str(policy_path.resolve()),
        "decision_policy_sha256": prerequisites.sha256_file(policy_path),
        "expected_epoch": 1,
        "expected_global_step_per_arm": 1,
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "authorized_purpose": None,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    mode_complete_path = publication / "COMPLETE.json"
    _write_json(mode_complete_path, mode_complete)
    ready = {
        "schema_name": "signtrajfield_stage_c_mode_ready",
        "schema_version": 1,
        "execution_mode": "smoke",
        "source_git_head": head,
        "pair_constraint": pair,
        "active_lease_claim_identity": claim["claim_identity"],
        "execution_lease_claim_identity": claim["claim_identity"],
        "complete_path": str(mode_complete_path.resolve()),
        "complete_sha256": prerequisites.sha256_file(mode_complete_path),
        "decision_path": str(decision_path.resolve()),
        "decision_sha256": prerequisites.sha256_file(decision_path),
        "decision_identity": decision["decision_identity"],
        "decision_status": "smoke_ready",
        "expected_epoch": 1,
        "expected_global_step_per_arm": 1,
        "one_optimizer_update_per_arm": True,
        "one_full_train_epoch_per_arm": False,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "authorized_purpose": None,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    ready_path = publication / "READY"
    _write_json(ready_path, ready)
    terminal_unsigned = {
        "schema_name": "signtrajfield_stage_c_execution_lease_terminal",
        "schema_version": 1,
        "outcome": "released",
        "claim_identity": claim["claim_identity"],
        "completion_path": str(mode_complete_path.resolve()),
        "completion_sha256": prerequisites.sha256_file(mode_complete_path),
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": prerequisites.sha256_file(ready_path),
    }
    terminal = {
        **terminal_unsigned,
        "terminal_identity": prerequisites.digest_json(terminal_unsigned),
    }
    terminal_path = control / (
        "active_execution_lease.history/"
        f"{claim['claim_identity']}.released/TERMINAL.json"
    )
    _write_json(terminal_path.parent / "owner.json", claim)
    _write_json(
        terminal_path.parent / "READY",
        {
            "schema_name": "signtrajfield_stage_c_execution_lease",
            "schema_version": 1,
            "claim_identity": claim["claim_identity"],
        },
    )
    _write_json(terminal_path, terminal)
    return {
        "smoke_ready": ready_path,
        "smoke_root": smoke_root,
        "source_git_head": head,
        "pair_constraint": pair,
    }, ready, complete, complete_path


def _restore_active_execution_claim(kwargs, ready):
    ready_path = Path(kwargs["smoke_ready"])
    mode_complete = json.loads(
        Path(ready["complete_path"]).read_text(encoding="utf-8")
    )
    attestation_path = Path(mode_complete["lease_attestation_path"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    claim = attestation["claim"]
    control = ready_path.parent.parent
    lease = control / "active_execution_lease"
    archived = control / (
        "active_execution_lease.history/"
        f"{claim['claim_identity']}.released"
    )
    _write_json(archived / "owner.json", claim)
    _write_json(
        archived / "READY",
        {
            "schema_name": "signtrajfield_stage_c_execution_lease",
            "schema_version": 1,
            "claim_identity": claim["claim_identity"],
        },
    )
    archived.rename(lease)
    return lease, attestation_path, claim


def test_smoke_gate_binds_complete_scope_and_rejects_tampering(tmp_path):
    kwargs, ready, complete, complete_path = _smoke_gate(tmp_path)
    mode_complete_path = Path(ready["complete_path"])
    mode_complete = json.loads(mode_complete_path.read_text(encoding="utf-8"))
    smoke_binding = prerequisites.validate_smoke(**kwargs)
    launch = json.loads(
        (complete_path.parent / "LAUNCH.json").read_text(encoding="utf-8")
    )
    assert smoke_binding["schema_name"] == (
        "signtrajfield_stage_c_prior_one_update_smoke_prerequisite"
    )
    assert smoke_binding["ready_path"] == str(Path(kwargs["smoke_ready"]).resolve())
    assert smoke_binding["ready_sha256"] == prerequisites.sha256_file(
        Path(kwargs["smoke_ready"])
    )
    assert smoke_binding["decision_identity"] == ready["decision_identity"]
    assert smoke_binding["decision_status"] == "smoke_ready"
    assert smoke_binding["source_git_head"] == kwargs["source_git_head"]
    assert smoke_binding["pair_constraint"] == kwargs["pair_constraint"]
    assert smoke_binding["memory_config_sha256"] == launch["artifacts"][
        "memory_config"
    ]["sha256"]
    assert smoke_binding["cpu_gate_sha256"] == launch["artifacts"]["cpu_gate"][
        "sha256"
    ]
    assert smoke_binding["training_network_profile"] == "dual"

    ready["one_optimizer_update_per_arm"] = False
    _write_json(kwargs["smoke_ready"], ready)
    with pytest.raises(prerequisites.PrerequisiteError, match="READY"):
        prerequisites.validate_smoke(**kwargs)

    ready["one_optimizer_update_per_arm"] = True
    complete["test_data_accessed"] = True
    _write_json(complete_path, complete)
    mode_complete["execution_complete_sha256"] = prerequisites.sha256_file(
        complete_path
    )
    _write_json(mode_complete_path, mode_complete)
    ready["complete_sha256"] = prerequisites.sha256_file(mode_complete_path)
    _write_json(kwargs["smoke_ready"], ready)
    with pytest.raises(prerequisites.PrerequisiteError, match="scope"):
        prerequisites.validate_smoke(**kwargs)

    complete["test_data_accessed"] = False
    complete["arms"]["memory"]["metrics.jsonl"] = "0" * 64
    _write_json(complete_path, complete)
    mode_complete["execution_complete_sha256"] = prerequisites.sha256_file(
        complete_path
    )
    _write_json(mode_complete_path, mode_complete)
    ready["complete_sha256"] = prerequisites.sha256_file(mode_complete_path)
    _write_json(kwargs["smoke_ready"], ready)
    with pytest.raises(prerequisites.PrerequisiteError, match="artifact hash"):
        prerequisites.validate_smoke(**kwargs)


def test_finalized_publication_requeue_validates_without_active_lease(tmp_path):
    kwargs, ready, complete, complete_path = _smoke_gate(tmp_path)
    control = Path(kwargs["smoke_ready"]).parent.parent
    lease = control / "active_execution_lease"
    result = execution_control.validate_finalized_publication(
        lease_path=lease,
        completion_path=Path(ready["complete_path"]),
        ready_path=Path(kwargs["smoke_ready"]),
        expected_source_git_head=kwargs["source_git_head"],
        expected_pair_constraint=kwargs["pair_constraint"],
    )
    assert result["decision_status"] == "smoke_ready"
    mode_complete = json.loads(Path(ready["complete_path"]).read_text(encoding="utf-8"))
    stage_c_pilot_decision._validate_current_execution_scope(
        complete=complete,
        execution_complete=complete_path,
        active_lease_attestation=Path(mode_complete["lease_attestation_path"]),
        expected_source_git_head=kwargs["source_git_head"],
        expected_source_remote_ref="origin/stage-c",
        expected_source_remote_head=kwargs["source_git_head"],
        expected_pair_constraint=kwargs["pair_constraint"],
    )
    with pytest.raises(
        execution_control.StageCExecutionControlError, match="source/pair"
    ):
        execution_control.validate_finalized_publication(
            lease_path=lease,
            completion_path=Path(ready["complete_path"]),
            ready_path=Path(kwargs["smoke_ready"]),
            expected_source_git_head=kwargs["source_git_head"],
            expected_pair_constraint="pair05",
        )
    runner = _script("run_csl_daily_stage_c_generator_adaptation_pilot.sh")
    assert "validate-finalized" in runner
    assert "STAGE_C_VALIDATED_FINALIZED_${EXECUTION_MODE^^}" in runner
    assert "exit 0" in runner and "exit 2" in runner


def test_execution_same_job_requeue_finalizes_released_terminal(
    tmp_path, monkeypatch
):
    kwargs, ready, _complete, _complete_path = _smoke_gate(tmp_path)
    lease, _attestation, claim = _restore_active_execution_claim(kwargs, ready)
    monkeypatch.setattr(
        execution_control,
        "_scheduler_job",
        lambda job_id: (
            ("RUNNING", 1, "JobState=RUNNING Restarts=1")
            if job_id == "202"
            else ("FAILED", 0, "JobState=FAILED Restarts=0")
        ),
    )
    result = execution_control.reconcile_published_execution(
        lease_path=lease,
        completion_path=Path(ready["complete_path"]),
        ready_path=Path(kwargs["smoke_ready"]),
        current_slurm_job_id="202",
        current_slurm_restart_count=1,
    )
    assert result["outcome"] == "released"
    assert not lease.exists()
    assert (
        lease.with_name(f"{lease.name}.history")
        / f"{claim['claim_identity']}.released/TERMINAL.json"
    ).is_file()


def test_reusable_completion_requires_current_source_and_pair(tmp_path):
    kwargs, ready, complete, complete_path = _smoke_gate(tmp_path)
    _lease, attestation, _claim = _restore_active_execution_claim(kwargs, ready)
    stage_c_pilot_decision._validate_current_execution_scope(
        complete=complete,
        execution_complete=complete_path,
        active_lease_attestation=attestation,
        expected_source_git_head=kwargs["source_git_head"],
        expected_source_remote_ref="origin/stage-c",
        expected_source_remote_head=kwargs["source_git_head"],
        expected_pair_constraint=kwargs["pair_constraint"],
    )
    with pytest.raises(stage_c_pilot_decision.StageCDecisionError, match="source/pair"):
        stage_c_pilot_decision._validate_current_execution_scope(
            complete=complete,
            execution_complete=complete_path,
            active_lease_attestation=attestation,
            expected_source_git_head="d" * 40,
            expected_source_remote_ref="origin/stage-c",
            expected_source_remote_head="d" * 40,
            expected_pair_constraint=kwargs["pair_constraint"],
        )
    with pytest.raises(stage_c_pilot_decision.StageCDecisionError, match="source/pair"):
        stage_c_pilot_decision._validate_current_execution_scope(
            complete=complete,
            execution_complete=complete_path,
            active_lease_attestation=attestation,
            expected_source_git_head=kwargs["source_git_head"],
            expected_source_remote_ref="origin/stage-c",
            expected_source_remote_head=kwargs["source_git_head"],
            expected_pair_constraint="pair05",
        )


def test_decision_reopens_every_roce_artifact_and_rank_log(tmp_path):
    _kwargs, _ready, complete, complete_path = _smoke_gate(tmp_path / "valid")
    result = stage_c_pilot_decision._validate_network_evidence(
        complete, complete_path, "smoke"
    )
    assert result["training_profile"] == "dual"
    assert result["all_five_log_audits_and_ten_rank_logs_valid"] is True

    _kwargs, _ready, tampered, tampered_path = _smoke_gate(tmp_path / "tampered")
    pair_path = tampered_path.parent / "network_preflight/PAIR.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    pair["pair_constraint"] = "pair05"
    _write_json(pair_path, pair)
    with pytest.raises(stage_c_pilot_decision.StageCDecisionError, match="hash"):
        stage_c_pilot_decision._validate_network_evidence(
            tampered, tampered_path, "smoke"
        )

    _kwargs, _ready, missing, missing_path = _smoke_gate(tmp_path / "missing")
    raw_log = missing_path.parent / "network_preflight/nccl.dual.node-a.100.log"
    raw_log.unlink()
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError, match="raw NCCL rank log"
    ):
        stage_c_pilot_decision._validate_network_evidence(
            missing, missing_path, "smoke"
        )

    _kwargs, _ready, same_host, same_host_path = _smoke_gate(
        tmp_path / "same_host"
    )
    audit_path = same_host_path.parent / "network_preflight/NCCL_LOGS_dual.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    old_log = Path(audit["logs"][1])
    replacement = old_log.with_name("nccl.dual.node-a.101.log")
    old_log.rename(replacement)
    audit["logs"][1] = str(replacement.resolve())
    audit["per_log_evidence"][1]["path"] = str(replacement.resolve())
    audit["per_log_evidence"][1]["sha256"] = prerequisites.sha256_file(
        replacement
    )
    audit["per_log_evidence"][1]["host"] = "node-a"
    _write_json(audit_path, audit)
    same_host["execution_artifacts"]["benchmark_logs_dual"] = (
        prerequisites.sha256_file(audit_path)
    )
    _write_json(same_host_path, same_host)
    with pytest.raises(
        stage_c_pilot_decision.StageCDecisionError, match="both exact pair ranks"
    ):
        stage_c_pilot_decision._validate_network_evidence(
            same_host, same_host_path, "smoke"
        )


@pytest.mark.parametrize(
    "case",
    (
        "missing_raw_artifact",
        "node_speed",
        "peer_connectivity",
        "empty_counter_schema",
        "benchmark_hosts",
    ),
)
def test_decision_recomputes_raw_roce_semantics(tmp_path, case):
    _kwargs, _ready, complete, complete_path = _smoke_gate(tmp_path)
    network = complete_path.parent / "network_preflight"
    if case == "missing_raw_artifact":
        complete["execution_artifacts"].pop("node_audit_0")
        _write_json(complete_path, complete)
        with pytest.raises(stage_c_pilot_decision.StageCDecisionError, match="set"):
            stage_c_pilot_decision._validate_network_evidence(
                complete, complete_path, "smoke"
            )
        return
    if case == "node_speed":
        artifact_key = "node_audit_0"
        path = network / "node-a.node.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["rails"]["rocep1s0f1"]["speed_mbit"] = 1_000
    elif case == "peer_connectivity":
        artifact_key = "connectivity_audit_0"
        path = network / "node-a.connectivity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["peer_connectivity"][0]["passed"] = False
    elif case == "empty_counter_schema":
        artifact_key = "counter_health"
        path = network / "COUNTER_HEALTH.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["harmful_counters"] = []
        value["traffic_counters"] = []
    else:
        artifact_key = "benchmark_dual"
        path = network / "NCCL_BENCHMARK_dual.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["hosts"] = ["node-a", "node-c"]
    _write_json(path, value)
    complete["execution_artifacts"][artifact_key] = prerequisites.sha256_file(path)
    _write_json(complete_path, complete)
    with pytest.raises(stage_c_pilot_decision.StageCDecisionError):
        stage_c_pilot_decision._validate_network_evidence(
            complete, complete_path, "smoke"
        )
