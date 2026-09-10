"""Regression coverage for the narrowly corrected Stage-C r6 recovery."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from NIAF.continuous_trajectory_field import relevance_calibration
from NIAF.continuous_trajectory_field.scripts import stage_c_calibration_control
from NIAF.continuous_trajectory_field.scripts import stage_c_execution_control
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_decision
from NIAF.continuous_trajectory_field.scripts import stage_c_pilot_prerequisites


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "NIAF" / "continuous_trajectory_field" / "configs"
POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_"
    "decision_policy_v1.json"
)
RECOVERY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_"
    "recovery_evidence_v1.json"
)
R5_ATTEMPT = ROOT / (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
    "protocol_v4_run_r5_smoke/sources/"
    "source_36d121dd361df2c08616031bd7c2076d85da78d0/attempts/143609/"
    "executions/restart_0"
)
R5_CONFIG_BINDINGS = {
    "memory": (
        "c363b261a437fcfcc12d1e6f164bbdc7ec615918196274855c0908b57a168e2c"
    ),
    "matched_off": (
        "a4d4ff251da117ec091a6d85425a9ad68be96f3ddd472aff3abfb5eecdbe2317"
    ),
}
R5_CHECKPOINT_BINDINGS = {
    "memory": (
        "18d288309eef7f8430cda4b35f469cf5041a59382644440ef0cf74a43a1f119b"
    ),
    "matched_off": (
        "5753e7ae9a8c4ca284e1e3eb9620d5f2be60f4d686243c64fab8b43e9d1bee7a"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recursive_diff_paths(left: object, right: object, prefix: tuple[str, ...] = ()):
    if type(left) is not type(right):
        return {prefix}
    if isinstance(left, dict):
        paths: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                paths.add((*prefix, str(key)))
            else:
                paths.update(_recursive_diff_paths(left[key], right[key], (*prefix, str(key))))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return {prefix}
        paths: set[tuple[str, ...]] = set()
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.update(
                _recursive_diff_paths(
                    left_item, right_item, (*prefix, str(index))
                )
            )
        return paths
    return set() if left == right else {prefix}


def _redigest_named_identity(value: dict) -> None:
    value["digest"] = stage_c_pilot_decision.digest_json(
        {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}
    )


def test_protocol_v5_real_r5_checkpoint_configs_normalize_only_approved_fields() -> None:
    """The retained failing run proves both the correction and its boundary."""

    configs = {}
    provenances = {}
    for arm, expected_sha256 in R5_CONFIG_BINDINGS.items():
        path = R5_ATTEMPT / arm / "config.resolved.json"
        assert path.is_file() and not path.is_symlink()
        assert _sha256(path) == expected_sha256
        configs[arm] = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_path = R5_ATTEMPT / arm / "checkpoints" / "last.pt"
        assert checkpoint_path.is_file() and not checkpoint_path.is_symlink()
        assert _sha256(checkpoint_path) == R5_CHECKPOINT_BINDINGS[arm]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["config"] == configs[arm]
        provenances[arm] = checkpoint["stage_c_provenance"]

    assert _recursive_diff_paths(configs["memory"], configs["matched_off"]) == {
        ("conditioning", "sentence_memory_dropout_probability"),
        ("conditioning", "sentence_memory_train_mode"),
        ("experiment_name",),
        ("output", "out_dir"),
        ("sentence_memory_safety", "stage_c", "arm"),
        (
            "sentence_memory_safety",
            "stage_c",
            "resolved_source_provenance",
            "declared_contract",
            "arm",
        ),
        (
            "sentence_memory_safety",
            "stage_c",
            "resolved_source_provenance",
            "digest",
        ),
    }
    normalized_memory = stage_c_pilot_decision._normalize_config(
        configs["memory"],
        arm="memory",
        checkpoint_provenance=provenances["memory"],
    )
    normalized_matched_off = stage_c_pilot_decision._normalize_config(
        configs["matched_off"],
        arm="matched_off",
        checkpoint_provenance=provenances["matched_off"],
    )
    assert normalized_memory == normalized_matched_off
    normalized_provenance = normalized_memory["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ]
    assert "arm" not in normalized_provenance["declared_contract"]
    assert stage_c_pilot_decision._named_identity(
        normalized_provenance, "normalized provenance"
    ) == normalized_provenance

    changed = copy.deepcopy(configs["matched_off"])
    changed["train"]["lr"] = 0.0002
    assert normalized_memory != stage_c_pilot_decision._normalize_config(
        changed,
        arm="matched_off",
        checkpoint_provenance=provenances["matched_off"],
    )

    base_config = configs["memory"]
    base_provenance = provenances["memory"]

    forged_digest = copy.deepcopy(base_config)
    forged_digest["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ]["digest"] = "0" * 64

    missing_digest = copy.deepcopy(base_config)
    missing_digest["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ].pop("digest")

    wrong_arm = copy.deepcopy(base_config)
    wrong_arm_provenance = copy.deepcopy(base_provenance)
    wrong_arm_embedded = wrong_arm["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ]
    for provenance in (wrong_arm_embedded, wrong_arm_provenance):
        provenance["declared_contract"]["arm"] = "matched_off"
        _redigest_named_identity(provenance)

    added_adjacent = copy.deepcopy(base_config)
    added_embedded = added_adjacent["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ]
    added_embedded["unexpected_adjacent_field"] = "forged"
    _redigest_named_identity(added_embedded)

    missing_adjacent = copy.deepcopy(base_config)
    missing_embedded = missing_adjacent["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ]
    missing_embedded.pop("scope")
    _redigest_named_identity(missing_embedded)

    mismatched_embedded = copy.deepcopy(base_config)
    mismatch = mismatched_embedded["sentence_memory_safety"]["stage_c"][
        "resolved_source_provenance"
    ]
    mismatch["declared_contract"]["huber_beta"] = 0.2
    _redigest_named_identity(mismatch)

    for bad_config, checkpoint_provenance in (
        (forged_digest, base_provenance),
        (missing_digest, base_provenance),
        (wrong_arm, wrong_arm_provenance),
        (added_adjacent, base_provenance),
        (missing_adjacent, base_provenance),
        (mismatched_embedded, base_provenance),
    ):
        with pytest.raises(stage_c_pilot_decision.StageCDecisionError):
            stage_c_pilot_decision._normalize_config(
                bad_config,
                arm="memory",
                checkpoint_provenance=checkpoint_provenance,
            )


def test_protocol_v5_configs_preserve_r5_science_in_fresh_namespaces() -> None:
    memory = CONFIGS / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_memory_protocol_v5_run_r6.yaml"
    )
    matched_off = CONFIGS / (
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        "adaptation_matched_off_protocol_v5_run_r6.yaml"
    )
    audit = stage_c_pilot_prerequisites._validate_protocol_v5_run_r6_configs(
        recovery_audit={"audit_identity": "a" * 64},
        source_root=ROOT,
        memory_config=memory,
        matched_off_config=matched_off,
    )

    assert audit["scientific_settings_equal_protocol_v4_run_r5"] is True
    assert audit["development_only"] is True
    assert audit["non_authorizing"] is True


def test_protocol_v5_controls_and_source_binding_are_registered(
    tmp_path: Path,
) -> None:
    profile = "stage_c_generator_adaptation_protocol_v5_run_r6"
    assert relevance_calibration.SOURCE_FILE_PROFILES[profile] == {
        "NIAF/continuous_trajectory_field/relevance_calibration.py",
        (
            "NIAF/continuous_trajectory_field/scripts/"
            "calibrate_sentence_memory_relevance.py"
        ),
        (
            "NIAF/continuous_trajectory_field/scripts/"
            "decide_centered_memory_stage.py"
        ),
        (
            "scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_"
            "protocol_v5_run_r6_sbatch.sh"
        ),
        "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh",
    }
    assert profile in stage_c_calibration_control.SOURCE_FILE_PROFILES
    assert POLICY_NAME in stage_c_execution_control.SMOKE_PREREQUISITE_POLICY_NAMES

    policy_path = tmp_path / POLICY_NAME
    policy_sha256 = "a" * 64
    marker_names = {
        POLICY_NAME,
        RECOVERY_NAME,
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v5_run_r6.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v5_run_r6.yaml"
        ),
        "run_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6.sh",
        (
            "calibrate_csl_daily_stage_c_generator_adaptation_"
            "protocol_v5_run_r6_sbatch.sh"
        ),
        "ARCHIVE.json",
    }
    bound = {
        str((tmp_path / name).resolve()): (
            policy_sha256 if name == POLICY_NAME else "b" * 64
        )
        for name in marker_names
    }
    bound.update(
        {
            str((tmp_path / f"r6_source_{index}.py").resolve()): "c" * 64
            for index in range(21)
        }
    )
    assert len(bound) == 30
    assert stage_c_pilot_decision._validate_source_binding_generation(
        bound_files=bound,
        decision_policy={
            "path": str(policy_path.resolve()),
            "sha256": policy_sha256,
        },
        mode="smoke",
    ) == 30

    mismatched = dict(bound)
    mismatched.pop(
        str(
            (
                tmp_path
                / "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_memory_protocol_v5_run_r6.yaml"
            ).resolve()
        )
    )
    mismatched[str((tmp_path / "replacement.py").resolve())] = "d" * 64
    with pytest.raises(stage_c_pilot_decision.StageCDecisionError):
        stage_c_pilot_decision._validate_source_binding_generation(
            bound_files=mismatched,
            decision_policy={
                "path": str(policy_path.resolve()),
                "sha256": policy_sha256,
            },
            mode="smoke",
        )


def test_protocol_v5_policy_archive_and_runtime_recovery_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CONFIGS / POLICY_NAME
    recovery = CONFIGS / RECOVERY_NAME
    manifest = json.loads(recovery.read_text(encoding="utf-8"))
    evidence_root = Path(manifest["canonical_evidence_root"])

    contract = stage_c_pilot_decision.validate_policy(policy)["recovery_contract"]
    assert contract["protocol_generation"] == "protocol_v5"
    assert contract["run_generation"] == "run_r6"
    assert contract["resume_authorized"] is False
    assert contract["same_protocol_retry_authorized"] is False
    assert contract["r5_artifact_reuse_authorized"] is False
    assert contract["incident_archive"]["sha256"] == (
        "9c4e2213ca51941b41281f0b869fc66aa4e440df846ff57dee2731fca5f1c41a"
    )
    assert manifest["incident_archive"]["bytes"] == 39_609
    assert manifest["incident_archive"]["schema_version"] == 4

    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_reopen_protocol_v5_archive_tree",
        lambda **_kwargs: pytest.fail("runtime must not rehash historical trees"),
    )
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_validate_standalone_source_clone",
        lambda **_kwargs: pytest.fail("runtime must not scan the archived r5 clone"),
    )
    audit = stage_c_pilot_prerequisites.validate_protocol_v5_run_r6_runtime_bindings(
        policy_path=policy,
        recovery_manifest=recovery,
        source_root=ROOT,
        evidence_root=evidence_root,
    )
    assert audit["incident_archive_sha256"] == contract["incident_archive"]["sha256"]
    assert audit["historical_evidence_rehashed_in_cpu_gate"] is False
    assert audit["prior_optimizer_updates_per_arm"] == {
        "memory": 1,
        "matched_off": 1,
    }


def test_protocol_v5_generic_recovery_dispatch_is_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "validate_protocol_v5_run_r6_recovery_evidence",
        lambda **kwargs: calls.append(kwargs) or {"generation": "run_r6"},
    )
    common = {
        "recovery_manifest": tmp_path / "recovery.json",
        "source_root": tmp_path / "source",
        "evidence_root": tmp_path / "evidence",
    }

    assert stage_c_pilot_prerequisites.validate_recovery_evidence(
        policy_path=tmp_path / POLICY_NAME,
        **common,
    ) == {"generation": "run_r6"}
    assert calls == [{"policy_path": tmp_path / POLICY_NAME, **common}]


def test_protocol_v5_cpu_recovery_path_requests_every_historical_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CONFIGS / POLICY_NAME
    recovery = CONFIGS / RECOVERY_NAME
    evidence_root = Path(
        json.loads(recovery.read_text(encoding="utf-8"))["canonical_evidence_root"]
    )
    tree_labels: list[str] = []
    clone_calls: list[dict] = []
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_reopen_protocol_v5_archive_tree",
        lambda **kwargs: tree_labels.append(kwargs["label"])
        or kwargs["evidence_root"],
    )
    monkeypatch.setattr(
        stage_c_pilot_prerequisites,
        "_validate_standalone_source_clone",
        lambda **kwargs: clone_calls.append(kwargs),
    )

    audit = (
        stage_c_pilot_prerequisites.validate_protocol_v5_run_r6_recovery_evidence(
            policy_path=policy,
            recovery_manifest=recovery,
            source_root=ROOT,
            evidence_root=evidence_root,
        )
    )
    runtime_audit = (
        stage_c_pilot_prerequisites.validate_protocol_v5_run_r6_runtime_bindings(
            policy_path=policy,
            recovery_manifest=recovery,
            source_root=ROOT,
            evidence_root=evidence_root,
        )
    )
    assert audit["historical_evidence_rehashed_in_cpu_gate"] is True
    assert audit["audit_identity"] == runtime_audit["audit_identity"]
    assert len(tree_labels) == 3
    assert {label.rsplit(" ", 1)[-1] for label in tree_labels} == {
        "calibration_artifact",
        "run_prerequisites",
        "smoke_attempt",
    }
    assert clone_calls == [
        {
            "clone": Path(
                "/media/cvpr/haomian/"
                "SignTrajField_centered_stage_c_v4_run_source_r5"
            ),
            "expected_head": "36d121dd361df2c08616031bd7c2076d85da78d0",
            "expected_remote_ref": (
                "origin/codex/csl-daily-centered-generator-stage-c-v4-run-r5"
            ),
            "local_metadata_timeout_seconds": 600,
            "remote_ref_timeout_seconds": 120,
        }
    ]


def test_protocol_v5_scripts_are_distinct_and_have_no_stale_runtime_namespace() -> None:
    scripts = sorted((ROOT / "scripts" / "NIAF").glob("*protocol_v5_run_r6*"))
    assert len(scripts) == 6
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "protocol_v5_run_r6" in text
        if script.name.endswith("_sbatch.sh"):
            assert "csl_stage_c_protocol_v5_run_r6" in text
        if not script.name.startswith(("smoke_", "train_")):
            assert "codex/csl-daily-centered-generator-stage-c-v5-run-r6" in text
        without_archive = "\n".join(
            line for line in text.splitlines() if "INCIDENT_ARCHIVE=" not in line
        )
        assert "protocol_v4_run_r5" not in without_archive
        assert "csl_stage_c_protocol_v4_run_r5" not in without_archive
