import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path("NIAF/continuous_trajectory_field/relevance_calibration.py")
SPEC = importlib.util.spec_from_file_location("niaf_relevance_calibration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RelevanceCalibrationError = MODULE.RelevanceCalibrationError
binary_auroc = MODULE.binary_auroc
calibrate_from_map = MODULE.calibrate_from_map
calibrated_probability = MODULE.calibrated_probability
deterministic_alternative_item_ids = MODULE.deterministic_alternative_item_ids
fit_balanced_logistic = MODULE.fit_balanced_logistic
publish_calibration = MODULE.publish_calibration
sha_query_partition = MODULE.sha_query_partition
validate_bank_index_order = MODULE.validate_bank_index_order
validate_relevance_calibration_artifact = MODULE.validate_relevance_calibration_artifact
validate_relevance_calibration_source = MODULE.validate_relevance_calibration_source


def _rows(count=20):
    return [
        {
            "bank_index": index,
            "motion_path": f"train/q{index}.npz",
            "name": f"q{index}",
            "semantic_group_id": f"semantic-{index}",
            "source_group_id": f"source-group-{index}",
            "source_id": f"source-{index}",
        }
        for index in range(count)
    ]


def test_bank_index_must_equal_metadata_row_position():
    rows = _rows(4)
    validate_bank_index_order(rows)
    rows[2]["bank_index"] = 7
    with pytest.raises(RelevanceCalibrationError, match="identity-preserving"):
        validate_bank_index_order(rows)


def test_sha_query_partition_is_deterministic_and_exact_90_10():
    rows = _rows(101)
    first, first_digests = sha_query_partition(rows)
    second, second_digests = sha_query_partition(copy.deepcopy(rows))
    assert np.array_equal(first, second)
    assert first_digests == second_digests
    assert int(first.sum()) == 10
    assert len(set(first_digests)) == len(rows)
    changed, _digests = sha_query_partition(rows, seed=1235)
    assert not np.array_equal(first, changed)


def test_full_alternatives_are_stable_unique_and_exclude_query_groups():
    rows = _rows(31)
    # Items 1 and 2 deliberately share the query's semantic/source identities.
    rows[1]["semantic_group_id"] = rows[0]["semantic_group_id"]
    rows[2]["source_group_id"] = rows[0]["source_group_id"]
    groups = np.arange(len(rows), dtype=np.int64)
    selected = deterministic_alternative_item_ids(
        query_item_id=0,
        query_row=rows[0],
        rows=rows,
        positive_item_ids=[3, 4],
        positive_group_ids=[3, 4],
        item_group_ids=groups,
        count=8,
    )
    repeated = deterministic_alternative_item_ids(
        query_item_id=0,
        query_row=copy.deepcopy(rows[0]),
        rows=copy.deepcopy(rows),
        positive_item_ids=[3, 4],
        positive_group_ids=[3, 4],
        item_group_ids=groups.copy(),
        count=8,
    )
    assert selected == repeated
    assert len(selected) == len(set(selected)) == 8
    assert not {0, 1, 2, 3, 4}.intersection(selected)


def test_balanced_float64_logistic_and_auroc_numerics():
    scores = np.asarray([-2.0, -1.0, -0.5, 0.1, 0.3, 0.8, 1.2, 2.0])
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8)
    fitted = fit_balanced_logistic(scores, labels)
    probability = calibrated_probability(
        scores,
        intercept=fitted["intercept"],
        slope=fitted["slope"],
    )
    assert probability.dtype == np.float64
    assert fitted["slope"] > 0.0
    assert fitted["a"] == fitted["slope"]
    assert fitted["intercept"] == -fitted["a"] * fitted["b"]
    assert fitted["formula"] == "sigmoid(a*(adjusted_score-b))"
    assert fitted["class_weight_positive_sum"] == pytest.approx(0.5)
    assert fitted["class_weight_negative_sum"] == pytest.approx(0.5)
    assert np.all(np.diff(probability) > 0.0)
    assert binary_auroc(labels, scores) == pytest.approx(0.9375)
    assert binary_auroc([0, 1], [1.0, 1.0]) == pytest.approx(0.5)


def _accepted_map():
    rng = np.random.default_rng(1234)
    queries = 40
    positive = rng.normal(1.0, 0.25, size=(queries, 8)).astype(np.float64)
    negative = rng.normal(-1.0, 0.25, size=(queries, 8)).astype(np.float64)
    split = np.zeros(queries, dtype=np.uint8)
    split[:4] = 1
    return {
        "negative_adjusted_scores": negative,
        "negative_item_ids": np.arange(queries * 8, dtype=np.int32).reshape(queries, 8),
        "positive_adjusted_scores": positive,
        "positive_item_ids": np.arange(
            queries * 8, 2 * queries * 8, dtype=np.int32
        ).reshape(queries, 8),
        "query_item_ids": np.arange(queries, dtype=np.int32),
        "query_partition": split,
    }


def test_calibration_acceptance_and_atomic_seal_detect_tampering(tmp_path):
    arrays = _accepted_map()
    result = calibrate_from_map(arrays)
    assert result["accepted"] is True
    assert result["holdout"]["auroc"] >= 0.75
    assert result["holdout"]["probability_gap"] >= 0.20
    output = tmp_path / "calibration"
    artifact = publish_calibration(
        output_dir=output,
        arrays=arrays,
        result=result,
        input_identity={"bank": {"bank_id": "bank"}},
        source_identity={"git_head": "a" * 40},
        seed=1234,
        duration_weight=0.05,
        split_nonce=MODULE.SPLIT_NONCE,
        alternative_nonce=MODULE.ALTERNATIVE_NONCE,
    )
    loaded = validate_relevance_calibration_artifact(
        output, expected_identity=artifact["identity"]
    )
    assert loaded["coefficients"] == artifact["coefficients"]
    ready = json.loads((output / "READY").read_text(encoding="utf-8"))
    assert ready["artifact_identity"] == artifact["identity"]
    assert (output.stat().st_mode & 0o222) == 0
    assert all((path.stat().st_mode & 0o222) == 0 for path in output.iterdir())

    map_path = output / "calibration_map.npz"
    map_path.chmod(0o644)
    map_path.write_bytes(map_path.read_bytes() + b"tamper")
    with pytest.raises(RelevanceCalibrationError, match="READY envelope"):
        validate_relevance_calibration_artifact(output)
    for path in output.iterdir():
        path.chmod(0o644)
    output.chmod(0o755)


def test_sealed_source_binding_rejects_commit_or_source_file_drift(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    files = {}
    for relative in MODULE.REQUIRED_SOURCE_FILES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"stable:{relative}\n", encoding="utf-8")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": MODULE.sha256_file(path),
        }
    source = {
        "git_head": "a" * 40,
        "remote_head": "a" * 40,
        "remote_ref": "origin/frozen",
        "repository_root": str(source_root),
        "source_files": files,
    }
    validated = validate_relevance_calibration_source(
        {"source": source},
        source_root=source_root,
        expected_git_head="a" * 40,
        expected_remote_ref="origin/frozen",
        expected_remote_head="a" * 40,
    )
    assert validated["git_head"] == "a" * 40
    assert validated["source_file_profile"] == MODULE.LEGACY_SOURCE_FILE_PROFILE
    with pytest.raises(RelevanceCalibrationError, match="commit"):
        validate_relevance_calibration_source(
            {"source": source},
            source_root=source_root,
            expected_git_head="b" * 40,
        )
    helper = source_root / sorted(MODULE.REQUIRED_SOURCE_FILES)[0]
    helper.write_text("drift\n", encoding="utf-8")
    with pytest.raises(RelevanceCalibrationError, match="source file changed"):
        validate_relevance_calibration_source(
            {"source": source},
            source_root=source_root,
            expected_git_head="a" * 40,
        )


def test_stage_c_source_profile_is_exact_and_rejects_mixed_or_unknown_sets(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    files = {}
    for relative in MODULE.STAGE_C_REQUIRED_SOURCE_FILES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"stage-c:{relative}\n", encoding="utf-8")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": MODULE.sha256_file(path),
        }
    source = {
        "git_head": "c" * 40,
        "remote_head": "c" * 40,
        "remote_ref": "origin/stage-c",
        "repository_root": str(source_root),
        "source_file_profile": MODULE.STAGE_C_SOURCE_FILE_PROFILE,
        "source_files": files,
    }
    validated = validate_relevance_calibration_source(
        {"source": source},
        source_root=source_root,
        expected_git_head="c" * 40,
        expected_remote_ref="origin/stage-c",
        expected_remote_head="c" * 40,
        expected_source_file_profile=MODULE.STAGE_C_SOURCE_FILE_PROFILE,
    )
    assert validated["source_file_profile"] == MODULE.STAGE_C_SOURCE_FILE_PROFILE

    retry2_source = copy.deepcopy(source)
    retry2_source["source_file_profile"] = MODULE.STAGE_C_RETRY2_SOURCE_FILE_PROFILE
    retry2 = validate_relevance_calibration_source(
        {"source": retry2_source},
        source_root=source_root,
        expected_git_head="c" * 40,
        expected_remote_ref="origin/stage-c",
        expected_remote_head="c" * 40,
        expected_source_file_profile=MODULE.STAGE_C_RETRY2_SOURCE_FILE_PROFILE,
    )
    assert retry2["source_file_profile"] == (
        MODULE.STAGE_C_RETRY2_SOURCE_FILE_PROFILE
    )
    with pytest.raises(RelevanceCalibrationError, match="profile changed"):
        validate_relevance_calibration_source(
            {"source": retry2_source},
            source_root=source_root,
            expected_git_head="c" * 40,
            expected_source_file_profile=(
                MODULE.STAGE_C_PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE
            ),
        )

    protocol_root = tmp_path / "protocol-v2-source"
    protocol_root.mkdir()
    protocol_files = {}
    for relative in MODULE.STAGE_C_PROTOCOL_V2_RUN_R3_REQUIRED_SOURCE_FILES:
        path = protocol_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protocol-v2:{relative}\n", encoding="utf-8")
        protocol_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": MODULE.sha256_file(path),
        }
    protocol_source = {
        "git_head": "d" * 40,
        "remote_head": "d" * 40,
        "remote_ref": "origin/protocol-v2",
        "repository_root": str(protocol_root),
        "source_file_profile": MODULE.STAGE_C_PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE,
        "source_files": protocol_files,
    }
    protocol = validate_relevance_calibration_source(
        {"source": protocol_source},
        source_root=protocol_root,
        expected_git_head="d" * 40,
        expected_remote_ref="origin/protocol-v2",
        expected_remote_head="d" * 40,
        expected_source_file_profile=(
            MODULE.STAGE_C_PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE
        ),
    )
    assert protocol["source_file_profile"] == (
        MODULE.STAGE_C_PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE
    )
    assert (
        "scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_"
        "protocol_v2_run_r3_sbatch.sh"
        in protocol["source_files"]
    )
    assert (
        "scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh"
        not in protocol["source_files"]
    )

    wrong_expected = copy.deepcopy(source)
    with pytest.raises(RelevanceCalibrationError, match="profile changed"):
        validate_relevance_calibration_source(
            {"source": wrong_expected},
            source_root=source_root,
            expected_git_head="c" * 40,
            expected_source_file_profile=MODULE.LEGACY_SOURCE_FILE_PROFILE,
        )

    unknown = copy.deepcopy(source)
    unknown["source_file_profile"] = "unapproved_v9"
    with pytest.raises(RelevanceCalibrationError, match="Unknown"):
        validate_relevance_calibration_source(
            {"source": unknown},
            source_root=source_root,
            expected_git_head="c" * 40,
        )

    missing = copy.deepcopy(source)
    missing["source_files"].pop(next(iter(missing["source_files"])))
    with pytest.raises(RelevanceCalibrationError, match="exact profile"):
        validate_relevance_calibration_source(
            {"source": missing},
            source_root=source_root,
            expected_git_head="c" * 40,
        )

    mixed = copy.deepcopy(source)
    legacy_launcher = (
        "scripts/NIAF/calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh"
    )
    path = source_root / legacy_launcher
    path.write_text("legacy launcher\n", encoding="utf-8")
    mixed["source_files"][legacy_launcher] = {
        "bytes": path.stat().st_size,
        "sha256": MODULE.sha256_file(path),
    }
    with pytest.raises(RelevanceCalibrationError, match="exact profile"):
        validate_relevance_calibration_source(
            {"source": mixed},
            source_root=source_root,
            expected_git_head="c" * 40,
        )


def test_calibration_publication_never_replaces_existing_artifact(tmp_path):
    arrays = _accepted_map()
    result = calibrate_from_map(arrays)
    output = tmp_path / "calibration"
    kwargs = dict(
        output_dir=output,
        arrays=arrays,
        result=result,
        input_identity={},
        source_identity={},
        seed=1234,
        duration_weight=0.05,
        split_nonce=MODULE.SPLIT_NONCE,
        alternative_nonce=MODULE.ALTERNATIVE_NONCE,
    )
    artifact = publish_calibration(**kwargs)
    ready_before = (output / "READY").read_bytes()
    with pytest.raises(RelevanceCalibrationError, match="already exists"):
        publish_calibration(**kwargs)
    assert (output / "READY").read_bytes() == ready_before
    assert (
        validate_relevance_calibration_artifact(output)["identity"]
        == artifact["identity"]
    )
    for path in output.iterdir():
        path.chmod(0o644)
    output.chmod(0o755)


def test_rejected_calibration_never_writes_ready(tmp_path):
    arrays = _accepted_map()
    result = calibrate_from_map(arrays)
    rejected = copy.deepcopy(result)
    rejected["accepted"] = False
    output = tmp_path / "rejected"
    publish_calibration(
        output_dir=output,
        arrays=arrays,
        result=rejected,
        input_identity={},
        source_identity={},
        seed=1234,
        duration_weight=0.05,
        split_nonce=MODULE.SPLIT_NONCE,
        alternative_nonce=MODULE.ALTERNATIVE_NONCE,
    )
    assert not (output / "READY").exists()
    assert (output / "REJECTED").is_file()
    assert (output.stat().st_mode & 0o222) == 0
    assert all((path.stat().st_mode & 0o222) == 0 for path in output.iterdir())
    with pytest.raises(RelevanceCalibrationError):
        validate_relevance_calibration_artifact(output)
    for path in output.iterdir():
        path.chmod(0o644)
    output.chmod(0o755)


def test_calibration_script_never_discovers_neighbor_splits():
    source = Path(
        "NIAF/continuous_trajectory_field/relevance_calibration.py"
    ).read_text(encoding="utf-8")
    body = source.split("def load_train_inputs(", 1)[1].split(
        "def build_calibration_map(", 1
    )[0]
    assert ".glob(" not in body
    assert ".iterdir(" not in body
    assert "neighbors_test" not in body
    assert "neighbors_val" not in body
