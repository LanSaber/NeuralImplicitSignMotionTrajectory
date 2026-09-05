import numpy as np

from flow.evaluate.dtw_mpjpe_t2m_default import (
    FACE_LANDMARK_COUNT,
    face_landmarks,
    frame_distance_matrix_pa,
    motion_path_and_jerk_diagnostics,
)


def test_body_pa_fits_the_same_body_keypoints_that_it_scores():
    gt_body = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pred_body = 2.0 * gt_body + np.asarray([10.0, -3.0, 5.0])

    # These unrelated points deliberately favor a different full-joint
    # transform. They must not influence a partwise body PA measurement.
    gt_other = np.asarray(
        [[float(i), float(i % 3), float(i % 5)] for i in range(1, 21)],
        dtype=np.float64,
    )
    pred_other = gt_other.copy()

    gt_parts = {
        "body": gt_body[None, ...],
        "full_joints": np.concatenate([gt_body, gt_other], axis=0)[None, ...],
    }
    pred_parts = {
        "body": pred_body[None, ...],
        "full_joints": np.concatenate([pred_body, pred_other], axis=0)[None, ...],
    }

    distance = frame_distance_matrix_pa(pred_parts, gt_parts, "body")

    np.testing.assert_allclose(distance, np.zeros((1, 1)), atol=1e-10)


def test_face_part_uses_appended_smplx_landmarks():
    joints = np.arange(2 * 130 * 3, dtype=np.float32).reshape(2, 130, 3)
    face = face_landmarks(joints)
    assert face.shape == (2, FACE_LANDMARK_COUNT, 3)
    np.testing.assert_array_equal(face, joints[:, -FACE_LANDMARK_COUNT:])


def test_motion_path_and_jerk_diagnostics_use_each_sequence_fps():
    time = np.arange(6, dtype=np.float64)
    gt = (time**3)[:, None, None]
    pred = (2.0 * time**3)[:, None, None]
    result = motion_path_and_jerk_diagnostics(
        pred,
        gt,
        pred_fps=10.0,
        gt_fps=20.0,
    )
    assert result["motion_path_ratio"] == 2.0
    assert result["motion_path_error"] == 1.0
    # The doubled coordinate magnitude is offset by the 1/8 time-scale
    # factor from (10 / 20)^3.
    assert result["jerk_magnitude_ratio"] == 0.25
