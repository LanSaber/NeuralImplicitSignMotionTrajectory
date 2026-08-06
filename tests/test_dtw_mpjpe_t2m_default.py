import numpy as np

from flow.evaluate.dtw_mpjpe_t2m_default import frame_distance_matrix_pa


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
