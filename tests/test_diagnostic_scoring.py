import numpy as np
from diagnose_watermark_position import score_violation_rate


def test_score_violation_rate_zero_when_constraint_satisfied():
    min_rgb = np.array([[200, 200], [200, 200]], dtype=np.float32)
    alpha = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    score = score_violation_rate(min_rgb, alpha, epsilon=1.0)
    assert score == 0.0


def test_score_violation_rate_positive_when_violated():
    min_rgb = np.array([[10, 200], [200, 10]], dtype=np.float32)
    alpha = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    score = score_violation_rate(min_rgb, alpha, epsilon=1.0)
    assert score == 0.5
