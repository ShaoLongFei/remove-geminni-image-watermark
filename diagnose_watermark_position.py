import numpy as np


def score_violation_rate(min_rgb: np.ndarray, alpha: np.ndarray, epsilon: float) -> float:
    threshold = alpha * 255.0 - epsilon
    violations = min_rgb < threshold
    return float(np.sum(violations)) / float(violations.size)
