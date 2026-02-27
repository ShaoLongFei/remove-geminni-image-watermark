import numpy as np


def score_violation_rate(min_rgb: np.ndarray, alpha: np.ndarray, epsilon: float) -> float:
    threshold = alpha * 255.0 - epsilon
    violations = min_rgb < threshold
    return float(np.sum(violations)) / float(violations.size)


def find_best_position(
    *,
    min_rgb: np.ndarray,
    alpha: np.ndarray,
    pred_x: int,
    pred_y: int,
    size: int,
    radius: int,
    coarse_step: int,
    epsilon: float,
) -> tuple[int, int, float]:
    height, width = min_rgb.shape
    best_x, best_y, best_score = pred_x, pred_y, 1.0

    def clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, val))

    start_x = clamp(pred_x - radius, 0, width - size)
    end_x = clamp(pred_x + radius, 0, width - size)
    start_y = clamp(pred_y - radius, 0, height - size)
    end_y = clamp(pred_y + radius, 0, height - size)

    for y in range(start_y, end_y + 1, coarse_step):
        for x in range(start_x, end_x + 1, coarse_step):
            patch = min_rgb[y : y + size, x : x + size]
            score = score_violation_rate(patch, alpha, epsilon)
            if score < best_score:
                best_x, best_y, best_score = x, y, score

    return best_x, best_y, best_score
