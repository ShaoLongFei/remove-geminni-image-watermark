import numpy as np

ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255.0


def score_clip_rate(rgb: np.ndarray, alpha: np.ndarray) -> float:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected rgb array with shape (H, W, 3)")
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("Alpha map shape does not match rgb region size")

    alpha = alpha.astype(np.float32)
    mask = alpha >= ALPHA_THRESHOLD
    if not np.any(mask):
        return 1.0

    alpha = np.minimum(alpha, MAX_ALPHA)
    one_minus = 1.0 - alpha
    original = (rgb.astype(np.float32) - alpha[..., None] * LOGO_VALUE) / one_minus[
        ..., None
    ]
    clipped = (original < 0) | (original > 255)

    clipped_channels = int(np.sum(clipped[mask]))
    total_channels = int(np.sum(mask)) * 3
    return float(clipped_channels) / float(total_channels)


def find_best_position(
    *,
    rgb: np.ndarray,
    alpha: np.ndarray,
    pred_x: int,
    pred_y: int,
    size: int,
    radius: int,
    coarse_step: int,
) -> tuple[int, int, float]:
    height, width = rgb.shape[0], rgb.shape[1]
    best_x, best_y, best_score = pred_x, pred_y, 1.0

    def clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, val))

    start_x = clamp(pred_x - radius, 0, width - size)
    end_x = clamp(pred_x + radius, 0, width - size)
    start_y = clamp(pred_y - radius, 0, height - size)
    end_y = clamp(pred_y + radius, 0, height - size)

    for y in range(start_y, end_y + 1, coarse_step):
        for x in range(start_x, end_x + 1, coarse_step):
            patch = rgb[y : y + size, x : x + size, :3]
            score = score_clip_rate(patch, alpha)
            if score < best_score:
                best_x, best_y, best_score = x, y, score

    return best_x, best_y, best_score
