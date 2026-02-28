import numpy as np
from scripts.diagnostics.position_visualize import score_clip_rate


def _apply_watermark(original: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_expanded = alpha[..., None]
    watermarked = alpha_expanded * 255.0 + (1.0 - alpha_expanded) * original
    return watermarked.astype(np.float32)


def test_score_clip_rate_zero_when_no_clipping():
    original = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.float32,
    )
    alpha = np.full((2, 2), 0.5, dtype=np.float32)
    watermarked = _apply_watermark(original, alpha)
    score = score_clip_rate(watermarked, alpha)
    assert score == 0.0


def test_score_clip_rate_positive_when_clipped():
    watermarked = np.zeros((2, 2, 3), dtype=np.float32)
    alpha = np.full((2, 2), 0.5, dtype=np.float32)
    score = score_clip_rate(watermarked, alpha)
    assert score == 1.0
