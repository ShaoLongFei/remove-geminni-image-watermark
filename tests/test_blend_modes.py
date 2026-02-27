import numpy as np
from gemini_watermark.blend_modes import unblend_region
from gemini_watermark.watermark_engine import WatermarkPosition


def _apply_watermark(original_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_expanded = alpha[..., None]
    watermarked = alpha_expanded * 255.0 + (1.0 - alpha_expanded) * original_rgb
    return np.clip(np.rint(watermarked), 0, 255).astype(np.float32)


def test_unblend_recovers_original_within_tolerance():
    original = np.array([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]],
    ], dtype=np.float32)
    alpha = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ], dtype=np.float32)
    watermarked = _apply_watermark(original, alpha)
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., :3] = watermarked
    rgba[..., 3] = 255.0

    pos = WatermarkPosition(x=0, y=0, width=2, height=2)
    unblend_region(rgba, alpha, pos)

    diff = np.abs(rgba[..., :3] - original)
    assert diff.max() <= 1.0


def test_unblend_skips_low_alpha():
    original = np.array([[[10, 20, 30]]], dtype=np.float32)
    alpha = np.array([[0.0]], dtype=np.float32)
    rgba = np.zeros((1, 1, 4), dtype=np.float32)
    rgba[..., :3] = original
    rgba[..., 3] = 255.0

    pos = WatermarkPosition(x=0, y=0, width=1, height=1)
    unblend_region(rgba, alpha, pos)

    assert np.allclose(rgba[..., :3], original)
