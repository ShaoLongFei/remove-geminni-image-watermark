import numpy as np
from gemini_watermark.alpha_map import calculate_alpha_map_from_rgba


def test_calculate_alpha_map_from_rgba():
    rgba = np.array([
        [[0, 10, 5, 255], [255, 0, 0, 255]]
    ], dtype=np.uint8)
    alpha = calculate_alpha_map_from_rgba(rgba)
    assert alpha.shape == (1, 2)
    assert np.allclose(alpha, np.array([[10 / 255.0, 1.0]], dtype=np.float32))
