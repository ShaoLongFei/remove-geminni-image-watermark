import numpy as np
from gemini_watermark.alpha_map import get_alpha_map_for_size


def test_get_alpha_map_for_size_shape_and_range():
    alpha = get_alpha_map_for_size(48)
    assert alpha.shape == (48, 48)
    assert alpha.dtype == np.float32
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0
    assert float(alpha.max()) > 0.0
