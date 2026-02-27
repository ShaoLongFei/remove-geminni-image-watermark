import numpy as np
from diagnose_watermark_position import find_best_position


def test_find_best_position_prefers_low_violation():
    min_rgb = np.full((10, 10), 200, dtype=np.float32)
    alpha = np.full((3, 3), 0.5, dtype=np.float32)

    # Inject violations at one candidate location
    min_rgb[5:8, 5:8] = 0

    pred_x, pred_y = 4, 4
    best_x, best_y, score = find_best_position(
        min_rgb=min_rgb,
        alpha=alpha,
        pred_x=pred_x,
        pred_y=pred_y,
        size=3,
        radius=2,
        coarse_step=1,
        epsilon=1.0,
    )

    # Best should avoid the violated block at (5,5)
    assert (best_x, best_y) != (5, 5)
    assert score <= 0.5
