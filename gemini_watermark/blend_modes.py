from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gemini_watermark.watermark_engine import WatermarkPosition

ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255.0


def unblend_region(
    rgba: np.ndarray,
    alpha_map: np.ndarray,
    position: "WatermarkPosition",
) -> None:
    x, y, width, height = position.x, position.y, position.width, position.height
    if width <= 0 or height <= 0:
        return

    region = rgba[y : y + height, x : x + width, :3]
    alpha = alpha_map.astype(np.float32)

    if alpha.shape != (height, width):
        raise ValueError("alpha_map shape does not match region size")

    mask = alpha >= ALPHA_THRESHOLD
    if not np.any(mask):
        return

    alpha = np.minimum(alpha, MAX_ALPHA)
    one_minus = 1.0 - alpha
    corrected = (region - alpha[..., None] * LOGO_VALUE) / one_minus[..., None]
    corrected = np.clip(np.rint(corrected), 0, 255)

    region[mask] = corrected[mask]
