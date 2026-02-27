import numpy as np


def calculate_alpha_map_from_rgba(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("Expected RGBA image with shape (H, W, 4)")
    rgb = rgba[..., :3].astype(np.float32)
    alpha = np.max(rgb, axis=2) / 255.0
    return alpha.astype(np.float32)
