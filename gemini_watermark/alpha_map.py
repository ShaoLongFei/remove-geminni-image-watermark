from pathlib import Path

import numpy as np
from PIL import Image

_ALPHA_MAP_CACHE: dict[int, np.ndarray] = {}


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def calculate_alpha_map_from_rgba(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("Expected RGBA image with shape (H, W, 4)")
    rgb = rgba[..., :3].astype(np.float32)
    alpha = np.max(rgb, axis=2) / 255.0
    return alpha.astype(np.float32)


def get_alpha_map_for_size(size: int) -> np.ndarray:
    if size in _ALPHA_MAP_CACHE:
        return _ALPHA_MAP_CACHE[size]
    if size not in (48, 96):
        raise ValueError("Unsupported watermark size")

    path = _assets_dir() / f"bg_{size}.png"
    with Image.open(path) as img:
        rgba = np.array(img.convert("RGBA"), dtype=np.float32)

    alpha = calculate_alpha_map_from_rgba(rgba)
    _ALPHA_MAP_CACHE[size] = alpha
    return alpha
