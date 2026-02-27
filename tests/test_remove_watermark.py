import numpy as np
from PIL import Image
from gemini_watermark.alpha_map import get_alpha_map_for_size
from gemini_watermark.watermark_engine import (
    detect_watermark_config,
    calculate_watermark_position,
)
from gemini_watermark.image_io import remove_watermark


def _apply_watermark(rgba: np.ndarray, alpha_map: np.ndarray, pos) -> np.ndarray:
    out = rgba.copy().astype(np.float32)
    region = out[pos.y : pos.y + pos.height, pos.x : pos.x + pos.width, :3]
    alpha = alpha_map.astype(np.float32)
    watermarked = alpha[..., None] * 255.0 + (1.0 - alpha[..., None]) * region
    out[pos.y : pos.y + pos.height, pos.x : pos.x + pos.width, :3] = np.clip(
        np.rint(watermarked), 0, 255
    )
    return out


def test_remove_watermark_recovers_background(tmp_path):
    size = 128
    original = np.zeros((size, size, 4), dtype=np.float32)
    original[..., :3] = [50, 100, 150]
    original[..., 3] = 255.0

    cfg = detect_watermark_config(size, size)
    pos = calculate_watermark_position(size, size, cfg)
    alpha = get_alpha_map_for_size(cfg.logo_size)
    watermarked = _apply_watermark(original, alpha, pos)

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(watermarked.astype(np.uint8), mode="RGBA").save(input_path)

    remove_watermark(str(input_path), str(output_path))

    result = np.array(Image.open(output_path).convert("RGBA"), dtype=np.float32)
    diff = np.abs(result - original)
    assert diff.max() <= 1.0
