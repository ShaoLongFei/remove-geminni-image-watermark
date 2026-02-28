from dataclasses import dataclass

import numpy as np

from gemini_watermark.alpha_map import get_alpha_map_for_size
from gemini_watermark.blend_modes import unblend_region


@dataclass(frozen=True)
class WatermarkConfig:
    logo_size: int
    margin_right: int
    margin_bottom: int


@dataclass(frozen=True)
class WatermarkPosition:
    x: int
    y: int
    width: int
    height: int


def detect_watermark_config(image_width: int, image_height: int) -> WatermarkConfig:
    if image_width >= 160 and image_height >= 160:
        return WatermarkConfig(logo_size=96, margin_right=64, margin_bottom=64)
    return WatermarkConfig(logo_size=48, margin_right=32, margin_bottom=32)


def calculate_watermark_position(
    image_width: int,
    image_height: int,
    config: WatermarkConfig,
) -> WatermarkPosition:
    return WatermarkPosition(
        x=image_width - config.margin_right - config.logo_size,
        y=image_height - config.margin_bottom - config.logo_size,
        width=config.logo_size,
        height=config.logo_size,
    )


def remove_watermark_from_array(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[0], rgba.shape[1]
    config = detect_watermark_config(width, height)
    position = calculate_watermark_position(width, height, config)

    if position.x < 0 or position.y < 0:
        return rgba
    if position.x + position.width > width or position.y + position.height > height:
        return rgba

    alpha_map = get_alpha_map_for_size(config.logo_size)
    out = rgba.copy()
    unblend_region(out, alpha_map, position)
    return out
