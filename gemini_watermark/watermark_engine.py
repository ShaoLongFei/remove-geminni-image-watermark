from dataclasses import dataclass


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
    if image_width > 1024 and image_height > 1024:
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
