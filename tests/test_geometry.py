from gemini_watermark.watermark_engine import (
    WatermarkConfig,
    WatermarkPosition,
    detect_watermark_config,
    calculate_watermark_position,
)


def test_detect_config_large():
    cfg = detect_watermark_config(1025, 1025)
    assert cfg == WatermarkConfig(logo_size=96, margin_right=64, margin_bottom=64)


def test_detect_config_small_when_one_side_below_min():
    cfg = detect_watermark_config(159, 200)
    assert cfg == WatermarkConfig(logo_size=48, margin_right=32, margin_bottom=32)


def test_calculate_position():
    cfg = WatermarkConfig(logo_size=48, margin_right=32, margin_bottom=32)
    pos = calculate_watermark_position(512, 512, cfg)
    assert pos == WatermarkPosition(x=432, y=432, width=48, height=48)
