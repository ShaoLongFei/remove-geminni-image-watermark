from gemini_watermark.watermark_engine import detect_watermark_config


def test_detect_watermark_config_prefers_96_when_fits() -> None:
    config = detect_watermark_config(256, 256)
    assert config.logo_size == 96
    assert config.margin_right == 64
    assert config.margin_bottom == 64


def test_detect_watermark_config_falls_back_to_48_when_too_small() -> None:
    config = detect_watermark_config(128, 200)
    assert config.logo_size == 48
    assert config.margin_right == 32
    assert config.margin_bottom == 32
