# Gemini Image Watermark Remover

[中文说明](README.zh-CN.md)

A small Python toolkit for removing Gemini image watermarks via reverse alpha blending. Includes batch removal and diagnostic utilities.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Batch removal:

```bash
python -m scripts.remove_watermark
```

Edit `INPUT_DIR` and `OUTPUT_DIR` in `scripts/remove_watermark.py`.

3. Diagnose watermark position:

```bash
python -m scripts.diagnostics.position_scan
```

Edit `INPUT_DIR` in `scripts/diagnostics/position_scan.py`.

4. Visualize positions:

```bash
python -m scripts.diagnostics.position_visualize
```

Edit `INPUT_DIR` and `OUTPUT_DIR` in `scripts/diagnostics/position_visualize.py`.

## Limitations

- Designed for Gemini-generated images with known watermark geometry and alpha behavior.
- If an image is too small, the 96px watermark may not fit and the 48px fallback is used; results may vary.
- Non-Gemini images or altered content may produce incorrect results.
