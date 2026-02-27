# JS Demo Watermark Removal (Node + Jimp) Design

**Date:** 2026-02-27

## Goal
Create a Node.js demo script that ports the upstream JS watermark-removal logic and processes all top-level images in a given directory. The demo will help verify whether the original JS logic produces correct results on the provided images.

## Scope and Constraints
- Use upstream JS algorithm verbatim (alpha map calculation, reverse alpha blending, watermark geometry rules).
- Process only top-level files (no recursion).
- Inputs/outputs are fixed paths:
  - Input: `/Users/shaolongfei/Downloads/横评PPT`
  - Output: `/Users/shaolongfei/Downloads/横评PPT_去除水印_js`
- Dependencies: Node.js + `jimp` only.
- Use existing assets from this repo: `gemini_watermark/assets/bg_48.png` and `bg_96.png`.

## Architecture
Single script `demo_js_remove_watermark.js` with the following components:

1. **Alpha Map**
   - `calculateAlphaMap(imageData)` mirrors `alphaMap.js` from upstream.
   - Uses max(R, G, B) / 255 for each pixel.
   - Caches alpha maps by size.

2. **Geometry**
   - `detectWatermarkConfig(width, height)`
   - `calculateWatermarkPosition(width, height, config)`
   - Mirrors README rules: 96/64 when both dimensions > 1024, else 48/32.

3. **Reverse Blending**
   - `removeWatermark(imageData, alphaMap, position)`
   - Same constants: `ALPHA_THRESHOLD = 0.002`, `MAX_ALPHA = 0.99`, `LOGO_VALUE = 255`.

4. **Batch Processing**
   - Top-level directory scan only.
   - Filter image extensions: `.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff`.
   - Use Jimp for read/write and access to RGBA buffer.

## Observability
- Print per-image info: filename, dimensions, config, and watermark position.
- Summary counters for processed/skipped/failed.
- Optional debug flag to print alpha-map min/max for the first image and a small pixel sample before/after.

## Error Handling
- Input directory must exist (else error and exit).
- Output directory created if missing.
- Per-file errors captured and reported without stopping the batch.

## Testing
- Minimal Node.js tests for:
  - `detectWatermarkConfig`
  - `calculateWatermarkPosition`
  - `calculateAlphaMap` basic correctness

## Usage
Run the demo:

```bash
node demo_js_remove_watermark.js
```
