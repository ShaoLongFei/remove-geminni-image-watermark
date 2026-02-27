# Watermark Position Diagnostic (Python) Design

**Date:** 2026-02-27

## Goal
Create a Python diagnostic script that automatically scans for the true Gemini watermark position by comparing observed pixel constraints against the expected alpha map. The script outputs per-image best coordinates and offsets to help confirm whether the watermark location rules are wrong.

## Scope and Constraints
- Diagnostic only; does not modify images or the removal logic.
- Processes top-level image files in `/Users/shaolongfei/Downloads/横评PPT`.
- Outputs results to terminal logs (no CSV).
- Scans around predicted position within ±128 pixels.
- Uses existing `bg_48.png` and `bg_96.png` alpha maps from the repo.

## Core Insight (Scoring)
From watermark formula:

```
watermarked = alpha * 255 + (1 - alpha) * original
```

We derive a pixel constraint:

```
min(R, G, B) >= alpha * 255
```

So for a candidate region, count the fraction of pixels where:

```
min_rgb < alpha*255 - epsilon
```

Lower violation rate → more likely watermark location.

## Architecture
Single script `diagnose_watermark_position.py` with:

1. **Image loading** using Pillow and numpy.
2. **Alpha maps** loaded via existing asset logic (48/96).
3. **Prediction** using current watermark rules (48/96 size + margins).
4. **Two-stage scan**
   - Coarse step (4 px) within ±128 px window
   - Fine step (1 px) within ±4 px of best coarse location
5. **Logging** per image with predicted vs best location and score.

## Output Format (Per Image)
Example:

```
1.png 1376x768 | size=48 pred=(1296,688) best=(1292,684) dx=-4 dy=-4 score=0.032
```

## Error Handling
- Input dir must exist and be a directory; else error.
- Non-image files are skipped.
- Images smaller than the watermark size are skipped for that size.
- Candidate positions clipped to valid image bounds.

## Performance
The scan is limited to a ±128 px window with coarse+fine search to keep runtime reasonable across batches.

## Future Use
If the diagnostic consistently finds systematic offsets (dx/dy), those offsets can be used to update the removal logic.
