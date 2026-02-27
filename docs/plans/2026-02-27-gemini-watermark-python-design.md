# Gemini Watermark Remover (Python) Design

**Date:** 2026-02-27

## Goal
Provide a small Python library that removes the Gemini image watermark by porting the upstream JS algorithm. The public API is a single function:

```
remove_watermark(input_path: str, output_path: str) -> None
```

It reads from a file path and writes the processed output to an explicit file path.

## Scope and Constraints
- Inputs are presumed to be Gemini-generated images using the known watermark geometry and alpha behavior.
- No watermark detection or adaptive heuristics are added.
- Dependencies are limited to Pillow and numpy.
- Output format matches the input format (extension and Pillow-inferred format).

## Architecture
Modules mirror the upstream JS structure for 1:1 logic mapping:

1. `alpha_map.py`
   - Port of JS `alphaMap` logic.
   - Computes an alpha map based on image width/height and fixed constants.
   - Exposes `compute_alpha_map(width, height) -> np.ndarray` with float32 values in [0, 1].

2. `blend_modes.py`
   - Port of JS `blendModes` logic.
   - Implements inverse alpha blending (“unblend”) and helpers for clamping and channel math.

3. `watermark_engine.py`
   - Orchestrates watermark removal for the computed region.
   - Applies inverse blending only to the watermark area, leaving the rest untouched.

4. `image_io.py`
   - Reads input file with Pillow, converts to RGBA, normalizes to float32 (0..1).
   - Calls the engine, converts back to uint8, writes output using the original format.

5. `__init__.py`
   - Re-exports `remove_watermark` as the public API.

## Data Flow
`input_path -> PIL.Image -> RGBA numpy array -> alpha map + inverse blend -> output array -> PIL.Image -> output_path`

## Error Handling and Edge Cases
- Read/write errors surface directly (no masking).
- Images are converted to RGBA to ensure blending math is valid.
- Values are clamped to [0, 1] to avoid overflow or underflow.
- If watermark region resolves to empty (e.g., very small images), the output is a no-op copy.
- Non-Gemini images may show artifacts; this is an expected limitation.

## Testing
No automated tests are added in this pass. The math is isolated in separate modules to enable future deterministic tests using synthetic images.

## Usage
```
from gemini_watermark import remove_watermark

remove_watermark("input.jpg", "output.jpg")
```
