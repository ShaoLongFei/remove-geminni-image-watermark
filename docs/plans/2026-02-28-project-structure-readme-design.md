# Project Structure + Bilingual README Design

**Date:** 2026-02-28

## Overview
Reorganize the repository to clearly separate library code from runnable scripts, and add bilingual documentation with English as the default. Keep behavior intact.

## Goals
- Move runnable scripts into `scripts/` and `scripts/diagnostics/`.
- Keep `gemini_watermark/` as the core library.
- Provide an English README with a link to a Chinese README.
- Keep usage simple with fixed input/output paths at top of scripts.

## Non-Goals
- No new CLI framework or packaging changes.
- No new features in the watermark algorithm.
- No changes to alpha map assets or blending math.

## Structure
- `gemini_watermark/`: core library (alpha maps, blending, removal engine).
- `scripts/remove_watermark.py`: batch removal script (current `demo_remove_watermark.py`).
- `scripts/diagnostics/position_scan.py`: watermark position scan (current `diagnose_watermark_position.py`).
- `scripts/diagnostics/position_visualize.py`: watermark position visualization (current `diagnose_watermark_visualize.py`).
- `README.md`: English default.
- `README.zh-CN.md`: Chinese version.

## Data Flow
Scripts load images from a directory, call library helpers (`remove_watermark` or scanning helpers), then write outputs to a separate directory. The library continues to compute watermark position and unblend using the alpha map.

## Error Handling
Batch scripts should:
- Skip non-image files.
- Create output directory if missing.
- Continue on per-file exceptions and summarize failures.

## Testing
- Update any tests that import script modules to use the new paths.
- Core library tests remain unchanged.

## Documentation
- English README includes: Intro, Usage, Limitations, Project Structure.
- Provide a clear language switch link to `README.zh-CN.md`.
- Chinese README mirrors the same minimal sections.

## Migration Steps (Implementation Outline)
1. Move scripts into `scripts/` and `scripts/diagnostics/`.
2. Update any imports/tests referencing old script locations.
3. Write bilingual READMEs with language switch links.
4. Run tests.
