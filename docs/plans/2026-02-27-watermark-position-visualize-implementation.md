# Watermark Position Visualization (Python) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python diagnostic script that outputs annotated images with predicted and best watermark boxes for 48/96 sizes separately, writing results to `/Users/shaolongfei/Downloads/横评PPT_诊断标注`.

**Architecture:** Add `diagnose_watermark_visualize.py` with clip-rate scoring and coarse+fine scan; draw red (pred) and green (best) rectangles on copies of the original image; output two PNGs per image when applicable.

**Tech Stack:** Python 3.x, numpy, Pillow

---

### Task 1: Add clip-rate scoring tests

**Files:**
- Create: `tests/test_diagnostic_visualize_scoring.py`

**Step 1: Write the failing test**

```python
import numpy as np
from diagnose_watermark_visualize import score_clip_rate


def _apply_watermark(original: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_expanded = alpha[..., None]
    watermarked = alpha_expanded * 255.0 + (1.0 - alpha_expanded) * original
    return watermarked.astype(np.float32)


def test_score_clip_rate_zero_when_no_clipping():
    original = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.float32,
    )
    alpha = np.full((2, 2), 0.5, dtype=np.float32)
    watermarked = _apply_watermark(original, alpha)
    score = score_clip_rate(watermarked, alpha)
    assert score == 0.0


def test_score_clip_rate_positive_when_clipped():
    watermarked = np.zeros((2, 2, 3), dtype=np.float32)
    alpha = np.full((2, 2), 0.5, dtype=np.float32)
    score = score_clip_rate(watermarked, alpha)
    assert score == 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostic_visualize_scoring.py::test_score_clip_rate_zero_when_no_clipping -v`

Expected: FAIL (ImportError or function missing).

**Step 3: Commit test later after passing**

---

### Task 2: Add scan helper tests

**Files:**
- Create: `tests/test_diagnostic_visualize_scan.py`

**Step 1: Write failing test**

```python
import numpy as np
from diagnose_watermark_visualize import find_best_position


def test_find_best_position_prefers_low_clip_rate():
    rgb = np.full((10, 10, 3), 180, dtype=np.float32)
    alpha = np.full((3, 3), 0.5, dtype=np.float32)

    # Inject clipping at one candidate location
    rgb[5:8, 5:8] = 0

    pred_x, pred_y = 4, 4
    best_x, best_y, score = find_best_position(
        rgb=rgb,
        alpha=alpha,
        pred_x=pred_x,
        pred_y=pred_y,
        size=3,
        radius=2,
        coarse_step=1,
    )

    assert (best_x, best_y) != (5, 5)
    assert score == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostic_visualize_scan.py::test_find_best_position_prefers_low_clip_rate -v`

Expected: FAIL (ImportError or function missing).

---

### Task 3: Implement script skeleton + helpers

**Files:**
- Create: `diagnose_watermark_visualize.py`

**Step 1: Add minimal stubs to satisfy imports**

```python
import numpy as np


def score_clip_rate(rgb: np.ndarray, alpha: np.ndarray) -> float:
    return 1.0


def find_best_position(*, rgb, alpha, pred_x, pred_y, size, radius, coarse_step):
    return 0, 0, 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostic_visualize_scoring.py tests/test_diagnostic_visualize_scan.py -v`

Expected: FAIL with assertions.

**Step 3: Implement correct logic**

Implement `score_clip_rate` (clip-rate), constants `ALPHA_THRESHOLD`, `MAX_ALPHA`, `LOGO_VALUE`, and `find_best_position` with scanning and clamping.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostic_visualize_scoring.py tests/test_diagnostic_visualize_scan.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add diagnose_watermark_visualize.py tests/test_diagnostic_visualize_scoring.py tests/test_diagnostic_visualize_scan.py
git commit -m "feat: add visualize scoring and scan helpers"
```

---

### Task 4: Add visualization output and integration test

**Files:**
- Modify: `diagnose_watermark_visualize.py`
- Create: `tests/test_diagnostic_visualize_output.py`

**Step 1: Write failing test**

```python
from pathlib import Path

from PIL import Image

from diagnose_watermark_visualize import process_directory


def test_process_directory_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    img.save(input_dir / "sample.png")

    process_directory(str(input_dir), str(output_dir))

    assert (output_dir / "sample_48.png").exists()
    assert not (output_dir / "sample_96.png").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostic_visualize_output.py::test_process_directory_writes_outputs -v`

Expected: FAIL (ImportError or missing function).

**Step 3: Implement full script**

Implement:
- constants for paths and colors
- `predicted_position`
- `annotate_image` to draw rectangles
- `process_image` to scan and output PNGs for 48 and 96 separately
- `process_directory` to iterate top-level images
- `main()` calling `process_directory` with hardcoded paths

**Step 4: Run tests**

Run: `pytest -q`

Expected: PASS

**Step 5: Commit**

```bash
git add diagnose_watermark_visualize.py tests/test_diagnostic_visualize_output.py
git commit -m "feat: add watermark visualization outputs"
```

---

### Task 5: Add usage docs

**Files:**
- Modify or Create: `README.md`

**Step 1: Add section**

```
## Watermark Position Visualization

```bash
python diagnose_watermark_visualize.py
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add visualization usage"
```
