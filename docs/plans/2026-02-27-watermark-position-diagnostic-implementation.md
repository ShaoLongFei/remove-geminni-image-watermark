# Watermark Position Diagnostic (Python) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python diagnostic script that scans for the most likely Gemini watermark position using alpha-map constraint violations and logs per-image results.

**Architecture:** Add a single script `diagnose_watermark_position.py` that loads images from the input directory, computes alpha maps (48/96), scans a ±128 px window around the predicted position with coarse+fine steps, and prints best coordinates and offsets.

**Tech Stack:** Python 3.x, numpy, Pillow

---

### Task 1: Add core scoring helper and tests

**Files:**
- Create: `diagnose_watermark_position.py`
- Create: `tests/test_diagnostic_scoring.py`

**Step 1: Write the failing test**

```python
import numpy as np
from diagnose_watermark_position import score_violation_rate


def test_score_violation_rate_zero_when_constraint_satisfied():
    min_rgb = np.array([[200, 200], [200, 200]], dtype=np.float32)
    alpha = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    score = score_violation_rate(min_rgb, alpha, epsilon=1.0)
    assert score == 0.0


def test_score_violation_rate_positive_when_violated():
    min_rgb = np.array([[10, 200], [200, 10]], dtype=np.float32)
    alpha = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    score = score_violation_rate(min_rgb, alpha, epsilon=1.0)
    assert score == 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostic_scoring.py::test_score_violation_rate_zero_when_constraint_satisfied -v`

Expected: FAIL (ImportError or function missing).

**Step 3: Add minimal stub to make test run (still fail)**

```python
import numpy as np


def score_violation_rate(min_rgb: np.ndarray, alpha: np.ndarray, epsilon: float) -> float:
    return 1.0
```

**Step 4: Run tests to verify they fail correctly**

Run: `pytest tests/test_diagnostic_scoring.py -v`

Expected: FAIL with assertion errors.

**Step 5: Implement minimal correct logic**

```python
def score_violation_rate(min_rgb: np.ndarray, alpha: np.ndarray, epsilon: float) -> float:
    threshold = alpha * 255.0 - epsilon
    violations = min_rgb < threshold
    return float(np.sum(violations)) / float(violations.size)
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_diagnostic_scoring.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add diagnose_watermark_position.py tests/test_diagnostic_scoring.py
git commit -m "feat: add diagnostic scoring helper"
```

---

### Task 2: Implement scanning logic and CLI behavior

**Files:**
- Modify: `diagnose_watermark_position.py`
- Create: `tests/test_diagnostic_scan.py`

**Step 1: Write failing test**

```python
import numpy as np
from diagnose_watermark_position import find_best_position


def test_find_best_position_prefers_low_violation():
    min_rgb = np.full((10, 10), 200, dtype=np.float32)
    alpha = np.full((3, 3), 0.5, dtype=np.float32)

    # Inject violations at one candidate location
    min_rgb[5:8, 5:8] = 0

    pred_x, pred_y = 4, 4
    best_x, best_y, score = find_best_position(
        min_rgb=min_rgb,
        alpha=alpha,
        pred_x=pred_x,
        pred_y=pred_y,
        size=3,
        radius=2,
        coarse_step=1,
        epsilon=1.0,
    )

    # Best should avoid the violated block at (5,5)
    assert (best_x, best_y) != (5, 5)
    assert score <= 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostic_scan.py::test_find_best_position_prefers_low_violation -v`

Expected: FAIL (ImportError or missing function).

**Step 3: Implement minimal scan stub (still fail)**

```python
def find_best_position(*args, **kwargs):
    return 0, 0, 1.0
```

**Step 4: Run tests to verify they fail correctly**

Run: `pytest tests/test_diagnostic_scan.py -v`

Expected: FAIL with assertion errors.

**Step 5: Implement minimal correct scan**

```python
def find_best_position(
    min_rgb: np.ndarray,
    alpha: np.ndarray,
    pred_x: int,
    pred_y: int,
    size: int,
    radius: int,
    coarse_step: int,
    epsilon: float,
):
    height, width = min_rgb.shape
    best_x, best_y, best_score = pred_x, pred_y, 1.0

    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    start_x = clamp(pred_x - radius, 0, width - size)
    end_x = clamp(pred_x + radius, 0, width - size)
    start_y = clamp(pred_y - radius, 0, height - size)
    end_y = clamp(pred_y + radius, 0, height - size)

    for y in range(start_y, end_y + 1, coarse_step):
        for x in range(start_x, end_x + 1, coarse_step):
            patch = min_rgb[y : y + size, x : x + size]
            score = score_violation_rate(patch, alpha, epsilon)
            if score < best_score:
                best_x, best_y, best_score = x, y, score

    return best_x, best_y, best_score
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_diagnostic_scan.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add diagnose_watermark_position.py tests/test_diagnostic_scan.py
git commit -m "feat: add diagnostic scan helper"
```

---

### Task 3: Integrate coarse+fine scan with real images

**Files:**
- Modify: `diagnose_watermark_position.py`

**Step 1: Add full script flow**

```python
from pathlib import Path
from PIL import Image
import numpy as np
from gemini_watermark.alpha_map import get_alpha_map_for_size
from gemini_watermark.watermark_engine import detect_watermark_config, calculate_watermark_position

INPUT_DIR = Path("/Users/shaolongfei/Downloads/横评PPT")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def find_best_position_fine(...):
    # coarse + fine around best
    ...


def process_image(path: Path):
    img = Image.open(path).convert("RGBA")
    rgba = np.array(img, dtype=np.float32)
    min_rgb = np.min(rgba[..., :3], axis=2)
    height, width = min_rgb.shape

    for size in (48, 96):
        if width < size or height < size:
            continue
        config = detect_watermark_config(width, height)
        if config.logo_size != size:
            # still scan both sizes for diagnostic
            pass
        pred = calculate_watermark_position(width, height, config)
        alpha = get_alpha_map_for_size(size)

        best_x, best_y, score = find_best_position(...radius=128, coarse_step=4)
        fine_x, fine_y, fine_score = find_best_position(...radius=4, coarse_step=1)

        dx = fine_x - pred.x
        dy = fine_y - pred.y
        print(f"{path.name} {width}x{height} | size={size} pred=({pred.x},{pred.y}) best=({fine_x},{fine_y}) dx={dx} dy={dy} score={fine_score:.4f}")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    for entry in INPUT_DIR.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            process_image(entry)

if __name__ == "__main__":
    main()
```

**Step 2: Run tests**

Run: `pytest -q`

Expected: PASS

**Step 3: Commit**

```bash
git add diagnose_watermark_position.py
git commit -m "feat: add diagnostic script for watermark position"
```

---

### Task 4: Add usage note

**Files:**
- Modify: `README.md`

**Step 1: Add section**

```
## Watermark Position Diagnostic

```bash
python diagnose_watermark_position.py
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add diagnostic usage"
```
