# Gemini Watermark Remover (Python) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python library function `remove_watermark(input_path, output_path)` that removes the Gemini watermark by porting the upstream JS algorithm.

**Architecture:** Port JS modules into Python: alpha map calculation from background images, reverse alpha blending for the watermark region, and an IO wrapper using Pillow + numpy. Include bg_48/bg_96 assets and cache alpha maps.

**Tech Stack:** Python 3.x, Pillow, numpy, pytest

---

### Task 1: Watermark geometry helpers

**Files:**
- Create: `gemini_watermark/watermark_engine.py`
- Create: `tests/test_geometry.py`

**Step 1: Write the failing tests**

```python
from gemini_watermark.watermark_engine import (
    WatermarkConfig,
    WatermarkPosition,
    detect_watermark_config,
    calculate_watermark_position,
)


def test_detect_config_large():
    cfg = detect_watermark_config(1025, 1025)
    assert cfg == WatermarkConfig(logo_size=96, margin_right=64, margin_bottom=64)


def test_detect_config_small_when_one_side_1024():
    cfg = detect_watermark_config(1024, 1025)
    assert cfg == WatermarkConfig(logo_size=48, margin_right=32, margin_bottom=32)


def test_calculate_position():
    cfg = WatermarkConfig(logo_size=48, margin_right=32, margin_bottom=32)
    pos = calculate_watermark_position(512, 512, cfg)
    assert pos == WatermarkPosition(x=432, y=432, width=48, height=48)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry.py::test_detect_config_large -v`

Expected: FAIL with ImportError or AttributeError because module/functions do not exist.

**Step 3: Add minimal stubs to make tests run (still fail)**

```python
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
    return WatermarkConfig(0, 0, 0)


def calculate_watermark_position(
    image_width: int,
    image_height: int,
    config: WatermarkConfig,
) -> WatermarkPosition:
    return WatermarkPosition(0, 0, 0, 0)
```

**Step 4: Run tests to verify they fail correctly**

Run: `pytest tests/test_geometry.py -v`

Expected: FAIL with assertion error (wrong values).

**Step 5: Implement minimal correct logic**

```python
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
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_geometry.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add gemini_watermark/watermark_engine.py tests/test_geometry.py
git commit -m "feat: add watermark geometry helpers"
```

---

### Task 2: Alpha map computation from RGBA

**Files:**
- Create: `gemini_watermark/alpha_map.py`
- Create: `tests/test_alpha_map.py`

**Step 1: Write the failing test**

```python
import numpy as np
from gemini_watermark.alpha_map import calculate_alpha_map_from_rgba


def test_calculate_alpha_map_from_rgba():
    rgba = np.array([
        [[0, 10, 5, 255], [255, 0, 0, 255]]
    ], dtype=np.uint8)
    alpha = calculate_alpha_map_from_rgba(rgba)
    assert alpha.shape == (1, 2)
    assert np.allclose(alpha, np.array([[10 / 255.0, 1.0]], dtype=np.float32))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_alpha_map.py::test_calculate_alpha_map_from_rgba -v`

Expected: FAIL with ImportError or AttributeError.

**Step 3: Add minimal stub to make test run (still fail)**

```python
import numpy as np


def calculate_alpha_map_from_rgba(rgba: np.ndarray) -> np.ndarray:
    return np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.float32)
```

**Step 4: Run tests to verify they fail correctly**

Run: `pytest tests/test_alpha_map.py -v`

Expected: FAIL with assertion error (wrong values).

**Step 5: Implement minimal correct logic**

```python
def calculate_alpha_map_from_rgba(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("Expected RGBA image with shape (H, W, 4)")
    rgb = rgba[..., :3].astype(np.float32)
    alpha = np.max(rgb, axis=2) / 255.0
    return alpha.astype(np.float32)
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_alpha_map.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add gemini_watermark/alpha_map.py tests/test_alpha_map.py
git commit -m "feat: add alpha map computation"
```

---

### Task 3: Reverse alpha blending for watermark region

**Files:**
- Create: `gemini_watermark/blend_modes.py`
- Create: `tests/test_blend_modes.py`

**Step 1: Write the failing tests**

```python
import numpy as np
from gemini_watermark.blend_modes import unblend_region
from gemini_watermark.watermark_engine import WatermarkPosition


def _apply_watermark(original_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_expanded = alpha[..., None]
    watermarked = alpha_expanded * 255.0 + (1.0 - alpha_expanded) * original_rgb
    return np.clip(np.rint(watermarked), 0, 255).astype(np.float32)


def test_unblend_recovers_original_within_tolerance():
    original = np.array([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]],
    ], dtype=np.float32)
    alpha = np.array([
        [0.2, 0.4],
        [0.6, 0.8],
    ], dtype=np.float32)
    watermarked = _apply_watermark(original, alpha)
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., :3] = watermarked
    rgba[..., 3] = 255.0

    pos = WatermarkPosition(x=0, y=0, width=2, height=2)
    unblend_region(rgba, alpha, pos)

    diff = np.abs(rgba[..., :3] - original)
    assert diff.max() <= 1.0


def test_unblend_skips_low_alpha():
    original = np.array([[[10, 20, 30]]], dtype=np.float32)
    alpha = np.array([[0.0]], dtype=np.float32)
    rgba = np.zeros((1, 1, 4), dtype=np.float32)
    rgba[..., :3] = original
    rgba[..., 3] = 255.0

    pos = WatermarkPosition(x=0, y=0, width=1, height=1)
    unblend_region(rgba, alpha, pos)

    assert np.allclose(rgba[..., :3], original)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_blend_modes.py::test_unblend_recovers_original_within_tolerance -v`

Expected: FAIL with ImportError or AttributeError.

**Step 3: Add minimal stub to make test run (still fail)**

```python
import numpy as np
from gemini_watermark.watermark_engine import WatermarkPosition

ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255.0


def unblend_region(rgba: np.ndarray, alpha_map: np.ndarray, position: WatermarkPosition) -> None:
    return None
```

**Step 4: Run tests to verify they fail correctly**

Run: `pytest tests/test_blend_modes.py -v`

Expected: FAIL with assertion error.

**Step 5: Implement minimal correct logic**

```python
def unblend_region(rgba: np.ndarray, alpha_map: np.ndarray, position: WatermarkPosition) -> None:
    x, y, width, height = position.x, position.y, position.width, position.height
    if width <= 0 or height <= 0:
        return

    region = rgba[y : y + height, x : x + width, :3]
    alpha = alpha_map.astype(np.float32)

    if alpha.shape != (height, width):
        raise ValueError("alpha_map shape does not match region size")

    mask = alpha >= ALPHA_THRESHOLD
    if not np.any(mask):
        return

    alpha = np.minimum(alpha, MAX_ALPHA)
    one_minus = 1.0 - alpha
    corrected = (region - alpha[..., None] * LOGO_VALUE) / one_minus[..., None]
    corrected = np.clip(np.rint(corrected), 0, 255)

    region[mask] = corrected[mask]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_blend_modes.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add gemini_watermark/blend_modes.py tests/test_blend_modes.py
git commit -m "feat: add reverse alpha blending"
```

---

### Task 4: Add background assets and alpha map loader

**Files:**
- Create: `gemini_watermark/assets/bg_48.png`
- Create: `gemini_watermark/assets/bg_96.png`
- Modify: `gemini_watermark/alpha_map.py`
- Create: `tests/test_alpha_map_loader.py`

**Step 1: Add assets from upstream repo**

```bash
mkdir -p gemini_watermark/assets
curl -L -o gemini_watermark/assets/bg_48.png https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_48.png
curl -L -o gemini_watermark/assets/bg_96.png https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_96.png
```

**Step 2: Write the failing test**

```python
import numpy as np
from gemini_watermark.alpha_map import get_alpha_map_for_size


def test_get_alpha_map_for_size_shape_and_range():
    alpha = get_alpha_map_for_size(48)
    assert alpha.shape == (48, 48)
    assert alpha.dtype == np.float32
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_alpha_map_loader.py::test_get_alpha_map_for_size_shape_and_range -v`

Expected: FAIL with AttributeError (function missing) or error due to missing loader.

**Step 4: Add minimal stub to make test run (still fail)**

```python
from pathlib import Path
from PIL import Image

_ALPHA_MAP_CACHE: dict[int, np.ndarray] = {}


def get_alpha_map_for_size(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)
```

**Step 5: Run tests to verify they fail correctly**

Run: `pytest tests/test_alpha_map_loader.py -v`

Expected: FAIL with assertion error (range/shape if incorrect).

**Step 6: Implement minimal correct logic**

```python
def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def get_alpha_map_for_size(size: int) -> np.ndarray:
    if size in _ALPHA_MAP_CACHE:
        return _ALPHA_MAP_CACHE[size]
    if size not in (48, 96):
        raise ValueError("Unsupported watermark size")

    path = _assets_dir() / f"bg_{size}.png"
    with Image.open(path) as img:
        rgba = np.array(img.convert("RGBA"), dtype=np.float32)

    alpha = calculate_alpha_map_from_rgba(rgba)
    _ALPHA_MAP_CACHE[size] = alpha
    return alpha
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_alpha_map_loader.py -v`

Expected: PASS

**Step 8: Commit**

```bash
git add gemini_watermark/assets/bg_48.png gemini_watermark/assets/bg_96.png gemini_watermark/alpha_map.py tests/test_alpha_map_loader.py
git commit -m "feat: add alpha map loader and assets"
```

---

### Task 5: End-to-end remove_watermark path and public API

**Files:**
- Create: `gemini_watermark/image_io.py`
- Modify: `gemini_watermark/watermark_engine.py`
- Modify: `gemini_watermark/__init__.py`
- Create: `tests/test_remove_watermark.py`

**Step 1: Write the failing test**

```python
import numpy as np
from PIL import Image
from gemini_watermark.alpha_map import get_alpha_map_for_size
from gemini_watermark.watermark_engine import (
    detect_watermark_config,
    calculate_watermark_position,
)
from gemini_watermark.image_io import remove_watermark


def _apply_watermark(rgba: np.ndarray, alpha_map: np.ndarray, pos) -> np.ndarray:
    out = rgba.copy().astype(np.float32)
    region = out[pos.y : pos.y + pos.height, pos.x : pos.x + pos.width, :3]
    alpha = alpha_map.astype(np.float32)
    watermarked = alpha[..., None] * 255.0 + (1.0 - alpha[..., None]) * region
    out[pos.y : pos.y + pos.height, pos.x : pos.x + pos.width, :3] = np.clip(
        np.rint(watermarked), 0, 255
    )
    return out


def test_remove_watermark_recovers_background(tmp_path):
    size = 128
    original = np.zeros((size, size, 4), dtype=np.float32)
    original[..., :3] = [50, 100, 150]
    original[..., 3] = 255.0

    cfg = detect_watermark_config(size, size)
    pos = calculate_watermark_position(size, size, cfg)
    alpha = get_alpha_map_for_size(cfg.logo_size)
    watermarked = _apply_watermark(original, alpha, pos)

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(watermarked.astype(np.uint8), mode="RGBA").save(input_path)

    remove_watermark(str(input_path), str(output_path))

    result = np.array(Image.open(output_path).convert("RGBA"), dtype=np.float32)
    diff = np.abs(result - original)
    assert diff.max() <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_remove_watermark.py::test_remove_watermark_recovers_background -v`

Expected: FAIL with ImportError or AttributeError.

**Step 3: Add minimal stubs to make tests run (still fail)**

```python
# gemini_watermark/image_io.py
from typing import Optional


def remove_watermark(input_path: str, output_path: str) -> None:
    raise NotImplementedError
```

**Step 4: Run tests to verify they fail correctly**

Run: `pytest tests/test_remove_watermark.py -v`

Expected: FAIL with NotImplementedError.

**Step 5: Implement minimal correct logic**

```python
# gemini_watermark/watermark_engine.py
import numpy as np
from gemini_watermark.alpha_map import get_alpha_map_for_size
from gemini_watermark.blend_modes import unblend_region


def remove_watermark_from_array(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[0], rgba.shape[1]
    config = detect_watermark_config(width, height)
    position = calculate_watermark_position(width, height, config)

    if position.x < 0 or position.y < 0:
        return rgba
    if position.x + position.width > width or position.y + position.height > height:
        return rgba

    alpha_map = get_alpha_map_for_size(config.logo_size)
    out = rgba.copy()
    unblend_region(out, alpha_map, position)
    return out
```

```python
# gemini_watermark/image_io.py
from pathlib import Path
from PIL import Image
import numpy as np
from gemini_watermark.watermark_engine import remove_watermark_from_array


def remove_watermark(input_path: str, output_path: str) -> None:
    input_path = str(input_path)
    output_path = str(output_path)

    with Image.open(input_path) as img:
        src_format = img.format
        rgba = img.convert("RGBA")

    data = np.array(rgba, dtype=np.float32)
    result = remove_watermark_from_array(data)
    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)

    out_img = Image.fromarray(result, mode="RGBA")

    if src_format and src_format.upper() in {"JPEG", "JPG"}:
        out_img = out_img.convert("RGB")

    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"format": src_format} if src_format else {}
    out_img.save(output_path, **save_kwargs)
```

```python
# gemini_watermark/__init__.py
from gemini_watermark.image_io import remove_watermark

__all__ = ["remove_watermark"]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_remove_watermark.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add gemini_watermark/image_io.py gemini_watermark/watermark_engine.py gemini_watermark/__init__.py tests/test_remove_watermark.py
git commit -m "feat: add remove_watermark API"
```

---

### Task 6: Add dependency manifests (runtime + dev)

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

**Step 1: Write requirements files**

```
# requirements.txt
numpy
Pillow
```

```
# requirements-dev.txt
pytest
```

**Step 2: Commit**

```bash
git add requirements.txt requirements-dev.txt
git commit -m "chore: add python requirements"
```

---

## Notes
- Follow @superpowers:test-driven-development strictly during implementation.
- If pytest is not available, install with `pip install -r requirements-dev.txt` before running tests.
