# Project Structure + Bilingual README Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize runnable scripts into `scripts/` (and `scripts/diagnostics/`) and add bilingual README files with English default and Chinese link.

**Architecture:** Keep `gemini_watermark/` as the library, move CLI-style scripts into a dedicated scripts tree, and update tests/imports accordingly. Provide README.md (EN) and README.zh-CN.md (ZH) with minimal sections and a language switch.

**Tech Stack:** Python 3.x, pytest, Markdown

---

### Task 1: Update tests to import new script locations (RED)

**Files:**
- Modify: `tests/test_demo_script.py`
- Modify: `tests/test_diagnostic_scoring.py`
- Modify: `tests/test_diagnostic_scan.py`
- Modify: `tests/test_diagnostic_visualize_scoring.py`
- Modify: `tests/test_diagnostic_visualize_scan.py`
- Modify: `tests/test_diagnostic_visualize_output.py`

**Step 1: Update imports (failing test)**

Update import lines to:
- `from scripts.remove_watermark import process_directory`
- `from scripts.diagnostics.position_scan import score_clip_rate, find_best_position`
- `from scripts.diagnostics.position_visualize import score_clip_rate, find_best_position, process_directory`

**Step 2: Run a test to verify it fails**

Run: `pytest tests/test_demo_script.py::test_process_directory_writes_outputs -v`

Expected: FAIL with ImportError for `scripts.*` (modules not found).

**Step 3: Commit**

```bash
git add tests/test_demo_script.py tests/test_diagnostic_scoring.py tests/test_diagnostic_scan.py \
  tests/test_diagnostic_visualize_scoring.py tests/test_diagnostic_visualize_scan.py \
  tests/test_diagnostic_visualize_output.py
git commit -m "test: update script imports for new layout"
```

---

### Task 2: Move scripts into `scripts/` and add packages (GREEN)

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/diagnostics/__init__.py`
- Move: `demo_remove_watermark.py` -> `scripts/remove_watermark.py`
- Move: `diagnose_watermark_position.py` -> `scripts/diagnostics/position_scan.py`
- Move: `diagnose_watermark_visualize.py` -> `scripts/diagnostics/position_visualize.py`

**Step 1: Create directories and move files**

```bash
mkdir -p scripts/diagnostics
mv demo_remove_watermark.py scripts/remove_watermark.py
mv diagnose_watermark_position.py scripts/diagnostics/position_scan.py
mv diagnose_watermark_visualize.py scripts/diagnostics/position_visualize.py
```

**Step 2: Add `__init__.py` files**

Create empty:
- `scripts/__init__.py`
- `scripts/diagnostics/__init__.py`

**Step 3: Run focused tests**

Run: `pytest tests/test_demo_script.py tests/test_diagnostic_scoring.py tests/test_diagnostic_scan.py \
  tests/test_diagnostic_visualize_scoring.py tests/test_diagnostic_visualize_scan.py \
  tests/test_diagnostic_visualize_output.py -v`

Expected: PASS

**Step 4: Commit**

```bash
git add scripts demo_remove_watermark.py diagnose_watermark_position.py diagnose_watermark_visualize.py
# Note: moved files should appear as deletes/adds

git commit -m "refactor: move scripts into scripts/ tree"
```

---

### Task 3: Add bilingual README files

**Files:**
- Modify: `README.md`
- Create: `README.zh-CN.md`

**Step 1: Update English README**

Include:
- Title + short intro
- Language switch link to `README.zh-CN.md`
- Usage: install deps + run scripts with new paths
- Limitations (Gemini-only, position assumptions, small images may skip)
- Project structure snippet

**Step 2: Add Chinese README**

Mirror sections in Chinese and link back to English.

**Step 3: Run tests**

Run: `pytest -q`

Expected: PASS

**Step 4: Commit**

```bash
git add README.md README.zh-CN.md
git commit -m "docs: add bilingual readme"
```

---

### Task 4: Final verification

**Step 1: Run full test suite**

Run: `pytest -q`

Expected: PASS

**Step 2: Report readiness for merge**

Summarize changes and verification output.
