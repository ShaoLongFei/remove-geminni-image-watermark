# Gemini 图片水印去除工具

[English](README.md)

一个用于去除 Gemini 图片水印的小型 Python 工具集，包含批量去水印和诊断工具。

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 批量去水印：

```bash
python -m scripts.remove_watermark
```

请在 `scripts/remove_watermark.py` 中修改 `INPUT_DIR` 和 `OUTPUT_DIR`。

3. 水印位置诊断：

```bash
python -m scripts.diagnostics.position_scan
```

请在 `scripts/diagnostics/position_scan.py` 中修改 `INPUT_DIR`。

4. 位置可视化：

```bash
python -m scripts.diagnostics.position_visualize
```

请在 `scripts/diagnostics/position_visualize.py` 中修改 `INPUT_DIR` 和 `OUTPUT_DIR`。

## 限制说明

- 仅适用于 Gemini 生成且水印几何与透明度一致的图片。
- 图片过小时 96px 水印可能不适用，会回退到 48px；结果可能有差异。
- 非 Gemini 或被改动的图片可能效果不佳。
