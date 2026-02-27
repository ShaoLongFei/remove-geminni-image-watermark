from pathlib import Path

import numpy as np
from PIL import Image

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
