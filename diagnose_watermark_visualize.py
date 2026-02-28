from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gemini_watermark.alpha_map import get_alpha_map_for_size

INPUT_DIR = Path("/Users/shaolongfei/Downloads/横评PPT")
OUTPUT_DIR = Path("/Users/shaolongfei/Downloads/横评PPT_诊断标注")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

PRED_COLOR = (255, 59, 48)
BEST_COLOR = (52, 199, 89)
LABEL_BG = (0, 0, 0)
LINE_WIDTH = 2

ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255.0


def score_clip_rate(rgb: np.ndarray, alpha: np.ndarray) -> float:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected rgb array with shape (H, W, 3)")
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("Alpha map shape does not match rgb region size")

    alpha = alpha.astype(np.float32)
    mask = alpha >= ALPHA_THRESHOLD
    if not np.any(mask):
        return 1.0

    alpha = np.minimum(alpha, MAX_ALPHA)
    one_minus = 1.0 - alpha
    original = (rgb.astype(np.float32) - alpha[..., None] * LOGO_VALUE) / one_minus[
        ..., None
    ]
    clipped = (original < 0) | (original > 255)

    clipped_channels = int(np.sum(clipped[mask]))
    total_channels = int(np.sum(mask)) * 3
    return float(clipped_channels) / float(total_channels)


def find_best_position(
    *,
    rgb: np.ndarray,
    alpha: np.ndarray,
    pred_x: int,
    pred_y: int,
    size: int,
    radius: int,
    coarse_step: int,
) -> tuple[int, int, float]:
    height, width = rgb.shape[0], rgb.shape[1]
    best_x, best_y, best_score = pred_x, pred_y, 1.0

    def clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, val))

    start_x = clamp(pred_x - radius, 0, width - size)
    end_x = clamp(pred_x + radius, 0, width - size)
    start_y = clamp(pred_y - radius, 0, height - size)
    end_y = clamp(pred_y + radius, 0, height - size)

    for y in range(start_y, end_y + 1, coarse_step):
        for x in range(start_x, end_x + 1, coarse_step):
            patch = rgb[y : y + size, x : x + size, :3]
            score = score_clip_rate(patch, alpha)
            if score < best_score:
                best_x, best_y, best_score = x, y, score

    return best_x, best_y, best_score


def predicted_position(width: int, height: int, size: int) -> tuple[int, int]:
    margin = 64 if size == 96 else 32
    return width - margin - size, height - margin - size


def clamp_position(width: int, height: int, size: int, x: int, y: int) -> tuple[int, int]:
    max_x = max(0, width - size)
    max_y = max(0, height - size)
    return max(0, min(max_x, x)), max(0, min(max_y, y))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 2
    label_w = text_w + pad * 2
    label_h = text_h + pad * 2

    tx = x
    ty = y - label_h - 2
    if ty < 0:
        ty = y + 2
    if tx + label_w > image_size[0]:
        tx = max(0, image_size[0] - label_w)
    if ty + label_h > image_size[1]:
        ty = max(0, image_size[1] - label_h)

    draw.rectangle([tx, ty, tx + label_w, ty + label_h], fill=LABEL_BG)
    draw.text((tx + pad, ty + pad), text, fill=color, font=font)


def annotate_image(
    img: Image.Image,
    *,
    pred_x: int,
    pred_y: int,
    best_x: int,
    best_y: int,
    size: int,
    score: float,
) -> None:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    width, height = img.size

    pred_x, pred_y = clamp_position(width, height, size, pred_x, pred_y)
    best_x, best_y = clamp_position(width, height, size, best_x, best_y)

    pred_rect = [pred_x, pred_y, pred_x + size - 1, pred_y + size - 1]
    best_rect = [best_x, best_y, best_x + size - 1, best_y + size - 1]

    draw.rectangle(pred_rect, outline=PRED_COLOR, width=LINE_WIDTH)
    draw.rectangle(best_rect, outline=BEST_COLOR, width=LINE_WIDTH)

    pred_label = f"pred ({pred_x},{pred_y})"
    best_label = f"best ({best_x},{best_y}) score={score:.4f}"

    _draw_label(draw, pred_label, pred_x, pred_y, PRED_COLOR, (width, height), font)
    _draw_label(draw, best_label, best_x, best_y, BEST_COLOR, (width, height), font)


def process_image(path: Path, output_dir: Path) -> None:
    img = Image.open(path).convert("RGB")
    rgb = np.array(img, dtype=np.float32)
    height, width = rgb.shape[0], rgb.shape[1]

    for size in (48, 96):
        if width < size or height < size:
            continue

        pred_x, pred_y = predicted_position(width, height, size)
        pred_x, pred_y = clamp_position(width, height, size, pred_x, pred_y)
        alpha = get_alpha_map_for_size(size).astype(np.float32)

        coarse_x, coarse_y, _ = find_best_position(
            rgb=rgb,
            alpha=alpha,
            pred_x=pred_x,
            pred_y=pred_y,
            size=size,
            radius=128,
            coarse_step=4,
        )
        fine_x, fine_y, fine_score = find_best_position(
            rgb=rgb,
            alpha=alpha,
            pred_x=coarse_x,
            pred_y=coarse_y,
            size=size,
            radius=4,
            coarse_step=1,
        )
        fine_x, fine_y = clamp_position(width, height, size, fine_x, fine_y)

        annotated = img.copy()
        annotate_image(
            annotated,
            pred_x=pred_x,
            pred_y=pred_y,
            best_x=fine_x,
            best_y=fine_y,
            size=size,
            score=fine_score,
        )

        output_path = output_dir / f"{path.stem}_{size}.png"
        annotated.save(output_path, format="PNG")

        dx = fine_x - pred_x
        dy = fine_y - pred_y
        print(
            f"{path.name} {width}x{height} | size={size} "
            f"pred=({pred_x},{pred_y}) best=({fine_x},{fine_y}) "
            f"dx={dx} dy={dy} score={fine_score:.4f}"
        )


def process_directory(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    for entry in input_path.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            process_image(entry, output_path)


def main() -> None:
    process_directory(str(INPUT_DIR), str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
