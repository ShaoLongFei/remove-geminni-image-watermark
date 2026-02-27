from pathlib import Path

import numpy as np
from PIL import Image

from gemini_watermark.alpha_map import get_alpha_map_for_size

INPUT_DIR = Path("/Users/shaolongfei/Downloads/横评PPT")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def score_violation_rate(min_rgb: np.ndarray, alpha: np.ndarray, epsilon: float) -> float:
    threshold = alpha * 255.0 - epsilon
    violations = min_rgb < threshold
    return float(np.sum(violations)) / float(violations.size)


def find_best_position(
    *,
    min_rgb: np.ndarray,
    alpha: np.ndarray,
    pred_x: int,
    pred_y: int,
    size: int,
    radius: int,
    coarse_step: int,
    epsilon: float,
) -> tuple[int, int, float]:
    height, width = min_rgb.shape
    best_x, best_y, best_score = pred_x, pred_y, 1.0

    def clamp(val: int, lo: int, hi: int) -> int:
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


def predicted_position(width: int, height: int, size: int) -> tuple[int, int]:
    if size == 96:
        margin = 64
    else:
        margin = 32
    return width - margin - size, height - margin - size


def process_image(path: Path) -> None:
    img = Image.open(path).convert("RGBA")
    rgba = np.array(img, dtype=np.float32)
    min_rgb = np.min(rgba[..., :3], axis=2)
    height, width = min_rgb.shape

    for size in (48, 96):
        if width < size or height < size:
            continue

        pred_x, pred_y = predicted_position(width, height, size)
        alpha = get_alpha_map_for_size(size)

        coarse_x, coarse_y, _ = find_best_position(
            min_rgb=min_rgb,
            alpha=alpha,
            pred_x=pred_x,
            pred_y=pred_y,
            size=size,
            radius=128,
            coarse_step=4,
            epsilon=1.0,
        )
        fine_x, fine_y, fine_score = find_best_position(
            min_rgb=min_rgb,
            alpha=alpha,
            pred_x=coarse_x,
            pred_y=coarse_y,
            size=size,
            radius=4,
            coarse_step=1,
            epsilon=1.0,
        )

        dx = fine_x - pred_x
        dy = fine_y - pred_y
        print(
            f"{path.name} {width}x{height} | size={size} "
            f"pred=({pred_x},{pred_y}) best=({fine_x},{fine_y}) "
            f"dx={dx} dy={dy} score={fine_score:.4f}"
        )


def main() -> None:
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    for entry in INPUT_DIR.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            process_image(entry)


if __name__ == "__main__":
    main()
