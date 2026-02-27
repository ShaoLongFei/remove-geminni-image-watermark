import os
from pathlib import Path

from gemini_watermark import remove_watermark

INPUT_DIR = Path("/Users/shaolongfei/Downloads/横评PPT")
OUTPUT_DIR = Path("/Users/shaolongfei/Downloads/横评PPT_去除水印")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def process_directory(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    failures: list[str] = []

    for entry in os.scandir(input_path):
        if not entry.is_file():
            continue

        ext = Path(entry.name).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            skipped += 1
            continue

        src = Path(entry.path)
        dst = output_path / entry.name
        try:
            remove_watermark(str(src), str(dst))
            processed += 1
        except Exception as exc:  # noqa: BLE001 - demo script should keep going
            failed += 1
            failures.append(f"{entry.name}: {exc}")

    print(
        "Done. "
        f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}. "
        f"Output: {output_path}"
    )
    if failures:
        print("Failures:")
        for item in failures:
            print(f"- {item}")


def main() -> None:
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    process_directory(str(INPUT_DIR), str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
