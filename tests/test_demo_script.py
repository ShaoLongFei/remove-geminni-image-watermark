from pathlib import Path

from PIL import Image

from demo_remove_watermark import process_directory


def test_process_directory_top_level_only(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image_path = input_dir / "sample.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(image_path)

    (input_dir / "note.txt").write_text("skip me")

    subdir = input_dir / "sub"
    subdir.mkdir()
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(subdir / "nested.png")

    process_directory(str(input_dir), str(output_dir))

    assert output_dir.exists()
    assert (output_dir / "sample.png").exists()
    assert not (output_dir / "note.txt").exists()
    assert not (output_dir / "sub" / "nested.png").exists()
