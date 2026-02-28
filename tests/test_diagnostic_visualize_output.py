from pathlib import Path

from PIL import Image

from scripts.diagnostics.position_visualize import process_directory


def test_process_directory_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    img.save(input_dir / "sample.png")

    process_directory(str(input_dir), str(output_dir))

    assert (output_dir / "sample_48.png").exists()
    assert not (output_dir / "sample_96.png").exists()
