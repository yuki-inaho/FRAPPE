import json
from pathlib import Path

from PIL import Image

from tools.resize_rgb_dataset import resize_dataset


def _image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (12, 34, 56)).save(path, format="PNG")


def test_resize_preserves_anonymous_splits_and_strips_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for dataset_index in range(1, 5):
        _image(source / "canonical" / f"dataset_{dataset_index:03d}" / "image_00000001.png", (800, 600))
    output = tmp_path / "output"
    report = resize_dataset(
        source, output, 640, 480,
        {"train": [1, 2], "validation": [3], "test": [4]},
    )
    target = output / "imagefolder" / "train" / "image_00000001.png"
    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.size == (640, 480)
        assert not image.info
    assert report["image_size"] == {"width": 640, "height": 480}
    assert report["splits"] == {"train": 2, "validation": 1, "test": 1}
    assert str(source) not in json.dumps(report)
