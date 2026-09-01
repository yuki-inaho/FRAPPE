from pathlib import Path

import datasets
from PIL import Image


def test_local_anonymous_imagefolder_has_expected_splits(tmp_path: Path) -> None:
    for split in ("train", "validation", "test"):
        directory = tmp_path / split
        directory.mkdir()
        Image.new("RGB", (32, 24), (1, 2, 3)).save(directory / "image_00000001.png")

    train = datasets.load_dataset(str(tmp_path), split="train")
    validation = datasets.load_dataset(str(tmp_path), split="validation")

    assert train.num_rows == 1
    assert validation.num_rows == 1
    assert train[0]["image"].mode == "RGB"
