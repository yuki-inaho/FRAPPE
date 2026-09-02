from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tools.prepare_rgb_dataset import (
    build_imagefolder,
    open_archive,
    ordered_members,
    prepare_archive,
    validate_output,
)


def png_bytes(color: tuple[int, int, int], metadata: bool = False) -> bytes:
    image = Image.new("RGB", (12, 10), color)
    output = io.BytesIO()
    pnginfo = PngImagePlugin.PngInfo()
    if metadata:
        pnginfo.add_text("private", "must not survive")
    image.save(output, format="PNG", pnginfo=pnginfo)
    return output.getvalue()


def make_zip(path: Path, names: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in names:
            archive.writestr(name, content)


def test_numeric_order_rgb_filter_and_metadata_removal(tmp_path: Path) -> None:
    archive = tmp_path / "private-input.zip"
    make_zip(archive, [
        ("private/Depth/00000002.png", png_bytes((9, 9, 9))),
        ("private/Color/00000010.png", png_bytes((10, 0, 0))),
        ("private/Color/00000002.png", png_bytes((2, 0, 0), metadata=True)),
        ("private/IR/00000002.png", png_bytes((8, 8, 8))),
    ])

    summary = prepare_archive(archive, tmp_path / "out", 1)

    assert summary["image_count"] == 2
    first = tmp_path / "out/canonical/dataset_001/image_00000001.png"
    second = tmp_path / "out/canonical/dataset_001/image_00000002.png"
    with Image.open(first) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (2, 0, 0)
        assert image.info == {}
    with Image.open(second) as image:
        assert image.getpixel((0, 0)) == (10, 0, 0)
    manifest = json.loads((first.parent / "manifest.json").read_text())
    assert "private-input" not in json.dumps(manifest)


def test_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    make_zip(archive, [("../Color/00000001.png", png_bytes((0, 0, 0)))])
    with open_archive(archive) as reader, pytest.raises(ValueError, match="unsafe member"):
        ordered_members(reader)


def test_rejects_tar_links(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as output:
        link = tarfile.TarInfo("private/Color/link.png")
        link.type = tarfile.SYMTYPE
        link.linkname = "elsewhere"
        output.addfile(link)
    with open_archive(archive) as reader, pytest.raises(ValueError, match="contains a link"):
        ordered_members(reader)


def test_anonymous_imagefolder_uses_hardlinks(tmp_path: Path) -> None:
    for dataset_index in (1, 2, 3, 4):
        directory = tmp_path / f"canonical/dataset_{dataset_index:03d}"
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8), (dataset_index, 0, 0)).save(
            directory / "image_00000001.png")

    counts = build_imagefolder(
        tmp_path, {"train": [1, 2], "validation": [3], "test": [4]})
    validate_output(tmp_path, counts)

    assert counts == {"train": 2, "validation": 1, "test": 1}
    assert (tmp_path / "imagefolder/train/image_00000001.png").stat().st_ino == (
        tmp_path / "canonical/dataset_001/image_00000001.png").stat().st_ino
