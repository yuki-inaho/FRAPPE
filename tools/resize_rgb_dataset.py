#!/usr/bin/env python3
"""Derive an anonymous fixed-size RGB ImageFolder without source metadata.

This reads only the already anonymized ``canonical/dataset_NNN`` images. It
never receives raw archive paths or source member names, and writes fresh PNG
payloads with the same anonymous sequential filenames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from PIL import Image


IMAGE_NAME = re.compile(r"image_[0-9]{8}\.png$")


def _valid_rgb_png(path: Path, width: int | None = None, height: int | None = None) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return (
                image.format == "PNG" and image.mode == "RGB" and not image.info
                and (width is None or image.width == width)
                and (height is None or image.height == height)
            )
    except (OSError, ValueError):
        return False


def _source_images(source_root: Path, dataset_index: int) -> list[Path]:
    directory = source_root / "canonical" / f"dataset_{dataset_index:03d}"
    images = sorted(path for path in directory.glob("*.png") if IMAGE_NAME.fullmatch(path.name))
    if not images:
        raise ValueError(f"dataset_{dataset_index:03d} has no anonymous RGB PNG files")
    expected = [f"image_{index:08d}.png" for index in range(1, len(images) + 1)]
    if [path.name for path in images] != expected:
        raise ValueError(f"dataset_{dataset_index:03d} does not use sequential anonymous image names")
    return images


def _write_resized(source: Path, destination: Path, width: int, height: int) -> None:
    with Image.open(source) as image:
        image.load()
        resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        # Reconstruct from pixels so EXIF/ICC/source info cannot carry over.
        clean = Image.frombytes("RGB", resized.size, resized.tobytes())
    temporary = destination.with_name(f".{destination.name}.tmp")
    clean.save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, destination)


def _link_view(source_files: list[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for index, source in enumerate(source_files, start=1):
        name = f"image_{index:08d}.png"
        expected.add(name)
        target = destination / name
        if target.exists():
            if os.path.samefile(source, target):
                continue
            target.unlink()
        os.link(source, target)
    for candidate in destination.glob("image_*.png"):
        if IMAGE_NAME.fullmatch(candidate.name) and candidate.name not in expected:
            candidate.unlink()
    return len(expected)


def resize_dataset(source_root: Path, output_root: Path, width: int, height: int,
                   split_map: dict[str, list[int]]) -> dict[str, object]:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("output must be a new directory; source anonymous data is preserved")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    expected_indices = [index for indices in split_map.values() for index in indices]
    if sorted(expected_indices) != [1, 2, 3, 4]:
        raise ValueError("dataset indices 1 through 4 must each be assigned to exactly one split")

    canonical: dict[int, list[Path]] = {}
    summaries: list[dict[str, object]] = []
    for dataset_index in range(1, 5):
        inputs = _source_images(source_root, dataset_index)
        destination = output_root / "canonical" / f"dataset_{dataset_index:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(inputs, start=1):
            target = destination / f"image_{index:08d}.png"
            if not _valid_rgb_png(target, width, height):
                _write_resized(source, target, width, height)
            if index % 250 == 0 or index == len(inputs):
                print(f"dataset_{dataset_index:03d}: {index}/{len(inputs)} RGB images", flush=True)
        canonical[dataset_index] = sorted(destination.glob("image_????????.png"))
        summaries.append({
            "dataset_id": f"dataset_{dataset_index:03d}",
            "image_count": len(inputs),
            "dimensions": {f"{width}x{height}": len(inputs)},
        })

    split_counts = {
        split: _link_view(
            [image for dataset_index in indices for image in canonical[dataset_index]],
            output_root / "imagefolder" / split,
        )
        for split, indices in split_map.items()
    }
    for split, expected_count in split_counts.items():
        images = sorted((output_root / "imagefolder" / split).glob("image_????????.png"))
        if len(images) != expected_count or not all(_valid_rgb_png(path, width, height) for path in images):
            raise RuntimeError(f"invalid {split} output")

    report = {
        "schema_version": 1,
        "format": "RGB PNG without source metadata",
        "derived_from": "anonymous RGB PNG input",
        "image_size": {"width": width, "height": height},
        "naming": "anonymous sequential image names",
        "datasets": summaries,
        "splits": split_counts,
        "split_groups": {
            split: [f"dataset_{index:03d}" for index in indices]
            for split, indices in split_map.items()
        },
    }
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "preparation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/home/kasm-user/Desktop/data/frappe_rgb"))
    parser.add_argument("--output", type=Path,
                        default=Path("/home/kasm-user/Desktop/data/frappe_rgb_640x480"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--train-datasets", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--validation-datasets", type=int, nargs="+", default=[3])
    parser.add_argument("--test-datasets", type=int, nargs="+", default=[4])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = resize_dataset(
        args.source, args.output, args.width, args.height,
        {
            "train": args.train_datasets,
            "validation": args.validation_datasets,
            "test": args.test_datasets,
        },
    )
    print(f"prepared {sum(report['splits'].values())} resized anonymous RGB images", flush=True)


if __name__ == "__main__":
    main()
