#!/usr/bin/env python3
"""Create an anonymous, metadata-free RGB ImageFolder dataset from archives.

The input archive and member names are intentionally never written to the
output tree. Images are decoded directly from ZIP/TAR members rather than
extracting the raw archive hierarchy.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import stat
import tarfile
import zipfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
GENERATED_IMAGE = re.compile(r"^image_[0-9]{8}\.png$")
NUMBER = re.compile(r"([0-9]+)")


@dataclass(frozen=True)
class Member:
    name: str
    group: tuple[str, ...]
    order: tuple[object, ...]


def _safe_parts(name: str) -> tuple[str, ...]:
    """Return normalized archive parts and reject unsafe member paths."""
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("archive contains an unsafe member path")
    return path.parts


def _candidate(parts: tuple[str, ...]) -> bool:
    suffix = Path(parts[-1]).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return False
    parents = {part.casefold() for part in parts[:-1]}
    return "color" in parents or parts[-1].casefold().startswith("color_")


def _group(parts: tuple[str, ...]) -> tuple[str, ...]:
    lower = [part.casefold() for part in parts]
    if "color" in lower[:-1]:
        return parts[: lower.index("color")]
    return parts[:-1]


def _natural_key(filename: str) -> tuple[object, ...]:
    chunks = NUMBER.split(filename.casefold())
    return tuple(int(chunk) if chunk.isdigit() else chunk for chunk in chunks)


class ArchiveReader(AbstractContextManager["ArchiveReader"]):
    def members(self) -> list[Member]:
        raise NotImplementedError

    def open(self, member: Member) -> BinaryIO:
        raise NotImplementedError


class ZipReader(ArchiveReader):
    def __init__(self, path: Path):
        self.archive = zipfile.ZipFile(path)

    def __exit__(self, *args: object) -> None:
        self.archive.close()

    def members(self) -> list[Member]:
        selected: list[Member] = []
        for info in self.archive.infolist():
            parts = _safe_parts(info.filename)
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise ValueError("ZIP archive contains a symbolic link")
            if info.is_dir() or not _candidate(parts):
                continue
            selected.append(Member(info.filename, _group(parts), _natural_key(parts[-1])))
        return selected

    def open(self, member: Member) -> BinaryIO:
        return self.archive.open(member.name)


class TarReader(ArchiveReader):
    def __init__(self, path: Path):
        self.archive = tarfile.open(path, mode="r:*")

    def __exit__(self, *args: object) -> None:
        self.archive.close()

    def members(self) -> list[Member]:
        selected: list[Member] = []
        for info in self.archive.getmembers():
            parts = _safe_parts(info.name)
            if info.issym() or info.islnk():
                raise ValueError("TAR archive contains a link")
            if not info.isfile() or not _candidate(parts):
                continue
            selected.append(Member(info.name, _group(parts), _natural_key(parts[-1])))
        return selected

    def open(self, member: Member) -> BinaryIO:
        stream = self.archive.extractfile(member.name)
        if stream is None:
            raise ValueError("unable to read an image member")
        return stream


def open_archive(path: Path) -> ArchiveReader:
    if zipfile.is_zipfile(path):
        return ZipReader(path)
    if tarfile.is_tarfile(path):
        return TarReader(path)
    raise ValueError("unsupported archive format")


def ordered_members(reader: ArchiveReader) -> tuple[list[Member], list[int]]:
    """Sort by anonymous sequence and numeric frame order."""
    members = reader.members()
    groups: dict[tuple[str, ...], list[Member]] = {}
    for member in members:
        groups.setdefault(member.group, []).append(member)
    ordered: list[Member] = []
    lengths: list[int] = []
    for key in sorted(groups):
        sequence = sorted(groups[key], key=lambda item: item.order)
        if len({item.order for item in sequence}) != len(sequence):
            raise ValueError("duplicate RGB frame index in an input sequence")
        ordered.extend(sequence)
        lengths.append(len(sequence))
    return ordered, lengths


def _is_valid_output(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == "PNG" and image.mode == "RGB" and not image.info
    except (OSError, ValueError):
        return False


def _write_clean_png(stream: BinaryIO, destination: Path) -> tuple[int, int]:
    """Decode to a fresh RGB object so no source metadata is retained."""
    with Image.open(stream) as source:
        source.load()
        converted = source.convert("RGB")
        clean = Image.frombytes("RGB", converted.size, converted.tobytes())
    temporary = destination.with_name(f".{destination.name}.tmp")
    clean.save(temporary, format="PNG", compress_level=1)
    os.replace(temporary, destination)
    return clean.size


def prepare_archive(path: Path, output_root: Path, dataset_index: int,
                    dry_run: bool = False) -> dict[str, object]:
    dataset_id = f"dataset_{dataset_index:03d}"
    destination = output_root / "canonical" / dataset_id
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    with open_archive(path) as reader:
        members, sequence_lengths = ordered_members(reader)
        if not members:
            raise ValueError(f"{dataset_id} contains no RGB candidates")
        dimensions: dict[str, int] = {}
        for index, member in enumerate(members, start=1):
            target = destination / f"image_{index:08d}.png"
            if not dry_run and not _is_valid_output(target):
                try:
                    with reader.open(member) as stream:
                        size = _write_clean_png(stream, target)
                except Exception as exc:
                    raise RuntimeError(f"failed to decode {dataset_id} image {index}") from exc
            elif dry_run:
                with reader.open(member) as stream:
                    with Image.open(io.BytesIO(stream.read())) as image:
                        size = image.size
            else:
                with Image.open(target) as image:
                    size = image.size
            dimensions[f"{size[0]}x{size[1]}"] = dimensions.get(f"{size[0]}x{size[1]}", 0) + 1
            if index % 250 == 0 or index == len(members):
                print(f"{dataset_id}: {index}/{len(members)} RGB images", flush=True)

    summary: dict[str, object] = {
        "dataset_id": dataset_id,
        "image_count": len(members),
        "sequence_count": len(sequence_lengths),
        "sequence_lengths": sequence_lengths,
        "dimensions": dimensions,
    }
    if not dry_run:
        manifest = destination / "manifest.json"
        manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _link_generated_view(source_files: Iterable[Path], destination: Path) -> int:
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
    for path in destination.iterdir():
        if path.is_file() and GENERATED_IMAGE.fullmatch(path.name) and path.name not in expected:
            path.unlink()
    return len(expected)


def build_imagefolder(output_root: Path, split_map: dict[str, list[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, dataset_indices in split_map.items():
        sources: list[Path] = []
        for dataset_index in dataset_indices:
            canonical = output_root / "canonical" / f"dataset_{dataset_index:03d}"
            sources.extend(sorted(canonical.glob("image_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png")))
        counts[split] = _link_generated_view(sources, output_root / "imagefolder" / split)
    return counts


def validate_output(output_root: Path, expected_counts: dict[str, int]) -> None:
    for split, expected in expected_counts.items():
        directory = output_root / "imagefolder" / split
        images = sorted(directory.glob("*.png"))
        if len(images) != expected:
            raise RuntimeError(f"{split} image count mismatch")
        for index, path in enumerate(images, start=1):
            if path.name != f"image_{index:08d}.png" or not _is_valid_output(path):
                raise RuntimeError(f"invalid anonymous output in {split} at image {index}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs=4)
    parser.add_argument("--output", type=Path,
                        default=Path("/home/kasm-user/Desktop/data/frappe_rgb"))
    parser.add_argument("--train-datasets", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--validation-datasets", type=int, nargs="+", default=[3])
    parser.add_argument("--test-datasets", type=int, nargs="+", default=[4])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for archive in args.archives:
        if not archive.is_file():
            raise FileNotFoundError("an input archive is missing")
    summaries = [
        prepare_archive(path, args.output, index, args.dry_run)
        for index, path in enumerate(args.archives, start=1)
    ]
    if args.dry_run:
        print(json.dumps({"datasets": summaries}, indent=2))
        return

    split_map = {
        "train": args.train_datasets,
        "validation": args.validation_datasets,
        "test": args.test_datasets,
    }
    assigned = [index for indices in split_map.values() for index in indices]
    if sorted(assigned) != [1, 2, 3, 4]:
        raise ValueError("dataset indices 1 through 4 must each be assigned to exactly one split")
    split_counts = build_imagefolder(args.output, split_map)
    validate_output(args.output, split_counts)
    report = {
        "schema_version": 1,
        "format": "RGB PNG without source metadata",
        "naming": "anonymous sequential image names",
        "datasets": summaries,
        "splits": split_counts,
        "split_groups": {
            split: [f"dataset_{index:03d}" for index in indices]
            for split, indices in split_map.items()
        },
    }
    report_dir = args.output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preparation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {sum(split_counts.values())} split images", flush=True)


if __name__ == "__main__":
    main()
