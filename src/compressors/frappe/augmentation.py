"""Albumentations training augmentation for RGB FRAPPE datasets.

The pipeline deliberately leaves normalization to the training script, where
images are converted from uint8 RGB to FRAPPE's ``[-1, 1]`` representation.
This keeps validation and codec evaluation free from data augmentation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

# Version polling is unrelated to a training run and otherwise adds an
# avoidable network timeout on offline compute nodes.  Respect an explicit user
# value while disabling it by default for this local pipeline.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from PIL import Image

# This explicit allow-list makes misspelled configuration names fail before a
# long GPU run starts, rather than silently dropping an augmentation.
SUPPORTED_TRANSFORMS = {
    "ColorJitter",
    "HueSaturationValue",
    "RGBShift",
    "ChannelShuffle",
    "HorizontalFlip",
    "VerticalFlip",
    "RandomGamma",
    "ToGray",
    "ToSepia",
}


def _transform_entries(spec: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield enabled transform specifications from a list or named mapping."""
    entries = spec.get("transforms", [])
    if isinstance(entries, Mapping):
        entries = entries.values()
    if not isinstance(entries, Iterable) or isinstance(entries, (str, bytes)):
        raise ValueError("augmentation transforms must be a list or mapping")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each augmentation transform must be a mapping")
        if entry.get("enabled", True):
            yield entry


def _build_transforms(spec: Mapping[str, Any]) -> list[A.BasicTransform]:
    transforms: list[A.BasicTransform] = []
    for entry in _transform_entries(spec):
        name = entry.get("name")
        if name not in SUPPORTED_TRANSFORMS:
            choices = ", ".join(sorted(SUPPORTED_TRANSFORMS))
            raise ValueError(f"unsupported Albumentations transform {name!r}; choose one of: {choices}")
        kwargs = {key: value for key, value in entry.items() if key not in {"name", "enabled"}}
        try:
            transforms.append(getattr(A, name)(**kwargs))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid configuration for Albumentations {name}: {error}") from error
    return transforms


class RGBTrainingAugmentation:
    """Random resize/crop plus a configurable Albumentations RGB pipeline."""

    def __init__(self, spec: Mapping[str, Any] | None = None) -> None:
        self.spec = dict(spec or {})
        self.transforms = _build_transforms(self.spec)

    @classmethod
    def from_json(cls, value: str) -> RGBTrainingAugmentation:
        try:
            spec = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("--augmentation_config must be valid JSON") from error
        if not isinstance(spec, Mapping):
            raise ValueError("--augmentation_config must contain a JSON object")
        return cls(spec)

    def describe(self) -> str:
        names = [transform.__class__.__name__ for transform in self.transforms]
        return ", ".join(names) if names else "none"

    @staticmethod
    def _resize_shape(source_height: int, source_width: int, target_height: int,
                      target_width: int, scale: float, max_size: int) -> tuple[int, int]:
        """Resize enough to cover the requested crop, then apply extra scale.

        At scale=1 a 640×480 image with a 640×480 target stays 640×480.
        This is important for the managed landscape input rather than using a
        square-crop shortest-edge rule that would always enlarge it.
        """
        resize_scale = max(target_height / source_height, target_width / source_width) * scale
        if round(max(source_height, source_width) * resize_scale) > max_size:
            resize_scale = max_size / max(source_height, source_width)
        return max(1, round(source_height * resize_scale)), max(1, round(source_width * resize_scale))

    def __call__(self, image: Image.Image, *, height: int, width: int,
                 scale: float, max_size: int) -> np.ndarray:
        """Return an HWC uint8 RGB crop, suitable for conversion to a tensor."""
        source = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        resized_h, resized_w = self._resize_shape(
            source.shape[0], source.shape[1], height, width, scale, max_size)
        geometric = [
            A.Resize(resized_h, resized_w, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.RandomCrop(height=height, width=width, p=1.0),
        ]
        augmented = A.Compose([*geometric, *self.transforms])(image=source)["image"]
        if augmented.shape != (height, width, 3) or augmented.dtype != np.uint8:
            raise RuntimeError(
                "Albumentations pipeline must return an uint8 HWC RGB image; "
                f"received shape={augmented.shape}, dtype={augmented.dtype}")
        return augmented
