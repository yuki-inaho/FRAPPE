"""Loading anonymous local image splits.

The datasets in this project are metadata-free PNG files with sequential
anonymous names, laid out as ``<root>/<split>/image_00000001.png``. Seven tools
had grown their own copy of ``sorted(glob("image_????????.png"))`` followed by
``Image.open``, which is not merely repetition: each copy also embedded its own
default root, so relocating the data meant editing seven files, and each copy
decided independently whether to convert to RGB, to normalise, and to which
device.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .annotations import Float, Tensor, UInt8, checked

#: The anonymous naming ``tools/prepare_rgb_dataset.py`` writes.
IMAGE_GLOB = "image_????????.png"

#: Where the prepared data lives, overridable without editing code. The
#: environment variable is the same one the Hydra data configs already read.
DEFAULT_DATASET_ROOT = "/workspace/data/frappe_rgb_800x608/imagefolder"


def default_dataset_root() -> Path:
    return Path(os.environ.get("FRAPPE_DATASET_ROOT", DEFAULT_DATASET_ROOT))


@checked
def to_signed(image: UInt8[Tensor, "*batch channels height width"]
              ) -> Float[Tensor, "*batch channels height width"]:
    """``uint8`` pixels to the ``[-1, 1]`` range the codec operates in."""
    return image.to(torch.float32) / 127.5 - 1.0


@checked
def to_pixels(image: Float[Tensor, "*batch channels height width"]
              ) -> UInt8[Tensor, "*batch channels height width"]:
    """The inverse of :func:`to_signed`, with the clamp a display needs."""
    return torch.clamp(torch.round((image.clamp(-1.0, 1.0) / 2 + 0.5) * 255.0),
                       0.0, 255.0).to(torch.uint8)


@dataclass
class AnonymousImageFolder:
    """One split of an anonymous ImageFolder, addressed by index.

    Reading is deliberately lazy and per-image: the splits hold thousands of
    800x608 PNGs, and a tool that wants sixteen of them should not pay for four
    thousand.
    """

    root: Path
    split: str = "validation"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._files = sorted((self.root / self.split).glob(IMAGE_GLOB))
        if not self._files:
            raise FileNotFoundError(
                f"no anonymous PNG images under {self.root / self.split}")

    def __len__(self) -> int:
        return len(self._files)

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    def path(self, index: int) -> Path:
        return self._files[index]

    def pil(self, index: int) -> Image.Image:
        with Image.open(self._files[index]) as handle:
            handle.load()
            return handle.convert("RGB")

    def pixels(self, index: int) -> UInt8[Tensor, "1 3 height width"]:
        """One image as ``uint8`` NCHW, the form the ONNX encoder takes."""
        array = np.array(self.pil(index), dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    def signed(self, index: int, device: str | torch.device = "cpu"
               ) -> Float[Tensor, "1 3 height width"]:
        """One image in ``[-1, 1]`` on ``device``, the form the model takes."""
        return to_signed(self.pixels(index)).to(device)

    def batch(self, count: int, device: str | torch.device = "cpu"
              ) -> Float[Tensor, "batch 3 height width"]:
        """The first ``count`` images stacked, for whole-split statistics.

        Every image in a split must then be the same size; that holds for the
        prepared datasets, which are materialised at a fixed resolution, and a
        mismatch here is a data-preparation bug worth failing on.
        """
        frames = [self.signed(index, device) for index in range(min(count, len(self)))]
        return torch.cat(frames, dim=0)
