"""Distortion and rate accumulation, with the averaging convention made explicit.

PSNR is convex in MSE, so averaging per-image PSNR and converting one aggregate
MSE are different numbers -- up to about 0.8 dB apart on the shipped Kodak
curve. The repository uses both: ``evaluate.py`` and the notebook average
per-image PSNR, while the joint-prefix tools convert an aggregate. Neither is
wrong, but a table that mixes them is, so the choice is named here rather than
implied by whichever function a tool happened to call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Averaging(str, Enum):
    """How per-image distortion becomes one number."""

    #: Convert one aggregate MSE. The convention of the joint-prefix tools.
    AGGREGATE_MSE = "aggregate_mse"
    #: Average per-image PSNR. The convention of ``evaluate.py``, the notebook,
    #: and therefore of the results shipped in ``results/``.
    MEAN_PSNR = "mean_psnr"


def psnr_from_mse(mse: float) -> float:
    """PSNR in dB for a mean squared error on the ``[0, 1]`` convention."""
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


@dataclass
class RatePoint:
    """One operating point: what it cost and what it delivered."""

    label: str | int
    psnr_db: float
    bpp: float
    images: int

    @property
    def compression_ratio(self) -> float:
        """Against 24 bpp uncompressed 8-bit RGB, as the codec's own code does."""
        return 24.0 / self.bpp if self.bpp > 0 else float("inf")

    def as_dict(self) -> dict:
        return {"label": self.label, "psnr_db": self.psnr_db, "bpp": self.bpp,
                "compression_ratio": self.compression_ratio, "images": self.images}


@dataclass
class RateDistortionAccumulator:
    """Collect per-image error and bytes for one operating point.

    Distortion and rate are accumulated over the *same* images by construction.
    Pairing a PSNR averaged over one set with a bitrate measured over another
    produces a point that is on no rate-distortion curve, which is a mistake this
    type makes impossible rather than merely discouraged.
    """

    averaging: Averaging = Averaging.AGGREGATE_MSE
    mse_total: float = 0.0
    psnr_total: float = 0.0
    bytes_total: int = 0
    pixels_total: int = 0
    images: int = 0
    _per_image: list[tuple[float, int]] = field(default_factory=list)

    def add(self, mse: float, byte_count: int, pixels: int) -> None:
        self.mse_total += mse
        self.psnr_total += psnr_from_mse(mse)
        self.bytes_total += byte_count
        self.pixels_total += pixels
        self.images += 1
        self._per_image.append((mse, byte_count))

    @property
    def psnr_db(self) -> float:
        if not self.images:
            return float("nan")
        if self.averaging is Averaging.MEAN_PSNR:
            return self.psnr_total / self.images
        return psnr_from_mse(self.mse_total / self.images)

    @property
    def bpp(self) -> float:
        return self.bytes_total * 8 / self.pixels_total if self.pixels_total else float("nan")

    def point(self, label: str | int) -> RatePoint:
        return RatePoint(label=label, psnr_db=self.psnr_db, bpp=self.bpp, images=self.images)
