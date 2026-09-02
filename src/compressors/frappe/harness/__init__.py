"""Shared pieces behind the command-line tools in ``tools/``.

``compressors.frappe`` proper is the codec: the transforms, the quantizer, the
entropy coder. This subpackage is everything the *experiments* need and the
codec does not -- loading an anonymous split, counting a bitstream, averaging a
metric, matching a reference codec's rate, printing a table.

It exists because those things were previously copied into every tool. Three
tools carried their own JPEG-LS byte count, seven their own image loader, and
the two copies of the byte count disagreed about whether the 4-byte length
prefix is part of the rate -- a disagreement that is invisible until someone
compares numbers from two tools. One implementation with the convention as an
explicit argument makes that a choice rather than an accident.
"""

from .bitstream import (
    BitstreamConvention,
    arrange_planes,
    encode_planes,
    measure_rate,
)
from .data import AnonymousImageFolder, default_dataset_root
from .metrics import RateDistortionAccumulator, RatePoint, psnr_from_mse

__all__ = [
    "AnonymousImageFolder",
    "BitstreamConvention",
    "RateDistortionAccumulator",
    "RatePoint",
    "arrange_planes",
    "default_dataset_root",
    "encode_planes",
    "measure_rate",
    "psnr_from_mse",
]
