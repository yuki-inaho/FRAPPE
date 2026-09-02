"""Measuring a joint-prefix codec on a local split.

This is the deployment path, not a proxy for it: the encoder produces true int8
codes, those codes are arranged and entropy coded exactly as the bitstream
defines, and the reconstruction is decoded from them. Nothing here estimates a
bitrate.

Distortion and rate are accumulated over the same images by construction, via
:class:`RateDistortionAccumulator`. When entropy coding only a prefix of the
split -- it is the slow part -- the distortion is restricted to that same prefix
rather than averaged over all of them, because a PSNR from one set paired with a
bitrate from another is a point on no curve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.nn import functional

from ..prefix import JointPrefixFRAPPE
from .bitstream import BitstreamConvention, measure_rate, prefix_channels
from .data import AnonymousImageFolder
from .metrics import Averaging, RateDistortionAccumulator, RatePoint

#: An operating point is a prefix length or an explicit set of kept channels.
OperatingPoint = int | Sequence[int]


def kept_channels(model: JointPrefixFRAPPE, point: OperatingPoint) -> list[int]:
    """How many channels of each scale group this operating point transmits."""
    if isinstance(point, int):
        return prefix_channels(model.scale_groups, point)
    wanted = {int(channel) for channel in point}
    return [sum(1 for channel in range(start, end) if channel + 1 in wanted)
            for _, start, end in model.scale_groups]


@torch.no_grad()
def evaluate_operating_points(
    model: JointPrefixFRAPPE,
    folder: AnonymousImageFolder,
    points: Sequence[OperatingPoint],
    images: int | None = None,
    device: str | torch.device = "cpu",
    averaging: Averaging = Averaging.AGGREGATE_MSE,
    convention: BitstreamConvention = BitstreamConvention.PAYLOAD_ONLY,
    rate_images: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[RatePoint]:
    """Rate and distortion for each operating point, from real bitstreams."""
    count = min(images or len(folder), len(folder))
    rate_count = min(rate_images or count, count)
    accumulators = {index: RateDistortionAccumulator(averaging)
                    for index in range(len(points))}
    channel_plans = [kept_channels(model, point) for point in points]

    for image_index in range(rate_count):
        x = folder.signed(image_index, device)
        pixels = x.shape[2] * x.shape[3]
        codes = model.integer_codes(x)
        adapted = model.adapt([code.to(torch.float) for code in codes])
        for index, (point, plan) in enumerate(zip(points, channel_plans)):
            reconstruction = (model.decode(adapted, point) if isinstance(point, int)
                              else model.decode_subset(adapted, point)).clamp(-1, 1)
            mse = functional.mse_loss(x / 2 + 0.5, reconstruction / 2 + 0.5).item()
            byte_count, _ = measure_rate([codes[g] for g in range(len(codes))],
                                         pixels, plan, convention)
            accumulators[index].add(mse, byte_count, pixels)
        if progress is not None:
            progress(image_index + 1, rate_count)

    return [accumulators[index].point(
        label=point if isinstance(point, int) else tuple(point))
        for index, point in enumerate(points)]


def monotonicity_violations(points: Sequence[RatePoint]) -> int:
    """How often a longer prefix reconstructs worse than a shorter one.

    The prefix property is the codec's selling point, so a ladder that is not
    monotone is a defect rather than a curiosity, and it is cheap to count.
    """
    values = [point.psnr_db for point in points]
    return sum(1 for earlier, later in zip(values[:-1], values[1:]) if later < earlier)
