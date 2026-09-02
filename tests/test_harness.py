"""The harness must agree with the code it was extracted from.

These are characterisation tests: they pin the behaviour the tools had before
the shared layer existed, so a refactor that changes a number fails here rather
than silently changing a published result. Every assertion compares the harness
against the original implementation rather than against a hand-written
expectation, because the original is what produced the numbers on record.
"""

from __future__ import annotations

import io
import struct

import pytest
import torch

from src.compressors.frappe import entropy_coding
from src.compressors.frappe.harness import bitstream, metrics
from src.compressors.frappe.harness.data import to_pixels, to_signed


def reference_jpegls_bytes(latents, n_channels, groups):
    """The byte count as tools/evaluate_joint_prefix.py computed it before the refactor."""
    import pillow_jpls  # noqa: F401
    from torchvision.transforms.v2.functional import to_pil_image

    total = 0
    remaining = n_channels
    for code, (_, start, end) in zip(latents, groups):
        if remaining <= 0:
            break
        width = min(end - start, remaining)
        plane = code[0, :width]
        flat = plane.reshape(plane.shape[0] * plane.shape[1], plane.shape[2])
        buffer = io.BytesIO()
        to_pil_image((flat.to(torch.long) + 127).to(torch.uint8)).save(
            buffer, format="JPEG-LS")
        total += len(buffer.getbuffer())
        remaining -= width
    return total


@pytest.fixture
def latents():
    """Latents shaped like the released schedule's five scale groups."""
    torch.manual_seed(5)
    widths, grids = (3, 6, 3, 6, 3), (4, 8, 16, 32, 64)
    return [torch.randint(-127, 128, (1, width, grid, grid + 2), dtype=torch.int8)
            for width, grid in zip(widths, grids)]


GROUPS = [(32, 0, 3), (16, 3, 9), (8, 9, 12), (4, 12, 18), (2, 18, 21)]


def test_arrange_plane_matches_the_entropy_coder(latents):
    """The harness layout is the entropy coder's layout, plus its uint8 shift."""
    reference = entropy_coding.arrange_latents(latents)
    for latent, expected in zip(latents, reference):
        shifted = (expected.to(torch.int16) + bitstream.CODE_OFFSET).to(torch.uint8)
        assert torch.equal(bitstream.arrange_plane(latent[0]), shifted)


def test_arranged_rows_are_channel_major(latents):
    """(c, h, w) -> row = c*H + h, col = w. The order JPEG-LS sees is the contract."""
    latent = latents[1][0]
    channels, grid_h, grid_w = latent.shape
    plane = bitstream.arrange_plane(latent)
    assert plane.shape == (channels * grid_h, grid_w)
    for c in range(channels):
        for h in range(grid_h):
            expected = (latent[c, h].to(torch.int16) + bitstream.CODE_OFFSET).to(torch.uint8)
            assert torch.equal(plane[c * grid_h + h], expected)


@pytest.mark.parametrize("n_channels", [1, 3, 4, 9, 12, 18, 21])
def test_measure_rate_matches_the_pre_refactor_byte_count(latents, n_channels):
    """Payload-only rate reproduces what the tools counted before the harness."""
    kept = bitstream.prefix_channels(GROUPS, n_channels)
    measured, _ = bitstream.measure_rate(latents, pixels=1, channels=kept)
    assert measured == reference_jpegls_bytes(latents, n_channels, GROUPS)


def test_measure_rate_with_prefixes_matches_encode_latents(latents):
    """The container convention reproduces entropy_coding.encode_latents exactly."""
    blob = entropy_coding.encode_latents(entropy_coding.arrange_latents(latents))
    measured, _ = bitstream.measure_rate(
        latents, pixels=1, convention=bitstream.BitstreamConvention.WITH_LENGTH_PREFIX)
    assert measured == len(blob)


def test_the_two_conventions_differ_by_exactly_the_prefixes(latents):
    """The whole disagreement is four bytes per scale group, and nothing else."""
    bare, _ = bitstream.measure_rate(latents, pixels=1)
    container, _ = bitstream.measure_rate(
        latents, pixels=1, convention=bitstream.BitstreamConvention.WITH_LENGTH_PREFIX)
    assert container - bare == bitstream.LENGTH_PREFIX_BYTES * len(latents)


def test_encoded_planes_round_trip_through_the_entropy_coder(latents):
    """Bytes written here are bytes entropy_coding.decode_latents can read back."""
    planes = bitstream.arrange_planes(latents)
    blob = bitstream.encode_planes(
        planes, bitstream.BitstreamConvention.WITH_LENGTH_PREFIX)
    recovered = entropy_coding.decode_latents(blob)
    for original, restored in zip(entropy_coding.arrange_latents(latents), recovered):
        assert torch.equal(original, restored)


def test_prefix_channels_splits_a_prefix_across_contiguous_groups():
    assert bitstream.prefix_channels(GROUPS, 1) == [1, 0, 0, 0, 0]
    assert bitstream.prefix_channels(GROUPS, 12) == [3, 6, 3, 0, 0]
    assert bitstream.prefix_channels(GROUPS, 21) == [3, 6, 3, 6, 3]
    assert bitstream.prefix_channels(GROUPS, 99) == [3, 6, 3, 6, 3]


def test_bare_length_prefix_is_big_endian(latents):
    """The prefix format is part of the bitstream, not an implementation detail."""
    planes = bitstream.arrange_planes(latents[:1])
    blob = bitstream.encode_planes(
        planes, bitstream.BitstreamConvention.WITH_LENGTH_PREFIX)
    (declared,) = struct.unpack_from(">I", blob, 0)
    assert declared == len(blob) - bitstream.LENGTH_PREFIX_BYTES


def test_pixel_conversions_round_trip():
    pixels = torch.randint(0, 256, (1, 3, 5, 7), dtype=torch.uint8)
    assert torch.equal(to_pixels(to_signed(pixels)), pixels)


def test_signed_conversion_spans_the_full_range():
    extremes = torch.tensor([[[[0, 255]]]], dtype=torch.uint8).expand(1, 3, 1, 2)
    signed = to_signed(extremes.contiguous())
    assert float(signed.min()) == pytest.approx(-1.0)
    assert float(signed.max()) == pytest.approx(1.0)


def test_the_two_averaging_conventions_are_both_available_and_differ():
    """PSNR is convex in MSE, so the choice of average is a real choice."""
    aggregate = metrics.RateDistortionAccumulator(metrics.Averaging.AGGREGATE_MSE)
    per_image = metrics.RateDistortionAccumulator(metrics.Averaging.MEAN_PSNR)
    for mse, byte_count in ((1e-2, 100), (1e-4, 400)):
        aggregate.add(mse, byte_count, pixels=1000)
        per_image.add(mse, byte_count, pixels=1000)
    assert aggregate.psnr_db == pytest.approx(metrics.psnr_from_mse((1e-2 + 1e-4) / 2))
    assert per_image.psnr_db == pytest.approx((20.0 + 40.0) / 2)
    assert per_image.psnr_db > aggregate.psnr_db
    assert aggregate.bpp == pytest.approx(500 * 8 / 2000)


def test_a_rate_point_reports_the_codec_s_own_compression_ratio():
    accumulator = metrics.RateDistortionAccumulator()
    accumulator.add(1e-3, byte_count=1000, pixels=8000)
    point = accumulator.point(label=16)
    assert point.bpp == pytest.approx(1.0)
    assert point.compression_ratio == pytest.approx(24.0)
    assert point.as_dict()["label"] == 16


def test_annotations_reject_the_wrong_dtype_and_rank():
    """The shape annotations are a predicate, not a comment."""
    from jaxtyping import TypeCheckError

    with pytest.raises(TypeCheckError):
        bitstream.arrange_plane(torch.zeros(3, 4, 5))  # float, not int8
    with pytest.raises(TypeCheckError):
        bitstream.arrange_plane(torch.zeros(4, 5, dtype=torch.int8))  # rank 2
