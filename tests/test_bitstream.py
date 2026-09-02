"""The JPEG-LS container, and the two rate conventions that live in it.

``src/compressors/frappe/bitstream.py`` restates in unsigned-plane terms what
``entropy_coding.py`` does in signed-latent terms, so a deployment can code the
exported encoder's output without importing torch. A restatement is only
acceptable while it agrees exactly, so the agreement is asserted here rather than
assumed -- the same standard ``test_prefix_model.py`` applies to the
``pixel_unshuffle`` rewrite of ``adapt_to_decoder``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.compressors.frappe.bitstream import (
    CODE_OFFSET,
    LENGTH_PREFIX_BYTES,
    decode_planes,
    encode_plane,
    encode_planes,
    measure,
)


def require_jpegls():
    """JPEG-LS comes from pillow-jpls; without it there is nothing to test."""
    pytest.importorskip("pillow_jpls", reason="pillow-jpls provides the JPEG-LS codec")


def sample_planes(seed: int = 0) -> list[np.ndarray]:
    """Planes shaped like a real five-group schedule, smoothed so JPEG-LS has
    something to predict -- uniform noise would exercise only the escape path."""
    rng = np.random.default_rng(seed)
    shapes = [(19, 25), (190, 50), (228, 100), (912, 200), (304, 400)]
    planes = []
    for rows, cols in shapes:
        ramp = np.linspace(0, 255, cols, dtype=np.float64)[None, :]
        noise = rng.normal(0, 6, size=(rows, cols))
        planes.append(np.clip(ramp + noise, 0, 255).astype(np.uint8))
    return planes


def test_planes_survive_a_round_trip_exactly():
    """NEAR=0 JPEG-LS is lossless; anything else would corrupt the codes."""
    require_jpegls()
    planes = sample_planes()
    restored = decode_planes(encode_planes(planes))
    assert len(restored) == len(planes)
    for original, roundtripped in zip(planes, restored):
        assert np.array_equal(original, roundtripped)


def test_the_length_prefix_costs_exactly_four_bytes_per_scale_group():
    require_jpegls()
    planes = sample_planes()
    bare = encode_planes(planes, length_prefix=False)
    prefixed = encode_planes(planes, length_prefix=True)
    assert len(prefixed) - len(bare) == LENGTH_PREFIX_BYTES * len(planes)


def test_a_bare_blob_cannot_be_decoded():
    """Without the prefixes the blob is not self-describing, and the reader must
    say so rather than return plausible garbage."""
    require_jpegls()
    with pytest.raises(ValueError, match="truncated"):
        decode_planes(encode_planes(sample_planes(), length_prefix=False)[:-1])


def test_measure_reports_both_conventions_and_they_differ_by_the_prefixes():
    """The gap the worklog quantifies: 32*G/(T1*T2) bits per pixel."""
    require_jpegls()
    planes = sample_planes()
    report = measure(planes, height=608, width=800)
    expected_gap = 8 * LENGTH_PREFIX_BYTES * len(planes) / (608 * 800)
    assert report["bpp_with_length_prefix"] - report["bpp_payload_only"] == \
        pytest.approx(expected_gap)
    assert expected_gap == pytest.approx(3.2895e-4, rel=1e-3)


def test_measure_ties_compression_ratio_to_the_same_convention():
    require_jpegls()
    report = measure(sample_planes(), height=608, width=800)
    assert report["compression_ratio_payload_only"] == \
        pytest.approx(24.0 / report["bpp_payload_only"])
    assert report["compression_ratio_with_length_prefix"] == \
        pytest.approx(24.0 / report["bpp_with_length_prefix"])


@pytest.mark.parametrize("bad", [
    np.zeros((8, 8), dtype=np.int8),
    np.zeros((8, 8), dtype=np.float32),
    np.zeros((8, 8), dtype=np.int16),
])
def test_signed_planes_are_refused(bad):
    """The failure this module exists to prevent: shifting the codes twice
    produces a valid stream that decodes to a plausible wrong image."""
    with pytest.raises(ValueError, match="uint8"):
        encode_plane(bad)


def test_a_batch_axis_of_one_is_accepted():
    """The exported graph carries the batch axis the exporter fixed at one."""
    require_jpegls()
    plane = sample_planes()[0]
    assert encode_plane(plane[None]) == encode_plane(plane)


def test_bytes_match_the_entropy_coding_path_they_restate():
    """The whole justification for a second implementation."""
    require_jpegls()
    torch = pytest.importorskip("torch", reason="entropy_coding is written on torch")

    from src.compressors.frappe.entropy_coding import encode_latents

    planes = sample_planes(seed=3)
    # entropy_coding works from signed arranged latents shaped (1, C, H, W); the
    # planes above are the (C*H, W) unsigned form of exactly that.
    signed = [torch.from_numpy(plane.astype(np.int16) - CODE_OFFSET).to(torch.int8)
              for plane in planes]
    assert encode_latents(signed) == encode_planes(planes, length_prefix=True)
