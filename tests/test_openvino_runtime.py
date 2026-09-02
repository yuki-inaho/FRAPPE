"""Contract tests for running the exported FRAPPE graphs through OpenVINO.

Two properties are worth a test here, and they are the two that a deployment
gets wrong quietly.

The first is that a requested device is the device that runs. OpenVINO ships
meta-devices (``AUTO``, ``HETERO``, ``MULTI``, ``BATCH``) whose entire purpose is
to pick a device for you and to move work when the first choice does not fit, so
asking for ``AUTO`` and reading back "it ran" tells you nothing about where. This
module refuses them and makes the caller state a preference order it can see the
resolution of.

The second is the plane geometry. ``tools/export_onnx.py`` gives the exporter the
paper's own relation -- ``rows = n_s * (max_ps / p_s) * units_h``, ``cols =
(max_ps / p_s) * units_w`` -- so the graph stays resolution independent. The
runtime has to reproduce that relation to pin a static shape for the NPU, and a
disagreement between the two would surface as a shape error at compile time on
one resolution and as silent misalignment on another.

Tests that need a real exported graph are skipped when none is configured;
``FRAPPE_ONNX_STEM`` points at one. Tests that need a particular accelerator are
skipped when OpenVINO does not enumerate it, so the suite is honest on a machine
without an NPU rather than green by omission.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.compressors.frappe.openvino_runtime import (
    META_DEVICES,
    DeviceUnavailableError,
    FrappeDecoder,
    FrappeEncoder,
    PrecisionUnavailableError,
    available_devices,
    bit_exact_properties,
    plane_shapes_for,
    select_device,
)
from src.compressors.frappe.openvino_runtime import _core as _openvino_core

# The released 16-channel CR-50 schedule, as it appears in the checkpoint config.
CR50_PS = [32, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4, 2]


def require_graph() -> Path:
    """The exported graph pair, or skip. Probed in the body, not in a marker, so
    the reason names what is missing rather than a collection-time constant."""
    stem = os.environ.get("FRAPPE_ONNX_STEM")
    if not stem:
        pytest.skip("set FRAPPE_ONNX_STEM to an exported graph pair")
    path = Path(stem)
    if not path.with_name(path.name + "_encoder.onnx").exists():
        pytest.skip(f"no encoder graph beside {path}")
    return path


# ---- device resolution -------------------------------------------------


def test_cpu_is_always_enumerated():
    """Without a CPU plugin nothing in this module can be tested at all."""
    assert "CPU" in available_devices()


def test_select_device_takes_the_first_available_preference():
    choice = select_device(["CPU"])
    assert choice.device == "CPU"


def test_select_device_records_why_earlier_preferences_were_skipped():
    """A fallback that leaves no trace is indistinguishable from a first choice."""
    choice = select_device(["NO_SUCH_DEVICE", "CPU"])
    assert choice.device == "CPU"
    assert dict(choice.considered)["NO_SUCH_DEVICE"]


def test_select_device_raises_when_no_preference_is_available():
    with pytest.raises(DeviceUnavailableError) as raised:
        select_device(["NO_SUCH_DEVICE", "ALSO_MISSING"])
    assert "NO_SUCH_DEVICE" in str(raised.value)
    assert "ALSO_MISSING" in str(raised.value)


@pytest.mark.parametrize("meta", sorted(META_DEVICES))
def test_meta_devices_are_refused(meta):
    """AUTO and friends resolve the device themselves, which is what we forbid."""
    with pytest.raises(ValueError, match="meta-device"):
        select_device([meta])


# ---- plane geometry ----------------------------------------------------


def test_plane_shapes_follow_the_papers_relation():
    """rows = n_s * (max_ps / p_s) * units_h, cols = (max_ps / p_s) * units_w."""
    shapes = plane_shapes_for(CR50_PS, height=608, width=800)
    # 800x608 with max_ps 32 is 25 x 19 units. Groups: 1x p32, 5x p16, 3x p8,
    # 6x p4, 1x p2.
    assert shapes == [
        (1 * 1 * 19, 1 * 25),      # p=32: factor 1
        (5 * 2 * 19, 2 * 25),      # p=16: factor 2
        (3 * 4 * 19, 4 * 25),      # p=8:  factor 4
        (6 * 8 * 19, 8 * 25),      # p=4:  factor 8
        (1 * 16 * 19, 16 * 25),    # p=2:  factor 16
    ]


def test_plane_shapes_scale_linearly_with_the_image():
    small = plane_shapes_for(CR50_PS, height=64, width=96)
    large = plane_shapes_for(CR50_PS, height=128, width=192)
    assert [(2 * r, 2 * c) for r, c in small] == large


def test_plane_shapes_reject_a_size_the_analysis_cannot_tile():
    """Non-overlapping analysis has no meaning on a ragged edge."""
    with pytest.raises(ValueError, match="multiple"):
        plane_shapes_for(CR50_PS, height=600, width=800)


def test_plane_shapes_reject_a_grid_the_reflect_padding_cannot_use():
    """The first decoder convolution reflects, which is undefined on one element."""
    with pytest.raises(ValueError, match="at least"):
        plane_shapes_for(CR50_PS, height=32, width=32)


@pytest.mark.parametrize("schedule", [
    CR50_PS,
    [32, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4, 2, 2],          # CR-40, 17ch
    [32] * 3 + [16] * 6 + [8] * 3 + [4] * 6 + [2] * 3,                   # released, 21ch
    [8],
])
def test_scale_grouping_agrees_with_the_torch_side_definition(schedule):
    """openvino_runtime re-derives the grouping so a deployment needs no torch.

    Re-derivation is only acceptable while the two agree, so the agreement is
    checked rather than assumed -- the same standard the repository already
    applies to the pixel_unshuffle rewrite of adapt_to_decoder.
    """
    pytest.importorskip("torch", reason="the ops.py definition pulls in torch")
    from src.compressors.frappe.ops import get_scale_groups

    from src.compressors.frappe.openvino_runtime import scale_groups_of

    expected = [(ps, end - start) for ps, start, end in
                get_scale_groups(schedule, len(schedule))]
    assert scale_groups_of(schedule) == expected


# ---- graph execution ---------------------------------------------------


def test_encoder_output_planes_match_the_declared_geometry():
    stem = require_graph()
    encoder = FrappeEncoder(stem, device="CPU", height=608, width=800)
    image = np.zeros((1, 3, 608, 800), dtype=np.uint8)
    planes = encoder(image)
    assert [plane.shape[1:] for plane in planes] == encoder.plane_shapes
    assert all(plane.dtype == np.uint8 for plane in planes)


def test_decoder_returns_the_image_geometry_it_was_pinned_to():
    stem = require_graph()
    encoder = FrappeEncoder(stem, device="CPU", height=608, width=800)
    decoder = FrappeDecoder(stem, device="CPU", plane_shapes=encoder.plane_shapes)
    planes = encoder(np.zeros((1, 3, 608, 800), dtype=np.uint8))
    reconstruction = decoder(planes)
    assert reconstruction.shape == (1, 3, 608, 800)
    assert reconstruction.dtype == np.uint8


def test_an_unavailable_device_fails_instead_of_running_somewhere_else():
    """The failure mode this module exists to prevent."""
    stem = require_graph()
    with pytest.raises(DeviceUnavailableError):
        FrappeEncoder(stem, device="NO_SUCH_DEVICE", height=608, width=800)


@pytest.mark.parametrize("device", ["CPU", "GPU", "NPU"])
def test_a_device_that_claims_fp32_reproduces_the_bitstream_exactly(device):
    """The encoder's planes are the file, so a device that can be exact must be.

    ``bit_exact_properties`` is the policy: it returns the settings that make a
    device reproduce the reference codes, and refuses for a device that cannot.
    Both branches are exercised here, so neither the capability nor its absence
    is taken on trust.
    """
    stem = require_graph()
    if device not in available_devices():
        pytest.skip(f"{device} is not present on this machine")
    try:
        properties = bit_exact_properties(device)
    except PrecisionUnavailableError:
        pytest.skip(f"{device} advertises no FP32; exactness is tested elsewhere")
    # Uniform noise is the adversarial input: it puts the most companded values
    # near a rounding boundary, where fp16 and fp32 disagree.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (1, 3, 608, 800), dtype=np.uint8)
    reference = FrappeEncoder(stem, device="CPU", height=608, width=800)(image)
    candidate = FrappeEncoder(stem, device=device, height=608, width=800,
                              properties=properties)(image)
    for index, (want, got) in enumerate(zip(reference, candidate)):
        mismatched = int((want != got).sum())
        assert mismatched == 0, (
            f"{device} plane {index}: {mismatched}/{want.size} symbols differ, "
            f"max |difference| {int(np.abs(want.astype(int) - got.astype(int)).max())}")


def test_a_device_without_fp32_is_refused_rather_than_given_a_useless_hint():
    """An NPU that advertises only FP16 will accept INFERENCE_PRECISION_HINT=f32
    and ignore it. Returning that hint would look like a fix and be none."""
    if "NPU" not in available_devices():
        pytest.skip("NPU is not present on this machine")
    capabilities = _openvino_core().get_property("NPU", "OPTIMIZATION_CAPABILITIES")
    if "FP32" in capabilities:
        pytest.skip(f"this NPU advertises FP32 ({list(capabilities)})")
    with pytest.raises(PrecisionUnavailableError, match="fp32"):
        bit_exact_properties("NPU")


@pytest.mark.parametrize("device", ["GPU", "NPU"])
def test_an_inexact_device_is_off_by_at_most_one_code_level(device):
    """What a device without fp32 costs, bounded rather than assumed.

    A code that differs by one is still a valid code -- it feeds a lossless
    entropy coder -- so the deployment question is the size of the effect, not
    its existence. Measured over real images it is about +0.0004 bpp and
    -0.0004 dB. What would be a defect is a larger deviation, so that is what is
    asserted here.
    """
    stem = require_graph()
    if device not in available_devices():
        pytest.skip(f"{device} is not present on this machine")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (1, 3, 608, 800), dtype=np.uint8)
    reference = FrappeEncoder(stem, device="CPU", height=608, width=800)(image)
    candidate = FrappeEncoder(stem, device=device, height=608, width=800)(image)
    symbols = sum(plane.size for plane in reference)
    mismatched = sum(int((want != got).sum())
                     for want, got in zip(reference, candidate))
    worst = max(int(np.abs(want.astype(np.int32) - got.astype(np.int32)).max())
                for want, got in zip(reference, candidate))
    assert worst <= 1, f"{device} is off by {worst} code levels, not a rounding boundary"
    assert mismatched <= 0.01 * symbols, (
        f"{device}: {mismatched}/{symbols} symbols differ, more than 1%")
