"""Contract tests for the JPEG-LS front half as an OpenVINO graph.

The failure mode that makes this module need more than a smoke test is silent
miscompilation: on this machine's NPU, comparison-based formulations of the
gradient quantiser were measured producing thousands of wrong samples per plane
while staying finite and integral, so checks that only look for NaNs or
non-integers pass on corrupted output. The only check that catches it is exact
equality with a reference computed on a device known to be right, which is why
``verify="all"`` is the contract's default and why every device test here runs
that path against an independently written CPU formulation.

The oracle is checked against a second, differently spelled implementation of
the same identities -- a ReLU-arithmetic MED instead of Minimum/Maximum, and a
comparison-based quantiser instead of the module's arithmetic one. Two
formulations agreeing exactly on integer inputs is what makes a transcription
mistake in either one visible.

Tests that need a particular accelerator are skipped when OpenVINO does not
enumerate it, so the suite is honest on a machine without an NPU rather than
green by omission.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.compressors.frappe.jpegls_openvino import (
    CONTEXT_RADIX,
    GRADIENT_THRESHOLDS,
    MedContextGraph,
    VerificationError,
    build_line_buffer,
    precompute,
    reference_med_and_context,
)
from src.compressors.frappe.openvino_runtime import available_devices


def _independent_oracle(line_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same identities as ``reference_med_and_context``, differently spelled.

    MED through the ReLU-arithmetic clamp of the reference concept note, the
    quantiser through the obvious comparison spelling. On CPU both spellings
    are exact integer arithmetic, so a disagreement is a transcription bug, not
    a precision artefact.
    """
    patch = np.asarray(line_buffer, dtype=np.int32)
    a = patch[1:, :-2]
    b = patch[:-1, 1:-1]
    c = patch[:-1, :-2]
    d = patch[:-1, 2:]
    upper = np.maximum(a - b, 0) + b
    lower = a + b - upper
    raised = np.maximum(a + b - c - lower, 0) + lower
    prediction = raised - np.maximum(raised - upper, 0)

    def quantise(gradient):
        magnitude = np.abs(gradient)
        steps = sum((magnitude > threshold).astype(np.int32) for threshold in (0, 2, 6, 20))
        return np.sign(gradient) * steps

    context = CONTEXT_RADIX**2 * quantise(d - b) + CONTEXT_RADIX * quantise(b - c) + quantise(c - a)
    return prediction, context


# ---- line buffer boundaries ---------------------------------------------


def test_the_first_line_is_the_all_zero_line():
    buffer = build_line_buffer(np.full((4, 7), 200, dtype=np.uint8))
    assert buffer.shape == (5, 9)
    assert np.all(buffer[0] == 0)


def test_the_right_border_repeats_the_last_column():
    plane = np.arange(12, dtype=np.uint8).reshape(3, 4)
    buffer = build_line_buffer(plane)
    assert np.array_equal(buffer[1:, -1], plane[:, -1])


def test_the_left_border_carries_the_previous_row_into_the_current_one():
    """Row ``i``'s left border is the first sample of row ``i - 1``; row 1's
    predecessor is the zero line, so its border is zero."""
    plane = np.arange(12, dtype=np.uint8).reshape(3, 4)
    buffer = build_line_buffer(plane)
    assert buffer[1, 0] == 0
    assert np.array_equal(buffer[2:, 0], plane[:-1, 0])


def test_the_image_itself_sits_in_the_middle_of_the_buffer():
    plane = np.arange(12, dtype=np.uint8).reshape(3, 4)
    buffer = build_line_buffer(plane)
    assert np.array_equal(buffer[1:, 1:-1], plane)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 4), dtype=np.int16),
        np.zeros((3, 4), dtype=np.float32),
        np.zeros(4, dtype=np.uint8),
    ],
)
def test_a_non_plane_is_refused(bad):
    with pytest.raises(ValueError, match="2D uint8"):
        build_line_buffer(bad)


# ---- the CPU oracle ------------------------------------------------------


def test_a_known_patch_gives_the_hand_computed_values():
    """Rows ``[3, 4, 6]`` and ``[1, 9, 2]``, hand-computed sample by sample.

    On row 0 the neighbours B, C, D all sit on the zero line, so the context is
    only ``quantise(C - A)`` -- the second sample gets 3 from clamping A+B-C to
    [0, 3] and -2 from sign(-3) * 2. Row 1 has real neighbours everywhere; the
    first sample reads A=3 B=3 C=0 D=4, giving prediction clamp(6, 3, 3) = 3
    and context 81*1 + 9*2 - 2 = 97, which is the one sample that exercises the
    leading 81 term. Hand arithmetic, so the oracle is anchored to something no
    formulation shares.
    """
    prediction, context = reference_med_and_context(
        build_line_buffer(np.array([[3, 4, 6], [1, 9, 2]], dtype=np.uint8))
    )
    assert prediction.tolist() == [[0, 3, 4], [3, 2, 9]]
    assert context.tolist() == [[0, -2, -2], [97, 91, 7]]


@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 3), (3, 2), (17, 1), (1, 17), (19, 25)])
def test_the_oracle_matches_an_independently_written_formulation(rows, cols):
    rng = np.random.default_rng(rows * 1000 + cols)
    plane = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    buffer = build_line_buffer(plane)
    want_prediction, want_context = _independent_oracle(buffer)
    got_prediction, got_context = reference_med_and_context(buffer)
    assert np.array_equal(got_prediction, want_prediction)
    assert np.array_equal(got_context, want_context)


def test_the_thresholds_are_the_eight_bit_lossless_defaults():
    """T1/T2/T3 = 3/7/21 as strict-inequality magnitudes, one step per test."""
    assert GRADIENT_THRESHOLDS == (0, 2, 6, 20)


# ---- running the graph ---------------------------------------------------


def test_precompute_on_cpu_matches_the_oracle_and_says_so():
    rng = np.random.default_rng(7)
    plane = rng.integers(0, 256, (19, 25), dtype=np.uint8)
    report = precompute(plane, device="CPU")
    want_prediction, want_context = reference_med_and_context(build_line_buffer(plane))
    assert np.array_equal(report["prediction"], want_prediction)
    assert np.array_equal(report["context"], want_context)
    assert report["device"] == "CPU"
    assert report["shape"] == [19, 25]
    assert report["verified"] is True
    assert report["execution_devices"]


def test_verifying_against_the_oracle_is_the_default():
    rng = np.random.default_rng(8)
    plane = rng.integers(0, 256, (19, 25), dtype=np.uint8)
    default = precompute(plane, device="CPU")
    explicit = precompute(plane, device="CPU", verify="all")
    assert np.array_equal(default["prediction"], explicit["prediction"])
    assert np.array_equal(default["context"], explicit["context"])
    assert explicit["verified"] is True


def test_verify_none_skips_the_oracle_and_says_so():
    rng = np.random.default_rng(9)
    plane = rng.integers(0, 256, (19, 25), dtype=np.uint8)
    report = precompute(plane, device="CPU", verify="none")
    assert report["verified"] is False


def test_an_unknown_verify_policy_is_refused():
    plane = np.zeros((4, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="'all' or 'none'"):
        precompute(plane, device="CPU", verify="spot")


def test_a_graph_pinned_to_one_shape_rejects_another():
    graph = MedContextGraph(4, 6, "CPU")
    with pytest.raises(ValueError, match="line buffer"):
        graph(np.zeros((4 + 1, 9 + 2), dtype=np.int16))


def test_an_unavailable_device_fails_instead_of_running_somewhere_else():
    from src.compressors.frappe.openvino_runtime import DeviceUnavailableError

    with pytest.raises(DeviceUnavailableError):
        MedContextGraph(4, 6, "NO_SUCH_DEVICE")


# ---- device exactness ----------------------------------------------------


@pytest.mark.parametrize("rows,cols", [(19, 25), (190, 50)])
def test_every_enumerated_device_matches_the_cpu_oracle(rows, cols):
    """The load-bearing test: exact equality, every device, not a spot check.

    On this machine's NPU the obvious comparison spellings of the quantiser
    were measured silently wrong at exactly these shapes, while staying finite
    and integral. A regression to such a formulation fails here, loudly, on
    every device it corrupts.
    """
    rng = np.random.default_rng(rows)
    plane = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    want_prediction, want_context = reference_med_and_context(build_line_buffer(plane))
    for device in available_devices():
        report = precompute(plane, device=device)
        assert np.array_equal(report["prediction"], want_prediction), device
        assert np.array_equal(report["context"], want_context), device


@pytest.mark.parametrize(
    "name,maker",
    [
        ("constant", lambda rows, cols: np.full((rows, cols), 128, dtype=np.uint8)),
        (
            "alternating",
            lambda rows, cols: (np.indices((rows, cols)).sum(0) % 2 * 255).astype(np.uint8),
        ),
        (
            "extremes",
            lambda rows, cols: np.where((np.indices((rows, cols)).sum(0)) % 2, 255, 0).astype(
                np.uint8
            ),
        ),
    ],
)
def test_degenerate_content_matches_the_cpu_oracle_everywhere(name, maker):
    """Flat, alternating and two-valued planes are where a predictor or an
    indicator function is most likely to diverge from its intent."""
    rows, cols = 19, 25
    plane = maker(rows, cols)
    want_prediction, want_context = reference_med_and_context(build_line_buffer(plane))
    for device in available_devices():
        report = precompute(plane, device=device)
        assert np.array_equal(report["prediction"], want_prediction), (name, device)
        assert np.array_equal(report["context"], want_context), (name, device)


def test_the_front_half_never_raises_verification_error_on_correct_devices():
    """``precompute`` raises ``VerificationError`` rather than returning wrong
    values; on the devices this suite accepts, it must not raise at all."""
    rng = np.random.default_rng(11)
    plane = rng.integers(0, 256, (2, 3), dtype=np.uint8)
    for device in available_devices():
        try:
            precompute(plane, device=device)
        except VerificationError as error:
            pytest.fail(f"{device} silently disagreed with the CPU: {error}")
