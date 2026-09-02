"""Contract tests for the native CharLS encoder and its thread parallelism.

Two properties carry the whole design. The native path must be byte-identical
to the PIL path it replaces -- the repository's rate numbers and its export
self-verification both hang off those exact bytes -- and it must actually run
concurrently, which is the only reason it exists: ctypes foreign calls release
the GIL, so five planes can encode at once where the PIL path forces them to
take turns.

The byte-identity is asserted against PIL directly, not against the harness's
own ``encode_plane`` -- once the harness prefers the native path, comparing
against it would be circular.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
import torch
from PIL import Image

from src.compressors.frappe.harness import (
    BitstreamConvention,
    charls_native,
    decode_planes,
    encode_planes,
)


def require_native() -> None:
    if not charls_native.available():
        pytest.skip(f"libcharls is not loadable: {charls_native._load_error}")


def pillow_encode(plane: np.ndarray) -> bytes:
    """What the PIL path writes, constructed here to keep the comparison honest."""
    import pillow_jpls  # noqa: F401

    buffer = io.BytesIO()
    Image.fromarray(plane, mode="L").save(buffer, format="JPEG-LS")
    return buffer.getvalue()


@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 3), (19, 25), (50, 190), (128, 128), (912, 200)])
def test_native_encoding_is_byte_identical_to_the_pil_path(rows, cols):
    require_native()
    rng = np.random.default_rng(rows * 1000 + cols)
    plane = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    assert charls_native.encode_plane(plane) == pillow_encode(plane)


@pytest.mark.parametrize("fill", [0, 128, 255])
def test_constant_planes_are_byte_identical_too(fill):
    require_native()
    plane = np.full((64, 96), fill, dtype=np.uint8)
    assert charls_native.encode_plane(plane) == pillow_encode(plane)


def test_a_non_plane_is_refused():
    require_native()
    with pytest.raises(ValueError, match="2D uint8"):
        charls_native.encode_plane(np.zeros(8, dtype=np.uint8))


def test_the_harness_path_round_trips_through_the_self_describing_form():
    """The deployed path: harness bytes in, the same planes out."""
    require_native()
    rng = np.random.default_rng(3)
    planes = [
        torch.from_numpy(rng.integers(0, 256, (rows, cols), dtype=np.uint8))
        for rows, cols in [(19, 25), (190, 50), (304, 400)]
    ]
    blob = encode_planes(planes, BitstreamConvention.WITH_LENGTH_PREFIX)
    assert all(torch.equal(want, got) for want, got in zip(planes, decode_planes(blob)))


def test_parallel_encoding_produces_the_sequential_bytes():
    """Scheduling must not be observable in the bitstream: the concatenation
    order is the caller's order whatever order the threads finish in."""
    require_native()
    rng = np.random.default_rng(5)
    planes = [
        torch.from_numpy(rng.integers(0, 256, (rows, cols), dtype=np.uint8))
        for rows, cols in [(19, 25), (190, 50), (228, 100), (912, 200), (304, 400)]
    ]
    parallel = encode_planes(planes, BitstreamConvention.PAYLOAD_ONLY)
    sequential = b"".join(pillow_encode(plane.numpy()) for plane in planes)
    assert parallel == sequential


def test_the_native_encode_releases_the_gil():
    """The property that makes a native wrapper worth its dependency: while a
    foreign encode runs, other Python threads make progress. Probed
    structurally rather than by wall-time ratio -- a background encode that is
    still running after a spin of Python on the main thread can only mean the
    GIL was released during the foreign call."""
    require_native()
    import threading

    rng = np.random.default_rng(11)
    plane = rng.integers(0, 256, (4800, 400), dtype=np.uint8)  # ~50 ms of encoding
    finished = []

    thread = threading.Thread(target=lambda: finished.append(charls_native.encode_plane(plane)))
    thread.start()
    time.sleep(0.005)  # let the thread enter the foreign call

    deadline = time.perf_counter() + 0.02
    while time.perf_counter() < deadline:
        pass  # 20 ms of pure Python on the main thread

    assert not finished, (
        "the encode finished during 20 ms of main-thread Python; "
        "it should take tens of ms, so the GIL was never released"
    )
    thread.join()
    assert finished


def test_five_planes_encode_to_the_sequential_bytes():
    """The harness path, whatever encoder backs it, must produce the same
    bitstream as the PIL path does plane by plane."""
    require_native()
    rng = np.random.default_rng(5)
    planes = [
        torch.from_numpy(rng.integers(0, 256, (rows, cols), dtype=np.uint8))
        for rows, cols in [(19, 25), (190, 50), (228, 100), (912, 200), (304, 400)]
    ]
    harness = encode_planes(planes, BitstreamConvention.PAYLOAD_ONLY)
    sequential = b"".join(pillow_encode(plane.numpy()) for plane in planes)
    assert harness == sequential
