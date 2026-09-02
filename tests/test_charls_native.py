"""CharLS native: same bytes as the Pillow path, explicit absence, no fallback.

The native path exists so a caller that must not hold the GIL during the
encode can opt in by naming ``charls-native``; every other consumer stays on
the portable Pillow baseline. These tests run in the deployment environment
(uv, group ``deploy``).
"""

import numpy as np
import pytest
import torch

from src.compressors.frappe.harness.bitstream import encode_plane

CASES = [
    np.zeros((1, 1), dtype=np.uint8),
    np.arange(6, dtype=np.uint8).reshape(2, 3),
    np.random.default_rng(0).integers(0, 256, (20, 16), dtype=np.uint8),
    np.full((5, 7), 9, dtype=np.uint8),  # constant plane, JPEG-LS's worst ffmpeg case
]


def test_native_and_pillow_write_identical_streams():
    """Same library behind both paths: the codestream must be byte-identical."""
    import src.compressors.frappe.harness.charls_native as native

    assert native.available(), f"libcharls.so.2 must be loadable: {native.unavailable_reason()}"
    for plane in CASES:
        assert native.encode_plane(plane) == encode_plane(torch.from_numpy(plane), backend="pillow")


def test_native_is_opt_in_and_fails_loud():
    """The backend argument is the only switch; a missing library is an error."""
    import src.compressors.frappe.harness.charls_native as native

    assert native.unavailable_reason() is None, "the library resolves on this machine"
    plane = np.zeros((4, 4), dtype=np.uint8)
    assert isinstance(encode_plane(torch.from_numpy(plane), backend="charls-native"), bytes)
    assert encode_plane(torch.from_numpy(plane), backend="pillow") == encode_plane(
        torch.from_numpy(plane)
    )


def test_missing_library_is_reported_not_hidden(monkeypatch):
    """With no loadable CharLS, the native backend says why instead of degrading."""
    import src.compressors.frappe.harness.charls_native as native

    def refusing():
        yield "does/not/exist/libcharls.so.2"

    monkeypatch.setattr(native, "_candidate_paths", refusing)
    monkeypatch.setattr(native, "_state", {"lib": None, "error": None})
    assert native.available() is False
    assert "does/not/exist" in native.unavailable_reason()
    with pytest.raises(RuntimeError, match="libcharls"):
        native.encode_plane(np.zeros((2, 2), dtype=np.uint8))
