"""CharLS through its C API, for encoding planes with the GIL released.

``pillow_jpls`` holds the GIL for the whole encode. Foreign calls made
through :mod:`ctypes` release it, which is the reason this module exists: the
GIL becomes available to other threads during the encode, and a caller that
wants to overlap encoding with other Python work (the encoder graph of the
next image, for one) can.

Thread-parallelising the planes themselves was measured and rejected: at
FRAPPE's operating points one plane carries 77% of the samples, JPEG-LS has
no parallelism within a plane, and on the real planes a thread pool made the
path slower (2.0-2.8 ms against 1.75 ms sequential) -- scheduling overhead
exceeds the at most 0.3 ms that perfect overlap could hide. So this module is
about the GIL, not about threads.

Two properties the harness depends on, both checked by tests rather than
assumed. The codestream CharLS writes is the one ``pillow_jpls`` writes too
(it is the same library), and the 44-byte SPIFF prefix the PIL path prepends
is reconstructed here byte for byte, with only height and width patched in --
so a caller that switches from the PIL path to this one changes nothing about
the bitstream, only about who holds the GIL while producing it.

The library is loaded from the pixi environment's ``lib`` directory and fails
soft: :func:`available` is False when it is missing, and the PIL path remains
the fallback. If the C API someday disagrees with the PIL path, the
byte-identity tests fail before any rate number can silently shift.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

#: The SPIFF-wrapped container pillow_jpls writes for an 8-bit grayscale
#: image: SOI, an APP8 SPIFF header, and a one-entry SPIFF directory. Height
#: and width are the only parts that vary with the plane.
_SPIFF_PREFIX_TEMPLATE = bytes.fromhex(
    "ffd8ffe80020535049464600020000010000001300000019080806010000006000000060ffe8000800000001"
)
_HEIGHT_OFFSET = 16
_WIDTH_OFFSET = 20


class FrameInfo(ctypes.Structure):
    """``charls_frame_info``: the geometry CharLS writes into the SOF segment."""

    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bits_per_sample", ctypes.c_int32),
        ("component_count", ctypes.c_int32),
    ]


_state: dict = {"lib": None, "error": None}


def _candidate_paths():
    yield Path(sys.prefix) / "lib" / "libcharls.so.2"
    yield "libcharls.so.2"


def _load():
    if _state["lib"] is None and _state["error"] is None:
        last = None
        for candidate in _candidate_paths():
            try:
                lib = ctypes.CDLL(candidate)
                _setup(lib)
                _state["lib"] = lib
                break
            except OSError as error:
                last = error
        if _state["lib"] is None:
            _state["error"] = str(last)
    return _state["lib"]


def _setup(lib) -> None:
    lib.charls_jpegls_encoder_create.restype = ctypes.c_void_p
    lib.charls_jpegls_encoder_create.argtypes = []
    lib.charls_jpegls_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.charls_jpegls_encoder_set_destination_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.charls_jpegls_encoder_set_frame_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(FrameInfo)]
    lib.charls_jpegls_encoder_set_near_lossless.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.charls_jpegls_encoder_encode_from_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.charls_jpegls_encoder_get_bytes_written.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.charls_get_error_message.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    for name in (
        "charls_jpegls_encoder_set_destination_buffer",
        "charls_jpegls_encoder_set_frame_info",
        "charls_jpegls_encoder_set_near_lossless",
        "charls_jpegls_encoder_encode_from_buffer",
        "charls_jpegls_encoder_get_bytes_written",
    ):
        getattr(lib, name).restype = ctypes.c_int
    lib.charls_get_error_message.restype = ctypes.c_int


def available() -> bool:
    """Whether the CharLS C API is loadable in this environment."""
    return _load() is not None


def _error_message(error_code: int) -> str:
    lib = _state["lib"]
    if lib is None:
        return f"charls error {error_code}"
    buffer = ctypes.create_string_buffer(512)
    lib.charls_get_error_message(error_code, buffer, len(buffer))
    return buffer.value.decode(errors="replace")


def _spiff_prefix(height: int, width: int) -> bytes:
    prefix = bytearray(_SPIFF_PREFIX_TEMPLATE)
    prefix[_HEIGHT_OFFSET : _HEIGHT_OFFSET + 4] = int(height).to_bytes(4, "big")
    prefix[_WIDTH_OFFSET : _WIDTH_OFFSET + 4] = int(width).to_bytes(4, "big")
    return bytes(prefix)


def encode_plane(plane: np.ndarray) -> bytes:
    """One 2D uint8 plane as the JPEG-LS stream the PIL path would write.

    The foreign calls release the GIL, so encodes from this function run
    concurrently across threads.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(f"libcharls is not loadable: {_state['error']}")
    plane = np.ascontiguousarray(plane)
    if plane.ndim != 2 or plane.dtype != np.uint8:
        raise ValueError(f"expected a 2D uint8 plane, got {plane.ndim}D {plane.dtype}")
    height, width = plane.shape

    # NEAR=0 on 8-bit samples never exceeds one byte per sample plus markers;
    # twice that plus a page is cheap insurance against a pathological run.
    destination = (ctypes.c_char * (plane.size * 2 + 65536))()
    frame_info = FrameInfo(width=width, height=height, bits_per_sample=8, component_count=1)
    encoder = lib.charls_jpegls_encoder_create()
    if not encoder:
        raise RuntimeError("charls_jpegls_encoder_create returned null")
    try:

        def check(error_code: int, what: str) -> None:
            if error_code != 0:
                raise RuntimeError(f"charls {what} failed: {_error_message(error_code)}")

        check(
            lib.charls_jpegls_encoder_set_destination_buffer(
                encoder, destination, len(destination)
            ),
            "set_destination_buffer",
        )
        check(
            lib.charls_jpegls_encoder_set_frame_info(encoder, ctypes.byref(frame_info)),
            "set_frame_info",
        )
        check(lib.charls_jpegls_encoder_set_near_lossless(encoder, 0), "set_near_lossless")
        check(
            lib.charls_jpegls_encoder_encode_from_buffer(encoder, plane.ctypes.data, plane.size, 0),
            "encode_from_buffer",
        )
        written = ctypes.c_size_t()
        check(
            lib.charls_jpegls_encoder_get_bytes_written(encoder, ctypes.byref(written)),
            "get_bytes_written",
        )
        return _spiff_prefix(height, width) + destination.raw[: written.value]
    finally:
        lib.charls_jpegls_encoder_destroy(encoder)
