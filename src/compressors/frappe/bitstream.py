"""The JPEG-LS container FRAPPE's code planes travel in.

Since the ONNX split point moved to the entropy coder, the encoder graph already
emits what JPEG-LS consumes: one uint8 grayscale plane per scale group, shaped
``(n_s * H/p_s, W/p_s)``, with the signed codes already shifted by +127. All that
is left outside the graph is the standard codec and the container -- which is
what this module is.

It exists as its own module for three reasons.

*The shift belongs to exactly one side.* ``entropy_coding.encode_latents`` takes
*signed* arranged latents and applies ``+127`` itself. Feeding it the graph's
output would shift twice, and the result is not an error: it is a valid JPEG-LS
stream that decodes to a plausible-looking wrong image. The functions here take
planes that are already unsigned, so the two entry points cannot be confused.

*The length prefix is a choice, not a detail.* ``encode_latents`` prefixes each
scale's stream with a four-byte big-endian length so the blob is self-describing;
``evaluate.py``, ``tools/evaluate_joint_prefix.py`` and the notebook sum the bare
payloads instead. The two families of bitrate differ by ``32 * G / (T1 * T2)`` --
3.29e-4 bpp for five scale groups at 800x608, under 0.07% of rate, but not zero.
``temp/worklog_Sep02-2026_frappe_40db.md`` asks for the convention to live in one
module behind an explicit argument rather than in four copies of a byte count.
That argument is :func:`encode_planes`'s ``length_prefix``, and
:func:`measure` reports both numbers so a rate can never be quoted without saying
which one it is.

*A deployment should not need torch.* Nothing here imports it; the planes are
numpy arrays and PIL does the coding. ``tests`` asserts that the bytes this
module produces are identical to the ones ``entropy_coding`` produces from the
corresponding signed latents, so the duplication is checked rather than assumed.
"""

from __future__ import annotations

import io
import struct

import numpy as np

#: Width of the big-endian length that precedes each scale's stream in the
#: self-describing form, matching ``entropy_coding.encode_latents``.
LENGTH_PREFIX_BYTES = 4

#: Codes are signed 8-bit and JPEG-LS is unsigned 8-bit, so streams carry
#: ``code + CODE_OFFSET``. The exported encoder graph applies this itself.
CODE_OFFSET = 127


def _pil():
    import pillow_jpls  # noqa: F401 -- registers the JPEG-LS plugin with PIL
    from PIL import Image

    return Image


def _as_plane(plane) -> np.ndarray:
    array = np.asarray(plane)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]  # the graph carries the batch axis the exporter fixed at 1
    if array.ndim != 2:
        raise ValueError(f"a JPEG-LS plane is 2D, got shape {tuple(array.shape)}")
    if array.dtype != np.uint8:
        raise ValueError(
            f"a JPEG-LS plane is uint8, got {array.dtype}. Signed codes come from "
            "entropy_coding.arrange_latents and still need the +127 shift; the "
            "exported encoder graph has already applied it.")
    return np.ascontiguousarray(array)


def encode_plane(plane) -> bytes:
    """The bare JPEG-LS stream for one already-arranged uint8 plane.

    The plane is validated before the codec is imported, so a caller that has the
    dtype wrong is told that rather than that pillow-jpls is missing.
    """
    array = _as_plane(plane)
    image = _pil()
    buffer = io.BytesIO()
    image.fromarray(array, mode="L").save(buffer, format="JPEG-LS")
    return buffer.getvalue()


def decode_plane(payload: bytes) -> np.ndarray:
    """Inverse of :func:`encode_plane`. The stream carries its own dimensions."""
    image = _pil()
    handle = image.open(io.BytesIO(payload))
    handle.load()
    return np.asarray(handle, dtype=np.uint8)


def encode_planes(planes, *, length_prefix: bool = True) -> bytes:
    """Concatenate each plane's JPEG-LS stream into one blob.

    With ``length_prefix`` the blob is self-describing -- each stream is preceded
    by its big-endian four-byte length, which is the form
    ``entropy_coding.encode_latents`` writes and the only form
    :func:`decode_planes` can read. Without it the result is the bare
    concatenation whose length is what ``tools/evaluate_joint_prefix.py`` and the
    rest of the reporting tools measure.
    """
    chunks: list[bytes] = []
    for plane in planes:
        payload = encode_plane(plane)
        if length_prefix:
            chunks.append(struct.pack(">I", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def decode_planes(blob: bytes) -> list[np.ndarray]:
    """Read a length-prefixed blob back into uint8 planes.

    The number of scale groups is implicit: the reader walks until the buffer is
    exhausted, exactly as ``entropy_coding.decode_latents`` does.
    """
    planes: list[np.ndarray] = []
    offset, total = 0, len(blob)
    while offset < total:
        if offset + LENGTH_PREFIX_BYTES > total:
            raise ValueError("truncated blob: missing the 4-byte length prefix")
        (length,) = struct.unpack_from(">I", blob, offset)
        offset += LENGTH_PREFIX_BYTES
        if offset + length > total:
            raise ValueError(
                f"truncated blob: a chunk of {length} bytes overruns the buffer")
        planes.append(decode_plane(blob[offset:offset + length]))
        offset += length
    return planes


def measure(planes, height: int, width: int) -> dict:
    """Both rate conventions for one image, so neither can be quoted unlabelled.

    ``height`` and ``width`` are the *image's*, not a plane's: bits per pixel is
    always per source pixel, which is what makes the number comparable with the
    reference codecs and with ``24 / bpp`` as a compression ratio.
    """
    payloads = [len(encode_plane(plane)) for plane in planes]
    pixels = int(height) * int(width)
    if pixels <= 0:
        raise ValueError("an image with no pixels has no bitrate")
    bare = sum(payloads)
    prefixed = bare + LENGTH_PREFIX_BYTES * len(payloads)
    return {
        "planes": len(payloads),
        "payload_bytes": payloads,
        "bytes_payload_only": bare,
        "bytes_with_length_prefix": prefixed,
        "bpp_payload_only": bare * 8 / pixels,
        "bpp_with_length_prefix": prefixed * 8 / pixels,
        "compression_ratio_payload_only": 24.0 * pixels / (bare * 8),
        "compression_ratio_with_length_prefix": 24.0 * pixels / (prefixed * 8),
    }
