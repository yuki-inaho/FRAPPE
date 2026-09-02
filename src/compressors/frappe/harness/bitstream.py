"""The one place that turns latents into bytes.

The codec's bitstream is defined in :mod:`compressors.frappe.entropy_coding`:
each scale's ``(1, C, H, W)`` latent becomes a single 2D grayscale plane
``(C*H, W)``, the signed codes are shifted into ``uint8``, and each scale's
JPEG-LS stream is written behind a 4-byte big-endian length. The paper states
the same thing as ``(n_s * T1/p_s, T2/p_s)`` with length-prefixed JPEG-LS.

Every tool that reports a bitrate needs a slice of that, and until this module
existed each one carried its own copy. The copies drifted: some counted the
length prefixes and some did not, which is a 4-byte-per-scale difference nobody
notices until two tools' numbers are put in one table. Here the convention is an
argument, so choosing it is deliberate and the choice travels with the number.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from .annotations import Int8, Shaped, Tensor, UInt8, checked

#: Signed codes live in ``[-127, 127]``; JPEG-LS wants unsigned 8-bit, so the
#: streams carry ``code + CODE_OFFSET``.
CODE_OFFSET = 127

#: Bytes ``encode_planes`` spends on each scale's big-endian length prefix.
LENGTH_PREFIX_BYTES = 4


@dataclass(frozen=True)
class BitstreamConvention:
    """How a measured bitrate is defined.

    ``count_length_prefix`` is the only thing the repository was inconsistent
    about. ``entropy_coding.encode_latents`` writes the prefixes and
    ``evaluate_rate_distortion`` measures the resulting blob, so its numbers
    include them; ``evaluate.py``, the notebook and the joint-prefix tools sum
    bare payloads, so theirs do not. The difference is
    ``8 * LENGTH_PREFIX_BYTES * groups / pixels`` -- 3.3e-4 bpp for five groups
    at 800x608, under 0.07% of rate. Small, but not zero, and not something two
    tables should silently disagree about.
    """

    count_length_prefix: bool = False

    #: The convention of ``evaluate.py``, the notebook, and the joint-prefix
    #: tools: bare JPEG-LS payloads.
    PAYLOAD_ONLY: ClassVar[BitstreamConvention]
    #: The convention of ``entropy_coding.encode_latents`` and the shipped
    #: rate-distortion results: a self-describing container.
    WITH_LENGTH_PREFIX: ClassVar[BitstreamConvention]

    def overhead_bytes(self, groups: int) -> int:
        return LENGTH_PREFIX_BYTES * groups if self.count_length_prefix else 0


BitstreamConvention.PAYLOAD_ONLY = BitstreamConvention(count_length_prefix=False)
BitstreamConvention.WITH_LENGTH_PREFIX = BitstreamConvention(count_length_prefix=True)


@checked
def arrange_plane(latent: Int8[Tensor, "channels grid_h grid_w"]
                  ) -> UInt8[Tensor, "rows cols"]:
    """One scale group's latent as the grayscale image JPEG-LS receives.

    ``(C, H, W) -> (C*H, W)`` with the codes shifted into ``uint8``: channel
    major, row minor, C-contiguous. This is ``entropy_coding.arrange_latents``
    followed by the shift ``encode_latents`` applies, which is the pair the ONNX
    encoder reproduces inside its graph.
    """
    channels, grid_h, grid_w = latent.shape
    flat = latent.reshape(channels * grid_h, grid_w).contiguous()
    return (flat.to(torch.int16) + CODE_OFFSET).to(torch.uint8)


def arrange_planes(latents: Iterable[Shaped[Tensor, ...]],
                   channels: Sequence[int] | None = None) -> list[Tensor]:
    """Arrange every scale group, optionally keeping only the first channels.

    ``latents`` may be ``(1, C, H, W)`` or ``(C, H, W)``; the batch axis is
    dropped because a bitstream describes one image. ``channels`` truncates each
    group, which is how a prefix shorter than the full schedule is measured: the
    groups are contiguous in channel index, so keeping the first ``k`` of a group
    keeps exactly the right global channels.
    """
    planes = []
    for index, latent in enumerate(latents):
        plane = latent[0] if latent.dim() == 4 else latent
        if channels is not None:
            keep = channels[index]
            if keep <= 0:
                continue
            plane = plane[:keep]
        planes.append(arrange_plane(plane.to(torch.int8).cpu()))
    return planes


def prefix_channels(scale_groups: Sequence[tuple[int, int, int]],
                    n_channels: int) -> list[int]:
    """How many channels of each scale group the prefix ``1:n_channels`` keeps."""
    kept, remaining = [], n_channels
    for _, start, end in scale_groups:
        take = max(0, min(end - start, remaining))
        kept.append(take)
        remaining -= take
    return kept


def encode_plane(plane: UInt8[Tensor, "rows cols"], backend: str = "pillow") -> bytes:
    """The bare JPEG-LS stream for one arranged plane, through the named backend.

    ``pillow`` is the portable baseline (``pillow_jpls``). ``charls-native``
    is an explicit opt-in that calls CharLS's C API through ctypes; when the
    library is not loadable it raises rather than silently degrading to
    Pillow, so a caller never gets a backend it did not name.
    """
    if backend == "pillow":
        import pillow_jpls  # noqa: F401 -- registers the JPEG-LS plugin with PIL
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(plane.numpy(), mode="L").save(buffer, format="JPEG-LS")
        return buffer.getvalue()
    if backend == "charls-native":
        from .charls_native import encode_plane as native_encode

        return native_encode(plane.numpy())
    raise ValueError(f"unknown JPEG-LS backend {backend!r}; expected 'pillow' or 'charls-native'")


def encode_planes(planes: Sequence[Tensor],
                  convention: BitstreamConvention = BitstreamConvention.PAYLOAD_ONLY,
                  ) -> bytes:
    """Serialise arranged planes, with or without the self-describing prefixes."""
    chunks = []
    for plane in planes:
        payload = encode_plane(plane)
        if convention.count_length_prefix:
            chunks.append(struct.pack(">I", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def measure_rate(latents: Iterable[Tensor], pixels: int,
                 channels: Sequence[int] | None = None,
                 convention: BitstreamConvention = BitstreamConvention.PAYLOAD_ONLY,
                 ) -> tuple[int, float]:
    """``(bytes, bits per pixel)`` for one image's latents.

    ``pixels`` is the image's ``height * width``, not its element count: the
    codec's rate is per pixel, and the three colour channels are what it is
    compressing rather than something to divide by.
    """
    planes = arrange_planes(latents, channels)
    total = len(encode_planes(planes, convention))
    return total, total * 8 / pixels
