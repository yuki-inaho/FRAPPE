"""Standard codecs to compare against, and how to line them up fairly.

A rate-distortion table only means something next to what an ordinary codec
spends for the same quality on the same images, so every reference here is a real
encode of a real file, measured by its size on disk.

Two tools needed this and one imported it from the other, which made a
command-line script into a library it was never shaped to be. The comparison
logic lives here instead.

Codecs are probed at start-up and skipped when unavailable, so the same code runs
on a machine with a different Pillow build or without ffmpeg.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import psnr_from_mse

DEFAULT_LADDERS = {
    "jpeg": [50, 70, 80, 85, 90, 93, 95, 97, 98],
    "webp": [50, 70, 80, 85, 90, 93, 95, 98, 100],
    "jpeg2000": [40, 30, 20, 15, 12, 10, 8, 6, 4, 3],
    "avif": [50, 42, 36, 30, 24, 18, 12, 8],
}


def _encode_pillow(image: Image.Image, codec: str, setting: int) -> tuple[bytes, Image.Image]:
    buffer = io.BytesIO()
    if codec == "jpeg":
        image.save(buffer, format="JPEG", quality=int(setting), subsampling=0,
                   optimize=True)
    elif codec == "webp":
        image.save(buffer, format="WEBP", quality=int(setting), method=6)
    elif codec == "jpeg2000":
        # ``quality_mode="rates"`` reads quality_layers as compression ratios,
        # so the requested operating point is a compression ratio directly.
        image.save(buffer, format="JPEG2000", quality_mode="rates",
                   quality_layers=[float(setting)], irreversible=True)
    else:
        raise ValueError(f"{codec} is not a Pillow codec")
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        decoded.load()
        return payload, decoded.convert("RGB")


def _encode_ffmpeg(image: Image.Image, encoder: str, setting: int,
                   container: str) -> tuple[bytes, Image.Image]:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.png"
        target = Path(directory) / f"target.{container}"
        decoded_path = Path(directory) / "decoded.png"
        image.save(source, format="PNG")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-c:v", encoder, "-crf", str(int(setting)), "-still-picture", "1",
             "-cpu-used", "4", str(target)],
            check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(target), str(decoded_path)], check=True)
        payload = target.read_bytes()
        with Image.open(decoded_path) as decoded:
            decoded.load()
            return payload, decoded.convert("RGB")


def available_codecs(requested: list[str]) -> list[str]:
    from PIL import features

    probe = Image.new("RGB", (64, 64), (120, 90, 60))
    usable = []
    for codec in requested:
        try:
            if codec == "avif":
                if shutil.which("ffmpeg") is None:
                    raise RuntimeError("ffmpeg is not installed")
                _encode_ffmpeg(probe, "libaom-av1", 30, "avif")
            else:
                if codec in {"jpeg", "webp"} and not features.check(
                        {"jpeg": "jpg", "webp": "webp"}[codec]):
                    raise RuntimeError(f"Pillow has no {codec} support")
                _encode_pillow(probe, codec, DEFAULT_LADDERS[codec][0])
            usable.append(codec)
        except Exception as error:
            print(f"  skipping {codec}: {type(error).__name__}: {error}", flush=True)
    return usable


def encode(image: Image.Image, codec: str, setting: int) -> tuple[bytes, Image.Image]:
    if codec == "avif":
        return _encode_ffmpeg(image, "libaom-av1", setting, "avif")
    return _encode_pillow(image, codec, setting)


def sweep(images: list[Image.Image], codec: str, ladder: list[int]) -> list[dict]:
    curve = []
    for setting in ladder:
        total_bytes, total_mse, pixels = 0, 0.0, 0
        for image in images:
            payload, decoded = encode(image, codec, setting)
            reference = np.asarray(image, dtype=np.float32) / 255.0
            candidate = np.asarray(decoded, dtype=np.float32) / 255.0
            if candidate.shape != reference.shape:
                raise RuntimeError(f"{codec} changed the image shape at setting {setting}")
            total_mse += float(((reference - candidate) ** 2).mean())
            total_bytes += len(payload)
            pixels = reference.shape[0] * reference.shape[1]
        bpp = total_bytes * 8 / (len(images) * pixels)
        curve.append({"setting": setting, "bpp": bpp,
                      "compression_ratio": 24.0 / bpp,
                      "psnr_db": psnr_from_mse(total_mse / len(images))})
        point = curve[-1]
        print(f"    {codec:9s} setting={setting:>4}  {point['bpp']:7.3f} bpp"
              f"  CR={point['compression_ratio']:7.2f}  PSNR={point['psnr_db']:6.2f} dB",
              flush=True)
    return sorted(curve, key=lambda point: point["bpp"])


def interpolate(curve: list[dict], bpp: float) -> float | None:
    """PSNR of a reference curve at ``bpp``, linear in (log bpp, dB)."""
    rates = [point["bpp"] for point in curve]
    if not curve or bpp < min(rates) or bpp > max(rates):
        return None
    return float(np.interp(math.log(bpp), [math.log(r) for r in rates],
                           [point["psnr_db"] for point in curve]))


#: (low, high) quality-control bounds and whether higher means larger files.
REFERENCE_BOUNDS = {"jpeg": (1, 100, True), "webp": (1, 100, True),
                    "jpeg2000": (1, 400, False), "avif": (0, 63, False)}


def match_rate(image: Image.Image, codec: str, target_bytes: int,
               steps: int = 12) -> tuple[Image.Image, int, int]:
    """Bisect the codec's quality control until its file size matches ``target_bytes``."""
    low, high, higher_is_larger = REFERENCE_BOUNDS[codec]
    best = None
    for _ in range(steps):
        setting = (low + high) // 2
        payload, decoded = encode(image, codec, setting)
        size = len(payload)
        if best is None or abs(size - target_bytes) < abs(best[1] - target_bytes):
            best = (decoded, size, setting)
        if (size < target_bytes) == higher_is_larger:
            low = setting + 1
        else:
            high = setting - 1
        if low > high:
            break
    return best


