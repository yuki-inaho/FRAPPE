#!/usr/bin/env python3
"""Rate-distortion reference curves for standard codecs on a local anonymous split.

FRAPPE numbers only mean something next to what an ordinary codec spends for the
same quality on the same images.  This sweeps the codecs available in the
environment over their quality controls, measures the real encoded file size and
the decoded PSNR, and writes a comparable rate-distortion curve.

PSNR uses the same convention as the FRAPPE training and evaluation scripts:
mean squared error on [0, 1] RGB, averaged over images after per-image PSNR is
*not* taken -- the aggregate MSE is converted once, so the number matches
``tools/evaluate_joint_prefix.py``.

Codecs are probed at start-up and silently skipped when unavailable, so the tool
runs unchanged on machines with a different Pillow build.  Pass
``--frappe-report`` to have the tool interpolate each reference curve at the
FRAPPE operating points and print the dB difference at matched rate.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

#: Default quality ladders. JPEG 2000 is driven by compression ratio, which lets
#: a target such as "compression ratio 10" be requested exactly.
DEFAULT_LADDERS = {
    "jpeg": [50, 70, 80, 85, 90, 93, 95, 97, 98],
    "webp": [50, 70, 80, 85, 90, 93, 95, 98, 100],
    "jpeg2000": [40, 30, 20, 15, 12, 10, 8, 6, 4, 3],
    "avif": [50, 42, 36, 30, 24, 18, 12, 8],
}


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


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
        except Exception as error:  # noqa: BLE001 -- probing is best-effort by design
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/workspace/data/frappe_rgb_800x608/imagefolder"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--codecs", nargs="+", default=list(DEFAULT_LADDERS),
                        choices=list(DEFAULT_LADDERS))
    parser.add_argument("--settings", type=int, nargs="+", default=None,
                        help="override the quality ladder for every selected codec")
    parser.add_argument("--ladder", action="append", default=[], metavar="CODEC=V1,V2,...",
                        help="override one codec's ladder, e.g. --ladder jpeg=5,10,20 "
                             "(repeatable; takes precedence over --settings)")
    parser.add_argument("--frappe-report", type=Path, default=None,
                        help="evaluation JSON from tools/evaluate_joint_prefix.py to compare against")
    parser.add_argument("--frappe-split", default="validation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    files = sorted((args.dataset_root / args.split).glob("image_????????.png"))[:args.images]
    if not files:
        raise SystemExit(f"no anonymous PNG images under {args.dataset_root / args.split}")
    images = []
    for path in files:
        with Image.open(path) as handle:
            handle.load()
            images.append(handle.convert("RGB"))
    print(f"{len(images)} images from {args.split}, {images[0].size[0]}x{images[0].size[1]}")

    overrides = {}
    for entry in args.ladder:
        codec, _, values = entry.partition("=")
        if codec not in DEFAULT_LADDERS or not values:
            raise SystemExit(f"--ladder expects CODEC=V1,V2,... with CODEC in "
                             f"{sorted(DEFAULT_LADDERS)}; got {entry!r}")
        overrides[codec] = [int(value) for value in values.split(",")]

    started = time.time()
    codecs = available_codecs(args.codecs)
    curves = {}
    for codec in codecs:
        print(f"  {codec}:", flush=True)
        ladder = overrides.get(codec) or args.settings or DEFAULT_LADDERS[codec]
        curves[codec] = sweep(images, codec, ladder)

    report = {"split": args.split, "images": len(images), "curves": curves,
              "seconds": time.time() - started}

    if args.frappe_report and args.frappe_report.is_file():
        evaluation = json.loads(args.frappe_report.read_text(encoding="utf-8"))
        points = evaluation["splits"][args.frappe_split]["curve"]
        comparison = []
        print(f"\n  FRAPPE operating points against the reference curves "
              f"(dB at matched rate):")
        header = "  ".join(f"{codec:>9}" for codec in codecs)
        print(f"    {'n':>3} {'bpp':>8} {'FRAPPE':>8}  {header}")
        for point in points:
            row = {"channels": point["channels"], "bpp": point["bpp"],
                   "frappe_psnr_db": point["psnr_db"], "reference": {}}
            cells = []
            for codec in codecs:
                value = interpolate(curves[codec], point["bpp"])
                row["reference"][codec] = value
                cells.append("      n/a" if value is None
                             else f"{point['psnr_db'] - value:+9.2f}")
            comparison.append(row)
            print(f"    {point['channels']:>3} {point['bpp']:8.3f} "
                  f"{point['psnr_db']:8.2f}  " + "  ".join(cells))
        print("    (positive means FRAPPE is ahead of that codec at the same bitrate)")
        report["frappe_comparison"] = comparison

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
