#!/usr/bin/env python3
"""Profile the encode path: OpenVINO encoder -> CharLS -> bytes, warmup excluded.

Every number here is a steady-state median -- the second half of back-to-back
timed calls -- because this hardware's cold path lies: the NPU needs about
fifteen inferences before it settles and a CPU idles its clocks between frames.
Warmup is excluded by construction, not by a fixed-size burn-in guess.

The stages, in the order the deployment meets them::

    image (uint8, in memory) -> encoder graph -> uint8 planes
                             -> CharLS        -> bitstream bytes

The tool answers three questions in one run. How fast is each stage, and the
path end to end? What does the path cost in size -- the compression ratio
against raw RGB (the paper's 24 bpp convention) and against the source PNG the
images are stored as? And when an ``--int8-encoder`` is given, what does
activation quantization do to the latency, the rate and the reconstruction --
measured interleaved with the fp32 encoder, so clock drift lands on both sides
of the ratio instead of all on one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from src.compressors.frappe.harness import (
    AnonymousImageFolder,
    BitstreamConvention,
    decode_planes,
    encode_planes,
)
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.metrics import psnr_from_mse
from src.compressors.frappe.harness.profiling import (
    RAW_BITS_PER_PIXEL,
    as_torch_planes,
    series,
    steady_median,
)
from src.compressors.frappe.harness.reporting import Table, write_report
from src.compressors.frappe.openvino_runtime import (
    FrappeDecoder,
    FrappeEncoder,
    bit_exact_properties,
    testbed,
)


@dataclass
class VariantStats:
    """Accumulated per-variant measurements, aggregated at the end."""

    encoder_ms: list[float] = field(default_factory=list)
    charls_ms: list[float] = field(default_factory=list)
    end_to_end_ms: list[float] = field(default_factory=list)
    bytes: int = 0
    drift: int = 0
    symbols: int = 0
    squared_error: float = 0.0
    error_samples: int = 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx-stem", type=Path, required=True, help="exported graph pair stem")
    parser.add_argument(
        "--int8-encoder",
        type=Path,
        default=None,
        help="an int8 encoder graph (see tools/quantize_encoder_int8.py); "
        "when given, it is profiled against the fp32 encoder",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--images", type=int, default=0, help="how many images to measure; 0 means the whole split"
    )
    parser.add_argument(
        "--iterations", type=int, default=30, help="back-to-back timed calls per stage and image"
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import openvino

    core = openvino.Core()
    folder = AnonymousImageFolder(args.dataset_root, args.split)
    count = min(args.images, len(folder)) if args.images else len(folder)
    if count == 0:
        raise SystemExit(f"the {args.split!r} split is empty")
    _b, _c, height, width = folder.pixels(0).shape
    pixels = height * width
    fp32 = FrappeEncoder(
        args.onnx_stem,
        "CPU",
        height,
        width,
        core=core,
        properties=bit_exact_properties("CPU", core),
    )
    int8 = (
        FrappeEncoder(args.int8_encoder, "CPU", height, width, core=core)
        if args.int8_encoder
        else None
    )
    # One shared fp32 decoder, so both variants' planes are judged by the same
    # synthesis path -- a deployment would not ship two decoders.
    decoder = FrappeDecoder(args.onnx_stem, "CPU", fp32.plane_shapes, core=core)

    stats = {"fp32": VariantStats(), "int8": VariantStats() if int8 else None}

    for index in range(count):
        image = folder.pixels(index).numpy()
        fp_planes = fp32(image)
        fp_tensors = as_torch_planes(fp_planes)

        # CharLS is shared by both variants; time it once, on the fp32 planes.
        charls = steady_median(
            series(
                lambda t=fp_tensors: encode_planes(t, BitstreamConvention.PAYLOAD_ONLY),
                args.iterations,
            )
        )

        # The two encoders run interleaved on the same image, so clock drift
        # lands on both sides of the ratio instead of all on one.
        fp_times, int8_times = [], []
        for _ in range(args.iterations):
            started = time.perf_counter()
            fp32(image)
            fp_times.append((time.perf_counter() - started) * 1000.0)
            if int8 is not None:
                started = time.perf_counter()
                int8(image)
                int8_times.append((time.perf_counter() - started) * 1000.0)

        for name, planes in (("fp32", fp_planes), ("int8", None if int8 is None else int8(image))):
            if planes is None:
                continue
            entry = stats[name]
            tensors = as_torch_planes(planes)
            entry.encoder_ms.append(steady_median(fp_times if name == "fp32" else int8_times))
            entry.charls_ms.append(charls)
            entry.end_to_end_ms.append(
                steady_median(
                    series(
                        lambda a=image, e=fp32 if name == "fp32" else int8: encode_planes(
                            as_torch_planes(e(a)), BitstreamConvention.PAYLOAD_ONLY
                        ),
                        args.iterations,
                    )
                )
            )
            entry.bytes += len(encode_planes(tensors, BitstreamConvention.PAYLOAD_ONLY))
            entry.symbols += sum(plane.size for plane in planes)
            blob = encode_planes(tensors, BitstreamConvention.WITH_LENGTH_PREFIX)
            restored = [plane.numpy() for plane in decode_planes(blob)]
            reconstruction = decoder(restored)[0].astype(np.float64)
            difference = (image[0].astype(np.float64) - reconstruction) / 255.0
            entry.squared_error += float((difference**2).sum())
            entry.error_samples += difference.size

        if int8 is not None:
            int8_planes = int8(image)
            stats["int8"].drift += sum(
                int((want != got).sum()) for want, got in zip(fp_planes, int8_planes)
            )

    med = statistics.median
    png_bytes = sum(path.stat().st_size for path in folder.files[:count])
    rows, report_variants = [], {}
    for name, entry in stats.items():
        if entry is None:
            continue
        encode, charls, e2e = med(entry.encoder_ms), med(entry.charls_ms), med(entry.end_to_end_ms)
        bpp = entry.bytes * 8 / (count * pixels)
        psnr = psnr_from_mse(entry.squared_error / entry.error_samples)
        report_variants[name] = {
            "encoder_ms": encode,
            "charls_ms": charls,
            "end_to_end_ms": e2e,
            "bytes_per_image": entry.bytes / count,
            "bits_per_sample": bpp,
            "compression_ratio_raw": RAW_BITS_PER_PIXEL / bpp,
            "compression_ratio_png": (png_bytes / count) / (entry.bytes / count),
            "psnr_db": psnr,
            "plane_drift_symbols": entry.drift if int8 else None,
            "plane_symbols": entry.symbols,
        }
        rows.append(
            {
                "variant": name,
                "encoder": f"{encode:.2f}",
                "charls": f"{charls:.2f}",
                "end_to_end": f"{e2e:.2f}",
                "B_per_image": f"{entry.bytes / count:.0f}",
                "bpp": f"{bpp:.3f}",
                "CR_raw": f"{RAW_BITS_PER_PIXEL / bpp:.2f}",
                "CR_png": f"{(png_bytes / count) / (entry.bytes / count):.2f}",
                "psnr": f"{psnr:.3f}",
                "drift": f"{entry.drift:,}" if int8 else "-",
            }
        )

    print(
        f"{count} image(s), {width}x{height}, {args.iterations} timed calls per stage "
        f"(steady = median over the second half; warmup excluded)"
    )
    print(
        Table(
            [
                ("variant", "variant", ""),
                ("encoder", "encoder", ">8"),
                ("charls", "charls", ">8"),
                ("end_to_end", "end_to_end", ">10"),
                ("B/img", "B_per_image", ">8"),
                ("bpp", "bpp", ">7"),
                ("CR raw", "CR_raw", ">7"),
                ("CR png", "CR_png", ">7"),
                ("PSNR", "psnr", ">8"),
                ("drift", "drift", ">7"),
            ]
        ).render(rows)
    )

    if int8 is not None:
        f, q = report_variants["fp32"], report_variants["int8"]
        print(
            f"\n  fp32 -> int8: encoder {f['encoder_ms'] / q['encoder_ms']:.2f}x, "
            f"end-to-end {f['end_to_end_ms'] / q['end_to_end_ms']:.2f}x, "
            f"rate {(q['bytes_per_image'] / f['bytes_per_image'] - 1) * 100:+.1f}%, "
            f"PSNR {q['psnr_db'] - f['psnr_db']:+.3f} dB, "
            f"drift {q['plane_drift_symbols']:,}/{q['plane_symbols']:,} symbols"
        )

    write_report(
        {
            "onnx_stem": str(args.onnx_stem),
            "int8_encoder": str(args.int8_encoder) if args.int8_encoder else None,
            "split": args.split,
            "images": count,
            "static_shape": [width, height],
            "iterations": args.iterations,
            "warmup_policy": "steady median of the second half",
            "variants": report_variants,
            "source_png_bytes_per_image": png_bytes / count,
            "testbed": testbed(core),
        },
        args.report,
    )


if __name__ == "__main__":
    main()
