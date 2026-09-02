#!/usr/bin/env python3
"""Compare the deployed codec against the standard ones, at the same resolution.

The paper's evaluation question, asked about this machine's deployment: where
do the operating points land on the rate-distortion plane, next to JPEG, WebP,
JPEG 2000 and AVIF measured on the same images with the same PSNR convention
(aggregate MSE on [0, 1] RGB, converted once -- the repository's convention)?

The FRAPPE side is the real deployment path: the exported OpenVINO encoder
graph (fp32 or int8 -- ``--encoder-graph`` takes either), CharLS entropy
coding, and the decoder graph for the reconstruction. Reference codecs run
through :mod:`harness.codecs`, which encodes real files and measures their
size, so every number in the table is a byte count or a reconstruction, never
an estimate.

Latency is reported next to the RD points because the deployment question is
usually "can I afford this": FRAPPE's row carries its steady-state end-to-end
encode latency, and the FRAPPE column next to each reference codec's PSNR
reads "+X.XX" -- the dB FRAPPE is ahead at that reference codec's nearest
measured rate. The two sides are not the same contract -- a reference codec is
a standalone file format, FRAPPE is a three-stage pipeline -- so the table
also reports FRAPPE's stage latencies separately in the JSON.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from src.compressors.frappe.harness import (
    AnonymousImageFolder,
    BitstreamConvention,
    decode_planes,
    encode_planes,
)
from src.compressors.frappe.harness import codecs as reference_codecs
from src.compressors.frappe.harness.cli import add_output_argument
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.metrics import psnr_from_mse
from src.compressors.frappe.harness.reporting import Table, write_report
from src.compressors.frappe.openvino_runtime import FrappeDecoder, FrappeEncoder, testbed

RAW_BITS_PER_PIXEL = 24.0

#: The quality ladders swept for the reference codecs, chosen so each curve
#: brackets FRAPPE's operating points from both sides at this resolution.
LADDERS = {
    "jpeg": [70, 80, 85, 90, 93, 95],
    "webp": [70, 80, 85, 90, 93],
    "jpeg2000": [10, 15, 20, 30, 40],
    "avif": [42, 36, 30, 24, 18],
}


def steady_median(times):
    return statistics.median(times[len(times) // 2 :])


def series(action, iterations):
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        action()
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx-stem",
        type=Path,
        required=True,
        help="exported graph pair (decoder; encoder unless --encoder-graph)",
    )
    parser.add_argument(
        "--encoder-graph",
        type=Path,
        default=None,
        help="an encoder graph other than the stem's own -- an int8 "
        "encoder, for instance; the planes, and so the rate, "
        "follow this graph",
    )
    parser.add_argument("--codecs", nargs="+", default=list(LADDERS), choices=list(LADDERS))
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--images", type=int, default=0, help="how many images to evaluate; 0 means the whole split"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="back-to-back timed calls for FRAPPE's stage latencies",
    )
    add_output_argument(parser, help_text="write the comparison here as JSON")
    return parser.parse_args(argv)


def as_torch_planes(planes):
    return [
        torch.from_numpy(np.ascontiguousarray(plane[0] if plane.ndim == 3 else plane))
        for plane in planes
    ]


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

    encoder_path = args.encoder_graph or Path(f"{args.onnx_stem}_encoder.onnx")
    encoder = FrappeEncoder(encoder_path, "CPU", height, width, core=core)
    decoder = FrappeDecoder(args.onnx_stem, "CPU", encoder.plane_shapes, core=core)

    # FRAPPE: rate, PSNR and steady stage latencies over the split.
    stage_encode, stage_charls, stage_all = [], [], []
    squared_error, error_samples, total_bytes = 0.0, 0, 0
    for index in range(count):
        image = folder.pixels(index).numpy()
        planes = encoder(image)
        tensors = as_torch_planes(planes)

        stage_encode.append(steady_median(series(lambda a=image: encoder(a), args.iterations)))
        stage_charls.append(
            steady_median(
                series(
                    lambda t=tensors: encode_planes(t, BitstreamConvention.PAYLOAD_ONLY),
                    args.iterations,
                )
            )
        )

        blob = encode_planes(tensors, BitstreamConvention.WITH_LENGTH_PREFIX)
        restored = [plane.numpy() for plane in decode_planes(blob)]
        reconstruction = decoder(restored)
        difference = (
            folder.pixels(index).to(torch.float64)
            - torch.from_numpy(reconstruction).to(torch.float64)
        ) / 255.0
        squared_error += float((difference**2).sum())
        error_samples += difference.numel()
        total_bytes += len(encode_planes(tensors, BitstreamConvention.PAYLOAD_ONLY))

        def full(a=image):
            fresh = encoder(a)
            encode_planes(as_torch_planes(fresh), BitstreamConvention.PAYLOAD_ONLY)

        stage_all.append(steady_median(series(full, args.iterations)))

    bpp = total_bytes * 8 / (count * pixels)
    frappe = {
        "name": "FRAPPE (this deployment)",
        "bytes_per_image": total_bytes / count,
        "bpp": bpp,
        "compression_ratio": RAW_BITS_PER_PIXEL / bpp,
        "psnr_db": psnr_from_mse(squared_error / error_samples),
        "encoder_ms": statistics.median(stage_encode),
        "charls_ms": statistics.median(stage_charls),
        "end_to_end_ms": statistics.median(stage_all),
    }

    # The reference codecs, encoded and decoded for real, on the same images.
    probe_images = [folder.pil(index) for index in range(count)]
    usable = reference_codecs.available_codecs(args.codecs)
    curves = {}
    started = time.time()
    for codec in usable:
        print(f"  sweeping {codec} ...", flush=True)
        curves[codec] = reference_codecs.sweep(probe_images, codec, LADDERS[codec])
    print(f"  reference sweep took {time.time() - started:.0f} s")

    # One table: each reference codec at the ladder point nearest FRAPPE's
    # rate, with the dB gap to FRAPPE at FRAPPE's own rate interpolated in.
    rows = [
        {
            "codec": "FRAPPE (deployed)",
            "bpp": f"{frappe['bpp']:.3f}",
            "PSNR": f"{frappe['psnr_db']:.2f}",
            "CR": f"{frappe['compression_ratio']:.1f}",
            "enc ms": f"{frappe['end_to_end_ms']:.2f}",
        }
    ]
    for codec, curve in curves.items():
        nearest = min(curve, key=lambda point: abs(point["bpp"] - frappe["bpp"]))
        rows.append(
            {
                "codec": codec,
                "bpp": f"{nearest['bpp']:.3f}",
                "PSNR": f"{nearest['psnr_db']:.2f}"
                f" (FRAPPE {frappe['psnr_db'] - nearest['psnr_db']:+.2f})",
                "CR": f"{nearest['compression_ratio']:.1f}",
                "enc ms": "sweep",
            }
        )
    print(
        f"\n{count} image(s), {width}x{height} -- FRAPPE {frappe['bpp']:.3f} bpp, "
        f"PSNR {frappe['psnr_db']:.2f} dB, {frappe['end_to_end_ms']:.2f} ms end to end"
    )
    print(
        Table(
            [
                ("codec", "codec", ""),
                ("bpp", "bpp", ">7"),
                ("PSNR", "PSNR", ">8"),
                ("CR", "CR", ">6"),
                ("enc ms", "enc_ms", ">7"),
            ]
        ).render(rows)
    )

    write_report(
        {
            "onnx_stem": str(args.onnx_stem),
            "encoder_graph": str(encoder_path),
            "split": args.split,
            "images": count,
            "static_shape": [width, height],
            "frappe": frappe,
            "reference_curves": curves,
            "reference_ladders": LADDERS,
            "iterations": args.iterations,
            "testbed": testbed(core),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
