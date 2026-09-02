#!/usr/bin/env python3
"""Run the whole FRAPPE codec on this machine's accelerators: image in, image out.

The round trip is the deployment, so this tool runs all of it and measures each
stage separately::

    image (uint8)
      -> encoder graph on --encoder-device   -> uint8 JPEG-LS planes
      -> JPEG-LS on the CPU                  -> bitstream
      -> JPEG-LS on the CPU                  -> the same planes, checked
      -> decoder graph on --decoder-device   -> reconstruction (uint8)

Encoder and decoder take separate devices because they have opposite cost
profiles: the analysis path is a handful of strided convolutions and the
synthesis path carries essentially all of the model's parameters and FLOPs, so
the placement that is best for one is not obviously best for the other. Naming
them separately is what lets a measurement answer that rather than a preference.

Devices are named, never inferred. OpenVINO's ``AUTO`` is refused, and a device
that cannot run is an error rather than a silent move to one that can -- a codec
whose encoder quietly migrated is a codec whose measurements describe something
other than what was deployed. ``--prefer`` states an explicit fallback order and
the report records which device won and why the earlier ones did not.

With ``--reference-checkpoint`` the planes are also checked against the PyTorch
model the graph came from, with the same zero-tolerance gate
``tools/export_onnx.py`` applies: the planes are the bitstream, so a device that
changes one symbol has changed the file. Only devices advertising ``FP32`` can
pass that gate; see ``openvino_runtime.bit_exact_properties``.

Private data never reaches this tool's output. It addresses images by split and
integer index, and the report records the split and the index, never a path.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import (
    AnonymousImageFolder,
    BitstreamConvention,
    arrange_planes,
    decode_planes,
    encode_planes,
    psnr_from_mse,
)
from src.compressors.frappe.harness.checkpoints import load_checkpoint
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.reporting import Table, write_report
from src.compressors.frappe.openvino_runtime import (
    FrappeDecoder,
    FrappeEncoder,
    bit_exact_properties,
    select_device,
    testbed,
)

#: 8 bits per channel, three channels: the raw rate a compression ratio is against.
RAW_BITS_PER_PIXEL = 24.0


def as_torch_planes(planes) -> list[torch.Tensor]:
    """OpenVINO's ``(1, rows, cols)`` uint8 arrays as the harness's 2D tensors."""
    return [torch.from_numpy(np.ascontiguousarray(plane[0] if plane.ndim == 3
                                                  else plane)) for plane in planes]


def rate_of(planes, pixels: int) -> dict:
    """Both rate conventions for one image, so neither can be quoted unlabelled.

    ``encode_latents`` prefixes each scale's stream with four bytes and the
    reporting tools sum bare payloads; the gap is 3.3e-4 bpp at 800x608. The
    headline follows the bare-payload convention, matching the released
    measurements, and the other number travels with it.
    """
    tensors = as_torch_planes(planes)
    bare = len(encode_planes(tensors, BitstreamConvention.PAYLOAD_ONLY))
    prefixed = bare + BitstreamConvention.WITH_LENGTH_PREFIX.overhead_bytes(len(tensors))
    return {"bytes": bare, "bytes_with_length_prefix": prefixed,
            "bpp": bare * 8 / pixels, "bpp_with_length_prefix": prefixed * 8 / pixels}


def reference_planes(checkpoint: Path, image: torch.Tensor) -> list[torch.Tensor]:
    """What the PyTorch model says the bitstream is, in the graph's own form."""
    loaded = load_checkpoint(checkpoint, "cpu")
    x = image.to(torch.float32) / 127.5 - 1.0
    with torch.no_grad():
        return arrange_planes(loaded.model.integer_codes(x))


def timed(call, warmup: int, repeats: int):
    """Median of ``repeats`` timed calls after ``warmup`` untimed ones."""
    result = call()
    for _ in range(max(0, warmup - 1)):
        result = call()
    samples = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return result, statistics.median(samples)


def resolve(requested: str, prefer, core) -> tuple[str, list]:
    """A concrete device, plus the record of anything that was skipped."""
    if requested.lower() == "prefer":
        if not prefer:
            raise SystemExit("--prefer needs at least one device when a device is 'prefer'")
        choice = select_device(prefer, core)
    else:
        choice = select_device([requested], core)
    return choice.device, [{"device": device, "reason": reason}
                           for device, reason in choice.considered]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-stem", type=Path, required=True,
                        help="path stem of the exported pair, without _encoder.onnx")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0,
                        help="first image index in the split; the report records this, "
                             "never a filename")
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--encoder-device", default="CPU",
                        help="a concrete OpenVINO device, or 'prefer' to use --prefer")
    parser.add_argument("--decoder-device", default="CPU",
                        help="a concrete OpenVINO device, or 'prefer' to use --prefer")
    parser.add_argument("--prefer", nargs="+", default=None)
    parser.add_argument("--bit-exact", action=argparse.BooleanOptionalAction, default=False,
                        help="ask the encoder device for the settings that reproduce the "
                             "reference codes; fails on a device that cannot")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the reconstruction of the first image here")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    import openvino

    core = openvino.Core()
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(args.cache_dir)})

    encoder_device, encoder_skipped = resolve(args.encoder_device, args.prefer, core)
    decoder_device, decoder_skipped = resolve(args.decoder_device, args.prefer, core)
    encoder_properties = bit_exact_properties(encoder_device, core) if args.bit_exact else {}

    folder = AnonymousImageFolder(args.dataset_root, args.split)
    indices = [args.index + offset for offset in range(args.images)]
    if indices[-1] >= len(folder):
        raise SystemExit(f"{args.split!r} has {len(folder)} images; "
                         f"indices {indices[0]}..{indices[-1]} do not fit")

    first = folder.pixels(indices[0])
    _batch, channels, height, width = first.shape
    pixels = height * width
    started_all = time.perf_counter()
    encoder = FrappeEncoder(args.onnx_stem, encoder_device, height, width,
                            core=core, properties=encoder_properties)
    decoder = FrappeDecoder(args.onnx_stem, decoder_device, encoder.plane_shapes, core=core)
    compile_seconds = time.perf_counter() - started_all
    print(f"pinned to {width}x{height}")
    print(f"  encoder {encoder.device} -> ran on {encoder.execution_devices}"
          f"{'  [bit-exact settings]' if args.bit_exact else ''}")
    print(f"  decoder {decoder.device} -> ran on {decoder.execution_devices}")
    print(f"  planes  {list(zip(encoder.plane_names, encoder.plane_shapes))}")

    rows, encode_ms, jpegls_ms, decode_ms = [], [], [], []
    pooled_error, pooled_samples, pooled_bytes, pooled_prefixed = 0.0, 0, 0, 0
    mismatched_total = 0
    for index in indices:
        image = first if index == indices[0] else folder.pixels(index)
        array = image.numpy()
        planes, encode_median = timed(lambda a=array: encoder(a),
                                      args.warmup, args.repeats)

        if args.reference_checkpoint:
            reference = reference_planes(args.reference_checkpoint, image)
            candidate = as_torch_planes(planes)
            mismatched = sum(int((want != got).sum())
                             for want, got in zip(reference, candidate))
            mismatched_total += mismatched
            if mismatched:
                worst = max(int((want.to(torch.int32) - got.to(torch.int32)).abs().max())
                            for want, got in zip(reference, candidate))
                raise SystemExit(
                    f"the {encoder.device} encoder does not reproduce the reference "
                    f"bitstream: {mismatched} symbols differ, max |difference| {worst}. "
                    "Encode on a device that advertises FP32, or drop --reference-checkpoint "
                    "and accept the measured inexactness.")

        tensors = as_torch_planes(planes)
        blob, jpegls_median = timed(
            lambda t=tensors: encode_planes(t, BitstreamConvention.WITH_LENGTH_PREFIX),
            0, max(1, args.repeats // 2))
        restored = decode_planes(blob)
        for want, got in zip(tensors, restored):
            if not torch.equal(want, got):
                raise SystemExit("JPEG-LS did not round-trip the planes; at NEAR=0 it "
                                 "is lossless, so this is a defect and not a tolerance")

        reconstruction, decode_median = timed(
            lambda r=restored: decoder([plane.numpy() for plane in r]),
            args.warmup, args.repeats)
        rate = rate_of(planes, pixels)
        difference = (image.to(torch.float64) - torch.from_numpy(reconstruction)
                      .to(torch.float64)) / 255.0
        pooled_error += float((difference ** 2).sum())
        pooled_samples += difference.numel()
        pooled_bytes += rate["bytes"]
        pooled_prefixed += rate["bytes_with_length_prefix"]
        encode_ms.append(encode_median)
        jpegls_ms.append(jpegls_median)
        decode_ms.append(decode_median)
        rows.append({"index": index,
                     "psnr_db": psnr_from_mse(float((difference ** 2).mean())),
                     **rate,
                     "compression_ratio": RAW_BITS_PER_PIXEL / rate["bpp"],
                     "latency_ms": {"encode": encode_median, "jpegls": jpegls_median,
                                    "decode": decode_median}})
        if index == indices[0] and args.output:
            from PIL import Image

            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reconstruction[0].transpose(1, 2, 0)).save(args.output)
            print(f"  wrote {args.output}")

    pooled_pixels = pixels * len(indices)
    bpp = pooled_bytes * 8 / pooled_pixels
    bpp_prefixed = pooled_prefixed * 8 / pooled_pixels
    latency = {"encode_median": statistics.median(encode_ms),
               "jpegls_median": statistics.median(jpegls_ms),
               "decode_median": statistics.median(decode_ms),
               "compile_seconds": compile_seconds,
               "n_warmup": args.warmup, "n_measurement": args.repeats}
    latency["total_median"] = (latency["encode_median"] + latency["jpegls_median"]
                               + latency["decode_median"])

    table = Table(["stage", "device", "ms", "Mpixel/s"])
    for label, device, value in (("encode", encoder.device, latency["encode_median"]),
                                 ("JPEG-LS", "CPU", latency["jpegls_median"]),
                                 ("decode", decoder.device, latency["decode_median"]),
                                 ("total", "-", latency["total_median"])):
        table.add(label, device, f"{value:.2f}", f"{pixels / value / 1000:.1f}")
    print()
    table.render()
    psnr = psnr_from_mse(pooled_error / pooled_samples)
    print(f"\n  {len(indices)} image(s)   PSNR {psnr:.3f} dB"
          f"   {bpp:.5f} bpp   CR {RAW_BITS_PER_PIXEL / bpp:.3f}   [payload only]")
    print(f"{'':22s}{bpp_prefixed:.5f} bpp   CR {RAW_BITS_PER_PIXEL / bpp_prefixed:.3f}"
          "   [length prefixes included]")

    write_report({
        "onnx_stem": str(args.onnx_stem),
        "split": args.split, "indices": indices, "images": len(indices),
        "static_shape": [width, height], "channels": channels,
        "encoder": {"device": encoder.device, "properties": encoder_properties,
                    "execution_devices": encoder.execution_devices,
                    "skipped": encoder_skipped},
        "decoder": {"device": decoder.device,
                    "execution_devices": decoder.execution_devices,
                    "skipped": decoder_skipped},
        "rate_convention": "payload_only",
        "psnr_db": psnr,
        "psnr_mean_of_images_db": float(np.mean([row["psnr_db"] for row in rows])),
        "bpp": bpp, "bpp_with_length_prefix": bpp_prefixed,
        "compression_ratio": RAW_BITS_PER_PIXEL / bpp,
        "compression_ratio_with_length_prefix": RAW_BITS_PER_PIXEL / bpp_prefixed,
        "bytes": pooled_bytes,
        "verification": {
            "jpegls_roundtrip_exact": True,
            "reference_checkpoint": (str(args.reference_checkpoint)
                                     if args.reference_checkpoint else None),
            "plane_mismatched_symbols": (mismatched_total if args.reference_checkpoint
                                         else None),
        },
        "latency_ms": latency, "per_image": rows,
        "testbed": testbed(core), "seconds": time.perf_counter() - started_all,
    }, args.report)


if __name__ == "__main__":
    main()
