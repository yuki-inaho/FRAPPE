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
them separately is what lets the benchmark answer that rather than assume it.

Devices are named, never inferred. OpenVINO's ``AUTO`` is refused, and a device
that cannot run is an error rather than a silent move to one that can -- a codec
whose encoder quietly migrated is a codec whose measurements describe something
other than what you deployed. ``--prefer`` states an explicit fallback order and
the report records which device won and why the earlier ones did not.

Two rates are reported for every measurement. ``entropy_coding.encode_latents``
prefixes each scale's stream with a four-byte length; the reporting tools in this
repository sum the bare payloads. The gap is 3.29e-4 bpp at 800x608 -- small, but
enough that quoting one against a table built from the other is wrong. The
headline numbers here use the bare-payload convention, matching
``tools/evaluate_joint_prefix.py`` and the released measurements.

With ``--reference-checkpoint`` the planes are also checked against the PyTorch
model that produced the graph, with the same zero-tolerance gate
``tools/export_onnx.py`` applies: the planes are the bitstream, so a device that
changes one symbol has changed the file. That check needs torch; without it the
tool runs on a machine that only has OpenVINO.

Private data never reaches this tool's output. It addresses images by split and
integer index, and the report records the split and the index, never a path or a
filename.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.bitstream import (  # noqa: E402
    decode_planes, encode_planes, measure)
from src.compressors.frappe.openvino_runtime import (  # noqa: E402
    FrappeDecoder, FrappeEncoder, available_devices, select_device)

#: 8 bits per channel, three channels: the raw rate a compression ratio is against.
RAW_BITS_PER_PIXEL = 24.0


def load_image(root: Path, split: str, index: int) -> np.ndarray:
    """One anonymous image as ``(1, 3, H, W)`` uint8, addressed by index.

    The filename is deliberately not returned. The dataset is private, and an
    index is all the report needs to be reproducible.
    """
    from PIL import Image

    files = sorted((root / split).glob("image_????????.png"))
    if not files:
        raise SystemExit(f"no anonymous PNG images under {root / split}")
    if not 0 <= index < len(files):
        raise SystemExit(f"index {index} is outside 0..{len(files) - 1} for {split!r}")
    with Image.open(files[index]) as handle:
        handle.load()
        array = np.array(handle.convert("RGB"), dtype=np.uint8)
    return array.transpose(2, 0, 1)[None]


def count_images(root: Path, split: str) -> int:
    return len(sorted((root / split).glob("image_????????.png")))


def squared_error(original: np.ndarray, reconstruction: np.ndarray) -> tuple[float, int]:
    """Summed squared error on ``[0, 1]`` intensities, and the sample count.

    Kept as a sum rather than a mean so a set of images can be pooled into one
    MSE. That is the convention ``tools/evaluate_joint_prefix.py`` reports and the
    one every cross-codec table in this repository uses; averaging per-image PSNRs
    instead gives a different number, so both are reported and labelled.
    """
    difference = (original.astype(np.float64) - reconstruction.astype(np.float64)) / 255.0
    return float((difference ** 2).sum()), int(difference.size)


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0 else -10.0 * np.log10(mse)


def testbed(core=None) -> dict:
    """What the numbers were measured on. Reproducibility needs this, not a path."""
    import openvino

    resolved = core or openvino.Core()
    devices = {}
    for device in available_devices(resolved):
        try:
            devices[device] = resolved.get_property(device, "FULL_DEVICE_NAME")
        except Exception as error:  # noqa: BLE001 -- report, do not fail a run over it
            devices[device] = f"<unavailable: {type(error).__name__}>"
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "openvino_version": openvino.__version__,
        "available_devices": devices,
    }


def resolve(requested: str, prefer: list[str] | None, core=None) -> tuple[str, list]:
    """A concrete device, plus the record of anything that was skipped."""
    if requested.lower() != "prefer":
        choice = select_device([requested], core)
    else:
        if not prefer:
            raise SystemExit("--prefer needs at least one device when a device is 'prefer'")
        choice = select_device(prefer, core)
    return choice.device, [{"device": device, "reason": reason}
                           for device, reason in choice.considered]


def reference_planes(checkpoint: Path, image: np.ndarray) -> list[np.ndarray]:
    """What the PyTorch model says the bitstream is, in the graph's own uint8 form."""
    import torch

    from src.compressors.frappe.entropy_coding import arrange_latents
    from tools.evaluate_joint_prefix import load_checkpoint

    model, _config, _state = load_checkpoint(checkpoint, "cpu")
    x = torch.from_numpy(image).to(torch.float32) / 127.5 - 1.0
    with torch.no_grad():
        arranged = arrange_latents(model.integer_codes(x))
    return [(plane.to(torch.int16) + 127).to(torch.uint8).numpy() for plane in arranged]


def timed(call, warmup: int, repeats: int):
    """Median of ``repeats`` timed calls after ``warmup`` untimed ones."""
    for _ in range(warmup):
        result = call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return result, statistics.median(samples), samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-stem", type=Path, required=True,
                        help="path stem of the exported pair, without _encoder.onnx")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0,
                        help="first image index in the split; the report records this, "
                             "never a filename")
    parser.add_argument("--images", type=int, default=1,
                        help="how many consecutive images to run from --index")
    parser.add_argument("--encoder-device", default="CPU",
                        help="a concrete OpenVINO device, or 'prefer' to use --prefer")
    parser.add_argument("--decoder-device", default="CPU",
                        help="a concrete OpenVINO device, or 'prefer' to use --prefer")
    parser.add_argument("--prefer", nargs="+", default=None,
                        help="explicit fallback order used when a device is 'prefer'; "
                             "the report records what was skipped and why")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="OpenVINO compiled-model cache; speeds up repeat runs")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-checkpoint", type=Path, default=None,
                        help="gate the planes against this PyTorch checkpoint; needs torch")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the reconstruction of the first image here")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import openvino

    core = openvino.Core()
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(args.cache_dir)})

    encoder_device, encoder_skipped = resolve(args.encoder_device, args.prefer, core)
    decoder_device, decoder_skipped = resolve(args.decoder_device, args.prefer, core)

    available = count_images(args.dataset_root, args.split)
    if not available:
        raise SystemExit(f"no anonymous PNG images under {args.dataset_root / args.split}")
    indices = [args.index + offset for offset in range(args.images)]
    if indices[-1] >= available:
        raise SystemExit(f"{args.split!r} has {available} images; "
                         f"indices {indices[0]}..{indices[-1]} do not fit")

    first = load_image(args.dataset_root, args.split, indices[0])
    _batch, channels, height, width = first.shape
    print(f"pinning graphs to {width}x{height}")
    started_all = time.perf_counter()
    encoder = FrappeEncoder(args.onnx_stem, encoder_device, height, width, core=core)
    decoder = FrappeDecoder(args.onnx_stem, decoder_device, encoder.plane_shapes, core=core)
    compile_seconds = time.perf_counter() - started_all
    print(f"  encoder on {encoder.device} (ran on {encoder.execution_devices})")
    print(f"  decoder on {decoder.device} (ran on {decoder.execution_devices})")
    print(f"  planes: {list(zip(encoder.plane_names, encoder.plane_shapes))}")

    per_image, pooled_error, pooled_samples = [], 0.0, 0
    pooled_payload, pooled_prefixed, pooled_pixels = 0, 0, 0
    encode_ms, jpegls_ms, decode_ms = [], [], []
    reference = None
    if args.reference_checkpoint:
        print(f"  gating planes against {args.reference_checkpoint.name}")

    for index in indices:
        image = first if index == indices[0] else \
            load_image(args.dataset_root, args.split, index)
        planes, encode_median, _ = timed(lambda: encoder(image), args.warmup, args.repeats)

        if args.reference_checkpoint:
            reference = reference_planes(args.reference_checkpoint, image)
            mismatched = sum(int((want != got).sum())
                             for want, got in zip(reference, planes))
            if mismatched:
                worst = max(int(np.abs(want.astype(np.int32) - got.astype(np.int32)).max())
                            for want, got in zip(reference, planes))
                raise SystemExit(
                    f"the {encoder.device} encoder does not reproduce the reference "
                    f"bitstream: {mismatched} symbols differ, max |difference| {worst}")

        blob, jpegls_median, _ = timed(lambda: encode_planes(planes, length_prefix=True),
                                       0, max(1, args.repeats // 2) if args.repeats else 1)
        restored = decode_planes(blob)
        for original_plane, restored_plane in zip(planes, restored):
            if not np.array_equal(original_plane[0] if original_plane.ndim == 3
                                  else original_plane, restored_plane):
                raise SystemExit("JPEG-LS did not round-trip the code planes; the codec "
                                 "must be lossless at NEAR=0")

        reconstruction, decode_median, _ = timed(lambda: decoder(restored),
                                                 args.warmup, args.repeats)
        rates = measure(planes, height, width)
        error, samples = squared_error(image, reconstruction)
        pooled_error += error
        pooled_samples += samples
        pooled_payload += rates["bytes_payload_only"]
        pooled_prefixed += rates["bytes_with_length_prefix"]
        pooled_pixels += height * width
        encode_ms.append(encode_median)
        jpegls_ms.append(jpegls_median)
        decode_ms.append(decode_median)
        per_image.append({
            "index": index,
            "psnr_db": psnr_from_mse(error / samples),
            "bpp": rates["bpp_payload_only"],
            "bpp_with_length_prefix": rates["bpp_with_length_prefix"],
            "compression_ratio": rates["compression_ratio_payload_only"],
            "bytes": rates["bytes_payload_only"],
            "bytes_with_length_prefix": rates["bytes_with_length_prefix"],
            "payload_bytes": rates["payload_bytes"],
            "latency_ms": {"encode": encode_median, "jpegls": jpegls_median,
                           "decode": decode_median},
        })
        if index == indices[0] and args.output:
            from PIL import Image

            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reconstruction[0].transpose(1, 2, 0)).save(args.output)
            print(f"  wrote {args.output}")

    pooled_bpp = pooled_payload * 8 / pooled_pixels
    pooled_bpp_prefixed = pooled_prefixed * 8 / pooled_pixels
    report = {
        "onnx_stem": str(args.onnx_stem),
        "split": args.split,
        "indices": indices,
        "images": len(indices),
        "static_shape": [width, height],
        "channels": channels,
        "encoder": {"device": encoder.device,
                    "execution_devices": encoder.execution_devices,
                    "skipped": encoder_skipped},
        "decoder": {"device": decoder.device,
                    "execution_devices": decoder.execution_devices,
                    "skipped": decoder_skipped},
        "rate_convention": "payload_only",
        "rate_convention_note": (
            "bpp and compression_ratio exclude the 4-byte per-scale length prefix, "
            "matching tools/evaluate_joint_prefix.py and the released measurements; "
            "the prefixed values are reported alongside"),
        "psnr_db": psnr_from_mse(pooled_error / pooled_samples),
        "psnr_mean_of_images_db": float(np.mean([entry["psnr_db"] for entry in per_image])),
        "bpp": pooled_bpp,
        "bpp_with_length_prefix": pooled_bpp_prefixed,
        "compression_ratio": RAW_BITS_PER_PIXEL / pooled_bpp,
        "compression_ratio_with_length_prefix": RAW_BITS_PER_PIXEL / pooled_bpp_prefixed,
        "bytes": pooled_payload,
        "verification": {
            "jpegls_roundtrip_exact": True,
            "reference_checkpoint": (str(args.reference_checkpoint)
                                     if args.reference_checkpoint else None),
            "plane_mismatched_symbols": 0 if args.reference_checkpoint else None,
        },
        "latency_ms": {
            "encode_median": statistics.median(encode_ms),
            "jpegls_median": statistics.median(jpegls_ms),
            "decode_median": statistics.median(decode_ms),
            "total_median": (statistics.median(encode_ms) + statistics.median(jpegls_ms)
                             + statistics.median(decode_ms)),
            "compile_seconds": compile_seconds,
            "n_warmup": args.warmup,
            "n_measurement": args.repeats,
        },
        "per_image": per_image,
        "testbed": testbed(core),
        "seconds": time.perf_counter() - started_all,
    }

    print(f"\n  {len(indices)} image(s), {args.split} split")
    print(f"    PSNR            {report['psnr_db']:8.3f} dB   "
          f"(mean of per-image {report['psnr_mean_of_images_db']:.3f} dB)")
    print(f"    rate            {report['bpp']:8.5f} bpp  CR {report['compression_ratio']:.3f}"
          f"   [payload only]")
    print(f"                    {report['bpp_with_length_prefix']:8.5f} bpp  "
          f"CR {report['compression_ratio_with_length_prefix']:.3f}   [length prefixes included]")
    print(f"    encode          {report['latency_ms']['encode_median']:8.2f} ms  "
          f"on {encoder.device}")
    print(f"    JPEG-LS         {report['latency_ms']['jpegls_median']:8.2f} ms  on CPU")
    print(f"    decode          {report['latency_ms']['decode_median']:8.2f} ms  "
          f"on {decoder.device}")
    print(f"    total           {report['latency_ms']['total_median']:8.2f} ms")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
