#!/usr/bin/env python3
"""Batch sweep for the deployed graphs: does batching pay, and does it change anything?

An image codec encodes one image at a time, but a deployment may hold several
frames in flight, so the batch axis matters for throughput even when it does
not for latency. This tool sweeps the batch over ``--batches`` on each device,
reports the steady per-image latency and throughput of both graphs, and checks
the property every batched deployment silently assumes: that a batch-N encoder
run produces, image by image, the same planes as a batch-one run. Devices that
refuse a batched graph are recorded and skipped, never substituted.

Every number is a steady median (the second half of back-to-back calls, after
a warmup long enough for the NPU's ramp), the same protocol as the other
benchmarks here.
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

from src.compressors.frappe.harness import AnonymousImageFolder
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.reporting import Table, write_report
from src.compressors.frappe.openvino_runtime import testbed


def steady(call, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        call()
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(times[len(times) // 2 :])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx-stem",
        type=Path,
        required=True,
        help="an exported pair with a dynamic batch dimension (export with --export-batch-max N)",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="inferences discarded before timing; the NPU needs ~15",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import openvino

    core = openvino.Core()
    folder = AnonymousImageFolder(args.dataset_root, args.split)
    largest = max(args.batches)
    if largest > len(folder):
        raise SystemExit(
            f"the {args.split!r} split has {len(folder)} images; batch {largest} does not fit"
        )
    _b, _c, height, width = folder.pixels(0).shape
    pixels = height * width
    images = np.concatenate([folder.pixels(i).numpy() for i in range(largest)])
    print(
        f"{width}x{height}, batches {args.batches}, devices {args.devices}, "
        f"{args.iterations} timed calls after {args.warmup} warmup\n"
    )

    rows = []
    entries: dict = {}
    for device in args.devices:
        # Per-device golden baseline: the same image run one at a time. An fp16
        # device drifts from a CPU reference for precision reasons, which would
        # masquerade as a batching effect, so each device is compared against
        # its own batch-one execution.
        single_model = core.read_model(f"{args.onnx_stem}_encoder.onnx")
        single_model.reshape({0: openvino.PartialShape([1, 3, height, width])})
        single = core.compile_model(single_model, device)
        single_in = single.inputs[0].get_any_name()
        plane_names = [port.get_any_name() for port in single.outputs]
        singles = {
            name: np.stack(
                [
                    np.array(single({single_in: images[i : i + 1]})[name].data, copy=True)[0]
                    for i in range(largest)
                ]
            )
            for name in plane_names
        }

        for batch in args.batches:
            entry: dict = {"batch": batch, "device": device}
            try:
                encoder_model = core.read_model(f"{args.onnx_stem}_encoder.onnx")
                encoder_model.reshape({0: openvino.PartialShape([batch, 3, height, width])})
                encoder = core.compile_model(encoder_model, device)
                encoder_in = encoder.inputs[0].get_any_name()

                ms = steady(
                    lambda e=encoder, n=encoder_in, b=batch: e({n: images[:b]}),
                    args.iterations,
                    args.warmup,
                )
                entry["encode_ms_per_image"] = ms / batch
                entry["encode_mpixel_s"] = batch * pixels / ms / 1000

                planes = {
                    name: np.array(encoder({encoder_in: images[:batch]})[name].data, copy=True)
                    for name in plane_names
                }
                # Batch-N planes must equal the same device's batch-one planes,
                # image for image. A drift of one code level on a few percent of
                # symbols still counts: the planes are the bitstream.
                drift = sum(
                    int((planes[name][i] != singles[name][i]).sum())
                    for name in plane_names
                    for i in range(batch)
                )
                worst = max(
                    int(
                        np.abs(
                            planes[name][i].astype(np.int32) - singles[name][i].astype(np.int32)
                        ).max()
                    )
                    for name in plane_names
                    for i in range(batch)
                )
                entry["plane_drift_vs_batch_one"] = drift
                entry["planes_match_batch_one"] = drift == 0
                entry["plane_drift_max_code_delta"] = worst

                decoder_model = core.read_model(f"{args.onnx_stem}_decoder.onnx")
                # A plane already carries the batch axis: (batch, rows, cols).
                decoder_model.reshape(
                    {
                        index: openvino.PartialShape(plane.shape)
                        for index, plane in enumerate(planes.values())
                    }
                )
                decoder = core.compile_model(decoder_model, device)
                ms = steady(lambda d=decoder, f=planes: d(f), args.iterations, args.warmup)
                entry["decode_ms_per_image"] = ms / batch
                entry["decode_mpixel_s"] = batch * pixels / ms / 1000
                entry["execution_devices"] = next(iter(encoder.get_property("EXECUTION_DEVICES")))
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"[:400]
            rows.append(entry)
            entries.setdefault(device, {})[str(batch)] = entry

    print(
        Table(
            [
                ("device", "device", ""),
                ("batch", "batch", ">5"),
                ("enc ms/img", "encode_ms_per_image", ">11"),
                ("enc Mpx/s", "encode_mpixel_s", ">10"),
                ("dec ms/img", "decode_ms_per_image", ">11"),
                ("dec Mpx/s", "decode_mpixel_s", ">9"),
                ("drift", "plane_drift_vs_batch_one", ">6"),
                ("error", "error", ""),
            ]
        ).render(
            [
                {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in row.items()}
                for row in rows
            ]
        )
    )

    identical = all(
        entry.get("planes_match_batch_one", True)
        for device_entries in entries.values()
        for entry in device_entries.values()
    )
    drifts = {
        f"{device}/{batch}": entry["plane_drift_vs_batch_one"]
        for device, device_entries in entries.items()
        for batch, entry in device_entries.items()
        if entry.get("plane_drift_vs_batch_one")
    }
    if identical:
        print(
            "\n  every batch produced the batch-one planes, byte for byte: "
            "batching changes throughput only, not the bitstream"
        )
    elif drifts:
        print(f"\n  batched runs drifted from their device's batch-one run: {drifts}")
    write_report(
        {
            "onnx_stem": str(args.onnx_stem),
            "static_shape": [width, height],
            "batches": args.batches,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "planes_match_batch_one": identical,
            "entries": entries,
            "testbed": testbed(core),
        },
        args.report,
    )


if __name__ == "__main__":
    main()
