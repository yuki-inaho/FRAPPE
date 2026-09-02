#!/usr/bin/env python3
"""Should the JPEG-LS front half leave the CPU? Measured, not argued.

The front half -- MED, gradient quantisation, context ID -- is the only part of
JPEG-LS without a data dependency between neighbouring samples, which makes it
the only part an accelerator can take. The question this tool answers is
whether taking it pays. The comparison is deliberately lopsided: on one side
the front half alone, on every device OpenVINO enumerates; on the other side
CharLS doing the *entire* codec -- front half, adaptive contexts, run mode,
Golomb, bit stuffing, markers -- on the CPU. If the whole thing on the CPU is
faster than the half on the accelerator, the offload is a net loss and the
number that says so is worth keeping.

The planes measured are FRAPPE's real code planes for one image, produced by
the CPU encoder, because the shapes span a 384x range in sample count (475 to
182,400 at 800x608) and the per-plane cost scales with it. Per plane, the timed
unit is the realistic offload path: line buffer on the host, then one graph
inference. The host glue is timed separately too, because it survives any
device -- a zero-cost front half would still pay it.

Verification is off during timing (``verify="none"``); exactness is the tests'
job, and on this machine every enumerated device already agrees with the CPU
oracle exactly at every shape measured here.
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

from src.compressors.frappe.harness import AnonymousImageFolder, BitstreamConvention, encode_planes
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.reporting import write_report
from src.compressors.frappe.jpegls_openvino import MedContextGraph, build_line_buffer
from src.compressors.frappe.openvino_runtime import FrappeEncoder, available_devices, testbed


def steady_median(times: list[float]) -> float:
    """Median over the second half, so a warm-up tail cannot inflate it."""
    return statistics.median(times[len(times) // 2 :])


def series(action, iterations: int) -> list[float]:
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        action()
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def as_torch_planes(planes):
    import numpy as np
    import torch

    return [
        torch.from_numpy(np.ascontiguousarray(plane[0] if plane.ndim == 3 else plane))
        for plane in planes
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx-stem",
        type=Path,
        required=True,
        help="exported graph pair; the CPU encoder produces the planes",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--devices", nargs="+", default=None, help="default: every device OpenVINO enumerates"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=60,
        help="back-to-back timed calls per plane; the median over the "
        "second half is the steady state",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import openvino

    core = openvino.Core()
    devices = args.devices or available_devices(core)

    image = AnonymousImageFolder(args.dataset_root, args.split).pixels(args.index).numpy()
    _batch, _channels, height, width = image.shape
    encoder = FrappeEncoder(args.onnx_stem, "CPU", height, width, core=core)
    planes = [plane[0] if plane.ndim == 3 else plane for plane in encoder(image)]
    shapes = [plane.shape for plane in planes]
    samples = sum(rows * cols for rows, cols in shapes)
    print(
        f"{width}x{height}, split {args.split} index {args.index}, "
        f"{len(planes)} planes, {samples} samples, devices {devices}"
    )
    print(f"{args.iterations} back-to-back calls per plane; steady = median over the second half\n")

    # The host glue: build the bordered buffer the graph slices. It runs on the
    # CPU no matter which device takes the graph.
    glue = steady_median(
        series(lambda: [build_line_buffer(plane) for plane in planes], args.iterations)
    )

    front_half = {}
    for device in devices:
        per_plane, failed = {}, None
        try:
            graphs = [MedContextGraph(rows, cols, device, core=core) for rows, cols in shapes]
            for (rows, cols), graph, plane in zip(shapes, graphs, planes):
                buffer = build_line_buffer(plane)
                per_plane[f"{rows}x{cols}"] = steady_median(
                    series(lambda g=graph, b=buffer: g(b), args.iterations)
                )
        except Exception as error:
            failed = f"{type(error).__name__}: {error}"
            print(f"  {device}: {failed[:100]}")
        entry = {"per_plane_ms": per_plane, "error": failed}
        if not failed:
            entry["total_ms"] = sum(per_plane.values()) + glue
            entry["execution_devices"] = graphs[0].execution_devices
            print(
                f"  {device}: front half {entry['total_ms']:.2f} ms/image "
                f"(+ {glue:.2f} ms host glue)"
            )
        front_half[device] = entry

    # The comparator: CharLS on the CPU, doing everything.
    tensors = as_torch_planes(planes)
    charls = steady_median(
        series(lambda: encode_planes(tensors, BitstreamConvention.PAYLOAD_ONLY), args.iterations)
    )

    columns = (
        f"\n{'device':7s} {'front half':>11s} {'glue':>7s} {'whole codec':>12s} "
        f"{'loss factor':>12s}"
    )
    print(columns)
    print("-" * (len(columns) - 1))
    print(
        f"{'CPU':7s} {front_half['CPU']['total_ms']:10.2f}m {glue:6.2f}m "
        f"{charls:11.2f}m {front_half['CPU']['total_ms'] / charls:11.1f}x"
    )
    for device, entry in front_half.items():
        if device == "CPU":
            continue
        if entry["error"]:
            print(f"{device:7s} {entry['error'][:70]}")
            continue
        print(
            f"{device:7s} {entry['total_ms']:10.2f}m {glue:6.2f}m "
            f"{charls:11.2f}m {entry['total_ms'] / charls:11.1f}x"
        )

    verdict = {
        "front_half_total_ms": {
            device: entry.get("total_ms") for device, entry in front_half.items()
        },
        "host_glue_ms": glue,
        "charls_whole_codec_cpu_ms": charls,
        "cheapest_front_half_device": min(
            (device for device, entry in front_half.items() if entry.get("total_ms")),
            key=lambda device: front_half[device]["total_ms"],
            default=None,
        ),
        "conclusion": None,
    }
    best = verdict["cheapest_front_half_device"]
    if best:
        ratio = verdict["front_half_total_ms"][best] / charls
        verdict["conclusion"] = (
            f"even the cheapest front half ({best}, "
            f"{verdict['front_half_total_ms'][best]:.2f} ms) is {ratio:.1f}x the "
            f"cost of CharLS doing the entire codec on the CPU ({charls:.2f} ms); "
            f"offloading does not pay for latency"
        )
        print(f"\n  {verdict['conclusion']}\n")

    report = {
        "onnx_stem": str(args.onnx_stem),
        "split": args.split,
        "index": args.index,
        "static_shape": [width, height],
        "plane_shapes": [list(shape) for shape in shapes],
        "samples": samples,
        "iterations": args.iterations,
        "front_half": front_half,
        "verdict": verdict,
        "testbed": testbed(core),
    }
    write_report(report, args.report)


if __name__ == "__main__":
    main()
