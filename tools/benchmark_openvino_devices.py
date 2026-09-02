#!/usr/bin/env python3
"""Where should each half of the FRAPPE codec run, once every device is warm?

The analysis and synthesis paths have opposite cost profiles -- a handful of
strided convolutions against a decoder holding essentially all of the model's
parameters -- so the right device for one is not the right device for the other,
and the answer is a measurement. This sweeps the encoder and the decoder
independently over the devices OpenVINO enumerates. Independently, not jointly:
the two graphs share no tensor, so the cost of a pairing is the sum of its halves
and measuring all nine combinations would spend three times the wall clock to
reproduce the same two columns.

The reason this tool is more than a loop around ``time.perf_counter`` is that on
this class of hardware a single number is a wrong number. Four effects move it,
and they move it in opposite directions:

``compile``
    Building the device blob. Measured cold and again against ``--cache-dir``,
    because the gap between them is the difference between a process that starts
    in milliseconds and one that stalls for seconds. On the Meteor Lake NPU the
    decoder blob has been observed to take several seconds cold.

``first inference``
    Buffer allocation and lazy device setup, paid once after compilation.

``ramp``
    The NPU does not run at its eventual rate immediately. A clean two-level
    trajectory has been observed -- a plateau for the first dozen or so
    inferences and then roughly half the latency, sustained. A benchmark that
    warms up for three iterations and reports the median measures the plateau and
    concludes the NPU is twice as slow as it is. ``iterations_to_steady`` is
    reported so the ramp is visible rather than averaged away, and the whole
    trajectory is kept in the report.

``duty cycle``
    Whether the device is kept busy. Back-to-back and gap-separated series are
    both measured because they do not agree and they do not disagree in the same
    direction on every device: a CPU left idle between frames drops its clocks
    and gets slower, while a warmed NPU has been observed to hold its rate across
    idle gaps. Which number applies depends on whether the deployment encodes a
    stream or the occasional frame, so both are reported and neither is called
    "the" latency.

A device that fails to compile is recorded with its error and the sweep
continues, because "the NPU plugin rejects this graph" is a result. It is never
silently replaced by one that works.

JPEG-LS is measured too, on the CPU, since it is the stage no accelerator here
takes and therefore the floor under any end-to-end number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import (
    AnonymousImageFolder,
    BitstreamConvention,
    encode_planes,
)
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.reporting import write_report
from src.compressors.frappe.openvino_runtime import (
    FrappeDecoder,
    FrappeEncoder,
    available_devices,
    testbed,
)


def as_torch_planes(planes):
    """OpenVINO's ``(1, rows, cols)`` uint8 arrays as the harness's 2D tensors."""
    import numpy as np
    import torch

    return [torch.from_numpy(np.ascontiguousarray(plane[0] if plane.ndim == 3 else plane))
            for plane in planes]

#: A trajectory has reached steady state at the first iteration from which every
#: later one stays within this fraction of the steady median. "Every later one"
#: rather than "the next one" because a ramp can dip once before settling.
STEADY_TOLERANCE = 0.10


def iterations_to_steady(trajectory: list[float], steady: float,
                         tolerance: float = STEADY_TOLERANCE) -> int:
    """First index from which the series never again exceeds the steady median.

    Returns ``0`` when the device is fast from its first inference, and
    ``len(trajectory)`` when it never settles inside the tolerance -- which is
    itself worth seeing, because it means the measurement window was too short.
    """
    limit = steady * (1.0 + tolerance)
    settled = len(trajectory)
    for index in range(len(trajectory) - 1, -1, -1):
        if trajectory[index] > limit:
            break
        settled = index
    return settled


def summarise(trajectory: list[float]) -> dict:
    """Split a back-to-back series into its ramp and its steady state.

    The steady median is taken over the last half of the series rather than over
    everything after the nominal warm-up, so it does not depend on guessing the
    warm-up length in advance -- the whole point is that the length is what we
    are trying to find out.
    """
    tail = trajectory[len(trajectory) // 2:]
    steady = statistics.median(tail)
    settled = iterations_to_steady(trajectory, steady)
    ramp = trajectory[:settled]
    return {
        "iterations": len(trajectory),
        "first_infer_ms": trajectory[0],
        "steady_median_ms": steady,
        "steady_min_ms": min(tail),
        "steady_max_ms": max(tail),
        "iterations_to_steady": settled,
        # What the ramp costs in total beyond running at the steady rate: the
        # latency a deployment eats before it is up to speed.
        "ramp_excess_ms": sum(ramp) - settled * steady if ramp else 0.0,
        "ramp_median_ms": statistics.median(ramp) if ramp else None,
        "trajectory_ms": trajectory,
    }


def run_series(stage, feed, count: int, gap: float = 0.0) -> list[float]:
    times = []
    for _ in range(count):
        if gap:
            time.sleep(gap)
        started = time.perf_counter()
        stage(feed)
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def profile_stage(build, feed, iterations: int, isolated_iterations: int,
                  gap: float) -> dict:
    """Compile, then measure the ramp, the steady state and the duty-cycle effect.

    ``build`` is called twice. The first call is the cold compile; the second is
    measured too, because with an OpenVINO cache directory set it is a cache hit
    and the pair of numbers is what tells a deployment whether to ship a cache.
    """
    started = time.perf_counter()
    stage = build()
    cold_compile_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    build()
    second_compile_ms = (time.perf_counter() - started) * 1000.0

    profile = summarise(run_series(stage, feed, iterations))
    profile["compile_ms"] = cold_compile_ms
    profile["compile_again_ms"] = second_compile_ms
    profile["execution_devices"] = stage.execution_devices

    # The device is warm now, so this isolates the duty-cycle effect from the
    # ramp: any difference here is about idling between calls, not about start-up.
    if isolated_iterations > 0:
        isolated = run_series(stage, feed, isolated_iterations, gap=gap)
        profile["isolated"] = {
            "iterations": isolated_iterations,
            "gap_seconds": gap,
            "median_ms": statistics.median(isolated),
            "trajectory_ms": isolated,
            "duty_cycle_ratio": statistics.median(isolated) / profile["steady_median_ms"],
        }
    return profile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-stem", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--devices", nargs="+", default=None,
                        help="default: every device OpenVINO enumerates")
    parser.add_argument("--iterations", type=int, default=60,
                        help="back-to-back inferences per stage. Must comfortably "
                             "exceed the ramp; the NPU has been seen to need ~15")
    parser.add_argument("--isolated-iterations", type=int, default=12,
                        help="gap-separated inferences, to measure the duty-cycle "
                             "effect once the device is already warm. 0 disables")
    parser.add_argument("--gap", type=float, default=0.2,
                        help="idle seconds between the isolated inferences")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="OpenVINO compiled-model cache. With it, compile_again_ms "
                             "is a cache hit and the pair shows what shipping a cache buys")
    parser.add_argument("--properties", type=str, default=None,
                        help='JSON of extra device properties, e.g. \'{"NPU_TURBO": true}\'')
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import openvino

    core = openvino.Core()
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(args.cache_dir)})
    properties = json.loads(args.properties) if args.properties else {}
    devices = args.devices or available_devices(core)

    image = AnonymousImageFolder(args.dataset_root, args.split).pixels(args.index).numpy()
    _batch, _channels, height, width = image.shape
    print(f"{width}x{height}, {args.split} split index {args.index}, devices {devices}")
    print(f"{args.iterations} back-to-back + {args.isolated_iterations} at "
          f"{args.gap}s gaps, properties {properties or '{}'}\n")

    # One CPU encoder first: every decoder must be fed the same planes for the
    # comparison to mean anything.
    baseline = FrappeEncoder(args.onnx_stem, "CPU", height, width, core=core)
    planes = baseline(image)
    plane_shapes = baseline.plane_shapes

    encoders, decoders = {}, {}
    for device in devices:
        for label, table, build, feed in (
            ("encode", encoders,
             lambda d=device: FrappeEncoder(args.onnx_stem, d, height, width,
                                            core=core, properties=properties), image),
            ("decode", decoders,
             lambda d=device: FrappeDecoder(args.onnx_stem, d, plane_shapes,
                                            core=core, properties=properties), planes),
        ):
            print(f"  profiling {label} on {device} ...", flush=True)
            try:
                table[device] = profile_stage(build, feed, args.iterations,
                                              args.isolated_iterations, args.gap)
            except Exception as error:
                table[device] = {"error": f"{type(error).__name__}: {error}"}
                print(f"    {table[device]['error'][:110]}")

    tensors = as_torch_planes(planes)
    started = time.perf_counter()
    for _ in range(max(1, args.iterations // 4)):
        encode_planes(tensors, BitstreamConvention.WITH_LENGTH_PREFIX)
    jpegls_ms = (time.perf_counter() - started) / max(1, args.iterations // 4) * 1000.0

    pixels = height * width
    columns = (f"\n{'stage':7s} {'device':7s} {'compile':>9s} {'again':>8s} {'first':>8s} "
               f"{'ramp':>6s} {'steady':>9s} {'idle-gap':>9s} {'Mpx/s':>8s}")
    print(columns)
    print("-" * (len(columns) - 1))
    for label, table in (("encode", encoders), ("decode", decoders)):
        for device, entry in table.items():
            if "error" in entry:
                print(f"{label:7s} {device:7s} {entry['error'][:60]}")
                continue
            steady = entry["steady_median_ms"]
            isolated = entry.get("isolated", {}).get("median_ms")
            print(f"{label:7s} {device:7s} {entry['compile_ms']:8.0f}m "
                  f"{entry['compile_again_ms']:7.0f}m {entry['first_infer_ms']:7.1f}m "
                  f"{entry['iterations_to_steady']:6d} {steady:8.2f}m "
                  f"{(f'{isolated:8.2f}m' if isolated is not None else '        -')} "
                  f"{pixels / steady / 1000:8.1f}")
    print(f"{'jpegls':7s} {'CPU':7s} {'-':>9s} {'-':>8s} {'-':>8s} {'-':>6s} "
          f"{jpegls_ms:8.2f}m {'-':>9s} {pixels / jpegls_ms / 1000:8.1f}")
    print("\n  ramp = inferences before the latency settles; a stage with a large ramp "
          "\n  is mis-measured by any benchmark that warms up fewer times than that.")

    def best(table):
        ran = {device: entry for device, entry in table.items() if "error" not in entry}
        return min(ran, key=lambda device: ran[device]["steady_median_ms"]) if ran else None

    best_encoder, best_decoder = best(encoders), best(decoders)
    summary = None
    if best_encoder and best_decoder:
        warm = (encoders[best_encoder]["steady_median_ms"] + jpegls_ms
                + decoders[best_decoder]["steady_median_ms"])
        summary = {"encoder": best_encoder, "decoder": best_decoder,
                   "warm_total_ms": warm, "warm_mpixel_s": pixels / warm / 1000}
        compile_cost = (encoders[best_encoder]["compile_ms"]
                        + decoders[best_decoder]["compile_ms"])
        ramp_cost = (encoders[best_encoder]["ramp_excess_ms"]
                     + decoders[best_decoder]["ramp_excess_ms"])
        print(f"\nfastest warm round trip: encode on {best_encoder}, JPEG-LS on CPU, "
              f"decode on {best_decoder}\n  {warm:.2f} ms "
              f"({pixels / warm / 1000:.1f} Mpixel/s) once both graphs are warm. "
              f"Cold, add {compile_cost:.0f} ms of compilation "
              f"and {ramp_cost:.0f} ms of ramp.")

    report = {
        "onnx_stem": str(args.onnx_stem),
        "split": args.split,
        "index": args.index,
        "static_shape": [width, height],
        "devices_requested": devices,
        "properties": properties,
        "encoder": encoders,
        "decoder": decoders,
        "jpegls_cpu_median_ms": jpegls_ms,
        "fastest_warm": summary,
        "steady_tolerance": STEADY_TOLERANCE,
        "cache_dir_used": bool(args.cache_dir),
        "testbed": testbed(core),
    }
    write_report(report, args.report)


if __name__ == "__main__":
    main()
