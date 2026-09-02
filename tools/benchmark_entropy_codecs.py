#!/usr/bin/env python3
"""Which entropy codec should carry the encoder's planes? CharLS against the rest.

The encoder graph ends at five uint8 planes and everything after is a choice of
lossless codec for them. CharLS holds that seat, and this tool is the standing
check on whether it should keep it: it times candidate codecs encoding and
decoding the same real planes on the CPU, in-process, and compares the bytes.
The question is narrow on purpose -- the planes are high-entropy quantised
latents, so a codec that looks strong on photographs can still lose here on
both speed and size.

Timing is the median of the second half of back-to-back calls, which is the
protocol every other benchmark in this repository uses. The analysis stage is
timed separately on a pre-loaded image so the encode-stage total does not
accidentally include the PNG decode of the source.

JPEG XL candidates use the ``pillow_jxl`` plugin (which bundles its own current
libjxl, so results do not depend on the system's potentially older cjxl). They
are probed and skipped when the plugin is missing, like every codec in
``harness.codecs``.
"""

from __future__ import annotations

import argparse
import io
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
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.reporting import write_report
from src.compressors.frappe.openvino_runtime import FrappeEncoder, bit_exact_properties, testbed


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
    return [
        torch.from_numpy(np.ascontiguousarray(plane[0] if plane.ndim == 3 else plane))
        for plane in planes
    ]


def jxl_available() -> bool:
    try:
        import pillow_jxl  # noqa: F401
    except ImportError:
        return False
    return True


def jxl_encode_plane(plane: np.ndarray, effort: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(plane, mode="L").save(buffer, format="JXL", lossless=True, effort=int(effort))
    return buffer.getvalue()


def jxl_decode_plane(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as decoded:
        decoded.load()
        return np.asarray(decoded)


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
    parser.add_argument(
        "--images", type=int, default=0, help="how many images to measure; 0 means the whole split"
    )
    parser.add_argument(
        "--jxl-efforts",
        type=int,
        nargs="+",
        default=[1],
        help="JPEG XL lossless efforts to measure, e.g. --jxl-efforts 1 3",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=16,
        help="back-to-back calls per stage and image; the steady median "
        "is taken over the second half",
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
    width, height = folder.pixels(0).shape[3], folder.pixels(0).shape[2]
    encoder = FrappeEncoder(
        args.onnx_stem,
        "CPU",
        height,
        width,
        core=core,
        properties=bit_exact_properties("CPU", core),
    )
    efforts = args.jxl_efforts if jxl_available() else []
    if args.jxl_efforts and not efforts:
        print("pillow_jxl is not installed; measuring CharLS only")

    names = ["charls"] + [f"jxl-e{effort}" for effort in efforts]
    totals = {name: {"encode_ms": [], "decode_ms": [], "bytes": 0} for name in names}
    analysis_ms = []
    samples_per_image = 0

    for index in range(count):
        image = folder.pixels(index).numpy()
        planes = [plane[0] if plane.ndim == 3 else plane for plane in encoder(image)]
        tensors = as_torch_planes(planes)
        samples_per_image = sum(plane.size for plane in planes)

        analysis_ms.append(steady_median(series(lambda a=image: encoder(a), args.iterations)))

        # CharLS: bare payloads are what the rate convention counts; the decode
        # path needs the self-describing form, so both are made here.
        bare = encode_planes(tensors, BitstreamConvention.PAYLOAD_ONLY)
        prefixed = encode_planes(tensors, BitstreamConvention.WITH_LENGTH_PREFIX)
        totals["charls"]["encode_ms"].append(
            steady_median(
                series(
                    lambda t=tensors: encode_planes(t, BitstreamConvention.PAYLOAD_ONLY),
                    args.iterations,
                )
            )
        )
        totals["charls"]["decode_ms"].append(
            steady_median(series(lambda b=prefixed: decode_planes(b), max(1, args.iterations // 2)))
        )
        totals["charls"]["bytes"] += len(bare)

        for effort in efforts:
            payloads = [jxl_encode_plane(plane, effort) for plane in planes]
            key = f"jxl-e{effort}"
            totals[key]["encode_ms"].append(
                steady_median(
                    series(
                        lambda p=planes, e=effort: [jxl_encode_plane(plane, e) for plane in p],
                        args.iterations,
                    )
                )
            )
            totals[key]["decode_ms"].append(
                steady_median(
                    series(
                        lambda pl=payloads: [jxl_decode_plane(payload) for payload in pl],
                        max(1, args.iterations // 2),
                    )
                )
            )
            totals[key]["bytes"] += sum(len(payload) for payload in payloads)

    images = count
    analysis = statistics.median(analysis_ms)
    columns = (
        f"\n{'codec':10s} {'encode':>9s} {'decode':>9s} {'+analysis':>10s} "
        f"{'bytes/img':>10s} {'b/sample':>9s}"
    )
    print(
        f"{images} image(s), {samples_per_image} plane samples each, "
        f"analysis {analysis:.2f} ms (CPU, bit-exact)"
    )
    print(columns)
    print("-" * (len(columns) - 1))
    results = {}
    for name in names:
        entry = totals[name]
        encode = statistics.median(entry["encode_ms"])
        decode = statistics.median(entry["decode_ms"])
        bps = entry["bytes"] * 8 / (images * samples_per_image)
        results[name] = {
            "encode_ms": encode,
            "decode_ms": decode,
            "encode_stage_ms": analysis + encode,
            "bytes_per_image": entry["bytes"] / images,
            "bits_per_sample": bps,
        }
        print(
            f"{name:10s} {encode:8.2f}m {decode:8.2f}m {analysis + encode:9.2f}m "
            f"{entry['bytes'] / images:9.0f} {bps:8.3f}"
        )

    baseline = results["charls"]
    for name, entry in results.items():
        if name == "charls":
            continue
        print(
            f"  {name} vs charls: encode {entry['encode_ms'] / baseline['encode_ms']:.2f}x, "
            f"decode {entry['decode_ms'] / baseline['decode_ms']:.2f}x, "
            f"bytes {entry['bytes_per_image'] / baseline['bytes_per_image']:.2f}x"
        )

    write_report(
        {
            "onnx_stem": str(args.onnx_stem),
            "split": args.split,
            "images": images,
            "static_shape": [width, height],
            "plane_samples_per_image": samples_per_image,
            "iterations": args.iterations,
            "analysis_ms": analysis,
            "codecs": results,
            "testbed": testbed(core),
        },
        args.report,
    )


if __name__ == "__main__":
    main()
