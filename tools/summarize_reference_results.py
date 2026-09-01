#!/usr/bin/env python3
"""Summarise the rate-distortion result files shipped in ``results/``.

Those JSON files are the repository's own evaluation of the released FRAPPE
weights and of the baseline codecs, produced by
``src/compressors/frappe/evaluate_rate_distortion.py``.  They are the reference
any locally trained model has to be compared against, and they fix the rate
convention: ``bpp`` there is the length of a real entropy-coded bitstream divided
by the pixel count, and the compression ratio is ``24 / bpp`` against
uncompressed 8-bit RGB.

Per-image records are averaged the same way the shipping harness averages them:
the MEAN OF PER-IMAGE PSNR, which is what ``evaluate.py`` returns
(``np.mean(psnrs)``) and what the notebook prints.  That is deliberately *not*
the convention of ``tools/evaluate_joint_prefix.py``,
``tools/evaluate_released_model.py`` and ``tools/benchmark_reference_codecs.py``,
which convert one aggregate MSE into a single PSNR.  The two differ by up to
about 0.8 dB on this Kodak curve because PSNR is convex in MSE, so numbers from
this tool are comparable with the paper and with each other, and numbers from
those tools are comparable with the paper and with each other -- but the two
families must not be mixed inside one table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_curves(results_root: Path, task: str = "rate_distortion") -> dict[str, dict]:
    curves: dict[str, dict] = {}
    for path in sorted(results_root.glob(f"*/{task}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        codec = payload.get("codec", path.parent.name)
        entries = []
        results = payload.get("results", {})
        for key in sorted(results, key=lambda value: float(value)):
            per_image = results[key].get("per_image", [])
            if not per_image:
                continue
            entry = {"operating_point": key, "images": len(per_image)}
            for metric in payload.get("metrics", []):
                values = [record[metric] for record in per_image
                          if record.get(metric) is not None and math.isfinite(record[metric])]
                if values:
                    entry[metric] = float(np.mean(values))
            if "bpp" in entry:
                entry["compression_ratio"] = 24.0 / entry["bpp"]
            entries.append(entry)
        curves[codec] = {"path": str(path), "dataset": payload.get("dataset"),
                         "n_images": payload.get("n_images"), "points": entries}
    return curves


def interpolate(points: list[dict], bpp: float, metric: str = "PSNR_dB") -> float | None:
    usable = [(point["bpp"], point[metric]) for point in points if metric in point]
    if len(usable) < 2:
        return None
    usable.sort()
    rates = [value for value, _ in usable]
    if bpp < rates[0] or bpp > rates[-1]:
        return None
    return float(np.interp(math.log(bpp), [math.log(r) for r in rates],
                           [q for _, q in usable]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--task", default="rate_distortion")
    parser.add_argument("--metric", default="PSNR_dB")
    parser.add_argument("--codecs", nargs="+", default=None)
    parser.add_argument("--compare-compression", type=float, nargs="+",
                        default=[3.5, 10.0, 20.0, 50.0, 100.0, 240.0],
                        help="compression ratios at which to line the codecs up")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    curves = load_curves(args.results_root, args.task)
    if args.codecs:
        curves = {name: value for name, value in curves.items() if name in args.codecs}
    if not curves:
        raise SystemExit(f"no {args.task} results under {args.results_root}")

    for codec, curve in curves.items():
        print(f"\n{codec}  ({curve['dataset']}, {curve['n_images']} images)")
        header = [key for key in ("bpp", "compression_ratio", args.metric)
                  if any(key in point for point in curve["points"])]
        print("    " + f"{'point':>7}" + "".join(f"{name:>20}" for name in header))
        for point in curve["points"]:
            cells = "".join(f"{point[name]:20.4f}" if name in point else f"{'-':>20}"
                            for name in header)
            print(f"    {point['operating_point']:>7}" + cells)

    print(f"\n{args.metric} lined up at equal compression ratio "
          f"(interpolated in log rate; '-' means outside the measured ladder):")
    names = sorted(curves)
    print("    " + f"{'CR':>7}{'bpp':>9}" + "".join(f"{name:>12}" for name in names))
    for ratio in args.compare_compression:
        bpp = 24.0 / ratio
        cells = []
        for name in names:
            value = interpolate(curves[name]["points"], bpp, args.metric)
            cells.append(f"{'-':>12}" if value is None else f"{value:12.2f}")
        print(f"    {ratio:7.1f}{bpp:9.4f}" + "".join(cells))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
