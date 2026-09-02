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
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import AnonymousImageFolder  # noqa: E402
from src.compressors.frappe.harness.cli import (  # noqa: E402
    add_dataset_arguments,
    add_output_argument,
)
from src.compressors.frappe.harness.codecs import (  # noqa: E402
    DEFAULT_LADDERS,
    available_codecs,
    interpolate,
    sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    add_dataset_arguments(parser)
    add_output_argument(parser)
    args = parser.parse_args()

    folder = AnonymousImageFolder(args.dataset_root, args.split)
    images = [folder.pil(index) for index in range(min(args.images, len(folder)))]
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
        print("\n  FRAPPE operating points against the reference curves "
              "(dB at matched rate):")
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
