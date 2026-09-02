#!/usr/bin/env python3
"""Evaluate a joint-prefix FRAPPE checkpoint on a full local split.

Every number comes from the deployment path: true int8 codes and a real
entropy-coded bitstream, with ``compression_ratio = 24 / bpp`` against
uncompressed 8-bit RGB. The report carries the whole prefix rate-distortion
ladder plus the monotonicity diagnostics -- a ladder that is not monotone means
the prefix property, which is the point of the architecture, has broken.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import AnonymousImageFolder
from src.compressors.frappe.harness.bitstream import BitstreamConvention
from src.compressors.frappe.harness.checkpoints import load_checkpoint
from src.compressors.frappe.harness.cli import (
    add_dataset_arguments,
    add_device_argument,
    add_output_argument,
    resolve_device,
)
from src.compressors.frappe.harness.evaluation import (
    evaluate_operating_points,
    monotonicity_violations,
)
from src.compressors.frappe.harness.metrics import Averaging
from src.compressors.frappe.harness.reporting import Table, write_report

LADDER = Table([("n", "label", "3d"), ("PSNR dB", "psnr_db", "8.2f"),
                ("bpp", "bpp", "9.4f"), ("CR", "compression_ratio", "9.2f")])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--prefixes", type=int, nargs="+", default=None,
                        help="operating points to measure; default: every prefix 1..N")
    parser.add_argument("--averaging", choices=[value.value for value in Averaging],
                        default=Averaging.AGGREGATE_MSE.value,
                        help="aggregate_mse matches the joint-prefix tools, mean_psnr "
                             "matches evaluate.py and the shipped results")
    parser.add_argument("--count-length-prefix", action="store_true",
                        help="include the 4-byte per-scale length prefixes in the rate, "
                             "as entropy_coding.encode_latents does")
    add_dataset_arguments(parser, images=None)
    parser.add_argument("--images", type=int, default=None,
                        help="how many images of each split to use; default: all")
    add_device_argument(parser)
    add_output_argument(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    points = args.prefixes or list(range(1, checkpoint.channels + 1))
    convention = (BitstreamConvention.WITH_LENGTH_PREFIX if args.count_length_prefix
                  else BitstreamConvention.PAYLOAD_ONLY)
    print(checkpoint.describe(), flush=True)

    report = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "iteration": checkpoint.iteration,
              "ps": list(checkpoint.config.ps), "averaging": args.averaging,
              "count_length_prefix": args.count_length_prefix, "splits": {}}
    for split in args.splits:
        folder = AnonymousImageFolder(args.dataset_root, split)
        count = min(args.images or len(folder), len(folder))

        def show(done: int, total: int) -> None:
            if done % 200 == 0 or done == total:
                print(f"  {split}: {done}/{total}", flush=True)  # noqa: B023

        curve = evaluate_operating_points(
            checkpoint.model, folder, points, images=count, device=device,
            averaging=Averaging(args.averaging), convention=convention, progress=show)
        violations = monotonicity_violations(curve)
        print(f"\n  {split}: {count} images")
        print(LADDER.render(point.as_dict() for point in curve))
        print(f"    monotonicity violations: {violations}/{max(len(curve) - 1, 0)}",
              flush=True)
        report["splits"][split] = {
            "images": count, "image_indices": list(range(count)),
            "curve": [point.as_dict() for point in curve],
            "final_psnr_db": curve[-1].psnr_db, "final_bpp": curve[-1].bpp,
            "final_compression_ratio": curve[-1].compression_ratio,
            "monotonicity_violations": violations,
        }
    write_report(report, args.output)


if __name__ == "__main__":
    main()
