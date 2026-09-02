#!/usr/bin/env python3
"""Package the official NPU INT8 encoder deployment for one operating point.

One command produces the deployment directory: ``encoder_fp32.onnx``,
``encoder_int8_qdq.onnx`` (the canonical INT8 encoder, analysis convolutions
only), ``encoder_int8_<W>x<H>.xml/.bin`` (OpenVINO IR frozen to the target
resolution), ``decoder.onnx``, ``manifest.json`` and ``package_report.json``.
The calibration split feeds statistics only and never the verification split;
the RD baseline in the manifest is measured through the shipped artifacts
themselves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args(argv: list[str] | None = None):

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="FP32 joint-prefix checkpoint to deploy; never modified",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="anonymous imagefolder; supplies calibration and verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="receives the package; do not reuse across checkpoints",
    )
    parser.add_argument(
        "--calibration-split",
        default="train",
        help="dataset split used only for PTQ statistics; train by contract",
    )
    parser.add_argument("--calibration-images", type=int, default=32)
    parser.add_argument("--verify-split", default="validation")
    parser.add_argument(
        "--verify-images",
        type=int,
        default=16,
        help="images measured through the shipped artifacts for the manifest's RD baseline",
    )
    parser.add_argument(
        "--height", type=int, default=608, help="resolution the OpenVINO IR is frozen to"
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument(
        "--prefix",
        type=int,
        default=None,
        help="operating point; default: the checkpoint's full n_channels",
    )
    parser.add_argument(
        "--target-device",
        default="NPU",
        choices=["NPU", "CPU", "ANY"],
        help="placement policy for the deployed graph, not this machine",
    )
    parser.add_argument("--ptq-preset", default="performance", choices=["performance", "mixed"])
    parser.add_argument("--bias-correction", default="fast", choices=["fast", "accurate", "none"])
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="torch device for the frozen decoder in the RD baseline; CPU is allowed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import torch

    from src.compressors.frappe.harness.checkpoints import load_checkpoint
    from src.compressors.frappe.harness.deployment_package import build_package

    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device} but CUDA is not available; refusing to fall back")

    loaded = load_checkpoint(args.checkpoint, "cpu")
    try:
        manifest = build_package(
            loaded.model,
            args.checkpoint,
            args.dataset_root,
            args.output_dir,
            calibration_split=args.calibration_split,
            calibration_images=args.calibration_images,
            verify_split=args.verify_split,
            verify_images=args.verify_images,
            height=args.height,
            width=args.width,
            prefix=args.prefix,
            target_device=args.target_device,
            ptq_preset=args.ptq_preset,
            bias_correction=args.bias_correction,
            decode_device=args.device,
            checkpoint_iteration=loaded.iteration,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    numbers = manifest["rd_baseline"]["conditions"]
    print(f"packaged {args.checkpoint} (prefix {manifest['prefix']}) into {args.output_dir}")
    for name, point in numbers.items():
        print(
            f"  {name}: {point['psnr_db']:.4f} dB @ {point['bpp']:.4f} bpp "
            f"({point['bytes_total']} bytes)"
        )
    delta = manifest["rd_baseline"]["deltas_vs_fp32"]["int8"]
    print(f"  int8 vs fp32: {delta['d_psnr_db']:+.4f} dB, {delta['d_bpp']:+.5f} bpp")


if __name__ == "__main__":
    main()
