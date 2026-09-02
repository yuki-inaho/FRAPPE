#!/usr/bin/env python3
"""One hybrid round trip: OpenVINO encoder, explicit JPEG-LS, CUDA decoder.

The pipeline the NPU PC will run: the packaged INT8 encoder graph on the
device the caller names, every plane through the JPEG-LS backend the caller
names (nothing else is used), and the packaged decoder under
``CUDAExecutionProvider``. The report records what actually executed, never a
fallback; if any requested device, provider or backend is unavailable the
command fails instead of downgrading.
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
        "--artifact-dir",
        type=Path,
        required=True,
        help="a packaged deployment directory (see tools/package_npu_int8.py)",
    )
    parser.add_argument(
        "--encoder-device",
        required=True,
        help="OpenVINO device for the encoder, e.g. CPU or NPU; never AUTO",
    )
    parser.add_argument(
        "--entropy-backend",
        required=True,
        choices=["pillow", "charls-native"],
        help="JPEG-LS writer; the named backend is the only one used",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--index", type=int, default=0, help="first anonymous image index to measure"
    )
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from src.compressors.frappe.experiment import atomic_json_dump
    from src.compressors.frappe.harness.hybrid_runtime import run_roundtrip

    args = parse_args(argv)
    report = run_roundtrip(
        args.artifact_dir,
        args.dataset_root,
        args.split,
        args.index,
        args.images,
        encoder_device=args.encoder_device,
        entropy_backend=args.entropy_backend,
    )
    atomic_json_dump(report, args.report)
    print(f"wrote {args.report}")
    print(
        f"  encoder {report['encoder']['requested']} -> "
        f"{report['encoder']['execution_devices']}, "
        f"decoder {report['decoder']['provider']}, backend {report['entropy_backend']}"
    )
    print(
        f"  {report['images']} images: {report['psnr_db']:.4f} dB @ "
        f"{report['bpp']:.4f} bpp payload-only "
        f"({report['bpp_with_length_prefix']:.4f} with prefixes), "
        f"JPEG-LS roundtrip exact = {report['jpegls_roundtrip_exact']}"
    )


if __name__ == "__main__":
    main()
