#!/usr/bin/env python3
"""Quantize a FRAPPE encoder graph to int8 (QDQ) with static calibration.

The deployment question this answers is whether the analysis path is faster at
int8, and by how much. The quantization is ONNX Runtime's static post-training
flow -- calibrate on real images from the anonymous split, wrap every
``Conv`` in QuantizeLinear/DequantizeLinear pairs, leave everything else alone.
QDQ is the format OpenVINO reads as native int8 on the CPU plugin, so the
output of this tool is an ONNX file that compiles wherever the fp32 one does.

NNCF was tried first and is unusable in this environment: its 1.x line compiles
C++ extensions against torch's private API (incompatible with torch 2.11) and
its 2.x/3.x lines pin numpy versions that conflict with the conda environment.
The ORT flow is pure Python and depends on nothing outside the environment.

The quantized encoder changes the planes -- activation quantization rounds a
few hundred companded values per image onto the other side of the uint8 cast --
so its output is a different, still-valid bitstream. Measuring what that does
to rate and PSNR is :mod:`tools.profile_encode_path`'s job, not this tool's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import AnonymousImageFolder
from src.compressors.frappe.harness.data import default_dataset_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx-stem",
        type=Path,
        required=True,
        help="exported graph pair stem; the encoder is quantized",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=32,
        help="how many real images to calibrate against",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the int8 ONNX; default is the encoder "
        "path with _int8 before the extension",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    encoder = (
        Path(str(args.onnx_stem) + "_encoder.onnx")
        if not str(args.onnx_stem).endswith("_encoder.onnx")
        else Path(args.onnx_stem)
    )
    if not encoder.is_file():
        raise SystemExit(f"no encoder graph at {encoder}")
    # The name keeps the _encoder.onnx suffix so FrappeEncoder passes the path
    # through unchanged instead of appending the suffix a second time.
    output = args.output or encoder.with_name(
        encoder.name.replace("_encoder.onnx", "_int8_encoder.onnx")
    )

    folder = AnonymousImageFolder(args.dataset_root, args.split)
    if args.calibration_images > len(folder):
        raise SystemExit(
            f"the {args.split!r} split has {len(folder)} images; "
            f"{args.calibration_images} were requested"
        )

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    class Reader:
        """ORT's calibration reader over real images, bounded to a count."""

        def __init__(self, count: int, input_name: str) -> None:
            self.count, self.input_name = count, input_name
            self.rewind()

        def rewind(self) -> None:
            self.index = 0

        def get_next(self):
            if self.index >= self.count:
                return None
            image = folder.pixels(self.index).numpy()
            self.index += 1
            return {self.input_name: image}

    # One pass through ORT only to learn the input's name -- every FRAPPE
    # encoder names it "image", but the graph is the authority.
    import onnxruntime as ort

    session = ort.InferenceSession(str(encoder), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    quantize_static(
        str(encoder),
        str(output),
        Reader(args.calibration_images, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv"],
    )
    before, after = encoder.stat().st_size, output.stat().st_size
    print(
        f"wrote {output} ({after / 1e6:.1f} MB from {before / 1e6:.1f} MB fp32, "
        f"calibrated on {args.calibration_images} images)"
    )


if __name__ == "__main__":
    main()
