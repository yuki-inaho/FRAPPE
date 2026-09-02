#!/usr/bin/env python3
"""Render the deployed codec's side-by-side comparison image for inspection.

The round trip measures; this tool shows. It runs the same deployment path as
``tools/roundtrip_openvino.py`` on one image -- encoder graph, JPEG-LS through
CharLS, decoder graph -- and writes a single PNG placing the original and the
reconstruction side by side, so a human can judge what the measured PSNR means.

The two panels may be rotated before placement, which is for photographs that
were captured sideways: rotating both panels the same way keeps the comparison
honest while making the subject upright. The gap between the panels is white
and is not part of either image.

Like every deployment tool, images are addressed by split and integer index --
never by filename -- so private data does not reach reports or logs. The
written PNG is the one deliberate exception: it is the tool's product, and its
path is given by the caller.

Encoder and decoder accept separate graph paths, which is how an experiment
mixes graphs -- an int8 encoder against the fp32 decoder, say. Each device is
named, never inferred; see ``tools/roundtrip_openvino.py`` for the reasoning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from PIL import Image

from src.compressors.frappe.harness import (
    AnonymousImageFolder,
    BitstreamConvention,
    decode_planes,
    default_dataset_root,
    encode_planes,
    psnr_from_mse,
)
from src.compressors.frappe.harness.profiling import as_torch_planes
from src.compressors.frappe.openvino_runtime import FrappeDecoder, FrappeEncoder

ROTATIONS = {
    "none": None,
    "cw90": Image.Transpose.ROTATE_270,
    "ccw90": Image.Transpose.ROTATE_90,
    "180": Image.Transpose.ROTATE_180,
}

#: Device presets. CPU is the settled deployment: the encoder is exact there and
#: the decoder is far faster on the GPU than on the NPU. NPU is the experimental
#: placement, kept selectable so its output can be inspected on the same terms --
#: the NPU encoder cannot reproduce the reference bitstream (no FP32), so its
#: bytes are its own and the printed PSNR says by how much.
MODES = {
    "cpu": ("CPU", "GPU"),
    "npu": ("NPU", "NPU"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--encoder-graph",
        type=Path,
        required=True,
        help="path to the encoder ONNX graph (any naming; the int8 graph works here)",
    )
    parser.add_argument(
        "--decoder-stem",
        type=Path,
        required=True,
        help="path stem of the decoder graph, without _decoder.onnx",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="cpu",
        help="device preset: 'cpu' encodes on the CPU and decodes on the GPU "
        "(the settled deployment); 'npu' runs both stages on the NPU. "
        "Explicit --encoder-device/--decoder-device override the preset",
    )
    parser.add_argument(
        "--encoder-device",
        default=None,
        help="a concrete OpenVINO device for the encoder (default: from --mode)",
    )
    parser.add_argument(
        "--decoder-device",
        default=None,
        help="a concrete OpenVINO device for the decoder (default: from --mode)",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="image index in the split; logs record this, never a filename",
    )
    parser.add_argument(
        "--rotate",
        choices=sorted(ROTATIONS),
        default="none",
        help="rotate both panels the same way before placement",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=12,
        help="white gap between the panels, in pixels (default: 12)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="where to write the side-by-side PNG",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    import openvino

    core = openvino.Core()
    folder = AnonymousImageFolder(args.dataset_root, args.split)
    if not 0 <= args.index < len(folder):
        raise SystemExit(
            f"{args.split!r} has {len(folder)} images, index {args.index} does not fit"
        )
    image = folder.pixels(args.index)
    _batch, channels, height, width = image.shape

    preset_encoder, preset_decoder = MODES[args.mode]
    encoder_device = args.encoder_device or preset_encoder
    decoder_device = args.decoder_device or preset_decoder
    encoder = FrappeEncoder(args.encoder_graph, encoder_device, height, width, core=core)
    decoder = FrappeDecoder(args.decoder_stem, decoder_device, encoder.plane_shapes, core=core)

    planes = encoder(image.numpy())
    tensors = as_torch_planes(planes)
    blob = encode_planes(tensors, BitstreamConvention.WITH_LENGTH_PREFIX)
    restored = decode_planes(blob)
    for want, got in zip(tensors, restored):
        if not torch.equal(want, got):
            raise SystemExit(
                "JPEG-LS did not round-trip the planes; at NEAR=0 it is lossless, "
                "so this is a defect and not a tolerance"
            )
    reconstruction = decoder([plane.numpy() for plane in restored])

    difference = (
        image.to(torch.float64) - torch.from_numpy(reconstruction).to(torch.float64)
    ) / 255.0
    psnr = psnr_from_mse(float((difference**2).mean()))
    bpp = len(blob) * 8 / (height * width)
    print(f"image {args.split}[{args.index}]  pinned {width}x{height}  channels {channels}")
    print(f"  encoder {encoder.device} -> {encoder.execution_devices}")
    print(f"  decoder {decoder.device} -> {decoder.execution_devices}")
    print(f"  bytes {len(blob)}  bpp {bpp:.5f}  PSNR {psnr:.3f} dB")

    rotation = ROTATIONS[args.rotate]
    original = Image.fromarray(image[0].permute(1, 2, 0).numpy())
    rebuilt = Image.fromarray(reconstruction[0].transpose(1, 2, 0))
    if rotation is not None:
        original, rebuilt = original.transpose(rotation), rebuilt.transpose(rotation)
    panel_width, panel_height = original.size
    canvas = Image.new("RGB", (panel_width * 2 + args.gap, panel_height), (255, 255, 255))
    canvas.paste(original, (0, 0))
    canvas.paste(rebuilt, (panel_width + args.gap, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
