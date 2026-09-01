#!/usr/bin/env python3
"""Export an anonymous local ImageFolder sample reconstructed by FRAPPE.

The script intentionally addresses a sample by its split-local index rather
than retaining or emitting an input filename.  It loads one of the saved
progressive decoder snapshots, applies the same integer latent quantization as
the validator, and writes a PNG reconstruction.  An optional side-by-side
reference is useful for visual inspection, but is not required for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import datasets
import PIL.Image
import PIL.ImageDraw
import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image

# Make ``python tools/export_local_reconstruction.py`` work from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.model import MergedAutoencoder
from src.compressors.frappe.quantize import srgb_to_linear


def load_snapshot(checkpoint: dict, channels: int, device: str) -> MergedAutoencoder:
    """Materialize one completed progressive snapshot on ``device``."""
    snapshots = checkpoint["merged_decoder_weights"]
    if not 1 <= channels <= len(snapshots):
        raise ValueError(
            f"--channels must be in 1..{len(snapshots)}; received {channels}"
        )
    model = MergedAutoencoder(checkpoint["config"], channels).to(device)
    snapshot = snapshots[channels - 1]
    for scale, state_dict in enumerate(snapshot["encoder_weights"]):
        model.encoders[scale].load_state_dict(state_dict)
    model.decoder.load_state_dict(snapshot["decoder_weights"])
    return model.eval()


def validated_image(image: PIL.Image.Image, config) -> PIL.Image.Image:
    """Apply the validator's deterministic resize and patch-size checks."""
    height = getattr(config, "validation_height", None)
    width = getattr(config, "validation_width", None)
    if (height is None) != (width is None):
        raise ValueError("checkpoint has only one validation dimension configured")
    image = image.convert("RGB")
    if height is not None:
        image = image.resize((width, height), PIL.Image.Resampling.BICUBIC)

    max_patch = max(config.ps)
    if image.width % max_patch or image.height % max_patch:
        raise ValueError(
            f"image size {image.width}x{image.height} must divide by max patch "
            f"size {max_patch}"
        )
    return image


def reconstruct(model: MergedAutoencoder, image: PIL.Image.Image, config, device: str):
    """Return normalized reference/reconstruction tensors and their PSNR."""
    reference = pil_to_tensor(image).to(torch.float32).unsqueeze(0).to(device)
    reference = reference / 127.5 - 1.0
    model_input = srgb_to_linear(reference) if config.linear_input else reference
    with torch.inference_mode():
        latents = model.encode(model_input)
        latents = [latent.round().clamp(-127, 127).to(torch.int8) for latent in latents]
        reconstructed = model.decode(latents).clamp(-1, 1)
    psnr = -10.0 * math.log10(F.mse_loss(reference, reconstructed).item())
    return reference, reconstructed, psnr


def select_median_psnr_index(scores: list[float]) -> tuple[int, float]:
    """Choose the lowest-index image whose PSNR is nearest the median PSNR."""
    if not scores:
        raise ValueError("cannot select a representative image from an empty split")
    median = statistics.median(scores)
    index = min(range(len(scores)), key=lambda i: (abs(scores[i] - median), i))
    return index, median


def find_representative(dataset, model: MergedAutoencoder, config, device: str) -> tuple[int, float]:
    """Score a split and return its median-quality image without filenames."""
    scores = []
    for index, sample in enumerate(dataset):
        image = validated_image(sample["image"], config)
        _, _, psnr = reconstruct(model, image, config, device)
        scores.append(psnr)
        if (index + 1) % 500 == 0 or index + 1 == dataset.num_rows:
            print(f"scored {index + 1}/{dataset.num_rows} images", flush=True)
    return select_median_psnr_index(scores)


def save_comparison(reference: PIL.Image.Image, reconstruction: PIL.Image.Image,
                    path: Path, psnr: float, channels: int) -> None:
    """Write a labelled, non-identifying visual comparison PNG."""
    label_height = 30
    canvas = PIL.Image.new("RGB", (reference.width * 2, reference.height + label_height), "white")
    canvas.paste(reference, (0, label_height))
    canvas.paste(reconstruction, (reference.width, label_height))
    draw = PIL.ImageDraw.Draw(canvas)
    draw.text((6, 8), "reference", fill="black")
    draw.text((reference.width + 6, 8), f"FRAPPE reconstruction ({channels}ch, {psnr:.2f} dB)", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="validation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int, default=None,
                        help="zero-based index within the requested split")
    selection.add_argument("--representative", action="store_true",
                        help="select the image closest to the split's median per-image PSNR")
    parser.add_argument("--channels", type=int, default=None,
                        help="completed progressive snapshot; default is the final one")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path,
                        help="PNG path for the reconstructed image")
    parser.add_argument("--comparison-output", type=Path, default=None,
                        help="optional side-by-side reference/reconstruction PNG")
    parser.add_argument("--metadata-output", type=Path, default=None,
                        help="optional JSON metadata path; contains no source filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = args.channels or len(checkpoint["merged_decoder_weights"])
    dataset = datasets.load_dataset(args.dataset_root, split=args.split)
    model = load_snapshot(checkpoint, channels, args.device)
    selection = "index"
    median_psnr = None
    if args.representative:
        args.index, median_psnr = find_representative(
            dataset, model, checkpoint["config"], args.device
        )
        selection = "median_per_image_psnr"
    elif args.index is None:
        args.index = 0
    if not 0 <= args.index < dataset.num_rows:
        raise IndexError(f"--index must be in 0..{dataset.num_rows - 1}; received {args.index}")

    image = validated_image(dataset[args.index]["image"], checkpoint["config"])
    reference, reconstruction, psnr = reconstruct(model, image, checkpoint["config"], args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image((reconstruction[0].cpu() + 1.0) / 2.0).save(args.output, format="PNG")
    if args.comparison_output is not None:
        save_comparison(
            to_pil_image((reference[0].cpu() + 1.0) / 2.0),
            to_pil_image((reconstruction[0].cpu() + 1.0) / 2.0),
            args.comparison_output,
            psnr,
            channels,
        )
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps({
            "checkpoint": args.checkpoint.name,
            "split": args.split,
            "sample_index": args.index,
            "selection": selection,
            "median_PSNR_dB": median_psnr,
            "channels": channels,
            "width": image.width,
            "height": image.height,
            "PSNR_dB": psnr,
        }, indent=2) + "\n")

    print(f"saved reconstruction: {args.output}")
    if args.comparison_output is not None:
        print(f"saved comparison: {args.comparison_output}")
    print(
        f"{args.split} sample index={args.index}, selection={selection}, "
        f"{channels}ch, PSNR={psnr:.3f} dB"
    )


if __name__ == "__main__":
    main()
