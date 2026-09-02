#!/usr/bin/env python3
"""Render what a codec actually produces, next to a reference codec at the same rate.

A rate-distortion table says a model wins by some number of decibels; it does not
say what the loss looks like. This renders one anonymous image through one or
more checkpoints and, for each, through a standard codec tuned to *the same
measured bitrate*, so the comparison is like for like rather than
quality-setting for quality-setting.

Three things are written per model: the reconstruction, an amplified absolute
error map (a 40 dB residual is invisible at unit gain), and a labelled strip
that puts the original, every model, and every rate-matched reference side by
side. A zoomed strip is written too, because at high compression the interesting
failures are in texture and edges, which a full-frame view at print size hides.

Reference codecs are rate-matched by bisecting their quality control on the
measured file size, not by assuming a setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import AnonymousImageFolder
from src.compressors.frappe.harness.bitstream import measure_rate
from src.compressors.frappe.harness.checkpoints import load_checkpoint
from src.compressors.frappe.harness.cli import (
    add_dataset_arguments,
    add_device_argument,
    resolve_device,
)
from src.compressors.frappe.harness.codecs import REFERENCE_BOUNDS, match_rate


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.clamp(-1, 1) / 2 + 0.5) * 255).round().to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy())


def psnr_between(reference: Image.Image, candidate: Image.Image) -> float:
    a = np.asarray(reference, dtype=np.float32) / 255.0
    b = np.asarray(candidate, dtype=np.float32) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


def label_strip(panels: list[tuple[str, Image.Image]], pad: int = 8,
                bar: int = 26) -> Image.Image:
    width, height = panels[0][1].size
    canvas = Image.new("RGB", (len(panels) * width + (len(panels) + 1) * pad,
                               height + bar + 2 * pad), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (caption, panel) in enumerate(panels):
        x = pad + index * (width + pad)
        canvas.paste(panel, (x, pad + bar))
        draw.text((x + 2, pad + 6), caption, fill=(235, 235, 235))
    return canvas


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--reference-codecs", nargs="+", default=["avif"],
                        choices=list(REFERENCE_BOUNDS))
    parser.add_argument("--zoom", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="crop for the detail strip; default is a centred quarter")
    parser.add_argument("--zoom-scale", type=int, default=2)
    parser.add_argument("--error-gain", type=float, default=8.0)
    add_dataset_arguments(parser, images=None)
    add_device_argument(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = args.labels or [path.parent.parent.name for path in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit("--labels must have one entry per checkpoint")
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folder = AnonymousImageFolder(args.dataset_root, args.split)
    original = folder.pil(args.index)
    array = np.array(original, dtype=np.uint8)
    x = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
         .to(device=device, dtype=torch.float32) / 127.5 - 1.0)
    pixels = x.shape[2] * x.shape[3]
    original.save(args.output_dir / "original.png")

    panels = [("original", original)]
    report = {"split": args.split, "anonymous_index": args.index,
              "size": list(original.size), "models": [], "references": []}

    for label, checkpoint in zip(labels, args.checkpoints):
        model = load_checkpoint(checkpoint, device).model
        codes = model.integer_codes(x)
        recon = model.decode(model.adapt([c.to(torch.float) for c in codes]),
                             model.n_channels).clamp(-1, 1)
        payload_bytes, bpp = measure_rate(codes, pixels)
        image = to_image(recon[0])
        image.save(args.output_dir / f"{label}_reconstruction.png")
        error = (x - recon).abs().mean(1, keepdim=True) * args.error_gain
        Image.fromarray((error[0, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()).save(
            args.output_dir / f"{label}_error_x{int(args.error_gain)}.png")
        value = -10.0 * math.log10(max(
            F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item(), 1e-12))
        caption = f"{label}  {value:.2f} dB  {bpp:.3f} bpp  CR {24 / bpp:.1f}"
        panels.append((caption, image))
        report["models"].append({"label": label, "checkpoint": str(checkpoint),
                                 "channels": model.n_channels, "psnr_db": value,
                                 "bytes": payload_bytes, "bpp": bpp,
                                 "compression_ratio": 24 / bpp})
        print(f"  {caption}", flush=True)

        for codec in args.reference_codecs:
            decoded, size, setting = match_rate(original, codec, payload_bytes)
            reference_bpp = size * 8 / pixels
            reference_psnr = psnr_between(original, decoded)
            decoded.save(args.output_dir / f"{codec}_at_{label}_rate.png")
            panels.append((f"{codec} @ same rate  {reference_psnr:.2f} dB  "
                           f"{reference_bpp:.3f} bpp", decoded))
            report["references"].append({
                "codec": codec, "matched_to": label, "setting": setting,
                "psnr_db": reference_psnr, "bytes": size, "bpp": reference_bpp,
                "compression_ratio": 24 / reference_bpp})
            print(f"    {codec} bisected to setting {setting}: {reference_psnr:.2f} dB, "
                  f"{reference_bpp:.3f} bpp ({size} B vs the model's {payload_bytes} B)",
                  flush=True)
        del model
        torch.cuda.empty_cache()

    label_strip(panels).save(args.output_dir / "comparison_full.png")

    width, height = original.size
    box = tuple(args.zoom) if args.zoom else (width // 3, height // 3, width // 3, height // 3)
    left, top, crop_w, crop_h = box
    zoomed = [(caption, panel.crop((left, top, left + crop_w, top + crop_h)).resize(
        (crop_w * args.zoom_scale, crop_h * args.zoom_scale), Image.Resampling.NEAREST))
        for caption, panel in panels]
    label_strip(zoomed).save(args.output_dir / "comparison_zoom.png")
    report["zoom_box"] = list(box)

    (args.output_dir / "samples.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output_dir}/comparison_full.png, comparison_zoom.png "
          f"and per-model images")


if __name__ == "__main__":
    main()
