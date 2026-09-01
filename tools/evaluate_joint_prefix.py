#!/usr/bin/env python3
"""Evaluate a joint-prefix FRAPPE checkpoint on a full local split.

Every number here comes from the deployment path: true int8 codes produced by
``integer_codes`` and real JPEG-LS bitstream lengths, never a rate proxy.  The
report contains the whole prefix rate-distortion ladder plus the monotonicity
diagnostics the theory note asks for ("各 prefix の RD curve, monotonicity
violation 数, 限界改善").
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from argparse import Namespace
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Executing ``python tools/evaluate_joint_prefix.py`` puts ``tools/`` on
# sys.path rather than the repository root, so make the documented invocation
# self-contained.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.prefix import JointPrefixFRAPPE


def load_checkpoint(path: Path, device: str) -> tuple[JointPrefixFRAPPE, Namespace, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    config = Namespace(**state["config"])
    model = JointPrefixFRAPPE(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config, state


def jpegls_bytes(codes, n_channels, groups) -> int:
    import pillow_jpls  # noqa: F401
    from torchvision.transforms.v2.functional import to_pil_image

    total = 0
    remaining = n_channels
    for code, (_, start, end) in zip(codes, groups):
        if remaining <= 0:
            break
        width = min(end - start, remaining)
        plane = code[0, :width]
        flat = plane.reshape(plane.shape[0] * plane.shape[1], plane.shape[2])
        buffer = io.BytesIO()
        to_pil_image((flat.to(torch.long) + 127).to(torch.uint8)).save(buffer, format="JPEG-LS")
        total += len(buffer.getbuffer())
        remaining -= width
    return total


def to_png(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.clamp(-1, 1) / 2 + 0.5) * 255).round().to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/workspace/data/frappe_rgb_800x608/imagefolder"))
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--images", type=int, default=None, help="default: the whole split")
    parser.add_argument("--prefixes", type=int, nargs="+", default=None,
                        help="default: every prefix 1..N")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--export-reconstruction", type=Path, default=None,
                        help="write an original/reconstruction pair for one image")
    parser.add_argument("--export-index", type=int, default=0)
    parser.add_argument("--export-prefix", type=int, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config, state = load_checkpoint(args.checkpoint, device)
    prefixes = args.prefixes or list(range(1, model.n_channels + 1))
    started = time.time()
    print(f"checkpoint iteration={state.get('iteration')} channels={model.n_channels} "
          f"decoder_ch={model.total_decoder_channels}", flush=True)

    report: dict[str, object] = {"checkpoint": str(args.checkpoint),
                                 "iteration": state.get("iteration"),
                                 "ps": list(config.ps), "splits": {}}
    for split in args.splits:
        files = sorted((args.dataset_root / split).glob("image_????????.png"))
        if args.images:
            files = files[:args.images]
        if not files:
            raise SystemExit(f"no anonymous PNG images under {args.dataset_root / split}")
        totals = {n: {"mse": 0.0, "bytes": 0} for n in prefixes}
        # Accumulated per image rather than taken from the last one, so a split
        # whose images differ in size is still normalised correctly.
        pixels = 0
        for index, path in enumerate(files):
            with Image.open(path) as handle:
                handle.load()
                array = np.array(handle.convert("RGB"), dtype=np.uint8)
            x = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
                 .to(device=device, dtype=torch.float32) / 127.5 - 1.0)
            pixels += x.shape[2] * x.shape[3]
            codes = model.integer_codes(x)
            y = model.adapt([code.to(torch.float) for code in codes])
            for n in prefixes:
                recon = model.decode(y, n).clamp(-1, 1)
                totals[n]["mse"] += F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
                totals[n]["bytes"] += jpegls_bytes(codes, n, model.scale_groups)
            if (index + 1) % 200 == 0 or index + 1 == len(files):
                print(f"  {split}: {index + 1}/{len(files)}", flush=True)

        count = len(files)
        curve = []
        for n in prefixes:
            mse = totals[n]["mse"] / count
            bpp = totals[n]["bytes"] * 8 / pixels
            curve.append({"channels": n, "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                          "bpp": bpp, "compression_ratio": 24.0 / bpp if bpp else None})
        psnrs = [point["psnr_db"] for point in curve]
        violations = sum(1 for a, b in zip(psnrs[:-1], psnrs[1:]) if b < a)
        gains = [b - a for a, b in zip(psnrs[:-1], psnrs[1:])]
        report["splits"][split] = {
            "images": count, "curve": curve,
            "final_psnr_db": psnrs[-1], "final_bpp": curve[-1]["bpp"],
            "final_compression_ratio": curve[-1]["compression_ratio"],
            "monotonicity_violations": violations,
            "max_marginal_gain_db": max(gains) if gains else None,
        }
        print(f"\n  {split}: {count} images")
        for point in curve:
            print(f"    n={point['channels']:2d}  PSNR={point['psnr_db']:6.2f} dB"
                  f"  bpp={point['bpp']:7.4f}  CR={point['compression_ratio']:8.2f}")
        print(f"    monotonicity violations: {violations}/{len(psnrs) - 1}", flush=True)

    if args.export_reconstruction:
        files = sorted((args.dataset_root / args.splits[0]).glob("image_????????.png"))
        with Image.open(files[args.export_index]) as handle:
            handle.load()
            array = np.array(handle.convert("RGB"), dtype=np.uint8)
        x = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
             .to(device=device, dtype=torch.float32) / 127.5 - 1.0)
        n = args.export_prefix or model.n_channels
        codes = model.integer_codes(x)
        recon = model.decode(model.adapt([c.to(torch.float) for c in codes]), n).clamp(-1, 1)
        mse = F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
        pair = Image.new("RGB", (array.shape[1] * 2 + 8, array.shape[0]), (16, 16, 16))
        pair.paste(to_png(x[0]), (0, 0))
        pair.paste(to_png(recon[0]), (array.shape[1] + 8, 0))
        args.export_reconstruction.parent.mkdir(parents=True, exist_ok=True)
        pair.save(args.export_reconstruction)
        sidecar = args.export_reconstruction.with_suffix(".json")
        sidecar.write_text(json.dumps({
            "split": args.splits[0], "anonymous_index": args.export_index, "prefix": n,
            "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
            "bytes": jpegls_bytes(codes, n, model.scale_groups),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.export_reconstruction} (left: original, right: reconstruction)")

    report["seconds"] = time.time() - started
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
