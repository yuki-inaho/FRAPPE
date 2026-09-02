#!/usr/bin/env python3
"""Evaluate the released FRAPPE weights on a local anonymous split.

The numbers shipped in ``results/`` were measured on Kodak.  To compare a locally
trained model against the released one, the released one has to be run on the
same images, with the same rate convention: real JPEG-LS bitstreams and
``compression_ratio = 24 / bpp``.

The released checkpoint stores one decoder snapshot per channel count
(``merged.{n}.decoder...``), so each prefix is a separate model that has to be
built and loaded in turn -- which is exactly the storage cost the joint-prefix
reformulation removes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness.bitstream import measure_rate  # noqa: E402
from src.compressors.frappe.harness.data import default_dataset_root  # noqa: E402
from src.compressors.frappe.model import load_from_hub, load_progressive_model  # noqa: E402
from src.compressors.frappe.quantize import srgb_to_linear  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                        help="anonymous ImageFolder root; defaults to $FRAPPE_DATASET_ROOT")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--prefixes", type=int, nargs="+", default=None,
                        help="channel counts to evaluate; default: every trained prefix")
    parser.add_argument("--repo-id", default="danjacobellis/FRAPPE")
    parser.add_argument("--subdir", default="FRAPPE")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    config, weights, n_trained = load_from_hub(args.repo_id, args.subdir)
    prefixes = args.prefixes or list(range(1, n_trained + 1))
    print(f"released weights: {n_trained} trained channels, ps={config.ps}", flush=True)

    files = sorted((args.dataset_root / args.split).glob("image_????????.png"))[:args.images]
    if not files:
        raise SystemExit(f"no anonymous PNG images under {args.dataset_root / args.split}")
    images = []
    for path in files:
        with Image.open(path) as handle:
            handle.load()
            array = np.array(handle.convert("RGB"), dtype=np.uint8)
        images.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
                      .to(device=device, dtype=torch.float32) / 127.5 - 1.0)
    max_ps = max(config.ps)
    height, width = images[0].shape[2], images[0].shape[3]
    if height % max_ps or width % max_ps:
        raise SystemExit(f"{width}x{height} is not a multiple of the largest patch size {max_ps}")
    print(f"{len(images)} images from {args.split}, {width}x{height}", flush=True)

    started = time.time()
    curve = []
    for n in prefixes:
        model = load_progressive_model(weights, config, n, device)
        total_mse, total_bytes = 0.0, 0
        for x in images:
            x_in = srgb_to_linear(x) if config.linear_input else x
            latents = [z.round().clamp(-127, 127).to(torch.int8) for z in model.encode(x_in)]
            recon = model.decode([z.to(torch.float) for z in latents]).clamp(-1, 1)
            total_mse += F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
            total_bytes += measure_rate(latents, height * width)[0]
        mse = total_mse / len(images)
        bpp = total_bytes * 8 / (len(images) * height * width)
        point = {"channels": n, "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                 "bpp": bpp, "compression_ratio": 24.0 / bpp}
        curve.append(point)
        print(f"  n={n:2d}  PSNR={point['psnr_db']:6.2f} dB  bpp={bpp:7.4f}"
              f"  CR={point['compression_ratio']:8.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    report = {"repo_id": args.repo_id, "split": args.split, "images": len(images),
              "image_size": [width, height], "ps": list(config.ps), "curve": curve,
              "seconds": time.time() - started}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
