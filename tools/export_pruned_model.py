#!/usr/bin/env python3
"""Prune a trained joint-prefix FRAPPE model down to a target channel count.

Selection is by default the exact greedy backward elimination of
``tools/prune_latent_channels.py`` -- it decodes and entropy-codes every
candidate rather than trusting a saliency proxy -- and the cheap criteria remain
available for comparison.

The result is a structurally smaller model: fewer analysis filters, fewer
companding parameters, a narrower first decoder convolution.  Before any
fine-tuning it reproduces the original model restricted to the kept channels
bit-exactly, which the tool verifies rather than assumes; fine-tuning is then
pure upside and is done by re-running the trainer on the emitted checkpoint with
``--ps <pruned schedule> --resume <checkpoint> --resume_model_only``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.experiment import atomic_json_dump, atomic_torch_save  # noqa: E402
from src.compressors.frappe.prefix import prune_channels  # noqa: E402
from tools.evaluate_joint_prefix import load_checkpoint  # noqa: E402
from tools.prune_latent_channels import (  # noqa: E402
    PROXY_CRITERIA, RateMeter, channel_rates, greedy_frontier, load_images, measure,
    proxy_scores, select_by_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--target-channels", type=int, default=None)
    parser.add_argument("--keep", type=int, nargs="+", default=None,
                        help="explicit 1-based channel list; overrides --target-channels")
    parser.add_argument("--criterion", default="oracle",
                        choices=["oracle", *PROXY_CRITERIA])
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/workspace/data/frappe_rgb_800x608/imagefolder"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.keep is None and args.target_channels is None:
        raise SystemExit("give either --target-channels or --keep")

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config, state = load_checkpoint(args.checkpoint, device)
    images = load_images(args.dataset_root, args.split, args.images, device)
    with torch.no_grad():
        codes_cache = [model.integer_codes(x) for x in images]
    meter = RateMeter(model)
    started = time.time()
    print(f"source: {args.checkpoint} (iteration {state.get('iteration')}), "
          f"{model.n_channels} channels", flush=True)

    frontier = None
    if args.keep is not None:
        kept = sorted({int(channel) for channel in args.keep})
        selection = "explicit"
    elif args.criterion == "oracle":
        print("  exact greedy backward elimination:", flush=True)
        frontier = greedy_frontier(model, images, codes_cache, meter, verbose=True)
        point = next(p for p in frontier if p["count"] == args.target_channels)
        kept, selection = point["channels"], "oracle"
    else:
        rates = channel_rates(model, images, codes_cache, meter)
        scores = proxy_scores(model, images, codes_cache, device)[args.criterion]
        order = sorted(range(model.n_channels), key=lambda i: -(scores[i] / rates[i]))
        kept = sorted(index + 1 for index in order[:args.target_channels])
        selection = args.criterion
    print(f"\n  keeping {len(kept)} channels by {selection}: {kept}", flush=True)

    before_psnr, before_bpp = measure(model, images, codes_cache, meter, kept)
    pruned = prune_channels(model, kept, config).eval()

    # Equivalence is a claim about the pruning code, so it is checked, not asserted.
    #
    # The gate is on the two things that are exactly reproducible: the integer
    # codes the encoder emits, and the resulting quality.  The float outputs are
    # only reproducible to float32 precision -- the first convolution sums a
    # different number of input channels in the pruned model, and although that
    # difference is ~1e-6, the LayerNorm inside each residual block divides by a
    # small standard deviation and amplifies it downstream.  Gating on the raw
    # output difference would therefore fail on arithmetic, not on correctness.
    with torch.no_grad():
        worst = 0.0
        codes_match = True
        for x, codes in zip(images, codes_cache):
            surviving = [codes[group][:, local]
                         for group, (ps, start, end) in enumerate(model.scale_groups)
                         for local in range(end - start) if start + local + 1 in kept]
            pruned_codes = pruned.integer_codes(x)
            emitted = [pruned_codes[group][:, local]
                       for group, (ps, start, end) in enumerate(pruned.scale_groups)
                       for local in range(end - start)]
            codes_match &= len(surviving) == len(emitted) and all(
                torch.equal(a, b) for a, b in zip(surviving, emitted))
            reference = model.decode_subset(model.adapt([c.to(torch.float) for c in codes]), kept)
            candidate = pruned.decode(pruned.adapt([c.to(torch.float) for c in pruned_codes]),
                                      pruned.n_channels)
            worst = max(worst, (reference - candidate).abs().max().item())
    after_psnr, after_bpp = measure(pruned, images, [pruned.integer_codes(x) for x in images],
                                    RateMeter(pruned), list(range(1, pruned.n_channels + 1)))
    print(f"  integer codes bit-identical: {codes_match}")
    print(f"  PSNR  masked {before_psnr:.4f} dB  vs pruned {after_psnr:.4f} dB "
          f"(difference {after_psnr - before_psnr:+.4f} dB)")
    print(f"  bpp   masked {before_bpp:.5f}     vs pruned {after_bpp:.5f}")
    print(f"  float32 output difference (informational): max {worst:.3e}")
    if not codes_match:
        raise SystemExit("pruned encoder does not emit the same integer codes")
    if abs(after_psnr - before_psnr) > 0.01:
        raise SystemExit("pruned model does not reproduce the masked original")

    source_parameters = sum(p.numel() for p in model.parameters())
    pruned_parameters = sum(p.numel() for p in pruned.parameters())
    analysis_before = sum(p.numel() for p in model.analysis.parameters())
    analysis_after = sum(p.numel() for p in pruned.analysis.parameters())

    pruned_config = argparse.Namespace(**vars(config))
    pruned_config.ps = list(pruned.ps)
    payload = {"iteration": state.get("iteration"), "model": pruned.state_dict(),
               "config": vars(pruned_config), "pruned_from": str(args.checkpoint),
               "kept_channels": kept, "selection": selection}
    atomic_torch_save(payload, args.output_checkpoint)

    report = {
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "selection": selection, "kept_channels": kept,
        "source_ps": list(config.ps), "pruned_ps": list(pruned.ps),
        "channels": {"before": model.n_channels, "after": pruned.n_channels},
        "decoder_input_channels": {"before": model.total_decoder_channels,
                                   "after": pruned.total_decoder_channels},
        "parameters": {"before": source_parameters, "after": pruned_parameters,
                       "analysis_before": analysis_before, "analysis_after": analysis_after},
        "measured": {"psnr_db": after_psnr, "bpp": after_bpp,
                     "compression_ratio": 24.0 / after_bpp if after_bpp else None,
                     "masked_psnr_db": before_psnr, "masked_bpp": before_bpp},
        "integer_codes_identical": codes_match,
        "max_output_difference": worst,
        "frontier": frontier,
        "seconds": time.time() - started,
    }
    print(f"\n  schedule  {list(config.ps)}\n         -> {list(pruned.ps)}")
    print(f"  latent channels        {model.n_channels:>8} -> {pruned.n_channels}")
    print(f"  decoder input channels {model.total_decoder_channels:>8} -> "
          f"{pruned.total_decoder_channels}")
    print(f"  analysis parameters    {analysis_before:>8} -> {analysis_after}")
    print(f"  total parameters       {source_parameters:>8} -> {pruned_parameters}")
    print(f"  measured on {len(images)} {args.split} images: {after_psnr:.2f} dB, "
          f"{after_bpp:.4f} bpp, CR {24.0 / after_bpp:.2f}")
    print(f"  wrote {args.output_checkpoint}")
    if args.report:
        atomic_json_dump(report, args.report)
        print(f"  wrote {args.report}")


if __name__ == "__main__":
    main()
