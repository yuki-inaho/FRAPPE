#!/usr/bin/env python3
"""Command line for the rate-normalised latent-channel pruning search.

The formulation, the criteria and the exact greedy frontier live in
:mod:`compressors.frappe.harness.pruning`; this file is the interface to them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness.checkpoints import load_checkpoint  # noqa: E402
from src.compressors.frappe.harness.cli import (  # noqa: E402
    add_dataset_arguments,
    add_device_argument,
    add_output_argument,
    resolve_device,
)
from src.compressors.frappe.harness.pruning import (  # noqa: E402
    PROXY_CRITERIA,
    RateMeter,
    channel_rates,
    greedy_frontier,
    load_images,
    measure,
    proxy_scores,
    select_by_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-compression", type=float, nargs="+",
                        default=[10.0, 20.0, 30.0, 50.0, 80.0],
                        help="compression ratios to select a channel subset for")
    parser.add_argument("--criteria", nargs="+", default=list(PROXY_CRITERIA),
                        choices=list(PROXY_CRITERIA))
    parser.add_argument("--skip-oracle", action="store_true")
    add_dataset_arguments(parser, images=8)
    add_device_argument(parser)
    add_output_argument(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model, config, state = checkpoint.model, checkpoint.config, checkpoint.state
    images = load_images(args.dataset_root, args.split, args.images, device)
    started = time.time()
    with torch.no_grad():
        codes_cache = [model.integer_codes(x) for x in images]
    meter = RateMeter(model)

    print(f"checkpoint iteration={state.get('iteration')}  {model.n_channels} latent channels, "
          f"{len(images)} images\n")
    rates = channel_rates(model, images, codes_cache, meter)
    scores = proxy_scores(model, images, codes_cache, device)

    print("  per-channel cost and importance (importance/bit in parentheses):")
    print(f"    {'ch':>3} {'ps':>3} {'bpp':>8}  " +
          "  ".join(f"{name:>22}" for name in args.criteria))
    group_ps = []
    for ps, start, end in model.scale_groups:
        group_ps.extend([ps] * (end - start))
    for channel in range(model.n_channels):
        cells = []
        for name in args.criteria:
            value = scores[name][channel]
            cells.append(f"{value:9.3e} ({value / rates[channel]:8.2e})")
        print(f"    {channel + 1:>3} {group_ps[channel]:>3} {rates[channel]:8.4f}  " +
              "  ".join(cells))

    report = {
        "checkpoint": str(args.checkpoint), "iteration": state.get("iteration"),
        "images": len(images), "ps": list(config.ps),
        "channel_rates_bpp": rates.tolist(),
        "scores": {name: scores[name].tolist() for name in args.criteria},
        "selections": {}, "oracle": None,
    }

    frontier = None
    if not args.skip_oracle:
        print("\n  exact greedy backward elimination (measured PSNR, measured bitstream):")
        frontier = greedy_frontier(model, images, codes_cache, meter, verbose=True)
        report["oracle"] = frontier

    print("\n  channel subsets selected at each target compression ratio:")
    for target in args.target_compression:
        budget = 24.0 / target
        print(f"\n    target CR {target:g} ({budget:.4f} bpp)")
        entries = {}
        for name in args.criteria:
            kept = select_by_score(scores[name], rates, budget)
            psnr, bpp = measure(model, images, codes_cache, meter, kept)
            entries[name] = {"channels": kept, "psnr_db": psnr, "bpp": bpp,
                             "compression_ratio": 24.0 / bpp}
            print(f"      {name:11s} keep {len(kept):2d}  {bpp:7.4f} bpp"
                  f"  CR {24.0 / bpp:7.2f}  {psnr:6.2f} dB   {kept}")
        prefix = [c for c in range(1, model.n_channels + 1)]
        best_prefix = None
        for n in range(1, model.n_channels + 1):
            psnr, bpp = measure(model, images, codes_cache, meter, prefix[:n])
            if bpp <= budget and (best_prefix is None or psnr > best_prefix["psnr_db"]):
                best_prefix = {"channels": prefix[:n], "psnr_db": psnr, "bpp": bpp,
                               "compression_ratio": 24.0 / bpp}
        if best_prefix:
            entries["prefix"] = best_prefix
            print(f"      {'prefix':11s} keep {len(best_prefix['channels']):2d}  "
                  f"{best_prefix['bpp']:7.4f} bpp  CR {best_prefix['compression_ratio']:7.2f}  "
                  f"{best_prefix['psnr_db']:6.2f} dB")
        if frontier:
            fits = [point for point in frontier if point["bpp"] <= budget]
            if fits:
                best = max(fits, key=lambda point: point["psnr_db"])
                entries["oracle"] = {"channels": best["channels"], "psnr_db": best["psnr_db"],
                                     "bpp": best["bpp"],
                                     "compression_ratio": 24.0 / best["bpp"]}
                print(f"      {'oracle':11s} keep {len(best['channels']):2d}  {best['bpp']:7.4f} bpp"
                      f"  CR {24.0 / best['bpp']:7.2f}  {best['psnr_db']:6.2f} dB   "
                      f"{best['channels']}")
        report["selections"][str(target)] = entries

    if frontier:
        oracle_order = {}
        for point in frontier[1:]:
            oracle_order[point["removed"]] = point["count"]
        ranks = np.array([oracle_order.get(c, 0) for c in range(1, model.n_channels + 1)],
                         dtype=float)
        print("\n  rank correlation with the oracle removal order (Spearman):")
        for name in args.criteria:
            proxy = scores[name] / rates
            correlation = float(np.corrcoef(
                np.argsort(np.argsort(-proxy)), np.argsort(np.argsort(ranks)))[0, 1])
            report.setdefault("rank_correlation", {})[name] = correlation
            print(f"    {name:11s} {correlation:+.3f}")

    report["seconds"] = time.time() - started
    report["jpegls_encodes"] = meter.calls
    print(f"\n  {meter.calls} JPEG-LS encodes, {time.time() - started:.0f} s")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
