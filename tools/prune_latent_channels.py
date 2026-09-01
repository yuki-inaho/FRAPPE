#!/usr/bin/env python3
"""Structured pruning of FRAPPE latent channels, scored per bit rather than per FLOP.

FORMULATION
-----------
A FRAPPE latent channel is a pruning *group* in the sense of Group Fisher (Liu
et al., ICML 2021, arXiv:2108.00708).  Removing latent channel ``i`` removes,
inseparably:

  * one filter row of the analysis convolution of its scale group,
  * that channel's companding parameters ``(sigma_i, gamma_i, beta_i)``,
  * the contiguous block of ``(p_d / p_i)^2`` decoder input channels it expands
    into after spatial adaption -- ``JointPrefixFRAPPE.channel_slices[i]``.

So the coupling that group pruning exists to handle is explicit in the model
rather than something a dependency tracer has to discover.

What changes for a codec is the *cost* side.  Network pruning normalises
importance by FLOPs or parameter count; here the resource being bought is
bitrate.  Channel ``i`` at patch size ``p_i`` emits ``H W / p_i^2`` symbols, so
one ``p=2`` channel costs 256x what one ``p=32`` channel costs.  Ranking by raw
importance would therefore always keep the finest channels, which are the
expensive ones.  Every criterion below is consequently divided by ``R_i``, the
channel's *measured* contribution to the real JPEG-LS bitstream, and selection
is the Lagrangian

    keep channel i  <=>  Delta_D_i / Delta_R_i  >  lambda,

realised as: sort by ``importance_i / R_i`` descending and take channels while
the budget ``sum R_i <= 24 / target_compression`` holds.

CRITERIA
--------
``l1`` / ``l2``   Norm of the analysis filter -- Li et al., ICLR 2017
                  (arXiv:1608.08710).  Data-free.
``activation``    RMS of the channel's adapted latent.  Data, no gradients.
``taylor``        ``|sum_over_group dL/dy * y|`` -- the first-order Taylor
                  estimate of the loss increase from zeroing the group, i.e.
                  Molchanov et al. (arXiv:1611.06440) lifted from single filters
                  to the coupled group.
``fisher``        ``sum_samples (sum_over_group dL/dy * y)^2`` -- the diagonal
                  empirical Fisher of the same group statistic, which is the
                  Group Fisher criterion.  Squaring *after* the group sum is what
                  makes it a group criterion rather than a sum of per-channel
                  ones; it is also why per-sample gradients are taken at batch
                  size one here.
``random``        Control.  A criterion that cannot beat this is not a criterion.
``oracle``        Not a proxy: greedy backward elimination that actually decodes
                  and actually entropy-codes every candidate, removing whichever
                  channel costs the least PSNR per bit saved.  With 21 groups the
                  exact frontier is affordable, so the proxies are scored against
                  ground truth instead of trusted.

PROCESSING FLOW
---------------
The order of operations matters more than the choice of criterion:

  1. Train with a rate objective first.  On a model trained without one, every
     channel spends 5-7 bits per symbol, the exact greedy frontier removes
     channels in strict reverse schedule order, and no non-prefix subset beats a
     prefix -- pruning has nothing to find.  Taylor and Fisher then actively
     mislead, picking non-prefix subsets that collapse by more than 10 dB,
     because the decoder has only ever seen nested prefix masks and a subset with
     holes in it is out of distribution.
  2. After rate-targeted training, the Lagrangian has already driven the channels
     that do not pay for themselves to (near) zero variance.  Those are what
     pruning removes, and removing them costs neither rate (they were already
     ~0 bits) nor quality -- it buys encoder and decoder compute.  This is
     Molchanov's stated objective, resource-efficient inference, rather than a
     rate reduction.
  3. Score, select, and hand the kept set to ``tools/export_pruned_model.py``,
     which builds the structurally smaller model and verifies that its integer
     codes are bit-identical to the masked original before anything is retrained.
  4. Fine-tune the pruned model (``train_joint_prefix.py --resume_model_only``).
     Nothing has to be recovered -- step 3 guarantees parity -- so fine-tuning is
     purely an opportunity to reallocate the freed capacity.

The oracle is the point of the tool.  Every cheap criterion is validated against
it by rank correlation and, more usefully, by the PSNR gap at matched bitrate.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_joint_prefix import load_checkpoint  # noqa: E402

PROXY_CRITERIA = ("l1", "l2", "activation", "taylor", "fisher", "random")


def load_images(root: Path, split: str, count: int, device: str) -> list[torch.Tensor]:
    files = sorted((root / split).glob("image_????????.png"))[:count]
    if not files:
        raise SystemExit(f"no anonymous PNG images under {root / split}")
    images = []
    for path in files:
        with Image.open(path) as handle:
            handle.load()
            array = np.array(handle.convert("RGB"), dtype=np.uint8)
        images.append((torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
                       .to(device=device, dtype=torch.float32) / 127.5 - 1.0))
    return images


class RateMeter:
    """Real JPEG-LS lengths for kept channel subsets, memoised per scale group.

    Dropping a channel only changes the plane of its own scale group, so the
    length of every other group is reused across the whole search.  Without this
    the greedy frontier would re-encode the finest plane thousands of times.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.cache: dict[tuple[int, int, tuple[int, ...]], int] = {}
        self.calls = 0

    def group_bytes(self, image_index: int, group_index: int,
                    plane: torch.Tensor, kept: tuple[int, ...]) -> int:
        key = (image_index, group_index, kept)
        if key not in self.cache:
            import pillow_jpls  # noqa: F401
            from torchvision.transforms.v2.functional import to_pil_image

            selected = plane[list(kept)]
            n, h, w = selected.shape
            flat = selected.reshape(n * h, w)
            buffer = io.BytesIO()
            to_pil_image((flat.to(torch.long) + 127).to(torch.uint8)).save(
                buffer, format="JPEG-LS")
            self.cache[key] = len(buffer.getbuffer())
            self.calls += 1
        return self.cache[key]

    def subset_bytes(self, image_index: int, codes: list[torch.Tensor],
                     channels) -> int:
        kept = {int(c) for c in channels}
        total = 0
        for group_index, (ps, start, end) in enumerate(self.model.scale_groups):
            local = tuple(sorted(c - start for c in range(start, end) if c + 1 in kept))
            if local:
                total += self.group_bytes(image_index, group_index,
                                          codes[group_index][0].cpu(), local)
        return total


@torch.no_grad()
def measure(model, images, codes_cache, meter, channels) -> tuple[float, float]:
    """Actual PSNR and actual bits per pixel for a kept channel subset."""
    kept = sorted({int(c) for c in channels})
    total_mse, total_bytes, pixels = 0.0, 0, 0
    for index, (x, codes) in enumerate(zip(images, codes_cache)):
        y = model.adapt([code.to(torch.float) for code in codes])
        recon = model.decode_subset(y, kept).clamp(-1, 1)
        total_mse += F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
        total_bytes += meter.subset_bytes(index, codes, kept)
        pixels = x.shape[2] * x.shape[3]
    mse = total_mse / len(images)
    return (-10.0 * math.log10(max(mse, 1e-12)),
            total_bytes * 8 / (len(images) * pixels))


def proxy_scores(model, images, codes_cache, device) -> dict[str, np.ndarray]:
    """Per-group importance under every cheap criterion, before rate normalisation."""
    n = model.n_channels
    scores = {name: np.zeros(n) for name in PROXY_CRITERIA}

    channel_of_group = []
    for group_index, (ps, start, end) in enumerate(model.scale_groups):
        for local in range(end - start):
            channel_of_group.append((group_index, local))

    for channel in range(n):
        group_index, local = channel_of_group[channel]
        weight = model.analysis[group_index].weight[local]
        scores["l1"][channel] = weight.abs().sum().item()
        scores["l2"][channel] = weight.pow(2).sum().sqrt().item()
    rng = np.random.default_rng(0)
    scores["random"] = rng.random(n)

    # Taylor and Fisher need gradients of the training distortion with respect to
    # the adapted latent.  Batch size one keeps the Fisher per-sample, which is
    # what the empirical Fisher is defined over.
    model.eval()
    for x, codes in zip(images, codes_cache):
        y = model.adapt([code.to(torch.float) for code in codes]).detach().requires_grad_(True)
        recon = model.decode(y, model.n_channels, masked=True)
        loss = F.mse_loss(recon, x).clamp_min(1e-12).log10()
        gradient, = torch.autograd.grad(loss, y)
        contribution = (gradient * y.detach())
        for channel, (start, end) in enumerate(model.channel_slices):
            block = contribution[:, start:end]
            group_sum = block.sum().item()
            scores["taylor"][channel] += abs(group_sum)
            scores["fisher"][channel] += group_sum ** 2
            scores["activation"][channel] += y.detach()[:, start:end].pow(2).mean().sqrt().item()
    for name in ("taylor", "fisher", "activation"):
        scores[name] /= len(images)
    return scores


def channel_rates(model, images, codes_cache, meter) -> np.ndarray:
    """Marginal bits per pixel attributable to each latent channel."""
    everything = list(range(1, model.n_channels + 1))
    pixels = images[0].shape[2] * images[0].shape[3]
    full = sum(meter.subset_bytes(i, c, everything) for i, c in enumerate(codes_cache))
    rates = np.zeros(model.n_channels)
    for channel in everything:
        without = [c for c in everything if c != channel]
        dropped = sum(meter.subset_bytes(i, c, without) for i, c in enumerate(codes_cache))
        rates[channel - 1] = (full - dropped) * 8 / (len(images) * pixels)
    return np.maximum(rates, 1e-9)


def greedy_frontier(model, images, codes_cache, meter, verbose: bool) -> list[dict]:
    """Exact greedy backward elimination on measured PSNR and measured bitrate."""
    kept = list(range(1, model.n_channels + 1))
    psnr, bpp = measure(model, images, codes_cache, meter, kept)
    frontier = [{"channels": list(kept), "count": len(kept), "psnr_db": psnr, "bpp": bpp}]
    if verbose:
        print(f"    keep {len(kept):2d}  {bpp:7.4f} bpp  {psnr:6.2f} dB   {kept}", flush=True)
    while len(kept) > 1:
        best = None
        for candidate in kept:
            trial = [c for c in kept if c != candidate]
            trial_psnr, trial_bpp = measure(model, images, codes_cache, meter, trial)
            saved = bpp - trial_bpp
            lost = psnr - trial_psnr
            # Cost of removal per bit saved, ordered lexicographically so the
            # free removals come first.  A plain ratio would invert here: when a
            # removal costs no quality (lost <= 0) the ratio is negative and the
            # smallest one belongs to the removal that saves the FEWEST bits,
            # which is exactly backwards. Free removals are therefore ranked
            # among themselves by how much rate they save, and only then are the
            # lossy ones ranked by cost per bit.
            key = (0, -saved) if lost <= 0 else (1, lost / max(saved, 1e-9))
            if best is None or key < best[0]:
                best = (key, candidate, trial_psnr, trial_bpp)
        _, candidate, psnr, bpp = best
        kept = [c for c in kept if c != candidate]
        frontier.append({"channels": list(kept), "count": len(kept),
                         "psnr_db": psnr, "bpp": bpp, "removed": candidate})
        if verbose:
            print(f"    keep {len(kept):2d}  {bpp:7.4f} bpp  {psnr:6.2f} dB"
                  f"   dropped ch{candidate}", flush=True)
    return frontier


def select_by_score(scores: np.ndarray, rates: np.ndarray, budget: float) -> list[int]:
    """Keep the highest score-per-bit channels that fit in ``budget`` bpp."""
    order = np.argsort(-(scores / rates))
    kept, spent = [], 0.0
    for channel in order:
        if spent + rates[channel] <= budget:
            kept.append(int(channel) + 1)
            spent += rates[channel]
    return sorted(kept) or [int(np.argmax(scores / rates)) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/workspace/data/frappe_rgb_800x608/imagefolder"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-compression", type=float, nargs="+",
                        default=[10.0, 20.0, 30.0, 50.0, 80.0],
                        help="compression ratios to select a channel subset for")
    parser.add_argument("--criteria", nargs="+", default=list(PROXY_CRITERIA),
                        choices=list(PROXY_CRITERIA))
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config, state = load_checkpoint(args.checkpoint, device)
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
