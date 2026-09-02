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

import io
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import AnonymousImageFolder

PROXY_CRITERIA = ("l1", "l2", "activation", "taylor", "fisher", "random")


def load_images(root: Path, split: str, count: int, device: str) -> list[torch.Tensor]:
    """The first ``count`` images of a split, one tensor each.

    A thin wrapper over :class:`AnonymousImageFolder` kept because the pruning
    search indexes the same images thousands of times and must not re-decode a
    PNG on every candidate subset.
    """
    folder = AnonymousImageFolder(root, split)
    return [folder.signed(index, device) for index in range(min(count, len(folder)))]


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
        for group_index, (_ps, start, end) in enumerate(self.model.scale_groups):
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


def proxy_scores(model, images, codes_cache) -> dict[str, np.ndarray]:
    """Per-group importance under every cheap criterion, before rate normalisation."""
    n = model.n_channels
    scores = {name: np.zeros(n) for name in PROXY_CRITERIA}

    channel_of_group = []
    for group_index, (_ps, start, end) in enumerate(model.scale_groups):
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


