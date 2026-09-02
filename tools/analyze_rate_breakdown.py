#!/usr/bin/env python3
"""Where the bits actually go in a trained FRAPPE codec, and what could remove them.

For each scale group this reports four rates in bits per pixel:

``raw``       ``8 * n_ch / p^2`` -- the int8 budget, before any entropy coding.
``jpegls``    what the shipped JPEG-LS coder actually spends.
``order0``    the empirical zeroth-order entropy of the codes.  A static
              per-channel factorised model with an arithmetic coder reaches this.
``residual``  the entropy of the horizontal first difference.  A cheap predictive
              coder with a static model reaches roughly this.

The gap between ``jpegls`` and the two entropy columns is what a better entropy
coder can recover without touching the model.  The size of ``raw`` per group is
what only a rate-distortion-trained compander (or fewer channels) can shrink.

A ``--bit-depth-sweep`` additionally measures, without retraining, what happens
when a group's codes are requantised to fewer levels: it traces the codec's
post-hoc rate-distortion slope and shows which group is worth attacking first.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness import AnonymousImageFolder  # noqa: E402
from src.compressors.frappe.harness.checkpoints import load_checkpoint  # noqa: E402
from src.compressors.frappe.harness.data import default_dataset_root  # noqa: E402
from src.compressors.frappe.harness.reporting import Table  # noqa: E402


def jpegls_group_bytes(plane: torch.Tensor) -> int:
    """One scale group's real JPEG-LS length, in the layout the codec ships."""
    import pillow_jpls  # noqa: F401
    from torchvision.transforms.v2.functional import to_pil_image

    n, h, w = plane.shape
    flat = plane.reshape(n * h, w)
    buffer = io.BytesIO()
    to_pil_image((flat.to(torch.long) + 127).to(torch.uint8)).save(buffer, format="JPEG-LS")
    return len(buffer.getbuffer())


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    A double ``argsort`` breaks ties by index instead of averaging them, which
    turns Spearman into a formality on any series that is monotone by
    construction -- and a prefix rate ladder is exactly that. Tied entries here
    are real: once a channel has been driven to zero variance, extending the
    prefix past it changes neither the estimate nor the measurement.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def entropy_bits(values: np.ndarray) -> float:
    """Zeroth-order empirical entropy in bits per symbol."""
    counts = np.array(list(Counter(values.ravel().tolist()).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


BREAKDOWN = Table([("scale", "ps", "5d"), ("ch", "channels", "3d"),
                   ("raw", "raw_bpp", "8.3f"), ("JPEG-LS", "jpegls_bpp", "9.3f"),
                   ("order-0", "order0_bpp", "9.3f"),
                   ("residual", "residual_bpp", "9.3f"),
                   ("b/sym", "bits_per_symbol_jpegls", "7.2f"),
                   ("|max|", "abs_max_code", "6d"),
                   ("levels", "distinct_codes", "7d")], indent=2)


@torch.no_grad()
def group_breakdown(model, folder, count: int, device: str
                    ) -> tuple[list[dict], dict, float]:
    """Where the bits go, per scale group, and what a better coder could recover.

    Four rates per group: the int8 budget before any coding, what JPEG-LS
    actually spends, the zeroth-order entropy of the codes, and the entropy of
    their horizontal first difference. The gap between JPEG-LS and the entropy
    columns is what an entropy coder can still win; the size of ``raw`` is what
    only rate-distortion training or fewer channels can shrink.
    """
    groups = model.scale_groups
    per_group = [{"ps": ps, "channels": end - start, "jpegls_bytes": 0,
                  "order0_bits": 0.0, "residual_bits": 0.0, "symbols": 0,
                  "abs_max": 0, "used_levels": 0}
                 for ps, start, end in groups]
    pixels = 0
    total_mse = 0.0
    for index in range(count):
        x = folder.signed(index, device)
        pixels = x.shape[2] * x.shape[3]
        codes = model.integer_codes(x)
        recon = model.decode(model.adapt([c.to(torch.float) for c in codes]),
                             model.n_channels).clamp(-1, 1)
        total_mse += F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
        for group_index, code in enumerate(codes):
            plane = code[0].cpu()
            entry = per_group[group_index]
            entry["jpegls_bytes"] += jpegls_group_bytes(plane)
            values = plane.numpy().astype(np.int16)
            for channel in values:
                difference = np.diff(channel, axis=-1)
                entry["order0_bits"] += entropy_bits(channel) * channel.size
                entry["residual_bits"] += entropy_bits(difference) * channel.size
            entry["symbols"] += values.size
            entry["abs_max"] = max(entry["abs_max"], int(np.abs(values).max()))
            entry["used_levels"] = max(entry["used_levels"], len(np.unique(values)))

    rows, totals = [], {"raw_bpp": 0.0, "jpegls_bpp": 0.0,
                        "order0_bpp": 0.0, "residual_bpp": 0.0}
    for entry in per_group:
        raw = 8.0 * entry["channels"] / (entry["ps"] ** 2)
        row = {"ps": entry["ps"], "channels": entry["channels"], "raw_bpp": raw,
               "jpegls_bpp": entry["jpegls_bytes"] * 8 / (count * pixels),
               "order0_bpp": entry["order0_bits"] / (count * pixels),
               "residual_bpp": entry["residual_bits"] / (count * pixels),
               "abs_max_code": entry["abs_max"], "distinct_codes": entry["used_levels"]}
        row["bits_per_symbol_jpegls"] = (row["jpegls_bpp"] * pixels
                                         / (entry["symbols"] / count))
        rows.append(row)
        for key in totals:
            totals[key] += row[key]
    return rows, totals, -10.0 * math.log10(max(total_mse / count, 1e-12))


@torch.no_grad()
def requantisation_sweep(model, folder, count: int, device: str,
                         shifts: list[int]) -> list[dict]:
    """The codec's rate-distortion slope without retraining anything.

    Dividing the finest groups' codes by a power of two and rounding again traces
    what the trained model would give at a coarser quantisation, which says
    whether rate is better bought by dropping channels or by spending fewer bits
    on the ones that are kept.
    """
    if not shifts:
        return []
    groups = model.scale_groups
    sweep = []
    print("\n  post-hoc requantisation of the finest groups (no retraining):")
    print(f"  {'shift':>6} {'applied to':>14} {'bpp':>8} {'CR':>7} {'PSNR':>8}")
    finest = [index for index, (ps, _, _) in enumerate(groups) if ps <= 4]
    for shift in [0, *shifts]:
        for scope, indices in (("p<=4", finest), ("p==2", finest[-1:])):
            if shift == 0 and scope != "p<=4":
                continue
            mse_sum, byte_sum, pixels = 0.0, 0, 0
            for index in range(count):
                x = folder.signed(index, device)
                pixels = x.shape[2] * x.shape[3]
                codes = model.integer_codes(x)
                coarse = []
                for group_index, code in enumerate(codes):
                    value = code.to(torch.float)
                    if shift and group_index in indices:
                        # The clamp bound must be an integer level: a fractional
                        # bound lets a saturated symbol reach the bitstream
                        # truncated while the decoder still sees its fraction,
                        # so rate and distortion would describe two codecs.
                        step = float(2 ** shift)
                        limit = float(127 // int(step))
                        value = torch.round(value / step).clamp(-limit, limit)
                        byte_sum += jpegls_group_bytes(value.to(torch.int8)[0].cpu())
                        value = value * step
                    else:
                        byte_sum += jpegls_group_bytes(code[0].cpu())
                    coarse.append(value)
                recon = model.decode(model.adapt(coarse), model.n_channels).clamp(-1, 1)
                mse_sum += F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
            bpp = byte_sum * 8 / (count * pixels)
            value = -10.0 * math.log10(max(mse_sum / count, 1e-12))
            sweep.append({"shift": shift, "scope": scope, "bpp": bpp,
                          "compression_ratio": 24 / bpp, "psnr_db": value})
            print(f"  {shift:>6} {scope:>14} {bpp:8.3f} {24 / bpp:7.2f} {value:8.2f}",
                  flush=True)
    return sweep


@torch.no_grad()
def surrogate_check(model, folder, count: int, device: str) -> list[dict]:
    """Score the differentiable rate surrogate against the real bitstream.

    The training loss cannot call JPEG-LS, so it optimises a surrogate. A
    surrogate need not be unbiased -- the dual ascent on the multiplier absorbs a
    constant factor -- but it must be monotone in the real cost, and that is
    testable rather than assumable.
    """
    groups = model.scale_groups
    surrogate = []
    print("\n  differentiable rate surrogate vs the real bitstream, by prefix:")
    print(f"    {'n':>3} {'estimate':>10} {'measured':>10} {'ratio':>8}")
    for n in range(1, model.n_channels + 1):
        estimate, measured_bytes, pixels = 0.0, 0, 0
        for index in range(count):
            x = folder.signed(index, device)
            pixels = x.shape[2] * x.shape[3]
            codes = [code.to(torch.float) for code in model.integer_codes(x)]
            estimate += model.rate_bpp(codes, n).item()
            remaining = n
            for code, (_, start, end) in zip(codes, groups):
                if remaining <= 0:
                    break
                width = min(end - start, remaining)
                measured_bytes += jpegls_group_bytes(
                    code[0, :width].to(torch.int8).cpu())
                remaining -= width
        estimate /= count
        measured = measured_bytes * 8 / (count * pixels)
        surrogate.append({"channels": n, "estimate_bpp": estimate,
                          "measured_bpp": measured,
                          "ratio": estimate / measured if measured else None})
        print(f"    {n:>3} {estimate:10.4f} {measured:10.4f} "
              f"{estimate / measured if measured else float('nan'):8.3f}")
    estimates = np.array([point["estimate_bpp"] for point in surrogate])
    measures = np.array([point["measured_bpp"] for point in surrogate])
    pearson = float(np.corrcoef(np.log(estimates), np.log(measures))[0, 1])
    spearman = float(np.corrcoef(average_ranks(estimates), average_ranks(measures))[0, 1])
    ties = len(estimates) - len(np.unique(estimates))
    print(f"    log-log Pearson {pearson:+.4f}   Spearman {spearman:+.4f}   "
          f"ties in the estimate: {ties}")
    return surrogate


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                        help="anonymous ImageFolder root; defaults to $FRAPPE_DATASET_ROOT")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bit-depth-sweep", type=int, nargs="*", default=[1, 2, 3, 4],
                        help="requantisation shifts to try on the two finest groups")
    parser.add_argument("--compare-rate-estimate", action="store_true",
                        help="score the differentiable rate surrogate used in training "
                             "against the real JPEG-LS bitrate, prefix by prefix")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = checkpoint.model
    folder = AnonymousImageFolder(args.dataset_root, args.split)
    count = min(args.images, len(folder))

    rows, totals, psnr = group_breakdown(model, folder, count, device)
    print(f"\ncheckpoint iteration={checkpoint.iteration}  images={count}  "
          f"full-prefix PSNR={psnr:.2f} dB\n")
    print(BREAKDOWN.header())
    for row in rows:
        print(BREAKDOWN.row(row))
    print(BREAKDOWN.row({"ps": None, "channels": model.n_channels, **totals}))
    print(f"\n  compression ratio: JPEG-LS {24 / totals['jpegls_bpp']:.2f}x, "
          f"order-0 bound {24 / totals['order0_bpp']:.2f}x, "
          f"residual bound {24 / totals['residual_bpp']:.2f}x")

    sweep = requantisation_sweep(model, folder, count, device, args.bit_depth_sweep)
    surrogate = surrogate_check(model, folder, count, device) \
        if args.compare_rate_estimate else []

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"checkpoint": str(args.checkpoint), "iteration": checkpoint.iteration,
             "images": count, "psnr_db": psnr, "groups": rows, "totals": totals,
             "requantisation_sweep": sweep, "rate_surrogate": surrogate}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
