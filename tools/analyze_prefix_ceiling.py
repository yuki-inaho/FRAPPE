#!/usr/bin/env python3
"""What a FRAPPE channel schedule can reach, and what the training method costs.

The analysis transform is a per-scale, non-overlapping linear patch projection,
so its reachable quality can be bounded without training a decoder at all.  This
tool measures three bounds on the same data, which together separate "the
schedule is too small" from "the optimizer is leaving quality on the table":

``greedy-klt``   Coarse-to-fine principal component analysis of the *residual*
                 patch covariance -- the training-free initializer of the theory
                 note's "DCT/KLT/PCA" section, and the linear ideal of the
                 published stagewise recipe, in which each scale only ever sees
                 what the coarser scales left behind.  Reported with float
                 coefficients and with int8 codes plus real JPEG-LS bitrates.

``joint-linear`` The same architecture -- per-scale strided analysis, untied
                 per-scale synthesis summed on the decoder grid -- with the
                 greedy ordering removed and both sides optimised together by
                 SGD.  This is eq. (superdecoder) with the trunk replaced by the
                 identity, so the gap to ``greedy-klt`` is exactly what stagewise
                 residual fitting costs, before any nonlinearity is involved.

``free-pca``     Principal components of whole 32x32 blocks at the same symbol
                 count, with no shift-tying and no per-scale structure.  Any
                 linear analysis of that size is dominated by it, so the gap to
                 ``joint-linear`` is what the FRAPPE structure itself costs.

Every number is a *lower* bound on the trained codec, whose decoder is nonlinear
and sees spatial context that none of these bounds can use.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

# Running ``python tools/<name>.py`` puts ``tools/`` on sys.path rather than the
# repository root, so the documented invocation is made self-contained here.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.harness.data import default_dataset_root

DEFAULT_PS = [32, 32, 32, 16, 16, 16, 16, 16, 16, 8, 8, 8,
              4, 4, 4, 4, 4, 4, 2, 2, 2]


def scale_groups(ps_list):
    groups, i = [], 0
    while i < len(ps_list):
        ps = ps_list[i]
        start = i
        while i < len(ps_list) and ps_list[i] == ps:
            i += 1
        groups.append((ps, start, i))
    return groups


def load_images(root: Path, split: str, count: int, device: str) -> torch.Tensor:
    files = sorted((root / split).glob("image_????????.png"))[:count]
    if not files:
        raise SystemExit(f"no anonymous PNG images under {root / split}")
    frames = []
    for path in files:
        with Image.open(path) as handle:
            handle.load()
            frames.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return batch.to(device=device, dtype=torch.float32) / 127.5 - 1.0


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    return rearrange(x, "b c (h p1) (w p2) -> (b h w) (c p1 p2)", p1=p, p2=p)


def unpatchify(rows: torch.Tensor, p: int, b: int, c: int, h: int, w: int) -> torch.Tensor:
    return rearrange(rows, "(b h w) (c p1 p2) -> b c (h p1) (w p2)",
                     b=b, h=h // p, w=w // p, c=c, p1=p, p2=p)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """PSNR on the [0, 1] convention used by the FRAPPE training script."""
    mse = torch.nn.functional.mse_loss(x / 2 + 0.5, y / 2 + 0.5).item()
    return float("inf") if mse <= 0 else -10.0 * float(np.log10(mse))


def robust_scale(plane: torch.Tensor, percentile: float) -> torch.Tensor:
    """Per-channel magnitude at ``percentile``, subsampled to bound memory."""
    n = plane.shape[1]
    flat = plane.permute(1, 0, 2, 3).reshape(n, -1).abs()
    if flat.shape[1] > 2_000_000:
        flat = flat[:, torch.randperm(flat.shape[1], device=flat.device)[:2_000_000]]
    return torch.quantile(flat.float(), percentile / 100.0, dim=-1).view(1, n, 1, 1).clamp_min(1e-8)


def jpegls_bpp(planes: list[torch.Tensor], n_pixels: int) -> float:
    """Real JPEG-LS bitstream length, matching the FRAPPE validate() layout."""
    import pillow_jpls  # noqa: F401
    from torchvision.transforms.v2.functional import to_pil_image

    total = 0
    for plane in planes:  # each (n, H, W) int8, one scale group
        n, h, w = plane.shape
        flat = plane.reshape(n * h, w)
        buffer = io.BytesIO()
        to_pil_image((flat.to(torch.long) + 127).to(torch.uint8)).save(buffer, format="JPEG-LS")
        total += len(buffer.getbuffer())
    return total * 8 / n_pixels


def greedy_klt_bound(fit: torch.Tensor, evaluate: torch.Tensor, ps_list: list[int],
                     groups, percentile: float, verbose: bool) -> dict:
    """Coarse-to-fine KLT of the residual patch covariance, with an int8 pass."""
    b, c, h, w = evaluate.shape
    fit_residual, eval_residual = fit.clone(), evaluate.clone()
    recon = torch.zeros_like(evaluate)
    curve, planes, bases = [], [], []
    channel_index = 0

    for ps, start, end in groups:
        n_group = end - start
        rows = patchify(fit_residual, ps)
        mean = rows.mean(dim=0, keepdim=True)
        centred = rows - mean
        covariance = (centred.T @ centred) / centred.shape[0]
        _, vectors = torch.linalg.eigh(covariance.double())
        basis = vectors.flip(-1)[:, :n_group].to(torch.float32)  # descending eigenvalue
        bases.append((ps, mean, basis))

        eval_rows = patchify(eval_residual, ps) - mean
        coefficients = eval_rows @ basis
        planes.append(rearrange(coefficients, "(b hh ww) n -> b n hh ww",
                                b=b, hh=h // ps, ww=w // ps))

        for k in range(n_group):
            component = coefficients[:, k:k + 1] @ basis[:, k:k + 1].T
            if k == 0:
                component = component + mean
            recon = recon + unpatchify(component, ps, b, c, h, w)
            channel_index += 1
            value = psnr(evaluate, recon.clamp(-1, 1))
            raw_bpp = 8.0 * sum(1.0 / (p * p) for p in ps_list[:channel_index])
            curve.append({"channels": channel_index, "ps": ps,
                          "psnr_db": value, "raw_bpp": raw_bpp})
            if verbose:
                print(f"  n={channel_index:2d} (ps={ps:2d})  greedy-KLT PSNR={value:6.2f} dB"
                      f"  raw={raw_bpp:6.3f} bpp", flush=True)

        eval_residual = eval_residual - unpatchify(
            coefficients @ basis.T + mean, ps, b, c, h, w)
        fit_residual = fit_residual - unpatchify(
            (centred @ basis) @ basis.T + mean, ps, fit.shape[0], c, fit.shape[2], fit.shape[3])

    quantized = torch.zeros_like(evaluate)
    int8_planes = []
    for (ps, mean, basis), plane in zip(bases, planes):
        scales = robust_scale(plane, percentile)
        codes = (plane / scales * 127.0).round().clamp(-127, 127)
        int8_planes.append(codes.to(torch.int8))
        rows = rearrange(codes / 127.0 * scales, "b n hh ww -> (b hh ww) n")
        quantized = quantized + unpatchify(rows @ basis.T + mean, ps, b, c, h, w)

    return {"curve": curve, "psnr_db": curve[-1]["psnr_db"],
            "int8_psnr_db": psnr(evaluate, quantized.clamp(-1, 1)),
            "int8_planes": int8_planes}


def joint_linear_bound(fit: torch.Tensor, evaluate: torch.Tensor, groups,
                       steps: int, batch: int, lr: float, verbose: bool) -> dict:
    """Optimise the FRAPPE analysis/synthesis pair jointly, with a linear trunk."""
    device = fit.device
    torch.manual_seed(0)
    analysis = [torch.nn.Parameter(torch.randn(n, fit.shape[1], p, p, device=device)
                                   * (fit.shape[1] * p * p) ** -0.5) for p, n in groups]
    synthesis = [torch.nn.Parameter(torch.randn(n, fit.shape[1], p, p, device=device)
                                    * n ** -0.5) for p, n in groups]
    bias = torch.nn.Parameter(torch.zeros(1, fit.shape[1], 1, 1, device=device))
    optimizer = torch.optim.Adam(analysis + synthesis + [bias], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=lr / 300)

    def reconstruct(x):
        out = bias.expand_as(x).clone()
        for (p, _), w, d in zip(groups, analysis, synthesis):
            out = out + F.conv_transpose2d(F.conv2d(x, w, stride=p), d, stride=p)
        return out

    trace = []
    for step in range(steps):
        index = torch.randint(0, fit.shape[0], (min(batch, fit.shape[0]),), device=device)
        loss = F.mse_loss(reconstruct(fit[index]), fit[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if verbose and (step % max(steps // 8, 1) == 0 or step == steps - 1):
            with torch.no_grad():
                value = psnr(evaluate, reconstruct(evaluate).clamp(-1, 1))
            trace.append({"step": step, "psnr_db": value})
            print(f"  joint-linear step {step:6d}/{steps}  val PSNR={value:6.2f} dB", flush=True)
    with torch.no_grad():
        value = psnr(evaluate, reconstruct(evaluate).clamp(-1, 1))
    return {"psnr_db": value, "steps": steps, "trace": trace}


def free_pca_bound(fit: torch.Tensor, evaluate: torch.Tensor, ps_list: list[int],
                   block: int) -> dict:
    """Unconstrained block PCA at the schedule's symbol count -- a linear upper bound."""
    fit_rows = patchify(fit, block)
    eval_rows = patchify(evaluate, block)
    mean = fit_rows.mean(dim=0, keepdim=True)
    centred = fit_rows - mean
    covariance = (centred.T @ centred) / centred.shape[0]
    _, vectors = torch.linalg.eigh(covariance.double())
    basis = vectors.flip(-1).to(torch.float32)
    symbols = int(sum((block // p) ** 2 for p in ps_list))
    kept = basis[:, :symbols]
    approximation = (eval_rows - mean) @ kept @ kept.T + mean
    b, c, h, w = evaluate.shape
    value = psnr(evaluate, unpatchify(approximation, block, b, c, h, w).clamp(-1, 1))
    return {"psnr_db": value, "block": block, "symbols_per_block": symbols,
            "block_dimension": int(fit_rows.shape[1]),
            "fraction_of_dimension": symbols / float(fit_rows.shape[1])}


def print_summary(args: argparse.Namespace, report: dict) -> None:
    """The three bounds side by side, and the two gaps between them.

    The gap from greedy to joint is what stagewise ordering costs; the gap
    from joint to free PCA is what the FRAPPE structure costs. Printing the
    subtractions is the point of running all three.
    """
    print(f"\n  schedule: {len(args.ps)} channels, "
          f"{report['raw_int8_bpp']:.3f} raw int8 bpp")
    greedy = report["bounds"].get("greedy_klt")
    if greedy:
        print(f"  greedy-KLT (stagewise ideal)   float {greedy['psnr_db']:6.2f} dB"
              f"   int8 {greedy['int8_psnr_db']:6.2f} dB"
              f"   {greedy['measured_jpegls_bpp']:6.3f} bpp"
              f"   CR {greedy['compression_ratio']:5.2f}")
    joint = report["bounds"].get("joint_linear")
    if joint:
        print(f"  joint-linear (same structure)  float {joint['psnr_db']:6.2f} dB")
    free = report["bounds"].get("free_pca")
    if free:
        print(f"  free PCA (linear upper bound)  float {free['psnr_db']:6.2f} dB"
              f"   ({free['symbols_per_block']}/{free['block_dimension']} dims kept)")
    if greedy and joint:
        print(f"\n  cost of greedy stagewise ordering: "
              f"{joint['psnr_db'] - greedy['psnr_db']:+.2f} dB")
    if joint and free:
        print(f"  cost of the FRAPPE structural constraint: "
              f"{free['psnr_db'] - joint['psnr_db']:+.2f} dB")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                        help="anonymous ImageFolder root; defaults to $FRAPPE_DATASET_ROOT")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--fit-images", type=int, default=96,
                        help="training-split images used to fit the bases; kept separate "
                             "from the evaluated split")
    parser.add_argument("--ps", type=int, nargs="+", default=DEFAULT_PS)
    parser.add_argument("--bounds", nargs="+", default=["greedy-klt", "joint-linear", "free-pca"],
                        choices=["greedy-klt", "joint-linear", "free-pca"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--saturation-percentile", type=float, default=99.9)
    parser.add_argument("--bpp-images", type=int, default=8)
    parser.add_argument("--joint-steps", type=int, default=20000)
    parser.add_argument("--joint-batch", type=int, default=8)
    parser.add_argument("--joint-lr", type=float, default=3e-3)
    parser.add_argument("--joint-fit-images", type=int, default=256,
                        help="the joint-linear bound is an optimisation, so it gets more data")
    parser.add_argument("--free-pca-block", type=int, default=None,
                        help="block size for the unconstrained bound; default: max(--ps)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    device = args.device if torch.cuda.is_available() else "cpu"
    groups = scale_groups(args.ps)
    started = time.time()
    torch.manual_seed(0)

    fit_count = max(args.fit_images,
                    args.joint_fit_images if "joint-linear" in args.bounds else 0)
    fit = load_images(args.dataset_root, "train", fit_count, device)
    evaluate = load_images(args.dataset_root, args.split, args.images, device)
    b, _c, h, w = evaluate.shape
    verbose = not args.quiet
    if verbose:
        print(f"fit={tuple(fit.shape)} eval={tuple(evaluate.shape)} groups={groups}", flush=True)

    raw_bpp = 8.0 * sum(1.0 / (p * p) for p in args.ps)
    report: dict = {"ps": args.ps, "channels": len(args.ps),
                    "fit_images": int(fit.shape[0]), "eval_images": int(b),
                    "image_size": [int(h), int(w)], "raw_int8_bpp": raw_bpp, "bounds": {}}

    if "greedy-klt" in args.bounds:
        if verbose:
            print("\ngreedy per-scale KLT of the residual patch covariance:", flush=True)
        greedy = greedy_klt_bound(fit[:args.fit_images], evaluate, args.ps, groups,
                                  args.saturation_percentile, verbose)
        n_bpp = min(b, args.bpp_images)
        measured = float(np.mean([
            jpegls_bpp([plane[i].cpu() for plane in greedy["int8_planes"]], h * w)
            for i in range(n_bpp)]))
        report["bounds"]["greedy_klt"] = {
            "psnr_db": greedy["psnr_db"], "int8_psnr_db": greedy["int8_psnr_db"],
            "measured_jpegls_bpp": measured,
            "compression_ratio": 24.0 / measured if measured else None,
            "prefix_curve": greedy["curve"]}

    if "joint-linear" in args.bounds:
        if verbose:
            print("\njointly optimised linear analysis/synthesis (same structure):", flush=True)
        report["bounds"]["joint_linear"] = joint_linear_bound(
            fit[:args.joint_fit_images], evaluate,
            [(ps, end - start) for ps, start, end in groups],
            args.joint_steps, args.joint_batch, args.joint_lr, verbose)

    if "free-pca" in args.bounds:
        block = args.free_pca_block or max(args.ps)
        if verbose:
            print(f"\nunconstrained {block}x{block} block PCA at the same symbol count:",
                  flush=True)
        report["bounds"]["free_pca"] = free_pca_bound(
            fit[:args.fit_images], evaluate, args.ps, block)

    report["seconds"] = time.time() - started

    print_summary(args, report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
