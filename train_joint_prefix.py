#!/usr/bin/env python3
"""Algorithm B: joint prefix quantization-aware training for FRAPPE.

Every latent channel exists from the first optimizer step.  Each update runs one
encoder pass and evaluates a small "sandwich" of prefixes -- the shortest, the
full one, and a few sampled uniformly in log symbol count -- through a single
full-width superdecoder.  The cost per update therefore scales with the number
of sampled prefixes, not with the channel count, and the decoder is never
re-initialized as channels are added because channels are never added.

The quantizer walks the note's Q0--Q4 continuation on a fraction-of-training
schedule, and every reported rate is a real JPEG-LS bitstream length.

Run from the repository root so ``src`` is importable::

    pixi run python train_joint_prefix.py --run_dir runs/joint_001 ...
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.compressors.frappe.experiment import (
    KBestCheckpointManager, ModelEMA, TensorBoardTracker, atomic_json_dump, atomic_torch_save)
from src.compressors.frappe.prefix import (
    QUANTIZATION_MODES, JointPrefixFRAPPE, calibrate_companders, klt_initialize)

RELEASED_PS = [32, 32, 32, 16, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4, 2, 2, 2]


# ---- data --------------------------------------------------------------


class CropDataset(torch.utils.data.Dataset):
    """Random crops from an anonymous local ImageFolder split."""

    def __init__(self, root: Path, split: str, crop: int, augment: bool = True,
                 limit: int | None = None) -> None:
        self.files = sorted((root / split).glob("image_????????.png"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise SystemExit(f"no anonymous PNG images under {root / split}")
        self.crop = crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as handle:
            handle.load()
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        size = self.crop
        if h < size or w < size:
            raise SystemExit(f"image {h}x{w} smaller than the requested {size} crop")
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        patch = image[top:top + size, left:left + size]
        if self.augment:
            if random.random() < 0.5:
                patch = patch[:, ::-1]
            if random.random() < 0.5:
                patch = patch[::-1]
        return torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def load_full_images(root: Path, split: str, count: int, device: str) -> torch.Tensor:
    files = sorted((root / split).glob("image_????????.png"))[:count]
    frames = []
    for path in files:
        with Image.open(path) as handle:
            handle.load()
            frames.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return batch.to(device=device, dtype=torch.float32) / 127.5 - 1.0


# ---- prefix sampling ---------------------------------------------------


class PrefixSampler:
    """Sandwich sampling over operating points spaced uniformly in log rate.

    Symbol counts differ by 256x across the schedule, so sampling the channel
    index uniformly would concentrate almost every sample on rates nobody uses.
    Sampling uniformly in ``log C_n`` spreads the sampled operating points over
    the rate axis instead.
    """

    def __init__(self, ps: list[int], extra: int = 1, seed: int = 0) -> None:
        self.n_channels = len(ps)
        symbols = np.cumsum([1.0 / (p * p) for p in ps])
        self.log_symbols = np.log(symbols)
        self.extra = extra
        self.rng = random.Random(seed)

    def sample(self, subset_prob: float = 0.0) -> list:
        prefixes = {1, self.n_channels}
        low, high = self.log_symbols[0], self.log_symbols[-1]
        extra = []
        for _ in range(self.extra):
            target = self.rng.uniform(low, high)
            n = int(np.abs(self.log_symbols - target).argmin()) + 1
            if subset_prob and self.rng.random() < subset_prob:
                # A random subset of the same size: pruning a codec to a
                # non-prefix channel set only works if the decoder has seen
                # non-prefix masks during training.
                extra.append(sorted(self.rng.sample(range(1, self.n_channels + 1), n)))
            else:
                prefixes.add(n)
        return sorted(prefixes) + extra


def continuation_stage(progress: float, boundaries: list[float],
                       alpha_range: tuple[float, float]) -> tuple[str, float, bool]:
    """Map training progress onto (quantization mode, soft-round alpha, frozen encoder).

    Stages are Q0 float, Q1 additive uniform noise, Q2 annealed soft rounding,
    Q3 hard rounding with a straight-through estimator, and Q4 hard calibration
    with the analysis path frozen so only the synthesis transform adapts.
    """
    q0, q1, q2, q3 = boundaries
    if progress < q0:
        return "float", 0.0, False
    if progress < q1:
        return "aun", 0.0, False
    if progress < q2:
        span = max(q2 - q1, 1e-6)
        ratio = (progress - q1) / span
        low, high = alpha_range
        return "soft", float(low * (high / low) ** ratio), False
    if progress < q3:
        return "hard", 0.0, False
    return "hard", 0.0, True


# ---- evaluation --------------------------------------------------------


def jpegls_bytes(codes: list[torch.Tensor], n_channels: int, groups) -> int:
    """Real JPEG-LS length for prefix ``1:n_channels``, FRAPPE's plane layout."""
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


@torch.no_grad()
def evaluate(model: JointPrefixFRAPPE, images: torch.Tensor, prefixes: list[int],
             rate_images: int = 8) -> dict[int, dict[str, float]]:
    """Deployment-path evaluation: true int8 codes and real bitstream lengths."""
    model.eval()
    results = {n: {"mse": 0.0, "rate_mse": 0.0, "bytes": 0} for n in prefixes}
    pixels = images.shape[2] * images.shape[3]
    # Entropy coding is the slow part, so it may run on a prefix of the images.
    # The reported operating point then has to use the SAME prefix for distortion:
    # pairing a PSNR averaged over one set with a bitrate measured on another is
    # not a point on any rate-distortion curve. The all-image PSNR is reported
    # separately, for variance, rather than mixed into the operating point.
    rate_images = min(rate_images, images.shape[0])
    for index in range(images.shape[0]):
        x = images[index:index + 1]
        codes = model.integer_codes(x)
        y = model.adapt([code.to(torch.float) for code in codes])
        for n in prefixes:
            recon = model.decode(y, n).clamp(-1, 1)
            error = F.mse_loss(x / 2 + 0.5, recon / 2 + 0.5).item()
            results[n]["mse"] += error
            if index < rate_images:
                results[n]["rate_mse"] += error
                results[n]["bytes"] += jpegls_bytes(codes, n, model.scale_groups)
    count = images.shape[0]
    report = {}
    for n in prefixes:
        matched = results[n]["rate_mse"] / rate_images if rate_images else float("nan")
        bpp = results[n]["bytes"] * 8 / (rate_images * pixels) if rate_images else float("nan")
        report[n] = {"psnr_db": -10.0 * math.log10(max(matched, 1e-12)), "bpp": bpp,
                     "compression_ratio": 24.0 / bpp if bpp > 0 else float("nan"),
                     "psnr_all_images_db": -10.0 * math.log10(max(results[n]["mse"] / count, 1e-12)),
                     "rate_images": rate_images, "images": count}
    model.train()
    return report


# ---- training ----------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset_root", type=Path,
                   default=Path("/workspace/data/frappe_rgb_800x608/imagefolder"))
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--ps", type=int, nargs="+", default=RELEASED_PS)
    p.add_argument("--input_channels", type=int, default=3)
    p.add_argument("--decoder_ps", type=int, default=8)
    p.add_argument("--decoder_dim", type=int, default=256)
    p.add_argument("--decoder_kernel_size", type=int, default=3)
    p.add_argument("--decoder_arch", default="CCCCCC")
    p.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    p.add_argument("--decoder_layerscale", type=lambda s: s.lower() in ("true", "1", "yes"),
                   default=True)
    p.add_argument("--decoder_layerscale_init", type=float, default=1e-6)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr_scale", type=float, default=1.0)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.5)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--extra_prefixes", type=int, default=1,
                   help="random prefixes per update, on top of the shortest and the full one")
    p.add_argument("--full_prefix_weight", type=float, default=2.0)
    p.add_argument("--lam_rate", type=float, default=0.0,
                   help="Lagrange multiplier on the differentiable bpp estimate; "
                        "the initial value when --target_bpp steers it")
    p.add_argument("--target_bpp", type=float, default=None,
                   help="steer --lam_rate by dual ascent until the measured JPEG-LS "
                        "bitrate at the target operating point reaches this value")
    p.add_argument("--target_operating_point", type=int, default=None,
                   help="prefix length --target_bpp refers to (default: the full prefix)")
    p.add_argument("--rate_dual_lr", type=float, default=0.7,
                   help="step size of the multiplicative dual ascent on --lam_rate")
    p.add_argument("--lam_rate_max", type=float, default=20.0)
    p.add_argument("--subset_prob", type=float, default=0.0,
                   help="probability that a sampled operating point is an arbitrary "
                        "channel subset rather than a prefix; needed if the codec is "
                        "meant to survive structured pruning to a non-prefix set")
    p.add_argument("--lam_distill", type=float, default=0.0)
    p.add_argument("--lam_mono", type=float, default=0.05)
    p.add_argument("--lam_sat", type=float, default=1e-3)
    p.add_argument("--mono_margin", type=float, default=0.0)
    p.add_argument("--continuation", type=float, nargs=4, default=[0.10, 0.30, 0.55, 0.90],
                   help="progress boundaries for float->AUN->soft->hard->calibration")
    p.add_argument("--alpha_range", type=float, nargs=2, default=[2.0, 64.0])
    p.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    p.add_argument("--init", choices=["klt", "random"], default="klt")
    p.add_argument("--init_images", type=int, default=64)
    p.add_argument("--compander_percentile", type=float, default=99.9)
    p.add_argument("--compander_knee", type=float, default=2.0)
    p.add_argument("--compander_target", type=float, default=100.0)
    p.add_argument("--validation_images", type=int, default=16)
    p.add_argument("--rate_images", type=int, default=8,
                   help="validation images that also get a real JPEG-LS bitstream measurement")
    p.add_argument("--validate_every", type=int, default=1000)
    p.add_argument("--target_psnr", type=float, default=40.0)
    p.add_argument("--keep_best_k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--resume_model_only", action="store_true",
                   help="load only the weights, not the optimizer state or the step "
                        "counter; this is how a pruned checkpoint is fine-tuned, since "
                        "its parameter shapes no longer match the old optimizer state")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device if torch.cuda.is_available() else "cpu"
    max_ps = max(args.ps)
    if args.crop % max_ps:
        raise SystemExit(f"--crop must be a multiple of the largest patch size {max_ps}")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = TensorBoardTracker(run_dir / "tensorboard", enabled=True)
    kbest = KBestCheckpointManager(run_dir / "checkpoints" / "best", k=args.keep_best_k, mode="max")

    config = argparse.Namespace(**{k: v for k, v in vars(args).items()})
    model = JointPrefixFRAPPE(config).to(device)

    calibration = load_full_images(args.dataset_root, "train", args.init_images, device)
    if args.init == "klt":
        print("initialising analysis filters from the deflated patch KLT", flush=True)
        klt_initialize(model, calibration, verbose=True)
    calibrate_companders(model, calibration, args.compander_percentile,
                         args.compander_knee, args.compander_target)
    del calibration
    torch.cuda.empty_cache()

    validation = load_full_images(args.dataset_root, "validation", args.validation_images, device)
    dataset = CropDataset(args.dataset_root, "train", args.crop)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        generator=generator, worker_init_fn=seed_worker)

    analysis_params = list(model.analysis.parameters()) + list(model.companders.parameters())
    analysis_ids = {id(p) for p in analysis_params}
    decoder_params = [p for p in model.parameters() if id(p) not in analysis_ids]
    optimizer = torch.optim.AdamW([
        {"params": analysis_params, "lr": args.lr * args.encoder_lr_scale},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.99))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    sampler = PrefixSampler(args.ps, args.extra_prefixes, args.seed)

    start_iteration = 0
    resumed_lam_rate = None
    if args.resume and not args.resume.is_file():
        # Silently starting from scratch would overwrite last.pth.tar in the same
        # run directory, destroying the thing the caller meant to continue.
        raise SystemExit(f"--resume {args.resume} does not exist")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if args.resume_model_only:
            print(f"loaded weights from {args.resume} "
                  f"(iteration {state.get('iteration')}); optimizer state discarded",
                  flush=True)
        else:
            optimizer.load_state_dict(state["optimizer"])
            start_iteration = int(state["iteration"])
            tracker.global_step = start_iteration
            if ema is not None and state.get("ema"):
                ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
            if state.get("lam_rate") is not None:
                resumed_lam_rate = float(state["lam_rate"])
            print(f"resumed {args.resume} at iteration {start_iteration}", flush=True)

    atomic_json_dump({
        "schema_version": 1,
        "algorithm": "joint-prefix-QAT",
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model": {"total_decoder_channels": model.total_decoder_channels,
                  "channels": model.n_channels,
                  "parameters": sum(p.numel() for p in model.parameters())},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
    }, run_dir / "run_metadata.json")

    report_prefixes = sorted({n for n in (1, 9, 15, 18, model.n_channels)
                              if n <= model.n_channels})
    print(f"\n{'=' * 66}\n  Joint prefix QAT: {model.n_channels} channels, "
          f"{model.total_decoder_channels} decoder ch, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params\n"
          f"  crop={args.crop} batch={args.batch_size} iterations={args.iterations} "
          f"amp={args.amp}\n  continuation boundaries={args.continuation} "
          f"target={args.target_psnr} dB\n{'=' * 66}\n", flush=True)

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if args.amp == "bf16" and device.startswith("cuda")
                else torch.autocast("cuda", enabled=False))
    lam_rate = resumed_lam_rate if resumed_lam_rate is not None else float(args.lam_rate)
    if args.target_bpp is not None and lam_rate <= 0:
        lam_rate = 0.05  # a starting point; dual ascent moves it within a few checks
    target_point = args.target_operating_point or model.n_channels

    model.train()
    iteration = start_iteration
    window: list[float] = []
    best_psnr = float("-inf")
    best_iteration, best_bpp = None, None
    frozen = False
    stream = iter(loader)
    while iteration < args.iterations:
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(loader)
            batch = next(stream)
        x = batch.to(device, non_blocking=True).float() / 127.5 - 1.0

        progress = iteration / max(args.iterations - 1, 1)
        mode, alpha, freeze = continuation_stage(progress, args.continuation, args.alpha_range)
        if freeze != frozen:
            for parameter in analysis_params:
                parameter.requires_grad_(not freeze)
            frozen = freeze
            print(f"  [{iteration}] Q4 hard calibration: analysis path "
                  f"{'frozen' if freeze else 'unfrozen'}", flush=True)

        scale = min(1.0, (iteration + 1) / max(args.warmup, 1))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = (args.min_lr + (base - args.min_lr) * cosine) * scale

        prefixes = sampler.sample(args.subset_prob)
        with autocast:
            reconstructions, codes = model.forward_operating_points(x, prefixes, mode, alpha)
        reconstructions = [r.float() for r in reconstructions]

        weights = torch.tensor(
            [args.full_prefix_weight if point == model.n_channels else 1.0
             for point in prefixes], device=device)
        weights = weights / weights.sum()
        distortions = [F.mse_loss(r, x).clamp_min(1e-12).log10() for r in reconstructions]
        loss = sum(w * d for w, d in zip(weights, distortions))

        rate_estimate = None
        if lam_rate > 0:
            rates = [model.rate_bpp(codes, point) for point in prefixes]
            rate_estimate = rates[-1] if prefixes[-1] == model.n_channels else rates[0]
            loss = loss + lam_rate * sum(w * r for w, r in zip(weights, rates))
        if args.lam_distill > 0 and len(prefixes) > 1:
            teacher = reconstructions[-1].detach()
            loss = loss + args.lam_distill * sum(
                (r - teacher).abs().mean() for r in reconstructions[:-1]) / (len(prefixes) - 1)
        if args.lam_mono > 0 and len(prefixes) > 1:
            violations = [torch.relu(b - a + args.mono_margin)
                          for a, b in zip(distortions[:-1], distortions[1:])]
            loss = loss + args.lam_mono * sum(violations) / len(violations)
        if args.lam_sat > 0:
            loss = loss + args.lam_sat * model.saturation_penalty()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], args.grad_clip).item()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        iteration += 1
        full_log_mse = distortions[-1].item()
        window.append(full_log_mse)
        step = tracker.next_step()
        tracker.scalar("train/loss", loss.item(), step)
        tracker.scalar("train/full_prefix_log_mse", full_log_mse, step)
        tracker.scalar("train/full_prefix_psnr", -10.0 * (full_log_mse + math.log10(0.25)), step)
        tracker.scalar("train/grad_norm", grad_norm, step)
        tracker.scalar("train/lr", optimizer.param_groups[-1]["lr"], step)
        if rate_estimate is not None:
            tracker.scalar("train/rate_estimate_bpp", rate_estimate.item(), step)
            tracker.scalar("train/lam_rate", lam_rate, step)

        if iteration % 100 == 0:
            mean_log_mse = float(np.mean(window[-200:]))
            print(f"  it {iteration:6d}/{args.iterations}  {mode:5s}"
                  f"{f' a={alpha:.1f}' if mode == 'soft' else '      '}"
                  f"  train_psnr={-10.0 * (mean_log_mse + math.log10(0.25)):6.2f} dB"
                  f"  loss={loss.item():7.3f}  lr={optimizer.param_groups[-1]['lr']:.2e}"
                  + (f"  rate~{rate_estimate.item():.3f}bpp lam={lam_rate:.3f}"
                     if rate_estimate is not None else "")
                  + f"  points={prefixes}", flush=True)

        if iteration % args.validate_every == 0 or iteration >= args.iterations:
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            report = evaluate(model, validation, report_prefixes, args.rate_images)
            full = report[model.n_channels]
            for n, values in report.items():
                tracker.scalar(f"validation/psnr/prefix_{n:02d}", values["psnr_db"], iteration)
                tracker.scalar(f"validation/bpp/prefix_{n:02d}", values["bpp"], iteration)
            print(f"  [{iteration}] validation  " + "  ".join(
                f"n={n}:{v['psnr_db']:.2f}dB/{v['bpp']:.3f}bpp" for n, v in report.items()),
                flush=True)
            if args.target_bpp is not None:
                measured = report[target_point]["bpp"]
                # Multiplicative dual ascent: over budget raises the price of a
                # bit, under budget lowers it.  Reported every check so the
                # trajectory of the multiplier is auditable, not a hidden knob.
                # The step is clipped so one check can change the price of a bit
                # by at most a factor of e^0.7, which keeps the multiplier from
                # slamming into its bounds on the first over-budget measurement.
                step_size = float(np.clip(
                    args.rate_dual_lr * (measured / args.target_bpp - 1.0), -0.7, 0.7))
                lam_rate = float(np.clip(lam_rate * math.exp(step_size),
                                         1e-6, args.lam_rate_max))
                print(f"  [{iteration}] rate target {args.target_bpp:.4f} bpp at n={target_point}: "
                      f"measured {measured:.4f} bpp -> lam_rate {lam_rate:.4f}", flush=True)
                tracker.scalar("validation/lam_rate", lam_rate, iteration)
            payload = {"iteration": iteration, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "config": vars(args),
                       "report": report, "lam_rate": lam_rate,
                       "ema": ema.snapshot() if ema else None}
            atomic_torch_save(payload, run_dir / "checkpoints" / "last.pth.tar")
            # With a rate target, PSNR alone is not a ranking: a checkpoint that
            # is over budget is not a better codec, it is a different one.  Only
            # checkpoints inside the budget compete.
            within_budget = (args.target_bpp is None
                             or report[target_point]["bpp"] <= args.target_bpp * 1.05)
            if within_budget:
                kbest.consider(full["psnr_db"], iteration, payload)
            atomic_json_dump(
                {"iteration": iteration, "elapsed_hours": (time.time() - started) / 3600,
                 "prefix_report": {str(k): v for k, v in report.items()}},
                run_dir / "latest_report.json")
            # Only checkpoints inside the rate budget are candidates for "best".
            # Without this the headline number can come from an early stage that
            # had not yet been pushed down to the target bitrate, which is a
            # different operating point rather than a better model.
            if within_budget and full["psnr_db"] > best_psnr:
                best_psnr = full["psnr_db"]
                best_iteration = iteration
                best_bpp = full["bpp"]
            if ema is not None:
                model.load_state_dict(backup)
            # Stop only once the quantization continuation has finished: the
            # target must be met by a model that has actually been trained
            # through hard rounding, not by a float-stage model that happens to
            # survive evaluation-time rounding.
            if (args.target_bpp is None and full["psnr_db"] >= args.target_psnr
                    and progress >= args.continuation[3]):
                print(f"  reached the {args.target_psnr} dB target at iteration {iteration} "
                      f"after the full Q0-Q4 continuation", flush=True)
                break

    tracker.close()
    total = time.time() - started
    budget = ("" if args.target_bpp is None
              else f" within the {args.target_bpp:.4f} bpp budget")
    detail = ("" if best_iteration is None
              else f" at iteration {best_iteration} ({best_bpp:.4f} bpp)")
    print(f"\n{'=' * 66}\n  FINISHED  best full-prefix validation PSNR = "
          f"{best_psnr:.2f} dB{budget}{detail}"
          f"\n  in {total / 3600:.2f} h\n{'=' * 66}\n", flush=True)


if __name__ == "__main__":
    main()
