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
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.compressors.frappe.experiment import (
    KBestCheckpointManager,
    ModelEMA,
    TensorBoardTracker,
    atomic_json_dump,
    atomic_torch_save,
)
from src.compressors.frappe.harness.data import default_dataset_root
from src.compressors.frappe.harness.training import (
    CropDataset,
    PrefixSampler,
    RateTarget,
    continuation_stage,
    seed_worker,
)
from src.compressors.frappe.prefix import (
    JointPrefixFRAPPE,
    calibrate_companders,
    klt_initialize,
)

RELEASED_PS = [32, 32, 32, 16, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4, 2, 2, 2]


# ---- data --------------------------------------------------------------


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
    p.add_argument("--dataset_root", type=Path, default=default_dataset_root(),
                   help="anonymous ImageFolder root; defaults to $FRAPPE_DATASET_ROOT")
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


def build_model(args: argparse.Namespace, device: str) -> JointPrefixFRAPPE:
    """Construct the model and give it a starting point worth optimising from.

    The KLT initialiser puts the analysis filters at the stagewise linear optimum
    instead of at noise, and the compander calibration sets each channel's code
    range from measured percentiles rather than from a guess. Both cost one pass
    over a handful of images and both are why the run does not spend its first
    thousand steps rediscovering them.
    """
    model = JointPrefixFRAPPE(argparse.Namespace(**vars(args))).to(device)
    calibration = load_full_images(args.dataset_root, "train", args.init_images, device)
    if args.init == "klt":
        print("initialising analysis filters from the deflated patch KLT", flush=True)
        klt_initialize(model, calibration, verbose=True)
    calibrate_companders(model, calibration, args.compander_percentile,
                         args.compander_knee, args.compander_target)
    del calibration
    torch.cuda.empty_cache()
    return model


def build_optimizer(model: JointPrefixFRAPPE, args: argparse.Namespace
                    ) -> tuple[list[torch.nn.Parameter], torch.optim.Optimizer]:
    """Two parameter groups, because the two halves want different rates.

    The analysis path is a handful of small convolutions and per-channel
    companding scalars; the synthesis path is the bulk of the model. Returning
    the analysis parameters as well is not incidental -- the last stage of the
    quantization continuation freezes exactly them.
    """
    analysis_params = list(model.analysis.parameters()) + list(model.companders.parameters())
    analysis_ids = {id(parameter) for parameter in analysis_params}
    decoder_params = [parameter for parameter in model.parameters()
                      if id(parameter) not in analysis_ids]
    optimizer = torch.optim.AdamW([
        {"params": analysis_params, "lr": args.lr * args.encoder_lr_scale},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.99))
    return analysis_params, optimizer


def build_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """Random crops, seeded so a run is reproducible from its --seed alone."""
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return torch.utils.data.DataLoader(
        CropDataset(args.dataset_root, "train", args.crop),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=generator, worker_init_fn=seed_worker)


def prefix_loss(model: JointPrefixFRAPPE, x: torch.Tensor, points, mode: str,
                alpha: float, autocast, args: argparse.Namespace, lam_rate: float,
                device: str):
    """The objective, for one batch, over the sampled operating points.

    Distortion is ``log10 MSE`` per point rather than plain MSE. The prefixes in
    one sample can differ in error by two orders of magnitude, and a sum of raw
    MSEs would let the lowest-rate prefix own the gradient; the logarithm puts
    them on one scale. It is also FRAPPE's own objective, so the comparison with
    the stagewise trainer stays honest.

    Returns the loss, the per-point distortions -- the monotonicity term needs
    them and so does the console line -- and the full-prefix rate estimate, which
    is what the dual ascent later compares against a real measurement.
    """
    with autocast:
        reconstructions, codes = model.forward_operating_points(x, points, mode, alpha)
    reconstructions = [reconstruction.float() for reconstruction in reconstructions]

    weights = torch.tensor(
        [args.full_prefix_weight if point == model.n_channels else 1.0
         for point in points], device=device)
    weights = weights / weights.sum()
    distortions = [F.mse_loss(reconstruction, x).clamp_min(1e-12).log10()
                   for reconstruction in reconstructions]
    loss = sum(weight * distortion for weight, distortion in zip(weights, distortions))

    rate_estimate = None
    if lam_rate > 0:
        rates = [model.rate_bpp(codes, point) for point in points]
        rate_estimate = rates[-1] if points[-1] == model.n_channels else rates[0]
        loss = loss + lam_rate * sum(w * rate for w, rate in zip(weights, rates))
    if args.lam_distill > 0 and len(points) > 1:
        teacher = reconstructions[-1].detach()
        loss = loss + args.lam_distill * sum(
            (reconstruction - teacher).abs().mean()
            for reconstruction in reconstructions[:-1]) / (len(points) - 1)
    if args.lam_mono > 0 and len(points) > 1:
        # A longer prefix that reconstructs worse breaks the property the
        # architecture exists for, so it is penalised rather than merely counted.
        violations = [torch.relu(later - earlier + args.mono_margin)
                      for earlier, later in zip(distortions[:-1], distortions[1:])]
        loss = loss + args.lam_mono * sum(violations) / len(violations)
    if args.lam_sat > 0:
        loss = loss + args.lam_sat * model.saturation_penalty()
    return loss, distortions, rate_estimate


@dataclass
class BestSoFar:
    """The best checkpoint seen, and only from inside the rate budget.

    Without the budget condition the headline number can come from an early
    stage that had not yet been pushed down to the target bitrate. That is a
    different operating point, not a better model.
    """

    psnr_db: float = float("-inf")
    iteration: int | None = None
    bpp: float | None = None


def validation_checkpoint(model, ema, optimizer, validation, report_prefixes,
                          args, iteration, started, tracker, kbest, rate_target,
                          lam_rate, target_point, run_dir, best):
    """Measure, price a bit, and write the checkpoints. Returns the new state.

    Validation runs on the EMA weights when there are any, then puts the online
    weights back: the smoothed model is what would be shipped, but it is not what
    training continues from.
    """
    backup = None
    if ema is not None:
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ema.copy_to(model)

    report = evaluate(model, validation, report_prefixes, args.rate_images)
    full = report[model.n_channels]
    for n, values in report.items():
        tracker.scalar(f"validation/psnr/prefix_{n:02d}", values["psnr_db"], iteration)
        tracker.scalar(f"validation/bpp/prefix_{n:02d}", values["bpp"], iteration)
    print(f"  [{iteration}] validation  " + "  ".join(
        f"n={n}:{v['psnr_db']:.2f}dB/{v['bpp']:.3f}bpp" for n, v in report.items()),
        flush=True)

    if rate_target is not None:
        # Multiplicative dual ascent: over budget raises the price of a bit,
        # under budget lowers it. Reported at every check so the trajectory of
        # the multiplier is auditable rather than a hidden knob.
        measured = report[target_point]["bpp"]
        lam_rate = rate_target.update(lam_rate, measured)
        print(f"  [{iteration}] rate target {args.target_bpp:.4f} bpp at "
              f"n={target_point}: measured {measured:.4f} bpp -> lam_rate {lam_rate:.4f}",
              flush=True)
        tracker.scalar("validation/lam_rate", lam_rate, iteration)

    payload = {"iteration": iteration, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "config": vars(args),
               "report": report, "lam_rate": lam_rate,
               "ema": ema.snapshot() if ema else None}
    atomic_torch_save(payload, run_dir / "checkpoints" / "last.pth.tar")
    within_budget = (args.target_bpp is None
                     or report[target_point]["bpp"] <= args.target_bpp * 1.05)
    if within_budget:
        kbest.consider(full["psnr_db"], iteration, payload)
        if full["psnr_db"] > best.psnr_db:
            best = BestSoFar(full["psnr_db"], iteration, full["bpp"])
    atomic_json_dump(
        {"iteration": iteration, "elapsed_hours": (time.time() - started) / 3600,
         "prefix_report": {str(key): value for key, value in report.items()}},
        run_dir / "latest_report.json")

    if backup is not None:
        model.load_state_dict(backup)
    return report, lam_rate, best


@dataclass
class TrainingContext:
    """Everything the loop needs, assembled once.

    The loop reads sixteen things and a function signature listing them is worse
    than a name for the collection. Grouping them also makes the loop's
    dependencies visible: it is not reaching into a module, it is given a run.
    """

    model: JointPrefixFRAPPE
    optimizer: torch.optim.Optimizer
    ema: ModelEMA | None
    loader: torch.utils.data.DataLoader
    sampler: PrefixSampler
    validation: torch.Tensor
    args: argparse.Namespace
    device: str
    tracker: TensorBoardTracker
    kbest: KBestCheckpointManager
    run_dir: Path
    analysis_params: list
    base_lrs: list
    report_prefixes: list
    target_point: int
    rate_target: RateTarget | None
    started: float


def resume_if_asked(model, optimizer, ema, tracker, args, device: str
                    ) -> tuple[int, float]:
    """Continue a run, or start one, and say which.

    A --resume pointing at nothing used to start from scratch and then
    overwrite last.pth.tar in the same directory, destroying the thing the
    caller meant to continue. The multiplier travels with the checkpoint so
    a resumed rate-targeted run does not restart its dual ascent.
    """
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
    lam_rate = resumed_lam_rate if resumed_lam_rate is not None else float(args.lam_rate)
    if args.target_bpp is not None and lam_rate <= 0:
        # A starting point only; the dual ascent moves it within a few checks.
        lam_rate = 0.05
    return start_iteration, lam_rate


def write_run_metadata(model, args, run_dir, device: str) -> None:
    """Everything needed to reproduce this run, written before it starts."""
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


def announce(model, args) -> None:
    print(f"\n{'=' * 66}\n  Joint prefix QAT: {model.n_channels} channels, "
          f"{model.total_decoder_channels} decoder ch, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params\n"
          f"  crop={args.crop} batch={args.batch_size} iterations={args.iterations} "
          f"amp={args.amp}\n  continuation boundaries={args.continuation} "
          f"target={args.target_psnr} dB\n{'=' * 66}\n", flush=True)


def run_training(ctx: TrainingContext, start_iteration: int, lam_rate: float) -> BestSoFar:
    """The optimisation loop: sample, step, and check in on a cadence.

    Everything that decides *what* is optimised -- the quantization stage, the
    operating points, the price of a bit -- is settled before the loop starts and
    handed to it. What is left here is the mechanics.
    """
    model, args, device = ctx.model, ctx.args, ctx.device
    optimizer, tracker, kbest = ctx.optimizer, ctx.tracker, ctx.kbest
    ema, loader, sampler = ctx.ema, ctx.loader, ctx.sampler
    validation, report_prefixes = ctx.validation, ctx.report_prefixes
    analysis_params, base_lrs = ctx.analysis_params, ctx.base_lrs
    rate_target, target_point = ctx.rate_target, ctx.target_point
    run_dir, started = ctx.run_dir, ctx.started
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if args.amp == "bf16" and device.startswith("cuda")
                else torch.autocast("cuda", enabled=False))
    model.train()
    iteration = start_iteration
    window: list[float] = []
    best = BestSoFar()
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
        loss, distortions, rate_estimate = prefix_loss(
            model, x, prefixes, mode, alpha, autocast, args, lam_rate, device)

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
            report, lam_rate, best = validation_checkpoint(
                model, ema, optimizer, validation, report_prefixes, args, iteration,
                started, tracker, kbest, rate_target, lam_rate, target_point,
                run_dir, best)
            full = report[model.n_channels]
            # Stop only once the quantization continuation has finished: the
            # target must be met by a model that has actually been trained
            # through hard rounding, not by a float-stage model that happens to
            # survive evaluation-time rounding.
            if (args.target_bpp is None and full["psnr_db"] >= args.target_psnr
                    and progress >= args.continuation[3]):
                print(f"  reached the {args.target_psnr} dB target at iteration {iteration} "
                      f"after the full Q0-Q4 continuation", flush=True)
                break
    return best


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

    model = build_model(args, device)

    validation = load_full_images(args.dataset_root, "validation", args.validation_images, device)
    loader = build_loader(args)
    analysis_params, optimizer = build_optimizer(model, args)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    sampler = PrefixSampler(args.ps, args.extra_prefixes, args.seed)

    start_iteration, lam_rate = resume_if_asked(
        model, optimizer, ema, tracker, args, device)
    write_run_metadata(model, args, run_dir, device)
    report_prefixes = sorted({n for n in (1, 9, 15, 18, model.n_channels)
                              if n <= model.n_channels})
    announce(model, args)

    context = TrainingContext(
        model=model, optimizer=optimizer, ema=ema, loader=loader, sampler=sampler,
        validation=validation, args=args, device=device, tracker=tracker, kbest=kbest,
        run_dir=run_dir, analysis_params=analysis_params,
        base_lrs=[group["lr"] for group in optimizer.param_groups],
        report_prefixes=report_prefixes,
        target_point=args.target_operating_point or model.n_channels,
        rate_target=(None if args.target_bpp is None
                     else RateTarget(args.target_bpp, args.rate_dual_lr, args.lam_rate_max)),
        started=started)
    best = run_training(context, start_iteration, lam_rate)

    tracker.close()
    total = time.time() - started
    best_psnr, best_iteration, best_bpp = best.psnr_db, best.iteration, best.bpp
    budget = ("" if args.target_bpp is None
              else f" within the {args.target_bpp:.4f} bpp budget")
    detail = ("" if best_iteration is None
              else f" at iteration {best_iteration} ({best_bpp:.4f} bpp)")
    print(f"\n{'=' * 66}\n  FINISHED  best full-prefix validation PSNR = "
          f"{best_psnr:.2f} dB{budget}{detail}"
          f"\n  in {total / 3600:.2f} h\n{'=' * 66}\n", flush=True)


if __name__ == "__main__":
    main()
