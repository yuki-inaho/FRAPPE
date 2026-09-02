#!/usr/bin/env python3
"""Quantization-aware fine-tuning of the FRAPPE encoder for OpenVINO deployment.

The decoder is frozen property: only the analysis convolutions, the companders
and NNCF's quantizer ranges receive updates, and the RD objective is the
trainer's own -- weighted ``log10 MSE`` per sampled prefix, the symbol-weighted
rate surrogate, and the saturation penalty. Validation and every stored score
come from the deployment path: real integer codes, a real JPEG-LS bitstream.

Subcommands are added as the pipeline grows (``export``, ``evaluate``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F

from src.compressors.frappe.experiment import (
    KBestCheckpointManager,
    TensorBoardTracker,
    atomic_json_dump,
)
from src.compressors.frappe.harness.bitstream import (
    BitstreamConvention,
    measure_rate,
    prefix_channels,
)
from src.compressors.frappe.harness.checkpoints import load_checkpoint
from src.compressors.frappe.harness.data import AnonymousImageFolder, default_dataset_root
from src.compressors.frappe.harness.metrics import Averaging, RateDistortionAccumulator
from src.compressors.frappe.harness.quantization import (
    TrainableEncoder,
    freeze_decoder,
    load_qat_checkpoint,
    quantize_encoder,
    save_qat_checkpoint,
    save_qat_state,
    sha256_of,
)
from src.compressors.frappe.harness.training import CropDataset, seed_worker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train", help="calibrate and fine-tune the encoder under fake quantization")
    train.add_argument("--checkpoint", type=Path, required=True,
                       help="FP32 joint-prefix checkpoint to quantize; never modified in place")
    train.add_argument("--dataset-root", type=Path, default=None,
                       help="anonymous imagefolder; default: the harness's dataset root")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda:0",
                       help="training device; a missing GPU is an error, not a fallback")
    train.add_argument("--calibration-images", type=int, default=32,
                       help="training-split images used for NNCF calibration")
    train.add_argument("--target-device", default="NPU", choices=["NPU", "CPU", "ANY"],
                       help="placement policy for the deployed graph, not this machine")
    train.add_argument("--iterations", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--crop", type=int, default=256)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--points-per-step", type=int, default=2,
                       help="sampled prefixes per step; one of them is always the full prefix")
    train.add_argument("--full-prefix-weight", type=float, default=1.5)
    train.add_argument("--lam-rate", type=float, default=0.0)
    train.add_argument("--lam-sat", type=float, default=1e-3)
    train.add_argument("--validation-images", type=int, default=2)
    train.add_argument("--validate-every", type=int, default=1)
    train.add_argument("--keep-best-k", type=int, default=3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--resume", type=Path, default=None,
                       help="QAT checkpoint written by this tool; continues at its iteration + 1")
    return parser.parse_args(argv)


def calibration_tensors(folder: AnonymousImageFolder, count: int) -> list[torch.Tensor]:
    """The calibration set: the first ``count`` signed training images, verbatim."""
    actual = min(count, len(folder))
    if actual < count:
        raise SystemExit(f"requested {count} calibration images but the split has {len(folder)}")
    return [folder.signed(index) for index in range(actual)]


def count_quantizers(module: torch.nn.Module) -> int:
    return sum(1 for candidate in module.modules()
               if type(candidate).__name__.endswith("Quantizer"))


def train_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit(f"--device {args.device} but CUDA is not available; refusing to fall back")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    model = checkpoint.model.eval()
    freeze_decoder(model)
    base_sha256 = sha256_of(args.checkpoint)

    dataset_root = args.dataset_root or default_dataset_root()
    calibration = calibration_tensors(AnonymousImageFolder(dataset_root, "train"),
                                      args.calibration_images)
    quantized = quantize_encoder(TrainableEncoder(model), calibration,
                                 target_device=args.target_device,
                                 subset_size=len(calibration))
    quantizers = count_quantizers(quantized)
    if quantizers == 0:
        raise SystemExit(
            "NNCF inserted no quantizers; refusing to fine-tune a fake-quant-free model")

    start_iteration = 0
    if args.resume is not None:
        resumed = load_qat_checkpoint(args.resume, TrainableEncoder(model))
        if resumed["base_checkpoint_sha256"] != base_sha256:
            raise SystemExit(
                "the resumed QAT checkpoint was built from a different base checkpoint")
        quantized = resumed["model"]
        start_iteration = resumed["iteration"]

    quantized.to(device)
    model.to(device)
    optimizer = torch.optim.AdamW(quantized.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.99))
    if args.resume is not None:
        optimizer.load_state_dict(resumed["optimizer_state"])

    run_dir = args.run_dir
    checkpoints = run_dir / "checkpoints"
    last_path = checkpoints / "last.pth.tar"

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed + 1)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        CropDataset(dataset_root, "train", args.crop),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        generator=loader_generator, worker_init_fn=seed_worker)
    validation_folder = AnonymousImageFolder(dataset_root, "validation")
    best = KBestCheckpointManager(checkpoints / "best", k=args.keep_best_k, mode="max")
    tracker = TensorBoardTracker(run_dir / "tensorboard", enabled=True)

    report: dict = {"checkpoint": str(args.checkpoint), "base_checkpoint_sha256": base_sha256,
                    "target_device": args.target_device, "quantizers": quantizers,
                    "seed": args.seed, "history": []}
    batches = iter(loader)
    n_channels = model.n_channels

    for iteration in range(start_iteration + 1, args.iterations + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        x = batch.to(device).float() / 127.5 - 1.0

        sampled = torch.randperm(n_channels - 1, generator=sampler_generator)
        sampled = sampled[:max(args.points_per_step - 1, 0)]
        points = sorted({int(point) + 1 for point in sampled} | {n_channels})
        weights = torch.tensor([args.full_prefix_weight if point == n_channels else 1.0
                                for point in points], device=device)
        weights = weights / weights.sum()

        codes = quantized(x)
        adapted = model.adapt(codes)
        loss = torch.zeros((), device=device)
        for point, weight in zip(points, weights):
            reconstruction = model.decode(adapted, point)
            distortion = F.mse_loss(reconstruction, x).clamp_min(1e-12).log10()
            loss = loss + weight * distortion
        if args.lam_rate > 0:
            loss = loss + args.lam_rate * sum(model.rate_bpp(codes, point) for point in points) \
                / len(points)
        if args.lam_sat > 0:
            loss = loss + args.lam_sat * sum(compander.saturation_penalty()
                                             for compander in quantized.companders)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        tracker.scalar("train/loss", float(loss.detach()), iteration)
        tracker.scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
        metrics = {"iteration": iteration, "loss": float(loss.detach()),
                   "lr": optimizer.param_groups[0]["lr"], "points": points}

        if iteration % args.validate_every == 0 or iteration == args.iterations:
            accumulator = RateDistortionAccumulator(Averaging.AGGREGATE_MSE)
            plan = prefix_channels(model.scale_groups, n_channels)
            for index in range(min(args.validation_images, len(validation_folder))):
                image = validation_folder.signed(index, device)
                pixels = image.shape[2] * image.shape[3]
                with torch.no_grad():
                    codes = quantized(image)
                    adapted = model.adapt(codes)
                    reconstruction = model.decode(adapted, n_channels).clamp(-1, 1)
                integer_codes = [code.round().clamp(-127, 127).to(torch.int8) for code in codes]
                mse = F.mse_loss(image / 2 + 0.5, reconstruction / 2 + 0.5).item()
                byte_count, _ = measure_rate(integer_codes, pixels, plan,
                                             BitstreamConvention.PAYLOAD_ONLY)
                accumulator.add(mse, byte_count, pixels)
            point = accumulator.point(label=n_channels)
            metrics.update({"psnr_db": point.psnr_db, "bpp": point.bpp,
                            "bytes_total": point.bytes_total,
                            "validation_images": accumulator.images})
            tracker.scalar("validation/psnr_db", point.psnr_db, iteration)
            tracker.scalar("validation/bpp", point.bpp, iteration)

            payload = {"qat_state": save_qat_state(quantized), "iteration": iteration,
                       "psnr_db": point.psnr_db, "bpp": point.bpp,
                       "base_checkpoint_sha256": base_sha256}
            best.consider(point.psnr_db, iteration, payload)
            report["kbest"] = [{**entry} for entry in best.entries]
        report["history"].append(metrics)
        atomic_json_dump(report, run_dir / "latest_report.json")

    save_qat_checkpoint(last_path, quantized, optimizer, args.iterations,
                        base_sha256, psnr_db=report["history"][-1].get("psnr_db"),
                        bpp=report["history"][-1].get("bpp"))
    print(f"trained {args.iterations - start_iteration} steps from {start_iteration}; "
          f"quantizers={quantizers}; last={last_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train_command(args)


if __name__ == "__main__":
    main()
