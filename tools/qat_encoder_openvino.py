#!/usr/bin/env python3
"""Quantization-aware fine-tuning of the FRAPPE encoder for OpenVINO deployment.

The decoder is frozen property: only the analysis convolutions, the companders
and NNCF's quantizer ranges receive updates, and the RD objective is the
trainer's own -- weighted ``log10 MSE`` per sampled prefix, the symbol-weighted
rate surrogate, and the saturation penalty. Validation and every stored score
come from the deployment path: real integer codes, a real JPEG-LS bitstream.

``train`` fine-tunes under fake quantization; ``evaluate`` compares deployed
conditions on one rate-distortion footing. The official deployment artifacts
are ``tools/package_npu_int8.py``'s job -- the torch hook graph is a
training-time device, not a shipped one.
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
    op_inventory,
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

    evaluate = sub.add_parser("evaluate",
                              help="rate-distortion of fp32 / PTQ / QAT encoders as deployed IRs")
    evaluate.add_argument("--base-checkpoint", type=Path, required=True,
                          help="FP32 joint-prefix checkpoint; source of the fp32 "
                               "condition and of the shared frozen decoder")
    evaluate.add_argument("--qat-checkpoint", type=Path, required=True,
                          help="QAT checkpoint from 'train'; source of the "
                               "trained condition")
    evaluate.add_argument("--dataset-root", type=Path, default=None)
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--images", type=int, default=16,
                          help="the same image indices are measured for every condition")
    evaluate.add_argument("--calibration-images", type=int, default=32,
                          help="training-split images for the PTQ condition's calibration")
    evaluate.add_argument("--calibration-split", default="train",
                          help="dataset split used only for PTQ statistics; "
                               "must differ from --split")
    evaluate.add_argument("--target-device", default="NPU", choices=["NPU", "CPU", "ANY"])
    evaluate.add_argument("--ptq-preset", default="performance",
                          choices=["performance", "mixed"],
                          help="NNCF ONNX activation/weight quantization preset")
    evaluate.add_argument("--bias-correction", default="fast",
                          choices=["fast", "accurate", "none"],
                          help="NNCF ONNX PTQ bias-correction policy")
    evaluate.add_argument("--device", default="cuda:0",
                          help="torch device for the shared frozen decoder; CPU is allowed")
    evaluate.add_argument("--artifact-dir", type=Path, default=None,
                          help="where the per-condition ONNX graphs are written; "
                               "default: beside --output")
    evaluate.add_argument("--output", type=Path, required=True)
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


def evaluate_command(args: argparse.Namespace) -> None:
    import json

    import openvino as ov

    from src.compressors.frappe.harness.deployment import (
        DecoderGraph,
        measure_deployed_conditions,
    )
    from src.compressors.frappe.harness.quantization import (
        TrainableEncoder,
        encoder_weights_from_qat,
        export_encoder_onnx,
        quantize_onnx_encoder,
        require_disjoint_calibration_samples,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device} but CUDA is not available; refusing to fall back")
    decode_device = torch.device(args.device)

    base_model = load_checkpoint(args.base_checkpoint, "cpu").model.eval()
    base_sha256 = sha256_of(args.base_checkpoint)
    dataset_root = args.dataset_root or default_dataset_root()
    folder = AnonymousImageFolder(dataset_root, args.split)
    calibration_folder = AnonymousImageFolder(dataset_root, args.calibration_split)
    count = min(args.images, len(folder))
    if count < args.images:
        raise SystemExit(f"requested {args.images} images but the split has {len(folder)}")
    n_channels = base_model.n_channels
    height, width = folder.pixels(0).shape[2], folder.pixels(0).shape[3]
    plan = prefix_channels(base_model.scale_groups, n_channels)

    artifacts = args.artifact_dir or args.output.parent
    artifacts.mkdir(parents=True, exist_ok=True)
    order = ["fp32", "ptq", "ptq_qat_weights"]

    # NNCF 3.3's torch-side hook quantization does not survive tracing (the IR
    # ignores the trained ranges), so the deployed graph quantizes the *trained
    # weights* afresh with NNCF's ONNX backend. The saved Q/DQ ONNX is then the
    # sole input to OpenVINO. Each condition gets its own model instance:
    # loading QAT weights into a shared module would rewrite fp32 too.
    base_weights = TrainableEncoder(load_checkpoint(args.base_checkpoint, "cpu").model.eval())
    trained_weights = TrainableEncoder(load_checkpoint(args.base_checkpoint, "cpu").model.eval())
    payload = torch.load(args.qat_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("base_checkpoint_sha256") != base_sha256:
        raise SystemExit("the QAT checkpoint was built from a different base checkpoint")
    trained_weights.load_state_dict(encoder_weights_from_qat(payload))

    if args.calibration_images > len(calibration_folder):
        raise SystemExit(
            f"requested {args.calibration_images} calibration images but "
            f"{args.calibration_split} has {len(calibration_folder)}")
    calibration_indices = list(range(args.calibration_images))
    try:
        require_disjoint_calibration_samples(
            args.calibration_split,
            calibration_indices,
            args.split,
            list(range(count)),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    calibration = [{"image": calibration_folder.pixels(index).numpy()}
                   for index in calibration_indices]
    onnx_base = artifacts / "base_weights_encoder.onnx"
    export_encoder_onnx(base_weights, onnx_base, torch.zeros(
        1, base_model.input_channels, height, width, dtype=torch.uint8))
    onnx_qat = artifacts / "qat_weights_encoder.onnx"
    export_encoder_onnx(trained_weights, onnx_qat, torch.zeros(
        1, base_model.input_channels, height, width, dtype=torch.uint8))

    ptq_base_path = artifacts / "ptq_base_encoder.onnx"
    ptq_base_info = quantize_onnx_encoder(
        onnx_base,
        ptq_base_path,
        calibration,
        target_device=args.target_device,
        subset_size=args.calibration_images,
        preset=args.ptq_preset,
        bias_correction=args.bias_correction,
    )
    ptq_qat_path = artifacts / "ptq_qat_weights_encoder.onnx"
    ptq_qat_info = quantize_onnx_encoder(
        onnx_qat,
        ptq_qat_path,
        calibration,
        target_device=args.target_device,
        subset_size=args.calibration_images,
        preset=args.ptq_preset,
        bias_correction=args.bias_correction,
    )

    core = ov.Core()
    deployment_onnx = {
        "fp32": onnx_base,
        "ptq": ptq_base_path,
        "ptq_qat_weights": ptq_qat_path,
    }
    compiled = {}
    ir_artifacts = {}
    for name, onnx_path in deployment_onnx.items():
        ir = core.read_model(str(onnx_path))
        xml_path = artifacts / f"{name}_encoder.xml"
        ov.save_model(ir, str(xml_path), compress_to_fp16=False)
        persisted = core.read_model(str(xml_path))
        compiled[name] = core.compile_model(persisted, "CPU")
        ir_artifacts[name] = {
            "xml": str(xml_path),
            "bin": str(xml_path.with_suffix(".bin")),
            "xml_sha256": sha256_of(xml_path),
            "bin_sha256": sha256_of(xml_path.with_suffix(".bin")),
            "ops": op_inventory(persisted.get_ops()),
        }

    # Built after the calibration forwards: DecoderGraph wraps the shared
    # base model, so moving it to the decode device moves every encoder the
    # calibration just ran on.
    decoder = DecoderGraph(base_model, n_channels, uint8_io=True).to(decode_device).eval()

    # fp32, first in the order, is every other condition's reference.
    conditions = measure_deployed_conditions(
        {name: compiled[name] for name in order}, decoder, folder, range(count),
        scale_groups=base_model.scale_groups, plan=plan, prefix=n_channels,
        height=height, width=width, device=decode_device)
    for name in order:
        numbers = conditions[name]
        print(f"  {name}: {numbers['psnr_db']:.4f} dB @ {numbers['bpp']:.4f} bpp "
              f"({numbers['bytes_total']} bytes, "
              f"{numbers['mismatched_symbols_vs_reference']} symbols vs fp32)")

    report = {
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": base_sha256,
        "qat_checkpoint": str(args.qat_checkpoint),
        "qat_iteration": payload.get("iteration"),
        "split": args.split, "images": count,
        "image_indices": list(range(count)),
        "calibration": {
            "split": args.calibration_split,
            "images": len(calibration_indices),
            "image_indices": calibration_indices,
        },
        "size": [width, height],
        "prefix": n_channels, "averaging": "aggregate_mse",
        "bitstream_convention": "PAYLOAD_ONLY",
        "decoder": "torch DecoderGraph (frozen, fp32), identical for every condition",
        "openvino_runtime": ov.__version__,
        "quantization": {
            "backend": "NNCF ONNX PTQ",
            "target_device": args.target_device,
            "preset": args.ptq_preset,
            "bias_correction": args.bias_correction,
            "ptq_base": {**ptq_base_info, "sha256": sha256_of(ptq_base_path)},
            "ptq_qat_weights": {**ptq_qat_info, "sha256": sha256_of(ptq_qat_path)},
        },
        "artifacts": {
            "onnx": {
                name: {"path": str(path), "sha256": sha256_of(path)}
                for name, path in deployment_onnx.items()
            },
            "openvino_ir": ir_artifacts,
        },
        "conditions": conditions,
        "deltas_vs_fp32": {
            name: {"d_psnr_db": conditions[name]["psnr_db"] - conditions["fp32"]["psnr_db"],
                   "d_bpp": conditions[name]["bpp"] - conditions["fp32"]["bpp"]}
            for name in order if name != "fp32"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)


if __name__ == "__main__":
    main()
