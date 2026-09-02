#!/usr/bin/env python3
"""Measure a safe per-GPU FRAPPE merged-decoder training batch size.

The benchmark uses random RGB crops but otherwise follows the expensive merged
decoder phase: encoders are frozen, latents are quantized, and only the decoder
is optimized. It deliberately does not touch the training dataset or runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from timm.optim import Adan
from torch.nn import functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compressors.frappe.experiment import ModelEMA
from src.compressors.frappe.model import MergedAutoencoder
from src.compressors.frappe.ops import decoder_channels_per_encoder, get_scale_groups
from train_rae_progressive import make_amuse_optimizer

PROFILES = {
    "managed_9ch": {
        "ps": [32, 32, 32, 16, 16, 16, 8, 8, 8],
        "decoder_ps": 32,
        "decoder_dim": 768,
        "decoder_arch": "CCCCCC",
        "decoder_layerscale": False,
    },
    "released_21ch": {
        "ps": [32, 32, 32, 16, 16, 16, 16, 16, 16, 8, 8, 8,
               4, 4, 4, 4, 4, 4, 2, 2, 2],
        "decoder_ps": 8,
        "decoder_dim": 768,
        "decoder_arch": "CCCCCCCCCCCC",
        "decoder_layerscale": True,
    },
}


def make_config(profile: str, optimizer: str) -> SimpleNamespace:
    selected = PROFILES[profile]
    return SimpleNamespace(
        input_channels=3,
        ps=selected["ps"],
        decoder_ps=selected["decoder_ps"],
        decoder_dim=selected["decoder_dim"],
        decoder_kernel_size=3,
        decoder_arch=selected["decoder_arch"],
        decoder_mlp_ratio=4.0,
        decoder_layerscale=selected["decoder_layerscale"],
        decoder_layerscale_init=1e-6,
        encoder_arch="SC8",
        optimizer=optimizer,
        amuse_beta=0.8,
        amuse_beta2=0.999,
        amuse_eps=1e-10,
        amuse_momentum=0.95,
        amuse_rho=0.3,
        amuse_r=0.0,
        amuse_weight_lr_power=2.0,
        amuse_warmup_ratio=0.05,
        amuse_weight_decay=0.0,
        amuse_weight_decay_at_y=0.0,
        amuse_aux_update_type="adamw",
        amuse_muon_min_ndim=2,
    )


def decoder_channels(config: SimpleNamespace) -> int:
    return sum(
        (end - start) * decoder_channels_per_encoder(ps, config.decoder_ps)
        for ps, start, end in get_scale_groups(config.ps, len(config.ps)))


def measure(config: SimpleNamespace, batch_size: int, height: int, width: int,
            ema_decay: float) -> dict[str, float]:
    device = torch.device("cuda:0")
    model = MergedAutoencoder(config, len(config.ps)).to(device)
    for encoder in model.encoders:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoder.eval()

    if config.optimizer == "amuse":
        optimizer = make_amuse_optimizer(
            [{"params": list(model.decoder.parameters()), "lr": 1.0}],
            max_lr=5e-4, total_steps=20, config=config)
    else:
        optimizer = Adan(model.decoder.parameters(), lr=5e-4)
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None

    x = torch.rand(batch_size, 3, height, width, device=device).mul_(2).sub_(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # Inlined rather than wrapped in a closure: the closure captured ``model``
    # by cell, and the ``del`` below that frees the GPU memory emptied that cell,
    # so any second call would have raised NameError. Used once, so a function
    # bought nothing and cost a latent bug.
    with torch.no_grad():
        latents = [latent.round() for latent in model.encode(x)]
    prediction = model.decode(latents)
    loss = functional.mse_loss(prediction, x)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if config.optimizer == "amuse":
        optimizer.eval()
    if ema is not None:
        ema.update(model)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    result = {"batch_size": batch_size, "peak_allocated_gib": peak,
              "peak_reserved_gib": reserved, "loss": float(loss.item())}
    del x, latents, prediction, loss, ema, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="managed_9ch")
    parser.add_argument("--optimizer", choices=["adan", "amuse"], default="amuse")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--candidates", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    config = make_config(args.profile, args.optimizer)
    results = []
    for batch_size in args.candidates:
        try:
            result = measure(config, batch_size, args.height, args.width, args.ema_decay)
            result["status"] = "ok"
            results.append(result)
            print(json.dumps(result), flush=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"batch_size": batch_size, "status": "oom"})
            print(json.dumps(results[-1]), flush=True)
            break
    successful = [item for item in results if item["status"] == "ok"]
    recommendation = successful[-1]["batch_size"] if successful else 0
    print(json.dumps({
        "profile": args.profile,
        "optimizer": args.optimizer,
        "image_size": {"width": args.width, "height": args.height},
        "ema_decay": args.ema_decay,
        "decoder_channels": decoder_channels(config),
        "results": results,
        "largest_tested_batch_size": recommendation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
