#!/usr/bin/env python3
"""Evaluate a managed FRAPPE checkpoint on local ImageFolder splits.

The training loop intentionally uses a small fixed validation subset for
iteration-based monitoring.  This utility evaluates the saved progressive
snapshots on a complete (or explicitly limited) validation/test split after
training, without exposing source archive or image names in the output.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import datasets
import torch

# Executing ``python tools/evaluate_local_checkpoint.py`` puts ``tools/`` on
# sys.path rather than the repository root.  Make the documented invocation
# self-contained so the frozen training validator can be reused directly.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from train_rae_progressive import validate
from src.compressors.frappe.model import MergedAutoencoder


def _load_snapshot(checkpoint, config, n_ch, device):
    model = MergedAutoencoder(config, n_ch).to(device)
    snapshot = checkpoint["merged_decoder_weights"][n_ch - 1]
    for scale, state_dict in enumerate(snapshot["encoder_weights"]):
        model.encoders[scale].load_state_dict(state_dict)
    model.decoder.load_state_dict(snapshot["decoder_weights"])
    model.eval()
    return model


def _split_dataset(root: str, split: str, samples: int | None):
    dataset = datasets.load_dataset(root, split=split)
    if samples is not None:
        dataset = dataset.select(range(min(samples, dataset.num_rows)))
    return dataset


def evaluate(*, checkpoint_path: Path, dataset_root: str, splits: list[str],
             device: str, samples: int | None, channels: list[int] | None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    n_trained = len(checkpoint["merged_decoder_weights"])
    if n_trained < 1:
        raise ValueError("checkpoint contains no completed merged-decoder snapshots")

    channel_counts = channels or list(range(1, n_trained + 1))
    invalid = [n for n in channel_counts if n < 1 or n > n_trained]
    if invalid:
        raise ValueError(f"channel counts out of range 1..{n_trained}: {invalid}")

    result = {
        "checkpoint": checkpoint_path.name,
        "channels_trained": n_trained,
        "metrics": {},
    }
    for split in splits:
        dataset = _split_dataset(dataset_root, split, samples)
        split_metrics = {}
        for n_ch in channel_counts:
            started = time.monotonic()
            model = _load_snapshot(checkpoint, config, n_ch, device)
            psnr, compression_ratio = validate(model, device, dataset, config)
            bpp = 24.0 / compression_ratio
            split_metrics[str(n_ch)] = {
                "num_images": dataset.num_rows,
                "PSNR_dB": float(psnr),
                "bpp": float(bpp),
                "compression_ratio": float(compression_ratio),
                "elapsed_seconds": time.monotonic() - started,
            }
            print(
                f"{split} ch={n_ch}/{n_trained}: "
                f"PSNR={psnr:.3f} dB bpp={bpp:.6f} "
                f"CR={compression_ratio:.2f} ({dataset.num_rows} images)",
                flush=True,
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result["metrics"][split] = split_metrics
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--channels", nargs="+", type=int, default=None,
                        help="channel counts to evaluate (default: every saved snapshot)")
    parser.add_argument("--samples", type=int, default=None,
                        help="evaluate only the first N images per split")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        splits=args.splits,
        device=args.device,
        samples=args.samples,
        channels=args.channels,
    )
    output = args.output or args.checkpoint.with_name("evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
