#!/usr/bin/env python3
"""Hydra entry point for reproducible FRAPPE progressive training."""

from __future__ import annotations

import json
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from train_rae_progressive import main as train_main


def _extend(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.append(flag)
    if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
        argv.extend(str(item) for item in value)
    elif isinstance(value, bool):
        argv.append("true" if value else "false")
    else:
        argv.append(str(value))


def build_argv(cfg: DictConfig) -> list[str]:
    argv: list[str] = []
    values = {
        "--device": cfg.device,
        "--input_channels": cfg.model.input_channels,
        "--ps": cfg.model.ps,
        "--decoder_ps": cfg.model.decoder_ps,
        "--decoder_dim": cfg.model.decoder_dim,
        "--decoder_kernel_size": cfg.model.decoder_kernel_size,
        "--decoder_arch": cfg.model.decoder_arch,
        "--decoder_mlp_ratio": cfg.model.decoder_mlp_ratio,
        "--decoder_layerscale": cfg.model.decoder_layerscale,
        "--decoder_layerscale_init": cfg.model.decoder_layerscale_init,
        "--encoder_arch": cfg.model.encoder_arch,
        "--linear_input": cfg.model.linear_input,
        "--lam": cfg.optimization.lam,
        "--rpe": cfg.optimization.rpe,
        "--encoder_lr_scale": cfg.optimization.encoder_lr_scale,
        "--sc_max_lr": cfg.optimization.sc_max_lr,
        "--sc_lr_pow": cfg.optimization.sc_lr_pow,
        "--md_max_lr": cfg.optimization.md_max_lr,
        "--md_lr_pow": cfg.optimization.md_lr_pow,
        "--min_lr": cfg.optimization.min_lr,
        "--grad_clip": cfg.optimization.grad_clip,
        "--optimizer": cfg.optimization.optimizer,
        "--ema_decay": cfg.optimization.ema_decay,
        "--amuse_beta": cfg.optimization.amuse_beta,
        "--amuse_beta2": cfg.optimization.amuse_beta2,
        "--amuse_eps": cfg.optimization.amuse_eps,
        "--amuse_momentum": cfg.optimization.amuse_momentum,
        "--amuse_rho": cfg.optimization.amuse_rho,
        "--amuse_r": cfg.optimization.amuse_r,
        "--amuse_weight_lr_power": cfg.optimization.amuse_weight_lr_power,
        "--amuse_warmup_ratio": cfg.optimization.amuse_warmup_ratio,
        "--amuse_weight_decay": cfg.optimization.amuse_weight_decay,
        "--amuse_weight_decay_at_y": cfg.optimization.amuse_weight_decay_at_y,
        "--amuse_aux_update_type": cfg.optimization.amuse_aux_update_type,
        "--amuse_muon_min_ndim": cfg.optimization.amuse_muon_min_ndim,
        "--early_stopping": cfg.early_stopping.enabled,
        "--early_stopping_patience": cfg.early_stopping.patience,
        "--early_stopping_min_delta": cfg.early_stopping.min_delta,
        "--early_stopping_min_epochs": cfg.early_stopping.min_epochs,
        "--early_stopping_samples": cfg.early_stopping.samples,
        "--early_stopping_min_score": cfg.early_stopping.min_score,
        "--iterations_single": cfg.training.iterations_single,
        "--iterations_merged": cfg.training.iterations_merged,
        "--validation_every_iterations": cfg.validation.every_iterations,
        "--batch_size": cfg.training.batch_size,
        "--num_workers": cfg.training.num_workers,
        "--min_aspect": cfg.augmentation.min_aspect,
        "--max_aspect": cfg.augmentation.max_aspect,
        "--min_size": cfg.augmentation.min_size,
        "--max_size": cfg.augmentation.max_size,
        "--min_width": cfg.augmentation.min_width,
        "--max_width": cfg.augmentation.max_width,
        "--min_scale": cfg.augmentation.min_scale,
        "--max_scale": cfg.augmentation.max_scale,
        "--validation_height": cfg.augmentation.validation_height,
        "--validation_width": cfg.augmentation.validation_width,
        "--augmentation_config": json.dumps(
            OmegaConf.to_container(cfg.augmentation, resolve=True),
            separators=(",", ":"), sort_keys=True,
        ),
        "--merged_decoder_noise": cfg.training.merged_decoder_noise,
        "--merged_decoder_round": cfg.training.merged_decoder_round,
        "--train_ds": cfg.data.root,
        "--val_ds": cfg.data.root,
        "--dataset_samples": cfg.training.dataset_samples,
        "--validation_samples": cfg.training.validation_samples,
        "--resume_checkpoint": cfg.checkpoint.resume_checkpoint,
        "--resume_channels": cfg.checkpoint.resume_channels,
        "--run_dir": cfg.run.dir,
        "--tensorboard": cfg.logging.tensorboard,
        "--keep_best_k": cfg.checkpoint.keep_best_k,
        "--seed": cfg.seed,
        "--max_batches_per_epoch": cfg.training.max_batches_per_epoch,
    }
    for flag, value in values.items():
        _extend(argv, flag, value)
    return argv


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_main(build_argv(cfg))


if __name__ == "__main__":
    main()
