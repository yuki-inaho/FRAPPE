#!/usr/bin/env python3
"""Progressive multi-scale training with patchified latents and unified decoder.

This is the frozen training script used to produce the final checkpoint whose
results are reported in the FRAPPE paper. Model, ops, and quantizer definitions
are imported from the frozen `src/compressors/frappe/` library shipped with this
repository — run this script from the repo root so the `src` package is
importable.

Finer-scale latents are rearranged (einops patchify) to the coarsest resolution,
then concatenated and fed through a single decoder identical to train_rae.py's
MergedAutoencoder. The decoder architecture never changes across scale transitions —
only the input channel count grows.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.5"

import argparse, atexit, io, json, random, time
import torch, datasets, PIL.Image, pillow_jpls, numpy as np
import bjontegaard as bd
from types import SimpleNamespace
from timm.optim import Adan
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image

from src.compressors.frappe.model import AutoencoderSingleChannel, MergedAutoencoder
from src.compressors.frappe.ops import get_scale_groups, decoder_channels_per_encoder
from src.compressors.frappe.quantize import srgb_to_linear
from src.compressors.frappe.augmentation import RGBTrainingAugmentation
from src.compressors.frappe.experiment import (
    KBestCheckpointManager,
    EarlyStopping,
    ModelEMA,
    TensorBoardTracker,
    atomic_json_dump,
    atomic_torch_save,
)
from src.compressors.frappe.third_party.amuse import AMUSE


def seed_worker(_worker_id):
    """Make Python/NumPy augmentations deterministic in each loader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def merge_encoder(merged, single_channel, scale_idx, ch_in_scale):
    sc_weight = single_channel.analysis_transform[0].weight.data
    sc_bias = single_channel.analysis_transform[0].bias.data
    merged.encoders[scale_idx][0].weight.data[ch_in_scale:ch_in_scale+1] = sc_weight
    merged.encoders[scale_idx][0].bias.data[ch_in_scale:ch_in_scale+1] = sc_bias
    for (mk, mp), (sk, sp) in zip(merged.encoders[scale_idx][1].named_parameters(),
                                   single_channel.analysis_transform[1].named_parameters()):
        mp.data[ch_in_scale:ch_in_scale+1] = sp.data


def make_amuse_optimizer(param_groups, max_lr, total_steps, config):
    """Create official AMUSE groups from FRAPPE's encoder/decoder LR groups."""
    warmup_steps = max(1, int(np.ceil(total_steps * config.amuse_warmup_ratio)))
    if param_groups and not isinstance(param_groups[0], dict):
        # torch.optim accepts a bare parameter iterable; the frozen merged
        # decoder path uses that convention, so normalize it here.
        param_groups = [{'params': param_groups, 'lr': 1.0}]
    groups = []
    for source_group in param_groups:
        lr_scale = float(source_group.get('lr', 1.0))
        params = [p for p in source_group['params'] if p.requires_grad]
        muon_params = [p for p in params if p.ndim >= config.amuse_muon_min_ndim]
        aux_params = [p for p in params if p.ndim < config.amuse_muon_min_ndim]
        common = {'lr': max_lr * lr_scale, 'weight_decay': config.amuse_weight_decay}
        if muon_params:
            groups.append({
                **common,
                'params': muon_params,
                'use_muon': True,
                'momentum': config.amuse_momentum,
                'aux_update_type': config.amuse_aux_update_type,
            })
        if aux_params:
            groups.append({
                **common,
                'params': aux_params,
                'use_muon': False,
                'update_type': config.amuse_aux_update_type,
                'beta2': config.amuse_beta2,
                'eps': config.amuse_eps,
            })
    if not groups:
        raise ValueError('AMUSE received no trainable parameters')
    optimizer = AMUSE(
        groups,
        beta1=config.amuse_beta,
        warmup_steps=warmup_steps,
        rho=config.amuse_rho,
        r=config.amuse_r,
        weight_lr_power=config.amuse_weight_lr_power,
        weight_decay_at_y=config.amuse_weight_decay_at_y,
    )
    optimizer.train()
    return optimizer


def train_one(model, param_groups, dataloader, device, config, total_steps, epochs, label,
              lam=0.0, max_lr=1e-3, lr_pow=2, rpe=0.3, tracker=None,
              epoch_evaluator=None, early_stopper=None):
    def rc_sched(i_step):
        t = i_step / total_steps
        return (max_lr - config.min_lr) * (1 - (np.cos(np.pi * t)) ** (2 * lr_pow)) + config.min_lr
    if config.optimizer == 'adan':
        optimizer = Adan(param_groups, lr=1.0)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: rc_sched(s))
    elif config.optimizer == 'amuse':
        optimizer = make_amuse_optimizer(param_groups, max_lr, total_steps, config)
        schedule = None
    else:
        raise ValueError(f'Unknown optimizer: {config.optimizer}')
    all_params = [p for g in optimizer.param_groups for p in g['params']]
    ema = ModelEMA(model, config.ema_decay) if config.ema_decay > 0 else None
    log_mse_losses = []
    rate_losses = []

    for i_epoch in range(epochs):
        model.train()
        n_batches = len(dataloader)
        epoch_loss_start = len(log_mse_losses)
        epoch_rate_start = len(rate_losses)
        for i_batch, x in enumerate(dataloader):
            if config.max_batches_per_epoch is not None and i_batch >= config.max_batches_per_epoch:
                break
            x = x.to(device)
            x_in = srgb_to_linear(x) if config.linear_input else x
            pred = model(x_in)
            target = model._target(x_in, x)
            log_mse_loss = torch.nn.functional.mse_loss(pred, target).log10()
            log_mse_losses.append(log_mse_loss.item())
            if lam > 0 and hasattr(model, '_last_z'):
                target_power = (target ** 2).mean().detach()
                rate = model._last_z.std().log2()
                rate_losses.append(rate.item())
                total_loss = log_mse_loss + lam * target_power ** rpe * rate
            else:
                total_loss = log_mse_loss
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params, config.grad_clip, norm_type=2.0).item()
            optimizer.step()
            if schedule is not None:
                schedule.step()
            if ema is not None:
                if config.optimizer == 'amuse':
                    optimizer.eval()
                    ema.update(model)
                    optimizer.train()
                else:
                    ema.update(model)

            if tracker is not None:
                step = tracker.next_step()
                tracker.scalar(f"train/{label}/log_mse", log_mse_loss.item(), step)
                tracker.scalar(f"train/{label}/loss", total_loss.item(), step)
                tracker.scalar(f"train/{label}/learning_rate", optimizer.param_groups[0]['lr'], step)
                tracker.scalar(f"train/{label}/grad_norm", grad_norm, step)
                if rate_losses:
                    tracker.scalar(f"train/{label}/rate", rate_losses[-1], step)

            if i_batch % 500 == 0:
                avg = np.mean(log_mse_losses[-100:]) if log_mse_losses else 0
                lr_now = optimizer.param_groups[0]['lr']
                rate_str = f" rate={np.mean(rate_losses[-100:]):.3f}" if rate_losses else ""
                print(f"  {label} epoch {i_epoch} batch {i_batch}/{n_batches} "
                      f"log_mse={avg:.3f}{rate_str} lr={lr_now:.2e}", flush=True)

            del x, x_in, pred, target, log_mse_loss, total_loss

        if tracker is not None and log_mse_losses:
            tracker.scalar(f'train/{label}/epoch_log_mse', np.mean(log_mse_losses[epoch_loss_start:]), i_epoch)
            if len(rate_losses) > epoch_rate_start:
                tracker.scalar(f'train/{label}/epoch_rate', np.mean(rate_losses[epoch_rate_start:]), i_epoch)

        should_stop = False
        if epoch_evaluator is not None:
            if config.optimizer == 'amuse':
                optimizer.eval()
            score = epoch_evaluator(i_epoch, model)
            if early_stopper is not None:
                should_stop = early_stopper.step(score, i_epoch, model, ema)
                if should_stop:
                    print(f"  {label} early stopping at epoch {i_epoch + 1}; "
                          f"best epoch={early_stopper.best_epoch + 1}, "
                          f"best_psnr={early_stopper.best_score:.3f}", flush=True)
            if config.optimizer == 'amuse' and not should_stop:
                optimizer.train()
        if should_stop:
            break

    if config.optimizer == 'amuse':
        optimizer.eval()
    if early_stopper is not None:
        early_stopper.restore(model, ema)
    if ema is not None:
        ema.copy_to(model)

    return log_mse_losses


def validate(merged, device, val_dataset, config):
    merged.eval()
    max_ps = max(config.ps)
    psnrs, bpps = [], []
    for sample in val_dataset:
        img = sample['image']
        if config.validation_height is not None:
            img = img.resize(
                (config.validation_width, config.validation_height),
                PIL.Image.Resampling.BICUBIC,
            )
        w, h = img.size
        if h % max_ps or w % max_ps:
            raise ValueError(
                f'validation image size {w}x{h} must divide by maximum patch size {max_ps}; '
                'set --validation_height/--validation_width to aligned dimensions')
        x = pil_to_tensor(img.convert("RGB")).to(torch.float).to(device).unsqueeze(0) / 127.5 - 1.0
        x_in = srgb_to_linear(x) if config.linear_input else x
        with torch.inference_mode():
            latents = merged.encode(x_in)
            latents = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]
            xhat = merged.decode(latents).clamp(-1, 1)
        x_01 = x / 2 + 0.5
        xhat_01 = xhat / 2 + 0.5
        psnr = -10 * torch.nn.functional.mse_loss(x_01, xhat_01).log10().item()
        psnrs.append(psnr)
        n_pixels = x.shape[2] * x.shape[3]
        total_bytes = 0
        for z in latents:
            z_2d = z[0].reshape(z.shape[1] * z.shape[2], z.shape[3])
            buff = io.BytesIO()
            to_pil_image((z_2d.long() + 127).to(torch.uint8)).save(buff, format='JPEG-LS')
            total_bytes += len(buff.getbuffer())
        bpp = total_bytes * 8 / n_pixels
        bpps.append(bpp)
    mean_bpp = np.mean(bpps)
    mean_cr = 24.0 / mean_bpp
    return np.mean(psnrs), mean_cr


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, required=True)
    p.add_argument('--input_channels', type=int, default=3)
    p.add_argument('--ps', type=int, nargs='+', required=True)
    p.add_argument('--decoder_ps', type=int, default=None,
                    help='Decoder operating resolution (default: max of --ps)')
    p.add_argument('--decoder_dim', type=int, default=768)
    p.add_argument('--decoder_kernel_size', type=int, default=3)
    p.add_argument('--decoder_arch', type=str, default='CCCCCC')
    p.add_argument('--decoder_mlp_ratio', type=float, default=4)
    p.add_argument('--decoder_layerscale', type=lambda s: s.lower() in ('true','1','yes'), default=False)
    p.add_argument('--decoder_layerscale_init', type=float, default=1e-6)
    p.add_argument('--lam', type=float, nargs='+', default=[0.008])
    p.add_argument('--rpe', type=float, nargs='+', default=[0.3])
    p.add_argument('--encoder_arch', type=str, default='SC8', choices=['LF8', 'SC8'])
    p.add_argument('--linear_input', type=lambda s: s.lower() in ('true','1','yes'), default=False)
    p.add_argument('--encoder_lr_scale', type=float, nargs='+', default=[0.3])
    p.add_argument('--merged_decoder_noise', type=lambda s: s.lower() in ('true','1','yes'), default=False)
    p.add_argument('--merged_decoder_round', type=lambda s: s.lower() in ('true','1','yes'), default=True)
    p.add_argument('--epochs_single', type=int, nargs='+', default=[2])
    p.add_argument('--epochs_merged', type=int, nargs='+', default=[4])
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--min_aspect', type=float, default=1.0)
    p.add_argument('--max_aspect', type=float, default=1.0)
    p.add_argument('--min_size', type=int, default=480)
    p.add_argument('--max_size', type=int, default=480)
    p.add_argument('--min_width', type=int, default=None,
                   help='minimum training crop width; defaults to height × aspect')
    p.add_argument('--max_width', type=int, default=None,
                   help='maximum training crop width; defaults to height × aspect')
    p.add_argument('--min_scale', type=float, default=1.0)
    p.add_argument('--max_scale', type=float, default=2.0)
    p.add_argument('--validation_height', type=int, default=None,
                   help='fixed validation resize height (managed default: 480)')
    p.add_argument('--validation_width', type=int, default=None,
                   help='fixed validation resize width (managed default: 640)')
    p.add_argument('--augmentation_config', type=str, default=json.dumps({
        'transforms': [{
            'name': 'ColorJitter', 'brightness': 0.4, 'contrast': 0.0,
            'saturation': 0.4, 'hue': 0.0, 'p': 1.0,
        }],
    }), help='JSON Albumentations configuration; managed runs generate this from Hydra')
    p.add_argument('--sc_max_lr', type=float, nargs='+', default=[5e-5])
    p.add_argument('--sc_lr_pow', type=float, nargs='+', default=[2])
    p.add_argument('--md_max_lr', type=float, nargs='+', default=[5e-4])
    p.add_argument('--md_lr_pow', type=float, nargs='+', default=[2])
    p.add_argument('--min_lr', type=float, default=1e-8)
    p.add_argument('--grad_clip', type=float, default=1.5)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--save_checkpoint_name', type=str, default=None)
    p.add_argument('--train_ds', type=str, default='danjacobellis/LSDIR')
    p.add_argument('--val_ds', type=str, default='danjacobellis/kodak')
    p.add_argument('--dataset_samples', type=int, default=None)
    p.add_argument('--validation_samples', type=int, default=None)
    p.add_argument('--resume_checkpoint', type=str, default=None)
    p.add_argument('--resume_channels', type=int, default=None)
    p.add_argument('--run_dir', type=str, default=None,
                   help='Managed run directory for TensorBoard and K-best checkpoints')
    p.add_argument('--tensorboard', type=lambda s: s.lower() in ('true','1','yes'), default=False)
    p.add_argument('--keep_best_k', type=int, default=0,
                   help='Keep the K highest validation-PSNR channel checkpoints')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_batches_per_epoch', type=int, default=None,
                   help='Debug/smoke-test limit; omit for full training')
    p.add_argument('--optimizer', type=str, choices=['adan', 'amuse'], default='adan')
    p.add_argument('--ema_decay', type=float, default=0.0,
                   help='Model EMA; zero disables it to preserve paper-script behavior')
    p.add_argument('--amuse_beta', type=float, default=0.8)
    p.add_argument('--amuse_beta2', type=float, default=0.999)
    p.add_argument('--amuse_eps', type=float, default=1e-10)
    p.add_argument('--amuse_momentum', type=float, default=0.95)
    p.add_argument('--amuse_rho', type=float, default=0.3)
    p.add_argument('--amuse_r', type=float, default=0.0)
    p.add_argument('--amuse_weight_lr_power', type=float, default=2.0)
    p.add_argument('--amuse_warmup_ratio', type=float, default=0.05)
    p.add_argument('--amuse_weight_decay', type=float, default=0.0)
    p.add_argument('--amuse_weight_decay_at_y', type=float, default=0.0)
    p.add_argument('--amuse_aux_update_type', type=str, choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--amuse_muon_min_ndim', type=int, default=2)
    p.add_argument('--early_stopping', type=lambda s: s.lower() in ('true','1','yes'), default=False)
    p.add_argument('--early_stopping_patience', type=int, default=2)
    p.add_argument('--early_stopping_min_delta', type=float, default=0.01)
    p.add_argument('--early_stopping_min_epochs', type=int, default=2)
    p.add_argument('--early_stopping_samples', type=int, default=128)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError('ema_decay must be in [0, 1)')
    if not 0.0 < args.amuse_beta < 1.0:
        raise ValueError('amuse_beta must be in (0, 1)')
    if not 0.0 < args.amuse_warmup_ratio <= 1.0:
        raise ValueError('amuse_warmup_ratio must be in (0, 1]')
    if args.amuse_muon_min_ndim < 2:
        raise ValueError('amuse_muon_min_ndim must be at least 2')
    if args.early_stopping and args.early_stopping_samples < 1:
        raise ValueError('early_stopping_samples must be positive when enabled')
    if (args.min_width is None) != (args.max_width is None):
        raise ValueError('min_width and max_width must be supplied together')
    if (args.validation_height is None) != (args.validation_width is None):
        raise ValueError('validation_height and validation_width must be supplied together')
    augmentation = RGBTrainingAugmentation.from_json(args.augmentation_config)

    run_dir = os.path.abspath(args.run_dir) if args.run_dir else None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        if args.save_checkpoint_name is None:
            args.save_checkpoint_name = os.path.join(run_dir, 'checkpoints', 'last.pth.tar')
    tracker = TensorBoardTracker(
        os.path.join(run_dir, 'tensorboard') if run_dir else None,
        enabled=args.tensorboard,
    )
    atexit.register(tracker.close)

    train_dataset = datasets.load_dataset(args.train_ds, split='train')
    if args.dataset_samples:
        train_dataset = train_dataset.select(range(min(args.dataset_samples, train_dataset.num_rows)))
    val_dataset = datasets.load_dataset(args.val_ds, split='validation')
    if args.validation_samples:
        val_dataset = val_dataset.select(range(min(args.validation_samples, val_dataset.num_rows)))
    early_stopping_dataset = val_dataset
    if args.early_stopping:
        early_stopping_dataset = val_dataset.select(
            range(min(args.early_stopping_samples, val_dataset.num_rows)))
    n_channels = len(args.ps)

    config = SimpleNamespace()
    for k, v in vars(args).items():
        if k not in ('device', 'save_checkpoint_name', 'train_ds', 'resume_checkpoint', 'resume_channels'):
            setattr(config, k, v)

    for attr in ('lam', 'rpe', 'encoder_lr_scale', 'sc_max_lr', 'sc_lr_pow',
                 'md_max_lr', 'md_lr_pow', 'epochs_single', 'epochs_merged'):
        lst = getattr(config, attr)
        if len(lst) < n_channels:
            lst = lst + [lst[-1]] * (n_channels - len(lst))
            setattr(config, attr, lst)

    config.decoder_ps = args.decoder_ps or max(config.ps)
    assert all(ps % config.decoder_ps == 0 or config.decoder_ps % ps == 0
               for ps in config.ps), "All encoder ps must be multiples/divisors of decoder_ps"

    num_batches = train_dataset.num_rows // config.batch_size
    if num_batches < 1:
        raise ValueError(
            f"training split has {train_dataset.num_rows} samples, fewer than batch_size={config.batch_size}")
    if config.max_batches_per_epoch is not None:
        if config.max_batches_per_epoch < 1:
            raise ValueError("max_batches_per_epoch must be positive")
        num_batches = min(num_batches, config.max_batches_per_epoch)
    config.save_checkpoint_name = args.save_checkpoint_name or f'checkpoint_patchify_{device}.pth'
    config.train_ds = args.train_ds
    config.val_ds = args.val_ds
    max_ps = max(config.ps)
    checkpoint_dir = os.path.dirname(os.path.abspath(config.save_checkpoint_name))
    kbest = KBestCheckpointManager(
        os.path.join(run_dir or checkpoint_dir, 'checkpoints', 'best')
        if run_dir else os.path.join(checkpoint_dir, 'best'),
        k=args.keep_best_k,
        mode='max',
    )
    if run_dir:
        atomic_json_dump({
            'schema_version': 1,
            'arguments': {k: v for k, v in vars(args).items()},
            'runtime': {
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            },
        }, os.path.join(run_dir, 'run_metadata.json'))

    all_groups = get_scale_groups(config.ps, n_channels)
    print(f"\n{'='*60}")
    print(f"  Progressive patchify training on {device}")
    print(f"  Total channels: {n_channels}")
    print(f"  Decoder ps: {config.decoder_ps}")
    print(f"  Scale groups:")
    for s, (ps_s, start, end) in enumerate(all_groups):
        n_g = end - start
        dec_ch_g = n_g * decoder_channels_per_encoder(ps_s, config.decoder_ps)
        if ps_s < config.decoder_ps:
            op = f"patchify {config.decoder_ps // ps_s}x"
        elif ps_s > config.decoder_ps:
            op = f"upsample {ps_s // config.decoder_ps}x"
        else:
            op = "identity"
        print(f"    Group {s}: ps={ps_s}, ch {start}-{end-1} ({n_g}ch), "
              f"{op} -> {dec_ch_g} decoder ch")
    print(f"  decoder_dim={config.decoder_dim}, mlp_ratio={config.decoder_mlp_ratio}")
    print(f"  decoder_arch={config.decoder_arch}")
    print(f"  encoder_arch={config.encoder_arch}")
    print(f"  optimizer={config.optimizer}, ema_decay={config.ema_decay}")
    print(f"  Albumentations={augmentation.describe()}")
    if args.early_stopping:
        print(f"  early_stopping=PSNR patience={args.early_stopping_patience} "
              f"min_delta={args.early_stopping_min_delta} "
              f"monitor_samples={early_stopping_dataset.num_rows}")
    print(f"  lam={config.lam}")
    print(f"  sc_max_lr={config.sc_max_lr}")
    print(f"  md_max_lr={config.md_max_lr}")
    print(f"  epochs_single={config.epochs_single}")
    print(f"  epochs_merged={config.epochs_merged}")
    print(f"  train_ds={args.train_ds} ({train_dataset.num_rows} samples)")
    print(f"  val_ds={args.val_ds}")
    if args.resume_checkpoint:
        print(f"  resume_checkpoint={args.resume_checkpoint} ({args.resume_channels} channels)")
    print(f"{'='*60}\n", flush=True)

    def train_collate_fn(batch):
        aspect = np.random.uniform(config.min_aspect, config.max_aspect)
        h = np.random.uniform(config.min_size, config.max_size)
        w = (np.random.uniform(config.min_width, config.max_width)
             if config.min_width is not None else h * aspect)
        # Each convolutional patch encoder needs both sides divisible by max_ps.
        # The managed 640×480 setting is already exactly aligned.
        h = max_ps * max(1, int(np.rint(h / max_ps)))
        w = max_ps * max(1, int(np.rint(w / max_ps)))
        x = []
        for sample in batch:
            image = augmentation(
                sample['image'], height=h, width=w,
                scale=np.random.uniform(config.min_scale, config.max_scale),
                max_size=1 + int(config.max_scale * max(
                    config.max_size,
                    config.max_width if config.max_width is not None else config.max_size * config.max_aspect,
                )),
            )
            xi = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
            x.append(xi)
        return torch.cat(x).to(torch.float) / 127.5 - 1.0

    loader_index = 0

    def make_dataloader():
        nonlocal loader_index
        generator = torch.Generator()
        generator.manual_seed(config.seed + loader_index)
        loader_index += 1
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            drop_last=True, shuffle=True, collate_fn=train_collate_fn,
            generator=generator, worker_init_fn=seed_worker,
        )

    if args.resume_checkpoint:
        assert args.resume_channels is not None, "--resume_channels required with --resume_checkpoint"
        start_channel = args.resume_channels
        ckpt = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        old_config = ckpt['config']

        assert start_channel <= len(ckpt['merged_decoder_weights']), \
            f"Checkpoint has {len(ckpt['merged_decoder_weights'])} trained channels, requested {start_channel}"
        old_decoder_ps = getattr(old_config, 'decoder_ps', max(old_config.ps))
        assert old_decoder_ps == config.decoder_ps, \
            f"decoder_ps mismatch: checkpoint={old_decoder_ps}, new={config.decoder_ps}"
        assert old_config.ps[:start_channel] == config.ps[:start_channel], \
            f"ps mismatch for resumed channels: checkpoint={old_config.ps[:start_channel]}, new={config.ps[:start_channel]}"
        assert old_config.encoder_arch == config.encoder_arch, \
            f"encoder_arch mismatch: checkpoint={old_config.encoder_arch}, new={config.encoder_arch}"
        assert old_config.input_channels == config.input_channels, \
            f"input_channels mismatch: checkpoint={old_config.input_channels}, new={config.input_channels}"
        # Handle old checkpoint format (list-valued decoder args)
        _s = lambda c, a: (getattr(c, a)[0] if isinstance(getattr(c, a), list) else getattr(c, a))
        assert _s(old_config, 'decoder_dim') == config.decoder_dim, \
            f"decoder_dim mismatch: checkpoint={_s(old_config, 'decoder_dim')}, new={config.decoder_dim}"
        assert _s(old_config, 'decoder_kernel_size') == config.decoder_kernel_size, \
            f"decoder_kernel_size mismatch: checkpoint={_s(old_config, 'decoder_kernel_size')}, new={config.decoder_kernel_size}"
        assert _s(old_config, 'decoder_arch') == config.decoder_arch, \
            f"decoder_arch mismatch: checkpoint={_s(old_config, 'decoder_arch')}, new={config.decoder_arch}"
        assert _s(old_config, 'decoder_mlp_ratio') == config.decoder_mlp_ratio, \
            f"decoder_mlp_ratio mismatch: checkpoint={_s(old_config, 'decoder_mlp_ratio')}, new={config.decoder_mlp_ratio}"
        old_ls = getattr(old_config, 'decoder_layerscale', False)
        assert old_ls == config.decoder_layerscale, \
            f"decoder_layerscale mismatch: checkpoint={old_ls}, new={config.decoder_layerscale}"

        train_losses = ckpt['train_losses'][:start_channel * 2]
        single_channel_weights = ckpt['single_channel_weights'][:start_channel]
        merged_decoder_weights = ckpt['merged_decoder_weights'][:start_channel]
        valid_psnr = ckpt['valid_psnr'][:start_channel]
        valid_cr = ckpt['valid_cr'][:start_channel]

        merged = MergedAutoencoder(config, start_channel).to(device)
        mdw = merged_decoder_weights[-1]
        for s, enc_sd in enumerate(mdw['encoder_weights']):
            merged.encoders[s].load_state_dict(enc_sd)
        merged.decoder.load_state_dict(mdw['decoder_weights'])

        print(f"Resumed from {args.resume_checkpoint} ({start_channel} channels)")
        for i, (p, c) in enumerate(zip(valid_psnr, valid_cr)):
            print(f"  [ch{i}] psnr={p:.2f} dB  cr={c:.2f}")
        del ckpt
    else:
        start_channel = 0
        merged = None
        train_losses = []
        single_channel_weights = []
        merged_decoder_weights = []
        valid_psnr = []
        valid_cr = []

    print(f"Startup took {time.time()-t0:.0f}s. Starting training...\n", flush=True)

    for i_channel in range(start_channel, n_channels):
        n_ch = i_channel + 1

        # ── Phase 1: Single-channel residual training ──
        ch_config = SimpleNamespace(
            input_channels=config.input_channels, ps=config.ps[i_channel],
            decoder_ps=config.decoder_ps,
            decoder_dim=config.decoder_dim,
            decoder_kernel_size=config.decoder_kernel_size,
            decoder_arch=config.decoder_arch,
            decoder_mlp_ratio=config.decoder_mlp_ratio,
            decoder_layerscale=config.decoder_layerscale,
            decoder_layerscale_init=config.decoder_layerscale_init,
            encoder_arch=config.encoder_arch,
        )
        single = AutoencoderSingleChannel(ch_config).to(device)

        if merged is not None:
            merged.eval()
            for p in merged.parameters():
                p.requires_grad_(False)
            single._target = lambda x_in, x, _m=merged: x - _m(x_in)
        else:
            single._target = lambda x_in, x: x.clone()

        enc_ps = config.ps[i_channel]
        dec_ch = decoder_channels_per_encoder(enc_ps, config.decoder_ps)
        print(f"\n--- Channel {i_channel} (ps={enc_ps}, "
              f"{dec_ch} dec ch): "
              f"training single-channel ---", flush=True)

        dataloader = make_dataloader()
        enc_params = list(single.analysis_transform.parameters())
        dec_params = [p for p in single.parameters() if not any(p is q for q in enc_params)]
        total_steps_single = config.epochs_single[i_channel] * num_batches
        sc_losses = train_one(
            single,
            [{'params': enc_params, 'lr': config.encoder_lr_scale[i_channel]},
             {'params': dec_params, 'lr': 1.0}],
            dataloader, device, config,
            total_steps_single, config.epochs_single[i_channel], f"ch{i_channel}",
            lam=config.lam[i_channel], max_lr=config.sc_max_lr[i_channel],
            lr_pow=config.sc_lr_pow[i_channel], rpe=config.rpe[i_channel], tracker=tracker,
        )
        train_losses.append(sc_losses)
        single_channel_weights.append({k: v.cpu() for k, v in single.state_dict().items()})

        # ── Phase 2: Merged decoder training ──
        total_dec_ch = sum(
            (end - start) * decoder_channels_per_encoder(ps_s, config.decoder_ps)
            for ps_s, start, end in get_scale_groups(config.ps, n_ch))
        print(f"\n--- Channel {i_channel}: merging encoder, training merged decoder "
              f"({total_dec_ch} decoder ch) ---", flush=True)

        new_merged = MergedAutoencoder(config, n_ch).to(device)
        new_groups = new_merged.scale_groups
        current_scale_idx = len(new_groups) - 1

        # Copy encoder weights from previous model
        if merged is not None:
            prev_groups = merged.scale_groups
            for s in range(len(prev_groups)):
                prev_n = prev_groups[s][2] - prev_groups[s][1]
                new_n = new_groups[s][2] - new_groups[s][1]
                if prev_n == new_n:
                    new_merged.encoders[s].load_state_dict(merged.encoders[s].state_dict())
                else:
                    new_merged.encoders[s][0].weight.data[:prev_n] = merged.encoders[s][0].weight.data
                    new_merged.encoders[s][0].bias.data[:prev_n] = merged.encoders[s][0].bias.data
                    for (mk, mp), (sk, sp) in zip(new_merged.encoders[s][1].named_parameters(),
                                                   merged.encoders[s][1].named_parameters()):
                        mp.data[:prev_n] = sp.data

        # Merge new channel's encoder
        ch_in_scale = i_channel - new_groups[current_scale_idx][1]
        merge_encoder(new_merged, single, current_scale_idx, ch_in_scale)

        # Free old models
        del single
        if merged is not None:
            del merged
        merged = None
        torch.cuda.empty_cache()

        # Freeze all encoders
        for enc in new_merged.encoders:
            for p in enc.parameters():
                p.requires_grad_(False)

        new_merged._target = lambda x_in, x: x.clone()
        original_forward = new_merged.forward

        def _decoder_forward(x, _m=new_merged):
            if config.merged_decoder_noise:
                for enc in _m.encoders:
                    enc.train()
            else:
                for enc in _m.encoders:
                    enc.eval()
            with torch.no_grad():
                latents = _m.encode(x)
            if config.merged_decoder_round:
                latents = [z.round() for z in latents]
            return _m.decode(latents)
        new_merged.forward = _decoder_forward

        decoder_params = list(new_merged.decoder.parameters())
        dataloader = make_dataloader()
        total_steps_merged = config.epochs_merged[i_channel] * num_batches
        early_stopper = None
        epoch_evaluator = None
        if args.early_stopping:
            early_stopper = EarlyStopping(
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                min_epochs=args.early_stopping_min_epochs,
            )

            def epoch_evaluator(i_epoch, _model, _channel=i_channel, _n_ch=n_ch):
                epoch_psnr, epoch_cr = validate(_model, device, early_stopping_dataset, config)
                tracker.scalar(f'early_stopping/ch{_channel}/psnr', epoch_psnr, i_epoch + 1)
                tracker.scalar(f'early_stopping/ch{_channel}/compression_ratio', epoch_cr, i_epoch + 1)
                print(f"  merge{_n_ch}ch epoch {i_epoch + 1} "
                      f"monitor_psnr={epoch_psnr:.3f} monitor_cr={epoch_cr:.2f}", flush=True)
                return epoch_psnr
        merge_losses = train_one(
            new_merged, decoder_params, dataloader, device, config,
            total_steps_merged, config.epochs_merged[i_channel], f"merge{n_ch}ch",
            lam=0.0, max_lr=config.md_max_lr[i_channel], lr_pow=config.md_lr_pow[i_channel],
            tracker=tracker, epoch_evaluator=epoch_evaluator, early_stopper=early_stopper,
        )
        train_losses.append(merge_losses)

        new_merged.forward = original_forward

        # Save merged decoder weights
        merged_decoder_weights.append({
            'encoder_weights': [{k: v.cpu() for k, v in enc.state_dict().items()}
                                for enc in new_merged.encoders],
            'decoder_weights': {k: v.cpu() for k, v in new_merged.decoder.state_dict().items()},
            'scale_groups': [(ps_s, end-start) for ps_s, start, end in new_groups],
            'adapt_factors': new_merged.adapt_factors,
            'total_decoder_ch': new_merged.n_ch,
        })

        merged = new_merged

        # Validation
        mean_psnr, mean_cr = validate(merged, device, val_dataset, config)
        valid_psnr.append(mean_psnr)
        valid_cr.append(mean_cr)
        tracker.scalar('validation/psnr', mean_psnr, i_channel + 1)
        tracker.scalar('validation/compression_ratio', mean_cr, i_channel + 1)
        tracker.scalar('validation/bpp', 24.0 / mean_cr, i_channel + 1)
        print(f"[ch{i_channel}] val psnr={mean_psnr:.2f} dB  cr={mean_cr:.2f}", flush=True)

        checkpoint_payload = {
            'i_channel': i_channel,
            'config': config,
            'train_losses': train_losses,
            'valid_psnr': valid_psnr,
            'valid_cr': valid_cr,
            'single_channel_weights': single_channel_weights,
            'merged_decoder_weights': merged_decoder_weights,
        }
        atomic_torch_save(checkpoint_payload, config.save_checkpoint_name)
        kbest.consider(mean_psnr, i_channel + 1, checkpoint_payload)

    total_time = time.time() - t0

    r_avif = [0.0441157, 0.06284926, 0.09121535, 0.22538418, 0.3614841, 0.67526669, 1.44184875, 2.65533447]
    d_avif = [24.23690836, 25.25622596, 26.32338345, 29.26768998, 31.15876267, 34.19219484, 38.55725398, 41.76542381]
    r_rae = np.array([24.0 / cr for cr in valid_cr])
    d_rae = np.array(valid_psnr)
    keep = [not any(r_rae[j] <= r_rae[i] and d_rae[j] >= d_rae[i] and j != i
                    for j in range(len(r_rae))) for i in range(len(r_rae))]
    r_rae_p, d_rae_p = r_rae[keep], d_rae[keep]
    try:
        bd_avif = bd.bd_rate(r_avif, d_avif, r_rae_p.tolist(), d_rae_p.tolist(),
                             method='pchip', require_matching_points=False, min_overlap=0.15)
        if np.isfinite(bd_avif):
            print(f"  BD-rate vs AVIF: {bd_avif:+.1f}%", flush=True)
        else:
            bd_avif = None
            print("  BD-rate vs AVIF: unavailable (insufficient overlapping points)", flush=True)
    except Exception as e:
        bd_avif = None
        print(f"  BD-rate vs AVIF: failed ({e})", flush=True)

    results = {
        'device': device,
        'config': {k: v for k, v in vars(config).items()},
        'valid_psnr': valid_psnr,
        'valid_cr': valid_cr,
        'final_psnr': valid_psnr[-1] if valid_psnr else None,
        'final_cr': valid_cr[-1] if valid_cr else None,
        'bd_avif': bd_avif,
        'total_time_hours': total_time / 3600,
    }
    if config.save_checkpoint_name.endswith('.pth.tar'):
        results_path = config.save_checkpoint_name[:-8] + '_results.json'
    elif config.save_checkpoint_name.endswith('.pth'):
        results_path = config.save_checkpoint_name[:-4] + '_results.json'
    else:
        results_path = config.save_checkpoint_name + '_results.json'
    atomic_json_dump(results, results_path)

    print(f"\n{'='*60}")
    print(f"  FINISHED")
    if valid_psnr:
        print(f"  Final PSNR: {valid_psnr[-1]:.2f} dB  CR: {valid_cr[-1]:.2f}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Results saved to {results_path}")
    print(f"{'='*60}\n", flush=True)
    tracker.close()


if __name__ == '__main__':
    main()
