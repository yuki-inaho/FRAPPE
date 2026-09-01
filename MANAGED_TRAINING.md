# Managed training on anonymous local RGB data

This workflow keeps the paper's `train_rae_progressive.py` command-line
interface available while adding a Hydra entry point, TensorBoard logging,
atomic checkpoints, and K-best retention.

## 1. Prepare the dataset

Do not put source archive names in a repository config or command log. Supply
the four local paths through private shell variables, in the intended
`dataset_001` through `dataset_004` order:

```bash
pixi run prepare-data "$ARCHIVE_1" "$ARCHIVE_2" "$ARCHIVE_3" "$ARCHIVE_4" \
  --output /home/kasm-user/Desktop/data/frappe_rgb
```

The tool streams only Color/RGB members from ZIP/TAR files, decodes them into
fresh RGB objects, and writes metadata-free PNG files with anonymous sequential
names. Raw archive paths and member names are not persisted. The generated
Hugging Face ImageFolder layout is:

```text
/home/kasm-user/Desktop/data/frappe_rgb/imagefolder/
├── train/
├── validation/
└── test/
```

The canonical per-source copies use generic `dataset_001` through
`dataset_004` identifiers. ImageFolder views are hard links, so they do not
duplicate the PNG payloads. Then generate the fixed-size dataset used by the
managed configuration:

```bash
pixi run resize-data
```

This preserves the anonymous 800×600 source dataset and creates
`/home/kasm-user/Desktop/data/frappe_rgb_640x480/imagefolder`. Each fresh PNG
is RGB, metadata-free, anonymously named, and physically resized with bicubic
interpolation to width 640 × height 480. Both dimensions divide by the default
maximum patch size (32), so neither training nor validation crops away rows or
columns.

## 2. Run a bounded GPU smoke test

```bash
pixi run train-managed \
  run.id=smoke_001 \
  'model.ps=[32]' \
  model.decoder_ps=32 \
  model.decoder_dim=64 \
  model.decoder_arch=C \
  'training.iterations_single=[1]' \
  'training.iterations_merged=[1]' \
  validation.every_iterations=1 \
  training.dataset_samples=1 \
  training.validation_samples=1 \
  training.batch_size=1 \
  training.max_batches_per_epoch=1 \
  training.num_workers=0 \
  checkpoint.keep_best_k=1
```

Remove the sample and batch limits for a real run.

## 3. Train the documented nine-channel smoke architecture

```bash
pixi run train-managed run.id=progressive_9ch_001
```

## 4. Train the released 21-channel architecture

```bash
pixi run train-managed model=progressive_21ch run.id=progressive_21ch_001
```

The 21-channel preset matches the released model architecture. It does not
claim exact paper-training reproduction because the complete per-channel
lambda, learning-rate, and epoch invocation was not published. Override those
lists only after a small pilot study.

## 5. Prepare an 800×608, eight-hour-class 21-channel run

The 21-channel long-run preset selects the released architecture, the
separately materialized anonymous 800×608 ImageFolder, AMUSE, and EMA=0.99.
It is fixed-update based: 100 single-channel plus 220 merged-decoder updates
per channel (6,720 updates total). On the exclusive RTX 5090 this is planned
as an approximately eight-hour run at 800×608; actual duration depends on
hardware and data-loader throughput. It does not begin merely by creating the
configuration.

First prepare the data once (the source is already anonymous RGB PNG data):

```bash
pixi run resize-data \
  --source /home/kasm-user/Desktop/data/frappe_rgb \
  --output /workspace/data/frappe_rgb_800x608 \
  --width 800 --height 608
```

When ready to occupy the GPU, launch the complete recipe:

```bash
pixi run train-managed \
  experiment=iteration_21ch_8h \
  run.id=progressive_21ch_800x608_001
```

The preset performs bounded 64-image early-stopping checks and 128-image
channel validation to preserve the time budget, retains the five best
channel-level checkpoints, and writes a resume-safe `last.pth.tar`. After the
run, evaluate the complete validation and test splits with:

```bash
pixi run python tools/evaluate_local_checkpoint.py \
  --checkpoint runs/progressive_21ch_800x608_001/checkpoints/last.pth.tar \
  --dataset-root /workspace/data/frappe_rgb_800x608/imagefolder
```

### Five-hour 21-channel run without EMA

The calibrated no-EMA run uses the same released 21-channel/five-scale model,
12-block LayerScale decoder and 800×608 data.  It enables early stopping only
after the 128-image monitor reaches 40 dB; below that floor it neither stops
nor restores a low-quality checkpoint.  It uses batch 8 (batch 10 OOMed in
the 800×608 preflight), and loads shuffled batches with eight worker processes:

```bash
pixi run train-managed \
  experiment=iteration_21ch_5h_noema \
  run.id=progressive_21ch_800x608_5h_noema_001
```

Its fixed budget is 750 single-channel plus 1,700 merged-decoder updates per
channel (51,450 total).  The preflight measured 0.317 seconds of GPU work per
full-model update; the five-hour estimate includes PNG loading,
Albumentations, validation, and cumulative checkpoint writes.  The training
DataLoader has deterministic `shuffle=True` in every phase.

## Exporting a reconstructed validation image

Export an anonymous validation sample without copying its source filename into
the result.  `--output` is the actual model reconstruction; the optional
comparison image places the reference beside it for visual inspection.

```bash
pixi run python tools/export_local_reconstruction.py \
  --checkpoint runs/iteration_9ch_1h_001/checkpoints/last.pth.tar \
  --dataset-root /home/kasm-user/Desktop/data/frappe_rgb_640x480/imagefolder \
  --split validation --index 0 \
  --output /home/kasm-user/Desktop/frappe_9ch_validation_reconstruction.png \
  --comparison-output /home/kasm-user/Desktop/frappe_9ch_validation_comparison.png
```

For a non-cherry-picked representative sample, replace `--index 0` with
`--representative`.  It scores the requested split and chooses the image whose
per-image PSNR is closest to the split median.

To inspect what a reported PSNR means visually, use `--target-psnr 14.0` in
place of `--index 0`.  This scans the split and exports the image closest to
14.0 dB; the sidecar metadata records the chosen anonymous index and exact PSNR.

## Merged-decoder warm start

When a channel is added, the published script builds a fresh
`MergedAutoencoder` and trains its decoder from scratch, so every channel pays
for relearning a decoder that already worked. `training.decoder_warm_start`
controls that:

| value | behaviour |
| --- | --- |
| `none` | reinitialise the merged decoder (the published behaviour) |
| `copy` | reuse the previous decoder, leave the new input columns at their random init |
| `zero_expand` | reuse it and zero the new input columns (managed default) |

`zero_expand` widens the first decoder convolution with zero columns, which
leaves the previous prefix's function bit-identical at the moment of widening
regardless of what the new encoder emits — the new channel starts from a working
codec instead of from noise.
`tests/test_prefix_model.py::test_stagewise_widening_preserves_the_previous_prefix_exactly`
asserts that equality. The direct CLI keeps `none` so the published behaviour is
still reachable without a flag:

```bash
pixi run train-managed training.decoder_warm_start=none run.id=published_behaviour
```

For training that abandons the channel-at-a-time schedule altogether, see
[JOINT_PREFIX_TRAINING.md](JOINT_PREFIX_TRAINING.md).

## AMUSE and EMA

AMUSE is vendored from the official Apache-2.0 implementation at a pinned
commit because its repository is not an installable Python package. It is the
managed-training default with an EMA of 0.99; select it explicitly when needed:

```bash
pixi run train-managed \
  experiment=amuse_ema \
  run.id=amuse_001
```

For the historical Adan behavior, use `experiment=managed` or set
`optimization.optimizer=adan` and `optimization.ema_decay=0.0`.

AMUSE is schedule-free: its internal warmup is controlled by
`optimization.amuse_warmup_ratio`, and the Adan cosine controls
`sc_lr_pow`/`md_lr_pow` do not apply. Matrix/convolution weights use the
official Muon update; scalar and vector parameters use the configured AMUSE
auxiliary update (`adamw` by default).

The managed default batches were measured at 640×480 on the exclusively
assigned RTX 5090, including AMUSE optimizer state and EMA:

| preset | selected batch | measured safe VRAM use |
| --- | ---: | ---: |
| `model=smoke` (9 channels) | 288 | 27.1 GiB reserved |
| `model=progressive_21ch` | 12 | 26.1 GiB reserved |

The 9-channel profile fit batch 368 but batch 384 OOMed. The 21-channel
profile fit batch 14 narrowly and batch 15 OOMed; defaults intentionally leave
headroom. Re-measure after changing architecture, image size, optimizer, EMA,
or GPU:

```bash
pixi run python tools/benchmark_batch_size.py \
  --profile managed_9ch --height 480 --width 640 --candidates 256 288 320 352 384
```

## Data augmentation

The network input is always width 640 × height 480. Resize/crop geometry and
Albumentations transforms are selected as a Hydra configuration group, so
augmentation settings are captured in each run's Hydra output and
`run_metadata.json`.

| profile | use | transforms |
| --- | --- | --- |
| `rgb_default` | default natural RGB training | color jitter, HSV/RGB colour shifts, channel shuffle, H/V flips, gamma |
| `rgb_strong` | orientation/colour order are not semantically fixed | stronger colour changes, channel shuffle, H/V flips, occasional gray |
| `geometry_only` | disable colour/orientation DA | only resize and random crop |

For example:

```bash
pixi run train-managed augmentation=rgb_strong run.id=rgb_strong_001
pixi run train-managed augmentation=geometry_only run.id=geometry_only_001
pixi run train-managed augmentation.transforms.channel_shuffle.p=0.25 run.id=shuffle_025
```

Supported transform names are `ColorJitter`, `HueSaturationValue`, `RGBShift`,
`ChannelShuffle`, `HorizontalFlip`, `VerticalFlip`, `RandomGamma`, `ToGray`,
and `ToSepia`. Profiles live in `configs/augmentation/`; invalid transform
names or parameters fail before the dataset/GPU training loop starts.

## Iteration control and validation

Managed FRAPPE training is controlled by fixed optimizer-update counts, not
dataset epochs. Each progressive channel has a single-channel and a
merged-decoder phase. A shorter list repeats its final value for later
channels, so the schedule stays identical if the dataset size changes:

```bash
pixi run train-managed \
  'training.iterations_single=[106,106,159]' \
  'training.iterations_merged=[212,212,265]' \
  validation.every_iterations=53 \
  run.id=iteration_control_001
```

Validation/early-stopping checks occur every
`validation.every_iterations` merged-decoder updates, plus the phase's final
update. TensorBoard records per-update loss and validation metrics at those
iteration numbers. `last.pth.tar` remains the valid resume boundary because a
new channel's encoder and merged decoder must be committed together.

For the real-data approximately 30-minute pilot on one channel, use:

```bash
pixi run train-managed \
  experiment=iteration_trial_30m \
  run.id=iteration_trial_30m_001
```

Its 200 single-channel plus 500 merged-decoder updates were calibrated from
the measured 640×480 RTX 5090 throughput. It validates after merged iterations
125, 250, 375, and 500 (unless early stopping ends it first).

## Early stopping

Early stopping is enabled by default for the merged-decoder phase. At every
iteration-based validation check it measures PSNR on a fixed first 128-image
validation subset and stops after two non-improving checks (`min_delta=0.01
dB`, with at least two checks). The best raw model and its EMA state are
restored before full validation and checkpointing. It is intentionally
per-channel: `last.pth.tar` still remains the completed-channel resume boundary.

```bash
pixi run train-managed early_stopping.enabled=false run.id=fixed_iterations_001
pixi run train-managed early_stopping.patience=4 early_stopping.samples=256 run.id=patience_4
```

## Checkpoints and resumption

Each managed run writes:

```text
runs/<run-id>/
├── checkpoints/
│   ├── last.pth.tar
│   └── best/
│       ├── index.json
│       └── best_step*.pth.tar
├── tensorboard/
└── run_metadata.json
```

`last.pth.tar` is the channel-level resume checkpoint. K-best entries are ranked
by validation PSNR and are kept separately. FRAPPE resumption continues by
adding channels; it is not conventional fine-tuning of already completed
channels.

To resume, set both values:

```bash
pixi run train-managed \
  run.id=progressive_21ch_001 \
  checkpoint.resume_checkpoint=/path/to/last.pth.tar \
  checkpoint.resume_channels=NUMBER
```

## TensorBoard

```bash
pixi run tensorboard
```

Open the displayed local URL. Training loss, rate proxy, learning rate,
gradient norm, validation PSNR, compression ratio, and bpp are recorded.

## Tests

```bash
pixi run test
```

Tests use synthetic images and archives. Private training images are never
copied into the repository.
