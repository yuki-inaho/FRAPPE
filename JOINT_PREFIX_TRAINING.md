# Joint prefix training for FRAPPE

`train_rae_progressive.py` trains FRAPPE the way the paper describes: one latent
channel at a time, each fitted to the residual the previous prefix leaves behind,
with the merged decoder rebuilt at every channel. `train_joint_prefix.py` trains
the *same inference architecture* with every channel present from the first
optimizer step, and every prefix supervised at once.

The inference graph is unchanged — per-scale non-overlapping linear analysis,
per-channel softsign companding, int8 codes, spatial adaption, one synthesis
transform — so a model trained either way is the same kind of codec. What
changes is the optimization.

## Why the stagewise schedule limits quality

Measure the ceiling before spending GPU hours:

```bash
pixi run prefix-ceiling --output temp/ceiling_21ch.json
```

On 800x608 anonymous RGB data with the released 21-channel schedule
(`ps=[32]*3+[16]*6+[8]*3+[4]*6+[2]*3`, 9.586 raw bpp):

| analysis | tied-linear decoder | note |
| --- | ---: | --- |
| greedy per-scale KLT of the residual | 36.43 dB | the stagewise ideal |
| the same, int8 codes | 36.11 dB | quantization costs 0.32 dB |
| jointly optimised, same structure | 42.63 dB | greedy ordering removed |
| free PCA, same symbol count | 48.95 dB | unconstrained linear bound |

The schedule is not the limit and int8 is not the limit. Greedy residual fitting
is: it costs about 6 dB before a nonlinear decoder is involved at all. That is
the whole reason this trainer exists.

## What each update does

```
x --> every encoder, once --> companding + stage quantizer --> adapt --> y
      |
      +--> prefix set S = {1, N, K sampled uniformly in log symbol count}
                |
                +--> block-prefix mask, stacked on the batch dimension
                          |
                          +--> one full-width superdecoder pass --> x_hat_n
```

Cost per update grows with `|S|`, not with the channel count. `|S| = 4` costs
about 55 ms on an RTX 5090 at 256x256 crops and batch 32.

Prefixes are sampled uniformly in `log C_n` where `C_n = sum_{i<=n} 1/p_i^2`.
One `p=2` channel carries 256x the symbols of one `p=32` channel, so sampling
the channel index uniformly would put almost every sample on rates nobody uses.

## Loss

Per sampled prefix the distortion term is `log10 MSE`, which is FRAPPE's own
objective and, unlike a plain MSE sum, does not let the lowest-rate prefix own
the gradient. On top of that:

| term | flag | default | purpose |
| --- | --- | ---: | --- |
| rate proxy | `--lam_rate` | 0.0 | symbol-weighted `log2 Std`, off by default |
| prefix distillation | `--lam_distill` | 0.0 | pull short prefixes toward the full one |
| monotonicity | `--lam_mono` | 0.05 | penalise a longer prefix that reconstructs worse |
| saturation | `--lam_sat` | 1e-3 | keep companded values inside the int8 range |

`--full_prefix_weight` (default 1.5) is the extra weight given to `n = N`.

## Quantization continuation

`--continuation A B C D` gives the training-progress boundaries of the five
stages. With the default `0.10 0.30 0.55 0.90`:

| progress | stage | forward |
| --- | --- | --- |
| `< 0.10` | Q0 float | companded value, no rounding |
| `< 0.30` | Q1 AUN | `v + U(-1/2, 1/2)` |
| `< 0.55` | Q2 soft | annealed soft rounding, alpha over `--alpha_range` |
| `< 0.90` | Q3 hard | integer forward, straight-through backward |
| `>= 0.90` | Q4 calibration | Q3 with the analysis path frozen |

Additive uniform noise is applied *after* the per-channel affine, not before it
as in the frozen `SC8` sequential. Rounding happens after the affine, and
additive uniform noise is a relaxation of rounding, so it has to live in the same
space to be a consistent relaxation.

Validation always uses the deployment path: `integer_codes` produces real int8
codes and every reported bitrate is a real JPEG-LS bitstream length.

## Initialisation

`--init klt` (default) sets the analysis filters to the deflated per-scale KLT of
the training patches and calibrates each compander from magnitude percentiles, so
training starts from the stagewise linear optimum rather than from noise. It
costs one eigendecomposition per scale. `--init random` is the ablation.

`--compander_knee` places the softsign knee at a multiple of the
`--compander_percentile` magnitude: large values approach uniform quantization
with a saturating tail, small values compand aggressively. `--compander_target`
is the code value that percentile maps to.

## Running

```bash
pixi run train-joint \
  --run_dir runs/joint_21ch_001 \
  --decoder_dim 256 --decoder_arch CCCCCC \
  --crop 256 --batch_size 32 --num_workers 16 \
  --iterations 60000 --extra_prefixes 2
```

The run directory holds `checkpoints/last.pth.tar`, `checkpoints/best/`,
`tensorboard/`, `run_metadata.json` and `latest_report.json`, matching the
managed stagewise runs. Resume with `--resume runs/<id>/checkpoints/last.pth.tar`.

Evaluate the whole split on the deployment path:

```bash
pixi run evaluate-joint \
  --checkpoint runs/joint_21ch_001/checkpoints/last.pth.tar \
  --splits validation test \
  --output runs/joint_21ch_001/evaluation.json \
  --export-reconstruction temp/joint_reconstruction.png
```

That prints the full prefix rate-distortion ladder, the number of monotonicity
violations, and the largest marginal gain, with every bitrate measured from a
real bitstream.

## Warm-starting from a stagewise checkpoint

`warm_start_from_merged` lifts a `MergedAutoencoder` trained on the first `n`
channels into the superdecoder: encoders are copied, the first decoder
convolution is widened with zero columns, and the rest of the decoder is copied
verbatim. The lifted model reproduces the stagewise codec *exactly* at prefix
`n` before any further optimization — `tests/test_prefix_model.py` asserts that
bit-exactly, along with the head-sum decomposition of the first convolution and
the claim that channels past a prefix cannot influence its reconstruction.

```bash
pixi run pytest tests/test_prefix_model.py -q
```
