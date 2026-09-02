# CR 100 / 30 dB / 200 fps development direction

This note defines the next optimisation target after the current CR-50 INT8
deployment package. It separates rate-distortion work from runtime work so that
graph optimisation does not become a substitute for improving the codec.

## Target contract

The intended deployed pipeline is:

```text
800x608 RGB
  -> OpenVINO INT8 encoder on Intel NPU
  -> lossless JPEG-LS / CharLS on CPU
  -> ONNX decoder on GPU
  -> reconstructed RGB
```

Acceptance uses the same aggregate-MSE convention as the current deployment:

- payload-only rate no greater than 0.24 bpp (CR 100 for 24-bit RGB);
- aggregate PSNR of at least 30 dB on a fixed validation set;
- 200 fps pipeline throughput as the goal and 100 fps as the floor;
- explicit OpenVINO NPU and ONNX Runtime GPU providers, without silent fallback;
- byte-exact JPEG-LS latent roundtrips.

The NPU is not available on the development PC. It can build and functionally
check the OpenVINO artifact, while final NPU timing remains an external-machine
acceptance step.

## Current gap

The deployed 16-channel INT8 model measures 28.6125 dB at 0.47411 bpp on the
16-image deployment validation sample. Reaching the target requires about a
49% rate reduction while gaining about 1.39 dB. Encoder PTQ costs only about
0.021 dB, so QAT and ONNX cleanup are not the main rate-distortion problem.

Post-training channel removal is also insufficient: an existing 21-channel
pruning curve reaches approximately 0.24074 bpp at only 20.74 dB. The target
therefore needs a model trained specifically around 0.24 bpp.

## Latent schedule experiments

For an 800x608 image, one latent channel contains `800*608/p^2` symbols. A p2
channel has 256 times as many symbols as a p32 channel. Channel count alone is
therefore not a useful rate budget; the patch-scale allocation is the important
quantity.

The first experiment should compare only these two 24-channel schedules:

| schedule | channels at `[p32,p16,p8,p4,p2]` | symbols/image |
| --- | --- | ---: |
| coarse-heavy | `[4,8,8,4,0]` | 199,500 |
| one-fine | `[4,8,7,4,1]` | 313,500 |

The current deployed 16-channel schedule is `[1,5,3,6,1]` and carries 336,775
symbols/image. A coarse-heavy 24-channel model can therefore have more feature
channels but fewer symbols and less JPEG-LS scan work.

Use `tools/analyze_prefix_ceiling.py` to screen the schedules and
`tools/analyze_rate_breakdown.py` once to separate model-rate limitations from
entropy-coder overhead. Then train both candidates with the same short
iteration budget and continue only the better candidate.

## Rate-distortion training

Train a single CR-100 operating point rather than spending capacity on every
prefix. Aim for 0.230--0.235 bpp during training to leave deployment margin.
Use MSE or log-MSE as the distortion term and the existing differentiable rate
proxy. Periodically update its multiplier with measured JPEG-LS rate through
`RateTarget`; do not invoke JPEG-LS on every training step.

First establish feasibility with a high-capacity decoder. It should reach about
30.2 dB at no more than 0.235 bpp before decoder compression begins. If it does
not, revise the latent schedule or rate training instead of optimising runtime
for an RD point that cannot meet the target.

## GPU decoder plan

The current 12-block, width-768, MLP-ratio-4 decoder is the likely throughput
bottleneck. The first student architecture should be:

```text
8 blocks / width 512 / MLP ratio 2
DWConv + BN -> 1x1 expand -> ReLU -> 1x1 shrink + BN -> residual
```

Its pointwise work is roughly 15% of the current decoder. Distil it from the
high-capacity teacher while using the same integer latents. Only if it remains
too slow should a width-384 variant be tried.

LayerNorm/GELU to BatchNorm/ReLU is a learned student conversion, not an exact
graph rewrite. ReLU is positively homogeneous and inference BatchNorm folds
exactly into a convolution; GELU is not positively homogeneous and LayerNorm
depends on each input's statistics.

After fine-tuning, perform exact deployment folding in the PyTorch model:

1. convolution plus BatchNorm;
2. LayerScale into the final pointwise convolution;
3. fixed-prefix scale and bias into the decoder's first convolution;
4. any linear RepConv branches before activation.

Use block and channel structured pruning that physically reduces tensor shapes.
Do not rely on unstructured sparsity without a matching runtime kernel.

## FRAPPE-specific head conversion

The decoder head has a 1x1 convolution followed by a non-overlapping
`ConvTranspose2d` whose kernel and stride are both eight. For this specific
composition, the two linear weights and biases can be combined offline into:

```text
fused Conv1x1 -> DepthToSpace(8) -> Hardtanh
```

This does not imply that a general transposed convolution can be replaced by a
1x1 convolution. Export only after checking numerical parity at several input
shapes. The rewrite is decoder-only and cannot change the JPEG-LS symbols.

## ONNX and runtime order

Perform semantic reparameterisation and pruning in PyTorch before ONNX export.
Apply `onnxsim` only as optional decoder cleanup after export, followed by ONNX
Runtime parity checking. Do not simplify the encoder before NNCF PTQ: its
compander exclusion currently depends on graph scope and names.

Benchmark an FP16 decoder first with ONNX Runtime CUDA. If the decoder plus
transfer cannot fit the approximately 4 ms budget left by the encoder and
JPEG-LS stages, compare the same ONNX model with TensorRT EP. Measure GPU kernel
time, transfers, end-to-end latency and pipelined throughput separately. Use I/O
binding, pinned buffers and double buffering only after the graph is stable.

## Minimal execution order

1. Screen the two latent schedules and run short target-rate pilots.
2. Train only the winning high-capacity model to the RD acceptance margin.
3. Distil the 8x512 BN/ReLU decoder and structurally prune it once.
4. Fold deploy-time affine operations and convert the decoder head.
5. Export and optionally simplify the decoder ONNX; verify parity.
6. Apply NNCF ONNX PTQ to the encoder and create the static OpenVINO IR.
7. Verify the hybrid path locally, then measure NPU and pipeline performance on
   the target PC.

QAT should be reconsidered only if the new encoder loses more than 0.1 dB after
PTQ. Avoid broad hyperparameter sweeps, unconditional deconvolution rewrites,
encoder simplification before PTQ, and pruning that does not reduce deployed
tensor dimensions.
