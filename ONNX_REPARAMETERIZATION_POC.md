# FRAPPE ONNX reparameterization PoC

This branch tests deployment-only graph rewrites. It does not alter the PyTorch
model, checkpoint, training path, or bitstream format. The implementation is in
`src/compressors/frappe/harness/onnx_reparameterization.py`; the guarded CLI is
`tools/reparameterize_onnx.py`.

## Conclusion

Four rewrites were evaluated. They should not be enabled as one undifferentiated
"optimizer" switch.

| Rewrite | Mathematical status | CR-50 OpenVINO CPU result | Recommendation |
|---|---|---|---|
| input `X / 127.5 - 1` into five analysis Convs | exact over real arithmetic; float evaluation order changes | 0.103% of latent values differ by 1; -0.00315 dB and -0.00048 bpp over 32 images | opt-in only, gated by rate-distortion validation |
| frozen prefix mask/scale/bias into first Conv | exact affine reparameterization | final uint8 output is bit-exact | adopt when the exact pattern is present |
| non-overlapping Deconv into phase Conv + DepthToSpace | exact polyphase decomposition | 4.64% of final pixels differ by 1; -0.00016 dB; indicative decoder median 24.52 to 22.40 ms | promising target-specific option, not a generic pass |
| expanded tanh-GELU into ONNX `Gelu` | same tanh approximation | final OpenVINO uint8 output is bit-exact, but no CPU speedup | useful canonicalization; not a CPU optimization |

The most important backend finding is that OpenVINO CPU already fuses both the
expanded and canonical GELU with the preceding convolution. Their compiled
runtime graphs have the same operation inventory, and each of the six GELUs is
listed in the corresponding convolution's `originalLayersNames`. Replacing the
ONNX arithmetic therefore improves graph readability but does not create a new
CPU fusion.

The latency figures above are warm, serial, single-process samples on one host,
not release claims. NPU performance must be measured on an NPU; CPU results must
not be extrapolated.

## Why onnxsim and onnxoptimizer are not enough

ONNX Simplifier 0.7.3 was run against both released encoders and decoders. It
did not remove the encoder's `Cast -> Div -> Sub`, did not absorb the decoder's
first post-Conv `Add`, and did not replace `ConvTranspose`. The encoder node
count remained 91 for CR-50 and 92 for CR-40; the decoder retained one
`ConvTranspose`, zero `DepthToSpace`, and 20 regular Convs.

This is expected. These are weight-changing algebraic rewrites, not constant
folding or redundant-shape cleanup. The public ONNX Optimizer pass registry has
no `ConvTranspose -> Conv + DepthToSpace` or input-affine-to-Conv pass. A custom
[ONNXScript rewriter pattern](https://microsoft.github.io/onnxscript/tutorial/rewriter/rewrite_patterns.html)
would be a reasonable production framework, but it would still need the same
guards, weight mapping, and backend validation implemented here. Relevant
operator contracts are [DepthToSpace](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html)
and [Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html).

## Proven transformations

### 1. Input normalization absorption

For a padding-free, group-one analysis Conv,

```text
y = Conv(X / s - o, W, b)
W' = W / s
b' = b - o * sum(W, axes=(input_channel, kernel_h, kernel_w))
y = Conv(float(X), W', b')
```

The implementation accepts only direct rank-4 weight initializers, a direct
bias, zero padding, dilation one, group one, and the expected constants
`s=127.5`, `o=1`. The uint8-to-float Cast remains. It also verifies that the
removed `Div` and `Sub` do not have hidden consumers or graph outputs.

This rewrite is not bit-exact on every runtime because the original and folded
forms use different floating-point reduction order and potentially different
kernels. On CR-50, ONNX Runtime happened to preserve all integer planes over
eight samples, while OpenVINO CPU changed 2,831 of 2,694,200 values by one.
CR-40 also showed rare ONNX Runtime differences, so ORT equality must not be
treated as a proof.

A calibration experiment estimated the median residual of each analysis Conv
channel on 32 images and added it to the folded bias. OpenVINO code mismatch
dropped from 0.1029% to 0.0758% on calibration images and from 0.1035% to
0.0753% on a disjoint 32-image holdout. This is a useful bias-correction method,
but it does not restore bit equality. If used later, calibration and acceptance
must target final integer codes, JPEG-LS bpp, and decoded PSNR—not intermediate
float MSE alone. Keeping the two normalization ops is the correct choice when
the small speed change does not justify a changed bitstream.

### 2. Frozen prefix affine absorption

The guarded pattern is:

```text
[Mul(input_mask) ->] [reflection Pad ->] Conv -> [Mul(output_scale) ->]
[Add(output_bias)]
```

For input mask `m`, output scale `a`, and output offset `c`:

```text
W'[out, in, kh, kw] = a[out] * m[in] * W[out, in, kh, kw]
b'[out] = a[out] * b[out] + c[out]
```

The released full-prefix decoder has already eliminated the runtime mask and
scale. Its remaining pattern is reflection Pad, first Conv, then one constant
Add. The PoC folds that Add into the bias and preserves the reflection Pad. It
is bit-exact in ONNX Runtime and OpenVINO CPU over all evaluated samples.

### 3. FRAPPE non-overlapping Deconv phase decomposition

The released decoder head has weight shape `(192, 3, 8, 8)`, kernel and stride
eight, zero padding, zero output padding, dilation one, and group one. Under
exactly those constraints it can be rewritten to a 1x1 Conv with 192 output
phase channels followed by `DepthToSpace(blocksize=8, mode="CRD")`:

```text
Q[out * r*r + kh*r + kw, in, 0, 0] = W[in, out, kh, kw]
phase_bias[out * r*r + kh*r + kw] = bias[out]
```

This is the special `kernel == stride` non-overlap case. It is not the false
claim that an arbitrary `ConvTranspose(k=4, stride=2, padding=1)` is equivalent
to a 1x1 Conv. Overlapping kernels, padding, output padding, dilation, groups,
and explicit output shapes are refused.

The preceding 1x1 Conv is deliberately not merged with the phase Conv in this
PoC. Such a merge is algebraically possible, but increases rounding drift and
couples an independently testable transformation to the decoder head rewrite.

### 4. Expanded tanh-GELU canonicalization

The matcher requires the complete exported expression, all exact data-flow
links, single-consumer intermediates, and float constants close to `3`,
`0.044715`, `sqrt(2/pi)`, `1`, and `0.5`:

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Only then is it replaced by `Gelu(x, approximate="tanh")`. Standard-domain
GELU requires opset 20, so the ONNX version converter upgrades a matched opset
18 graph before replacement. Unrelated Tanh expressions are left unchanged.
The released decoder contains six matches, reducing `Pow` from 6 to 0, `Tanh`
from 6 to 0, and `Mul` from 29 to 5.

ONNX Runtime produced a one-level final uint8 difference in 75 of 11,673,600
values in the eight-image probe. OpenVINO CPU was bit-exact because it already
recognizes both graph forms. The indicative OpenVINO median was 23.77 ms for
the expanded graph and 24.33 ms for the canonical graph, i.e. no demonstrated
speed benefit.

## Rate-distortion probe

The CR-50 OpenVINO CPU probe used 32 validation images and real JPEG-LS payload
bytes. Values are aggregate-MSE PSNR and payload-only bpp.

| Graph | PSNR dB | bpp | delta PSNR | delta bpp |
|---|---:|---:|---:|---:|
| baseline | 28.67931 | 0.473185 | — | — |
| input normalization folded | 28.67616 | 0.472703 | -0.00315 | -0.000482 |
| prefix bias folded | 28.67931 | 0.473185 | 0 | 0 |
| Deconv phase decomposition | 28.67916 | 0.473185 | -0.00016 | 0 |
| prefix + Deconv | 28.67916 | 0.473185 | -0.00016 | 0 |

The lower bpp of the input fold is not a free improvement: it is a different
integer code stream and therefore a slightly different operating point.

## Usage

The CLI refuses in-place overwrite and refuses a requested rewrite when no
proven pattern matches. It validates the resulting model, runs shape inference,
writes atomically, and emits a JSON report with SHA-256 hashes, operation counts,
match details, tensor shapes, modes, and opset changes.

```bash
python tools/reparameterize_onnx.py \
  --input build/encoder.onnx \
  --output build/encoder-input-folded.onnx \
  --fold-input-normalization

python tools/reparameterize_onnx.py \
  --input build/decoder.onnx \
  --output build/decoder-reparameterized.onnx \
  --fold-fixed-prefix \
  --fuse-tanh-gelu \
  --replace-nonoverlap-convtranspose
```

Generated ONNX and report files are evaluation artifacts and should not be
committed. A release candidate should additionally pass:

1. ONNX checker and shape inference.
2. ONNX Runtime comparison against the source graph.
3. OpenVINO comparison on the actual target device.
4. Final integer-code mismatch, JPEG-LS bpp, and reconstruction PSNR gates on a
   calibration set and a disjoint holdout.
5. Warmup-aware latency and memory measurement on the target device.

Suggested initial gates for this repository are `max_abs_code <= 1`, latent
mismatch below 0.2%, absolute delta bpp below 0.001, and absolute delta PSNR
below 0.01 dB. These are engineering gates inferred from this PoC, not values
claimed by the FRAPPE paper. A rewrite must still demonstrate a target-device
benefit; passing an accuracy gate alone is not a reason to deploy it.
