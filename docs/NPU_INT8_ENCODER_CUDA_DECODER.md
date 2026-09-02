# Deploying the INT8 encoder on an Intel NPU PC with a CUDA decoder

This document hands a packaged FRAPPE operating point to a machine that has
an Intel NPU and an NVIDIA GPU. The deployment mixes three runtimes on
purpose: OpenVINO runs the INT8 encoder, CharLS writes and PIL reads the
JPEG-LS planes, and ONNX Runtime runs the decoder under the CUDA execution
provider. Every one of those three is named by the caller; nothing falls back
to anything else.

## Architecture

```text
image (uint8, 1x3x608x800)
  │  OpenVINO  --encoder-device CPU (locally) / NPU (on the NPU PC)
  ▼
5 uint8 planes, output index order 0..4
  │  CharLS NEAR=0 lossless JPEG-LS, one stream per plane  --entropy-backend charls-native | pillow
  ▼
JPEG-LS streams (4-byte big-endian length prefixes optional)
  │  decoded back to planes, byte-exact
  ▼
ONNX Runtime decoder  --providers=["CUDAExecutionProvider"] only
  ▼
reconstruction (uint8, 1x3x608x800)
```

The encoder graph is quantized with a mixed-precision contract: the five
analysis convolutions go int8 through ONNX Q/DQ pairs (one shared image
quantizer plus one weight pair per conv), while the SoftsignCompander, the
`Round`, the `Clip` and the plane layout stay ordinary floating-point /
integer ops. They are the codec's bitstream definition; a quantizer inside
them would change what the integer codes mean. The OpenVINO IR is frozen to
`NCHW = [1, 3, 608, 800]` because a shape-fixed target compiles statically;
other resolutions are not served by this package.

Planes are bound by **output index**, never by name. The manifest's
`plane_order: [0, 1, 2, 3, 4]` is the binding; the runtime checks each
output's shape and dtype against `plane_shapes` before any use.

## Artifact layout

Produced by `tools/package_npu_int8.py`:

```text
npu_int8_800x608/
├── manifest.json            # sha256 of every artifact, plane geometry,
│                            # plane_order, prefix, calibration indices 0..31,
│                            # versions, quantization boundary, RD baseline
├── encoder_fp32.onnx        # reference graph, dynamic H and W
├── encoder_int8_qdq.onnx    # the canonical INT8 encoder (portable Q/DQ ONNX)
├── encoder_int8_800x608.xml # OpenVINO IR frozen to 800x608  ┐ static NPU input
├── encoder_int8_800x608.bin #                              └─ [1,3,608,800]
├── decoder.onnx             # planes -> reconstruction, dynamic H and W
├── package_report.json      # packaging provenance (op inventories, calibration)
└── local_*.json / npu_roundtrip.json   # roundtrip reports, written per run
```

The manifest carries relative file names only -- no absolute paths, no image
names. Every consumer verifies those hashes before compiling anything and
refuses to run on a mismatch.

## Environment (uv)

```bash
uv sync --group deploy
```

The deploy group brings torch (cu128), `onnxruntime-gpu` 1.29 and the CUDA 13
userland it links against (`nvidia-cublas>=13`, `nvidia-cuda-runtime>=13`).
Two notes about that pairing:

- `onnxruntime-gpu` 1.29 requires **CUDA 13**, so the CUDA 13 libraries ship
  as pip packages beside torch's cu128 wheels. The hybrid runtime preloads
  them by absolute path with `RTLD_GLOBAL` before creating the decoder
  session, so no `LD_LIBRARY_PATH` hand-configuration is needed.
- cuDNN stays at torch's `nvidia-cudnn-cu12` 9.x: the soname
  `libcudnn.so.9` is shared, and the CUDA 13 cuDNN wheel would overwrite
  torch's libraries in the same directory. Do not add `nvidia-cudnn-cu13`.

The decoder session is created with `providers=["CUDAExecutionProvider"]`
and its actual provider is checked after construction; a session that fell
back to CPU is an error, not a degraded run.

CharLS native needs the system library `libcharls2` (Debian/Ubuntu:
`sudo apt install libcharls2 libcharls-dev`). The loader looks in the
environment's `lib` directory first and then in the system loader paths, and
`--entropy-backend charls-native` raises rather than degrading when the
library is absent. The portable baseline is `--entropy-backend pillow`,
which always works because `pillow-jpls` bundles the same library.

## NPU PC acceptance

On a machine whose `ov.Core().available_devices` contains `NPU` (this
requires the OS-side Intel NPU driver; if `NPU` does not appear, that is a
driver or firmware problem to fix first, not a reason to switch devices):

```bash
uv run --group deploy python tools/roundtrip_hybrid.py \
  --artifact-dir runs/joint_16ch_cr50_ft/deployment/npu_int8_800x608 \
  --encoder-device NPU --entropy-backend charls-native \
  --dataset-root <anonymous imagefolder root> \
  --split validation --index 0 --images 1 \
  --report runs/joint_16ch_cr50_ft/deployment/npu_int8_800x608/npu_roundtrip.json
```

Acceptance: the report's `encoder.requested` and
`encoder.execution_devices` say NPU, `decoder.provider` is
`CUDAExecutionProvider`, `jpegls_roundtrip_exact` is `true`, and the plane
count/shape/dtype agree with the manifest. The INT8 encoder is not required
to reproduce the fp32 planes symbol for symbol -- report its PSNR/bpp against
the manifest's `rd_baseline` instead.

Local verification on the packaging machine runs the same command with
`--encoder-device CPU`. If any requested device, provider or backend is
unavailable, the command fails with the reason and never downgrades: that is
the whole point of the explicit arguments.

## What was deliberately not adopted

- **JPEG XL** as the entropy coder: measured 3.2x slower and 41% larger than
  CharLS on the real planes (see the `npu` branch's worklog).
- **JPEG-LS in OpenVINO**: the front half alone costs ~146 ms on the NPU
  against ~1.8 ms on the CPU; the arithmetic-rewrite needed for the NPU's
  broken comparison ops was verified but the offload loses on cost.
- **NPU or OpenVINO for the decoder**: the decoder is the compute-heavy side;
  ONNX Runtime's CUDA provider covers it, and an OpenVINO decoder adds a
  second compiled IR to keep in sync for no measured gain.
- **Retraining under fake quantization (QAT)** for this deployment: the
  adopted artifact is plain PTQ on the base weights; the PTQ-over-QAT-weights
  variant measured better PSNR but 6% more rate and is not a true QAT
  deployment result.