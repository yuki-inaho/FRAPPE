#!/usr/bin/env python3
"""Export a joint-prefix FRAPPE codec to ONNX and check it against onnxruntime.

The codec is exported as two graphs, because that is how it is deployed: the
analysis side has to run on a cheap CPU while the synthesis side may run
anywhere.

``<stem>_encoder.onnx``
    ``image (N, 3, H, W)`` float in ``[-1, 1]``  ->  one ``int8`` code tensor per
    scale group.  This is the whole encoder: strided convolution, softsign
    companding, per-channel affine, rounding and clipping into the signed 8-bit
    range.  Its outputs are exactly the symbols the entropy coder sees, so the
    ONNX graph ends where JPEG-LS begins.

``<stem>_decoder.onnx``
    the same ``int8`` code tensors  ->  ``reconstruction (N, 3, H, W)``.  Spatial
    adaption, channel concatenation, the prefix scale/bias, the trunk and the
    output head.

Splitting there keeps the entropy coder out of the graph, which is correct:
JPEG-LS is a byte-exact standard codec, not something to approximate in ONNX.

``H`` and ``W`` are dynamic, but both must stay divisible by the largest patch
size -- the analysis convolutions are non-overlapping, so a ragged edge has no
defined meaning. The tool verifies every exported graph against PyTorch on real
images through onnxruntime rather than trusting the exporter, and times the
encoder on CPU because cheap CPU encoding is the property the architecture
exists to provide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_joint_prefix import load_checkpoint  # noqa: E402
from tools.prune_latent_channels import load_images  # noqa: E402


class EncoderGraph(torch.nn.Module):
    """Analysis path only: image in, int8 codes out."""

    def __init__(self, model) -> None:
        super().__init__()
        self.analysis = model.analysis
        self.companders = model.companders

    def forward(self, image: torch.Tensor):
        codes = []
        for analysis, compander in zip(self.analysis, self.companders):
            value = compander.companded(analysis(image))
            codes.append(torch.clamp(torch.round(value), -127.0, 127.0).to(torch.int8))
        return tuple(codes)


class DecoderGraph(torch.nn.Module):
    """Synthesis path only: int8 codes in, reconstruction out.

    The prefix is fixed at export time.  A deployed decoder serves one operating
    point, and freezing the mask lets it fold into the first convolution instead
    of surviving as a runtime multiply.
    """

    def __init__(self, model, prefix: int) -> None:
        super().__init__()
        self.model = model
        self.prefix = prefix

    def forward(self, *codes: torch.Tensor):
        adapted = self.model.adapt([code.to(torch.float32) for code in codes])
        return self.model.decode(adapted, self.prefix)


def export(module: torch.nn.Module, sample: tuple, path: Path, input_names: list[str],
           output_names: list[str], dynamic_axes: dict, opset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, sample, str(path), input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes,
                      opset_version=opset, do_constant_folding=True, dynamo=False)


def run_onnx(path: Path, feeds: dict[str, np.ndarray], threads: int) -> list[np.ndarray]:
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(str(path), options,
                                           providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=None,
                        help="operating point to freeze into the decoder (default: full)")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/workspace/data/frappe_rgb_800x608/imagefolder"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--verify-images", type=int, default=4)
    parser.add_argument("--export-height", type=int, default=608)
    parser.add_argument("--export-width", type=int, default=800)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads for the timing report")
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    model, config, state = load_checkpoint(args.checkpoint, "cpu")
    model.eval()
    prefix = args.prefix or model.n_channels
    max_ps = max(model.ps)
    if args.export_height % max_ps or args.export_width % max_ps:
        raise SystemExit(f"export size must be a multiple of the largest patch size {max_ps}")

    code_names = [f"codes_p{ps}" for ps, _, _ in model.scale_groups]
    sample_image = torch.zeros(1, model.input_channels, args.export_height, args.export_width)
    encoder = EncoderGraph(model).eval()
    with torch.no_grad():
        sample_codes = tuple(code.clone() for code in encoder(sample_image))

    encoder_path = args.output_stem.with_name(args.output_stem.name + "_encoder.onnx")
    decoder_path = args.output_stem.with_name(args.output_stem.name + "_decoder.onnx")

    # Each scale group has its own grid size (H/p, W/p), so they must not share
    # symbolic dimension names: onnxruntime would otherwise treat H/32 and H/2 as
    # the same unknown and refuse to reuse buffers correctly.
    code_axes = {name: {0: "batch", 2: f"grid_h_p{ps}", 3: f"grid_w_p{ps}"}
                 for name, (ps, _, _) in zip(code_names, model.scale_groups)}
    export(encoder, (sample_image,), encoder_path, ["image"], code_names,
           {"image": {0: "batch", 2: "height", 3: "width"}, **code_axes}, args.opset)
    export(DecoderGraph(model, prefix).eval(), sample_codes, decoder_path,
           code_names, ["reconstruction"],
           {**code_axes, "reconstruction": {0: "batch", 2: "height", 3: "width"}},
           args.opset)
    print(f"wrote {encoder_path} ({encoder_path.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {decoder_path} ({decoder_path.stat().st_size / 1e6:.2f} MB)")

    images = load_images(args.dataset_root, args.split, args.verify_images, "cpu")
    worst_codes, worst_recon, mismatched = 0, 0.0, 0
    for x in images:
        with torch.no_grad():
            reference_codes = encoder(x)
            reference_recon = DecoderGraph(model, prefix).eval()(*reference_codes)
        onnx_codes = run_onnx(encoder_path, {"image": x.numpy()}, args.threads)
        for reference, candidate in zip(reference_codes, onnx_codes):
            difference = int(np.abs(reference.numpy().astype(np.int32)
                                    - candidate.astype(np.int32)).max())
            worst_codes = max(worst_codes, difference)
            mismatched += int((reference.numpy() != candidate).sum())
        onnx_recon, = run_onnx(decoder_path,
                               {name: code for name, code in zip(code_names, onnx_codes)},
                               args.threads)
        worst_recon = max(worst_recon,
                          float(np.abs(reference_recon.numpy() - onnx_recon).max()))
    total_symbols = sum(int(code.numel()) for code in reference_codes) * len(images)
    print(f"\n  onnxruntime vs PyTorch over {len(images)} images:")
    print(f"    encoder: max |code difference| = {worst_codes}"
          f"   mismatched symbols = {mismatched}/{total_symbols}")
    print(f"    decoder: max |reconstruction difference| = {worst_recon:.3e}")
    if worst_codes != 0:
        raise SystemExit("ONNX encoder does not reproduce the integer codes exactly")
    if worst_recon > 1e-3:
        raise SystemExit("ONNX decoder output diverges from PyTorch")

    feeds = {"image": images[0].numpy()}
    for _ in range(3):
        run_onnx(encoder_path, feeds, args.threads)
    import onnxruntime
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    encoder_session = onnxruntime.InferenceSession(str(encoder_path), options,
                                                   providers=["CPUExecutionProvider"])
    decoder_session = onnxruntime.InferenceSession(str(decoder_path), options,
                                                   providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    for _ in range(args.timing_repeats):
        codes = encoder_session.run(None, feeds)
    encode_ms = (time.perf_counter() - started) / args.timing_repeats * 1000
    decoder_feeds = {name: code for name, code in zip(code_names, codes)}
    for _ in range(2):
        decoder_session.run(None, decoder_feeds)
    started = time.perf_counter()
    for _ in range(args.timing_repeats):
        decoder_session.run(None, decoder_feeds)
    decode_ms = (time.perf_counter() - started) / args.timing_repeats * 1000
    pixels = args.export_height * args.export_width
    print(f"\n  CPU latency at {args.export_width}x{args.export_height}, "
          f"{args.threads} thread(s):")
    print(f"    encode {encode_ms:8.2f} ms  ({pixels / encode_ms / 1000:.1f} Mpixel/s)")
    print(f"    decode {decode_ms:8.2f} ms  ({pixels / decode_ms / 1000:.1f} Mpixel/s)")

    report = {
        "checkpoint": str(args.checkpoint), "prefix": prefix, "ps": list(model.ps),
        "opset": args.opset,
        "files": {"encoder": str(encoder_path), "decoder": str(decoder_path),
                  "encoder_bytes": encoder_path.stat().st_size,
                  "decoder_bytes": decoder_path.stat().st_size},
        "verification": {"images": len(images), "max_code_difference": worst_codes,
                         "mismatched_symbols": mismatched,
                         "max_reconstruction_difference": worst_recon},
        "cpu_latency_ms": {"encode": encode_ms, "decode": decode_ms,
                           "threads": args.threads,
                           "size": [args.export_width, args.export_height]},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
