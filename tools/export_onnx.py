#!/usr/bin/env python3
"""Export a joint-prefix FRAPPE codec to ONNX and check it against onnxruntime.

The codec is exported as two graphs, because that is how it is deployed: the
analysis side has to run on a cheap CPU while the synthesis side may run
anywhere.

The split is placed exactly at the entropy coder, not one step before it. The
paper describes the bitstream as "reshape each scale to a single 2D grayscale
plane ``(n_s * T1/p_s, T2/p_s)`` and apply length-prefixed JPEG-LS", and
``src/compressors/frappe/entropy_coding.py`` implements that as a reshape of the
``(1, C, H, W)`` latent to ``(C*H, W)`` followed by a shift of the signed codes
into ``uint8``. Both of those are pure tensor operations, so both belong inside
the graph: the encoder's outputs are then literally the grayscale images JPEG-LS
consumes, and a deployment has no arithmetic left to get wrong between the model
and the coder. What stays outside is JPEG-LS itself and its 4-byte length
prefix, which are a byte-exact standard codec and a container, not arithmetic.

``<stem>_encoder.onnx``
    ``image (N, 3, H, W) uint8``  ->  one ``uint8`` plane ``(N, n_s*H/p_s, W/p_s)``
    per scale group, ready to hand to JPEG-LS.

``<stem>_decoder.onnx``
    the same planes  ->  ``reconstruction (N, 3, H, W) uint8``.

``--io float`` keeps the research-facing form instead: ``[-1, 1]`` float images
in and out, and signed ``int8`` code planes. The layout is identical either way;
only the dtypes and the normalisation differ.

``H`` and ``W`` are genuinely dynamic, not merely declared so. Both must stay
divisible by the largest patch size -- the analysis convolutions are
non-overlapping, so a ragged edge has no defined meaning -- and each scale
group's plane has ``rows = n_s * H / p_s`` and ``cols = W / p_s``, which is an
affine relation the exporter is told about explicitly. Without it the tracer
resolves the reshape from the traced tensor and bakes one image size into the
graph. The batch axis is fixed at one because the model specialises it; an image
codec encodes one image at a time.

Every exported graph is checked against the reference implementation on real
images: the encoder's planes must be byte-identical to
``entropy_coding.arrange_latents`` plus the shift, and the JPEG-LS payloads they
produce must be byte-identical to the reference bitstream. The graphs are then
re-run at other resolutions, because a graph that only works at the size it was
traced at is the failure this export is written to avoid.
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

from src.compressors.frappe.harness import AnonymousImageFolder  # noqa: E402
from src.compressors.frappe.harness.bitstream import (  # noqa: E402
    CODE_OFFSET,
    BitstreamConvention,
    arrange_planes,
    encode_plane,
    encode_planes,
)
from src.compressors.frappe.harness.checkpoints import load_checkpoint  # noqa: E402
from src.compressors.frappe.harness.data import default_dataset_root  # noqa: E402



class EncoderGraph(torch.nn.Module):
    """Image in, JPEG-LS-ready grayscale planes out."""

    def __init__(self, model, uint8_io: bool) -> None:
        super().__init__()
        self.analysis = model.analysis
        self.companders = model.companders
        self.uint8_io = uint8_io

    def forward(self, image: torch.Tensor):
        x = image.to(torch.float32) / 127.5 - 1.0 if self.uint8_io else image
        planes = []
        for analysis, compander in zip(self.analysis, self.companders):
            codes = torch.clamp(torch.round(compander.companded(analysis(x))),
                                -CODE_OFFSET, CODE_OFFSET)
            # (N, C, h, w) -> (N, C*h, w). flatten of two adjacent axes stays
            # shape-agnostic under export, where an explicit reshape target would
            # bake in the traced spatial size.
            plane = torch.flatten(codes, 1, 2)
            planes.append((plane + CODE_OFFSET).to(torch.uint8) if self.uint8_io
                          else plane.to(torch.int8))
        return tuple(planes)


class DecoderGraph(torch.nn.Module):
    """Grayscale planes in, reconstruction out, at one fixed operating point.

    A deployed decoder serves one prefix, so freezing it lets the channel mask
    fold into the first convolution instead of surviving as a runtime multiply.
    """

    def __init__(self, model, prefix: int, uint8_io: bool) -> None:
        super().__init__()
        self.model = model
        self.prefix = prefix
        self.uint8_io = uint8_io
        self.widths = [end - start for _, start, end in model.scale_groups]

    def forward(self, *planes: torch.Tensor):
        latents = []
        for plane, width in zip(planes, self.widths):
            value = plane.to(torch.float32)
            if self.uint8_io:
                value = value - CODE_OFFSET
            latents.append(value.unflatten(1, (width, -1)))  # (N, C*h, w) -> (N, C, h, w)
        reconstruction = self.model.decode(self.model.adapt(latents), self.prefix)
        if self.uint8_io:
            return torch.clamp(
                torch.round((reconstruction.clamp(-1.0, 1.0) / 2 + 0.5) * 255.0),
                0.0, 255.0).to(torch.uint8)
        return reconstruction


def dynamic_shapes(model, max_units: int = 4096):
    """Affine shape relations the exporter needs to keep ``H`` and ``W`` symbolic.

    Sizes are expressed in units of the largest patch size, which is the only
    granularity the non-overlapping analysis admits. Each scale group's plane is
    then ``rows = n_s * (max_ps / p_s) * units_h`` by
    ``cols = (max_ps / p_s) * units_w``. The lower bound of two is not cosmetic:
    the first decoder convolution pads by reflection, which is undefined once the
    grid is a single element, and the exporter refuses to emit a graph whose
    guards it cannot satisfy.
    """
    from torch.export import Dim

    max_ps = max(model.ps)
    units_h = Dim("units_h", min=2, max=max_units)
    units_w = Dim("units_w", min=2, max=max_units)
    image = {2: max_ps * units_h, 3: max_ps * units_w}
    planes = tuple({1: (end - start) * (max_ps // ps) * units_h,
                    2: (max_ps // ps) * units_w}
                   for ps, start, end in model.scale_groups)
    return image, planes


def export(module: torch.nn.Module, sample: tuple, path: Path, input_names: list[str],
           output_names: list[str], shapes, opset: int, simplify: bool) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, sample, str(path), input_names=input_names,
                      output_names=output_names, dynamic_shapes=shapes,
                      opset_version=opset, dynamo=True)
    raw_bytes = path.stat().st_size
    info = {"bytes": raw_bytes, "simplified": False}
    if simplify:
        import onnx
        from onnxsim import simplify as onnx_simplify

        model = onnx.load(str(path))
        before = len(model.graph.node)
        simplified, ok = onnx_simplify(model)
        if ok:
            onnx.save(simplified, str(path))
            info.update({"simplified": True, "nodes_before": before,
                         "nodes_after": len(simplified.graph.node),
                         "bytes": path.stat().st_size, "bytes_before": raw_bytes})
        else:
            info["simplify_error"] = "onnxsim could not validate the simplified model"
    return info


def session(path: Path, threads: int):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(str(path), options,
                                        providers=["CPUExecutionProvider"])


def describe(path: Path) -> dict:
    """The graph's declared input and output signature, as deployed code sees it."""
    import onnx

    model = onnx.load(str(path))

    def signature(values):
        described = []
        for value in values:
            tensor = value.type.tensor_type
            dims = [d.dim_param or d.dim_value for d in tensor.shape.dim]
            described.append({"name": value.name,
                              "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                              "shape": dims})
        return described

    initialisers = {i.name for i in model.graph.initializer}
    return {"inputs": signature([v for v in model.graph.input
                                 if v.name not in initialisers]),
            "outputs": signature(model.graph.output),
            "opset": model.opset_import[0].version}


def verify_against_reference(model, decoder, folder, encoder_session, decoder_session,
                             plane_names, count, uint8_io) -> dict:
    """The exported encoder must reproduce the reference bitstream exactly.

    Not "closely": the planes are the bytes JPEG-LS receives, so anything short
    of byte identity is a different codec. The decoder is held to a weaker bar
    on purpose -- it reduces channels in a different order than PyTorch does, so
    a float32 value on a rounding boundary lands on either side of it.
    """
    worst_plane = worst_recon = 0
    mismatched = symbols = payload_mismatch = 0
    differing_pixels = total_pixels = 0
    reference_blob = 0
    for index in range(min(count, len(folder))):
        x = folder.signed(index)
        codes = model.integer_codes(x)
        reference_planes = [plane.numpy() for plane in arrange_planes(codes)]
        if index == 0:
            reference_blob = len(encode_planes(
                arrange_planes(codes), BitstreamConvention.WITH_LENGTH_PREFIX))
        feed = folder.pixels(index).numpy() if uint8_io else x.numpy()
        onnx_planes = encoder_session.run(None, {"image": feed})
        for reference, candidate in zip(reference_planes, onnx_planes):
            emitted = candidate[0] if uint8_io else (
                candidate[0].astype(np.int32) + int(CODE_OFFSET)).astype(np.uint8)
            worst_plane = max(worst_plane, int(np.abs(
                emitted.astype(np.int32) - reference.astype(np.int32)).max()))
            mismatched += int((emitted != reference).sum())
            symbols += reference.size
            if encode_plane(torch.from_numpy(emitted)) != encode_plane(
                    torch.from_numpy(reference)):
                payload_mismatch += 1
        onnx_recon, = decoder_session.run(
            None, dict(zip(plane_names, onnx_planes)))
        with torch.no_grad():
            expected = decoder(*[torch.from_numpy(plane) for plane in onnx_planes])
        difference = np.abs(expected.numpy().astype(np.float64)
                            - onnx_recon.astype(np.float64))
        worst_recon = max(worst_recon, float(difference.max()))
        differing_pixels += int((difference > 0).sum())
        total_pixels += int(difference.size)

    print(f"\n  verified against src/compressors/frappe/entropy_coding.py "
          f"on {min(count, len(folder))} images:")
    print(f"    encoder planes : max |difference| = {worst_plane}   "
          f"mismatched symbols = {mismatched}/{symbols}")
    payload_note = ("byte-identical" if payload_mismatch == 0
                    else f"{payload_mismatch} streams differ")
    print(f"    JPEG-LS payload: {payload_note}   (reference blob for image 0: "
          f"{reference_blob} B, length prefixes included)")
    print(f"    decoder output : max |difference| vs PyTorch = {worst_recon:g}"
          f"   differing = {differing_pixels}/{total_pixels}"
          f" ({100 * differing_pixels / max(total_pixels, 1):.4f}%)")
    if worst_plane or payload_mismatch:
        raise SystemExit("ONNX encoder does not reproduce the reference bitstream")
    if uint8_io and (worst_recon > 1 or differing_pixels > 1e-4 * total_pixels):
        raise SystemExit("ONNX decoder output diverges from PyTorch")
    if not uint8_io and worst_recon > 1e-3:
        raise SystemExit("ONNX decoder output diverges from PyTorch")
    return {"images": min(count, len(folder)), "max_plane_difference": worst_plane,
            "mismatched_symbols": mismatched, "symbols": symbols,
            "jpegls_streams_differing": payload_mismatch,
            "max_reconstruction_difference": worst_recon,
            "differing_pixels": differing_pixels, "total_pixels": total_pixels,
            "reference_blob_bytes_image0": reference_blob}


def verify_resolutions(model, encoder_session, decoder_session, plane_names, uint8_io,
                       height: int, width: int) -> list[list[int]]:
    """A graph that only runs at the traced size is the defect this export avoids."""
    max_ps = max(model.ps)
    sizes = [(height, width), (max_ps * 2, max_ps * 3), (max_ps * 10, max_ps * 15),
             (1088, 1920)]
    print("\n  same graphs re-run at other resolutions:")
    for probe_height, probe_width in sizes:
        probe = np.zeros((1, model.input_channels, probe_height, probe_width),
                         dtype=np.uint8 if uint8_io else np.float32)
        try:
            planes = encoder_session.run(None, {"image": probe})
            recon = decoder_session.run(None, dict(zip(plane_names, planes)))[0]
            expected = (1, model.input_channels, probe_height, probe_width)
            status = "ok" if tuple(recon.shape) == expected else f"shape {recon.shape}"
        except Exception as error:
            status = f"{type(error).__name__}: {error}"
        print(f"    {probe_width:5d}x{probe_height:<5d} {status}")
        if status != "ok":
            raise SystemExit("the exported graphs are not resolution independent")
    return [[w, h] for h, w in sizes]


def measure_latency(encoder_session, decoder_session, plane_names, image,
                    repeats: int) -> tuple[float, float]:
    """Encode and decode wall-clock, warmed up, at the caller's thread count."""
    feed = {"image": image.numpy()}
    for _ in range(3):
        planes = encoder_session.run(None, feed)
    started = time.perf_counter()
    for _ in range(repeats):
        planes = encoder_session.run(None, feed)
    encode_ms = (time.perf_counter() - started) / repeats * 1000
    decoder_feed = dict(zip(plane_names, planes))
    for _ in range(2):
        decoder_session.run(None, decoder_feed)
    started = time.perf_counter()
    for _ in range(repeats):
        decoder_session.run(None, decoder_feed)
    return encode_ms, (time.perf_counter() - started) / repeats * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=None,
                        help="operating point to freeze into the decoder (default: full)")
    parser.add_argument("--io", choices=["uint8", "float"], default="uint8",
                        help="uint8: images and code planes as bytes, normalisation "
                             "inside the graph. float: [-1,1] images and int8 planes")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                        help="anonymous ImageFolder root; defaults to $FRAPPE_DATASET_ROOT")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--verify-images", type=int, default=4)
    parser.add_argument("--export-height", type=int, default=608)
    parser.add_argument("--export-width", type=int, default=800)
    parser.add_argument("--opset", type=int, default=18,
                        help="the dynamo exporter emits opset 18 natively; asking for "
                             "an older one makes onnx run a version converter that has "
                             "no downgrade adapter for Pad")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads for the timing report")
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, "cpu").model
    model.eval()
    prefix = args.prefix or model.n_channels
    uint8_io = args.io == "uint8"
    max_ps = max(model.ps)
    if args.export_height % max_ps or args.export_width % max_ps:
        raise SystemExit(f"export size must be a multiple of the largest patch size {max_ps}")

    plane_names = [f"plane_p{ps}" for ps, _, _ in model.scale_groups]
    image_shape, plane_shapes = dynamic_shapes(model)
    encoder = EncoderGraph(model, uint8_io).eval()
    decoder = DecoderGraph(model, prefix, uint8_io).eval()
    sample_image = (torch.zeros(1, model.input_channels, args.export_height,
                                args.export_width, dtype=torch.uint8) if uint8_io
                    else torch.zeros(1, model.input_channels, args.export_height,
                                     args.export_width))
    with torch.no_grad():
        sample_planes = tuple(plane.clone() for plane in encoder(sample_image))

    encoder_path = args.output_stem.with_name(args.output_stem.name + "_encoder.onnx")
    decoder_path = args.output_stem.with_name(args.output_stem.name + "_decoder.onnx")
    encoder_info = export(encoder, (sample_image,), encoder_path, ["image"], plane_names,
                          (image_shape,), args.opset, args.simplify)
    decoder_info = export(decoder, sample_planes, decoder_path, plane_names,
                          ["reconstruction"], (plane_shapes,), args.opset, args.simplify)

    for label, path, info in (("encoder", encoder_path, encoder_info),
                              ("decoder", decoder_path, decoder_info)):
        note = (f"  simplified {info['nodes_before']} -> {info['nodes_after']} nodes, "
                f"{info['bytes_before'] / 1e6:.2f} -> {info['bytes'] / 1e6:.2f} MB"
                if info.get("simplified") else "  not simplified")
        print(f"wrote {path} ({info['bytes'] / 1e6:.2f} MB)\n{note}")
        signature = describe(path)
        for direction in ("inputs", "outputs"):
            for entry in signature[direction]:
                print(f"    {direction[:-1]:6s} {entry['name']:14s} "
                      f"{entry['dtype']:7s} {entry['shape']}")

    encoder_session = session(encoder_path, args.threads)
    decoder_session = session(decoder_path, args.threads)
    folder = AnonymousImageFolder(args.dataset_root, args.split)
    verification = verify_against_reference(
        model, decoder, folder, encoder_session, decoder_session,
        plane_names, args.verify_images, uint8_io)
    resolutions = verify_resolutions(model, encoder_session, decoder_session, plane_names,
                                     uint8_io, args.export_height, args.export_width)
    encode_ms, decode_ms = measure_latency(
        encoder_session, decoder_session, plane_names,
        folder.pixels(0) if uint8_io else folder.signed(0), args.timing_repeats)
    pixels = args.export_height * args.export_width
    print(f"\n  CPU latency at {args.export_width}x{args.export_height}, "
          f"{args.threads} thread(s):")
    print(f"    encode {encode_ms:8.2f} ms  ({pixels / encode_ms / 1000:.1f} Mpixel/s)")
    print(f"    decode {decode_ms:8.2f} ms  ({pixels / decode_ms / 1000:.1f} Mpixel/s)")

    report = {
        "checkpoint": str(args.checkpoint), "prefix": prefix, "ps": list(model.ps),
        "io": args.io, "opset": args.opset,
        "encoder": {"path": str(encoder_path), **encoder_info, **describe(encoder_path)},
        "decoder": {"path": str(decoder_path), **decoder_info, **describe(decoder_path)},
        "verification": verification,
        "resolutions_verified": resolutions,
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
