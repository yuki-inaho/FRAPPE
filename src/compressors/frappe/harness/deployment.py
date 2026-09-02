"""The deployment-shaped views of the codec, and the shape algebra they need.

``JointPrefixFRAPPE`` is the research object: it holds every prefix at once, it
takes ``[-1, 1]`` floats, and it hands back latents. None of that is what ships.
What ships is two graphs -- image bytes to JPEG-LS-ready planes, planes back to
image bytes -- at one frozen operating point, and this module is where those two
graphs are defined.

They live here rather than in the ONNX exporter because the exporter is no
longer their only consumer. Quantizing the encoder means running the *same*
module through NNCF, and a second copy of the definition would be a second
chance for the deployed arithmetic to disagree with the exported arithmetic --
exactly the class of defect that putting the bitstream arrangement inside the
graph was meant to eliminate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .bitstream import CODE_OFFSET

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..prefix import JointPrefixFRAPPE


def plane_names(model: JointPrefixFRAPPE) -> list[str]:
    """The graph output names, one per scale group, named by patch size."""
    return [f"plane_p{ps}" for ps, _, _ in model.scale_groups]


class EncoderGraph(torch.nn.Module):
    """Image in, JPEG-LS-ready grayscale planes out.

    Everything up to but excluding the entropy coder: the per-scale analysis
    convolutions, the per-channel companding, the rounding and clamping that
    *define* the integer codes, and the reshape to ``(C*h, w)`` that turns each
    scale group into the single grayscale plane the paper describes.
    """

    def __init__(self, model: JointPrefixFRAPPE, uint8_io: bool) -> None:
        super().__init__()
        self.analysis = model.analysis
        self.companders = model.companders
        self.uint8_io = uint8_io
        # The shape algebra rides along so one exporter can serve both this
        # graph and its quantized twin.
        self.ps = model.ps
        self.scale_groups = model.scale_groups
        self.input_channels = model.input_channels

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

    def __init__(self, model: JointPrefixFRAPPE, prefix: int, uint8_io: bool) -> None:
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


def dynamic_shapes(model: JointPrefixFRAPPE, max_units: int = 4096):
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


def measure_deployed_conditions(compiled_conditions: dict[str, object], decoder, folder,
                                indices, *, scale_groups, plan, prefix, height, width,
                                device) -> dict[str, dict]:
    """Rate-distortion of every compiled encoder condition over the same images.

    Every condition is an already-compiled inference model that maps a uint8
    image to the uint8 planes; the decoder is the one frozen fp32 torch graph
    shared by all conditions. The first condition in the mapping is the
    reference: later conditions additionally report how many integer codes
    they flip against it, which is how arithmetic drift between runtimes is
    measured rather than assumed away. Distortion follows the harness
    convention (aggregate MSE on ``[0, 1]``) and rate counts payload-only
    bytes.
    """
    import numpy as np
    import torch.nn.functional as F

    from .bitstream import BitstreamConvention, measure_rate
    from .metrics import Averaging, RateDistortionAccumulator

    results: dict[str, dict] = {}
    reference_planes: list[list] = []
    names = list(compiled_conditions)
    for name in names:
        compiled = compiled_conditions[name]
        accumulator = RateDistortionAccumulator(Averaging.AGGREGATE_MSE)
        mismatched = max_difference = saturated = 0
        for position, index in enumerate(indices):
            image = folder.pixels(index)
            result = compiled({0: image.numpy()})
            planes = [result[i] for i in range(len(scale_groups))]
            if name == names[0]:
                reference_planes.append(planes)
            else:
                for got, want in zip(planes, reference_planes[position]):
                    difference = np.abs(got.astype(np.int32) - want.astype(np.int32))
                    mismatched += int((difference > 0).sum())
                    max_difference = max(max_difference, int(difference.max()))
            latents = []
            for plane, (ps, start, end) in zip(planes, scale_groups):
                h, w = height // ps, width // ps
                saturated += int((plane == 255).sum())
                latents.append(torch.from_numpy(
                    plane.reshape(1, end - start, h, w).astype(np.int64) - CODE_OFFSET
                ).to(torch.int8))
            byte_count, _ = measure_rate(latents, height * width, plan,
                                         BitstreamConvention.PAYLOAD_ONLY)
            with torch.no_grad():
                # DecoderGraph ships image bytes (uint8 0..255); the distortion
                # convention elsewhere in the harness works on [0, 1].
                reconstruction = decoder(*[torch.from_numpy(plane).to(device)
                                           for plane in planes]).float() / 255.0
            image_signed = folder.signed(index, device)
            mse = F.mse_loss(image_signed / 2 + 0.5, reconstruction).item()
            accumulator.add(mse, byte_count, height * width)
        point = accumulator.point(label=prefix)
        results[name] = {
            "psnr_db": point.psnr_db, "bpp": point.bpp,
            "bytes_total": point.bytes_total,
            "compression_ratio": point.compression_ratio,
            "saturated_symbols": saturated,
            "mismatched_symbols_vs_reference": mismatched,
            "max_symbol_difference_vs_reference": max_difference,
        }
    return results


def describe(path: str | Path) -> dict:
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
