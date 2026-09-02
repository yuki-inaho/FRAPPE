"""The encoder-side quantization-aware-training boundary.

FRAPPE's bitstream is defined by the compander's integer codes; NNCF's fake
quantizers instead emulate INT8 arithmetic *inside* the convolutions. These are
two different mechanisms that must never be confused, so this module keeps the
seam small and explicit: a trainable encoder view whose forward runs the
existing straight-through rounding, a deployable view of that same encoder
producing the shipped uint8 planes, and the NNCF calibration / checkpoint
plumbing that turns the first into the second.

NNCF is a deployment-only dependency (the uv ``deploy`` group), so every touch
of it happens inside a function body: the research environment must be able to
import this module without it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .bitstream import CODE_OFFSET

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..prefix import JointPrefixFRAPPE

#: The attributes of ``JointPrefixFRAPPE`` that belong to the frozen side. A
#: parameter whose qualified name starts with one of these is decoder property.
_DECODER_PREFIXES = ("first", "trunk", "head", "prefix_scale", "prefix_bias")


class TrainableEncoder(torch.nn.Module):
    """The analysis convolutions and companders, alone, as one module.

    Quantizing the encoder -- and only the encoder -- is then a property of the
    module boundary: NNCF is handed this module, so no quantizer can appear in
    the decoder without someone going out of their way. The forward is the
    training path: per-scale float codes whose rounding carries a straight-
    through gradient, exactly what ``JointPrefixFRAPPE.forward_operating_points``
    runs at ``mode="hard"``.
    """

    def __init__(self, model: JointPrefixFRAPPE) -> None:
        super().__init__()
        self.analysis = model.analysis
        self.companders = model.companders
        # The shape algebra travels with the encoder, so a quantized wrapper
        # still answers for its scale groups when the exporter asks.
        self.ps = model.ps
        self.scale_groups = model.scale_groups
        self.input_channels = model.input_channels

    def forward(self, image: torch.Tensor, mode: str = "hard", alpha: float = 8.0):
        return [compander(analysis(image), mode, alpha)
                for analysis, compander in zip(self.analysis, self.companders)]


class DeployableEncoder(torch.nn.Module):
    """Wraps a (possibly fake-quantized) trainable encoder for deployment.

    The wrapping, not a copy, is the point: the quantizers the trainer updated
    are the quantizers the exporter sees. The forward calls the encoder's own
    ``__call__`` -- not its submodules -- because NNCF's hooks fire only there;
    stepping around the parent would silently export the FP32 arithmetic. The
    codes it returns are already the integer bitstream values, so the view only
    flattens and shifts them into the planes JPEG-LS consumes.
    """

    def __init__(self, encoder: TrainableEncoder, uint8_io: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.uint8_io = uint8_io
        self.ps = encoder.ps
        self.scale_groups = encoder.scale_groups
        self.input_channels = encoder.input_channels

    def forward(self, image: torch.Tensor):
        x = image.to(torch.float32) / 127.5 - 1.0 if self.uint8_io else image
        planes = []
        for code in self.encoder(x):
            plane = torch.flatten(code, 1, 2)
            planes.append((plane + CODE_OFFSET).to(torch.uint8) if self.uint8_io
                          else plane.to(torch.int8))
        return tuple(planes)


def freeze_decoder(model: JointPrefixFRAPPE) -> None:
    """Detach the decoder side from training, in place, by requires_grad alone.

    Forwarding through a ``no_grad`` decoder would also cut the encoder's
    gradient; freezing the parameters keeps the graph alive end to end while
    making every decoder update impossible.
    """
    for name, param in model.named_parameters():
        if not name.startswith(("analysis", "companders")):
            param.requires_grad_(False)


def quantize_encoder(encoder: TrainableEncoder, calibration: list[torch.Tensor],
                     target_device: str = "NPU", subset_size: int = 32):
    """Calibrate fake quantizers onto the encoder and return the wrapped model.

    ``target_device`` names where the folded result is meant to run -- NPU is
    the deployment target's placement policy, not a claim about this machine.
    """
    import nncf

    return nncf.quantize(encoder, nncf.Dataset(calibration),
                         subset_size=subset_size,
                         target_device=nncf.TargetDevice(target_device.upper()))


def save_qat_state(quantized: torch.nn.Module) -> dict[str, Any]:
    """The serializable pair that recovers a quantized model: weights + config."""
    from nncf.torch import get_config

    return {"state_dict": quantized.state_dict(), "config": get_config(quantized)}


def restore_qat_state(encoder: TrainableEncoder, state: dict[str, Any]):
    """Rebuild the quantized module on a fresh encoder and reload its weights."""
    from nncf.torch import load_from_config

    restored = load_from_config(encoder, state["config"])
    restored.load_state_dict(state["state_dict"])
    return restored


def save_qat_checkpoint(path: str | Path, quantized: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, iteration: int,
                        base_checkpoint_sha256: str, **extra: Any) -> dict[str, Any]:
    """Write the full QAT resume state atomically: NNCF pair, optimizer, bookkeeping."""
    import nncf
    from nncf.torch import get_config

    payload = {"state_dict": quantized.state_dict(), "config": get_config(quantized),
               "optimizer": optimizer.state_dict(), "iteration": iteration,
               "base_checkpoint_sha256": base_checkpoint_sha256,
               "nncf_version": nncf.__version__, "torch_version": torch.__version__,
               **extra}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return payload


def load_qat_checkpoint(path: str | Path, encoder: TrainableEncoder) -> dict[str, Any]:
    """Restore the quantized encoder from a QAT checkpoint.

    The optimizer must be built over the *restored* model's parameters before
    its saved state is loaded into it, so the raw optimizer state travels back
    with the payload instead of being attached here.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = restore_qat_state(encoder, payload)
    return {"model": restored, "iteration": payload["iteration"],
            "optimizer_state": payload["optimizer"],
            "base_checkpoint_sha256": payload["base_checkpoint_sha256"]}


def export_encoder_onnx(quantized: torch.nn.Module, output_path: str | Path,
                       sample_image: torch.Tensor, opset: int = 17) -> dict[str, Any]:
    """Export a (possibly fake-quantized) encoder as the deployed ONNX graph.

    A fake-quantized input yields ``FakeQuantize`` operators in the graph; a
    plain encoder yields the clean FP32 arithmetic. Either way the same trace

    The TorchScript tracer is deliberate: it executes NNCF's hook machinery for
    real while tracing, so the fake-quant arithmetic lands in the graph as
    ``FakeQuantize`` operators -- the form OpenVINO folds into true INT8.
    ``torch.export`` cannot trace the hook executor's function mode at all, and
    NNCF 3.3 offers no way through: ``StripFormat.DQ`` rejects these quantizers
    (``half_range``), and a ``NATIVE`` strip leaves the mode active. The
    quantized model itself stays untouched; H and W stay symbolic through
    ``dynamic_axes``.
    """
    from .deployment import plane_names

    deploy = DeployableEncoder(quantized, uint8_io=True).eval()
    axes = {"image": {2: "height", 3: "width"},
            **{f"plane_p{ps}": {1: f"rows_p{ps}", 2: f"cols_p{ps}"}
               for ps, _, _ in quantized.scale_groups}}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(deploy, (sample_image,), str(output_path), input_names=["image"],
                      output_names=plane_names(quantized), dynamic_axes=axes,
                      opset_version=opset, dynamo=False)
    return {"path": str(output_path), "opset": opset}


def encoder_weights_from_qat(qat_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """The encoder's own weights from a QAT checkpoint, without hook state.

    NNCF's hook parameters (``__nncf_hooks.*``) carry the *training-time*
    quantization ranges, which the traceable projection does not reproduce;
    deployment therefore quantizes the trained weights afresh, and only these
    plain analysis/compander tensors travel over.
    """
    return {key: value for key, value in qat_state["state_dict"].items()
            if not key.startswith("__nncf_hooks")}


_COMPANDER_SCOPE_PATTERN = r".*/companders\..*"


def require_disjoint_calibration_samples(
    calibration_split: str,
    calibration_indices: Sequence[int],
    evaluation_split: str,
    evaluation_indices: Sequence[int],
) -> None:
    """Reject calibration/evaluation leakage before reporting codec quality."""
    if calibration_split != evaluation_split:
        return
    overlap = sorted(set(calibration_indices) & set(evaluation_indices))
    if overlap:
        preview = overlap[:8]
        raise ValueError(
            f"calibration and evaluation overlap in {calibration_split}: {preview}")


def _remove_compander_output_qdq(model: Any) -> int:
    """Remove Q/DQ pairs NNCF propagates across an ignored compander.

    Ignoring every compander node prevents quantizers inside the nonlinear
    mapping, but NNCF still places one Q/DQ pair at each ignored subgraph's
    output. Those outputs are already FRAPPE's rounded integer codes. A second
    learned quantization grid changes the bitstream, so reconnect consumers to
    the rounded value and discard only those boundary pairs.
    """
    quantizers = [
        node for node in model.graph.node
        if node.op_type == "QuantizeLinear" and "/companders." in node.name
    ]
    removed_nodes = []
    for quantizer in quantizers:
        dequantizers = [
            node for node in model.graph.node
            if node.op_type == "DequantizeLinear" and node.input[0] == quantizer.output[0]
        ]
        if len(dequantizers) != 1:
            raise ValueError(
                f"expected one DequantizeLinear after {quantizer.name}, got {len(dequantizers)}")
        dequantizer = dequantizers[0]
        for consumer in model.graph.node:
            for index, value in enumerate(consumer.input):
                if value == dequantizer.output[0]:
                    consumer.input[index] = quantizer.input[0]
        for output in model.graph.output:
            if output.name == dequantizer.output[0]:
                output.name = quantizer.input[0]
        removed_nodes.extend((quantizer, dequantizer))

    for node in removed_nodes:
        model.graph.node.remove(node)

    used_inputs = {value for node in model.graph.node for value in node.input}
    used_inputs.update(output.name for output in model.graph.output)
    for initializer in list(model.graph.initializer):
        if initializer.name not in used_inputs:
            model.graph.initializer.remove(initializer)
    return len(quantizers)


def quantize_onnx_encoder(
    input_path: str | Path,
    output_path: str | Path,
    calibration_images: Sequence[Mapping[str, np.ndarray]],
    *,
    target_device: str = "NPU",
    subset_size: int | None = None,
    preset: str = "performance",
    bias_correction: str = "fast",
) -> dict[str, Any]:
    """PTQ an FP32 encoder ONNX and save the portable Q/DQ model.

    ONNX is the durable quantization boundary: OpenVINO consumes this exact
    artifact rather than repeating calibration inside its model backend.
    Analysis convolutions are quantized while the companders, codec rounding,
    clamp and uint8 plane layout remain ordinary floating-point/integer ops.
    """
    import nncf
    import onnx

    if not calibration_images:
        raise ValueError("calibration_images must not be empty")
    actual_subset = len(calibration_images) if subset_size is None else subset_size
    if actual_subset <= 0 or actual_subset > len(calibration_images):
        raise ValueError(
            f"subset_size must be in [1, {len(calibration_images)}], got {actual_subset}")
    if bias_correction not in {"fast", "accurate", "none"}:
        raise ValueError(f"unsupported bias correction mode: {bias_correction}")

    model = onnx.load(str(input_path))
    advanced = nncf.AdvancedQuantizationParameters(
        quantize_outputs=False,
        disable_bias_correction=bias_correction == "none",
    )
    quantized = nncf.quantize(
        model,
        nncf.Dataset(list(calibration_images)),
        subset_size=actual_subset,
        target_device=nncf.TargetDevice(target_device.upper()),
        preset=nncf.QuantizationPreset(preset.lower()),
        fast_bias_correction=bias_correction != "accurate",
        ignored_scope=nncf.IgnoredScope(patterns=[_COMPANDER_SCOPE_PATTERN]),
        advanced_parameters=advanced,
    )
    removed = _remove_compander_output_qdq(quantized)
    onnx.checker.check_model(quantized)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(quantized, str(output_path))
    op_types = [node.op_type for node in quantized.graph.node]
    return {
        "path": str(output_path),
        "nncf_version": nncf.__version__,
        "target_device": target_device.upper(),
        "preset": preset.lower(),
        "bias_correction": bias_correction,
        "calibration_images": actual_subset,
        "ignored_scope_patterns": [_COMPANDER_SCOPE_PATTERN],
        "removed_compander_output_qdq": removed,
        "quantize_linear": op_types.count("QuantizeLinear"),
        "dequantize_linear": op_types.count("DequantizeLinear"),
    }


def sha256_of(path: str | Path) -> str:
    """The digest recorded next to every derived artifact, so comparisons can't drift."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
