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
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def sha256_of(path: str | Path) -> str:
    """The digest recorded next to every derived artifact, so comparisons can't drift."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
