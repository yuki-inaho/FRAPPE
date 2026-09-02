"""Run the exported FRAPPE graphs on a named OpenVINO device.

``tools/export_onnx.py`` places the split exactly at the entropy coder: the
encoder graph emits the uint8 grayscale planes JPEG-LS consumes and the decoder
graph consumes the same planes, so a deployment has no arithmetic of its own.
This module is the other half of that -- it loads those two graphs, pins them to
a device and a resolution, and runs them. It deliberately holds no model
knowledge beyond the plane geometry, and it does not import torch, because the
machine that runs a codec should not need the framework that trained it.

Two decisions are worth stating.

**The requested device is the device that runs.** OpenVINO's meta-devices
(``AUTO``, ``HETERO``, ``MULTI``, ``BATCH``) exist to choose a device for you and
to move work when the first choice does not fit. That is precisely the behaviour
this module refuses: asking for ``AUTO`` and reading back "it succeeded" tells
you nothing about where it ran, and a codec whose encoder silently migrated to
another device is a codec whose bitrate measurements mean something different
than you think. A caller that wants a preference order states one to
:func:`select_device` and gets back both the winner and the reason each earlier
choice was skipped. Falling back is then a caller's policy with a paper trail,
not a runtime behaviour.

**Shapes are pinned before compilation.** The exported graphs are genuinely
resolution independent -- height and width are carried as multiples of the
largest patch size -- but the NPU plugin compiles for a fixed shape, so a
deployment has to choose one. :func:`plane_shapes_for` reproduces the same
affine relation the exporter declares, which is the paper's own:
``rows = n_s * (max_ps / p_s) * units_h`` and ``cols = (max_ps / p_s) * units_w``.
It is duplicated here rather than imported from :mod:`compressors.frappe.ops`
only because that module pulls in torch; ``tests`` asserts the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: OpenVINO devices that resolve to some other device at run time. Refused,
#: because the point of naming a device is to know which one ran.
META_DEVICES = frozenset({"AUTO", "MULTI", "HETERO", "BATCH"})

#: The exporter's file naming, so a caller can pass one stem for both graphs.
ENCODER_SUFFIX = "_encoder.onnx"
DECODER_SUFFIX = "_decoder.onnx"


class DeviceUnavailableError(RuntimeError):
    """A named device is not enumerated by this OpenVINO installation.

    Raised instead of quietly choosing another device. The message carries every
    device that was tried and every device that exists, because the usual cause
    is a missing runtime package rather than a typo.
    """


class PrecisionUnavailableError(RuntimeError):
    """A device cannot reproduce the reference bitstream, whatever it is asked.

    Distinct from :class:`DeviceUnavailableError` because the device is present
    and will happily run -- it simply cannot be exact, so a caller that needs
    exactness has to choose a different device rather than a different setting.
    """


def bit_exact_properties(device: str, core=None) -> dict:
    """Properties that make ``device`` reproduce the reference codes exactly.

    The encoder's output planes *are* the bitstream, so a device that rounds one
    companded value differently writes a different file. Whether that matters is
    a caller's decision -- the codes feed a lossless entropy coder, so a code
    that differs by one is a valid code, and measured over real images the whole
    effect is about +0.0004 bpp and -0.0004 dB, the same order as the four-byte
    length prefixes this repository already treats as negligible. What is not a
    decision is which devices *can* be exact:

    ``CPU``  exact at its default precision (FP32 is its native path).
    ``GPU``  exact once ``INFERENCE_PRECISION_HINT`` is ``f32``; at its default
             it computes the analysis path in fp16 and a few hundred symbols per
             image land on the other side of a rounding boundary.
    ``NPU``  cannot. Its ``OPTIMIZATION_CAPABILITIES`` are ``FP16`` and ``INT8``
             with no ``FP32`` entry, so the request has nothing to bind to.

    Raises :class:`PrecisionUnavailableError` for the last case rather than
    returning a hint that will be silently ignored.
    """
    resolved = require_device(device, core)
    capabilities = _core(core).get_property(resolved, "OPTIMIZATION_CAPABILITIES")
    if "FP32" not in capabilities:
        raise PrecisionUnavailableError(
            f"{resolved} advertises {list(capabilities)} and cannot compute the "
            "analysis path in fp32, so it cannot reproduce the reference codes. "
            "Encode on a device that can (CPU, or GPU with INFERENCE_PRECISION_HINT "
            "f32), or accept the measured inexactness explicitly.")
    return {"INFERENCE_PRECISION_HINT": "f32"}


@dataclass(frozen=True)
class DeviceChoice:
    """Which device was selected, and why the earlier preferences were not.

    ``considered`` is ordered and holds only the rejections, so an empty tuple
    means the first preference won.
    """

    device: str
    considered: tuple[tuple[str, str], ...] = ()


def _core(core=None):
    if core is not None:
        return core
    import openvino

    return openvino.Core()


def available_devices(core=None) -> list[str]:
    """Devices this OpenVINO installation enumerates, meta-devices excluded."""
    return [device for device in _core(core).available_devices
            if device.split(".")[0] not in META_DEVICES]


def _reject_meta(device: str) -> None:
    if device.split(".")[0].upper() in META_DEVICES:
        raise ValueError(
            f"{device!r} is an OpenVINO meta-device: it selects a device at run time, "
            "which defeats naming one. Pass a concrete device, or a preference list "
            "to select_device() which records what it chose and why.")


def require_device(device: str, core=None) -> str:
    """Return ``device`` if it can run here, else raise. Never substitutes."""
    _reject_meta(device)
    present = available_devices(core)
    if device in present or device.split(".")[0] in present:
        return device
    raise DeviceUnavailableError(
        f"OpenVINO does not enumerate {device!r} on this machine; it has {present}")


def select_device(preferences, core=None) -> DeviceChoice:
    """First available device in ``preferences``, with the rejections recorded.

    This is where a fallback policy belongs -- in the caller's explicit ordering,
    returning evidence of what happened. The graph classes below never fall back.
    """
    preferences = [str(preference) for preference in preferences]
    if not preferences:
        raise ValueError("select_device needs at least one preference")
    for preference in preferences:
        _reject_meta(preference)
    resolved = _core(core)
    considered: list[tuple[str, str]] = []
    for preference in preferences:
        try:
            return DeviceChoice(require_device(preference, resolved), tuple(considered))
        except DeviceUnavailableError as unavailable:
            considered.append((preference, str(unavailable)))
    raise DeviceUnavailableError(
        "none of the requested devices is available: "
        + "; ".join(f"{device}: {reason}" for device, reason in considered))


def scale_groups_of(patch_sizes) -> list[tuple[int, int]]:
    """``(patch size, channel count)`` per contiguous run, coarse to fine.

    Mirrors :func:`compressors.frappe.ops.get_scale_groups`, which returns index
    bounds instead of counts because it slices tensors; here only the width of
    each group matters.
    """
    groups: list[tuple[int, int]] = []
    for patch_size in patch_sizes:
        if groups and groups[-1][0] == patch_size:
            groups[-1] = (patch_size, groups[-1][1] + 1)
        else:
            groups.append((int(patch_size), 1))
    return groups


def plane_shapes_for(patch_sizes, height: int, width: int) -> list[tuple[int, int]]:
    """``(rows, cols)`` of each scale group's JPEG-LS plane at this resolution.

    The relation is the exporter's, which is the paper's: sizes are counted in
    units of the largest patch size because the non-overlapping analysis admits
    no finer granularity, and each group's plane is
    ``n_s * (max_ps / p_s) * units_h`` rows by ``(max_ps / p_s) * units_w``
    columns.
    """
    groups = scale_groups_of(patch_sizes)
    if not groups:
        raise ValueError("an empty channel schedule has no planes")
    max_ps = max(patch_size for patch_size, _ in groups)
    if height % max_ps or width % max_ps:
        raise ValueError(
            f"{width}x{height} is not a multiple of the largest patch size {max_ps}; "
            "the analysis convolutions do not overlap, so a ragged edge has no "
            "defined meaning")
    units_h, units_w = height // max_ps, width // max_ps
    if units_h < 2 or units_w < 2:
        raise ValueError(
            f"{width}x{height} is {units_w}x{units_h} units of {max_ps}px; at least "
            "2x2 is required because the first decoder convolution pads by "
            "reflection, which is undefined on a single element")
    shapes = []
    for patch_size, channels in groups:
        factor = max_ps // patch_size
        shapes.append((channels * factor * units_h, factor * units_w))
    return shapes


class _PinnedGraph:
    """One ONNX graph, reshaped to static, compiled for one device.

    Compilation is the expensive step and it is done once here, so a caller that
    encodes many images pays it once. ``execution_devices`` is read back from the
    compiled model rather than assumed, because that is the only statement about
    where the work actually landed that does not come from us.

    Ports are addressed by index, never by name. The exporter names each plane
    ``plane_p{patch_size}`` from the scale groups, which is unique for a monotone
    schedule but not in general -- a schedule where a patch size reappears after
    another would emit two identically named outputs, and a name lookup would
    then bind the wrong plane silently. Index order is the graph's own order and
    is unambiguous. Names are still read, for reports and error messages.
    """

    def __init__(self, path: Path, device: str, input_shapes,
                 core=None, properties: dict | None = None) -> None:
        import openvino

        self.path = Path(path)
        self.core = _core(core)
        self.device = require_device(device, self.core)
        model = self.core.read_model(self.path)
        input_shapes = [tuple(shape) for shape in input_shapes]
        if len(input_shapes) != len(model.inputs):
            raise ValueError(f"{self.path.name} has {len(model.inputs)} input(s), "
                             f"{len(input_shapes)} shape(s) were given")
        model.reshape({index: openvino.PartialShape(list(shape))
                       for index, shape in enumerate(input_shapes)})
        still_dynamic = [index for index, port in enumerate(model.outputs)
                         if port.get_partial_shape().is_dynamic]
        if still_dynamic:
            raise ValueError(
                f"{self.path.name}: output(s) at index {still_dynamic} stayed dynamic "
                f"after pinning inputs to {input_shapes}. The exported graph should "
                "carry the affine relation between image and plane extents; without it "
                "the NPU plugin cannot compile.")
        self.compiled = self.core.compile_model(model, self.device, properties or {})
        self.request = self.compiled.create_infer_request()
        self.input_names = [port.get_any_name() for port in self.compiled.inputs]
        self.output_names = [port.get_any_name() for port in self.compiled.outputs]
        self.input_shapes = input_shapes
        self.output_shapes = [tuple(port.get_shape()) for port in self.compiled.outputs]

    @property
    def execution_devices(self) -> list[str]:
        value = self.compiled.get_property("EXECUTION_DEVICES")
        return list(value) if not isinstance(value, str) else [value]

    def __call__(self, arrays) -> list[np.ndarray]:
        arrays = list(arrays)
        if len(arrays) != len(self.input_shapes):
            raise ValueError(f"{self.path.name} needs {len(self.input_shapes)} input(s), "
                             f"{len(arrays)} were given")
        for index, (array, shape) in enumerate(zip(arrays, self.input_shapes)):
            if tuple(array.shape) != shape:
                raise ValueError(f"{self.path.name} input {index} "
                                 f"({self.input_names[index]}) was pinned to {shape}, "
                                 f"got {tuple(array.shape)}")
            self.request.set_input_tensor(index, _tensor(array))
        self.request.infer()
        return [np.array(self.request.get_output_tensor(index).data, copy=True)
                for index in range(len(self.output_names))]


def _tensor(array: np.ndarray):
    import openvino

    return openvino.Tensor(np.ascontiguousarray(array))


def _graph_path(stem_or_path, suffix: str) -> Path:
    path = Path(stem_or_path)
    return path if path.name.endswith(suffix) else path.with_name(path.name + suffix)


class FrappeEncoder:
    """Image in, JPEG-LS-ready uint8 planes out, on one device at one resolution."""

    def __init__(self, stem_or_path, device: str, height: int, width: int,
                 core=None, properties: dict | None = None) -> None:
        self.height, self.width = int(height), int(width)
        self.graph = _PinnedGraph(_graph_path(stem_or_path, ENCODER_SUFFIX), device,
                                  [(1, 3, self.height, self.width)],
                                  core=core, properties=properties)
        self.plane_names = list(self.graph.output_names)
        self.plane_shapes = [shape[1:] for shape in self.graph.output_shapes]

    @property
    def device(self) -> str:
        return self.graph.device

    @property
    def execution_devices(self) -> list[str]:
        return self.graph.execution_devices

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        expected = (1, 3, self.height, self.width)
        if image.shape != expected:
            raise ValueError(f"encoder was pinned to {expected}, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError("the deployed encoder takes uint8; the [-1,1] float form "
                             "is the --io float export, which is a different graph")
        return self.graph([image])


class FrappeDecoder:
    """JPEG-LS planes in, uint8 reconstruction out, at one frozen operating point.

    The prefix is baked into the exported graph, so a decoder object serves one
    rate. ``plane_shapes`` normally comes from the paired encoder; pass it
    explicitly when the decoder is deployed on its own.
    """

    def __init__(self, stem_or_path, device: str, plane_shapes,
                 core=None, properties: dict | None = None) -> None:
        self.plane_shapes = [tuple(shape) for shape in plane_shapes]
        self.graph = _PinnedGraph(
            _graph_path(stem_or_path, DECODER_SUFFIX), device,
            [(1, *shape) for shape in self.plane_shapes],
            core=core, properties=properties)
        self.plane_names = list(self.graph.input_names)

    @property
    def device(self) -> str:
        return self.graph.device

    @property
    def execution_devices(self) -> list[str]:
        return self.graph.execution_devices

    def __call__(self, planes) -> np.ndarray:
        planes = list(planes)
        if len(planes) != len(self.plane_shapes):
            raise ValueError(f"decoder was pinned to {len(self.plane_shapes)} planes, "
                             f"got {len(planes)}")
        # A plane is (rows, cols) as JPEG-LS sees it; the graph carries the batch
        # axis the exporter fixed at one, so accept either form.
        arrays = [np.asarray(plane)[None] if np.asarray(plane).ndim == 2
                  else np.asarray(plane) for plane in planes]
        return self.graph(arrays)[0]
