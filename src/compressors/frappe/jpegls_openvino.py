"""JPEG-LS's regular front half as an OpenVINO graph, for iGPU or NPU.

JPEG-LS is not a transform codec. It predicts each sample from its already-coded
neighbours and codes the residual with an adaptive Golomb code, so most of it is
a sequential state machine and none of that belongs on an accelerator. What does
belong is the front half, which is the same closed-form expression at every
sample:

    neighbours A/B/C/D  ->  MED prediction  ->  three gradients  ->  quantise
                                                                 ->  context Qs

That is what this module computes. Everything after it -- the per-context A/B/C/N
state, run mode, the limited Golomb code, the seven-bit stuffing after ``0xFF``,
the markers -- stays on the CPU, where the data dependency between consecutive
samples actually lives. The split, and the branch-free formulations below, follow
``npu-jpegls-offload`` (MIT), which prototyped the same idea against Intel's now
archived NPU acceleration library; this is the same arithmetic against OpenVINO.

Two things are worth stating, because both are load-bearing.

**MED is written with Minimum/Maximum, not with ReLU.** The reference's concept
note gives the branch-free predictor as a chain of ReLU differences, which is
algebraically identical to ``clamp(A + B - C, min(A,B), max(A,B))``. On this
machine's NPU the ReLU form is *numerically wrong* on some plane shapes -- it was
measured producing 286 of 475 samples incorrectly, with a maximum error of 212,
on a 19x25 plane -- while the Minimum/Maximum form is exact on CPU, iGPU and NPU
at every shape tried. Two formulations of one identity are not interchangeable
once a compiler is between them.

**Every output is verified against the CPU, not spot-checked.** ``verify="all"``
is the default rather than an option, because the failure above is silent: the
predictions stay finite and integral, so a check that only tests those passes on
corrupted output. The only check that catches it is equality with a reference
computed on a device known to be right.

Whether to *use* this is a separate question from whether it works, and the
measurement is not encouraging: on this machine the front half alone costs about
2.3 ms on CPU, 4.2 ms on iGPU and 40.7 ms on NPU for FRAPPE's five code planes,
while CharLS does the entire JPEG-LS codec on the same data in about 6.2 ms.
``tools/benchmark_jpegls_openvino.py`` reproduces that comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Default gradient thresholds T1, T2, T3 for 8-bit lossless JPEG-LS. The
#: quantiser compares against ``T - 1`` because it tests strict inequality.
GRADIENT_THRESHOLDS = (0, 2, 6, 20)

#: ``Qs = RADIX**2 * Q1 + RADIX * Q2 + Q3`` with each ``Q`` in ``[-4, 4]``.
CONTEXT_RADIX = 9


def build_line_buffer(plane: np.ndarray) -> np.ndarray:
    """The bordered buffer JPEG-LS predicts from, for a whole uint8 plane.

    Row 0 is the all-zero line that precedes the image. Every later row is one
    image row surrounded by the standard's boundary samples: the right border
    repeats the last column, and the left border of row ``i`` is the first sample
    of row ``i - 1``. Shape ``(H + 1, W + 2)``, which is what the graph slices.
    """
    image = np.asarray(plane)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"expected a 2D uint8 plane, got {image.ndim}D {image.dtype}")
    height, width = image.shape
    lines = np.zeros((height + 1, width + 2), dtype=np.int16)
    lines[1:, 1 : width + 1] = image
    lines[1:, width + 1] = image[:, width - 1]
    if height > 1:
        lines[2:, 0] = image[:-1, 0]
    return lines


def _quantise(gradient: np.ndarray) -> np.ndarray:
    magnitude = np.abs(gradient.astype(np.int32))
    steps = sum((magnitude > threshold).astype(np.int32) for threshold in GRADIENT_THRESHOLDS)
    return np.sign(gradient.astype(np.int32)) * steps


def reference_med_and_context(line_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The CPU oracle: exact integer MED predictions and signed context IDs.

    Neighbour names follow ISO/IEC 14495-1::

        C B D
        A X
    """
    patch = np.asarray(line_buffer, dtype=np.int32)
    if patch.ndim != 2 or patch.shape[0] < 2 or patch.shape[1] < 3:
        raise ValueError("line buffer must have shape (rows + 1, columns + 2)")
    a = patch[1:, :-2]
    b = patch[:-1, 1:-1]
    c = patch[:-1, :-2]
    d = patch[:-1, 2:]
    prediction = np.minimum(np.maximum(a + b - c, np.minimum(a, b)), np.maximum(a, b))
    context = (
        CONTEXT_RADIX**2 * _quantise(d - b) + CONTEXT_RADIX * _quantise(b - c) + _quantise(c - a)
    )
    return prediction.astype(np.int32), context.astype(np.int32)


def _build_model(rows: int, cols: int):
    """The graph: one bordered plane in, prediction and context out, both i32.

    Neighbours come from four slices rather than a four-channel unit-weight
    convolution. The convolution is what the reference builds and it is exact
    here too, but it buys nothing over a slice and costs a layout the compiler
    then has to undo.

    The conversion to ``i32`` happens inside the graph. Returning floats and
    rounding on the host is what the reference does, and it puts the one step
    that can be wrong -- the float-to-integer decision -- outside the thing being
    verified.
    """
    import openvino as ov
    from openvino import opset14 as ops

    source = ops.parameter([1, 1, rows + 1, cols + 2], ov.Type.f32, name="line_buffer")

    def crop(top: int, left: int):
        return ops.slice(
            source,
            start=ops.constant([top, left], ov.Type.i32),
            stop=ops.constant([top + rows, left + cols], ov.Type.i32),
            step=ops.constant([1, 1], ov.Type.i32),
            axes=ops.constant([2, 3], ov.Type.i32),
        )

    a, b, c, d = crop(1, 0), crop(0, 1), crop(0, 0), crop(0, 2)

    # MED as a clamp, not as a chain of ReLU differences. See the module
    # docstring: the two are algebraically identical and are not numerically
    # identical once this NPU's compiler is between them.
    gradient_predictor = ops.subtract(ops.add(a, b), c)
    prediction = ops.minimum(ops.maximum(gradient_predictor, ops.minimum(a, b)), ops.maximum(a, b))

    def quantise(gradient):
        """``sign(g) * #{t : |g| > t}``, written without a single comparison.

        ``|g| > t`` is the obvious spelling and it is wrong on this NPU: both
        ``Convert(Greater(...))`` and ``Select(Greater(...), 1, 0)`` were measured
        producing incorrect results, 9345 and 1366 samples out of 9500 on a
        190x50 plane. Every arithmetic spelling of the same indicator is exact.
        ``Sign(ReLU(|g| - t))`` is used because the values are integers, where
        ``|g| > t`` and ``|g| - t >= 1`` are the same statement, and because
        ``Sign`` and ``ReLU`` are both exact on all three devices.
        """
        magnitude = ops.absolute(gradient)
        steps = None
        for threshold in GRADIENT_THRESHOLDS:
            above = ops.sign(
                ops.relu(
                    ops.subtract(magnitude, ops.constant([[[[float(threshold)]]]], ov.Type.f32))
                )
            )
            steps = above if steps is None else ops.add(steps, above)
        return ops.multiply(ops.sign(gradient), steps)

    context = ops.add(
        ops.add(
            ops.multiply(
                quantise(ops.subtract(d, b)),
                ops.constant([[[[float(CONTEXT_RADIX**2)]]]], ov.Type.f32),
            ),
            ops.multiply(
                quantise(ops.subtract(b, c)),
                ops.constant([[[[float(CONTEXT_RADIX)]]]], ov.Type.f32),
            ),
        ),
        quantise(ops.subtract(c, a)),
    )

    outputs = [
        ops.result(ops.convert(prediction, ov.Type.i32), name="prediction"),
        ops.result(ops.convert(context, ov.Type.i32), name="context"),
    ]
    return ov.Model(outputs, [source], "jpegls_med_context")


class VerificationError(RuntimeError):
    """A device's front half disagrees with the CPU reference.

    Never absorbed into a fallback: a wrong prediction produces a valid JPEG-LS
    stream that decodes to the wrong image, so continuing would ship corruption.
    """


@dataclass
class MedContextGraph:
    """The front half compiled for one device and one exact plane shape.

    One compiled model per plane shape, because the NPU plugin compiles for a
    static shape. FRAPPE has five code planes per operating point, so this is
    five models, not the reference's 256x256 tiling -- tiling exists there
    because the archived library demanded one fixed shape, and OpenVINO does not.
    """

    rows: int
    cols: int
    device: str = "CPU"
    core: object = None
    properties: dict | None = None

    def __post_init__(self) -> None:
        from .openvino_runtime import _core, require_device

        self.core = _core(self.core)
        self.device = require_device(self.device, self.core)
        self.compiled = self.core.compile_model(
            _build_model(self.rows, self.cols), self.device, self.properties or {}
        )
        self.request = self.compiled.create_infer_request()

    @property
    def execution_devices(self) -> list[str]:
        value = self.compiled.get_property("EXECUTION_DEVICES")
        return list(value) if not isinstance(value, str) else [value]

    def __call__(self, line_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        buffer = np.asarray(line_buffer)
        if buffer.shape != (self.rows + 1, self.cols + 2):
            raise ValueError(
                f"graph was compiled for a {self.rows}x{self.cols} plane, "
                f"whose line buffer is {(self.rows + 1, self.cols + 2)}; "
                f"got {buffer.shape}"
            )
        self.request.infer([buffer.astype(np.float32)[None, None]])
        prediction = np.array(self.request.get_output_tensor(0).data, copy=True)[0, 0]
        context = np.array(self.request.get_output_tensor(1).data, copy=True)[0, 0]
        return prediction, context


def precompute(
    plane: np.ndarray,
    device: str = "CPU",
    core=None,
    verify: str = "all",
    properties: dict | None = None,
) -> dict:
    """Run the front half for one uint8 plane and check it against the CPU.

    ``verify="all"`` is the default rather than an option because the failure
    mode it guards against is silent. Set ``verify="none"`` only to time the
    device without the oracle's cost, and never on a path that writes a file.
    """
    if verify not in {"all", "none"}:
        raise ValueError(f"verify must be 'all' or 'none', got {verify!r}")
    plane = np.asarray(plane)
    line_buffer = build_line_buffer(plane)
    graph = MedContextGraph(plane.shape[0], plane.shape[1], device, core, properties)
    prediction, context = graph(line_buffer)
    report = {
        "device": graph.device,
        "execution_devices": graph.execution_devices,
        "shape": list(plane.shape),
        "verified": verify == "all",
    }
    if verify == "all":
        want_prediction, want_context = reference_med_and_context(line_buffer)
        for name, want, got in (
            ("prediction", want_prediction, prediction),
            ("context", want_context, context),
        ):
            wrong = int((want != got).sum())
            if wrong:
                where = np.argwhere(want != got)[0]
                raise VerificationError(
                    f"{graph.device} {name}: {wrong}/{want.size} samples differ on a "
                    f"{plane.shape[0]}x{plane.shape[1]} plane; first at "
                    f"{tuple(int(v) for v in where)} want {want[tuple(where)]} "
                    f"got {got[tuple(where)]}, max |difference| "
                    f"{int(np.abs(want - got).max())}"
                )
    return {**report, "prediction": prediction, "context": context}
