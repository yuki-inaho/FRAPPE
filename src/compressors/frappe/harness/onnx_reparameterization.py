"""Guarded, deployment-only reparameterizations for exported FRAPPE graphs.

These are not generic ONNX optimizer passes.  Each rewrite matches the narrow
algebra FRAPPE actually exports, derives new constant weights, and refuses graph
shapes whose equivalence has not been established.  The training model and its
checkpoint stay untouched; callers receive a deep-copied ``ModelProto``.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

GELU_CUBIC_COEFFICIENT = 0.044715
GELU_SQRT_TWO_OVER_PI = 0.7978845608028654


class RewriteError(ValueError):
    """The requested graph is close to a known pattern but violates its proof."""


@dataclass(frozen=True)
class RewriteResult:
    """One transformed model and the evidence needed to audit the match."""

    model: Any
    changed: int
    details: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {"changed": self.changed, "details": _json_compatible(list(self.details))}


def _json_compatible(value: Any) -> Any:
    """Turn ONNX attribute values into deterministic JSON-compatible values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value


def op_counts(model: Any) -> dict[str, int]:
    """A stable operation inventory for reports and regression tests."""
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def _onnx_modules():
    import onnx
    from onnx import helper, numpy_helper

    return onnx, helper, numpy_helper


def _attributes(node: Any) -> dict[str, Any]:
    _, helper, _ = _onnx_modules()
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _initializers(model: Any) -> dict[str, Any]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _producers(model: Any) -> dict[str, Any]:
    return {output: node for node in model.graph.node for output in node.output}


def _consumers(model: Any) -> dict[str, list[Any]]:
    users: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for value in node.input:
            users.setdefault(value, []).append(node)
    return users


def _constant_array(name: str, initializers: dict[str, Any]) -> np.ndarray | None:
    _, _, numpy_helper = _onnx_modules()
    value = initializers.get(name)
    return None if value is None else np.asarray(numpy_helper.to_array(value))


def _scalar(name: str, initializers: dict[str, Any]) -> float | None:
    value = _constant_array(name, initializers)
    if value is None or value.size != 1:
        return None
    return float(value.reshape(()))


def _channel_vector(value: np.ndarray | None, channels: int, label: str) -> np.ndarray:
    if value is None or value.size != channels:
        shape = None if value is None else tuple(value.shape)
        raise RewriteError(f"{label} must contain exactly {channels} channel values, got {shape}")
    return value.reshape(channels)


def _other_constant_input(node: Any, data_name: str, initializers: dict[str, Any]):
    if len(node.input) != 2 or data_name not in node.input:
        return None
    other = node.input[1] if node.input[0] == data_name else node.input[0]
    value = _constant_array(other, initializers)
    return None if value is None else (other, value)


def _scalar_and_data_input(
    node: Any,
    expected: float,
    initializers: dict[str, Any],
) -> str | None:
    """Return the non-constant input of a binary node with one expected scalar."""
    if len(node.input) != 2:
        return None
    for constant_index, data_index in ((0, 1), (1, 0)):
        value = _scalar(node.input[constant_index], initializers)
        if value is not None and np.isclose(value, expected, rtol=1e-6, atol=1e-7):
            return node.input[data_index]
    return None


def _unique_name(model: Any, base: str) -> str:
    occupied = {
        value
        for node in model.graph.node
        for value in (*node.input, *node.output)
        if value
    }
    occupied.update(initializer.name for initializer in model.graph.initializer)
    if base not in occupied:
        return base
    index = 1
    while f"{base}_{index}" in occupied:
        index += 1
    return f"{base}_{index}"


def _append_initializer(model: Any, name: str, value: np.ndarray) -> str:
    _, _, numpy_helper = _onnx_modules()
    unique = _unique_name(model, name)
    model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value), unique))
    return unique


def _remove_nodes(model: Any, removed: set[int]) -> None:
    kept = [node for node in model.graph.node if id(node) not in removed]
    model.graph.ClearField("node")
    model.graph.node.extend(kept)


def _remove_unused_initializers(model: Any) -> None:
    used = {value for node in model.graph.node for value in node.input}
    used.update(value.name for value in model.graph.output)
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept)


def _validated(model: Any) -> Any:
    onnx, _, _ = _onnx_modules()
    _remove_unused_initializers(model)
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model)


def fold_encoder_input_normalization(
    source: Any,
    *,
    expected_scale: float = 127.5,
    expected_offset: float = 1.0,
) -> RewriteResult:
    """Fold ``Cast(X) / scale - offset`` into padding-free analysis convolutions.

    For each matched convolution, ``W' = W / scale`` and
    ``b' = b - offset * sum(W)``.  The uint8-to-float Cast remains.  Direct
    initializer weights are deliberate: applying this to a Q/DQ or FakeQuantize
    weight path without also transforming its scales would change the model.
    """
    model = copy.deepcopy(source)
    initializers = _initializers(model)
    producers = _producers(model)
    consumers = _consumers(model)
    removed: set[int] = set()
    details: list[dict[str, Any]] = []

    for sub in list(model.graph.node):
        if sub.op_type != "Sub" or len(sub.input) != 2:
            continue
        offset = _scalar(sub.input[1], initializers)
        div = producers.get(sub.input[0])
        if offset is None or div is None or div.op_type != "Div" or len(div.input) != 2:
            continue
        scale = _scalar(div.input[1], initializers)
        cast = producers.get(div.input[0])
        if scale is None or cast is None or cast.op_type != "Cast":
            continue
        if not np.isclose(scale, expected_scale) or not np.isclose(offset, expected_offset):
            continue
        graph_outputs = {value.name for value in model.graph.output}
        if div.output[0] in graph_outputs or sub.output[0] in graph_outputs:
            raise RewriteError("normalization intermediates must not also be graph outputs")
        if consumers.get(div.output[0], []) != [sub]:
            raise RewriteError("normalization Div must be consumed only by its matching Sub")

        convs = consumers.get(sub.output[0], [])
        if not convs or any(node.op_type != "Conv" for node in convs):
            raise RewriteError("normalization output must be consumed only by analysis Conv nodes")

        prepared = []
        for conv in convs:
            attrs = _attributes(conv)
            pads = tuple(attrs.get("pads", (0, 0, 0, 0)))
            dilations = tuple(attrs.get("dilations", (1, 1)))
            group = int(attrs.get("group", 1))
            if pads != (0, 0, 0, 0) or dilations != (1, 1) or group != 1:
                raise RewriteError(
                    f"{conv.name or conv.output[0]} is not a padding-free, dilation-1, group-1 Conv"
                )
            if len(conv.input) < 2:
                raise RewriteError("analysis Conv has no constant weight input")
            weight = _constant_array(conv.input[1], initializers)
            if weight is None or weight.ndim != 4:
                raise RewriteError("analysis Conv weight must be a direct rank-4 initializer")
            if len(conv.input) >= 3:
                bias = _constant_array(conv.input[2], initializers)
                if bias is None:
                    raise RewriteError("analysis Conv bias must be a direct initializer")
            else:
                bias = np.zeros(weight.shape[0], dtype=weight.dtype)
            if bias.shape != (weight.shape[0],):
                raise RewriteError("analysis Conv bias shape does not match its output channels")
            prepared.append((conv, weight, bias, attrs))

        for conv, weight, bias, attrs in prepared:
            folded_weight = (weight / np.asarray(scale, dtype=weight.dtype)).astype(
                weight.dtype, copy=False
            )
            folded_bias = (
                bias - np.asarray(offset, dtype=weight.dtype) * weight.sum(axis=(1, 2, 3))
            ).astype(bias.dtype, copy=False)
            weight_name = _append_initializer(
                model, f"{conv.input[1]}_input_normalization_folded", folded_weight
            )
            bias_base = conv.input[2] if len(conv.input) >= 3 else f"{conv.name}_bias"
            bias_name = _append_initializer(
                model, f"{bias_base}_input_normalization_folded", folded_bias
            )
            conv.input[0] = cast.output[0]
            conv.input[1] = weight_name
            if len(conv.input) >= 3:
                conv.input[2] = bias_name
            else:
                conv.input.append(bias_name)
            details.append(
                {
                    "conv": conv.name or conv.output[0],
                    "weight_shape": list(weight.shape),
                    "scale": scale,
                    "offset": offset,
                    "attributes": _json_compatible(attrs),
                }
            )
        removed.update((id(sub), id(div)))

    if removed:
        _remove_nodes(model, removed)
    return RewriteResult(_validated(model), len(details), tuple(details))


def fold_fixed_prefix_affine(source: Any, *, conv_name: str | None = None) -> RewriteResult:
    """Fold a frozen prefix mask and post-Conv channel affine into one Conv.

    The supported graph is ``[Mul(mask) ->] [Pad(reflect) ->] Conv
    [-> Mul(scale)] [-> Add(beta)]``.  Auto-discovery is intentionally limited
    to a Conv reached through reflection Pad; a caller may name a Conv explicitly
    for a synthetic or already-normalized graph.
    """
    model = copy.deepcopy(source)
    initializers = _initializers(model)
    producers = _producers(model)
    consumers = _consumers(model)
    removed: set[int] = set()
    details: list[dict[str, Any]] = []

    for conv in list(model.graph.node):
        if conv.op_type != "Conv" or (conv_name is not None and conv.name != conv_name):
            continue
        attrs = _attributes(conv)
        if int(attrs.get("group", 1)) != 1:
            continue
        pad = producers.get(conv.input[0])
        reflect_pad = pad is not None and pad.op_type == "Pad" and _attributes(pad).get(
            "mode", b"constant"
        ) == b"reflect"
        if conv_name is None and not reflect_pad:
            continue

        weight = _constant_array(conv.input[1], initializers) if len(conv.input) >= 2 else None
        if weight is None or weight.ndim != 4:
            raise RewriteError(f"{conv.name or conv.output[0]} weight must be a rank-4 initializer")
        out_channels, in_channels = weight.shape[:2]
        if len(conv.input) >= 3:
            bias = _constant_array(conv.input[2], initializers)
            if bias is None:
                raise RewriteError("first Conv bias must be a direct initializer")
        else:
            bias = np.zeros(out_channels, dtype=weight.dtype)
        if bias.shape != (out_channels,):
            raise RewriteError("first Conv bias shape does not match output channels")

        pre_value = pad.input[0] if reflect_pad else conv.input[0]
        mask_node = producers.get(pre_value)
        mask_value = None
        dynamic_input = None
        if mask_node is not None and mask_node.op_type == "Mul":
            for candidate in mask_node.input:
                other = (
                    mask_node.input[1]
                    if candidate == mask_node.input[0]
                    else mask_node.input[0]
                )
                constant = _constant_array(other, initializers)
                if constant is not None:
                    dynamic_input, mask_value = candidate, constant
                    break

        scale_node = None
        beta_node = None
        scale_value = None
        beta_value = None
        current_output = conv.output[0]
        current_users = consumers.get(current_output, [])
        if len(current_users) == 1 and current_users[0].op_type == "Mul":
            candidate = _other_constant_input(current_users[0], current_output, initializers)
            if candidate is not None:
                scale_node = current_users[0]
                scale_value = candidate[1]
                current_output = scale_node.output[0]
                current_users = consumers.get(current_output, [])
        if len(current_users) == 1 and current_users[0].op_type == "Add":
            candidate = _other_constant_input(current_users[0], current_output, initializers)
            if candidate is not None:
                beta_node = current_users[0]
                beta_value = candidate[1]
                current_output = beta_node.output[0]

        if mask_value is None and scale_value is None and beta_value is None:
            continue
        if mask_node is not None and mask_value is not None:
            if len(consumers.get(mask_node.output[0], [])) != 1:
                raise RewriteError("prefix mask Mul must have exactly one consumer")
            mask = _channel_vector(mask_value, in_channels, "prefix mask")
            weight = weight * mask.reshape(1, in_channels, 1, 1)
        if scale_node is not None:
            scale = _channel_vector(scale_value, out_channels, "prefix scale")
            weight = weight * scale.reshape(out_channels, 1, 1, 1)
            bias = bias * scale
        if beta_node is not None:
            beta = _channel_vector(beta_value, out_channels, "prefix bias")
            bias = bias + beta

        weight_name = _append_initializer(model, f"{conv.input[1]}_fixed_prefix_folded", weight)
        bias_base = conv.input[2] if len(conv.input) >= 3 else f"{conv.name}_bias"
        bias_name = _append_initializer(model, f"{bias_base}_fixed_prefix_folded", bias)
        conv.input[1] = weight_name
        if len(conv.input) >= 3:
            conv.input[2] = bias_name
        else:
            conv.input.append(bias_name)
        if dynamic_input is not None:
            if reflect_pad:
                pad.input[0] = dynamic_input
            else:
                conv.input[0] = dynamic_input
            removed.add(id(mask_node))
        if scale_node is not None:
            removed.add(id(scale_node))
        if beta_node is not None:
            removed.add(id(beta_node))
        conv.output[0] = current_output
        details.append(
            {
                "conv": conv.name or conv.output[0],
                "weight_shape": list(weight.shape),
                "folded_mask": mask_value is not None,
                "folded_scale": scale_value is not None,
                "folded_bias": beta_value is not None,
                "reflect_pad_preserved": reflect_pad,
            }
        )

    if removed:
        _remove_nodes(model, removed)
    return RewriteResult(_validated(model), len(details), tuple(details))


def replace_nonoverlap_convtranspose(source: Any, *, strict: bool = True) -> RewriteResult:
    """Replace FRAPPE's non-overlapping Deconv with Conv1x1 + DepthToSpace.

    Only ``kernel == stride``, zero padding/output-padding, dilation one and
    group one are proven here.  The ONNX ConvTranspose weight layout is
    ``(C_in, C_out, kH, kW)``; CRD DepthToSpace needs phase channels ordered as
    ``out * r**2 + kh * r + kw``.
    """
    model = copy.deepcopy(source)
    _, helper, _ = _onnx_modules()
    initializers = _initializers(model)
    replacements: dict[int, list[Any]] = {}
    details: list[dict[str, Any]] = []

    for index, node in enumerate(model.graph.node):
        if node.op_type != "ConvTranspose":
            continue
        attrs = _attributes(node)
        weight = _constant_array(node.input[1], initializers) if len(node.input) >= 2 else None
        reasons = []
        if weight is None or weight.ndim != 4:
            reasons.append("weight is not a direct rank-4 initializer")
        else:
            kernel = tuple(attrs.get("kernel_shape", weight.shape[2:]))
            strides = tuple(attrs.get("strides", (1, 1)))
            if kernel[0] != kernel[1] or strides != kernel:
                phase_kernel = tuple(
                    (size + stride - 1) // stride
                    for size, stride in zip(kernel, strides, strict=True)
                )
                reasons.append(
                    f"current pass requires square kernel == stride, got kernel {kernel} "
                    f"and strides {strides}; a general polyphase rewrite would need "
                    f"phase kernels up to {phase_kernel} plus boundary handling"
                )
        if tuple(attrs.get("pads", (0, 0, 0, 0))) != (0, 0, 0, 0):
            reasons.append("padding must be zero")
        if tuple(attrs.get("output_padding", (0, 0))) != (0, 0):
            reasons.append("output_padding must be zero")
        if tuple(attrs.get("dilations", (1, 1))) != (1, 1):
            reasons.append("dilation must be one")
        if int(attrs.get("group", 1)) != 1:
            reasons.append("group must be one")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", b""):
            reasons.append("auto_pad must be NOTSET")
        if "output_shape" in attrs:
            reasons.append("explicit output_shape is unsupported")
        if reasons:
            if strict:
                raise RewriteError(f"{node.name or node.output[0]}: " + "; ".join(reasons))
            continue

        in_channels, out_channels, r, _ = weight.shape
        if len(node.input) >= 3:
            bias = _constant_array(node.input[2], initializers)
            if bias is None or bias.shape != (out_channels,):
                raise RewriteError("ConvTranspose bias must be a direct output-channel initializer")
        else:
            bias = np.zeros(out_channels, dtype=weight.dtype)
        phase_weight = weight.transpose(1, 2, 3, 0).reshape(
            out_channels * r * r, in_channels, 1, 1
        )
        phase_bias = np.repeat(bias, r * r)
        weight_name = _append_initializer(model, f"{node.input[1]}_phase_1x1", phase_weight)
        bias_base = node.input[2] if len(node.input) >= 3 else f"{node.name}_bias"
        bias_name = _append_initializer(model, f"{bias_base}_phase_1x1", phase_bias)
        intermediate = _unique_name(model, f"{node.output[0]}_phase_channels")
        conv = helper.make_node(
            "Conv",
            [node.input[0], weight_name, bias_name],
            [intermediate],
            name=f"{node.name or node.output[0]}_phase_1x1",
            group=1,
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
        )
        depth_to_space = helper.make_node(
            "DepthToSpace",
            [intermediate],
            list(node.output),
            name=f"{node.name or node.output[0]}_depth_to_space",
            blocksize=r,
            mode="CRD",
        )
        replacements[index] = [conv, depth_to_space]
        details.append(
            {
                "convtranspose": node.name or node.output[0],
                "weight_shape": list(weight.shape),
                "phase_weight_shape": list(phase_weight.shape),
                "blocksize": r,
                "mode": "CRD",
            }
        )

    if replacements:
        nodes = []
        for index, node in enumerate(model.graph.node):
            nodes.extend(replacements.get(index, [node]))
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
    return RewriteResult(_validated(model), len(details), tuple(details))


def _match_tanh_gelu(
    root: Any,
    producers: dict[str, Any],
    consumers: dict[str, list[Any]],
    initializers: dict[str, Any],
) -> tuple[str, tuple[Any, ...]] | None:
    """Match PyTorch's fully expanded tanh-approximate GELU ending at ``root``."""
    if root.op_type != "Mul" or len(root.input) != 2:
        return None
    for data, half_output in ((root.input[0], root.input[1]), (root.input[1], root.input[0])):
        half = producers.get(half_output)
        if half is None or half.op_type != "Mul":
            continue
        plus_one_output = _scalar_and_data_input(half, 0.5, initializers)
        plus_one = producers.get(plus_one_output)
        if plus_one is None or plus_one.op_type != "Add":
            continue
        tanh_output = _scalar_and_data_input(plus_one, 1.0, initializers)
        tanh = producers.get(tanh_output)
        if tanh is None or tanh.op_type != "Tanh" or len(tanh.input) != 1:
            continue
        scaled = producers.get(tanh.input[0])
        if scaled is None or scaled.op_type != "Mul":
            continue
        inner_output = _scalar_and_data_input(scaled, GELU_SQRT_TWO_OVER_PI, initializers)
        inner = producers.get(inner_output)
        if inner is None or inner.op_type != "Add" or data not in inner.input:
            continue
        cubic_output = inner.input[1] if inner.input[0] == data else inner.input[0]
        cubic = producers.get(cubic_output)
        if cubic is None or cubic.op_type != "Mul":
            continue
        power_output = _scalar_and_data_input(cubic, GELU_CUBIC_COEFFICIENT, initializers)
        power = producers.get(power_output)
        if power is None or power.op_type != "Pow" or len(power.input) != 2:
            continue
        if power.input[0] != data or _scalar(power.input[1], initializers) != 3.0:
            continue
        chain = (power, cubic, inner, scaled, tanh, plus_one, half)
        if any(consumers.get(node.output[0], []) != [next_node]
               for node, next_node in zip(chain, (*chain[1:], root))):
            continue
        return data, chain
    return None


def fuse_tanh_gelu(source: Any) -> RewriteResult:
    """Replace the exact exported tanh-GELU expansion with ONNX ``Gelu``.

    Standard-domain ``Gelu`` was added in opset 20.  A graph with a proven
    match is upgraded through ONNX's version converter before nodes are
    replaced.  This is primarily graph canonicalization: OpenVINO may already
    recognize and fuse the expanded arithmetic internally, so speedup is never
    assumed without a backend benchmark.
    """
    onnx, helper, _ = _onnx_modules()
    probe = copy.deepcopy(source)
    initializers = _initializers(probe)
    producers = _producers(probe)
    consumers = _consumers(probe)
    if not any(
        _match_tanh_gelu(node, producers, consumers, initializers) is not None
        for node in probe.graph.node
    ):
        return RewriteResult(_validated(probe), 0, ())

    default_opset = next(
        (entry.version for entry in probe.opset_import if entry.domain in ("", "ai.onnx")),
        0,
    )
    model = (
        onnx.version_converter.convert_version(probe, 20)
        if default_opset < 20
        else probe
    )
    initializers = _initializers(model)
    producers = _producers(model)
    consumers = _consumers(model)
    removed: set[int] = set()
    replacements: dict[int, Any] = {}
    details: list[dict[str, Any]] = []
    for index, root in enumerate(model.graph.node):
        match = _match_tanh_gelu(root, producers, consumers, initializers)
        if match is None:
            continue
        data, chain = match
        removed.update(id(node) for node in chain)
        replacements[index] = helper.make_node(
            "Gelu",
            [data],
            list(root.output),
            name=root.name or f"{root.output[0]}_gelu",
            approximate="tanh",
        )
        details.append(
            {
                "root": root.name or root.output[0],
                "input": data,
                "removed_nodes": len(chain) + 1,
                "approximate": "tanh",
                "source_opset": default_opset,
                "output_opset": 20 if default_opset < 20 else default_opset,
            }
        )

    nodes = []
    for index, node in enumerate(model.graph.node):
        if index in replacements:
            nodes.append(replacements[index])
        elif id(node) not in removed:
            nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    return RewriteResult(_validated(model), len(details), tuple(details))
