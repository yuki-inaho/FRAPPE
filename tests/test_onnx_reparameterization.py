"""Proofs and refusal tests for the deployment-only ONNX rewrites."""

from __future__ import annotations

import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from src.compressors.frappe.harness.onnx_reparameterization import (  # noqa: E402
    RewriteError,
    fold_encoder_input_normalization,
    fold_fixed_prefix_affine,
    fuse_tanh_gelu,
    op_counts,
    replace_nonoverlap_convtranspose,
)


def model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "fixture", inputs, outputs, initializers)
    result = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    result.ir_version = 10
    onnx.checker.check_model(result)
    return onnx.shape_inference.infer_shapes(result)


def run(graph, feeds):
    session = ort.InferenceSession(
        graph.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


def encoder_fixture(*, padded=False):
    rng = np.random.default_rng(12)
    weights = rng.normal(0, 0.2, (2, 3, 2, 2)).astype(np.float32)
    bias = rng.normal(0, 0.1, (2,)).astype(np.float32)
    pads = [1, 1, 1, 1] if padded else [0, 0, 0, 0]
    output_shape = [1, 2, 3, 3] if padded else [1, 2, 2, 2]
    initializers = [
        numpy_helper.from_array(np.asarray(127.5, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "offset"),
        numpy_helper.from_array(weights, "weight"),
        numpy_helper.from_array(bias, "bias"),
    ]
    nodes = [
        helper.make_node("Cast", ["image"], ["image_float"], name="cast", to=TensorProto.FLOAT),
        helper.make_node("Div", ["image_float", "scale"], ["scaled"], name="normalize_scale"),
        helper.make_node("Sub", ["scaled", "offset"], ["normalized"], name="normalize_offset"),
        helper.make_node(
            "Conv",
            ["normalized", "weight", "bias"],
            ["analysis"],
            name="analysis_p2",
            pads=pads,
            strides=[2, 2],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node("Round", ["analysis"], ["codes"], name="round"),
        helper.make_node("Cast", ["codes"], ["plane"], name="plane_cast", to=TensorProto.INT8),
    ]
    return model(
        nodes,
        [helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("plane", TensorProto.INT8, output_shape)],
        initializers,
    )


def prefix_fixture():
    rng = np.random.default_rng(23)
    mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 1, 1)
    weight = rng.normal(0, 0.15, (2, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(0, 0.1, (2,)).astype(np.float32)
    scale = np.asarray([0.8, 1.2], dtype=np.float32).reshape(1, 2, 1, 1)
    beta = np.asarray([-0.1, 0.2], dtype=np.float32).reshape(1, 2, 1, 1)
    initializers = [
        numpy_helper.from_array(mask, "mask"),
        numpy_helper.from_array(np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), "pads"),
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(scale, "prefix_scale"),
        numpy_helper.from_array(beta, "prefix_bias"),
    ]
    nodes = [
        helper.make_node("Mul", ["input", "mask"], ["masked"], name="prefix_mask"),
        helper.make_node("Pad", ["masked", "pads"], ["padded"], name="reflect_pad", mode="reflect"),
        helper.make_node("Conv", ["padded", "weight", "bias"], ["first"], name="first_conv"),
        helper.make_node("Mul", ["first", "prefix_scale"], ["scaled"], name="prefix_scale_mul"),
        helper.make_node("Add", ["scaled", "prefix_bias"], ["output"], name="prefix_bias_add"),
    ]
    return model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializers,
    )


def deconv_fixture(*, kernel=2, stride=2):
    rng = np.random.default_rng(34)
    weight = rng.normal(0, 0.2, (2, 3, kernel, kernel)).astype(np.float32)
    bias = rng.normal(0, 0.1, (3,)).astype(np.float32)
    nodes = [
        helper.make_node(
            "ConvTranspose",
            ["input", "weight", "bias"],
            ["output"],
            name="head_deconv",
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[0, 0, 0, 0],
            output_padding=[0, 0],
            dilations=[1, 1],
            group=1,
        )
    ]
    size = (3 - 1) * stride + kernel
    return model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, size, size])],
        [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")],
    )


def gelu_fixture(*, cubic_coefficient=0.044715):
    constants = {
        "three": 3.0,
        "coefficient": cubic_coefficient,
        "sqrt_two_over_pi": 0.7978845608028654,
        "one": 1.0,
        "half": 0.5,
    }
    initializers = [
        numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)
        for name, value in constants.items()
    ]
    nodes = [
        helper.make_node("Pow", ["input", "three"], ["power"]),
        helper.make_node("Mul", ["coefficient", "power"], ["cubic"]),
        helper.make_node("Add", ["input", "cubic"], ["inner"]),
        helper.make_node("Mul", ["sqrt_two_over_pi", "inner"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["tanh"]),
        helper.make_node("Add", ["tanh", "one"], ["plus_one"]),
        helper.make_node("Mul", ["half", "plus_one"], ["half_path"]),
        helper.make_node("Mul", ["input", "half_path"], ["output"], name="gelu_root"),
    ]
    return model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4])],
        initializers,
    )


def test_input_normalization_fold_preserves_integer_planes():
    original = encoder_fixture()
    rewritten = fold_encoder_input_normalization(original)
    assert rewritten.changed == 1
    assert op_counts(rewritten.model)["Conv"] == 1
    assert "Div" not in op_counts(rewritten.model)
    assert "Sub" not in op_counts(rewritten.model)
    image = np.random.default_rng(1).integers(0, 256, (1, 3, 4, 4), dtype=np.uint8)
    expected, = run(original, {"image": image})
    actual, = run(rewritten.model, {"image": image})
    np.testing.assert_array_equal(actual, expected)
    json.dumps(rewritten.as_dict())


def test_input_normalization_fold_refuses_padding():
    with pytest.raises(RewriteError, match="padding-free"):
        fold_encoder_input_normalization(encoder_fixture(padded=True))


def test_fixed_prefix_mask_scale_and_bias_fold_into_first_conv():
    original = prefix_fixture()
    rewritten = fold_fixed_prefix_affine(original)
    assert rewritten.changed == 1
    assert op_counts(rewritten.model) == {"Conv": 1, "Pad": 1}
    sample = np.random.default_rng(2).normal(size=(1, 3, 4, 4)).astype(np.float32)
    expected, = run(original, {"input": sample})
    actual, = run(rewritten.model, {"input": sample})
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


def test_nonoverlap_deconv_becomes_phase_conv_and_depth_to_space():
    original = deconv_fixture()
    rewritten = replace_nonoverlap_convtranspose(original)
    assert rewritten.changed == 1
    assert op_counts(rewritten.model) == {"Conv": 1, "DepthToSpace": 1}
    depth_to_space = next(
        node for node in rewritten.model.graph.node if node.op_type == "DepthToSpace"
    )
    attributes = {attribute.name: helper.get_attribute_value(attribute)
                  for attribute in depth_to_space.attribute}
    assert attributes == {"blocksize": 2, "mode": b"CRD"}
    sample = np.random.default_rng(3).normal(size=(1, 2, 3, 3)).astype(np.float32)
    expected, = run(original, {"input": sample})
    actual, = run(rewritten.model, {"input": sample})
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_overlapping_deconv_is_refused():
    with pytest.raises(RewriteError, match="phase kernels up to \\(2, 2\\)"):
        replace_nonoverlap_convtranspose(deconv_fixture(kernel=4, stride=2))


def test_expanded_tanh_gelu_becomes_standard_opset20_gelu():
    original = gelu_fixture()
    rewritten = fuse_tanh_gelu(original)
    assert rewritten.changed == 1
    assert op_counts(rewritten.model) == {"Gelu": 1}
    assert next(entry.version for entry in rewritten.model.opset_import if not entry.domain) == 20
    sample = np.random.default_rng(4).normal(size=(2, 3, 4)).astype(np.float32)
    expected, = run(original, {"input": sample})
    actual, = run(rewritten.model, {"input": sample})
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def test_non_gelu_tanh_expression_is_not_fused():
    assert fuse_tanh_gelu(gelu_fixture(cubic_coefficient=0.04)).changed == 0
