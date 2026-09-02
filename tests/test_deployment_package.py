"""The packaged NPU INT8 deployment: one directory, one manifest, no drift.

These tests run in the deployment environment (uv, group ``deploy``) because
they exercise NNCF's ONNX backend and OpenVINO. Fixtures are the small model
and synthetic images only -- no real checkpoints or private data.
"""

import json

import pytest
import torch
from test_quantization import SCHEDULE, make_config, write_anonymous_split

from src.compressors.frappe.harness.quantization import sha256_of

REQUIRED = ["manifest.json", "encoder_fp32.onnx", "encoder_int8_qdq.onnx",
            "encoder_int8_64x64.xml", "encoder_int8_64x64.bin", "decoder.onnx",
            "package_report.json"]


@pytest.fixture
def model():
    torch.manual_seed(0)
    from src.compressors.frappe.prefix import JointPrefixFRAPPE

    return JointPrefixFRAPPE(make_config()).eval()


def package(model, tmp_path, extra_args=()):
    """Package the small model through the official tool and return the manifest."""
    torch.save({"config": vars(make_config()), "model": model.state_dict(),
                "iteration": 4}, tmp_path / "base.pth.tar")
    write_anonymous_split(tmp_path / "data", "train", 2)
    write_anonymous_split(tmp_path / "data", "validation", 1, seed=1)
    output = tmp_path / "package"
    from tools.package_npu_int8 import main as package_main

    package_main([
        "--checkpoint", str(tmp_path / "base.pth.tar"),
        "--dataset-root", str(tmp_path / "data"),
        "--calibration-split", "train", "--calibration-images", "2",
        "--output-dir", str(output),
        "--height", "64", "--width", "64",
        "--target-device", "NPU", "--ptq-preset", "performance",
        "--bias-correction", "fast",
        *extra_args,
    ])
    return json.loads((output / "manifest.json").read_text()), output


def test_package_emits_the_contract_directory(model, tmp_path):
    """The five artifacts plus decoder and report, with a hash-bound manifest."""
    manifest, output = package(model, tmp_path)
    assert sorted(path.name for path in output.iterdir()) == sorted(REQUIRED)
    for name, info in manifest["artifacts"].items():
        assert "/" not in name, "manifest carries relative file names only"
        assert info["sha256"] == sha256_of(output / name)
        assert info["bytes"] == (output / name).stat().st_size
    assert manifest["image_shape_nchw"] == [1, 3, 64, 64]
    assert manifest["plane_order"] == list(range(len(SCHEDULE)))
    assert manifest["prefix"] == model.n_channels
    assert manifest["calibration"] == {"split": "train", "image_indices": [0, 1]}


def test_qdq_package_keeps_the_codec_boundary_out_of_qdq(model, tmp_path):
    """Analysis convolutions go int8; compander, round and clamp stay untouched."""
    import onnx

    _, output = package(model, tmp_path)
    quantized = onnx.load(str(output / "encoder_int8_qdq.onnx"))
    onnx.checker.check_model(quantized)
    quantizers = [node for node in quantized.graph.node if node.op_type == "QuantizeLinear"]
    assert len(quantizers) == 1, "one shared image quantizer feeds every conv"
    weight_dequantizers = [node for node in quantized.graph.node
                           if node.op_type == "DequantizeLinear"
                           and "encoder.analysis" in node.name]
    assert len(weight_dequantizers) == len(SCHEDULE)
    assert all("/companders." not in node.name for node in quantizers
               + weight_dequantizers)
    fp32_ops = [node.op_type for node in onnx.load(
        str(output / "encoder_fp32.onnx")).graph.node]
    int8_ops = [node.op_type for node in quantized.graph.node]
    assert fp32_ops.count("Clip") == int8_ops.count("Clip") == len(SCHEDULE)
    assert fp32_ops.count("Round") == int8_ops.count("Round") == len(SCHEDULE)


def test_static_ir_carries_the_frozen_resolution(model, tmp_path):
    """The NPU IR is compiled at the manifest's NCHW and compiled on CPU."""
    import openvino as ov

    manifest, output = package(model, tmp_path)
    ir = ov.Core().read_model(str(output / "encoder_int8_64x64.xml"))
    assert ir.input(0).shape == [1, 3, 64, 64]
    assert [tuple(result.shape) for result in
            ov.Core().compile_model(ir, "CPU")(
                {0: torch.zeros(1, 3, 64, 64, dtype=torch.uint8).numpy()})] == \
        [tuple(shape) for shape in manifest["plane_shapes"]]


def test_decoder_ships_in_the_same_package(model, tmp_path):
    """decoder.onnx takes the packaged planes in index order, uint8 to uint8."""
    import onnx
    import onnxruntime

    manifest, output = package(model, tmp_path)
    decoder = onnx.load(str(output / "decoder.onnx"))
    onnx.checker.check_model(decoder)
    session = onnxruntime.InferenceSession(str(output / "decoder.onnx"),
                                           providers=["CPUExecutionProvider"])
    assert [entry.name for entry in session.get_inputs()] == manifest["plane_names"]
    output = session.get_outputs()[0]
    assert output.name == "reconstruction"
    # H and W stay symbolic in the decoder: it serves every resolution the
    # frozen encoder can produce, which the analysis patch algebra guarantees.
    assert output.shape[:2] == manifest["image_shape_nchw"][:2]
    assert all(isinstance(dim, str) for dim in output.shape[2:])
