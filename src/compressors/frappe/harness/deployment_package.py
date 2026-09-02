"""The official INT8 deployment package: five artifacts, one manifest, one hash chain.

A packaged operating point is what travels: the FP32 encoder graph, its
Q/DQ-quantized twin, the OpenVINO IR compiled from that twin at one frozen
resolution, the decoder graph the planes feed, and a manifest binding every
file to the checkpoint, calibration set and measurements it came from. The
quantization boundary is the mixed-precision contract: the five analysis
convolutions go int8 through ONNX Q/DQ pairs, while the companders, the
rounding and the clamp stay ordinary fp32/integer ops because they *are* the
codec's bitstream definition.

The legacy QAT trace export (a torch hook graph carried into ONNX) is a
debugging device, not a package artifact; it is not produced here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..experiment import atomic_json_dump
from .bitstream import prefix_channels
from .deployment import plane_names
from .quantization import (
    TrainableEncoder,
    export_encoder_onnx,
    op_inventory,
    quantize_onnx_encoder,
    require_disjoint_calibration_samples,
    save_openvino_ir,
    sha256_of,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..prefix import JointPrefixFRAPPE


def export_graph(
    module: torch.nn.Module,
    sample: tuple,
    path: Path,
    input_names: list[str],
    output_names: list[str],
    shapes,
    opset: int = 18,
) -> dict:
    """One graph through the dynamo exporter with the repository's shape contract."""
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        sample,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=shapes,
        opset_version=opset,
        dynamo=True,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return {"path": path.name, "bytes": path.stat().st_size, "opset": opset}


def build_package(
    model: JointPrefixFRAPPE,
    checkpoint_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    calibration_split: str,
    calibration_images: int,
    verify_split: str,
    verify_images: int,
    height: int,
    width: int,
    prefix: int | None = None,
    target_device: str = "NPU",
    ptq_preset: str = "performance",
    bias_correction: str = "fast",
    decode_device: str | torch.device = "cpu",
    checkpoint_iteration: int | None = None,
) -> dict:
    """Produce the whole deployment directory and return its manifest.

    The calibration split feeds NNCF's statistics only; the verification split
    is measured through the shipped artifacts and must not overlap it. The
    decoder is exported for ``prefix`` channels and the encoder is frozen to
    ``[1, 3, height, width]`` in its OpenVINO IR.
    """
    import onnx
    import openvino as ov

    from .data import AnonymousImageFolder
    from .deployment import DecoderGraph, EncoderGraph, dynamic_shapes, measure_deployed_conditions

    model.eval()
    prefix = prefix or model.n_channels
    max_ps = max(model.ps)
    if height % max_ps or width % max_ps:
        raise ValueError(
            f"resolution {width}x{height} must be a multiple of the largest patch size {max_ps}"
        )
    if prefix < 1 or prefix > model.n_channels:
        raise ValueError(f"prefix {prefix} outside 1..{model.n_channels}")

    calibration_folder = AnonymousImageFolder(dataset_root, calibration_split)
    if calibration_images > len(calibration_folder):
        raise ValueError(
            f"requested {calibration_images} calibration images but "
            f"{calibration_split} has {len(calibration_folder)}"
        )
    calibration_indices = list(range(calibration_images))
    verify_folder = AnonymousImageFolder(dataset_root, verify_split)
    verify_indices = list(range(min(verify_images, len(verify_folder))))
    require_disjoint_calibration_samples(
        calibration_split, calibration_indices, verify_split, verify_indices
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    names = plane_names(model)
    _, plane_shapes = dynamic_shapes(model)
    sample_image = torch.zeros(1, model.input_channels, height, width, dtype=torch.uint8)
    with torch.no_grad():
        sample_planes = tuple(
            plane.clone() for plane in EncoderGraph(model, uint8_io=True)(sample_image)
        )

    # The encoder ships through the tracer export: the adopted INT8 PTQ was
    # produced from exactly this graph, and NNCF's compander scope is
    # name-based -- the dynamo exporter emits anonymous node names, which
    # would leave the codec boundary un-ignorable. The decoder carries no
    # quantization, so it takes the dynamo exporter and its dynamic_shapes.
    fp32_path = output_dir / "encoder_fp32.onnx"
    encoder_info = export_encoder_onnx(TrainableEncoder(model), fp32_path, sample_image)
    encoder_info["path"] = fp32_path.name
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    decoder_path = output_dir / "decoder.onnx"
    decoder_info = export_graph(
        DecoderGraph(model, prefix, uint8_io=True).eval(),
        sample_planes,
        decoder_path,
        names,
        ["reconstruction"],
        (plane_shapes,),
    )
    # The dynamo exporter externalises weights into <name>.onnx.data; a
    # deployment reads one file per graph, so the weights are folded back in.
    decoder_model = onnx.load(str(decoder_path), load_external_data=True)
    onnx.save_model(decoder_model, str(decoder_path), save_as_external_data=False)
    stray = decoder_path.with_name(decoder_path.name + ".data")
    if stray.is_file():
        stray.unlink()
    onnx.checker.check_model(onnx.load(str(decoder_path)))

    calibration = [
        {"image": calibration_folder.pixels(index).numpy()} for index in calibration_indices
    ]
    qdq_path = output_dir / "encoder_int8_qdq.onnx"
    quantization = quantize_onnx_encoder(
        fp32_path,
        qdq_path,
        calibration,
        target_device=target_device,
        subset_size=calibration_images,
        preset=ptq_preset,
        bias_correction=bias_correction,
    )
    if quantization["quantize_linear"] == 0:
        raise ValueError("the quantized graph carries no QuantizeLinear; refusing to package it")

    image_shape_nchw = [1, model.input_channels, height, width]
    xml_path = output_dir / f"encoder_int8_{width}x{height}.xml"
    ir_info = save_openvino_ir(qdq_path, xml_path, static_shape=image_shape_nchw)
    if ir_info["input_shape"] != image_shape_nchw:
        raise ValueError("the IR's input shape drifted from the requested NCHW")

    # The manifest's geometry comes from the shipped IR itself, not from a
    # paper derivation: what the NPU PC receives is what is recorded here.
    core = ov.Core()
    compiled = {
        "fp32": core.compile_model(core.read_model(str(fp32_path)), "CPU"),
        "int8": core.compile_model(core.read_model(ir_info["xml"]), "CPU"),
    }
    static_outputs = core.read_model(ir_info["xml"]).outputs
    static_shapes = [list(result.partial_shape.get_min_shape()) for result in static_outputs]

    plan = prefix_channels(model.scale_groups, prefix)
    decoder = DecoderGraph(model, prefix, uint8_io=True).to(decode_device).eval()
    conditions = measure_deployed_conditions(
        compiled,
        decoder,
        verify_folder,
        verify_indices,
        scale_groups=model.scale_groups,
        plan=plan,
        prefix=prefix,
        height=height,
        width=width,
        device=torch.device(decode_device),
    )

    artifacts = {
        path.name: {"sha256": sha256_of(path), "bytes": path.stat().st_size}
        for path in (fp32_path, qdq_path, Path(ir_info["xml"]), Path(ir_info["bin"]), decoder_path)
    }
    manifest = {
        "artifacts": artifacts,
        "image_shape_nchw": image_shape_nchw,
        "plane_order": list(range(len(names))),
        "plane_names": names,
        "plane_shapes": static_shapes,
        "plane_dtypes": ["uint8"] * len(names),
        "prefix": prefix,
        "ps_schedule": [ps for ps, _, _ in model.scale_groups],
        "quantization_boundary": {
            "encoder_backend": "NNCF ONNX QDQ PTQ",
            "int8_ops": ["Conv (analysis)"],
            "fp32_ops": ["SoftsignCompander", "Round", "Clip", "plane layout"],
            "preset": ptq_preset,
            "target_device": target_device.upper(),
            "bias_correction": bias_correction,
            "nncf_version": quantization["nncf_version"],
            "quantize_linear": quantization["quantize_linear"],
            "dequantize_linear": quantization["dequantize_linear"],
            "removed_compander_output_qdq": quantization["removed_compander_output_qdq"],
            "ignored_scope_patterns": quantization["ignored_scope_patterns"],
        },
        "calibration": {"split": calibration_split, "image_indices": calibration_indices},
        "rd_baseline": {
            "split": verify_split,
            "image_indices": verify_indices,
            "size": [width, height],
            "bitstream_convention": "PAYLOAD_ONLY",
            "averaging": "aggregate_mse",
            "decoder": "torch DecoderGraph (frozen, fp32)",
            "conditions": conditions,
            "deltas_vs_fp32": {
                "int8": {
                    "d_psnr_db": conditions["int8"]["psnr_db"] - conditions["fp32"]["psnr_db"],
                    "d_bpp": conditions["int8"]["bpp"] - conditions["fp32"]["bpp"],
                }
            },
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "openvino": ov.__version__,
            "numpy": np.__version__,
        },
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")

    report = {
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "checkpoint_iteration": checkpoint_iteration,
        "prefix": prefix,
        "io": "uint8",
        "encoder_fp32": {
            **encoder_info,
            "op_types": op_inventory(onnx.load(str(fp32_path)).graph.node),
        },
        "decoder": {
            **decoder_info,
            "op_types": op_inventory(onnx.load(str(decoder_path)).graph.node),
        },
        "encoder_int8": {
            "quantization": quantization,
            "op_types": op_inventory(onnx.load(str(qdq_path)).graph.node),
            "openvino_ir": {
                **ir_info,
                "op_types": op_inventory(core.read_model(ir_info["xml"]).get_ops()),
            },
        },
        "calibration": {"split": calibration_split, "image_indices": calibration_indices},
        "rd_baseline": manifest["rd_baseline"],
        "versions": manifest["versions"],
    }
    atomic_json_dump(report, output_dir / "package_report.json")
    return manifest
