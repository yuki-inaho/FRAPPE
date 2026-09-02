"""The hybrid runtime: explicit devices, index-bound planes, no silent fallback.

OpenVINO compiles only the device it is asked for, ONNX Runtime only the CUDA
provider, and the JPEG-LS backend only what the caller names. These tests run
in the deployment environment (uv, group ``deploy``) and use a packaged small
model plus a fake ORT session where a real GPU would be needed.
"""

import json

import numpy as np
import pytest
import torch
from test_deployment_package import package


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    """One packaged small-model artifact directory shared by the runtime tests."""
    from test_quantization import make_config

    from src.compressors.frappe.prefix import JointPrefixFRAPPE

    torch.manual_seed(0)
    model = JointPrefixFRAPPE(make_config()).eval()
    root = tmp_path_factory.mktemp("hybrid")
    manifest, output = package(model, root)
    return {"manifest": manifest, "output": output, "model": model, "root": root}


def test_meta_devices_are_rejected(packaged):
    """AUTO, HETERO, MULTI and BATCH are meta-devices; naming one is an error."""
    from src.compressors.frappe.harness.hybrid_runtime import OpenVINOEncoder

    manifest, output = packaged["manifest"], packaged["output"]
    for device in ("AUTO", "HETERO", "MULTI", "BATCH", "auto"):
        with pytest.raises(ValueError, match=r"meta-device|device"):
            OpenVINOEncoder(output, device, manifest)


def test_missing_device_is_refused_not_fallen_back(packaged):
    """A device that is not in available_devices is an error, never a downgrade."""
    import openvino as ov

    from src.compressors.frappe.harness.hybrid_runtime import OpenVINOEncoder

    manifest, output = packaged["manifest"], packaged["output"]
    core = ov.Core()
    absent = next(
        (
            name
            for name in ("NPU", "GPU")
            if not any(candidate.startswith(name) for candidate in core.available_devices)
        ),
        None,
    )
    if absent is None:
        pytest.skip("every requested device exists on this machine")
    with pytest.raises(RuntimeError, match=absent):
        OpenVINOEncoder(output, absent, manifest)


def test_planes_bind_by_output_index(packaged):
    """Runtime planes are taken by index and must match the reference graph exactly."""
    from src.compressors.frappe.harness.data import AnonymousImageFolder
    from src.compressors.frappe.harness.deployment import EncoderGraph
    from src.compressors.frappe.harness.hybrid_runtime import OpenVINOEncoder

    manifest, output, model = packaged["manifest"], packaged["output"], packaged["model"]
    data = AnonymousImageFolder(packaged["root"] / "data", "validation")
    encoder = OpenVINOEncoder(output, "CPU", manifest)
    image = data.pixels(0)
    planes = encoder.encode(image.numpy())
    assert [plane.shape for plane in planes] == [tuple(shape) for shape in manifest["plane_shapes"]]
    assert all(plane.dtype == np.uint8 for plane in planes)
    reference = EncoderGraph(model, uint8_io=True)(image)
    for produced, want in zip(planes, reference):
        assert np.array_equal(produced, want.numpy())


def test_cpu_provider_cannot_run_the_cuda_decoder(packaged, monkeypatch):
    """A decoder session that would not run on CUDA is constructed never."""
    import onnxruntime as ort

    from src.compressors.frappe.harness.hybrid_runtime import CudaOnnxDecoder

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider", "AzureExecutionProvider"]
    )
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        CudaOnnxDecoder(packaged["output"], packaged["manifest"])


def test_cuda_provider_is_passed_and_inputs_are_checked(packaged, monkeypatch):
    """The session request names only CUDA; manifest geometry is verified up front."""
    import onnxruntime as ort

    from src.compressors.frappe.harness.hybrid_runtime import CudaOnnxDecoder

    manifest = packaged["manifest"]
    requested = {}

    class FakeInput:
        def __init__(self, shape, name):
            self.shape = shape
            self.name = name
            self.type = "tensor(uint8)"

    class FakeSession:
        # path and options are how ORT builds sessions; irrelevant to this stub
        def __init__(self, path=None, options=None, providers=None):  # noqa: ARG002
            requested["providers"] = providers
            self.inputs = [
                FakeInput(shape, name)
                for shape, name in zip(manifest["plane_shapes"], manifest["plane_names"])
            ]

        def get_inputs(self):
            return self.inputs

        def run(self, output_names=None, feed=None):  # noqa: ARG002
            return [np.zeros(manifest["image_shape_nchw"], dtype=np.uint8)]

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    decoder = CudaOnnxDecoder(packaged["output"], manifest)
    assert requested["providers"] == ["CUDAExecutionProvider"]
    planes = [np.zeros(tuple(shape), dtype=np.uint8) for shape in manifest["plane_shapes"]]
    with pytest.raises(ValueError, match="index"):
        decoder.reconstruct(planes[::-1])
    decoder.reconstruct(planes)


def test_corrupted_manifest_hash_refuses_to_run(packaged):
    """A manifest whose hashes no longer match the artifacts stops before compile."""
    from src.compressors.frappe.harness.hybrid_runtime import OpenVINOEncoder

    manifest, output = packaged["manifest"], packaged["output"]
    tampered = json.loads(json.dumps(manifest))
    first = next(iter(tampered["artifacts"]))
    tampered["artifacts"][first]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sha256"):
        OpenVINOEncoder(output, "CPU", tampered)
