"""The hybrid runtime: OpenVINO encoder, explicit JPEG-LS, CUDA ONNX Runtime decoder.

Every device, provider and entropy backend here is something the caller named.
OpenVINO compiles only the requested device and refuses meta-devices and
absent hardware outright; ONNX Runtime is asked for exactly
``CUDAExecutionProvider`` and never quietly runs the decoder on the CPU; and
the JPEG-LS backend is only what the caller names, with the native library
raising rather than degrading. Plane geometry comes from the package manifest
and is checked before any inference, with planes bound by output index --
the manifest's ``plane_order`` -- rather than by name.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .quantization import sha256_of


def verify_manifest(artifact_dir: Path, manifest: dict) -> None:
    """Every manifest artifact must exist on disk and match its recorded sha256."""
    for name, info in manifest["artifacts"].items():
        path = Path(artifact_dir) / name
        if not path.is_file():
            raise RuntimeError(f"package artifact {name} is missing from {artifact_dir}")
        if sha256_of(path) != info["sha256"]:
            raise RuntimeError(f"package artifact {name} does not match its manifest sha256")


def load_manifest(artifact_dir: str | Path) -> dict:
    """The package's manifest, with every artifact hash verified against disk."""
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    verify_manifest(artifact_dir, manifest)
    return manifest


_META_DEVICES = {"AUTO", "HETERO", "MULTI", "BATCH"}


def _preload_cuda_libraries() -> None:
    """Load the CUDA 13 userland ORT links against, before it looks for them.

    ``onnxruntime-gpu`` 1.29 links CUDA 13, while torch's cu128 wheels install
    CUDA 12; the two userlands are separate pip packages. LD_LIBRARY_PATH must
    be set before the process starts, which a caller cannot do mid-process, so
    the libraries are loaded by absolute path with ``RTLD_GLOBAL``: once they
    are in the address space, the provider library resolves its dependencies
    from there instead of failing into a silent CPU fallback.
    """
    import contextlib
    import ctypes
    import sysconfig

    site_packages = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    sonames = ("libcublasLt.so.13", "libcublas.so.13", "libcudart.so.13", "libcudnn.so.9")
    for soname in sonames:
        for library in sorted(site_packages.rglob(soname)):
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


class OpenVINOEncoder:
    """The packaged INT8 encoder compiled on the one requested OpenVINO device."""

    def __init__(self, artifact_dir: str | Path, device: str, manifest: dict) -> None:
        import openvino as ov

        self.manifest = manifest
        verify_manifest(Path(artifact_dir), manifest)
        requested = device.upper()
        if requested in _META_DEVICES:
            raise ValueError(f"{device} is a meta-device, not hardware; name a real device")
        core = ov.Core()
        present = [candidate.upper().split(".")[0] for candidate in core.available_devices]
        if requested not in present:
            raise RuntimeError(
                f"{device} is not among OpenVINO devices {core.available_devices}; "
                "refusing to fall back"
            )
        self.requested_device = requested
        ir = core.read_model(
            str(
                Path(artifact_dir) / "encoder_int8_"
                f"{manifest['image_shape_nchw'][3]}x"
                f"{manifest['image_shape_nchw'][2]}.xml"
            )
        )
        self.ir_shape = list(ir.input(0).shape)
        if self.ir_shape != list(manifest["image_shape_nchw"]):
            raise RuntimeError(
                f"the encoder IR's input {self.ir_shape} disagrees with the manifest's "
                f"{manifest['image_shape_nchw']}"
            )
        self.compiled = core.compile_model(ir, device)
        self.execution_devices = self.compiled.get_property("EXECUTION_DEVICES")
        self.output_count = len(manifest["plane_order"])

    def encode(self, image: np.ndarray) -> list[np.ndarray]:
        """uint8 NCHW image in, uint8 planes out, bound by output index."""
        result = self.compiled({0: np.ascontiguousarray(image)})
        if len(result) != self.output_count:
            raise RuntimeError(
                f"the encoder produced {len(result)} outputs; the manifest binds "
                f"{self.output_count}"
            )
        shapes = self.manifest["plane_shapes"]
        planes = []
        for index in range(self.output_count):
            plane = result[index]
            if plane.shape != tuple(shapes[index]):
                raise RuntimeError(
                    f"plane at output index {index} has shape {plane.shape}, "
                    f"the manifest declares {tuple(shapes[index])}"
                )
            if plane.dtype != np.uint8:
                raise RuntimeError(
                    f"plane at output index {index} is {plane.dtype}, expected uint8"
                )
            planes.append(np.ascontiguousarray(plane))
        return planes


class CudaOnnxDecoder:
    """The packaged decoder under exactly one ONNX Runtime provider: CUDA."""

    def __init__(self, artifact_dir: str | Path, manifest: dict) -> None:
        import onnxruntime as ort

        self.manifest = manifest
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available (providers: {available}); "
                "the decoder must not silently run on CPU"
            )
        _preload_cuda_libraries()
        self.session = ort.InferenceSession(
            str(Path(artifact_dir) / "decoder.onnx"), providers=["CUDAExecutionProvider"]
        )
        # A provider that fails to load degrades to CPU with only a warning;
        # ask the session what it actually runs, and refuse anything else.
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"the decoder session fell back to {self.session.get_providers()[0]}; "
                "CUDA is the only accepted provider"
            )
        self._check_inputs(self.session.get_inputs())

    def _check_inputs(self, entries) -> None:
        shapes, names = self.manifest["plane_shapes"], self.manifest["plane_names"]
        if len(entries) != len(names):
            raise RuntimeError(
                f"the decoder takes {len(entries)} inputs; the manifest binds {len(names)}"
            )
        for index, entry in enumerate(entries):
            if entry.name != names[index]:
                raise RuntimeError(
                    f"decoder input index {index} is {entry.name!r}; the manifest binds "
                    f"{names[index]} there"
                )
            if entry.type != "tensor(uint8)":
                raise RuntimeError(
                    f"decoder input index {index} has dtype {entry.type}, expected uint8"
                )
            # The decoder keeps H and W symbolic (it serves every resolution
            # the frozen encoder can produce), so only the static dims are
            # declared; the actual arrays are checked against the manifest in
            # reconstruct().
            for position, actual in enumerate(entry.shape):
                declared = int(shapes[index][position])
                if isinstance(actual, int) and int(actual) != declared:
                    raise RuntimeError(
                        f"decoder input index {index} dim {position} is {actual}, "
                        f"the manifest declares {declared}"
                    )

    def reconstruct(self, planes: list[np.ndarray]) -> np.ndarray:
        """Planes in the manifest's index order in, uint8 reconstruction out."""
        shapes = self.manifest["plane_shapes"]
        if len(planes) != len(shapes):
            raise ValueError(f"expected {len(shapes)} planes in manifest order, got {len(planes)}")
        for index, plane in enumerate(planes):
            if plane.shape != tuple(int(dim) for dim in shapes[index]):
                raise ValueError(
                    f"plane at input index {index} has shape {plane.shape}, "
                    f"the manifest declares {tuple(int(dim) for dim in shapes[index])}"
                )
            if plane.dtype != np.uint8:
                raise ValueError(
                    f"plane at input index {index} has dtype {plane.dtype}, expected uint8"
                )
        feed = {entry.name: plane for entry, plane in zip(self.session.get_inputs(), planes)}
        return self.session.run(None, feed)[0]


def jpegls_roundtrip(
    planes: list[np.ndarray], backend: str
) -> tuple[list[bytes], list[np.ndarray], bool]:
    """Encode every plane through the named backend and decode it back exactly.

    The decode always goes through the portable Pillow path, so the bytes a
    backend writes are proven to be the codestream ``pillow_jpls`` reads. A
    plane that does not survive the round trip is an error, not a count.
    """
    import pillow_jpls  # noqa: F401 -- registers the JPEG-LS plugin with PIL
    import torch
    from PIL import Image

    from .bitstream import encode_plane

    payloads = [
        encode_plane(torch.from_numpy(plane.reshape(plane.shape[-2:])), backend=backend)
        for plane in planes
    ]
    decoded = []
    for payload, plane in zip(payloads, planes):
        with Image.open(io.BytesIO(payload)) as handle:
            restored = np.ascontiguousarray(np.asarray(handle.convert("L")))
        flat = plane.reshape(plane.shape[-2:])
        if restored.shape != flat.shape or not np.array_equal(restored, flat):
            raise RuntimeError(
                f"the JPEG-LS roundtrip is not exact for a plane of shape {flat.shape}"
            )
        decoded.append(restored)
    return payloads, decoded, True


def run_roundtrip(
    artifact_dir: str | Path,
    dataset_root: str | Path,
    split: str,
    first_index: int,
    images: int,
    *,
    encoder_device: str,
    entropy_backend: str,
) -> dict:
    """Encode → JPEG-LS → decode → reconstruct, with an explicit device/backend each."""
    import torch
    import torch.nn.functional as F

    from .data import AnonymousImageFolder, to_signed

    manifest = load_manifest(artifact_dir)
    encoder = OpenVINOEncoder(artifact_dir, encoder_device, manifest)
    decoder = CudaOnnxDecoder(artifact_dir, manifest)
    folder = AnonymousImageFolder(dataset_root, split)
    indices = list(range(first_index, min(first_index + images, len(folder))))
    if len(indices) < images:
        raise RuntimeError(
            f"requested {images} images from index {first_index} but "
            f"{split} holds {len(folder)} from there"
        )

    height, width = manifest["image_shape_nchw"][2], manifest["image_shape_nchw"][3]
    pixels_total = bytes_total = bytes_with_prefix = 0
    mse_total = 0.0
    length_prefix_bytes = 4 * len(manifest["plane_order"])
    for index in indices:
        pixels = folder.pixels(index)
        planes = encoder.encode(pixels.numpy())
        payloads, restored, exact = jpegls_roundtrip(planes, entropy_backend)
        payload = sum(len(chunk) for chunk in payloads)
        # the decoder binds the planes in their batched (1, rows, cols) form
        reconstruction = (
            decoder.reconstruct([plane[None] for plane in restored]).astype(np.float32) / 255.0
        )
        target = to_signed(pixels).float() / 2 + 0.5
        mse = F.mse_loss(target, torch.from_numpy(reconstruction)).item()
        mse_total += mse * height * width
        pixels_total += height * width
        bytes_total += payload
        bytes_with_prefix += payload + length_prefix_bytes
    psnr = -10 * np.log10(mse_total / pixels_total)
    # The report travels: the package is identified by its manifest hash and
    # the record says what executed, never where the data lived.
    return {
        "manifest_sha256": sha256_of(Path(artifact_dir) / "manifest.json"),
        "split": split,
        "images": len(indices),
        "image_indices": indices,
        "size": [width, height],
        "prefix": manifest["prefix"],
        "plane_order": manifest["plane_order"],
        "encoder": {"requested": encoder_device, "execution_devices": encoder.execution_devices},
        "decoder": {"provider": "CUDAExecutionProvider"},
        "entropy_backend": entropy_backend,
        "jpegls_roundtrip_exact": exact,
        "psnr_db": psnr,
        "bytes_total": int(bytes_total),
        "bpp": bytes_total * 8 / pixels_total,
        "bytes_total_with_length_prefix": int(bytes_with_prefix),
        "bpp_with_length_prefix": bytes_with_prefix * 8 / pixels_total,
    }
