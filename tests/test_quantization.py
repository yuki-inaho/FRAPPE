"""The QAT boundary: gradients through the codec's rounding, a frozen decoder,
and NNCF state that survives a round trip.

These tests run in the deployment environment (uv, group ``deploy``) because
NNCF lives only there. The module under test imports NNCF lazily, so the
research environment stays importable without it.
"""

from types import SimpleNamespace

import pytest
import torch

from src.compressors.frappe.harness.deployment import EncoderGraph
from src.compressors.frappe.prefix import JointPrefixFRAPPE

SCHEDULE = [32, 16]


def make_config(**overrides):
    config = SimpleNamespace(
        ps=SCHEDULE, input_channels=3, decoder_ps=8, decoder_dim=32, decoder_kernel_size=3,
        decoder_arch="C", decoder_mlp_ratio=2.0, decoder_layerscale=True,
        decoder_layerscale_init=1e-6, encoder_arch="SC8",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def model():
    torch.manual_seed(0)
    return JointPrefixFRAPPE(make_config()).eval()


def first_conv(model):
    return model.analysis[0]


def decoder_parameters(model):
    """Every parameter that belongs to the frozen side (decoder + adapt masks)."""
    return [param for name, param in model.named_parameters()
            if not name.startswith("analysis") and not name.startswith("companders")]


def test_qat_encoder_backpropagates_through_codec_rounding(model):
    """One hard-STE step puts a finite, non-zero gradient on the analysis conv."""
    from src.compressors.frappe.harness.quantization import TrainableEncoder

    encoder = TrainableEncoder(model)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    codes = encoder(x)
    assert all(code.dtype == torch.float32 for code in codes)
    reconstruction = model.decode(model.adapt(codes), model.n_channels)
    loss = torch.nn.functional.mse_loss(reconstruction, x).clamp_min(1e-12).log10()
    loss.backward()
    weight = first_conv(model).weight
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0


def test_decoder_is_frozen_during_qat_step(model):
    """The decoder receives no gradient and keeps its weights across an update."""
    from src.compressors.frappe.harness.quantization import TrainableEncoder, freeze_decoder

    freeze_decoder(model)
    encoder = TrainableEncoder(model)
    before = [param.detach().clone() for param in decoder_parameters(model)]
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    reconstruction = model.decode(model.adapt(encoder(x)), model.n_channels)
    loss = torch.nn.functional.mse_loss(reconstruction, x).clamp_min(1e-12).log10()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1e-3)
    loss.backward()
    assert all(param.grad is None for param in decoder_parameters(model))
    optimizer.step()
    for kept, now in zip(before, decoder_parameters(model)):
        assert torch.equal(kept, now.detach())


def test_deployment_view_matches_encoder_graph_planes(model):
    """Without fake quantizers the deployable view is the shipped graph, bit for bit."""
    from src.compressors.frappe.harness.quantization import DeployableEncoder

    image = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    cases = [(True, image),
             (False, image.to(torch.float32) / 127.5 - 1.0)]
    for uint8_io, feed in cases:
        deploy = DeployableEncoder(model, uint8_io=uint8_io)
        reference = EncoderGraph(model, uint8_io=uint8_io)
        produced, expected = deploy(feed), reference(feed)
        assert len(produced) == len(expected) == len(SCHEDULE)
        for plane, want in zip(produced, expected):
            assert plane.dtype == want.dtype
            assert plane.shape == want.shape
            assert torch.equal(plane, want)


def test_nncf_inserts_weight_quantizers_on_every_analysis_conv(model):
    """Calibration wraps each analysis conv's weight port, and never the decoder."""
    pytest.importorskip("nncf.torch", reason="NNCF ships in the uv deploy group only")
    import nncf

    from src.compressors.frappe.harness.quantization import TrainableEncoder

    torch.manual_seed(1)
    items = [torch.rand(1, 3, 32, 32) * 2 - 1 for _ in range(2)]
    encoder = TrainableEncoder(model)
    quantized = nncf.quantize(encoder, nncf.Dataset(items), subset_size=2,
                              target_device=nncf.TargetDevice.NPU)
    # NNCF 3.3's torch backend attaches quantizers as hooks under __nncf_hooks
    # rather than replacing modules; a conv's weight port is input port 1.
    quantizer_names = [name for name, module in quantized.named_modules()
                       if type(module).__name__ == "SymmetricQuantizer"]
    conv_weight_hooks = [name for name in quantizer_names
                         if name.startswith("__nncf_hooks.pre_hooks")
                         and "/conv2d/" in name and name.endswith("__1.0")]
    assert len(conv_weight_hooks) == len(SCHEDULE)
    assert all(type(m).__name__ != "SymmetricQuantizer"
               for name, m in model.named_modules() if "trunk" in name or "head" in name)


def test_nncf_state_survives_save_and_restore(model):
    """Saved QAT state reproduces the identical integer planes when restored."""
    pytest.importorskip("nncf.torch", reason="NNCF ships in the uv deploy group only")
    import nncf

    from src.compressors.frappe.harness.quantization import (
        DeployableEncoder,
        TrainableEncoder,
        restore_qat_state,
        save_qat_state,
    )

    torch.manual_seed(1)
    items = [torch.rand(1, 3, 32, 32) * 2 - 1 for _ in range(2)]
    quantized = nncf.quantize(TrainableEncoder(model), nncf.Dataset(items), subset_size=2,
                              target_device=nncf.TargetDevice.NPU)
    state = save_qat_state(quantized)
    restored = restore_qat_state(TrainableEncoder(model), state)
    image = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    before = DeployableEncoder(quantized, uint8_io=True)(image)
    after = DeployableEncoder(restored, uint8_io=True)(image)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
