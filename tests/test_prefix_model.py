"""The theory note's reproducibility checklist, as executable assertions.

Each test corresponds to one algebraic claim that the joint-prefix
reformulation relies on.  They are checked bit-exactly (or to float tolerance
where an operation is reassociated) rather than asserted in prose.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.compressors.frappe.model import MergedAutoencoder
from src.compressors.frappe.prefix import (
    JointPrefixFRAPPE,
    expansion_heads,
    soft_round,
    warm_start_from_merged,
    zero_expand_first_conv,
)


def make_config(ps, **overrides):
    config = SimpleNamespace(
        ps=ps, input_channels=3, decoder_ps=8, decoder_dim=32, decoder_kernel_size=3,
        decoder_arch="C", decoder_mlp_ratio=2.0, decoder_layerscale=True,
        decoder_layerscale_init=1e-6, encoder_arch="SC8",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


SCHEDULE = [32, 32, 16, 16, 8, 8, 4, 4, 2, 2]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return JointPrefixFRAPPE(make_config(SCHEDULE)).eval()


def test_channel_blocks_tile_the_decoder_input(model):
    """Every decoder input channel belongs to exactly one latent channel."""
    covered = []
    for start, end in model.channel_slices:
        covered.extend(range(start, end))
    assert sorted(covered) == list(range(model.total_decoder_channels))
    assert len(model.channel_slices) == len(SCHEDULE)


def test_prefix_masks_are_nested(model):
    """Prefix n+1 keeps everything prefix n keeps -- the ordering is nested."""
    for n in range(1, model.n_channels):
        assert torch.all(model.masks[n] >= model.masks[n - 1])
        assert model.masks[n].sum() > model.masks[n - 1].sum()


def test_head_sum_matches_concat_convolution(model):
    """Proposition "concat-first-conv の head-sum 分解", eq. (headsum)."""
    y = torch.randn(2, model.total_decoder_channels, 8, 8)
    for n in (1, 5, model.n_channels):
        concat_first = model.first(model.mask_prefix(y, n))
        head_sum = expansion_heads(model, y, n)
        assert torch.allclose(concat_first, head_sum, atol=1e-5, rtol=0)


def test_masking_ignores_channels_beyond_the_prefix(model):
    """Latent channels past the prefix cannot influence the reconstruction."""
    y = torch.randn(2, model.total_decoder_channels, 8, 8)
    perturbed = y.clone()
    start, _ = model.channel_slices[5]
    perturbed[:, start:] += 7.0
    with torch.no_grad():
        assert torch.equal(model.decode(y, 5), model.decode(perturbed, 5))


def test_zero_column_expansion_preserves_the_function():
    """Proposition "ゼロ列拡張による関数保存", eq. (functionpreserve)."""
    torch.manual_seed(1)
    narrow = torch.nn.Conv2d(6, 5, 3, padding="same", padding_mode="reflect")
    wide = torch.nn.Conv2d(11, 5, 3, padding="same", padding_mode="reflect")
    zero_expand_first_conv(narrow, wide)
    leading = torch.randn(2, 6, 9, 9)
    trailing = torch.randn(2, 5, 9, 9)
    with torch.no_grad():
        assert torch.equal(narrow(leading), wide(torch.cat([leading, trailing], dim=1)))


def test_warm_start_reproduces_the_stagewise_codec_exactly():
    """A stagewise checkpoint lifted into the superdecoder is bit-identical."""
    torch.manual_seed(2)
    config = make_config(SCHEDULE)
    trained_channels = 6
    merged = MergedAutoencoder(make_config(SCHEDULE[:trained_channels]), trained_channels).eval()
    joint = JointPrefixFRAPPE(config).eval()
    assert warm_start_from_merged(merged, joint) == trained_channels

    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    with torch.no_grad():
        reference = merged.decode([z.round() for z in merged.encode(x)])
        codes = joint.encode(x, "hard")
        lifted = joint.decode(joint.adapt(codes), trained_channels)
    assert torch.allclose(reference, lifted, atol=1e-6, rtol=0)


def test_integer_codes_are_representable_int8(model):
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    for code in model.integer_codes(x):
        assert code.dtype == torch.int8
        assert int(code.min()) >= -127 and int(code.max()) <= 127


def test_hard_rounding_is_exact_and_passes_gradient(model):
    """Q3: integer forward, straight-through backward."""
    model.train()
    x = torch.randn(2, 3, 32, 32, requires_grad=True).clamp(-1, 1)
    codes = model.encode(x, "hard")
    for code in codes:
        assert torch.equal(code, code.round())
    codes[0].sum().backward()
    assert model.analysis[0].weight.grad is not None
    assert torch.isfinite(model.analysis[0].weight.grad).all()
    assert model.analysis[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["float", "aun", "soft", "hard"])
def test_every_continuation_stage_is_differentiable(model, mode):
    model.train()
    reconstruction, _ = model.forward_prefixes(
        torch.randn(1, 3, 32, 32).clamp(-1, 1), [model.n_channels], mode, alpha=4.0)
    reconstruction[0].square().mean().backward()
    assert torch.isfinite(model.first.weight.grad).all()


def test_soft_round_interpolates_between_identity_and_round():
    # Half-integers are genuine ties: the symmetric soft quantizer sits on the
    # midpoint there while torch.round breaks the tie, so they are excluded.
    x = torch.arange(-30, 31).float() * 0.1 + 0.037
    assert torch.allclose(soft_round(x, 0.0), x)
    assert torch.allclose(soft_round(x, 400.0), x.round(), atol=1e-3)


def test_prefix_adapter_is_identity_at_initialisation(model):
    """A freshly built model is exactly the plain shared-decoder baseline."""
    y = torch.randn(1, model.total_decoder_channels, 4, 4)
    with torch.no_grad():
        adapted = model.decode(y, 4)
        plain = model.head(model.trunk(model.first(model.mask_prefix(y, 4))))
    assert torch.equal(adapted, plain)


def test_batched_prefix_pass_matches_individual_decodes(model):
    """forward_prefixes is a pure batching optimisation, not a different graph."""
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    prefixes = [1, 4, model.n_channels]
    with torch.no_grad():
        batched, codes = model.forward_prefixes(x, prefixes, "float")
        y = model.adapt(codes)
        individual = [model.decode(y, n) for n in prefixes]
    for a, b in zip(batched, individual):
        assert torch.allclose(a, b, atol=1e-6, rtol=0)


def test_rate_proxy_depends_only_on_the_prefix(model):
    """The prefix rate must not be able to see channels it does not transmit."""
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    codes = model.encode(x, "hard")
    perturbed = [code.clone() for code in codes]
    perturbed[-1] = perturbed[-1] * 5.0  # only the finest scale changes
    for n in (2, 6):
        assert model.rate_proxy(codes, n).item() == pytest.approx(
            model.rate_proxy(perturbed, n).item())
    assert model.rate_proxy(codes, model.n_channels).item() != pytest.approx(
        model.rate_proxy(perturbed, model.n_channels).item())


def test_rate_estimate_is_symbol_weighted(model):
    """One fine channel must outweigh one coarse channel, by its symbol count."""
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    codes = codes_of(model, x)
    coarse = model.rate_bpp(codes, 1)
    finest = model.rate_bpp(codes, model.n_channels) - model.rate_bpp(codes, model.n_channels - 1)
    assert abs(finest) > abs(coarse)


def test_rate_estimate_is_non_negative_and_vanishes_on_dead_channels(model):
    """A channel that carries no information must cost no bits, never negative bits."""
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    codes = model.encode(x, "hard")
    dead = [torch.zeros_like(code) for code in codes]
    assert model.rate_bpp(dead, model.n_channels).item() == pytest.approx(0.0, abs=1e-9)
    for n in range(1, model.n_channels + 1):
        assert model.rate_bpp(codes, n).item() >= 0.0


def test_rate_estimate_increases_with_code_scale(model):
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    codes = model.encode(x, "hard")
    louder = [code * 4.0 for code in codes]
    assert (model.rate_bpp(louder, model.n_channels).item()
            > model.rate_bpp(codes, model.n_channels).item())


def codes_of(model, x):
    return model.encode(x, "hard")


def test_stagewise_widening_preserves_the_previous_prefix_exactly():
    """Algorithm A applied to the stagewise trainer: adding a channel is free.

    Widening the merged decoder with zero columns and copying the rest means the
    codec for the previous prefix is bit-identical the instant the new channel
    appears, whatever the new encoder emits.
    """
    torch.manual_seed(3)
    narrow = MergedAutoencoder(make_config(SCHEDULE[:4]), 4).eval()
    wide = MergedAutoencoder(make_config(SCHEDULE[:6]), 6).eval()
    with torch.no_grad():
        zero_expand_first_conv(narrow.decoder[0], wide.decoder[0])
        for target, source in zip(list(wide.decoder)[1:], list(narrow.decoder)[1:]):
            target.load_state_dict(source.state_dict())

    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    with torch.no_grad():
        old_latents = [z.round() for z in narrow.encode(x)]
        widened = []
        for index, template in enumerate(wide.encode(x)):
            if index >= len(old_latents):
                widened.append(torch.randn_like(template).round())
                continue
            carried = old_latents[index]
            if carried.shape[1] < template.shape[1]:
                extra = torch.randn_like(template[:, carried.shape[1]:]).round()
                carried = torch.cat([carried, extra], dim=1)
            widened.append(carried)
        assert torch.allclose(narrow.decode(old_latents), wide.decode(widened),
                              atol=1e-6, rtol=0)


def test_subset_mask_generalises_the_prefix_mask(model):
    """Prefixes are the special case of an arbitrary kept-channel set."""
    for n in (1, 4, model.n_channels):
        assert torch.equal(model.subset_mask(range(1, n + 1)), model.masks[n - 1])


def test_subset_decoding_ignores_dropped_channels(model):
    """A pruned channel cannot influence the reconstruction, whatever it carries."""
    y = torch.randn(2, model.total_decoder_channels, 8, 8)
    kept = [1, 2, 5, 9]
    perturbed = y.clone()
    for channel in range(1, model.n_channels + 1):
        if channel not in kept:
            start, end = model.channel_slices[channel - 1]
            perturbed[:, start:end] += 11.0
    with torch.no_grad():
        assert torch.equal(model.decode_subset(y, kept), model.decode_subset(perturbed, kept))


def test_subset_rejects_out_of_range_channels(model):
    with pytest.raises(ValueError):
        model.subset_mask([0])
    with pytest.raises(ValueError):
        model.subset_mask([model.n_channels + 1])
    with pytest.raises(ValueError):
        model.subset_mask([])


def test_operating_points_accept_prefixes_and_subsets(model):
    """Prefix training and pruned-subset training go through one code path."""
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    with torch.no_grad():
        batched, codes = model.forward_operating_points(x, [3, [1, 2, 3], [1, 5, 9]], "float")
        y = model.adapt(codes)
        assert torch.allclose(batched[0], batched[1], atol=1e-6, rtol=0)
        assert torch.allclose(batched[2], model.decode_subset(y, [1, 5, 9]), atol=1e-6, rtol=0)


def test_rate_estimate_matches_between_prefix_and_equivalent_subset(model):
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    codes = model.encode(x, "hard")
    assert model.rate_bpp(codes, 5).item() == pytest.approx(
        model.rate_bpp(codes, [1, 2, 3, 4, 5]).item())


def test_pruning_reproduces_the_masked_model_exactly(model):
    """Structured pruning is a size change, not a behaviour change."""
    from src.compressors.frappe.prefix import prune_channels

    kept = [1, 3, 4, 5, 7, 8, 9, 10]
    pruned = prune_channels(model, kept, make_config(SCHEDULE)).eval()
    assert pruned.n_channels == len(kept)
    assert pruned.ps == [SCHEDULE[channel - 1] for channel in kept]
    assert pruned.total_decoder_channels < model.total_decoder_channels

    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    with torch.no_grad():
        reference = model.decode_subset(model.adapt(model.encode(x, "hard")), kept)
        candidate = pruned.decode(pruned.adapt(pruned.encode(x, "hard")), pruned.n_channels)
    assert torch.allclose(reference, candidate, atol=1e-6, rtol=0)


def test_pruning_drops_empty_scale_groups(model):
    """A scale whose channels are all pruned disappears from the schedule."""
    from src.compressors.frappe.prefix import prune_channels

    finest = model.scale_groups[-1][0]
    kept = [c for c in range(1, model.n_channels + 1) if SCHEDULE[c - 1] != finest]
    pruned = prune_channels(model, kept, make_config(SCHEDULE))
    assert finest not in pruned.ps
    assert len(pruned.scale_groups) == len(model.scale_groups) - 1


def test_pruning_rejects_an_empty_or_out_of_range_selection(model):
    from src.compressors.frappe.prefix import prune_channels

    with pytest.raises(ValueError):
        prune_channels(model, [], make_config(SCHEDULE))
    with pytest.raises(ValueError):
        prune_channels(model, [model.n_channels + 1], make_config(SCHEDULE))


def test_layernorm_is_unchanged_by_the_positive_index_rewrite():
    """The ONNX-friendly movedim rewrite must not move a single bit."""
    from src.compressors.frappe.model import LayerNormND

    torch.manual_seed(7)
    norm = LayerNormND(5)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.3, 0.3)
    for shape in [(2, 5, 7), (2, 5, 7, 6), (2, 5, 4, 6, 3)]:
        x = torch.randn(*shape)
        reference = torch.nn.functional.layer_norm(
            x.movedim(1, -1), (5,), norm.weight, norm.bias, norm.eps).movedim(-1, 1)
        assert torch.equal(norm(x), reference)


def test_encode_stays_in_float32_under_autocast(model):
    """QAT must round the values the deployment path rounds, not bf16 ones."""
    if not torch.cuda.is_available():
        pytest.skip("autocast pinning is only observable on CUDA")
    cuda_model = JointPrefixFRAPPE(make_config(SCHEDULE)).cuda().eval()
    x = torch.randn(1, 3, 32, 32, device="cuda").clamp(-1, 1)
    with torch.no_grad():
        reference = cuda_model.encode(x, "hard")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            under_autocast = cuda_model.encode(x, "hard")
    for a, b in zip(reference, under_autocast):
        assert b.dtype == torch.float32
        assert torch.equal(a, b)


def test_average_ranks_averages_ties():
    """A double argsort would rank ties by index and inflate any correlation."""
    from tools.analyze_rate_breakdown import average_ranks

    ranks = average_ranks(np.array([3.0, 1.0, 3.0, 2.0]))
    assert list(ranks) == [2.5, 0.0, 2.5, 1.0]
    constant = average_ranks(np.array([5.0, 5.0, 5.0]))
    assert list(constant) == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("encoder_ps,decoder_ps", [(2, 8), (4, 8), (8, 8), (16, 8), (32, 8)])
def test_adapt_to_decoder_matches_the_einops_formulation(encoder_ps, decoder_ps):
    """The ONNX-friendly rewrite of adapt_to_decoder must be bit-identical.

    einops resolves the spatial extent from the tensor, which bakes the traced
    image size into the exported graph; pixel_unshuffle and repeat_interleave do
    not. They are only an acceptable substitution if they agree exactly.
    """
    from einops import rearrange
    from torch.nn import functional

    from src.compressors.frappe.ops import adapt_to_decoder

    torch.manual_seed(11)
    grid = 6
    z = torch.randn(2, 3, grid * max(1, decoder_ps // encoder_ps),
                    grid * max(1, decoder_ps // encoder_ps))
    if encoder_ps < decoder_ps:
        f = decoder_ps // encoder_ps
        expected = rearrange(z, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=f, p2=f)
    elif encoder_ps > decoder_ps:
        expected = functional.interpolate(z, scale_factor=encoder_ps // decoder_ps,
                                          mode="nearest")
    else:
        expected = z
    assert torch.equal(adapt_to_decoder(z, encoder_ps, decoder_ps), expected)
