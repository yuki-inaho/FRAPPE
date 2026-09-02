"""Joint-prefix FRAPPE: full-width superdecoder with block-prefix masking.

This is the reformulation of the theory note "FRAPPE の逐次学習を並列化・標準化
するための再定式化".  The inference graph is unchanged from
:class:`compressors.frappe.model.MergedAutoencoder` -- per-scale non-overlapping
linear analysis, per-channel softsign companding, int8 codes, spatial adaption,
one heavy synthesis transform.  What changes is that *every* channel exists from
the first optimizer step and every prefix is a first-class output:

``decode(y, n)`` reconstructs from prefix ``1:n`` by zeroing the decoder-channel
blocks of latent channels ``> n`` (eq. "prefixorig"), so a single set of weights
serves all operating points.  Because the first decoder convolution is linear in
its input channels, masking then convolving equals summing per-channel expansion
heads (proposition "concat-first-conv の head-sum 分解"), and widening the first
convolution with zero columns preserves the previous function exactly
(proposition "ゼロ列拡張による関数保存").  Both identities are exercised
bit-exactly in ``tests/test_prefix_model.py``.

The quantizer implements the Q0--Q4 continuation of the note's "QAT
continuation" box as a single module whose forward is selected per step.
"""

from __future__ import annotations

import math

import torch

from .model import make_decoder_blocks
from .ops import adapt_to_decoder, decoder_channels_per_encoder, get_scale_groups

#: Continuation stages, coarse to fine, in the order the schedule walks them.
QUANTIZATION_MODES = ("float", "aun", "soft", "hard")


def soft_round(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Differentiable approximation of ``round`` that sharpens with ``alpha``.

    ``alpha -> 0`` is the identity and ``alpha -> inf`` is hard rounding, so a
    single annealed parameter walks stage Q2 of the continuation.
    """
    if alpha < 1e-3:
        return x
    m = torch.floor(x) + 0.5
    r = x - m
    return m + torch.tanh(alpha * r) / (2.0 * math.tanh(alpha / 2.0))


class SoftsignCompander(torch.nn.Module):
    """Per-channel companding and quantization, eq. (sc8general).

    ``v = gamma * phi_sigma(u) + beta`` with ``phi_sigma(u) = r u / (sigma + |u|)``
    followed by a stage-dependent quantizer.  Unlike the frozen ``SC8``
    sequential, the training noise is applied to ``v`` rather than to
    ``phi_sigma(u)``: additive uniform noise is the standard relaxation *of the
    rounding operation*, and rounding happens after the affine, so the noise
    must live in the same space to be a consistent relaxation.
    """

    def __init__(self, num_channels: int, bits: int = 8) -> None:
        super().__init__()
        self.bits = bits
        self.r = 2 ** (bits - 1) - 1
        self._sigma = torch.nn.Parameter(torch.full((num_channels,), float(self.r - 1)))
        self.gamma = torch.nn.Parameter(torch.ones(num_channels))
        self.beta = torch.nn.Parameter(torch.zeros(num_channels))

    def _shape(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(1, -1, 1, 1)

    def companded(self, u: torch.Tensor) -> torch.Tensor:
        sigma = self._shape(self._sigma.abs() + 1e-6)
        return self._shape(self.gamma) * (self.r * u / (sigma + u.abs())) + self._shape(self.beta)

    def forward(self, u: torch.Tensor, mode: str = "hard", alpha: float = 8.0) -> torch.Tensor:
        v = self.companded(u)
        self._last_v = v
        if mode == "float" or (mode in {"aun", "soft"} and not self.training):
            q = v if mode == "float" else torch.round(v)
        elif mode == "aun":
            q = v + torch.empty_like(v).uniform_(-0.5, 0.5)
        elif mode == "soft":
            noised = soft_round(v, alpha) + torch.empty_like(v).uniform_(-0.5, 0.5)
            q = soft_round(noised, alpha)
        elif mode == "hard":
            q = v + (torch.round(v) - v).detach()
        else:
            raise ValueError(f"unknown quantization mode {mode!r}")
        # Clip forward, identity backward: the saturation penalty, not a dead
        # gradient, is what pulls codes back into the representable range.
        return q + (q.clamp(-self.r, self.r) - q).detach()

    def saturation_penalty(self, margin: float = 2.0) -> torch.Tensor:
        """``E[relu(|v| - 127 + m)^2]`` of eq. (fullloss)."""
        v = getattr(self, "_last_v", None)
        if v is None:
            return torch.zeros((), device=self.gamma.device)
        return torch.relu(v.abs() - (self.r - margin)).pow(2).mean()


class JointPrefixFRAPPE(torch.nn.Module):
    """Full-width superdecoder trained on every prefix simultaneously."""

    def __init__(self, config) -> None:
        super().__init__()
        self.ps = [int(p) for p in config.ps]
        self.n_channels = len(self.ps)
        self.input_channels = int(config.input_channels)
        self.decoder_ps = int(config.decoder_ps)
        self.scale_groups = get_scale_groups(self.ps, self.n_channels)
        if not all(p % self.decoder_ps == 0 or self.decoder_ps % p == 0 for p in self.ps):
            raise ValueError("every encoder patch size must divide or be divided by decoder_ps")

        self.analysis = torch.nn.ModuleList()
        self.companders = torch.nn.ModuleList()
        slices: list[tuple[int, int]] = []
        offset = 0
        for ps, start, end in self.scale_groups:
            width = end - start
            self.analysis.append(torch.nn.Conv2d(self.input_channels, width, ps, stride=ps))
            self.companders.append(SoftsignCompander(width))
            per_channel = decoder_channels_per_encoder(ps, self.decoder_ps)
            for index in range(width):
                slices.append((offset + index * per_channel, offset + (index + 1) * per_channel))
            offset += width * per_channel
        self.total_decoder_channels = offset
        self.channel_slices = slices

        # masks[n - 1] keeps the decoder-channel blocks of latent channels 1..n.
        masks = torch.zeros(self.n_channels, self.total_decoder_channels)
        for n in range(1, self.n_channels + 1):
            for start, end in slices[:n]:
                masks[n - 1, start:end] = 1.0
        self.register_buffer("masks", masks, persistent=False)

        dim = int(config.decoder_dim)
        kernel = int(config.decoder_kernel_size)
        self.first = torch.nn.Conv2d(self.total_decoder_channels, dim, kernel_size=kernel,
                                     stride=1, padding="same", padding_mode="reflect")
        self.trunk = torch.nn.Sequential(*make_decoder_blocks(
            config.decoder_arch, dim, kernel_size=kernel,
            mlp_ratio=config.decoder_mlp_ratio,
            layerscale=config.decoder_layerscale,
            layerscale_init=config.decoder_layerscale_init))
        dps = self.decoder_ps
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(dim, self.input_channels * dps * dps, kernel_size=1),
            torch.nn.ConvTranspose2d(self.input_channels * dps * dps, self.input_channels,
                                     kernel_size=dps, stride=dps),
            torch.nn.Hardtanh(),
        )
        # Cheapest prefix adapter of the note's ordering: a prefix-specific
        # scale and bias on the first decoder layer.  Identity at init, so a
        # freshly built model is exactly the shared-decoder baseline.
        self.prefix_scale = torch.nn.Parameter(torch.ones(self.n_channels, dim))
        self.prefix_bias = torch.nn.Parameter(torch.zeros(self.n_channels, dim))

    # ---- analysis ------------------------------------------------------

    def encode(self, x: torch.Tensor, mode: str = "hard", alpha: float = 8.0) -> list[torch.Tensor]:
        """Per-scale code tensors.  Every channel is computed from ``x`` directly.

        The analysis path is pinned to float32 even when the caller runs under
        autocast.  Quantization-aware training is only meaningful if the integer
        it rounds to is the integer the deployment path produces, and a bf16
        companded value lands on a different integer for a noticeable fraction of
        symbols -- the codes would then be trained against a quantizer that is
        never shipped.  The decoder is free to run in bf16; it consumes integers.
        """
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            return [compander(analysis(x), mode, alpha)
                    for analysis, compander in zip(self.analysis, self.companders)]

    def adapt(self, codes: list[torch.Tensor]) -> torch.Tensor:
        adapted = [adapt_to_decoder(code.to(torch.float), ps, self.decoder_ps)
                   for code, (ps, _, _) in zip(codes, self.scale_groups)]
        return torch.cat(adapted, dim=1)

    def mask_prefix(self, y: torch.Tensor, n: int) -> torch.Tensor:
        if not 1 <= n <= self.n_channels:
            raise ValueError(f"prefix length {n} outside 1..{self.n_channels}")
        return y * self.masks[n - 1].view(1, -1, 1, 1)

    def subset_mask(self, channels) -> torch.Tensor:
        """Decoder-channel mask for an arbitrary set of latent channels.

        Prefixes are the special case ``{1, ..., n}``.  Structured pruning needs
        the general form: the rate-optimal set of latent channels at a given
        bitrate is not required to be a prefix, and one bit per channel in the
        stream header is enough to signal which set was used.
        """
        kept = sorted({int(c) for c in channels})
        if not kept or kept[0] < 1 or kept[-1] > self.n_channels:
            raise ValueError(f"channels must be a non-empty subset of 1..{self.n_channels}")
        mask = torch.zeros(self.total_decoder_channels, device=self.masks.device,
                           dtype=self.masks.dtype)
        for channel in kept:
            start, end = self.channel_slices[channel - 1]
            mask[start:end] = 1.0
        return mask

    def mask_subset(self, y: torch.Tensor, channels) -> torch.Tensor:
        return y * self.subset_mask(channels).view(1, -1, 1, 1)

    def decode_subset(self, y: torch.Tensor, channels) -> torch.Tensor:
        """Reconstruct from an arbitrary channel set.

        The prefix adapter is indexed by how many channels are kept, so a subset
        of size ``n`` reuses the adapter trained for prefix ``n``.  That keeps
        the pruned codec inside the weights that already exist instead of
        requiring a new adapter per subset.
        """
        kept = sorted({int(c) for c in channels})
        return self.decode(self.mask_subset(y, kept), len(kept), masked=True)

    # ---- synthesis -----------------------------------------------------

    def decode(self, y: torch.Tensor, n: int, masked: bool = False) -> torch.Tensor:
        """Reconstruct from prefix ``1:n``.  ``masked`` skips a redundant re-mask."""
        h = self.first(y if masked else self.mask_prefix(y, n))
        h = h * self.prefix_scale[n - 1].view(1, -1, 1, 1) + self.prefix_bias[n - 1].view(1, -1, 1, 1)
        return self.head(self.trunk(h))

    @staticmethod
    def _operating_point(point) -> list[int]:
        """Normalise an operating point to the sorted list of channels it keeps."""
        if isinstance(point, int):
            return list(range(1, point + 1))
        return sorted({int(channel) for channel in point})

    def forward_operating_points(self, x: torch.Tensor, points, mode: str = "hard",
                                 alpha: float = 8.0):
        """One encoder pass, one batched decoder pass over several operating points.

        Each point is either an ``int`` prefix length or an explicit set of kept
        channels, so prefix training and pruned-subset training share one path.
        Stacking the masked latents on the batch dimension keeps a single trunk
        call, so cost grows with the number of sampled points and not with the
        channel count.
        """
        codes = self.encode(x, mode, alpha)
        y = self.adapt(codes)
        kept_sets = [self._operating_point(point) for point in points]
        stacked = torch.cat([y * self.subset_mask(kept).view(1, -1, 1, 1)
                             for kept in kept_sets], dim=0)
        h = self.first(stacked)
        index = [len(kept) - 1 for kept in kept_sets]
        scale = torch.cat([self.prefix_scale[i].expand(x.shape[0], -1) for i in index])
        bias = torch.cat([self.prefix_bias[i].expand(x.shape[0], -1) for i in index])
        h = h * scale[:, :, None, None] + bias[:, :, None, None]
        out = self.head(self.trunk(h))
        return list(out.chunk(len(points), dim=0)), codes

    def forward_prefixes(self, x: torch.Tensor, prefixes, mode: str = "hard",
                         alpha: float = 8.0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prefix-only special case of :meth:`forward_operating_points`."""
        return self.forward_operating_points(x, [int(n) for n in prefixes], mode, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes = self.encode(x, "hard")
        return self.decode(self.adapt(codes), self.n_channels)

    # ---- integer codes and rate ---------------------------------------

    @torch.no_grad()
    def integer_codes(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Deployment path: true int8 codes, exactly what the entropy coder sees."""
        return [compander.companded(analysis(x)).round().clamp(-127, 127).to(torch.int8)
                for analysis, compander in zip(self.analysis, self.companders)]

    #: ``log2(2 * pi * e)``, the Gaussian entropy constant in bits.
    _GAUSSIAN_ENTROPY_CONSTANT = math.log2(2.0 * math.pi * math.e)

    def rate_bpp(self, codes: list[torch.Tensor], n) -> torch.Tensor:
        """Differentiable bits-per-pixel estimate for prefix ``1:n``.

        Each channel's codes are modelled as zero-mean Gaussian at a unit
        quantization step, whose discrete entropy is well approximated by
        ``0.5 * log2(1 + 2 pi e var)``.  Two properties matter here.  It is
        non-negative and tends to zero as a channel collapses onto a single code,
        unlike a bare ``log2 Std`` proxy, which rewards shrinking a channel's
        scale without bound and will happily drive the whole latent to zero.  And
        because each channel is weighted by its symbols per pixel ``1 / p_i^2``,
        the sum is on the same scale as the measured JPEG-LS bitrate, so the
        Lagrange multiplier that multiplies it has a meaning and can be steered
        by comparing the estimate with a real bitstream measurement.
        """
        kept = set(self._operating_point(n))
        total = torch.zeros((), device=codes[0].device)
        for code, (ps, start, end) in zip(codes, self.scale_groups):
            local = [c - start for c in range(start, end) if c + 1 in kept]
            if not local:
                continue
            per_channel = code[:, local].transpose(0, 1).reshape(len(local), -1)
            variance = per_channel.var(dim=-1, unbiased=False)
            bits = 0.5 * torch.log2(1.0 + variance * (2.0 * math.pi * math.e))
            total = total + bits.sum() / (ps * ps)
        return total

    #: Retained name for the rate term used by the training script.
    rate_proxy = rate_bpp

    def saturation_penalty(self, margin: float = 2.0) -> torch.Tensor:
        return sum(c.saturation_penalty(margin) for c in self.companders) / len(self.companders)


# ---- Algorithm A helpers ----------------------------------------------


def expansion_heads(model: JointPrefixFRAPPE, y: torch.Tensor, n: int) -> torch.Tensor:
    """``b + sum_{i<=n} Conv(K_i, y_i)`` of eq. (headsum).

    Provided so the algebraic identity in the note can be checked against the
    concat-then-convolve implementation rather than only asserted.
    """
    weight = model.first.weight
    padding = weight.shape[-1] // 2
    padded = torch.nn.functional.pad(y, (padding,) * 4, mode="reflect")
    accumulated = model.first.bias.view(1, -1, 1, 1).expand(
        y.shape[0], -1, y.shape[2], y.shape[3]).clone()
    for start, end in model.channel_slices[:n]:
        accumulated = accumulated + torch.nn.functional.conv2d(
            padded[:, start:end], weight[:, start:end])
    return accumulated


@torch.no_grad()
def zero_expand_first_conv(source: torch.nn.Conv2d, target: torch.nn.Conv2d) -> None:
    """Function-preserving widening, eq. (zeroexpand).

    Copies ``source`` into the leading input channels of ``target`` and zeroes
    the rest, so the widened decoder reproduces the narrower one exactly on any
    input whose leading blocks agree -- independent of what the new encoder,
    compander or rounding produce.
    """
    old = source.weight.shape[1]
    if target.weight.shape[1] < old:
        raise ValueError("target convolution is narrower than the source")
    target.weight.zero_()
    target.weight[:, :old].copy_(source.weight)
    if source.bias is not None and target.bias is not None:
        target.bias.copy_(source.bias)


@torch.no_grad()
def warm_start_from_merged(merged, joint: JointPrefixFRAPPE) -> int:
    """Algorithm A warm start: lift a stagewise checkpoint into the superdecoder.

    ``merged`` is a :class:`compressors.frappe.model.MergedAutoencoder` trained
    on the first ``n`` channels.  Its adapted-latent layout is the leading block
    of the joint model's, so copying the encoders, zero-expanding the first
    convolution and copying the remaining decoder layers reproduces the stagewise
    codec *exactly* at prefix ``n`` before any further optimization.  Returns
    ``n`` so the caller can assert on the right prefix.
    """
    trained = sum(end - start for _, start, end in merged.scale_groups)
    for index, encoder in enumerate(merged.encoders):
        width = encoder[0].weight.shape[0]
        joint.analysis[index].weight[:width].copy_(encoder[0].weight)
        joint.analysis[index].bias[:width].copy_(encoder[0].bias)
        source = dict(encoder[1].named_parameters())
        target = joint.companders[index]
        if "0._σ" in source:  # SC8 sequential: Softsign -> noise -> ChannelAffine
            target._sigma[:width].copy_(source["0._σ"])
            target.gamma[:width].copy_(source["2.γ"])
            if "2.β" in source:
                target.beta[:width].copy_(source["2.β"])
    decoder = list(merged.decoder)
    zero_expand_first_conv(decoder[0], joint.first)
    for target_block, source_block in zip(joint.trunk, decoder[1:-3]):
        target_block.load_state_dict(source_block.state_dict())
    for target_layer, source_layer in zip(joint.head, decoder[-3:]):
        target_layer.load_state_dict(source_layer.state_dict())
    joint.prefix_scale.fill_(1.0)
    joint.prefix_bias.zero_()
    return trained


# ---- Algorithm C step 1--3: calibration-based initialisation ------------


@torch.no_grad()
def klt_initialize(model: JointPrefixFRAPPE, images: torch.Tensor, verbose: bool = False) -> None:
    """Set the analysis filters to the deflated KLT of the residual patch covariance.

    This is the training-free initializer of the note's "DCT/KLT/PCA" section:
    coarse to fine, each scale takes the leading eigenvectors of the covariance
    of what the coarser scales left behind.  It starts joint training from the
    stagewise linear optimum instead of from noise, which is a strictly better
    starting point and costs one eigendecomposition per scale.
    """
    from einops import rearrange

    residual = images.clone()
    for index, (ps, start, end) in enumerate(model.scale_groups):
        width = end - start
        rows = rearrange(residual, "b c (h p1) (w p2) -> (b h w) (c p1 p2)", p1=ps, p2=ps)
        mean = rows.mean(dim=0, keepdim=True)
        centred = rows - mean
        covariance = (centred.T @ centred) / centred.shape[0]
        _, vectors = torch.linalg.eigh(covariance.double())
        basis = vectors.flip(-1)[:, :width].to(rows.dtype)          # (d, width)

        conv = model.analysis[index]
        conv.weight.copy_(basis.T.reshape(width, *conv.weight.shape[1:]))
        conv.bias.copy_(-(mean @ basis).squeeze(0))
        coefficients = centred @ basis
        approximation = coefficients @ basis.T + mean
        residual = residual - rearrange(
            approximation, "(b h w) (c p1 p2) -> b c (h p1) (w p2)",
            b=images.shape[0], h=images.shape[2] // ps, w=images.shape[3] // ps,
            c=images.shape[1], p1=ps, p2=ps)
        if verbose:
            mse = residual.pow(2).mean().item() / 4.0
            print(f"  KLT init scale ps={ps:2d} ({width}ch): residual PSNR="
                  f"{-10 * math.log10(max(mse, 1e-12)):6.2f} dB", flush=True)


@torch.no_grad()
def calibrate_companders(model: JointPrefixFRAPPE, images: torch.Tensor,
                         percentile: float = 99.9, knee: float = 2.0,
                         target_code: float = 100.0, batch: int = 8) -> None:
    """Set ``sigma`` and ``gamma`` from calibration statistics rather than by training.

    ``sigma = knee * u_P`` places the companding knee a chosen multiple of the
    ``percentile`` magnitude, and ``gamma`` then maps that magnitude to
    ``target_code``.  A large ``knee`` approaches uniform quantization with a
    saturating tail; a small one is aggressively companded.
    """
    for index, analysis in enumerate(model.analysis):
        magnitudes = []
        for offset in range(0, images.shape[0], batch):
            u = analysis(images[offset:offset + batch])
            magnitudes.append(u.transpose(0, 1).reshape(u.shape[1], -1).abs())
        pooled = torch.cat(magnitudes, dim=1)
        if pooled.shape[1] > 2_000_000:
            keep = torch.randperm(pooled.shape[1], device=pooled.device)[:2_000_000]
            pooled = pooled[:, keep]
        u_p = torch.quantile(pooled.float(), percentile / 100.0, dim=-1).clamp_min(1e-6)
        compander = model.companders[index]
        compander._sigma.copy_(knee * u_p)
        compander.gamma.copy_(torch.full_like(u_p, target_code * (knee + 1.0) / compander.r))
        compander.beta.zero_()


# ---- structured pruning ------------------------------------------------


@torch.no_grad()
def prune_channels(model: JointPrefixFRAPPE, kept, config) -> JointPrefixFRAPPE:
    """Build a structurally smaller model holding only the ``kept`` latent channels.

    This is the pruning step proper, not a mask: the returned model has fewer
    analysis filters, fewer companding parameters, and a narrower first decoder
    convolution, so it is cheaper to run rather than merely equivalent.

    Why it is exactly equivalent
    ----------------------------
    Latent channel ``i`` occupies a *contiguous* block of decoder input channels,
    because ``adapt_to_decoder`` maps one encoder channel to ``(p_d / p_i)^2``
    consecutive decoder channels in all three of its branches (space-to-depth,
    identity, nearest upsampling).  Concatenation then lays the groups out in
    schedule order.  So the adapted latent of the pruned model is the original's
    with the dropped blocks deleted, and the first convolution -- linear in its
    input channels -- only needs the surviving columns, in the same order.
    Everything downstream of it is copied verbatim.  Hence

        pruned(x)  ==  original.decode_subset(original.adapt(codes), kept)

    holds before any optimizer step, up to float32 reassociation in the first
    convolution's channel reduction.  The integer codes are bit-identical, which
    is the property worth gating on; see ``tools/export_pruned_model.py``.

    Groups that lose every channel disappear from the schedule entirely, so a
    model whose finest scale has been rate-optimised into silence comes back with
    that scale's convolution removed rather than zeroed.

    ``config`` is the original model's configuration; only ``ps`` is rewritten.
    """
    import copy

    kept = sorted({int(channel) for channel in kept})
    if not kept or kept[0] < 1 or kept[-1] > model.n_channels:
        raise ValueError(f"kept channels must be a subset of 1..{model.n_channels}")

    # Which channels of which original scale group survive, in order.
    survivors: list[tuple[int, list[int]]] = []
    for group_index, (_ps, start, end) in enumerate(model.scale_groups):
        local = [channel - 1 - start for channel in kept if start < channel <= end]
        if local:
            survivors.append((group_index, local))

    pruned_config = copy.deepcopy(config)
    pruned_config.ps = [model.scale_groups[group_index][0]
                        for group_index, local in survivors for _ in local]
    pruned = JointPrefixFRAPPE(pruned_config).to(model.masks.device)

    for new_index, (group_index, local) in enumerate(survivors):
        source_analysis = model.analysis[group_index]
        target_analysis = pruned.analysis[new_index]
        target_analysis.weight.copy_(source_analysis.weight[local])
        target_analysis.bias.copy_(source_analysis.bias[local])
        source_compander = model.companders[group_index]
        target_compander = pruned.companders[new_index]
        target_compander._sigma.copy_(source_compander._sigma[local])
        target_compander.gamma.copy_(source_compander.gamma[local])
        target_compander.beta.copy_(source_compander.beta[local])

    # The first convolution keeps exactly the input columns of the kept blocks,
    # in the order the pruned model will concatenate them.
    pruned.first.weight.zero_()
    offset = 0
    for channel in kept:
        start, end = model.channel_slices[channel - 1]
        width = end - start
        pruned.first.weight[:, offset:offset + width].copy_(model.first.weight[:, start:end])
        offset += width
    if offset != pruned.total_decoder_channels:
        raise RuntimeError("pruned decoder width does not match the kept blocks")
    pruned.first.bias.copy_(model.first.bias)
    pruned.trunk.load_state_dict(model.trunk.state_dict())
    pruned.head.load_state_dict(model.head.state_dict())
    pruned.prefix_scale.copy_(model.prefix_scale[:len(kept)])
    pruned.prefix_bias.copy_(model.prefix_bias[:len(kept)])
    return pruned
