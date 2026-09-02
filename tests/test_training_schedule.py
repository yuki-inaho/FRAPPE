"""What the trainer's schedules do, pinned before they are moved.

The quantization continuation and the prefix sampler decide, respectively, which
quantizer the model is trained against and which operating points it is trained
on. Both are pure functions of the step count and the seed, so both can be pinned
exactly -- and they must be, because a schedule that silently shifts by one stage
changes a run's results without changing its configuration.
"""

from __future__ import annotations

import pytest

from src.compressors.frappe.harness.training import (
    PrefixSampler, RateTarget, continuation_stage)

BOUNDARIES = [0.10, 0.30, 0.55, 0.90]
ALPHA_RANGE = (2.0, 64.0)


@pytest.mark.parametrize("progress,expected_mode,frozen", [
    (0.00, "float", False), (0.09, "float", False),
    (0.10, "aun", False), (0.29, "aun", False),
    (0.30, "soft", False), (0.54, "soft", False),
    (0.55, "hard", False), (0.89, "hard", False),
    (0.90, "hard", True), (1.00, "hard", True),
])
def test_the_continuation_walks_its_stages_at_the_stated_boundaries(
        progress: float, expected_mode: str, frozen: bool) -> None:
    mode, _, freeze = continuation_stage(progress, BOUNDARIES, ALPHA_RANGE)
    assert (mode, freeze) == (expected_mode, frozen)


def test_the_soft_stage_anneals_geometrically_across_its_span() -> None:
    """alpha sharpens from the low bound to the high one, not linearly.

    A soft quantizer's sharpness is multiplicative: going from 2 to 4 changes the
    approximation as much as going from 32 to 64, so the schedule interpolates in
    the exponent.
    """
    low, high = ALPHA_RANGE
    start = continuation_stage(0.30, BOUNDARIES, ALPHA_RANGE)[1]
    middle = continuation_stage(0.425, BOUNDARIES, ALPHA_RANGE)[1]
    end = continuation_stage(0.5499, BOUNDARIES, ALPHA_RANGE)[1]
    assert start == pytest.approx(low)
    assert middle == pytest.approx((low * high) ** 0.5, rel=1e-3)
    assert end == pytest.approx(high, rel=1e-2)


def test_stages_other_than_soft_report_no_temperature() -> None:
    for progress in (0.0, 0.2, 0.6, 0.95):
        mode, alpha, _ = continuation_stage(progress, BOUNDARIES, ALPHA_RANGE)
        if mode != "soft":
            assert alpha == 0.0


def test_a_zero_width_float_stage_starts_at_additive_noise() -> None:
    """--target_bpp runs skip Q0: a rate term is meaningless without rounding.

    In the float stage the codes are never rounded, so the model can shrink their
    scale for free and the rate term drives the latent to zero while the
    evaluation, which does round, collapses.
    """
    mode, _, _ = continuation_stage(0.0, [0.0, 0.25, 0.55, 0.90], ALPHA_RANGE)
    assert mode == "aun"


def test_the_sampler_always_offers_the_shortest_and_the_full_prefix() -> None:
    sampler = PrefixSampler([32] * 3 + [16] * 6 + [8] * 3 + [4] * 6 + [2] * 3, extra=2)
    for _ in range(50):
        points = sampler.sample()
        assert 1 in points and 21 in points


def test_the_sampler_is_reproducible_from_its_seed() -> None:
    schedule = [32] * 3 + [16] * 6 + [8] * 3 + [4] * 6 + [2] * 3
    first = [PrefixSampler(schedule, extra=2, seed=7).sample() for _ in range(20)]
    second = [PrefixSampler(schedule, extra=2, seed=7).sample() for _ in range(20)]
    assert first == second
    assert first != [PrefixSampler(schedule, extra=2, seed=8).sample() for _ in range(20)]


def test_the_sampler_spreads_operating_points_evenly_over_rate() -> None:
    """One p=2 channel carries 256x the symbols of one p=32 channel.

    The sampler draws uniformly in log symbol count, so its extra points should
    land evenly across the rate axis rather than evenly across the channel index.
    Splitting the log-rate range into four equal bins separates the two. Measured
    over 4000 draws, with the always-present shortest and full prefixes excluded
    because a random draw that hits them is absorbed by the set:

        log-rate sampling   0.165  0.300  0.297  0.237
        uniform-in-index    0.104  0.317  0.269  0.310

    The coarse bin and the fine bin are where they part, so those are what the
    bounds check: a sampler that walked the channel index would starve the first
    and overfill the last.
    """
    import numpy as np

    schedule = [32] * 3 + [16] * 6 + [8] * 3 + [4] * 6 + [2] * 3
    log_symbols = np.log(np.cumsum([1.0 / (p * p) for p in schedule]))
    edges = np.linspace(log_symbols[0], log_symbols[-1], 5)
    bin_of = np.clip(np.digitize(log_symbols, edges[1:-1]), 0, 3)

    sampler = PrefixSampler(schedule, extra=1, seed=0)
    drawn = [n for _ in range(4000) for n in sampler.sample() if n not in (1, 21)]
    shares = np.bincount(bin_of[np.array(drawn) - 1], minlength=4) / len(drawn)
    assert shares[0] > 0.13, f"the coarsest rate band is starved: {shares.round(3)}"
    assert shares[3] < 0.27, f"the finest rate band dominates: {shares.round(3)}"


def test_subset_sampling_yields_non_prefix_operating_points() -> None:
    """Pruning to a non-prefix set only works if the decoder has seen one."""
    schedule = [32] * 3 + [16] * 6 + [8] * 3 + [4] * 6 + [2] * 3
    sampler = PrefixSampler(schedule, extra=2, seed=3)
    points = [point for _ in range(60) for point in sampler.sample(subset_prob=1.0)]
    subsets = [point for point in points if isinstance(point, list)]
    assert subsets, "subset_prob=1.0 produced no subsets"
    assert any(sorted(subset) != list(range(1, len(subset) + 1)) for subset in subsets)


def test_the_rate_target_moves_the_price_of_a_bit_toward_the_budget() -> None:
    """Over budget makes bits dearer, under budget makes them cheaper."""
    target = RateTarget(target_bpp=0.48)
    assert target.update(1.0, measured_bpp=0.96) > 1.0
    assert target.update(1.0, measured_bpp=0.24) < 1.0
    assert target.update(1.0, measured_bpp=0.48) == pytest.approx(1.0)


def test_the_rate_target_step_is_clipped() -> None:
    """One wild first measurement must not slam the multiplier into its bound.

    Without the clip an initial rate four times the budget would multiply the
    price by e^2.1, and the run would then spend thousands of steps unwinding it.
    """
    target = RateTarget(target_bpp=0.48, dual_lr=0.7, step_clip=0.7)
    import math

    assert target.update(1.0, measured_bpp=100.0) == pytest.approx(math.exp(0.7))
    # A measured rate of zero is the extreme in the other direction: every
    # channel has been driven to a single code, which the surrogate can reach.
    assert target.update(1.0, measured_bpp=0.0) == pytest.approx(math.exp(-0.7))


def test_the_rate_target_reproduces_the_multiplier_trajectory_on_record() -> None:
    """The CR-50 run's log is the reference; the extracted class must match it.

    runs/joint_21ch_cr50 started at 0.05 and measured 2.0184 bpp against a 0.48
    target at its first check, which its log records as lam_rate 0.1007.
    """
    target = RateTarget(target_bpp=0.48)
    assert target.update(0.05, measured_bpp=2.0184) == pytest.approx(0.1007, abs=5e-5)


def test_the_rate_target_never_leaves_its_bounds() -> None:
    target = RateTarget(target_bpp=0.48, maximum=20.0)
    multiplier = 1.0
    for _ in range(100):
        multiplier = target.update(multiplier, measured_bpp=50.0)
    assert multiplier == pytest.approx(20.0)
    for _ in range(200):
        multiplier = target.update(multiplier, measured_bpp=1e-9)
    assert multiplier >= 1e-6
