"""The schedules a joint-prefix run is steered by.

Three things decide what a run actually optimises, and all three were buried in
the trainer's ``main``: which quantizer the model is trained against, which
operating points it is trained on, and how much a bit costs. Each is a pure
function of the step count, the seed and a measurement, so each can be tested on
its own -- and needs to be, because a schedule that shifts by one stage changes a
run's results without changing its configuration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import IMAGE_GLOB

#: The continuation's stages, coarse to fine, in schedule order.
STAGES = ("float", "aun", "soft", "hard")


class CropDataset(torch.utils.data.Dataset):
    """Random crops from an anonymous local ImageFolder split."""

    def __init__(self, root: Path, split: str, crop: int, augment: bool = True,
                 limit: int | None = None) -> None:
        self.files = sorted((root / split).glob(IMAGE_GLOB))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise SystemExit(f"no anonymous PNG images under {root / split}")
        self.crop = crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as handle:
            handle.load()
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        size = self.crop
        if h < size or w < size:
            raise SystemExit(f"image {h}x{w} smaller than the requested {size} crop")
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        patch = image[top:top + size, left:left + size]
        if self.augment:
            if random.random() < 0.5:
                patch = patch[:, ::-1]
            if random.random() < 0.5:
                patch = patch[::-1]
        return torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class PrefixSampler:
    """Sandwich sampling over operating points spaced uniformly in log rate.

    Symbol counts differ by 256x across the schedule, so sampling the channel
    index uniformly would concentrate almost every sample on rates nobody uses.
    Sampling uniformly in ``log C_n`` spreads the sampled operating points over
    the rate axis instead.
    """

    def __init__(self, ps: list[int], extra: int = 1, seed: int = 0) -> None:
        self.n_channels = len(ps)
        symbols = np.cumsum([1.0 / (p * p) for p in ps])
        self.log_symbols = np.log(symbols)
        self.extra = extra
        self.rng = random.Random(seed)

    def sample(self, subset_prob: float = 0.0) -> list:
        prefixes = {1, self.n_channels}
        low, high = self.log_symbols[0], self.log_symbols[-1]
        extra = []
        for _ in range(self.extra):
            target = self.rng.uniform(low, high)
            n = int(np.abs(self.log_symbols - target).argmin()) + 1
            if subset_prob and self.rng.random() < subset_prob:
                # A random subset of the same size: pruning a codec to a
                # non-prefix channel set only works if the decoder has seen
                # non-prefix masks during training.
                extra.append(sorted(self.rng.sample(range(1, self.n_channels + 1), n)))
            else:
                prefixes.add(n)
        return sorted(prefixes) + extra


def continuation_stage(progress: float, boundaries: list[float],
                       alpha_range: tuple[float, float]) -> tuple[str, float, bool]:
    """Map training progress onto (quantization mode, soft-round alpha, frozen encoder).

    Stages are Q0 float, Q1 additive uniform noise, Q2 annealed soft rounding,
    Q3 hard rounding with a straight-through estimator, and Q4 hard calibration
    with the analysis path frozen so only the synthesis transform adapts.
    """
    q0, q1, q2, q3 = boundaries
    if progress < q0:
        return "float", 0.0, False
    if progress < q1:
        return "aun", 0.0, False
    if progress < q2:
        span = max(q2 - q1, 1e-6)
        ratio = (progress - q1) / span
        low, high = alpha_range
        return "soft", float(low * (high / low) ** ratio), False
    if progress < q3:
        return "hard", 0.0, False
    return "hard", 0.0, True


@dataclass
class RateTarget:
    """Steer the Lagrange multiplier until a measured bitrate hits its target.

    The rate term is a price on bits, and the price that lands a model on a
    chosen operating point is not knowable in advance -- so it is not guessed.
    After each validation the real JPEG-LS bitrate is compared with the target
    and the multiplier is moved multiplicatively toward closing the gap.

    The step is clipped because the first measurement is often far off: without
    it, one over-budget check would slam the multiplier into its bound and the
    run would spend thousands of steps recovering. A factor of ``e^0.7`` per
    check, roughly a doubling, converges within a handful of validations while
    staying stable.
    """

    target_bpp: float
    dual_lr: float = 0.7
    maximum: float = 20.0
    step_clip: float = 0.7

    def update(self, multiplier: float, measured_bpp: float) -> float:
        step = float(np.clip(self.dual_lr * (measured_bpp / self.target_bpp - 1.0),
                             -self.step_clip, self.step_clip))
        return float(np.clip(multiplier * math.exp(step), 1e-6, self.maximum))
