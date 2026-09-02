"""Measurement plumbing the benchmark tools share.

Four tools grew the same three pieces: the steady-median timing protocol
(median over the second half of back-to-back calls, so a warm-up tail cannot
inflate a number), the OpenVINO-to-tensor conversion every graph caller needs,
and the 24-bit raw rate that every compression ratio here is measured against.
Copied into each tool, the definitions started to drift -- ``steady`` here,
``timed`` there, different warmup defaults -- which is exactly the failure the
harness exists to prevent. One implementation, named for what it does.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable

import numpy as np
import torch

#: 8 bits per channel, three channels: the raw rate a compression ratio is against.
RAW_BITS_PER_PIXEL = 24.0


def as_torch_planes(planes: Iterable[np.ndarray]) -> list:
    """OpenVINO's ``(1, rows, cols)`` uint8 arrays as the harness's 2D tensors."""
    return [
        torch.from_numpy(np.ascontiguousarray(plane[0] if plane.ndim == 3 else plane))
        for plane in planes
    ]


def steady_median(times: list[float]) -> float:
    """Median over the second half, so a warm-up tail cannot inflate it.

    The protocol every benchmark in this repository reports under: the caller
    discards warm-up by construction (the second half of a back-to-back series
    is the steady state) rather than by guessing a burn-in length in advance.
    """
    return statistics.median(times[len(times) // 2 :])


def series(action, iterations: int) -> list[float]:
    """Wall time of ``iterations`` calls, in milliseconds."""
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        action()
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def timed(call, warmup: int, repeats: int):
    """``(result, median_ms)``: run ``call``, warm up, then take the median.

    Returns the last call's result alongside the median of ``repeats`` timed
    calls, which is the shape every round-trip loop wants: the value to keep
    processing, and the latency it cost.
    """
    result = call()
    for _ in range(max(0, warmup - 1)):
        result = call()
    samples = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return result, statistics.median(samples)
