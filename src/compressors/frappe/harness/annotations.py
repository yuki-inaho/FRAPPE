"""Shape-and-dtype annotations for the harness, checked at runtime in tests.

``jaxtyping`` turns a tensor annotation into a shape contract: ``UInt8[Tensor,
"batch 3 height width"]`` is not documentation, it is a predicate, and the
repeated axis names inside one signature must agree. ``beartype`` is what
evaluates it.

The cost is per call, so it is not paid in the training inner loop. Public
harness entry points carry :func:`checked`, which is cheap next to the work they
do -- decoding a PNG, running JPEG-LS over a plane -- and the test suite installs
the import hook to extend the same checking to everything else.

Naming the axes is the point. ``arrange_planes`` returning ``UInt8[Tensor, "rows
cols"]`` from ``Int8[Tensor, "channels grid_h grid_w"]`` says the rank changes;
a comment saying the same thing does not fail when the code stops being true.
"""

from __future__ import annotations

from beartype import beartype
from jaxtyping import Float, Int8, Shaped, UInt8, jaxtyped
from torch import Tensor

__all__ = ["Float", "Int8", "Shaped", "Tensor", "UInt8", "checked"]


def checked(function):
    """Enforce this function's shape and dtype annotations at call time."""
    return jaxtyped(typechecker=beartype)(function)
