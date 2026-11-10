"""Random draws shared by the simulators and the emission samplers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def select_p(rng: np.random.Generator, p: NDArray[np.float64]) -> int:
    """`select_p.m`: draw one index from a discrete distribution `p` (summing
    to 1). Ported as `searchsorted(cumsum(p), u, side="right")` rather than
    MATLAB's `find(cumsum(p) > rand)`'s first-match -- the two are the same
    rule (`searchsorted(..., "right")` returns the count of cumulative
    entries `<= u`, i.e. exactly the smallest index whose cumulative sum
    exceeds `u`), just without building the intermediate boolean array.
    Clips the result to `len(p)-1` defensively: if `p`'s cumulative sum falls
    just short of `1.0` by rounding and `u` draws into that gap,
    `searchsorted` would otherwise return `len(p)` -- one past the end."""
    u = rng.random()
    idx = int(np.searchsorted(np.cumsum(p), u, side="right"))
    return min(idx, len(p) - 1)
