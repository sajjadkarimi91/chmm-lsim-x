"""`ViterbiResult`: the decoded state path returned by every Viterbi decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ViterbiResult:
    states: NDArray[np.int64]  # (C, T), zero-based
    log_score: float  # see module docstring: a sum of per-channel maxima, not a joint log-probability
