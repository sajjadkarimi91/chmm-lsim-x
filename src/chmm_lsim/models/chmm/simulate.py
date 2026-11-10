"""Sampling from a CHMM (`generate_chmm_time_series.m`)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.sampling import select_p
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches
from chmm_lsim.emissions.base import emission_model_for
from chmm_lsim.models.chmm.params import CHMMParams


def simulate_chmm(
    structure: LSIMStructure,
    params: CHMMParams,
    T: int,
    *,
    seed: int,
    validate: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Samples one sequence of length `T` from a CHMM
    (`generate_chmm_time_series.m`).

    Returns `(Y, X)`: observations `(D, T)` and zero-based hidden states
    `(C, T)`. Draws use `np.random.default_rng(seed)` only, so equal seeds
    give equal samples (not the MATLAB random stream)."""
    if T < 1:
        raise ValidationError(f"simulate_chmm requires T >= 1, got T={T}")
    check_structure_matches(structure, params)
    if validate:
        params.validate()
    rng = np.random.default_rng(seed)
    emission_model = emission_model_for(params.emissions)
    C = structure.C
    dims = structure.channel_state_num
    X = np.zeros((C, T), dtype=np.int64)
    Y = np.zeros((structure.D, T))
    for c in range(C):
        X[c, 0] = select_p(rng, params.pi_0[structure.state_slice(c)])
        Y[structure.dim_slice(c), 0] = emission_model.sample(rng, params.emissions[c], X[c, 0], past=Y[:, :0])
    for t in range(1, T):
        j = int(np.ravel_multi_index(tuple(int(x) for x in X[:, t - 1]), dims))
        for c in range(C):
            X[c, t] = select_p(rng, params.transition[c][j])
            Y[structure.dim_slice(c), t] = emission_model.sample(rng, params.emissions[c], X[c, t], past=Y[:, :t])
    return Y, X
