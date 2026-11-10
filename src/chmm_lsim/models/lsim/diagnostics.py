"""Joint hidden-path/observation log-probability (`common function/
compute_joint_state_observation_prob.m`), plus the reusable invariant/metric
helpers Layer-B verification needs on top of it: marginalizing a joint-state
posterior down to packed per-channel marginals, per-channel total-variation
distance, and a small SED-vs-exact summary.

**Works on models of any size, unlike `equivalent.py`.** `compute_
joint_state_observation_prob.m` never forms the Cartesian-product conditional
transition table `equivalent.py` builds -- it only ever needs `p(x_t^z |
x_{t-1})` at the *one* specific joint previous state a given hidden path
actually visits, evaluated directly from the LSIM transition law (`Σ_c
θ[c,z]·A[c,z][x_{t-1}^c, x_t^z]`, `simulate.py`'s formula, restated here for
one path instead of applied generatively). So `joint_log_probability` has no
`max_joint_states` ceiling and works on real, EM-fitted models with as many
channels and states as `em.py` itself supports -- only the Layer-B helpers
below (which take a joint-state posterior as input, something only
`equivalent.py`'s bounded-size equivalent HMM produces) are restricted by
that ceiling, indirectly, through their caller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches, validate_observations
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.params import LSIMParams


def joint_log_probability(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    X: NDArray[np.int64],
    *,
    validate: bool = True,
) -> float:
    """`compute_joint_state_observation_prob.m`, from `LSIMParams`/packed `Y`
    /zero-based hidden states `X` `(C, T)` directly. `-inf` for a
    structurally impossible path (a zero `pi_0`, transition, or emission
    density anywhere along it) -- `log(0)` is `-inf`, and `-inf` plus any
    further finite term stays `-inf`, so this propagates correctly with no
    NaN risk (checked explicitly in tests, not just assumed).

    `Y`'s shape/dtype and `X`'s shape/dtype/range are always checked
    (cheap, and load-bearing: a floating `X` previously escaped straight
    into `IndexError` from `params.pi_0[s0 + X[c,0]]` below, since NumPy
    indexing silently truncates or rejects a float index depending on
    exactly how it's used, and an integer `Y` was silently accepted despite
    `§2.1.1` requiring observations to be floating-point). `check_structure_
    matches` runs unconditionally too (cheap, an `O(1)` tuple comparison);
    `validate=True` (the default) additionally runs the more expensive
    `params.validate()`, skippable by a caller that has already validated
    `params` once and is calling this in a loop."""
    check_structure_matches(structure, params)
    if validate:
        params.validate()
    validate_observations(structure, Y)

    if X.ndim != 2:
        raise ValidationError(f"X.shape={X.shape}, expected 2-D (C, T)")
    if not np.issubdtype(X.dtype, np.integer):
        raise ValidationError(f"X.dtype={X.dtype}, expected an integer dtype (zero-based hidden states)")
    if X.shape != (structure.C, Y.shape[1]):
        raise ValidationError(f"X.shape={X.shape}, expected (C,T)=({structure.C},{Y.shape[1]})")
    for c in range(structure.C):
        if np.any(X[c] < 0) or np.any(X[c] >= structure.channel_state_num[c]):
            raise ValidationError(f"X[{c}] contains a state outside [0, {structure.channel_state_num[c]})")

    B, _ = compute_emissions(structure, params, Y)
    T = Y.shape[1]

    with np.errstate(divide="ignore"):
        log_prob = 0.0
        for c in range(structure.C):
            s0 = structure.state_offsets[c]
            log_prob += float(np.log(params.pi_0[s0 + X[c, 0]])) + float(np.log(B[s0 + X[c, 0], 0]))

        for t in range(1, T):
            prev_global = structure.state_offsets[:-1] + X[:, t - 1]  # (C,), one global index per channel
            for c in range(structure.C):
                s0 = structure.state_offsets[c]
                next_global = s0 + X[c, t]
                p = float(params.coupling[:, c] @ params.transition[prev_global, next_global])
                log_prob += float(np.log(p)) + float(np.log(B[next_global, t]))

    return log_prob
