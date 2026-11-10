"""LSIM time-series simulation (`chmm/generate_chmm_time_series.m`, plus its
local `select_p`/`gmm_gen` helpers).

**Seeding, explicitly.** Every draw goes through an `np.random.Generator`
passed in (or built once, at the top of `simulate_lsim`, from the caller's
`seed`) -- never `np.random`'s global state, so two calls with the same seed
reproduce identical observations and states, and a caller doing many
independent simulations in one process never has one call's draws depend on
how many draws a previous, unrelated call happened to make.

**The transition law, applied directly** (not via the Cartesian-product
equivalent HMM of `equivalent.py` -- simulation needs exactly one column of
that object per timestep, not the whole `(J, M_z)` table, and building the
full object would impose `equivalent.py`'s `max_joint_states` ceiling on
simulation for no reason): for target channel `z`,
`p(x_t^z | x_{t-1}) = Σ_c θ[c,z]·A[c,z][x_{t-1}^c, :]`
(`im_para_eqhmm.m:46-51`'s formula, evaluated at the one previous joint state
that was actually drawn, rather than tabulated for every possible one).

**Not bit-reproducible against MATLAB, by design.** `gmm_gen.m`'s full-
covariance branch samples via `Sigma^0.5` (a symmetric matrix square root,
via eigendecomposition) applied to `randn`; this module's `_symmetric_sqrt`
uses the same construction (`eigh`, not `np.linalg.cholesky` -- see its own
docstring for why: unlike Cholesky, it accepts the singular-but-PSD
covariances `params.validate(matlab_compat=True)` legitimately allows).
Even so, cross-language sample identity was never a goal or a claim here:
MATLAB's own `rand`/`randn` streams don't agree with NumPy's `Generator` at
all, and `eigh`'s eigenvectors for a matrix with repeated or near-repeated
eigenvalues are themselves an implementation/library convention (LAPACK is
free to return any orthonormal basis of the corresponding eigenspace), so
even the "same" mathematical operation on the same input can legitimately
produce a different, equally valid square root across libraries. What *is*
preserved exactly is the algorithm: component selection, per-state/component
mean and covariance, and the transition law above.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.sampling import select_p
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches
from chmm_lsim.emissions.base import emission_model_for
from chmm_lsim.models.lsim.params import LSIMParams


_SUM_ATOL = 1e-8


def simulate_lsim(
    structure: LSIMStructure,
    params: LSIMParams,
    T: int,
    *,
    seed: int,
    validate: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """`generate_chmm_time_series.m`, driven directly by `LSIMParams` instead
    of the cell-array CHMM API. Returns `(Y, X)`: `Y` packed `(D, T)`
    observations (§0.1's layout, same as every other packed array in this
    port); `X` zero-based hidden states `(C, T)`, `X[c, t]` in
    `[0, channel_state_num[c])`.

    `validate=True` (the default) checks `T >= 1`, runs `params.validate()`,
    and -- after generating -- that every state is in range and that every
    per-step transition distribution actually summed to `1` (§0.4.11-style
    sanity: a bug in the coupling/transition slicing below would silently
    produce an invalid distribution for `select_p` to draw from otherwise)."""
    if T < 1:
        raise ValidationError(f"simulate_lsim requires T >= 1, got T={T}")
    check_structure_matches(structure, params)
    if validate:
        params.validate()

    rng = np.random.default_rng(seed)
    emission_model = emission_model_for(params.emissions)
    C, D = structure.C, structure.D
    X = np.zeros((C, T), dtype=np.int64)
    Y = np.zeros((D, T), dtype=np.float64)

    for c in range(C):
        pi0_c = params.pi_0[structure.state_slice(c)]
        X[c, 0] = select_p(rng, pi0_c)
        Y[structure.dim_slice(c), 0] = emission_model.sample(rng, params.emissions[c], X[c, 0], past=Y[:, :0])

    for t in range(1, T):
        for c in range(C):
            p = np.zeros(structure.channel_state_num[c])
            for cp in range(C):
                theta_cp_c = params.coupling[cp, c]
                if theta_cp_c == 0.0:
                    continue
                A_block = params.transition[structure.state_slice(cp), structure.state_slice(c)]
                p += theta_cp_c * A_block[X[cp, t - 1], :]
            if validate:
                total = float(p.sum())
                if not np.isclose(total, 1.0, rtol=0, atol=_SUM_ATOL):
                    raise ValidationError(
                        f"simulate_lsim: p(x_t^{c} | x_{{t-1}}) at t={t} sums to {total}, expected 1.0 -- "
                        f"check coupling column-stochasticity and transition row-stochasticity"
                    )
            X[c, t] = select_p(rng, p)
            Y[structure.dim_slice(c), t] = emission_model.sample(rng, params.emissions[c], X[c, t], past=Y[:, :t])

    if validate:
        for c in range(C):
            if np.any(X[c] < 0) or np.any(X[c] >= structure.channel_state_num[c]):
                raise ValidationError(f"simulate_lsim: channel {c} produced a state outside [0, {structure.channel_state_num[c]})")

    return Y, X
