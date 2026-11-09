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

from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError, check_structure_matches

_SUM_ATOL = 1e-8


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


def _symmetric_sqrt(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """`Sigma^0.5` via symmetric eigendecomposition, matching MATLAB's own
    `^0.5` operator on a symmetric matrix -- unlike `np.linalg.cholesky`,
    this is defined for *singular* PSD matrices too (a rank-deficient
    covariance, e.g. from a degenerate EM fit or a hand-built test case).
    `params.validate(matlab_compat=True)` already accepts a covariance whose
    smallest eigenvalue is merely close to zero (within `_EIG_REL_TOL`,
    §0.4.10) but rejects anything genuinely negative-definite beyond that
    tolerance (`validate.py`'s own `_validate_full_sigma`), so any negative
    eigenvalue reaching here is floating-point noise around zero, not a real
    negative-definite direction -- clipped to `0.0` rather than left to turn
    `sqrt` into `nan`."""
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ (np.sqrt(eigvals)[:, None] * eigvecs.T)


def _gmm_sample(rng: np.random.Generator, gmm: ChannelGMM, state: int) -> NDArray[np.float64]:
    """`gmm_gen.m`: pick a mixture component for `state` via `select_p`, then
    sample that component's Gaussian -- `sqrt(variance)` elementwise for
    diagonal covariance, `_symmetric_sqrt(Sigma)` for full (see module
    docstring for why this isn't bit-reproducible against MATLAB, and
    `_symmetric_sqrt`'s own docstring for why it's used here instead of the
    more common Cholesky factor)."""
    k = select_p(rng, gmm.weights[state, :])
    mu = gmm.mu[:, state, k]
    z = rng.standard_normal(mu.shape[0])
    if gmm.diag:
        return mu + np.sqrt(gmm.sigma[:, state, k]) * z
    L = _symmetric_sqrt(gmm.sigma[:, :, state, k])
    return mu + L @ z


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
    C, D = structure.C, structure.D
    X = np.zeros((C, T), dtype=np.int64)
    Y = np.zeros((D, T), dtype=np.float64)

    for c in range(C):
        pi0_c = params.pi_0[structure.state_slice(c)]
        X[c, 0] = select_p(rng, pi0_c)
        Y[structure.dim_slice(c), 0] = _gmm_sample(rng, params.emissions[c], X[c, 0])

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
            Y[structure.dim_slice(c), t] = _gmm_sample(rng, params.emissions[c], X[c, t])

    if validate:
        for c in range(C):
            if np.any(X[c] < 0) or np.any(X[c] >= structure.channel_state_num[c]):
                raise ValidationError(f"simulate_lsim: channel {c} produced a state outside [0, {structure.channel_state_num[c]})")

    return Y, X
