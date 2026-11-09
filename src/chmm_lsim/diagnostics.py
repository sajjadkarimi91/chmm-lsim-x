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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.emissions import compute_emissions
from chmm_lsim.params import LSIMParams
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError, check_structure_matches, validate_observations


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


def marginalize_joint_to_channels(dims: tuple[int, ...], joint_posterior: NDArray[np.float64]) -> NDArray[np.float64]:
    """`(J, T)` joint-state posterior -> packed `(S, T)` per-channel
    marginals, `S = Σ dims`. `joint_posterior` need not be normalized per
    channel already (each channel's marginal is summed independently, so a
    caller that only has an unnormalized joint mass still gets *relative*
    per-channel mass out -- normalization, if wanted, is the caller's own
    concern, matching every other packed posterior array in this port)."""
    J, T = joint_posterior.shape
    if int(np.prod(dims)) != J:
        raise ValidationError(f"joint_posterior has J={J} rows, but Π dims={int(np.prod(dims))}")
    states_per_dim = np.array(np.unravel_index(np.arange(J), dims))  # (C, J)

    S = int(sum(dims))
    marginal = np.zeros((S, T))
    offset = 0
    for c, M_c in enumerate(dims):
        indicator = states_per_dim[c][None, :] == np.arange(M_c)[:, None]  # (M_c, J)
        marginal[offset : offset + M_c, :] = indicator.astype(np.float64) @ joint_posterior
        offset += M_c
    return marginal


def total_variation_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-timestep total variation distance between two `(N, T)` discrete
    distributions over the same `N` outcomes: `0.5 * Σ_i |p[i,t] - q[i,t]|`.
    Returns `(T,)`."""
    if p.shape != q.shape:
        raise ValidationError(f"total_variation_distance: shape mismatch {p.shape} vs {q.shape}")
    return 0.5 * np.abs(p - q).sum(axis=0)


def per_channel_total_variation(
    structure: LSIMStructure, marginal_a: NDArray[np.float64], marginal_b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Total variation distance restricted to each channel's own block of a
    packed `(S, T)` marginal pair. Returns `(C, T)`."""
    T = marginal_a.shape[1]
    out = np.zeros((structure.C, T))
    for c in range(structure.C):
        sl = structure.state_slice(c)
        out[c, :] = total_variation_distance(marginal_a[sl, :], marginal_b[sl, :])
    return out


@dataclass
class LayerBSummary:
    """Mean (over `T`) per-channel total variation distance of an
    approximate posterior (typically SED) and a baseline against the exact
    joint posterior's marginal -- and whether the approximate one is
    actually closer, the concrete claim Layer-B verification exists to
    check, not just report numbers for a human to eyeball."""

    per_channel_tv_approx: NDArray[np.float64]  # (C,)
    per_channel_tv_baseline: NDArray[np.float64]  # (C,)
    mean_tv_approx: float
    mean_tv_baseline: float
    approx_closer_than_baseline: bool


def summarize_layer_b(
    structure: LSIMStructure,
    exact_marginal: NDArray[np.float64],
    approx_marginal: NDArray[np.float64],
    baseline_marginal: NDArray[np.float64],
) -> LayerBSummary:
    """`exact_marginal`/`approx_marginal`/`baseline_marginal` are all packed
    `(S, T)`, already-marginalized-to-channels posteriors on the *same*
    observations -- `approx_marginal` typically `forward_backward_sed`'s
    `gamma`, `baseline_marginal` an explicitly-defined, deliberately weaker
    reference (e.g. each channel decoded on its own self-transition block
    alone, ignoring coupling entirely -- see `tests/test_diagnostics.py` for
    the one this port's own Layer-B tests use)."""
    tv_approx = per_channel_total_variation(structure, exact_marginal, approx_marginal).mean(axis=1)
    tv_baseline = per_channel_total_variation(structure, exact_marginal, baseline_marginal).mean(axis=1)
    mean_approx = float(tv_approx.mean())
    mean_baseline = float(tv_baseline.mean())
    return LayerBSummary(
        per_channel_tv_approx=tv_approx,
        per_channel_tv_baseline=tv_baseline,
        mean_tv_approx=mean_approx,
        mean_tv_baseline=mean_baseline,
        approx_closer_than_baseline=mean_approx < mean_baseline,
    )
