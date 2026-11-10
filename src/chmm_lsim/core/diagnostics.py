"""Model-independent verification helpers: marginalizing a joint-state posterior
to stacked per-channel marginals, total-variation distances and a small
approximate-vs-exact posterior summary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError


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
