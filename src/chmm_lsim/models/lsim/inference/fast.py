"""Fast/SVI forward-backward (`forward_backward_lsim_kernel.m`'s `variant='fast'`
branch): the backward message uses *fixed* coupling weights (no per-timestep
delta solve, no pairwise-conditional `P_cond`), raised to a fixed per-channel
exponent.

Shares its forward pass, emissions and `pi_0` regularization with `sed.py`
outright (both are `variant`-independent branches of the same MATLAB kernel,
`forward_backward_lsim_kernel.m:69-286`) -- imported from there rather than
duplicated, since `sed.py`'s versions are already P3a-verified against every
`sed` fixture and this is the same code path.

For source channel `z`, the fixed weight over target channels `w` is
`theta[z,:] / (svi_eps + sum(theta[z,:]))` (`forward_backward_lsim_kernel.m:111`),
and the fixed exponent is `w_opt[z] = sum(svi_eps + theta[z,:])`
(`forward_backward_lsim_kernel.m:114-115,135`) -- both computed once per call,
not per timestep, since neither depends on `t`. `theta[z,:] = 0` (the delta
solve's `flag_nan` case in `sed.py`) is not a hazard here: the `svi_eps` guard
in the denominator keeps the weight finite regardless.

No `beta_floor`, no `P_cond`, no `f`/`h`, no delta solve -- those are `sed`-only
constructs (§0.3's pairwise-conditional message) that the fast/SVI backward
message never forms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.numpy_ import block_sum, broadcast_block, build_W
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.inference.base import Diagnostics, Posterior
from chmm_lsim.models.lsim.inference.registry import register_from_emissions, register_inference
from chmm_lsim.models.lsim.inference.sed import _assert_block_sum_one, _forward_pass, _regularize_pi0
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import check_structure_matches, validate_observations


@register_inference("fast")
def forward_backward_fast(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants | None = None,
    *,
    validate: str = "params",
) -> Posterior:
    if fb_constants is None:
        fb_constants = FbConstants(variant="fast")
    if fb_constants.variant != "fast":
        raise ValueError(f"forward_backward_fast requires fb_constants.variant='fast', got {fb_constants.variant!r}")
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    check_structure_matches(structure, params)

    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, B_comp = compute_emissions(structure, params, Y)
    return fast_from_emissions(structure, params.pi_0, params.transition, params.coupling, B, B_comp, fb_constants,
                               validate=validate)


@register_from_emissions("fast")
def fast_from_emissions(
    structure: LSIMStructure,
    pi_0_raw: NDArray[np.float64],
    transition: NDArray[np.float64],
    coupling: NDArray[np.float64],
    B: NDArray[np.float64],
    B_comp: NDArray[np.float64],
    fb_constants: FbConstants,
    *,
    log_scale: NDArray[np.float64] | None = None,
    validate: str = "params",
) -> Posterior:
    """The fast/SVI recursions on precomputed emissions (see `sed_from_emissions`)."""
    S, T = B.shape
    C = structure.C
    W = build_W(transition, coupling, structure.block_of_state)

    diagnostics = Diagnostics()
    pi_0 = _regularize_pi0(pi_0_raw, structure, fb_constants.pi0_reg)

    alpha, scale, loglik = _forward_pass(pi_0, W, B, structure, validate=validate)

    # fixed coupling weights and exponent, per *source* channel z (kernel.m:111,114-115,135);
    # `theta[z,:]` is coupling's row z, since `build_W` reads `coupling[block(i),block(j)]`
    # as `coupling[source(i),target(j)]` -- the same convention `sed.py`'s `flag_nan`
    # (`coupling.sum(axis=1)`, a row sum) already relies on.
    row_sum = coupling.sum(axis=1)  # (C,)
    weights = coupling / (fb_constants.svi_eps + row_sum[:, None])  # (C,C), weights[z,w]
    weights = np.where(np.isnan(weights), 1.0 / C, weights)  # inert unless coupling itself has NaN (kernel.m:133)
    w_opt = np.sum(fb_constants.svi_eps + coupling, axis=1)  # (C,)

    weights_per_state = weights[structure.block_of_state]  # (S,C)
    w_opt_per_state = w_opt[structure.block_of_state]  # (S,)

    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    beta[:, T - 1] = scale[:, T - 1]
    gamma[:, T - 1] = alpha[:, T - 1] * beta[:, T - 1]
    if validate == "all":
        _assert_block_sum_one("gamma", gamma[:, T - 1], structure.state_offsets, T - 1)

    for tau in range(T - 2, -1, -1):
        if validate == "all":
            # fast's own beta formula never needs abar (only sed's P_cond does) --
            # but the underlying invariant (§0.2.1(i): abar block-sums to 1) holds
            # for any valid alpha/scale regardless of backward-message variant, and
            # P3a's contract checks it at every backward-relevant timestep. Computed
            # here purely for that check, matching sed.py's own validate="all" branch.
            abar = alpha[:, tau] * scale[:, tau]
            _assert_block_sum_one("abar", abar, structure.state_offsets, tau)

        beta_next = beta[:, tau + 1]
        beta_C = block_sum(transition * beta_next[None, :], structure.state_offsets, axis=1)  # (S,C)
        mix = np.sum(beta_C * weights_per_state, axis=1)  # (S,)
        beta_tau = scale[:, tau] * mix**w_opt_per_state

        # unconditional NaN guard (kernel.m:271) -- unlike sed's, not gated behind flag_nan
        nan_mask = np.isnan(beta_tau)
        if nan_mask.any():
            diagnostics.nan_guard_hits += int(nan_mask.sum())
            beta_tau[nan_mask] = scale[nan_mask, tau]

        # operation order preserved from kernel.m:276-282: normalizer is derived from the
        # *unnormalized* beta*alpha product, beta is divided by it, and alpha_T/gamma is
        # then recomputed from the already-normalized beta -- not by dividing the first
        # product directly. Mathematically identical to sed's order, kept separate for
        # float-level parity with MATLAB.
        alpha_T_temp = beta_tau * alpha[:, tau]
        normalizer = broadcast_block(block_sum(alpha_T_temp, structure.state_offsets), structure.block_of_state)
        beta_tau = beta_tau / normalizer
        gamma_tau = beta_tau * alpha[:, tau]

        beta[:, tau] = beta_tau
        gamma[:, tau] = gamma_tau

        if validate == "all":
            _assert_block_sum_one("gamma", gamma_tau, structure.state_offsets, tau)

    if log_scale is not None:
        loglik = loglik + float(np.sum(log_scale))
    return Posterior(
        loglik=loglik,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        scale=scale,
        B=B,
        B_comp=B_comp,
        log_scale=log_scale,
        diagnostics=diagnostics,
        structure=structure,
    )
