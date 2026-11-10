"""SED forward-backward (`forward_backward_lsim_kernel.m`'s `variant='sed'`
branch): pairwise-conditional backward message, delta weights solved per
channel from `x = (diag(f-h) + h*ones(C,C)) \\ f`.

Two ways to build the per-timestep `(S,S)` conditional `P_cond`, cross-checked
against each other in the tests. Neither stores a
`(S,S,T)` tensor the way `forward_backward_lsim_kernel.m` does -- both
reconstruct one `(S,S)` matrix per backward step, from the already-stored
`(S,T)` `alpha`/`scale`, at every `t`:
  - `method="reference"`: `P_cond = W + (diag_zero * abar) @ W` -- the literal
    `(S,S)@(S,S)` matrix product the MATLAB kernel performs to build that
    timestep's slice of its `P_vz_tm1_vw_t_O_cond` tensor, an `O(S^3)` step
    per `t`.
  - `method="optimized"` (default): `P_cond = W + alpha_next - broadcast(Gz)`
    (§0.2) -- an `O(S^2)` elementwise construction with no matrix-matrix
    product, `Gz` (`(C,S)`) being the only extra per-`t` quantity.
Both feed the same beta_C/f/h/delta-solve logic once `P_cond` exists, so a
mismatch between the two isolates the bug to `P_cond`'s construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.numpy_ import block_sum, broadcast_block, build_W
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.inference.base import Diagnostics, Posterior
from chmm_lsim.models.lsim.inference.registry import register_from_emissions, register_inference
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches, validate_observations

_VALID_METHODS = ("optimized", "reference")


@register_inference("sed")
def forward_backward_sed(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants | None = None,
    *,
    validate: str = "params",
    method: str = "optimized",
) -> Posterior:
    if fb_constants is None:
        fb_constants = FbConstants(variant="sed")
    if fb_constants.variant != "sed":
        raise ValueError(f"forward_backward_sed requires fb_constants.variant='sed', got {fb_constants.variant!r}")
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    check_structure_matches(structure, params)

    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, B_comp = compute_emissions(structure, params, Y)
    return sed_from_emissions(structure, params.pi_0, params.transition, params.coupling, B, B_comp, fb_constants,
                              validate=validate, method=method)


@register_from_emissions("sed")
def sed_from_emissions(
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
    method: str = "optimized",
) -> Posterior:
    """The SED recursions on precomputed emissions `B` (S,T) / `B_comp` (S,max_K,T).
    `B`/`B_comp` may be rescaled per (channel, t) by `exp(-log_scale)` ((C,T)); the
    recursions are invariant to that and `log_scale.sum()` is added back to the
    log-likelihood."""
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    S, T = B.shape
    C = structure.C
    W = build_W(transition, coupling, structure.block_of_state)
    diag_zero = 1.0 - (structure.block_of_state[:, None] == structure.block_of_state[None, :]).astype(np.float64)
    flag_nan = bool(np.any(coupling.sum(axis=1) <= 1e-10))

    diagnostics = Diagnostics()
    pi_0 = _regularize_pi0(pi_0_raw, structure, fb_constants.pi0_reg)

    alpha, scale, loglik = _forward_pass(pi_0, W, B, structure, validate=validate)

    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    beta[:, T - 1] = scale[:, T - 1]
    gamma[:, T - 1] = alpha[:, T - 1] * beta[:, T - 1]
    if validate == "all":
        _assert_block_sum_one("gamma", gamma[:, T - 1], structure.state_offsets, T - 1)

    for tau in range(T - 2, -1, -1):
        abar = alpha[:, tau] * scale[:, tau]
        alpha_next = alpha[:, tau + 1]
        diagnostics.zero_alpha_next_count += int(np.sum(alpha_next == 0))

        if validate == "all":
            _assert_block_sum_one("abar", abar, structure.state_offsets, tau)
            v2 = block_sum(alpha_next * beta[:, tau + 1], structure.state_offsets)
            _assert_close_to_one("V2", v2, tau + 1)

        if method == "reference":
            p_cond = W + (diag_zero * abar[None, :]) @ W
        else:
            Gz = block_sum(abar[:, None] * W, structure.state_offsets, axis=0)  # (C,S)
            p_cond = W + alpha_next[None, :] - broadcast_block(Gz, structure.block_of_state, axis=0)

        beta_next = beta[:, tau + 1]
        beta_C = block_sum(p_cond * beta_next[None, :], structure.state_offsets, axis=1)  # (S,C)

        if C > 1:
            weighted = (p_cond * abar[:, None]) ** 2
            num = np.zeros_like(weighted)
            np.divide(weighted, alpha_next[None, :], out=num, where=alpha_next[None, :] != 0)
            f = block_sum(block_sum(num, structure.state_offsets, axis=1), structure.state_offsets, axis=0)
            f = f + fb_constants.f_reg  # (C,C), rows=z, cols=w
            h = block_sum(abar**2, structure.state_offsets)  # (C,)

            if validate == "all":
                if not (np.all(np.isfinite(f)) and np.all(np.isfinite(h))):
                    raise ValidationError(f"f or h at t={tau} contains NaN/Inf -- f={f.tolist()}, h={h.tolist()}")
                if np.any(f <= h[:, None]):
                    raise ValidationError(
                        f"f<=h at t={tau}: this is unreachable for valid parameters (§0.3.3) -- "
                        f"check coupling column-stochasticity and transition row-stochasticity"
                    )
            diagnostics.f_le_h_violations += int(np.sum(f <= h[:, None]))

            delta = _solve_delta(f, h)  # (C,C)
            weights_subsystems = broadcast_block(delta, structure.block_of_state, axis=0)  # (S,C)
        else:
            weights_subsystems = np.ones((S, 1))

        raw = np.sum(beta_C * weights_subsystems, axis=1)
        diagnostics.beta_floor_hits += int(np.sum(raw < fb_constants.beta_floor))
        beta_tau = scale[:, tau] * np.maximum(fb_constants.beta_floor, raw)

        if flag_nan:
            nan_mask = np.isnan(beta_tau)
            if nan_mask.any():
                diagnostics.nan_guard_hits += int(nan_mask.sum())
                beta_tau[nan_mask] = scale[nan_mask, tau]

        gamma_tau = beta_tau * alpha[:, tau]
        normalizer = broadcast_block(block_sum(gamma_tau, structure.state_offsets), structure.block_of_state)
        beta[:, tau] = beta_tau / normalizer
        gamma[:, tau] = gamma_tau / normalizer

        if validate == "all":
            _assert_block_sum_one("gamma", gamma[:, tau], structure.state_offsets, tau)

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


def _regularize_pi0(pi_0: NDArray[np.float64], structure: LSIMStructure, pi0_reg: float) -> NDArray[np.float64]:
    if pi0_reg <= 0:
        return pi_0.copy()
    out = pi_0.copy()
    for c in range(structure.C):
        sl = structure.state_slice(c)
        block = pi0_reg + pi_0[sl]
        out[sl] = block / block.sum()
    return out


def _forward_pass(
    pi_0: NDArray[np.float64],
    W: NDArray[np.float64],
    B: NDArray[np.float64],
    structure: LSIMStructure,
    *,
    validate: str = "params",
) -> tuple[NDArray, NDArray, float]:
    S, T = B.shape
    alpha = np.zeros((S, T))
    scale = np.zeros((S, T))
    p_ot_c_cond_past = np.zeros((structure.C, T))

    alpha[:, 0] = pi_0
    p_ot_c_cond_past[:, 0] = block_sum(alpha[:, 0] * B[:, 0], structure.state_offsets)
    scale[:, 0] = B[:, 0] / broadcast_block(p_ot_c_cond_past[:, 0], structure.block_of_state)
    if validate == "all":
        _assert_block_sum_one("alpha", alpha[:, 0], structure.state_offsets, 0)

    for t in range(1, T):
        abar = alpha[:, t - 1] * scale[:, t - 1]
        alpha[:, t] = W.T @ abar
        p_ot_c_cond_past[:, t] = block_sum(alpha[:, t] * B[:, t], structure.state_offsets)
        scale[:, t] = B[:, t] / broadcast_block(p_ot_c_cond_past[:, t], structure.block_of_state)
        if validate == "all":
            _assert_block_sum_one("alpha", alpha[:, t], structure.state_offsets, t)

    loglik = float(np.sum(np.log(p_ot_c_cond_past)))
    if np.isnan(loglik):
        raise ValidationError("forward pass produced a NaN log-likelihood (P_O_model)")
    return alpha, scale, loglik


def _assert_block_sum_one(name: str, x: NDArray[np.float64], offsets: NDArray, t: int) -> None:
    """§0.2.1(i)/(ii): per-block sums of `alpha`/`abar`/`gamma` (and, via the
    caller, `V2`) are exactly 1. Checked explicitly rather than left to a
    `<=`/`>` comparison, since `NaN <= x` and `NaN > x` are both `False` --
    a non-finite value must fail loudly, not silently pass a threshold check."""
    if not np.all(np.isfinite(x)):
        raise ValidationError(f"{name} at t={t} contains NaN/Inf: {x.tolist()}")
    sums = block_sum(x, offsets)
    _assert_close_to_one(name, sums, t)


def _assert_close_to_one(name: str, sums: NDArray[np.float64], t: int) -> None:
    if not np.all(np.isfinite(sums)):
        raise ValidationError(f"{name} per-block sums at t={t} contain NaN/Inf: {sums.tolist()}")
    if not np.allclose(sums, 1.0, rtol=0, atol=1e-8):
        raise ValidationError(f"{name} per-block sums at t={t} are {sums.tolist()}, expected all 1.0 (§0.2.1)")


def reconstruct_sed_diagnostics(
    structure: LSIMStructure,
    params: LSIMParams,
    alpha: NDArray[np.float64],
    scale: NDArray[np.float64],
    fb_constants: FbConstants,
    t_samples: NDArray[np.int64],
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Reconstructs `f`, `h`, `delta`, `Gz` at 0-based timestep indices `t_samples`
    from the public `alpha`/`scale` outputs of `forward_backward_sed` -- the same
    method as `tools/matlab/check_sed_invariants.m` and the exporter's
    `local_sed_diagnostics`, so this can be compared against a fixture's
    `diag_f`/`diag_h`/`diag_delta`/`diag_Gz` without depending on this module's
    own backward-loop internals matching MATLAB by construction."""
    W = build_W(params.transition, params.coupling, structure.block_of_state)
    n = len(t_samples)
    C, S = structure.C, structure.S
    f_out = np.zeros((n, C, C))
    h_out = np.zeros((n, C))
    delta_out = np.zeros((n, C, C))
    Gz_out = np.zeros((n, C, S))

    for idx, t in enumerate(t_samples):
        t = int(t)
        abar = alpha[:, t] * scale[:, t]
        alpha_next = alpha[:, t + 1]
        Gz = block_sum(abar[:, None] * W, structure.state_offsets, axis=0)  # (C,S)
        p_cond = W + alpha_next[None, :] - broadcast_block(Gz, structure.block_of_state, axis=0)

        weighted = (p_cond * abar[:, None]) ** 2
        num = np.zeros_like(weighted)
        np.divide(weighted, alpha_next[None, :], out=num, where=alpha_next[None, :] != 0)
        f = block_sum(block_sum(num, structure.state_offsets, axis=1), structure.state_offsets, axis=0)
        f = f + fb_constants.f_reg
        h = block_sum(abar**2, structure.state_offsets)

        f_out[idx] = f
        h_out[idx] = h
        delta_out[idx] = _solve_delta(f, h)
        Gz_out[idx] = Gz

    return f_out, h_out, delta_out, Gz_out


def _solve_delta(f: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise closed-form solve of `(diag(f_z - h_z) + h_z*ones(C,C)) @ x_z
    = f_z` via Sherman-Morrison (§0.3.3), for every channel z at once.
    `f`, `d = f - h[:,None]` are `(C,C)`, rows indexed by z, columns by w."""
    d = f - h[:, None]
    s1 = np.sum(f / d, axis=1)  # (C,) per-z
    s2 = np.sum(1.0 / d, axis=1)  # (C,)
    g = h * s1 / (1.0 + h * s2)  # (C,)
    return (f - g[:, None]) / d
