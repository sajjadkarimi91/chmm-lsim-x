"""POISE forward-backward (`forward_backward_lsim_kernel.m`'s `variant='poise'`
branch, and the public `forward_backward_lsim_poise.m`): the backward message
raises the *outgoing-influence mixture* to the channel's own total outgoing
coupling strength, with an optional one-shot observation-conditioned refinement.

Shares its forward pass, emissions and `pi_0` regularization with `sed.py`
outright -- POISE changes only the backward recursion, exactly as the MATLAB
kernel does (one shared forward, three `variant` branches).

**Raw recursion** (`forward_backward_lsim_kernel.m:279-297`). For source channel
`z`, with `theta = coupling`:

    theta_out[z] = sum_c theta[z,c]                  total outgoing strength
    rho[z,c]     = theta[z,c] / theta_out[z]         outgoing mixture weights
    q_t[i]       = sum_c rho[z(i),c] * sum_{n in block c} transition[i,n] * beta_{t+1}[n]
    beta_t[i]    = b_t[i] * q_t[i] ** theta_out[z(i)]

`theta_out[z] == 0` (a channel with no outgoing coupling at all -- the exact-zero
branch `c3_coupling_zero_row` exercises) takes `beta_t[i] = b_t[i]` instead:
`0 ** 0 == 1` would silently make that channel's message the bare observation
term anyway, but the kernel writes the branch explicitly and so does this port.
`rho` for such a channel stays all-zero, never `0/0`.

Unlike `sed`, POISE forms no pairwise-conditional `(S,S)` message, no `f`/`h` and
no delta solve, and unlike `fast` it uses no `svi_eps` guard: the zero case is
handled exactly rather than regularized. Consequently POISE reports **no**
`Diagnostics` counters -- there is no `beta_floor` clamp, no `f<=h` case and no
NaN guard in the MATLAB POISE branch to count (`fast` has the NaN guard, `sed`
has all four).

**One-shot refinement** (`forward_backward_lsim_kernel.m:299-324`, off by default
in this port -- see `FbConstants.poise_refinement`). Every propagated marginal is
formed from the *frozen* raw smoothed marginals, never from an already refined
slice, so the pass is genuinely one-shot rather than a second recursion:

    propagated_t = normalize_blocks( b_t * (alpha_T_raw[t-1] @ W) )     for t = 1..T-1
    w1[z]        = max(0, 0.75 - theta_out[z])
    alpha_T[t]   = w1 * propagated_t + (1 - w1) * alpha_T_raw[t]

`t = 0` keeps its raw marginal (the MATLAB loop starts at `t = 2`), and `T == 1`
skips the pass entirely. The `0.75` is the literal weight rule of the MATLAB
kernel, kept as a constant here rather than exposed as a tunable: there is one
canonical POISE definition.

**Normalizer guard.** Both the raw recursion and the refinement go through
`check_channel_normalizer`, the port of MATLAB's `normalize_blocks` guard: a
nonpositive or non-finite per-channel normalizer raises rather than producing
`0/0`. It is unconditional, like MATLAB's. Note the *unified* MATLAB kernel
(`forward_backward_lsim_kernel.m`'s POISE branch) divided unguarded while the
standalone `forward_backward_lsim_poise.m` guarded -- P12 added the guard there
too, so the two MATLAB paths agree with each other and with this port.

**`beta` with refinement on.** EM's transition statistic reads `beta`, so the
kernel reconstructs the *effective* message whose one-slice marginal is exactly
the refined `alpha_T`: `beta = alpha_T / alpha` where `alpha > 0`, and `0`
elsewhere (`forward_backward_lsim_kernel.m:320-324`). That is what this port
returns, so `gamma == alpha * beta` holds wherever `alpha > 0` but *not* at
states with `alpha == 0`, where a refined `gamma` can be nonzero while `beta` is
0. With refinement off the raw recursion gives `gamma == alpha * beta`
identically, as `sed`/`fast` do.
"""

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
from chmm_lsim.core.validate import ValidationError, check_structure_matches, validate_observations

_REFINE_W0 = 0.75
"""The refinement's weight rule constant: `w1 = max(0, 0.75 - theta_out)`
(`forward_backward_lsim_kernel.m:302`). Hardcoded on purpose -- MATLAB has no
option for it, so neither does this port."""


def check_channel_normalizer(
    normalizer_channel: NDArray[np.float64], t: int, *, stage: str, sequence: int | None = None
) -> None:
    """MATLAB's `normalize_blocks` guard (`forward_backward_lsim_poise.m:180-188`),
    ported literally: a per-channel normalizer that is nonpositive or non-finite is
    an **error**, not something to divide by.

    This is unconditional -- it does not depend on `validate=` -- because MATLAB's
    is, and because the alternative is silent corruption rather than a slow check:
    the normalizer is a channel's total smoothed mass, so a zero one makes every
    state of that channel `0/0`, and with the refinement on that NaN then spreads
    through `propagated` to *every* channel and timestep. It is reachable with a
    perfectly valid model and a finite forward log-likelihood -- see
    `tests/test_p12_poise_normalizer_guard.py` for the exact C=2 construction --
    so it cannot be left to the caller to notice.

    Note this does not floor, clamp or otherwise alter the algorithm: it reports.
    """
    bad = ~np.isfinite(normalizer_channel) | (normalizer_channel <= 0.0)
    if not np.any(bad):
        return
    c = int(np.flatnonzero(bad)[0])
    where = "" if sequence is None else f"sequence {sequence}, "
    raise ValidationError(
        f"POISE {stage}: {where}channel {c} normalizer at t={t} is nonpositive or non-finite "
        f"({normalizer_channel[c]!r}); a channel's total smoothed mass must be finite and "
        f"positive. Dividing by it would return NaN posteriors from a model whose forward "
        f"log-likelihood is finite. This is MATLAB forward_backward_lsim_poise.m's own "
        f"'nonpositive or nonfinite channel normalizer' error."
    )


def poise_backward_constants(
    structure: LSIMStructure, coupling: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """The three per-state quantities POISE's backward recursion needs, computed
    once per call (none depend on `t`): the outgoing mixture weights `rho`
    `(S,C)`, the outgoing strength exponent `theta_out` `(S,)`, and the
    exact-zero mask. Shared by every backend so the three implementations cannot
    drift in how they read `coupling`'s rows."""
    theta_out_channel = coupling.sum(axis=1)  # (C,) -- sum over TARGET channels of source row z
    rho = np.zeros_like(coupling)
    nonzero = theta_out_channel > 0
    rho[nonzero] = coupling[nonzero] / theta_out_channel[nonzero, None]
    rho_per_state = rho[structure.block_of_state]  # (S,C)
    theta_out = theta_out_channel[structure.block_of_state]  # (S,)
    return rho_per_state, theta_out, theta_out == 0


@register_inference("poise")
def forward_backward_poise(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants | None = None,
    *,
    validate: str = "params",
) -> Posterior:
    if fb_constants is None:
        fb_constants = FbConstants(variant="poise")
    if fb_constants.variant != "poise":
        raise ValueError(f"forward_backward_poise requires fb_constants.variant='poise', got {fb_constants.variant!r}")
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    check_structure_matches(structure, params)

    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, B_comp = compute_emissions(structure, params, Y)
    return poise_from_emissions(structure, params.pi_0, params.transition, params.coupling, B, B_comp, fb_constants,
                                validate=validate)


@register_from_emissions("poise")
def poise_from_emissions(
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
    """The POISE recursions on precomputed emissions (see `sed_from_emissions`)."""
    S, T = B.shape
    W = build_W(transition, coupling, structure.block_of_state)

    diagnostics = Diagnostics()
    pi_0 = _regularize_pi0(pi_0_raw, structure, fb_constants.pi0_reg)

    alpha, scale, loglik = _forward_pass(pi_0, W, B, structure, validate=validate)

    rho_per_state, theta_out, zero_theta = poise_backward_constants(structure, coupling)

    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    beta[:, T - 1] = scale[:, T - 1]
    gamma[:, T - 1] = alpha[:, T - 1] * beta[:, T - 1]
    if validate == "all":
        _assert_block_sum_one("gamma", gamma[:, T - 1], structure.state_offsets, T - 1)

    for tau in range(T - 2, -1, -1):
        if validate == "all":
            # POISE's own message never needs abar, but §0.2.1(i)'s invariant holds for any
            # valid alpha/scale regardless of backward variant -- checked at every
            # backward-relevant timestep exactly as sed.py/fast.py do.
            _assert_block_sum_one("abar", alpha[:, tau] * scale[:, tau], structure.state_offsets, tau)

        beta_next = beta[:, tau + 1]
        # child_messages[i,c] = sum_{n in block c} transition[i,n]*beta_next[n]
        # (kernel.m:283's `transitions_matrices * mat_mult`, whose (S,C) layout is exactly
        # this per-target-channel block sum -- no (S,S) intermediate is formed).
        child_messages = block_sum(transition * beta_next[None, :], structure.state_offsets, axis=1)  # (S,C)
        q = np.sum(child_messages * rho_per_state, axis=1)  # (S,)

        beta_tau = scale[:, tau] * q**theta_out
        beta_tau[zero_theta] = scale[zero_theta, tau]  # kernel.m:287-288, the exact-zero branch

        # kernel.m:290-296's operation order: alpha_T is formed from the UNNORMALIZED beta,
        # then beta and alpha_T are both divided by the same per-channel normalizer -- not by
        # recomputing gamma from an already-normalized beta (fast.py's order). Mathematically
        # identical, kept as the kernel writes it for float-level parity with MATLAB.
        alpha_T_tau = beta_tau * alpha[:, tau]
        normalizer_channel = block_sum(alpha_T_tau, structure.state_offsets)
        check_channel_normalizer(normalizer_channel, tau, stage="backward recursion")
        normalizer = broadcast_block(normalizer_channel, structure.block_of_state)
        beta[:, tau] = beta_tau / normalizer
        gamma[:, tau] = alpha_T_tau / normalizer

        if validate == "all":
            _assert_block_sum_one("gamma", gamma[:, tau], structure.state_offsets, tau)

    if fb_constants.poise_refinement and T > 1:
        gamma, beta = _refine(structure, W, alpha, scale, gamma, theta_out, validate=validate)

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


def _refine(
    structure: LSIMStructure,
    W: NDArray[np.float64],
    alpha: NDArray[np.float64],
    scale: NDArray[np.float64],
    gamma_raw: NDArray[np.float64],
    theta_out: NDArray[np.float64],
    *,
    validate: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The one-shot refinement pass and the effective-`beta` reconstruction
    (`forward_backward_lsim_kernel.m:299-324`). `gamma_raw` is left untouched --
    every propagated slice reads the frozen raw marginals."""
    T = gamma_raw.shape[1]
    w1 = np.maximum(0.0, _REFINE_W0 - theta_out)  # (S,)
    gamma = gamma_raw.copy()

    for t in range(1, T):
        # kernel.m:306-308: the forward's own one-step prediction applied to the previous RAW
        # smoothed marginal (`transitions' * mat_mult` then the incoming-coupling mixture is
        # exactly `x @ W`), times the observation term, block-normalized.
        propagated = scale[:, t] * (gamma_raw[:, t - 1] @ W)
        normalizer_channel = block_sum(propagated, structure.state_offsets)
        check_channel_normalizer(normalizer_channel, t, stage="refinement")
        normalizer = broadcast_block(normalizer_channel, structure.block_of_state)
        propagated = propagated / normalizer
        gamma[:, t] = w1 * propagated + (1.0 - w1) * gamma_raw[:, t]

        if validate == "all":
            _assert_block_sum_one("gamma", gamma[:, t], structure.state_offsets, t)

    # kernel.m:320-324: EM reads `beta`, so return the effective message whose one-slice
    # marginal is exactly the refined `gamma`. Zero where alpha is zero (no message to
    # reconstruct there), which is why `gamma == alpha*beta` holds only where `alpha > 0`.
    beta = np.zeros_like(gamma)
    positive = alpha > 0
    beta[positive] = gamma[positive] / alpha[positive]
    return gamma, beta
