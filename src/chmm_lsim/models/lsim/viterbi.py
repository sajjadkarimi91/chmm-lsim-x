"""Viterbi decoding: the approximate per-channel algorithm actually present
in `viterbi_chmm.m` (`viterbi_lsim` here is a thin, packed-parameter API
around that algorithm -- **there is no `viterbi_lsim.m` in this repository**,
and this module does not claim to port one), plus an independent, exact
joint-state Viterbi built on `equivalent.py`'s Cartesian-product HMM, for
small models.

**`viterbi_chmm.m`'s own forward pass is shared with `sed.py`/`fast.py`,
bit-for-bit** (`Alpha`/`b_c_ot_nc` there are the same `W.T @ abar` recursion
as `sed.py`'s `_forward_pass`, using the same `pi_0` with no `pi0_reg`
regularization at all -- `viterbi_chmm.m` never calls anything like
`lsim_fb_constants`) -- reused via import, not re-derived.

**The backward/Viterbi recursion is a genuine per-channel approximation, not
exact joint decoding**, and its own score is named to say so.
`viterbi_chmm.m:225` maximizes, for every channel `z` independently, over
that channel's *own* previous state only; the contribution every *other*
channel makes to a candidate transition is a single aggregate scalar already
computed by the forward pass (`a_alpha_b(:,:,t-1) .* cuopling_repmat_zero`,
an alpha-weighted expectation over the other channels' own history, not a
tracked maximum) -- added to the self-transition probability *before* the
log, matching the LSIM transition law's own `Σ_c θ[c,z]·A[c,z][...]`
decomposition into a self term (`c=z`, tracked via this channel's Viterbi)
and an others term (`c≠z`, only ever marginalized, never re-maximized here).
So `viterbi_chmm.m`'s own `P_star_model` -- a *sum*, across channels, of each
channel's own independently-best final `log_Phi` -- is not the log-probability
of one coherent joint path; it is named `log_score` here, not `loglik` or
`log_prob`, and documented as such rather than left to be misread as one.

Vectorization note: `viterbi_chmm.m` computes this per-channel recursion with
MATLAB-array padding (`max(state_numbers)`-wide, `permute`/`reshape`/`squeeze`
scatter-gather) because MATLAB has no ragged arrays; this port uses
`LSIMStructure.state_slice` to work with each channel's own, unpadded
`(M_z, M_z)` block directly -- mathematically the same recursion, no padding
machinery to get bit-exactly right.

**Zero/underflowed emission blocks never produce a silent NaN, and the
error names exactly where.** If every state in some channel's block has
emission density `0` at a timestep (a real possibility with float64
densities on an outlying observation, not a contrived case -- linear-domain,
not log-domain, emissions are what P2/P3 actually implement until P8), the
shared forward pass's own rescaling (`b_c_ot_nc = B / P_ot_c_cond_past`)
would divide `0` by `0`. `viterbi_lsim` inspects each channel's emission
block directly, at every timestep, *before* the forward pass ever runs, and
raises a `ValidationError` naming the exact channel and timestep --
checking `B` itself this way (rather than only the forward pass's own
generic NaN-loglik check, which fires on *any* NaN anywhere without saying
where, and misses the case entirely when the underflow is only at the very
last timestep, since a single `-inf` term in a log-sum doesn't produce
`NaN`) is what makes the error message accurate rather than a coincidence
of where the NaN happened to first appear. This is a stopgap appropriate to
the current linear-domain emissions, not a long-term fix -- P8's planned
log-domain emissions avoid the underflow itself rather than merely
detecting it after the fact.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.numpy_ import build_W
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches, validate_observations
from chmm_lsim.core.viterbi import ViterbiResult
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.equivalent import EquivalentHMM, build_equivalent_hmm
from chmm_lsim.models.lsim.inference.sed import _forward_pass
from chmm_lsim.models.lsim.params import LSIMParams


def viterbi_lsim(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64], *, validate: str = "params") -> ViterbiResult:
    """`viterbi_chmm.m`, from `LSIMParams`/packed `Y` directly. Zero-based
    `states`. Handles `T=1` (the backward loop never runs, matching
    MATLAB's `for t=2:T` on `T=1`) and `C=1` (the "others" contribution is
    identically zero -- there are no other channels -- reducing exactly to
    single-chain Viterbi on that channel's own transition block) without
    special-casing either: both fall out of the same code path below."""
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    check_structure_matches(structure, params)
    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, _ = compute_emissions(structure, params, Y)
    S, T = B.shape
    C = structure.C

    # Detect a fully-zero or non-finite emission block for some channel at
    # some timestep *before* running the forward pass at all: viterbi_chmm.m's
    # own rescaling (b_c_ot_nc = B / P_ot_c_cond_past) divides 0 by 0 into NaN
    # whenever every state in a channel's block has emission density 0 (or
    # non-finite) at a timestep. Checking B directly, per channel and
    # timestep, lets this name exactly which channel/timestep is at fault --
    # the forward pass's own generic NaN-loglik check (below, kept as a
    # backstop for other causes) only fires once a NaN has already
    # propagated into a *later* timestep's running sum, and never fires at
    # all if the underflow is only at the very last timestep (loglik is
    # -inf there, not NaN, since just one term of the sum is affected).
    bad_blocks = []
    for c in range(C):
        sl_c = structure.state_slice(c)
        block = B[sl_c, :]  # (M_c, T)
        block_ok = np.isfinite(block).all(axis=0) & (block.sum(axis=0) > 0)
        for t in np.nonzero(~block_ok)[0]:
            bad_blocks.append((c, int(t)))
    if bad_blocks:
        bad_str = ", ".join(f"channel {c} at t={t}" for c, t in bad_blocks[:5])
        raise ValidationError(
            f"viterbi_lsim: the emission block underflowed to all-zero (or is non-finite) at "
            f"{bad_str}{', ...' if len(bad_blocks) > 5 else ''} -- no state in that channel can "
            f"explain the observation there (0/0 in the forward pass's own rescaling); refusing to "
            f"propagate a silent NaN (see module docstring)"
        )

    W = build_W(params.transition, params.coupling, structure.block_of_state)

    alpha, scale, _ = _forward_pass(params.pi_0, W, B, structure, validate=validate)
    b_c_ot_nc = scale  # viterbi_chmm.m's own name for this quantity
    if not np.all(np.isfinite(b_c_ot_nc)):
        bad_t = np.nonzero(~np.all(np.isfinite(b_c_ot_nc), axis=0))[0]
        raise ValidationError(
            f"viterbi_lsim: the forward pass's own rescaling underflowed to non-finite at "
            f"timestep(s) {bad_t.tolist()[:5]}{'...' if len(bad_t) > 5 else ''} for a reason other "
            f"than a single channel's emission block (already checked above) -- refusing to "
            f"propagate a silent NaN"
        )

    # a_alpha_b[j, c] = sum_{i in channel c} transition[i, j] * abar_prev[i] (viterbi_chmm.m:199,
    # shared with sed.py/fast.py's forward pass in spirit, computed fresh here per t since only
    # this module needs it laid out as (S, C) rather than folded into alpha's own recursion).
    channel_onehot = np.eye(C)[structure.block_of_state]  # (S, C)
    other_channel_mask = 1.0 - channel_onehot  # excludes c == block_of_state[j]

    log_phi = np.zeros((S, T))
    si = np.zeros((S, T), dtype=np.int64)  # local (within-channel) backpointer, 0-based

    with np.errstate(divide="ignore"):
        log_phi[:, 0] = np.log(b_c_ot_nc[:, 0] * params.pi_0)

        for t in range(1, T):
            abar_prev = alpha[:, t - 1] * b_c_ot_nc[:, t - 1]
            a_alpha_b = np.zeros((S, C))
            for c in range(C):
                sl_c = structure.state_slice(c)
                a_alpha_b[:, c] = params.transition[sl_c, :].T @ abar_prev[sl_c]
            other_contrib = (a_alpha_b * other_channel_mask * params.coupling[:, structure.block_of_state].T).sum(axis=1)

            for z in range(C):
                sl_z = structure.state_slice(z)
                M_z = structure.channel_state_num[z]
                self_trans = params.coupling[z, z] * params.transition[sl_z, sl_z].T  # (M_z, M_z): [i_next, i_prev]
                candidate = np.log(self_trans + other_contrib[sl_z][:, None]) + log_phi[sl_z, t - 1][None, :]
                best = candidate.max(axis=1)
                si[sl_z, t] = candidate.argmax(axis=1)
                log_phi[sl_z, t] = np.log(b_c_ot_nc[sl_z, t]) + best

    states = np.zeros((C, T), dtype=np.int64)
    log_score = 0.0
    for c in range(C):
        sl_c = structure.state_slice(c)
        best_local = int(np.argmax(log_phi[sl_c, T - 1]))
        log_score += float(log_phi[sl_c, T - 1][best_local])
        states[c, T - 1] = best_local

    for t in range(T - 2, -1, -1):
        for c in range(C):
            sl_c = structure.state_slice(c)
            states[c, t] = si[sl_c, t + 1][states[c, t + 1]]

    return ViterbiResult(states=states, log_score=log_score)

def viterbi_exact(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64], *, validate: bool = True) -> ViterbiResult:
    """Exact joint-state Viterbi (standard log-domain recursion, no
    approximation) on `equivalent.py`'s Cartesian-product HMM -- only
    feasible for small models, bounded by `equivalent.build_equivalent_hmm`'s
    own `MAX_JOINT_STATES`/`MAX_JOINT_MIXTURE_COMPONENTS` guards.

    Always builds its own `EquivalentHMM` from `structure`/`params` --
    there is deliberately no `equiv=` parameter here, so `params` is always
    the one and only source of truth for the model actually decoded. (An
    earlier version accepted an optional pre-built `equiv` to let a caller
    that already has one, e.g. also running `equivalent.exact_posterior`,
    avoid reconstructing it -- but `equiv` and `params` can each encode a
    *different* parameter value for the same `structure`, and only
    `structure`, not the actual parameter values, was ever compared between
    them; a stale or mismatched `equiv` was silently preferred over `params`
    with no error. Callers who already have an `equiv` and want to avoid
    rebuilding it should call `viterbi_exact_from_equiv` directly instead --
    that function takes no `params` at all, so there is no second source of
    truth to go stale.)"""
    check_structure_matches(structure, params)
    if validate:
        validate_observations(structure, Y)
    equiv = build_equivalent_hmm(structure, params, validate=validate)
    return _viterbi_exact_core(equiv, Y)

def viterbi_exact_from_equiv(equiv: EquivalentHMM, Y: NDArray[np.float64], *, validate: bool = True) -> ViterbiResult:
    """Exact joint-state Viterbi directly from a pre-built `EquivalentHMM`,
    with no `structure`/`params` argument at all -- `equiv` is unambiguously
    the only model in play, so there is no way to pass a second, conflicting
    parameter source (contrast `viterbi_exact`'s own docstring, which
    documents exactly the bug this split avoids). Use this to avoid
    rebuilding `equiv` when the caller already has one on hand, e.g.
    alongside `equivalent.exact_posterior(equiv, Y)` on the same model.
    `validate=True` (the default) still runs `validate_observations` against
    `equiv.structure`; it does not re-validate `equiv` itself (that already
    happened, if at all, when `equiv` was built)."""
    if validate:
        validate_observations(equiv.structure, Y)
    return _viterbi_exact_core(equiv, Y)

def _viterbi_exact_core(equiv: EquivalentHMM, Y: NDArray[np.float64]) -> ViterbiResult:
    joint_structure, joint_params = equiv.as_single_channel_params()
    B, _ = compute_emissions(joint_structure, joint_params, Y)
    J, T = B.shape

    with np.errstate(divide="ignore"):
        log_pi0 = np.log(equiv.pi0)
        log_A = np.log(equiv.transition)  # (J,J): [prev, next]
        log_B = np.log(B)

        delta = np.zeros((J, T))
        psi = np.zeros((J, T), dtype=np.int64)
        delta[:, 0] = log_pi0 + log_B[:, 0]
        for t in range(1, T):
            candidate = delta[:, t - 1][:, None] + log_A  # (J_prev, J_next)
            psi[:, t] = candidate.argmax(axis=0)
            delta[:, t] = candidate.max(axis=0) + log_B[:, t]

    best_final = int(np.argmax(delta[:, T - 1]))
    log_score = float(delta[best_final, T - 1])

    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = best_final
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]

    states_per_dim = np.array(np.unravel_index(path, equiv.dims))  # (C, T)
    return ViterbiResult(states=states_per_dim, log_score=log_score)
