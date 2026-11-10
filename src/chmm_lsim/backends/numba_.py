"""Numba execution backend (P7, optional `numba` extra): `njit(cache=True)`-
compiled structured forward/backward recursions for both the `sed` and
`fast` inference variants, unbatched, float64-only, single-sequence
`Posterior` mode. Mirrors `inference/sed.py`/`inference/fast.py`'s NumPy
algorithms operation-for-operation -- same summation grouping wherever
numba's own explicit-loop semantics allow it to be preserved -- not a
re-derivation; cross-checked against every committed posterior fixture (both
variants, every trial) in `tests/test_backends_numba.py`.

**`nogil=True` (added for P10b).** Every compiled kernel below is called
directly from Python (`_forward_backward_sed_numba`/`_forward_backward_fast_
numba`), so it is the GIL-release boundary that matters for `em.e_step`'s
`executor="thread"` path: without `nogil=True`, numba's default behavior
holds the GIL for the kernel's entire duration, so multiple Python threads
each running one trial through this backend would serialize on the GIL
exactly as if `n_jobs=1` -- concurrency in name only. `nogil=True` changes
nothing about the compiled numerics (same machine code either way; the flag
only controls whether the calling thread holds the GIL while it runs), so
it does not affect this file's own existing fixture-parity guarantees.

**Optional, lazy dependency.** `import chmm_lsim` never imports this module --
only `backends.base.get_backend("numba")`/`run_backend("numba", ...)` do,
lazily, the first time a caller actually requests `backend="numba"` -- so a
core-only install (no `numba`) stays fully functional. If `numba` isn't
installed, importing this module raises `MissingBackendDependency` (an
`ImportError` subclass) with an actionable install instruction, rather than
a bare, unexplained `ModuleNotFoundError`.

**Scope (P7).** `sed` and `fast` variants, `float64` only, unbatched single-
sequence `Posterior` mode, no GPU, no sequence batching. `sed`'s
`method="reference"` is *not* implemented here -- only `"optimized"` (the
default `P_cond` construction, §0.2's `O(S^2)` elementwise form, not the
`(S,S)@(S,S)` matrix product `"reference"` performs) -- declared via
`BackendCapabilities.sed_methods=("optimized",)`, so requesting
`method="reference"` on this backend raises `UnsupportedCombination` from
`backends.base.check_capability` before this module's own code ever runs;
NumPy already exists as the numerically-identical (P3a-verified) way to run
`"reference"` if that specific matrix-product form is ever needed.

**No allocations inside the timestep loop.** Every array a timestep needs
(`abar`, `p_cond`, `Gz`, `beta_C`, `f`, `h`, `delta`, `weights_subsystems`,
`v2` (the `validate="all"`-only workspace), ...) is a workspace allocated
once, before the `for t`/`for tau` loop begins, and overwritten in place at
every step -- not a fresh `np.zeros(...)` call per iteration the way the
vectorized NumPy code naturally allocates one. `sed`'s backward pass is
still, unavoidably, an `O(S^2)` step per timestep (`P_cond` genuinely is an
`(S,S)` quantity -- see `inference/sed.py`'s own module docstring for why),
but it is *one* `(S,S)` workspace reused `T-1` times, not `T-1` fresh
allocations. (A review round found `v2` itself allocated fresh inside the
`tau` loop, contradicting this guarantee -- fixed by hoisting it next to the
other workspaces above.)

**`validate="all"` is not a no-op here.** The compiled kernels themselves
check per-block sums of `alpha`/`abar`/`gamma` (`rtol=0, atol=1e-8`, the
same tolerance as `inference.sed._assert_close_to_one`), `sed`'s own
`V2=1` invariant, and `f`/`h` finiteness and `f>h` -- returning a compact
integer failure code and the offending timestep (a numba `njit` function
cannot itself raise a rich Python exception object usefully; returning a
code and having the *wrapper* raise `ValidationError` is the documented
escape hatch) rather than silently skipping the checks or returning NaN.

**Scalar division follows NumPy semantics, not Python's.** Every `@njit`
here is compiled with `error_model="numpy"`. Numba's *default* error model
(`"python"`) makes float division by zero raise a raw `ZeroDivisionError`,
matching plain Python `/` -- but `inference.sed`/`inference.fast`'s own
NumPy code relies on IEEE-754 float division never raising at all (`0/0`
-> `nan`, `x/0` -> `+-inf`), e.g. when every state's emission density
underflows to `0` at the final timestep (`scale = B / p_ot_c_cond_past`
becomes `0/0`) -- a real, previously-observed scenario, not hypothetical: a
review round confirmed NumPy returns `loglik=-inf` with a non-finite
`scale`/`beta`/`gamma` there (the deliberate, established, documented
behavior for that case -- see `_forward_pass`'s own module-level discussion
in `inference/sed.py` -- which this backend must reproduce, not "fix"),
while the *default*-error-model numba kernel instead raised a raw
`ZeroDivisionError` from deep inside a scalar division with no useful
message. `error_model="numpy"` applies to every division in every kernel
here -- `_forward_pass_kernel`'s `scale` computation, `_solve_delta_kernel`'s
Sherman-Morrison solve (`f/d`, `1/d` -- `d` can be `~0` for a
sufficiently pathological `f_reg`/coupling, though never for valid
parameters), and both backward kernels' own normalizer divisions -- not
merely the one call site the underflow scenario above happens to exercise
first."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import BackendCapabilities, MissingBackendDependency, register_backend

try:
    from numba import njit
except ImportError as exc:
    raise MissingBackendDependency(
        "the 'numba' execution backend requires the optional numba dependency -- install it via "
        "`pip install 'chmm-lsim[numba]'` (tested with numba==0.66.0 on Python 3.13.5; numba's own PyPI "
        "metadata declares support for Python 3.10-3.14, covering this project's 3.12-3.14 target)"
    ) from exc

from chmm_lsim.backends.numpy_ import build_W
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.inference.base import Diagnostics, Posterior
from chmm_lsim.models.lsim.inference.poise import _REFINE_W0 as _POISE_REFINE_W0
from chmm_lsim.models.lsim.inference.poise import poise_backward_constants
from chmm_lsim.models.lsim.inference.sed import _regularize_pi0
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches, validate_observations

_BLOCK_SUM_ATOL = 1e-8  # matches inference.sed._assert_close_to_one (rtol=0)

# Failure codes the compiled kernels return for validate="all" -- translated
# into a chmm_lsim.validate.ValidationError by the Python wrapper, with wording
# mirroring inference.sed's own messages for the same checks (see
# _raise_for_fail_code below).
_OK = 0
_FAIL_ALPHA_BLOCK_SUM = 1
_FAIL_ABAR_BLOCK_SUM = 2
_FAIL_GAMMA_BLOCK_SUM = 3
_FAIL_V2_BLOCK_SUM = 4
_FAIL_F_H_NONFINITE = 5
_FAIL_F_LE_H = 6
_FAIL_POISE_NORMALIZER = 7
_FAIL_POISE_REFINE_NORMALIZER = 8


def _raise_for_fail_code(fail_code: int, fail_t: int) -> None:
    if fail_code == _OK:
        return
    messages = {
        _FAIL_ALPHA_BLOCK_SUM: f"alpha at t={fail_t} per-block sums are not all 1.0 (or contain NaN/Inf) (§0.2.1)",
        _FAIL_ABAR_BLOCK_SUM: f"abar at t={fail_t} per-block sums are not all 1.0 (or contain NaN/Inf) (§0.2.1)",
        _FAIL_GAMMA_BLOCK_SUM: f"gamma at t={fail_t} per-block sums are not all 1.0 (or contain NaN/Inf) (§0.2.1)",
        _FAIL_V2_BLOCK_SUM: f"V2 per-block sums at t={fail_t} are not all 1.0 (or contain NaN/Inf)",
        _FAIL_F_H_NONFINITE: f"f or h at t={fail_t} contains NaN/Inf",
        _FAIL_F_LE_H: f"f<=h at t={fail_t}: this is unreachable for valid parameters (§0.3.3) -- check coupling column-stochasticity and transition row-stochasticity",
    }
    raise ValidationError(f"[backend=numba] {messages[fail_code]}")


def _raise_for_poise_fail_code(fail_code: int, fail_t: int, fail_c: int) -> None:
    """POISE's own two failures additionally name the channel, because the quantity that
    failed is per channel (see `inference/poise.check_channel_normalizer`)."""
    if fail_code == _OK:
        return
    stage = {
        _FAIL_POISE_NORMALIZER: "backward recursion",
        _FAIL_POISE_REFINE_NORMALIZER: "refinement",
    }.get(fail_code)
    if stage is None:
        _raise_for_fail_code(fail_code, fail_t)
        return
    raise ValidationError(
        f"[backend=numba] POISE {stage}: channel {fail_c} normalizer at t={fail_t} is "
        f"nonpositive or non-finite; a channel's total smoothed mass must be finite and "
        f"positive. Dividing by it would return NaN posteriors from a model whose forward "
        f"log-likelihood is finite. This is MATLAB forward_backward_lsim_poise.m's own "
        f"'nonpositive or nonfinite channel normalizer' error."
    )


@njit(cache=True, error_model="numpy", nogil=True)
def _check_block_sum(x: NDArray[np.float64], state_offsets: NDArray[np.int64]) -> bool:
    """`True` iff every per-channel block of `x` is finite and sums to `1.0`
    within `_BLOCK_SUM_ATOL` (`rtol=0`) -- the compiled equivalent of
    `inference.sed._assert_block_sum_one`, returning a bool instead of
    raising (njit functions can't raise a Python `ValidationError` usefully)."""
    C = state_offsets.shape[0] - 1
    for c in range(C):
        s = 0.0
        for i in range(state_offsets[c], state_offsets[c + 1]):
            v = x[i]
            if not np.isfinite(v):
                return False
            s += v
        if abs(s - 1.0) > _BLOCK_SUM_ATOL:
            return False
    return True


@njit(cache=True, error_model="numpy", nogil=True)
def _check_all_close_to_one(x: NDArray[np.float64]) -> bool:
    """`True` iff every entry of `x` is finite and within `_BLOCK_SUM_ATOL`
    (`rtol=0`) of `1.0` -- the compiled equivalent of
    `inference.sed._assert_close_to_one`, for values that are *already*
    per-channel scalars (sed's `V2`), as opposed to `_check_block_sum`'s
    per-state array that still needs reducing into blocks first."""
    for i in range(x.shape[0]):
        v = x[i]
        if not np.isfinite(v) or abs(v - 1.0) > _BLOCK_SUM_ATOL:
            return False
    return True


@njit(cache=True, error_model="numpy", nogil=True)
def _forward_pass_kernel(
    pi_0: NDArray[np.float64],
    W: NDArray[np.float64],
    B: NDArray[np.float64],
    state_offsets: NDArray[np.int64],
    block_of_state: NDArray[np.int64],
    check_all: bool,
):
    """`inference.sed._forward_pass`, compiled: `alpha[:,t] = W.T @ abar`
    (an explicit matvec, not a full `(S,S)` temporary), `p_ot_c_cond_past`
    accumulated per channel block, `scale = B / broadcast(p_ot_c_cond_past)`.
    `abar` is a single `(S,)` workspace reused every `t`, not reallocated."""
    S, T = B.shape
    C = state_offsets.shape[0] - 1
    alpha = np.zeros((S, T))
    scale = np.zeros((S, T))
    p_ot_c = np.zeros((C, T))
    fail_code = _OK
    fail_t = -1

    for i in range(S):
        alpha[i, 0] = pi_0[i]
    for c in range(C):
        s = 0.0
        for i in range(state_offsets[c], state_offsets[c + 1]):
            s += alpha[i, 0] * B[i, 0]
        p_ot_c[c, 0] = s
    for i in range(S):
        scale[i, 0] = B[i, 0] / p_ot_c[block_of_state[i], 0]

    if check_all and not _check_block_sum(alpha[:, 0], state_offsets):
        fail_code = _FAIL_ALPHA_BLOCK_SUM
        fail_t = 0

    abar = np.zeros(S)
    if fail_code == _OK:
        for t in range(1, T):
            for i in range(S):
                abar[i] = alpha[i, t - 1] * scale[i, t - 1]
            for j in range(S):
                s = 0.0
                for i in range(S):
                    s += W[i, j] * abar[i]
                alpha[j, t] = s
            for c in range(C):
                s = 0.0
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    s += alpha[i, t] * B[i, t]
                p_ot_c[c, t] = s
            for i in range(S):
                scale[i, t] = B[i, t] / p_ot_c[block_of_state[i], t]

            if check_all and not _check_block_sum(alpha[:, t], state_offsets):
                fail_code = _FAIL_ALPHA_BLOCK_SUM
                fail_t = t
                break

    return alpha, scale, p_ot_c, fail_code, fail_t


@njit(cache=True, error_model="numpy", nogil=True)
def _solve_delta_kernel(f: NDArray[np.float64], h: NDArray[np.float64], delta_out: NDArray[np.float64]) -> None:
    """`inference.sed._solve_delta`, compiled: row-wise Sherman-Morrison
    solve of `(diag(f_z-h_z) + h_z*ones(C,C)) @ x_z = f_z`, for every channel
    `z` at once. Writes into the caller's `delta_out` workspace in place."""
    C = h.shape[0]
    for z in range(C):
        s1 = 0.0
        s2 = 0.0
        for w in range(C):
            d = f[z, w] - h[z]
            s1 += f[z, w] / d
            s2 += 1.0 / d
        g = h[z] * s1 / (1.0 + h[z] * s2)
        for w in range(C):
            d = f[z, w] - h[z]
            delta_out[z, w] = (f[z, w] - g) / d


@njit(cache=True, error_model="numpy", nogil=True)
def _backward_fast_kernel(
    alpha: NDArray[np.float64],
    scale: NDArray[np.float64],
    transition: NDArray[np.float64],
    state_offsets: NDArray[np.int64],
    weights_per_state: NDArray[np.float64],
    w_opt_per_state: NDArray[np.float64],
    check_all: bool,
):
    """`inference.fast.forward_backward_fast`'s backward loop, compiled.
    `beta_C` (`(S,C)`) is the one per-`tau` workspace; no `(S,S)` temporary
    is ever formed (`transition` itself, built once outside this kernel, is
    reused directly)."""
    S, T = alpha.shape
    C = state_offsets.shape[0] - 1
    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    nan_guard_hits = 0
    fail_code = _OK
    fail_t = -1

    for i in range(S):
        beta[i, T - 1] = scale[i, T - 1]
        gamma[i, T - 1] = alpha[i, T - 1] * beta[i, T - 1]
    if check_all and not _check_block_sum(gamma[:, T - 1], state_offsets):
        fail_code = _FAIL_GAMMA_BLOCK_SUM
        fail_t = T - 1

    beta_C = np.zeros((S, C))
    abar = np.zeros(S)

    if fail_code == _OK:
        for tau in range(T - 2, -1, -1):
            if check_all:
                for i in range(S):
                    abar[i] = alpha[i, tau] * scale[i, tau]
                if not _check_block_sum(abar, state_offsets):
                    fail_code = _FAIL_ABAR_BLOCK_SUM
                    fail_t = tau
                    break

            for i in range(S):
                for c in range(C):
                    s = 0.0
                    for j in range(state_offsets[c], state_offsets[c + 1]):
                        s += transition[i, j] * beta[j, tau + 1]
                    beta_C[i, c] = s
            for i in range(S):
                mix = 0.0
                for c in range(C):
                    mix += beta_C[i, c] * weights_per_state[i, c]
                beta_tau_i = scale[i, tau] * mix ** w_opt_per_state[i]
                if np.isnan(beta_tau_i):
                    nan_guard_hits += 1
                    beta_tau_i = scale[i, tau]
                beta[i, tau] = beta_tau_i
            for c in range(C):
                s = 0.0
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    s += beta[i, tau] * alpha[i, tau]
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    beta[i, tau] = beta[i, tau] / s
                    gamma[i, tau] = beta[i, tau] * alpha[i, tau]

            if check_all and not _check_block_sum(gamma[:, tau], state_offsets):
                fail_code = _FAIL_GAMMA_BLOCK_SUM
                fail_t = tau
                break

    return beta, gamma, nan_guard_hits, fail_code, fail_t


@njit(cache=True, error_model="numpy", nogil=True)
def _backward_poise_kernel(
    alpha: NDArray[np.float64],
    scale: NDArray[np.float64],
    transition: NDArray[np.float64],
    W: NDArray[np.float64],
    state_offsets: NDArray[np.int64],
    rho_per_state: NDArray[np.float64],
    theta_out_per_state: NDArray[np.float64],
    refinement: bool,
    refine_w0: float,
    check_all: bool,
):
    """`inference.poise.forward_backward_poise`'s backward loop and its optional
    one-shot refinement, compiled. Like the fast kernel, `child` (`(S,C)`) is the
    only per-`tau` workspace -- POISE forms no `(S,S)` message either. The
    refinement needs `W` (the forward's own transition/coupling product, built
    once outside this kernel) and keeps the raw marginals frozen in `gamma_raw`,
    exactly as the MATLAB kernel does."""
    S, T = alpha.shape
    C = state_offsets.shape[0] - 1
    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    fail_code = _OK
    fail_t = -1
    fail_c = -1

    for i in range(S):
        beta[i, T - 1] = scale[i, T - 1]
        gamma[i, T - 1] = alpha[i, T - 1] * beta[i, T - 1]
    if check_all and not _check_block_sum(gamma[:, T - 1], state_offsets):
        fail_code = _FAIL_GAMMA_BLOCK_SUM
        fail_t = T - 1

    child = np.zeros((S, C))
    abar = np.zeros(S)

    if fail_code == _OK:
        for tau in range(T - 2, -1, -1):
            if check_all:
                for i in range(S):
                    abar[i] = alpha[i, tau] * scale[i, tau]
                if not _check_block_sum(abar, state_offsets):
                    fail_code = _FAIL_ABAR_BLOCK_SUM
                    fail_t = tau
                    break

            for i in range(S):
                for c in range(C):
                    acc = 0.0
                    for j in range(state_offsets[c], state_offsets[c + 1]):
                        acc += transition[i, j] * beta[j, tau + 1]
                    child[i, c] = acc
            for i in range(S):
                q = 0.0
                for c in range(C):
                    q += child[i, c] * rho_per_state[i, c]
                if theta_out_per_state[i] == 0.0:
                    beta[i, tau] = scale[i, tau]  # the exact-zero outgoing-strength branch
                else:
                    beta[i, tau] = scale[i, tau] * q ** theta_out_per_state[i]
            # alpha_T from the UNNORMALIZED beta, then both divided by the same
            # per-channel normalizer (the MATLAB POISE branch's operation order).
            for c in range(C):
                acc = 0.0
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    gamma[i, tau] = beta[i, tau] * alpha[i, tau]
                    acc += gamma[i, tau]
                # MATLAB's normalize_blocks guard: never divide by a nonpositive or
                # non-finite channel normalizer (unconditional, as MATLAB's is).
                if not (acc > 0.0) or np.isinf(acc):
                    fail_code = _FAIL_POISE_NORMALIZER
                    fail_t = tau
                    fail_c = c
                    break
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    beta[i, tau] = beta[i, tau] / acc
                    gamma[i, tau] = gamma[i, tau] / acc
            if fail_code != _OK:
                break

            if check_all and not _check_block_sum(gamma[:, tau], state_offsets):
                fail_code = _FAIL_GAMMA_BLOCK_SUM
                fail_t = tau
                break

    if fail_code == _OK and refinement and T > 1:
        gamma_raw = gamma.copy()  # frozen: every propagated slice reads these, never a refined one
        propagated = np.zeros(S)
        for t in range(1, T):
            for i in range(S):
                acc = 0.0
                for j in range(S):
                    acc += gamma_raw[j, t - 1] * W[j, i]
                propagated[i] = scale[i, t] * acc
            for c in range(C):
                acc = 0.0
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    acc += propagated[i]
                if not (acc > 0.0) or np.isinf(acc):
                    fail_code = _FAIL_POISE_REFINE_NORMALIZER
                    fail_t = t
                    fail_c = c
                    break
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    propagated[i] = propagated[i] / acc
            if fail_code != _OK:
                break
            for i in range(S):
                w1 = refine_w0 - theta_out_per_state[i]
                if w1 < 0.0:
                    w1 = 0.0
                gamma[i, t] = w1 * propagated[i] + (1.0 - w1) * gamma_raw[i, t]

            if check_all and not _check_block_sum(gamma[:, t], state_offsets):
                fail_code = _FAIL_GAMMA_BLOCK_SUM
                fail_t = t
                break

        # effective message whose one-slice marginal is exactly the refined gamma
        for i in range(S):
            for t in range(T):
                if alpha[i, t] > 0.0:
                    beta[i, t] = gamma[i, t] / alpha[i, t]
                else:
                    beta[i, t] = 0.0

    return beta, gamma, fail_code, fail_t, fail_c


@njit(cache=True, error_model="numpy", nogil=True)
def _backward_sed_kernel(
    alpha: NDArray[np.float64],
    scale: NDArray[np.float64],
    W: NDArray[np.float64],
    state_offsets: NDArray[np.int64],
    block_of_state: NDArray[np.int64],
    beta_floor: float,
    f_reg: float,
    flag_nan: bool,
    check_all: bool,
):
    """`inference.sed.forward_backward_sed`'s backward loop, `method=
    "optimized"` only, compiled: `p_cond = W + alpha_next - broadcast(Gz)`,
    `f`/`h`/Sherman-Morrison `delta` solve for `C>1`. `p_cond`/`Gz`/`beta_C`/
    `f`/`h`/`delta`/`weights_subsystems`/`abar` are all workspaces allocated
    once, before the `tau` loop, and overwritten in place every iteration --
    `p_cond` in particular is genuinely `(S,S)` (§0.2's own construction),
    but is one allocation reused `T-1` times, not `T-1` allocations."""
    S, T = alpha.shape
    C = state_offsets.shape[0] - 1
    beta = np.zeros((S, T))
    gamma = np.zeros((S, T))
    beta_floor_hits = 0
    nan_guard_hits = 0
    zero_alpha_next_count = 0
    f_le_h_violations = 0
    fail_code = _OK
    fail_t = -1

    for i in range(S):
        beta[i, T - 1] = scale[i, T - 1]
        gamma[i, T - 1] = alpha[i, T - 1] * beta[i, T - 1]
    if check_all and not _check_block_sum(gamma[:, T - 1], state_offsets):
        fail_code = _FAIL_GAMMA_BLOCK_SUM
        fail_t = T - 1

    p_cond = np.zeros((S, S))
    Gz = np.zeros((C, S))
    beta_C = np.zeros((S, C))
    f = np.zeros((C, C))
    h = np.zeros(C)
    delta = np.zeros((C, C))
    weights_subsystems = np.zeros((S, C))
    abar = np.zeros(S)
    v2 = np.zeros(C)  # validate="all"-only workspace, allocated once here rather than fresh on every tau (see module docstring's "no allocations inside the timestep loop")

    if fail_code == _OK:
        for tau in range(T - 2, -1, -1):
            for i in range(S):
                abar[i] = alpha[i, tau] * scale[i, tau]

            if check_all:
                if not _check_block_sum(abar, state_offsets):
                    fail_code = _FAIL_ABAR_BLOCK_SUM
                    fail_t = tau
                    break
                for c in range(C):
                    s = 0.0
                    for i in range(state_offsets[c], state_offsets[c + 1]):
                        s += alpha[i, tau + 1] * beta[i, tau + 1]
                    v2[c] = s
                if not _check_all_close_to_one(v2):
                    fail_code = _FAIL_V2_BLOCK_SUM
                    fail_t = tau + 1
                    break

            for i in range(S):
                if alpha[i, tau + 1] == 0.0:
                    zero_alpha_next_count += 1

            for c in range(C):
                for j in range(S):
                    s = 0.0
                    for i in range(state_offsets[c], state_offsets[c + 1]):
                        s += abar[i] * W[i, j]
                    Gz[c, j] = s

            for i in range(S):
                c_i = block_of_state[i]
                for j in range(S):
                    p_cond[i, j] = W[i, j] + alpha[j, tau + 1] - Gz[c_i, j]

            for i in range(S):
                for c in range(C):
                    s = 0.0
                    for j in range(state_offsets[c], state_offsets[c + 1]):
                        s += p_cond[i, j] * beta[j, tau + 1]
                    beta_C[i, c] = s

            if C > 1:
                for z in range(C):
                    for w in range(C):
                        s = 0.0
                        for i in range(state_offsets[z], state_offsets[z + 1]):
                            for j in range(state_offsets[w], state_offsets[w + 1]):
                                an = alpha[j, tau + 1]
                                if an != 0.0:
                                    val = p_cond[i, j] * abar[i]
                                    s += (val * val) / an
                        f[z, w] = s + f_reg
                for z in range(C):
                    s = 0.0
                    for i in range(state_offsets[z], state_offsets[z + 1]):
                        s += abar[i] * abar[i]
                    h[z] = s

                if check_all:
                    all_finite = True
                    for z in range(C):
                        if not np.isfinite(h[z]):
                            all_finite = False
                        for w in range(C):
                            if not np.isfinite(f[z, w]):
                                all_finite = False
                    if not all_finite:
                        fail_code = _FAIL_F_H_NONFINITE
                        fail_t = tau
                        break
                    violated = False
                    for z in range(C):
                        for w in range(C):
                            if f[z, w] <= h[z]:
                                violated = True
                    if violated:
                        fail_code = _FAIL_F_LE_H
                        fail_t = tau
                        break

                viol_count = 0
                for z in range(C):
                    for w in range(C):
                        if f[z, w] <= h[z]:
                            viol_count += 1
                f_le_h_violations += viol_count

                _solve_delta_kernel(f, h, delta)
                for i in range(S):
                    c_i = block_of_state[i]
                    for c in range(C):
                        weights_subsystems[i, c] = delta[c_i, c]
            else:
                for i in range(S):
                    weights_subsystems[i, 0] = 1.0

            for i in range(S):
                raw = 0.0
                for c in range(C):
                    raw += beta_C[i, c] * weights_subsystems[i, c]
                if raw < beta_floor:
                    beta_floor_hits += 1
                    raw = beta_floor
                beta_tau_i = scale[i, tau] * raw
                if flag_nan and np.isnan(beta_tau_i):
                    nan_guard_hits += 1
                    beta_tau_i = scale[i, tau]
                beta[i, tau] = beta_tau_i
                gamma[i, tau] = beta_tau_i * alpha[i, tau]

            for c in range(C):
                s = 0.0
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    s += gamma[i, tau]
                for i in range(state_offsets[c], state_offsets[c + 1]):
                    beta[i, tau] = beta[i, tau] / s
                    gamma[i, tau] = gamma[i, tau] / s

            if check_all and not _check_block_sum(gamma[:, tau], state_offsets):
                fail_code = _FAIL_GAMMA_BLOCK_SUM
                fail_t = tau
                break

    return beta, gamma, beta_floor_hits, nan_guard_hits, zero_alpha_next_count, f_le_h_violations, fail_code, fail_t


def _forward_backward_sed_numba(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants,
    *,
    validate: str = "params",
) -> Posterior:
    check_structure_matches(structure, params)
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, B_comp = compute_emissions(structure, params, Y)
    W = build_W(params.transition, params.coupling, structure.block_of_state)
    flag_nan = bool(np.any(params.coupling.sum(axis=1) <= 1e-10))
    pi_0 = _regularize_pi0(params.pi_0, structure, fb_constants.pi0_reg)
    check_all = validate == "all"

    alpha, scale, p_ot_c, fail_code, fail_t = _forward_pass_kernel(
        pi_0, np.ascontiguousarray(W), np.ascontiguousarray(B), structure.state_offsets, structure.block_of_state, check_all
    )
    _raise_for_fail_code(fail_code, fail_t)
    loglik = float(np.sum(np.log(p_ot_c)))
    if np.isnan(loglik):
        raise ValidationError("[backend=numba] forward pass produced a NaN log-likelihood (P_O_model)")

    beta, gamma, beta_floor_hits, nan_guard_hits, zero_alpha_next_count, f_le_h_violations, fail_code, fail_t = _backward_sed_kernel(
        alpha,
        scale,
        np.ascontiguousarray(W),
        structure.state_offsets,
        structure.block_of_state,
        fb_constants.beta_floor,
        fb_constants.f_reg,
        flag_nan,
        check_all,
    )
    _raise_for_fail_code(fail_code, fail_t)

    diagnostics = Diagnostics(
        zero_alpha_next_count=zero_alpha_next_count,
        beta_floor_hits=beta_floor_hits,
        f_le_h_violations=f_le_h_violations,
        nan_guard_hits=nan_guard_hits,
    )
    return Posterior(
        loglik=loglik, alpha=alpha, beta=beta, gamma=gamma, scale=scale, B=B, B_comp=B_comp,
        log_scale=None, diagnostics=diagnostics, structure=structure,
    )


def _forward_backward_fast_numba(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants,
    *,
    validate: str = "params",
) -> Posterior:
    check_structure_matches(structure, params)
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, B_comp = compute_emissions(structure, params, Y)
    W = build_W(params.transition, params.coupling, structure.block_of_state)
    pi_0 = _regularize_pi0(params.pi_0, structure, fb_constants.pi0_reg)
    check_all = validate == "all"

    alpha, scale, p_ot_c, fail_code, fail_t = _forward_pass_kernel(
        pi_0, np.ascontiguousarray(W), np.ascontiguousarray(B), structure.state_offsets, structure.block_of_state, check_all
    )
    _raise_for_fail_code(fail_code, fail_t)
    loglik = float(np.sum(np.log(p_ot_c)))
    if np.isnan(loglik):
        raise ValidationError("[backend=numba] forward pass produced a NaN log-likelihood (P_O_model)")

    C = structure.C
    row_sum = params.coupling.sum(axis=1)
    weights = params.coupling / (fb_constants.svi_eps + row_sum[:, None])
    weights = np.where(np.isnan(weights), 1.0 / C, weights)
    w_opt = np.sum(fb_constants.svi_eps + params.coupling, axis=1)
    weights_per_state = np.ascontiguousarray(weights[structure.block_of_state])
    w_opt_per_state = np.ascontiguousarray(w_opt[structure.block_of_state])

    beta, gamma, nan_guard_hits, fail_code, fail_t = _backward_fast_kernel(
        alpha, scale, np.ascontiguousarray(params.transition), structure.state_offsets, weights_per_state, w_opt_per_state, check_all
    )
    _raise_for_fail_code(fail_code, fail_t)

    diagnostics = Diagnostics(nan_guard_hits=nan_guard_hits)
    return Posterior(
        loglik=loglik, alpha=alpha, beta=beta, gamma=gamma, scale=scale, B=B, B_comp=B_comp,
        log_scale=None, diagnostics=diagnostics, structure=structure,
    )


def _forward_backward_poise_numba(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants,
    *,
    validate: str = "params",
) -> Posterior:
    check_structure_matches(structure, params)
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, B_comp = compute_emissions(structure, params, Y)
    W = build_W(params.transition, params.coupling, structure.block_of_state)
    pi_0 = _regularize_pi0(params.pi_0, structure, fb_constants.pi0_reg)
    check_all = validate == "all"

    alpha, scale, p_ot_c, fail_code, fail_t = _forward_pass_kernel(
        pi_0, np.ascontiguousarray(W), np.ascontiguousarray(B), structure.state_offsets, structure.block_of_state, check_all
    )
    _raise_for_fail_code(fail_code, fail_t)
    loglik = float(np.sum(np.log(p_ot_c)))
    if np.isnan(loglik):
        raise ValidationError("[backend=numba] forward pass produced a NaN log-likelihood (P_O_model)")

    # the same shared constants helper the numpy kernel uses, so the two cannot drift
    rho_per_state, theta_out_per_state, _zero = poise_backward_constants(structure, params.coupling)

    beta, gamma, fail_code, fail_t, fail_c = _backward_poise_kernel(
        alpha, scale, np.ascontiguousarray(params.transition), np.ascontiguousarray(W), structure.state_offsets,
        np.ascontiguousarray(rho_per_state), np.ascontiguousarray(theta_out_per_state),
        bool(fb_constants.poise_refinement), _POISE_REFINE_W0, check_all,
    )
    _raise_for_poise_fail_code(fail_code, fail_t, fail_c)

    # POISE has no beta_floor clamp, no delta solve and no NaN guard to count (see
    # inference/poise.py) -- its Diagnostics are all zero by construction.
    return Posterior(
        loglik=loglik, alpha=alpha, beta=beta, gamma=gamma, scale=scale, B=B, B_comp=B_comp,
        log_scale=None, diagnostics=Diagnostics(), structure=structure,
    )


def _run_numba(variant, structure, params, Y, fb_constants, *, validate="params", dtype="float64", **kwargs):
    # `dtype` is forwarded uniformly by `run_backend` (P8a: torch needs it);
    # already validated as `"float64"` (this backend's only declared dtype)
    # by `check_capability` before this runner is ever reached, so accepting
    # and discarding it here is a no-op.
    kwargs.pop("method", None)  # already checked against sed_methods=("optimized",) by check_capability; only "optimized" is ever implemented
    if kwargs:
        raise TypeError(f"forward_backward (backend='numba') got unexpected keyword arguments: {sorted(kwargs)}")
    if variant == "poise":
        return _forward_backward_poise_numba(structure, params, Y, fb_constants, validate=validate)
    if variant == "sed":
        return _forward_backward_sed_numba(structure, params, Y, fb_constants, validate=validate)
    elif variant == "fast":
        return _forward_backward_fast_numba(structure, params, Y, fb_constants, validate=validate)
    raise ValueError(f"backend='numba' runner got unexpected variant {variant!r}")  # unreachable: check_capability already filtered this


def _loglik_numba(variant, structure, params, Y, fb_constants, *, validate="params"):
    """Forward-only log-likelihood: exactly the first half of
    `_forward_backward_sed_numba`/`_forward_backward_fast_numba` (their
    forward halves are identical), stopping before the backward kernel.
    Bit-identical to those runners' own `.loglik`."""
    if fb_constants.variant != variant:
        raise ValueError(f"loglik runner requires fb_constants.variant={variant!r}, got {fb_constants.variant!r}")
    check_structure_matches(structure, params)
    if validate not in ("params", "off"):
        # "all" is served by the full posterior runner (backends.base.run_backend_loglik)
        raise ValueError(f"forward-only loglik supports validate='params' or 'off', got {validate!r}")
    if validate != "off":
        params.validate()
        validate_observations(structure, Y)

    B, _ = compute_emissions(structure, params, Y)
    W = build_W(params.transition, params.coupling, structure.block_of_state)
    pi_0 = _regularize_pi0(params.pi_0, structure, fb_constants.pi0_reg)
    _, _, p_ot_c, fail_code, fail_t = _forward_pass_kernel(
        pi_0, np.ascontiguousarray(W), np.ascontiguousarray(B), structure.state_offsets, structure.block_of_state, False
    )
    _raise_for_fail_code(fail_code, fail_t)
    loglik = float(np.sum(np.log(p_ot_c)))
    if np.isnan(loglik):
        raise ValidationError("[backend=numba] forward pass produced a NaN log-likelihood (P_O_model)")
    return loglik


register_backend(
    BackendCapabilities(
        name="numba",
        variants=("sed", "fast", "poise"),
        dtypes=("float64",),
        modes=("posterior", "loglik"),
        batched=False,
        gpu=False,
        sed_methods=("optimized",),
        emission_models=("gmm",),
    ),
    _run_numba,
    loglik_runner=_loglik_numba,
)
