"""Joint (Cartesian-product) state machinery shared by the exact-inference paths.

Joint state `j` for per-channel states `(m_0, ..., m_{C-1})` is the row-major
("C"-order) flattening of `(M_0, ..., M_{C-1})`: the last channel varies
fastest, matching MATLAB's `kron`-based construction. Also provides the size
guards for joint models and a textbook scaled forward-backward over a single
chain of joint states, used as the exact reference for the approximate LSIM
recursions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.validate import ValidationError


MAX_JOINT_STATES = 4096

MAX_JOINT_MIXTURE_COMPONENTS = 4096

MAX_ESTIMATED_BYTES = 512 * 1024 * 1024  # 512 MiB, before any of mu/sigma/weights/transition is allocated


class EquivalentModelTooLargeError(ValueError):
    """Raised by `build_equivalent_hmm` when `J = Π channel_state_num` or
    `K_joint = Π num_gmm_component` would exceed its own guard, or when the
    two together would still make `mu`/`sigma`/`weights`/`transition`
    multi-gigabyte (`MAX_ESTIMATED_BYTES`) even with both individually under
    their ceiling -- the equivalent HMM is only ever meant for small Layer-B
    verification models, not as a general inference path (that is exactly
    what P3a/P3b's approximate kernels are for)."""

def joint_state_index(states: NDArray[np.int64], dims: tuple[int, ...]) -> int:
    """The joint index of per-channel (0-based) states `states`, under the
    row-major/last-channel-fastest convention (see module docstring)."""
    return int(np.ravel_multi_index(tuple(int(s) for s in states), dims))

def joint_state_from_index(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of `joint_state_index`: the per-channel (0-based) states for
    one joint index."""
    return tuple(int(x) for x in np.unravel_index(index, dims))

def _cartesian_combine(arrays: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """`arrays[c]` is `(N, K_c)`; returns `(N, Π K_c)` where
    `result[n, j] = Π_c arrays[c][n, m_c(j)]`, `j`'s per-`c` component
    `m_c(j)` under the same row-major/last-fastest convention as everywhere
    else in this module (`np.ravel_multi_index`'s default order). Used both
    for a genuine batch (`N` joint states, combining per-channel *mixture*
    weights already gathered at each state) and, with `N=1`, for the single
    joint `pi_0` (a pure Cartesian product of per-channel vectors)."""
    result = arrays[0]
    for arr in arrays[1:]:
        result = (result[:, :, None] * arr[:, None, :]).reshape(result.shape[0], -1)
    return result

def scaled_forward_backward(
    pi0: NDArray[np.float64], transition: NDArray[np.float64], B: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Independent, textbook scaled HMM forward-backward (Rabiner's
    `alpha_hat`/`beta_hat` scaling) on a *flat* `N`-state chain -- no channel
    blocks, no coupling, no `inference/sed.py`/`inference/fast.py` code or
    helpers, by design (this is the Layer-B reference those must be checked
    against, so it must not share their recursion). `pi0` is `(N,)`,
    `transition` is `(N,N)` row-stochastic, `B` is `(N,T)` emission
    likelihoods (not necessarily normalized). Returns `(gamma, loglik)`,
    `gamma` `(N,T)` the smoothed posterior.

    Raises `ValidationError` -- never returns NaN, and never escapes as a
    raw `ValueError`/`IndexError` from unpacking or indexing a malformed
    array -- if: `B` isn't 2-D; `B` has `N=0` or `T=0` states/timesteps;
    `B`/`pi0`/`transition` contain a negative or non-finite entry; `pi0`/
    `transition` don't have the shapes `B` implies; or the scaling mass
    `c[t]` (`alpha_hat[:,t]`'s own sum, before it's normalized away) is ever
    zero or non-finite: that happens exactly when *no* state can explain the
    observations up to `t` at all (e.g. every state's emission density has
    underflowed to `0`) -- `alpha_hat[:,t] /= c[t]` would silently divide `0`
    by `0` into NaN otherwise, and every later `t` would inherit it silently.
    All of this is checked *before* `B.shape` is even unpacked, since a 1-D
    `B` can't be unpacked into `(N, T)` at all, and an `(N, 0)` `B` unpacks
    fine but then fails on `B[:, 0]` with a raw `IndexError` further down."""
    if B.ndim != 2:
        raise ValidationError(f"B.shape={B.shape}, expected 2-D (N, T)")
    N, T = B.shape
    if N < 1:
        raise ValidationError(f"B.shape={B.shape}: N=0, expected at least 1 state")
    if T < 1:
        raise ValidationError(f"B.shape={B.shape}: T=0, expected at least 1 timestep")

    def _check_probability_array(name: str, arr: NDArray[np.float64]) -> None:
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"{name} contains a non-finite (NaN/Inf) entry")
        if np.any(arr < 0):
            raise ValidationError(f"{name} contains a negative entry")

    _check_probability_array("B", B)

    if pi0.shape != (N,):
        raise ValidationError(f"pi0.shape={pi0.shape}, expected (N,)=({N},) to match B.shape={B.shape}")
    if transition.shape != (N, N):
        raise ValidationError(f"transition.shape={transition.shape}, expected (N,N)=({N},{N}) to match B.shape={B.shape}")
    _check_probability_array("pi0", pi0)
    _check_probability_array("transition", transition)

    alpha_hat = np.zeros((N, T))
    c = np.zeros(T)

    def _check_scaling_mass(t: int) -> None:
        if not np.isfinite(c[t]) or c[t] <= 0:
            raise ValidationError(
                f"scaled_forward_backward: scaling mass c[{t}]={c[t]} is zero or non-finite -- no state "
                f"can explain the observations up to t={t} (e.g. every emission density has underflowed "
                f"to 0); refusing to divide by it rather than silently producing NaN"
            )

    alpha_hat[:, 0] = pi0 * B[:, 0]
    c[0] = alpha_hat[:, 0].sum()
    _check_scaling_mass(0)
    alpha_hat[:, 0] /= c[0]
    for t in range(1, T):
        alpha_hat[:, t] = (transition.T @ alpha_hat[:, t - 1]) * B[:, t]
        c[t] = alpha_hat[:, t].sum()
        _check_scaling_mass(t)
        alpha_hat[:, t] /= c[t]

    beta_hat = np.zeros((N, T))
    beta_hat[:, T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta_hat[:, t] = transition @ (B[:, t + 1] * beta_hat[:, t + 1]) / c[t + 1]

    gamma = alpha_hat * beta_hat
    loglik = float(np.sum(np.log(c)))
    return gamma, loglik
