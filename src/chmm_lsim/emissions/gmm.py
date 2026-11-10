"""Gaussian-mixture emissions (`ChannelGMM`), the built-in emission model.

Each channel has, per hidden state, a mixture of `K_c` Gaussians with
diagonal or full covariances (one covariance type for the whole model).
This module holds everything specific to that family: the densities
(`gmm_pdf_diag`/`gmm_pdf_full`, ports of `gmm_pdf_fast.m`/`gmm_pdf.m`), the
mergeable EM sufficient statistics and the emission M-step, parameter
validation, sampling (`gmm_gen.m`), the free-parameter count, and the
`GaussianMixtureEmissions` model registered as ``"gmm"``.

NaN in an observation means "missing": a missing dimension contributes zero
log-density (it is marginalized out), and a channel missing in every
dimension at a time step has likelihood 1 for every state.

The EM accumulators are central moments, never raw `sum(y)`/`sum(y**2)`
moments: running `(weight, mean, M2)` triples (Welford/Chan) for diagonal
channels and `(Q, M, C)` pairwise triples for full-covariance channels, merged
with the parallel-variance formula, so precision stays proportional to the
data's own scale rather than its offset from zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from chmm_lsim.core.sampling import select_p
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import _EIG_REL_TOL, _SUM_ATOL, ValidationError, _check_no_nan_inf, _iter_validated_observation_chunks
from chmm_lsim.emissions.base import EmissionModel, register_emission_model

if TYPE_CHECKING:
    from chmm_lsim.learning.em_loop import ExpectedStats
    from chmm_lsim.models.lsim.params import LSIMParams


_LOG_2PI = float(np.log(2 * np.pi))


# --------------------------------------------------------------------------- parameters


@dataclass
class ChannelGMM:
    """One channel's GMM emission model, unpadded: exactly (M_c, K_c) etc,
    never `(max_M, max_K)`. See §2.1's "deviation from MATLAB, deliberate"."""

    weights: NDArray[np.float64]  # (M_c, K_c)
    mu: NDArray[np.float64]  # (d_c, M_c, K_c)
    sigma: NDArray[np.float64]  # (d_c, M_c, K_c) diag, or (d_c, d_c, M_c, K_c) full
    diag: bool

    def __post_init__(self) -> None:
        M_c, K_c = self.weights.shape
        if self.mu.shape[1:] != (M_c, K_c):
            raise ValueError(f"mu.shape={self.mu.shape} inconsistent with weights.shape={self.weights.shape}")
        d_c = self.mu.shape[0]
        expected_sigma_shape = (d_c, M_c, K_c) if self.diag else (d_c, d_c, M_c, K_c)
        if self.sigma.shape != expected_sigma_shape:
            raise ValueError(f"sigma.shape={self.sigma.shape} != {expected_sigma_shape} (diag={self.diag})")

    @property
    def d_c(self) -> int:
        return self.mu.shape[0]

    @property
    def M_c(self) -> int:
        return self.weights.shape[0]

    @property
    def K_c(self) -> int:
        return self.weights.shape[1]

    def copy(self) -> ChannelGMM:
        return ChannelGMM(self.weights.copy(), self.mu.copy(), self.sigma.copy(), self.diag)


def _check_gmm_shapes(structure: LSIMStructure, emissions: list[ChannelGMM]) -> None:
    """Raises `ValueError` unless `emissions` has one GMM per channel, each of
    the channel's own `(M_c, K_c, d_c)`, all with the same covariance type."""
    C = structure.C
    if len(emissions) != C:
        raise ValueError(f"len(emissions)={len(emissions)} != C={C}")
    for c, gmm in enumerate(emissions):
        if gmm.M_c != structure.channel_state_num[c]:
            raise ValueError(f"emissions[{c}].M_c={gmm.M_c} != channel_state_num[{c}]={structure.channel_state_num[c]}")
        if gmm.d_c != structure.channel_obs_dim[c]:
            raise ValueError(f"emissions[{c}].d_c={gmm.d_c} != channel_obs_dim[{c}]={structure.channel_obs_dim[c]}")
        if gmm.K_c != structure.num_gmm_component[c]:
            raise ValueError(f"emissions[{c}].K_c={gmm.K_c} != num_gmm_component[{c}]={structure.num_gmm_component[c]}")
    if any(gmm.diag != emissions[0].diag for gmm in emissions):
        raise ValueError("all channels must agree on diagonal-vs-full covariance (packed.sigma_diag is model-wide, not per-channel)")


# --------------------------------------------------------------------------- densities


def gmm_pdf_diag(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Diagonal-covariance emissions. Returns (B (S,T), B_comp (S,max_K,T))."""
    if not params.sigma_diag:
        raise ValueError("gmm_pdf_diag requires params.sigma_diag=True; use gmm_pdf_full for full covariance")
    return _gmm_pdf_diag(structure, params.emissions, Y)


def _gmm_pdf_diag(structure: LSIMStructure, emissions: list, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    T = Y.shape[1]
    B_comp = np.zeros((structure.S, structure.max_K, T))

    for c in range(structure.C):
        gmm = emissions[c]
        Yc = Y[structure.dim_slice(c), :]  # (d_c, T)
        nan_mask = np.isnan(Yc)
        s0 = structure.state_offsets[c]
        for s in range(gmm.M_c):
            for k in range(gmm.K_c):
                mu = gmm.mu[:, s, k][:, None]  # (d_c, 1)
                sigma = gmm.sigma[:, s, k][:, None]  # (d_c, 1) variances
                quad = (Yc - mu) ** 2 / sigma
                log_sqrt_det = 0.5 * np.log(sigma)
                log_dens = -0.5 * quad - log_sqrt_det - 0.5 * _LOG_2PI  # (d_c, T)
                log_dens = np.where(nan_mask, 0.0, log_dens)
                loglik = log_dens.sum(axis=0)  # (T,)
                B_comp[s0 + s, k, :] = gmm.weights[s, k] * np.exp(loglik)

    B = B_comp.sum(axis=1)
    return B, B_comp


def gmm_pdf_full(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Full-covariance emissions. Returns (B (S,T), B_comp (S,max_K,T))."""
    if params.sigma_diag:
        raise ValueError("gmm_pdf_full requires params.sigma_diag=False; use gmm_pdf_diag for diagonal covariance")
    return _gmm_pdf_full(structure, params.emissions, Y)


def _gmm_pdf_full(structure: LSIMStructure, emissions: list, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    T = Y.shape[1]
    B_comp = np.zeros((structure.S, structure.max_K, T))

    for c in range(structure.C):
        gmm: ChannelGMM = emissions[c]
        Yc = Y[structure.dim_slice(c), :].T  # (T, d_c)
        nan_mask = np.isnan(Yc)
        patterns, inverse = np.unique(nan_mask, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        s0 = structure.state_offsets[c]

        for s in range(gmm.M_c):
            for k in range(gmm.K_c):
                mu = gmm.mu[:, s, k]
                sigma = gmm.sigma[:, :, s, k]
                comp_vals = np.empty(T)
                for p_idx in range(patterns.shape[0]):
                    rows = inverse == p_idx
                    missing = patterns[p_idx]
                    observed = ~missing
                    if not observed.any():
                        comp_vals[rows] = 1.0
                        continue
                    X = Yc[np.ix_(rows, observed)]
                    mu_sub = mu[observed]
                    sigma_sub = sigma[np.ix_(observed, observed)]
                    comp_vals[rows] = multivariate_normal.pdf(X, mean=mu_sub, cov=sigma_sub, allow_singular=True)
                B_comp[s0 + s, k, :] = gmm.weights[s, k] * comp_vals

    B = B_comp.sum(axis=1)
    return B, B_comp


# --------------------------------------------------------------------------- validation


def _validate_gmm_emissions(emissions, *, matlab_compat: bool = True) -> None:
    """Per-channel GMM checks shared by LSIM and CHMM parameters: finite
    values, mixture weights that are non-negative and sum to one per state,
    and valid covariances (strictly positive variances, or symmetric
    positive (semi-)definite matrices -- see `_validate_full_sigma`)."""
    for c, gmm in enumerate(emissions):
        _check_no_nan_inf(f"emissions[{c}].weights", gmm.weights)
        _check_no_nan_inf(f"emissions[{c}].mu", gmm.mu)
        _check_no_nan_inf(f"emissions[{c}].sigma", gmm.sigma)

        if (gmm.weights < 0).any():
            raise ValidationError(f"emissions[{c}].weights has negative entries")
        weight_sums = gmm.weights.sum(axis=1)
        if not np.allclose(weight_sums, 1.0, rtol=0, atol=_SUM_ATOL):
            raise ValidationError(
                f"emissions[{c}].weights rows (per state) sum to {weight_sums.tolist()}, expected all 1.0"
            )

        if gmm.diag:
            if (gmm.sigma <= 0).any():
                raise ValidationError(f"emissions[{c}].sigma (diagonal) must be strictly positive")
        else:
            _validate_full_sigma(c, gmm, matlab_compat=matlab_compat)


def _validate_full_sigma(c: int, gmm, *, matlab_compat: bool) -> None:
    """A genuinely negative eigenvalue means no Gaussian density exists at all
    -- MATLAB's own `mvnpdf` errors on that matrix too (§0.4.10's `eig(...)<=0`
    threshold is the M-step's *repair* decision, not a claim that gmm_pdf can
    evaluate an indefinite covariance; a fixture with one would simply fail to
    export). So a negative
    eigenvalue is rejected unconditionally. What `matlab_compat` actually gates
    is whether a numerically-singular-but-PSD matrix (min eigenvalue within
    `_EIG_REL_TOL` of zero, e.g. the near-singular `c3_sigma_repair_*` fixtures) is
    tolerated -- `gmm_pdf_full`'s `allow_singular=True` can genuinely evaluate
    that case; `matlab_compat=False` asks for strict positive-definiteness."""
    d_c, _, M_c, K_c = gmm.sigma.shape
    for s in range(M_c):
        for k in range(K_c):
            sigma = gmm.sigma[:, :, s, k]
            if not np.allclose(sigma, sigma.T, rtol=0, atol=_SUM_ATOL):
                raise ValidationError(f"emissions[{c}].sigma[:,:,{s},{k}] is not symmetric")
            eigvals = np.linalg.eigvalsh(sigma)
            eig_tol = _EIG_REL_TOL * max(1.0, float(np.abs(eigvals).max()))
            min_eig = float(eigvals.min())
            if min_eig < -eig_tol:
                raise ValidationError(
                    f"emissions[{c}].sigma[:,:,{s},{k}] has a negative eigenvalue {min_eig} "
                    f"-- no valid Gaussian density exists (not a matlab_compat question)"
                )
            if min_eig < eig_tol:
                if not matlab_compat:
                    raise ValidationError(
                        f"emissions[{c}].sigma[:,:,{s},{k}] is numerically singular (min eigenvalue {min_eig}); "
                        f"matlab_compat=False requires strict positive-definiteness"
                    )


# --------------------------------------------------------------------------- sampling


def _symmetric_sqrt(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """`Sigma^0.5` via symmetric eigendecomposition, matching MATLAB's own
    `^0.5` operator on a symmetric matrix -- unlike `np.linalg.cholesky`,
    this is defined for *singular* PSD matrices too (a rank-deficient
    covariance, e.g. from a degenerate EM fit or a hand-built test case).
    `params.validate(matlab_compat=True)` already accepts a covariance whose
    smallest eigenvalue is merely close to zero (within `_EIG_REL_TOL`,
    §0.4.10) but rejects anything genuinely negative-definite beyond that
    tolerance (`_validate_full_sigma` above), so any negative
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


# --------------------------------------------------------------------------- EM statistics


_EPS = float(np.finfo(np.float64).eps)

_LOW_MASS_GAMMA_THRESHOLD = 1e-8

_LOW_MASS_STATE_COUNT = 3

_SIGMA_RIDGE = 0.001


@dataclass
class ChannelExpectedStats:
    """Raw, mergeable sufficient statistics for one channel's GMM M-step
    (§2.4.2). Every field is either a plain sum or a central-moment triple
    combined via the mergeable parallel-variance formula -- never a raw
    `Σy`/`Σy²` moment -- so this object's size is independent of `T_all` and
    it can be combined with another chunk's via `merge_channel_stats`.
    `nanvar_y`, `fallback_mu` and `fallback_sigma_full` are `None` until
    `finalize_channel_stats` runs -- they are *derived*, not accumulated.

    **Diagonal channels** (`sigma_diag=True`) use the per-`(state,component,
    dim)` univariate triples `weighted_W`/`weighted_mean`/`weighted_M2`
    (weight `= γ·o`); `pairwise_*` fields are `None`.

    **Full-covariance channels** use the per-`(state,component)` pairwise
    tensors `pairwise_Q`/`pairwise_M`/`pairwise_C` (weight `= γ`,
    co-observation mask `o_i·o_j`) for *everything*, including `mu` (its
    diagonal `M[i,i]` is exactly the univariate weighted mean); `weighted_*`
    fields are `None`.

    Both share `n_obs`/`welford_mean`/`welford_M2` (observed-only, per
    channel dimension -- feeds `nanvar_y`'s `0.001·var` ridge and the
    `matlab_compat=False` fallback denominator) and `allzero_n`/
    `allzero_mean`/`allzero_M2` (over *every* `T_all` timestep with missing
    values zero-filled, weight 1 -- feeds the canonical `matlab_compat=True`
    fallback, matching `em_lsim.m`'s literal "add 1 to every cell after
    zero-filling"). Full covariance additionally carries the *unweighted*
    pairwise tensor `u_pairwise_Q`/`u_pairwise_M`/`u_pairwise_C` for its own
    independent fallback (§0.4.2's denominator disagreement; the denominator
    itself stays `resp + T_all`, unmasked, matching canonical MATLAB, not
    `u_pairwise_Q`).
    """

    resp: NDArray[np.float64]  # (M_c,K_c) = Σ_t γ[s,k,t]
    resp_pos: NDArray[np.int64]  # (M_c,K_c) = #{t: γ[s,k,t] > 1e-8}
    n_w_pos_diag: NDArray[np.int64]  # (M_c,K_c) = #{(dim,t): γ·o > 0}
    n_w_pos_full: NDArray[np.int64]  # (M_c,K_c) = d_c * #{t: γ > 0}

    # diagonal-only (None for full-covariance channels):
    weighted_W: NDArray[np.float64] | None  # (M_c,K_c,d_c) = Σ_t γ·o
    weighted_mean: NDArray[np.float64] | None  # (M_c,K_c,d_c)
    weighted_M2: NDArray[np.float64] | None  # (M_c,K_c,d_c) = Σ_t γ·o·(y-mean)²

    # full-covariance-only (None for diagonal channels):
    pairwise_Q: NDArray[np.float64] | None  # (M_c,K_c,d_c,d_c) = Σ_t γ·o_i·o_j
    pairwise_M: NDArray[np.float64] | None  # (M_c,K_c,d_c,d_c) weighted pairwise mean of y_i under mask (i,j)
    pairwise_C: NDArray[np.float64] | None  # (M_c,K_c,d_c,d_c) weighted pairwise central cross-moment
    u_pairwise_Q: NDArray[np.float64] | None  # (d_c,d_c) = Σ_t o_i·o_j, unweighted, per channel
    u_pairwise_M: NDArray[np.float64] | None  # (d_c,d_c)
    u_pairwise_C: NDArray[np.float64] | None  # (d_c,d_c)

    # shared (both diag and full), per channel:
    n_obs: NDArray[np.float64]  # (d_c,) = Σ_t o -- observed-timestep count
    welford_mean: NDArray[np.float64]  # (d_c,) running mean of observed y, parallel-mergeable
    welford_M2: NDArray[np.float64]  # (d_c,) running Σ(y-mean)² of observed y, parallel-mergeable
    allzero_n: NDArray[np.float64]  # (d_c,) == running T_all (same for every dim; kept per-dim for uniform merging)
    allzero_mean: NDArray[np.float64]  # (d_c,) running mean over every T_all timestep, missing=0
    allzero_M2: NDArray[np.float64]  # (d_c,) running Σ(y_zerofilled-mean)² over every T_all timestep

    # derived -- None until finalize_channel_stats:
    nanvar_y: NDArray[np.float64] | None = None  # (d_c,) sample variance, omitnan, ddof=1, MATLAB's n=1->0 convention
    fallback_mu: NDArray[np.bool_] | None = None  # (M_c,K_c)
    fallback_sigma_full: NDArray[np.bool_] | None = None  # (M_c,K_c), None when diagonal

def _empty_channel_stats(M_c: int, K_c: int, d_c: int, sigma_diag: bool) -> ChannelExpectedStats:
    pair_shape = (M_c, K_c, d_c, d_c)
    return ChannelExpectedStats(
        resp=np.zeros((M_c, K_c)),
        resp_pos=np.zeros((M_c, K_c), dtype=np.int64),
        n_w_pos_diag=np.zeros((M_c, K_c), dtype=np.int64),
        n_w_pos_full=np.zeros((M_c, K_c), dtype=np.int64),
        weighted_W=np.zeros((M_c, K_c, d_c)) if sigma_diag else None,
        weighted_mean=np.zeros((M_c, K_c, d_c)) if sigma_diag else None,
        weighted_M2=np.zeros((M_c, K_c, d_c)) if sigma_diag else None,
        pairwise_Q=None if sigma_diag else np.zeros(pair_shape),
        pairwise_M=None if sigma_diag else np.zeros(pair_shape),
        pairwise_C=None if sigma_diag else np.zeros(pair_shape),
        u_pairwise_Q=None if sigma_diag else np.zeros((d_c, d_c)),
        u_pairwise_M=None if sigma_diag else np.zeros((d_c, d_c)),
        u_pairwise_C=None if sigma_diag else np.zeros((d_c, d_c)),
        n_obs=np.zeros(d_c),
        welford_mean=np.zeros(d_c),
        welford_M2=np.zeros(d_c),
        allzero_n=np.zeros(d_c),
        allzero_mean=np.zeros(d_c),
        allzero_M2=np.zeros(d_c),
    )

def _welford_batch(weight: NDArray[np.float64], value: NDArray[np.float64]) -> tuple[NDArray, NDArray, NDArray]:
    """One batch's `(W, mean, M2)` along the last axis, for a general
    non-negative `weight` (a boolean/0-1 mask is the special case of an
    observed-count accumulator; `weight≡1` is the special case of a plain,
    unweighted accumulator over every column). `value` must already be
    NaN-free (zero-filled at any position `weight` doesn't count) --
    `weight * value` with a real `NaN` stays `NaN` even at zero weight."""
    W = weight.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.divide((weight * value).sum(axis=-1), W, out=np.zeros_like(W), where=W > 0)
    diff = value - mean[..., None]
    M2 = np.sum(weight * diff**2, axis=-1)
    return W, mean, M2

def _welford_merge(
    W_a: NDArray, mean_a: NDArray, M2_a: NDArray, W_b: NDArray, mean_b: NDArray, M2_b: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Chan et al.'s parallel-variance merge, vectorized -- the mergeable form
    of Welford's algorithm, avoiding the Σy² cancellation a naive
    `Σy² - (Σy)²/n` formula suffers when `y` is far from zero. `W` is a
    general weight (a count is the special case of `weight≡1`)."""
    W = W_a + W_b
    delta = mean_b - mean_a
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = mean_a + np.divide(delta * W_b, W, out=np.zeros_like(W, dtype=np.float64), where=W > 0)
        M2 = M2_a + M2_b + np.divide(delta**2 * W_a * W_b, W, out=np.zeros_like(W, dtype=np.float64), where=W > 0)
    return W, mean, M2

def _pairwise_batch(
    weight: NDArray[np.float64], o: NDArray[np.bool_], Y_masked: NDArray[np.float64]
) -> tuple[NDArray, NDArray, NDArray]:
    """One batch's `(Q, M, C)` pairwise triple (the bivariate generalization
    of `_welford_batch`): `Q[i,j] = Σ_t weight(t)·o_i(t)·o_j(t)`, `M[i,j]` the
    weighted mean of `y_i` under that mask, `C[i,j] = Σ_t
    weight·o_i·o_j·(y_i-M[i,j])·(y_j-M[j,i])` -- computed by directly
    recentering (never forming a `y_i·y_j` raw moment), fully vectorized via
    a `(d,d,T)` broadcast rather than a `(d,d)` Python loop."""
    o_f = o.astype(np.float64)
    wo = weight[None, :] * o_f  # (d,T)
    Q = wo @ o_f.T  # (d,d)
    num_M = (wo * Y_masked) @ o_f.T  # (d,d)
    M = np.divide(num_M, Q, out=np.zeros_like(Q), where=Q > 0)
    Y_diff = Y_masked[:, None, :] - M[:, :, None]  # (d,d,T): [i,j,t] = y_i(t) - M[i,j]
    Y_diff_partner = np.transpose(Y_diff, (1, 0, 2))  # [i,j,t] = y_j(t) - M[j,i]
    weight_full = weight[None, None, :] * o_f[:, None, :] * o_f[None, :, :]  # (d,d,T)
    C = np.sum(weight_full * Y_diff * Y_diff_partner, axis=2)
    return Q, M, C

def _pairwise_merge(
    Q_a: NDArray, M_a: NDArray, C_a: NDArray, Q_b: NDArray, M_b: NDArray, C_b: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Bivariate parallel merge of two `(Q,M,C)` triples. `Q[i,j]` merges as
    a plain sum; `M[i,j]` merges as an ordinary univariate weighted mean
    (weight `Q[i,j]`); `C[i,j]` needs *both* deltas -- `M_b[i,j]-M_a[i,j]`
    (this pair's own mean-of-`i`) and `M_b[j,i]-M_a[j,i]` (the mean-of-`j`
    stored at the transposed entry, since mask `o_i·o_j` is symmetric).

    Works both on a single `(d,d)` pair-tensor and on a batched
    `(...,d,d)` one (`merge_channel_stats` merges all `(state,component)`
    pairs at once) -- `.swapaxes(-1,-2)`, not `.T`, since plain `.T` on an
    N-d array reverses *every* axis, not just the last two.
    """
    Q = Q_a + Q_b
    delta_i = M_b - M_a
    delta_j = delta_i.swapaxes(-1, -2)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = M_a + np.divide(delta_i * Q_b, Q, out=np.zeros_like(Q), where=Q > 0)
        C = C_a + C_b + np.divide(delta_i * delta_j * Q_a * Q_b, Q, out=np.zeros_like(Q), where=Q > 0)
    return Q, M, C

def _accumulate_channel_trial(
    stats: ChannelExpectedStats, structure: LSIMStructure, zee: int, post, Y_trial: NDArray[np.float64], sigma_diag: bool
) -> None:
    """Adds one trial's contribution to `stats`, in place."""
    zee_sl = structure.state_slice(zee)
    dim_sl = structure.dim_slice(zee)
    M_zee = structure.channel_state_num[zee]
    K_zee = structure.num_gmm_component[zee]

    Y_c = Y_trial[dim_sl, :]  # (d_c, T)
    o = ~np.isnan(Y_c)  # (d_c, T)
    o_f = o.astype(np.float64)
    Y_masked = np.where(o, Y_c, 0.0)
    T = Y_c.shape[1]

    obs_W, obs_mean, obs_M2 = _welford_batch(o_f, Y_masked)
    stats.n_obs, stats.welford_mean, stats.welford_M2 = _welford_merge(
        stats.n_obs, stats.welford_mean, stats.welford_M2, obs_W, obs_mean, obs_M2
    )

    allzero_W, allzero_mean, allzero_M2 = _welford_batch(np.ones_like(Y_masked), Y_masked)
    stats.allzero_n, stats.allzero_mean, stats.allzero_M2 = _welford_merge(
        stats.allzero_n, stats.allzero_mean, stats.allzero_M2, allzero_W, allzero_mean, allzero_M2
    )

    if not sigma_diag:
        u_Q, u_M, u_C = _pairwise_batch(np.ones(T), o, Y_masked)
        stats.u_pairwise_Q, stats.u_pairwise_M, stats.u_pairwise_C = _pairwise_merge(
            stats.u_pairwise_Q, stats.u_pairwise_M, stats.u_pairwise_C, u_Q, u_M, u_C
        )

    for i in range(M_zee):
        state_idx = zee_sl.start + i
        gamma_state = post.gamma[state_idx, :]
        B_state = post.B[state_idx, :]
        for k in range(K_zee):
            ratio = np.divide(
                post.B_comp[state_idx, k, :], B_state, out=np.full_like(B_state, np.nan), where=B_state != 0
            )
            g = gamma_state * ratio
            g = np.where(np.isnan(g), 0.0, g)  # (T,)

            stats.resp[i, k] += np.sum(g)
            stats.resp_pos[i, k] += int(np.sum(g > _LOW_MASS_GAMMA_THRESHOLD))
            stats.n_w_pos_diag[i, k] += int(np.sum(o & (g[None, :] > 0)))
            stats.n_w_pos_full[i, k] += Y_c.shape[0] * int(np.sum(g > 0))

            if sigma_diag:
                gW, gmean, gM2 = _welford_batch(g[None, :] * o_f, Y_masked)
                stats.weighted_W[i, k], stats.weighted_mean[i, k], stats.weighted_M2[i, k] = _welford_merge(
                    stats.weighted_W[i, k], stats.weighted_mean[i, k], stats.weighted_M2[i, k], gW, gmean, gM2
                )
            else:
                gQ, gM, gC = _pairwise_batch(g, o, Y_masked)
                stats.pairwise_Q[i, k], stats.pairwise_M[i, k], stats.pairwise_C[i, k] = _pairwise_merge(
                    stats.pairwise_Q[i, k], stats.pairwise_M[i, k], stats.pairwise_C[i, k], gQ, gM, gC
                )

def _merge_field(chunks: list, get, *, pairwise: bool) -> tuple:
    """Reduces `[get(c) for c in chunks]` (each a `(W,mean,M2)` triple, or --
    when `pairwise=True` -- a `(Q,M,C)` triple) with the appropriate merge,
    left to right. `pairwise` is passed explicitly rather than guessed from
    shape: a `(M_c,K_c,d_c)` univariate triple and a `(d_c,d_c)` pairwise one
    aren't reliably distinguishable by shape alone (`K_c` can equal `d_c`)."""
    triples = [get(c) for c in chunks]
    W, mean, M2 = triples[0]
    merge = _pairwise_merge if pairwise else _welford_merge
    for w, m, m2 in triples[1:]:
        W, mean, M2 = merge(W, mean, M2, w, m, m2)
    return W, mean, M2

def merge_channel_stats(chunks: list[ChannelExpectedStats]) -> ChannelExpectedStats:
    """Combines raw (unfinalized) per-channel stats from disjoint chunks of
    trials. Raises if any chunk was already finalized -- merging derived
    fields (a mean, a boolean trigger) by summing them would be wrong."""
    if any(c.nanvar_y is not None for c in chunks):
        raise ValueError("merge_channel_stats requires raw (unfinalized) stats -- finalize only after merging")
    sigma_diag = chunks[0].weighted_W is not None

    n_obs, w_mean, w_M2 = _merge_field(chunks, lambda c: (c.n_obs, c.welford_mean, c.welford_M2), pairwise=False)
    az_n, az_mean, az_M2 = _merge_field(chunks, lambda c: (c.allzero_n, c.allzero_mean, c.allzero_M2), pairwise=False)

    if sigma_diag:
        weighted_W, weighted_mean, weighted_M2 = _merge_field(
            chunks, lambda c: (c.weighted_W, c.weighted_mean, c.weighted_M2), pairwise=False
        )
        pairwise_Q = pairwise_M = pairwise_C = None
        u_Q = u_M = u_C = None
    else:
        weighted_W = weighted_mean = weighted_M2 = None
        pairwise_Q, pairwise_M, pairwise_C = _merge_field(
            chunks, lambda c: (c.pairwise_Q, c.pairwise_M, c.pairwise_C), pairwise=True
        )
        u_Q, u_M, u_C = _merge_field(chunks, lambda c: (c.u_pairwise_Q, c.u_pairwise_M, c.u_pairwise_C), pairwise=True)

    return ChannelExpectedStats(
        resp=sum(c.resp for c in chunks),
        resp_pos=sum(c.resp_pos for c in chunks),
        n_w_pos_diag=sum(c.n_w_pos_diag for c in chunks),
        n_w_pos_full=sum(c.n_w_pos_full for c in chunks),
        weighted_W=weighted_W,
        weighted_mean=weighted_mean,
        weighted_M2=weighted_M2,
        pairwise_Q=pairwise_Q,
        pairwise_M=pairwise_M,
        pairwise_C=pairwise_C,
        u_pairwise_Q=u_Q,
        u_pairwise_M=u_M,
        u_pairwise_C=u_C,
        n_obs=n_obs,
        welford_mean=w_mean,
        welford_M2=w_M2,
        allzero_n=az_n,
        allzero_mean=az_mean,
        allzero_M2=az_M2,
    )

def _nanvar_from_welford(n_obs: NDArray[np.float64], welford_M2: NDArray[np.float64]) -> NDArray[np.float64]:
    """MATLAB's `var(y, [], 2, 'omitnan')` from a merged observed-only
    Welford `(n_obs, welford_M2)` pair: sample variance (`ddof=1`) where
    `n_obs>1`, `0.0` (not `NaN`) where `n_obs==1` (MATLAB's own convention,
    confirmed directly against MATLAB: `var([5],[],2,'omitnan')==0`), and
    `NaN` where `n_obs==0`. Shared by `finalize_channel_stats` (from a
    mini-batch/full-dataset E-step's own accumulators) and `compute_global_
    nanvar` (accumulated directly from `Y_trials`, no E-step at all) -- one
    formula, not two copies that could drift apart."""
    nanvar_y = np.full_like(n_obs, np.nan)
    nanvar_y = np.where(n_obs > 1, np.divide(welford_M2, np.maximum(n_obs - 1, 1)), nanvar_y)
    nanvar_y = np.where(n_obs == 1, 0.0, nanvar_y)
    return nanvar_y

def finalize_channel_stats(stats: ChannelExpectedStats, *, sigma_diag: bool) -> ChannelExpectedStats:
    """Computes `nanvar_y` (from the merged observed-only Welford state,
    MATLAB's `n_obs==1 -> 0` convention, `n_obs==0 -> NaN`) and the two
    independent fallback flags (§0.4.3), once, after all merging. Returns a
    new object rather than mutating -- a finalized `ChannelExpectedStats` is
    a distinct state from a raw one (`merge_channel_stats` rejects the latter)."""
    n_obs = stats.n_obs
    nanvar_y = _nanvar_from_welford(n_obs, stats.welford_M2)

    d_c = n_obs.shape[0]
    fallback_mu = (stats.resp_pos <= _LOW_MASS_STATE_COUNT) | (stats.n_w_pos_diag < 2 * d_c)
    fallback_sigma_full = None if sigma_diag else ((stats.resp_pos <= _LOW_MASS_STATE_COUNT) | (stats.n_w_pos_full < 2 * d_c))

    return dataclasses_replace(stats, nanvar_y=nanvar_y, fallback_mu=fallback_mu, fallback_sigma_full=fallback_sigma_full)


# --------------------------------------------------------------------------- M-step


def repair_covariance(sig_temp: NDArray[np.float64], ridge_diag: NDArray[np.float64], *, matlab_compat: bool) -> NDArray[np.float64]:
    """§0.4.10 / quirk 10: `em_lsim.m:388`'s `if sum(eig(sig_temp)<=0)>1`
    repairs only when *at least two* eigenvalues are non-positive, leaving a
    singular or indefinite covariance in place when exactly one is --
    `matlab_compat=True` reproduces that threshold exactly; `matlab_compat=False`
    is the corrected `>0` (repairs on any non-positive eigenvalue). Symmetrizes
    first, matching `sig_temp = (sig_temp+sig_temp')/2` in the same line."""
    sig_temp = (sig_temp + sig_temp.T) / 2.0
    eigvals = np.linalg.eigvalsh(sig_temp)
    threshold = 1 if matlab_compat else 0
    if np.sum(eigvals <= 0) > threshold:
        sig_temp = sig_temp + np.diag(-eigvals.min() + ridge_diag)
    return sig_temp

def _fallback_univariate(cs: ChannelExpectedStats, *, matlab_compat: bool) -> tuple[NDArray, NDArray, NDArray]:
    """The `(W,mean,M2)` triple the fallback merges into the weighted one --
    the zero-filled all-`T_all` accumulator under canonical MATLAB
    (`matlab_compat=True`), or the observed-only accumulator under the
    corrected denominator (`matlab_compat=False`, quirk 1)."""
    if matlab_compat:
        return cs.allzero_n, cs.allzero_mean, cs.allzero_M2
    return cs.n_obs, cs.welford_mean, cs.welford_M2

def _gmm_m_step_channel(cs: ChannelExpectedStats, *, T_all: int, sigma_diag: bool, matlab_compat: bool) -> ChannelGMM:
    if cs.nanvar_y is None:
        raise ValueError("_gmm_m_step_channel requires finalized stats -- call finalize_expected_stats first")
    M_c, K_c = cs.resp.shape
    d_c = cs.n_obs.shape[0]

    weights = np.zeros((M_c, K_c))
    mu = np.zeros((d_c, M_c, K_c))
    sigma = np.zeros((d_c, M_c, K_c)) if sigma_diag else np.zeros((d_c, d_c, M_c, K_c))

    fb_W, fb_mean, fb_M2 = _fallback_univariate(cs, matlab_compat=matlab_compat)

    for i in range(M_c):
        for k in range(K_c):
            fb_mu = bool(cs.fallback_mu[i, k])

            if sigma_diag:
                W_ik, mean_ik, M2_ik = cs.weighted_W[i, k], cs.weighted_mean[i, k], cs.weighted_M2[i, k]
                if fb_mu:
                    _, mu_ik, M2_final = _welford_merge(W_ik, mean_ik, M2_ik, fb_W, fb_mean, fb_M2)
                else:
                    mu_ik, M2_final = mean_ik, M2_ik
                mu[:, i, k] = mu_ik
                sigma[:, i, k] = _SIGMA_RIDGE * cs.nanvar_y + M2_final / (W_ik + (fb_W if fb_mu else 0.0))
            else:
                Q, M, C = cs.pairwise_Q[i, k], cs.pairwise_M[i, k], cs.pairwise_C[i, k]
                W_diag, mean_diag, M2_diag = np.diag(Q), np.diag(M), np.diag(C)
                if fb_mu:
                    _, mu_ik, _ = _welford_merge(W_diag, mean_diag, M2_diag, fb_W, fb_mean, fb_M2)
                else:
                    mu_ik = mean_diag
                mu[:, i, k] = mu_ik

                fb_sigma = bool(cs.fallback_sigma_full[i, k])
                delta_i = M - mu_ik[:, None]
                delta_j = M.T - mu_ik[None, :]
                numerator = C + Q * delta_i * delta_j
                sigma_denom = cs.resp[i, k] + (T_all if fb_sigma else 0.0)
                if fb_sigma:
                    u_delta_i = cs.u_pairwise_M - mu_ik[:, None]
                    u_delta_j = cs.u_pairwise_M.T - mu_ik[None, :]
                    numerator = numerator + cs.u_pairwise_C + cs.u_pairwise_Q * u_delta_i * u_delta_j
                sig_temp = np.diag(_SIGMA_RIDGE * cs.nanvar_y) + numerator / sigma_denom
                sig_temp = repair_covariance(sig_temp, _SIGMA_RIDGE * cs.nanvar_y, matlab_compat=matlab_compat)
                sigma[:, :, i, k] = sig_temp

        resp_row = cs.resp[i, :]
        weights[i, :] = (resp_row + T_all * _EPS) / (np.sum(resp_row) + K_c * T_all * _EPS)

    return ChannelGMM(weights=weights, mu=mu, sigma=sigma, diag=sigma_diag)

def _emission_param_count(structure: LSIMStructure, sigma_diag: bool) -> int:
    """Free emission parameters as counted by `em_lsim.m`/`em_chmm.m`: means and
    covariances (`M*K*2*d` diagonal, `M*K*(d+d^2)` full) plus mixture weights
    (`M*(K-1)`)."""
    M = np.array(structure.channel_state_num)
    K = np.array(structure.num_gmm_component)
    d = np.array(structure.channel_obs_dim)
    if sigma_diag:
        return int(np.sum(M * K * 2 * d)) + int(np.sum(M * (K - 1)))
    return int(np.sum(M * K * (d + d**2))) + int(np.sum(M * (K - 1)))


# --------------------------------------------------------------------------- mini-batch covariance ridge


def compute_global_nanvar(
    structure: LSIMStructure, Y_trials: list[NDArray[np.float64]], *, max_chunk_observations: int | None = None
) -> list[NDArray[np.float64]]:
    """Precomputes each channel's whole-dataset observed-only `nanvar_y` --
    the canonical MATLAB mini-batch covariance-ridge policy (P10c, §2.6).
    `em_lsim_par.m`'s own covariance ridge (`0.001*var(...)`, §0.4.1) reads
    `all_observation` -- the WHOLE dataset's concatenated observations,
    built once outside the mini-batch loop (`em_lsim_par.m:169`) -- even
    though every OTHER M-step quantity for that same update (`Recentering`,
    `weight_update`, the responsibility-weighted mean) reads
    `all_observation_batch`, the CURRENT mini-batch's own subset
    (`em_lsim_par.m:394-401`, consumed at `:511`/`:517`/`:520`; contrast the
    ridge itself at `:508`/`:517`/`:520`, which reads `all_observation`, not
    `all_observation_batch`) -- confirmed by reading `em_lsim_par.m`
    directly during the P10b correction round, not guessed.

    **No inference (§P10c correction round).** `nanvar_y` depends only on
    which values were OBSERVED (`Y`/`structure`), never on `params`/`gamma`
    at all -- the first version of this function nonetheless ran one
    complete whole-dataset forward-backward E-step (`e_step`, discarding
    every gamma-dependent field it computed) just to reach `finalize_
    expected_stats`'s own `nanvar_y` formula, which is real, unnecessary
    inference work: a full pass over every trial, potentially spawning its
    own `ProcessPoolExecutor` or an extra Torch batched pass, directly
    undermining the ~100,000-trial scalability goal this whole phase exists
    for. This version accumulates the SAME per-channel observed-only
    Welford state (`_welford_batch`/`_welford_merge`, the exact functions
    `_accumulate_channel_trial`'s own `n_obs`/`welford_mean`/`welford_M2`
    fields use) directly from `Y_trials`, with NO `params`, `FbConstants`,
    `backend`, `n_jobs`, `executor`, or `backend_options` needed at all --
    and applies the shared `_nanvar_from_welford` formula (MATLAB sample
    variance, `ddof=1`; `n_obs==1 -> 0.0`, not `NaN`; `n_obs==0 -> NaN`;
    numerically stable at large offsets, since Welford's algorithm never
    forms a raw `sum(y**2)` moment). Accumulator memory is O(model
    dimensions) -- one `(d_c,)` triple per channel, reused across every
    trial -- never O(len(Y_trials)) or O(T_all).

    Must be computed exactly ONCE, before mini-batch training begins, and
    never recomputed from an individual mini-batch's own subset --
    `em_lsim_minibatch` calls this once and substitutes the result into
    every mini-batch's own finalized stats via `_substitute_global_nanvar`,
    before every `m_step` call.

    **Validates every trial before touching it (§P10c final correction).**
    This function runs before the very first E-step of a training run, so
    it is the first place a malformed `Y_trials` would otherwise be
    discovered ONLY by an obscure downstream symptom rather than a clear
    error here: a scalar or 1-D `Y` would raise a raw, confusing indexing
    error from `Y[dim_sl, :]`; a wrong-`D` array would be sliced against
    the wrong dimension boundaries and silently produce a wrong (not
    obviously wrong) `nanvar_y`; an integer-dtype `Y` would silently violate
    this package's floating-observation contract (`§2.1.1`); `+/-Inf` would
    silently produce `NaN`/`Inf` variance rather than a caught error;
    `T=0` would silently contribute nothing. `validate_observations`
    (already this package's own single source of truth for exactly these
    checks -- shape, dtype, `D`, `T>=1`, no `Inf`) is called on every trial
    before any accumulation begins, re-raising with that trial's own
    position in `Y_trials` named, so a large (e.g. ~100,000-trial) dataset's
    one bad trial is immediately identifiable. Legal `NaN` missingness is
    untouched -- `validate_observations` itself only rejects `Inf`, never
    `NaN`. An empty `Y_trials` is rejected outright (there is no such thing
    as a whole-dataset variance of zero trials -- silently returning
    all-`NaN` would be a `nanvar_y` for a dataset that doesn't exist).

    **Chunk-vectorized (P10f).** The previous body validated every trial with
    a Python-level `validate_observations` call and then merged one Welford
    batch per trial per channel -- a per-trial Python loop that dominated
    CUDA training at large N (~8 s per 20,000 trials; ~600 s of a 631 s
    two-epoch run at N=1,000,000). Now `validate.
    _iter_validated_observation_chunks` validates bounded chunks of trials
    (at most `max_chunk_observations` values, default `validate.DEFAULT_MAX_
    CHUNK_OBSERVATIONS`; a single larger trial forms its own chunk) with the
    identical checks, messages and error order, and each validated chunk is
    folded into the same float64 observed-only Welford/Chan accumulators by
    one `_welford_batch` over the chunk's concatenated timesteps followed by
    `_welford_merge` -- the same mergeable central-moment semantics, never the
    whole dataset concatenated, and no observation reaches an accumulator
    before its chunk passed validation. `n_obs==0 -> NaN`, `n_obs==1 -> 0.0`
    and ragged/missing handling are unchanged (`_nanvar_from_welford`)."""
    if not Y_trials:
        raise ValueError("Y_trials must contain at least one trial")
    accumulators = [
        (np.zeros(structure.channel_obs_dim[c]), np.zeros(structure.channel_obs_dim[c]), np.zeros(structure.channel_obs_dim[c]))
        for c in range(structure.C)
    ]
    for _start, _stop, X, nan_mask in _iter_validated_observation_chunks(
        structure, Y_trials, label="Y_trials", max_chunk_observations=max_chunk_observations
    ):
        for c in range(structure.C):
            dim_sl = structure.dim_slice(c)
            o = ~nan_mask[dim_sl, :]
            X_c = X[dim_sl, :]
            # Center each dimension on one of its own observed values before the chunk-wide sum (exact algebra:
            # the mean shifts back, the central moment M2 is shift-invariant), so a chunk of thousands of
            # large-offset values does not lose the precision the per-trial merge kept.
            has_obs = o.any(axis=1)
            shift = np.where(has_obs, X_c[np.arange(X_c.shape[0]), o.argmax(axis=1)], 0.0)
            obs_W, obs_mean, obs_M2 = _welford_batch(o.astype(np.float64), np.where(o, X_c - shift[:, None], 0.0))
            obs_mean = np.where(obs_W > 0, obs_mean + shift, 0.0)
            accumulators[c] = _welford_merge(*accumulators[c], obs_W, obs_mean, obs_M2)
    return [_nanvar_from_welford(n_obs, welford_M2) for n_obs, _mean, welford_M2 in accumulators]

def _substitute_global_nanvar(stats: ExpectedStats, global_nanvar: list[NDArray[np.float64]]) -> ExpectedStats:
    """Overrides each channel's own derived `nanvar_y` (computed by
    `finalize_expected_stats` from THIS mini-batch's own accumulators) with
    the precomputed, immutable whole-dataset value from `compute_global_
    nanvar` -- `stats` must already be finalized (so `nanvar_y` is non-None
    and about to be overwritten, not left uncomputed)."""
    if any(c.nanvar_y is None for c in stats.channels):
        raise ValueError("_substitute_global_nanvar requires already-finalized stats")
    return dataclasses_replace(
        stats, channels=[dataclasses_replace(c, nanvar_y=global_nanvar[i]) for i, c in enumerate(stats.channels)]
    )


# --------------------------------------------------------------------------- registered model


class GaussianMixtureEmissions(EmissionModel):
    """Per-channel Gaussian mixtures with diagonal or full covariances.

    Model-specific keyword options: ``sigma_diag`` (covariance type) for the
    statistics and the M-step, ``matlab_compat`` for the M-step and
    validation, and the initialization controls of
    `chmm_lsim.initialize_lsim_params` (``sigma_diag``, ``matlab_compat``,
    ``max_samples_per_channel``, ``n_jobs``)."""

    name = "gmm"
    channel_type = ChannelGMM
    stats_type = ChannelExpectedStats
    serial_global_keys = ("sigma_diag",)
    serial_channel_fields = ("weights", "mu", "sigma")

    # -- parameters --------------------------------------------------------------------------------
    def check_shapes(self, structure, emissions):
        _check_gmm_shapes(structure, emissions)

    def validate(self, emissions, *, matlab_compat=True):
        _validate_gmm_emissions(emissions, matlab_compat=matlab_compat)

    def copy(self, emission):
        return emission.copy()

    def n_free_parameters(self, structure, *, sigma_diag, **options):
        return _emission_param_count(structure, sigma_diag)

    # -- densities and sampling --------------------------------------------------------------------
    def densities(self, structure, emissions, Y):
        if emissions[0].diag:
            return _gmm_pdf_diag(structure, emissions, Y)
        return _gmm_pdf_full(structure, emissions, Y)

    def sample(self, rng, emission, state, *, past=None):
        return _gmm_sample(rng, emission, state)

    # -- EM statistics and M-step ------------------------------------------------------------------
    def new_stats(self, structure, c, emission):
        return _empty_channel_stats(
            structure.channel_state_num[c], structure.num_gmm_component[c], structure.channel_obs_dim[c], emission.diag
        )

    def accumulate(self, stats, structure, c, posterior, Y, emission):
        _accumulate_channel_trial(stats, structure, c, posterior, Y, emission.diag)

    def merge_stats(self, chunks):
        return merge_channel_stats(chunks)

    def finalize_stats(self, stats, *, sigma_diag, **options):
        return finalize_channel_stats(stats, sigma_diag=sigma_diag)

    def m_step(self, structure, stats, *, T_all, sigma_diag, matlab_compat=True, **options):
        return [
            _gmm_m_step_channel(stats[zee], T_all=T_all, sigma_diag=sigma_diag, matlab_compat=matlab_compat)
            for zee in range(structure.C)
        ]

    def global_statistics(self, structure, Y_trials, **options):
        return compute_global_nanvar(structure, Y_trials, **options)

    def apply_global_statistics(self, stats, global_stats):
        return _substitute_global_nanvar(stats, global_stats)

    # -- initialization ----------------------------------------------------------------------------
    def check_init_options(self, structure, *, sigma_diag, matlab_compat=True, max_samples_per_channel=None, n_jobs=1):
        from chmm_lsim.emissions.gmm_init import _check_init_options

        return _check_init_options(structure, sigma_diag=sigma_diag, matlab_compat=matlab_compat,
                                   max_samples_per_channel=max_samples_per_channel, n_jobs=n_jobs)

    def initialize(self, structure, Y_trials, channel_seeds, **options):
        from chmm_lsim.emissions.gmm_init import _initialize_gmm_emissions

        return _initialize_gmm_emissions(structure, Y_trials, channel_seeds, **options)

    # -- serialization -----------------------------------------------------------------------------
    def as_float64(self, emissions):
        return [
            ChannelGMM(weights=np.asarray(g.weights, dtype=np.float64), mu=np.asarray(g.mu, dtype=np.float64),
                       sigma=np.asarray(g.sigma, dtype=np.float64), diag=g.diag)
            for g in emissions
        ]

    def to_arrays(self, emissions):
        global_arrays = {"sigma_diag": np.array(bool(emissions[0].diag), dtype=np.bool_)}
        return global_arrays, [{"weights": g.weights, "mu": g.mu, "sigma": g.sigma} for g in emissions]

    def read_global_arrays(self, path, data, fail):
        sd = data["sigma_diag"]
        if sd.shape != () or sd.dtype != np.bool_:
            fail(path, f"sigma_diag must be a 0-d numpy.bool_, got shape {sd.shape} dtype {sd.dtype}")
        return {"sigma_diag": bool(sd[()])}

    def channel_array_shapes(self, M, d, K, options):
        diag = options["sigma_diag"]
        return {"weights": (M, K), "mu": (d, M, K), "sigma": (d, M, K) if diag else (d, d, M, K)}

    def from_arrays(self, channel_arrays, options):
        diag = options["sigma_diag"]
        return [
            ChannelGMM(weights=a["weights"].copy(), mu=a["mu"].copy(), sigma=a["sigma"].copy(), diag=diag)
            for a in channel_arrays
        ]


GMM = register_emission_model(GaussianMixtureEmissions())
