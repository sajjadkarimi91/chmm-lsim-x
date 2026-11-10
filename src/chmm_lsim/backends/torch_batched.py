"""Batched Torch execution backend (CPU and CUDA; float64 for inference,
statistics and EM, float32 for forward-only scoring).

Unlike `backends.numpy_`/`backends.numba_` (one sequence per call), this
backend's whole reason to exist is running many sequences *at once*, batched
over a leading dimension, on a single device -- `forward_backward_batch` is
the real public API (§2 below); `forward_backward(..., backend="torch")`
(dispatched through `backends.base.run_backend`, same as `numpy`/`numba`) is
a thin batch-of-one adapter over it, for API-contract parity with the other
two backends, not the intended way to use this one.

**Optional, lazy dependency**, same discipline as `backends.numba_`: `import
lsim` never imports `torch`; only `backends.base.get_backend("torch")`/
`run_backend("torch", ...)` do, lazily, the first time a caller actually
requests `backend="torch"`. If `torch` isn't installed, importing this
module raises `MissingBackendDependency` naming `pip install 'chmm-lsim[torch]'`.

**Device support, honestly scoped.** `device="cpu"` is verified here, at
`dtype=float64`, against every committed posterior fixture and against
per-sequence NumPy on ragged batches -- see `tests/test_backends_torch.py`.
`device="cuda"`/`"mps"` are *implemented* (availability checked at runtime;
requesting an unavailable device raises rather than silently falling back to
CPU) but **not validated here** -- CUDA correctness, its fp32-vs-fp64 error
study, and the `N=100,000` cluster benchmarks are P8b, to be run on real GPU
hardware (`benchmarks/cuda_validation_cli.py`, written but never executed in
this environment). `torch.backends.mps.is_available()` on this development
machine is `True`, so a handful of MPS *smoke* tests do run here for real
(P8a's own optional, non-sign-off-required allowance) -- they are not a
substitute for P8b's CUDA validation and are labeled as smoke coverage only.

**Design (§3/§4 of the P8a request).** `TorchKernel.build(structure, params,
device, dtype)` converts every immutable, structure/parameter-derived
quantity (`pi_0`, `W`, `transition`, `coupling`, the one-hot channel-
membership matrix `E`, per-channel emission tensors) into device tensors
*once*, at the top of a `forward_backward_batch` call -- never per sequence,
never per timestep. The time recursion is a plain Python `for t`/`for tau`
loop (unavoidably sequential -- LSIM's own forward-backward recursion is
sequential in time by construction, on every backend in this project); the
loop body is a batched tensor op over every sequence in the chunk and every
state at once, with small, explicit Python loops only over the channel axis
`C` (always small -- a handful of channels, never thousands), exactly as
`backends.numba_`'s own docstring justifies the same pattern there.

**Ragged lengths, via one deliberate mathematical fact, not per-timestep
masking machinery.** Padding a sequence to a longer `T_max` with all-`NaN`
columns and running the *unmodified* algorithm over the padded length
reproduces the true-length result *exactly* (to floating-point noise) --
confirmed directly, by construction of the existing emissions/forward-
backward math: a timestep where every dimension of every channel is `NaN`
makes every state's rescaled emission block sum to the same per-channel
constant (see `_gmm_log_comp_diag`/`_gmm_log_comp_full`'s own docstrings),
which the forward pass's own per-channel block-sum-to-1 invariant (already
established behavior, not new to this backend) turns into an *exact*
zero-loglik, neutral-`scale` contribution at every padded step -- so it
"freezes" the recursion's own conclusions without this module ever needing
to special-case padded timesteps in the recursion itself. Extracting a
trimmed `Posterior` for one sequence is then just slicing `[:T_i]` --
`BatchedPosteriorChunk.extract` does exactly that.

**Log-domain emissions, for `float32` numerical safety.** `_gmm_log_comp_diag`/
`_gmm_log_comp_full` compute every state/component's log-density directly
(never forming a raw, possibly-underflowing linear-domain density first).
Per `(batch, channel, timestep)`, the channel-block maximum log-density is
subtracted before exponentiating (`B_comp`/`B` are therefore the *rescaled*
values, not the raw MATLAB-scale ones -- exactly `inference.base.Posterior`'s
own, already-designed-for-this `log_scale`/`B_raw()` mechanism, reused
verbatim, not reinvented); the removed maxima are added back into `loglik`
directly (`Posterior.loglik` is always the true, unscaled value, per its own
docstring). `alpha`/`beta`/`gamma`/`scale` are provably invariant to this
per-channel-per-timestep rescaling (the same multiplicative constant cancels
in every ratio the recursion ever forms), and the `B_comp/B` responsibility
ratio the GMM M-step (`emissions/gmm.py`) actually uses is invariant too, for the same reason --
confirmed directly against `emissions/gmm.py`'s linear-domain NumPy computation
via `B_raw()`/`B_comp` reconstruction, not merely asserted.

**`mode="stats"` reduces genuinely on-device.** The batched forward-backward
recursion runs fully batched, on device, for the whole chunk at once
(`e_step(..., backend="torch")` calls this *once* per chunk, never once per
sequence) -- and, since a review round found an earlier version of this
module transferred every chunk's full `alpha`/`beta`/`gamma`/`B`/`B_comp` to
CPU and reduced them there (one sequence at a time, via the GMM
`_accumulate_channel_trial`) despite documenting that as "device-reduction,"
`_reduce_chunk_stats_device` now performs the *entire* reduction --
`zeta`/`gamma_first`/`gamma_sum`/`T_all`/`n_transitions` and every
`ChannelExpectedStats` field (`resp`/`resp_pos`/`n_w_pos_diag`/
`n_w_pos_full`, the diagonal `weighted_W`/`mean`/`M2` triple or the full-
covariance `pairwise_Q`/`M`/`C` + `u_pairwise_Q`/`M`/`C` triples, `n_obs`/
`welford_mean`/`M2`, `allzero_mean`/`M2`) -- as Torch tensor ops on-device,
via `_weighted_batch_torch`/`_pairwise_batch_torch` (direct central-moment
ports of `emissions.gmm._welford_batch`/`_pairwise_batch`, batched over `(batch,
time)` jointly instead of one sequence's own `T` axis at a time -- never a
raw `Σy²` moment, preserving the same P4-verified large-offset stability).
Only the small, fixed-size *reduced* dict per chunk ever reaches CPU;
`alpha`/`beta`/`gamma`/`B`/`B_comp` themselves never do. Chunks are merged
incrementally (`em.merge_expected_stats`, one pair at a time) as they're
produced, not collected into a list of every chunk's stats first. This
keeps chunk memory at `O(chunk*S*T)` (never `O(N*S*T)`), matching the
original design goal -- the earlier version's actual gap was in the
*reduction step* specifically, not the recursion (which was always
genuinely batched and on-device), and is what this fixes.

One subtlety `_reduce_chunk_stats_device` handles explicitly: unlike the
recursion's own quantities (whose padding is self-masking via `NaN`, see
above), a fully-missing (padded) channel-timestep's `B_comp/B` responsibility
ratio is a *finite*, nonzero value (`weight[s,k]`, not `NaN`) -- so every
per-timestep quantity derived from `gamma`/the responsibility ratio (`zeta`,
`gamma_sum`, `resp`/`resp_pos`/`n_w_pos_*`, `weighted_*`/`pairwise_*`,
`allzero_*`) is masked by an explicit `valid_mask = t < lengths[b]` before
accumulation; only the plain `o = ~isnan(Y)` mask (feeding `n_obs`/
`welford_*`/`u_pairwise_*`) is already correctly zero at padded positions
by construction, needing no extra masking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import (
    BackendCapabilities,
    MissingBackendDependency,
    UnsupportedCombination,
    check_capability,
    emission_model_name,
    register_backend,
)

try:
    import torch
except ImportError as exc:
    raise MissingBackendDependency(
        "the 'torch' execution backend requires the optional torch dependency -- install it via "
        "`pip install 'chmm-lsim[torch]'` (tested with torch==2.13.0 on Python 3.13.5; torch's own PyPI "
        "metadata declares support for Python 3.10-3.15, covering this project's 3.12-3.14 target)"
    ) from exc

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.emissions.gmm import _LOW_MASS_GAMMA_THRESHOLD, ChannelExpectedStats, ChannelGMM
from chmm_lsim.learning.em_loop import ExpectedStats, merge_expected_stats
from chmm_lsim.models.lsim.inference.base import Diagnostics, Posterior
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.models.lsim.inference.poise import _REFINE_W0 as _POISE_REFINE_W0
from chmm_lsim.core.validate import ValidationError, _is_prevalidated, check_structure_matches, validate_observation_list, validate_observations

_LOG_2PI = math.log(2 * math.pi)
_BLOCK_SUM_ATOL = 1e-8  # matches inference.sed._assert_close_to_one (rtol=0)
_DEFAULT_CPU_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024  # 512 MiB, conservative, documented default for CPU/MPS
_CUDA_MEMORY_HEADROOM = 0.8  # use at most 80% of torch.cuda.mem_get_info()'s free bytes


# ==========================================================================
# Device / dtype resolution
# ==========================================================================


def _resolve_device(device: str | "torch.device") -> "torch.device":
    if isinstance(device, torch.device):
        device = str(device.type)
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise UnsupportedCombination(
                "device='cuda' was requested but torch.cuda.is_available() is False on this machine -- "
                "refusing to silently fall back to CPU; either install/enable CUDA or request device='cpu'"
            )
        return torch.device("cuda")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise UnsupportedCombination(
                "device='mps' was requested but torch.backends.mps.is_available() is False on this machine -- "
                "refusing to silently fall back to CPU; request device='cpu' or a machine with MPS"
            )
        return torch.device("mps")
    raise UnsupportedCombination(f"backend='torch' does not support device={device!r} (supported: 'cpu', 'cuda', 'mps')")


_DTYPE_MAP = {"float64": torch.float64, "float32": torch.float32}


def _resolve_dtype(dtype: str) -> "torch.dtype":
    if dtype not in _DTYPE_MAP:
        raise UnsupportedCombination(f"backend='torch' does not support dtype={dtype!r} (supported: {sorted(_DTYPE_MAP)})")
    return _DTYPE_MAP[dtype]


def _check_device_dtype_combination(device: "torch.device", dtype: "torch.dtype") -> None:
    """`float64` is not implemented on MPS at all (as of this torch release);
    requesting it must raise naming both axes, not silently downcast to
    `float32` or run on CPU instead."""
    if device.type == "mps" and dtype == torch.float64:
        raise UnsupportedCombination(
            "backend='torch' does not support dtype='float64' on device='mps' (MPS has no float64 support in "
            "this torch build) -- use dtype='float32' on 'mps', or device='cpu'/'cuda' for float64"
        )


# ==========================================================================
# TorchKernel: every structure/params-derived tensor, built once per call.
# ==========================================================================


@dataclass
class _TorchChannelGMM:
    weights: "torch.Tensor"  # (M_c, K_c)
    mu: "torch.Tensor"  # (d_c, M_c, K_c)
    sigma: "torch.Tensor"  # (d_c, M_c, K_c) diag, or (d_c, d_c, M_c, K_c) full
    diag: bool
    M_c: int
    K_c: int
    d_c: int


@dataclass
class TorchKernel:
    """Every immutable, `(structure, params)`-derived tensor a batched
    forward-backward call needs, built once (`TorchKernel.build`) at the top
    of `forward_backward_batch`, then reused for every chunk -- never
    reconstructed per sequence or per timestep. Caching here is scoped to a
    single `forward_backward_batch` call, not a persistent cross-call
    registry: `em_lsim` replaces `params` with a brand-new object every
    M-step anyway, so a params-keyed global cache would be a guaranteed miss
    across iterations regardless -- the actual, real performance win (no
    per-sequence/per-timestep reconstruction within one E-step's worth of
    chunks) is already fully captured by per-call scoping."""

    structure: LSIMStructure
    device: "torch.device"
    dtype: "torch.dtype"
    pi_0: "torch.Tensor"  # (S,)
    W: "torch.Tensor"  # (S,S)
    transition: "torch.Tensor"  # (S,S)
    coupling: "torch.Tensor"  # (C,C)
    E: "torch.Tensor"  # (S,C) one-hot channel membership
    block_of_state: "torch.Tensor"  # (S,) int64
    sigma_diag: bool
    emissions: list[_TorchChannelGMM]
    max_K: int
    flag_nan: bool  # coupling.sum(axis=1) has a near-zero row (sed's own flag_nan)
    # P8c: the per-channel state blocks, cached once so the indexed one-hot forms
    # (see `_onehot_ops`) never rebuild them inside the timestep loop.
    channel_slices: tuple = ()
    equal_block_size: int | None = None  # M when every channel has the same M_c (then block sums are a reshape), else None

    @staticmethod
    def build(
        structure: LSIMStructure, params: LSIMParams, *, device: "torch.device", dtype: "torch.dtype", pi0_reg: float = 0.0
    ) -> "TorchKernel":
        """`pi0_reg` (P8b fix): the per-channel `pi_0` smoothing the NumPy/Numba
        kernels apply before the forward pass (`inference.sed._regularize_pi0`,
        computed here in float64 on CPU, then transferred). `forward_backward_
        batch` passes `fb_constants.pi0_reg`; the default `0.0` stores
        `params.pi_0` unchanged. Before this fix the batched path never applied
        it: invisible on the strictly-positive committed fixtures (a ~`pi0_reg`
        absolute effect), but an exactly-zero `pi_0` entry -- routine after EM
        updates -- gave exact-zero `alpha`/`gamma` on Torch instead of NumPy's
        smoothed values, and a different transition row downstream."""
        from chmm_lsim.models.lsim.inference.sed import _regularize_pi0

        def t(arr: NDArray) -> "torch.Tensor":
            return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)

        S, C = structure.S, structure.C
        E_np = np.zeros((S, C), dtype=np.float64)
        E_np[np.arange(S), structure.block_of_state] = 1.0

        emissions = []
        for c in range(C):
            gmm: ChannelGMM = params.emissions[c]
            emissions.append(
                _TorchChannelGMM(
                    weights=t(gmm.weights), mu=t(gmm.mu), sigma=t(gmm.sigma), diag=gmm.diag,
                    M_c=gmm.M_c, K_c=gmm.K_c, d_c=gmm.d_c,
                )
            )

        from chmm_lsim.backends.numpy_ import build_W

        W_np = build_W(params.transition, params.coupling, structure.block_of_state)
        flag_nan = bool(np.any(params.coupling.sum(axis=1) <= 1e-10))

        return TorchKernel(
            structure=structure, device=device, dtype=dtype,
            pi_0=t(_regularize_pi0(params.pi_0, structure, pi0_reg)), W=t(W_np), transition=t(params.transition), coupling=t(params.coupling),
            E=t(E_np), block_of_state=torch.as_tensor(np.array(structure.block_of_state), dtype=torch.long, device=device),
            sigma_diag=params.sigma_diag, emissions=emissions, max_K=structure.max_K, flag_nan=flag_nan,
            channel_slices=tuple(structure.state_slice(c) for c in range(C)),
            equal_block_size=(int(structure.channel_state_num[0]) if len(set(structure.channel_state_num)) == 1 else None),
        )


# ==========================================================================
# Log-domain emissions
# ==========================================================================


def _eig_logpdf(x: "torch.Tensor", mu: "torch.Tensor", sigma: "torch.Tensor") -> "torch.Tensor":
    """log N(x; mu, sigma) via symmetric eigendecomposition -- handles both
    well-conditioned and singular-but-PSD `sigma`, matching
    `scipy.stats.multivariate_normal`'s own `allow_singular=True` internal
    `_PSD`/`_support_mask` machinery *exactly*, not merely approximately:

    - the eigenvalue cutoff is `eps = cond * max(abs(eigvals))`, with
      `cond = factor * finfo(dtype).eps` and `factor = 1e3` (`float32`) or
      `1e6` (`float64`) -- `scipy.stats._multivariate._eigvalsh_to_eps`'s
      own dtype-dependent formula, not a single fixed `rel_tol` (a review
      round found the previous fixed `rel_tol=1e-8` disagreed with `scipy`
      by many orders of magnitude on both sides of realistic near-singular
      eigenvalues -- confirmed directly with `sigma=diag(1,e)`: at `e=1e-8`
      the fixed-tolerance version wrongly zeroed out a real, meaningfully
      large eigenvalue that `scipy` keeps, while at `e=1e-10` it wrongly
      kept an eigenvalue `scipy` -- and this function, now -- correctly
      drops, `scipy` returning `-inf` there since the point is off-support).
    - eigenvalues at or below `eps` are excluded from the log-determinant,
      the rank, and the pseudo-inverse quadratic form (as before) -- but
      **also** define a null-space basis (`eigvecs[:, ~mask]`). A review
      round found the previous version silently dropped the null-space
      contribution to the quadratic form without ever checking whether the
      observation's deviation actually has a nonzero component along that
      null space -- `scipy` calls that "off support" and returns `-inf`
      there (a real difference: the density is exactly `0` outside the
      subspace a singular Gaussian's mass concentrates on, not merely
      "small"). This function now computes the same `_support_mask` check
      `scipy` does: the deviation's projection onto the null-space basis,
      compared against a *looser* threshold (`1e3 * eps`, `scipy`'s own
      choice -- deliberately looser than the eigenvalue-retention cutoff
      itself, since "is this deviation numerically within the support"
      tolerates more floating-point slop than "is this eigenvalue nonzero"
      does), and returns `-inf` wherever that check fails, exactly as
      `scipy` does.

    `x`: `(N, d)`, `mu`: `(d,)`, `sigma`: `(d, d)`. **Confirmed to match
    `scipy` exactly for valid (positive-semidefinite) `sigma`** (`~1e-15`,
    and exact `-inf` agreement, not merely a close finite number) on:
    well-conditioned, rank-1 singular (both on- and off-support), the
    literal `sigma=[[1,1],[1,1]]`/`x=[1,1]` (on-support) vs `x=[1,0]`
    (off-support) case, and `sigma=diag(1,e)`/`x=[0,1e-4]` at
    `e in {1e-7,1e-8,1e-9,1e-10}` -- see `tests/test_backends_torch.py`.
    **Not claimed for an invalid, non-PSD `sigma`** (one with a materially
    negative eigenvalue, `< -eps`): `scipy`'s own `_PSD.__init__` raises
    `ValueError` there rather than returning a density; this function does
    not reproduce that raise (a materially negative eigenvalue simply fails
    `mask = eigvals > eps` and is folded into the null-space handling above,
    same as a legitimately-zero eigenvalue) and so does not match `scipy`
    on that specific invalid-input boundary. This is deliberately left
    unenforced *here*: `LSIMParams.validate()` already rejects a non-PSD
    covariance before any emission computation is reached on any backend
    (`numpy`/`numba` included), so this function is never exercised with
    one in practice -- narrowing this claim to valid PSD inputs, rather
    than adding a redundant second PSD check inside this hot per-component
    path, is the documentation fix a review round asked for here."""
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    dtype = sigma.dtype
    factor = 1e3 if dtype == torch.float32 else 1e6
    eps = factor * torch.finfo(dtype).eps * eigvals.abs().max()
    mask = eigvals > eps
    safe_eigvals = torch.where(mask, eigvals, torch.ones_like(eigvals))
    inv_eigvals = torch.where(mask, 1.0 / safe_eigvals, torch.zeros_like(eigvals))
    logdet = torch.where(mask, torch.log(safe_eigvals), torch.zeros_like(eigvals)).sum()
    rank = mask.sum().to(x.dtype)
    diff = x - mu[None, :]
    proj = diff @ eigvecs
    quad = (proj**2 * inv_eigvals[None, :]).sum(dim=1)
    logpdf = -0.5 * quad - 0.5 * logdet - 0.5 * rank * _LOG_2PI

    null_mask = ~mask
    if bool(null_mask.any()):
        support_eps = 1e3 * eps  # scipy's own, deliberately looser, support-check tolerance
        residual = torch.linalg.norm(proj[:, null_mask], dim=1)
        out_of_support = residual >= support_eps
        logpdf = torch.where(out_of_support, torch.full_like(logpdf, float("-inf")), logpdf)
    return logpdf


def _gmm_log_comp_diag(kernel: TorchKernel, Y: "torch.Tensor") -> "torch.Tensor":
    """Diagonal-covariance log-component-densities, batched over `(batch,
    timestep)` at once per `(channel, state, component)` -- direct log-space
    port of `emissions.gmm_pdf_diag`. A fully-`NaN` channel at some
    `(batch, timestep)` (real missingness, *or* this module's own ragged-
    length padding convention -- indistinguishable, and deliberately so, see
    module docstring) contributes log-density `log(weight[s,k])` per
    component -- not `0` -- since the density itself (not its log) is what
    "contributes 1" when every dimension is marginalized out; padded
    `k >= K_c` component slots are left at `-inf` (never written), so they
    exponentiate to exactly `0`, matching `B_comp`'s own zero-padding."""
    Bn, D, T = Y.shape
    S = kernel.structure.S
    log_B_comp = torch.full((Bn, S, kernel.max_K, T), float("-inf"), dtype=Y.dtype, device=Y.device)
    for c in range(kernel.structure.C):
        gmm = kernel.emissions[c]
        sl_dim = kernel.structure.dim_slice(c)
        s0 = int(kernel.structure.state_offsets[c])
        Yc = Y[:, sl_dim, :]  # (Bn,d_c,T)
        nan_mask = torch.isnan(Yc)
        for s in range(gmm.M_c):
            for k in range(gmm.K_c):
                mu = gmm.mu[:, s, k][None, :, None]
                sigma = gmm.sigma[:, s, k][None, :, None]
                quad = (Yc - mu) ** 2 / sigma
                log_sqrt_det = 0.5 * torch.log(sigma)
                log_dens = -0.5 * quad - log_sqrt_det - 0.5 * _LOG_2PI
                log_dens = torch.where(nan_mask, torch.zeros_like(log_dens), log_dens)
                ll = log_dens.sum(dim=1)  # (Bn,T)
                log_B_comp[:, s0 + s, k, :] = torch.log(gmm.weights[s, k]) + ll
    return log_B_comp


def _gmm_log_comp_full(kernel: TorchKernel, Y: "torch.Tensor") -> "torch.Tensor":
    """Full-covariance log-component-densities -- direct log-space port of
    `emissions.gmm_pdf_full`'s own per-channel/state/component/NaN-pattern
    loop (grouping by unique missingness pattern per channel, restricting to
    the observed sub-block via `_eig_logpdf`), batched over `(batch,
    timestep)` flattened into one axis per channel. A fully-missing pattern
    contributes log-density `0` directly (the density itself is `1`, so its
    log genuinely is `0` here -- unlike the diagonal case above, there is no
    per-dimension weight/density split to preserve, `gmm_pdf_full`'s own
    `comp_vals[rows] = 1.0` for a fully-missing pattern already *is* the
    per-component density, weighted separately by `gmm.weights[s,k]`)."""
    Bn, D, T = Y.shape
    S = kernel.structure.S
    log_B_comp = torch.full((Bn, S, kernel.max_K, T), float("-inf"), dtype=Y.dtype, device=Y.device)
    for c in range(kernel.structure.C):
        gmm = kernel.emissions[c]
        sl_dim = kernel.structure.dim_slice(c)
        s0 = int(kernel.structure.state_offsets[c])
        Yc = Y[:, sl_dim, :]  # (Bn,d_c,T)
        Yc_flat = Yc.permute(0, 2, 1).reshape(Bn * T, gmm.d_c)  # (N,d_c)
        nan_mask_flat = torch.isnan(Yc_flat)
        patterns, inverse = torch.unique(nan_mask_flat, dim=0, return_inverse=True)
        for s in range(gmm.M_c):
            for k in range(gmm.K_c):
                mu = gmm.mu[:, s, k]
                sigma = gmm.sigma[:, :, s, k]
                comp_vals = torch.zeros(Bn * T, dtype=Y.dtype, device=Y.device)
                for p_idx in range(patterns.shape[0]):
                    rows = inverse == p_idx
                    observed = ~patterns[p_idx]
                    if not bool(observed.any()):
                        continue  # fully-missing pattern: log-density 0, already the init value
                    X = Yc_flat[rows][:, observed]
                    mu_sub = mu[observed]
                    sigma_sub = sigma[observed][:, observed]
                    comp_vals[rows] = _eig_logpdf(X, mu_sub, sigma_sub)
                log_w = torch.log(gmm.weights[s, k])
                log_B_comp[:, s0 + s, k, :] = (log_w + comp_vals).reshape(Bn, T)
    return log_B_comp


def compute_emissions_torch(kernel: TorchKernel, Y: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Returns `(B_rescaled, B_comp_rescaled, log_scale)` -- `B`/`B_comp` are
    the *rescaled* (channel-block-max-subtracted, then exponentiated)
    emission arrays actually fed to the recursion, `log_scale` is `(Bn,C,T)`
    -- the same field `Posterior.log_scale`/`B_raw()` already exists for.
    `alpha`/`beta`/`gamma`/`scale` are unaffected by this rescaling (the
    same multiplicative-per-channel-per-timestep constant cancels in every
    ratio those quantities are ever built from); `loglik`'s caller must add
    `log_scale.sum(dim=(1,2))` back in -- `_run_batched_forward_backward`
    does this once, so every caller of this module always sees the true,
    unscaled `loglik`, matching `Posterior.loglik`'s own documented contract."""
    log_comp_fn = _gmm_log_comp_diag if kernel.sigma_diag else _gmm_log_comp_full
    log_B_comp = log_comp_fn(kernel, Y)  # (Bn,S,max_K,T)
    Bn, S, max_K, T = log_B_comp.shape
    C = kernel.structure.C

    max_log = torch.empty((Bn, C, T), dtype=Y.dtype, device=Y.device)
    for c in range(C):
        sl = kernel.structure.state_slice(c)
        max_log[:, c, :] = log_B_comp[:, sl, :, :].amax(dim=(1, 2))

    max_log_per_state = max_log[:, kernel.block_of_state, :]  # (Bn,S,T)
    log_B_comp_rescaled = log_B_comp - max_log_per_state[:, :, None, :]
    B_comp = torch.where(torch.isfinite(log_B_comp_rescaled), torch.exp(log_B_comp_rescaled), torch.zeros_like(log_B_comp_rescaled))
    B = B_comp.sum(dim=2)
    return B, B_comp, max_log


# ==========================================================================
# Batched forward pass, shared by sed and fast (mirrors inference.sed._forward_pass).
# ==========================================================================


def _check_block_sum_batched(x: "torch.Tensor", E: "torch.Tensor", atol: float = _BLOCK_SUM_ATOL) -> "torch.Tensor":
    """`x`: `(Bn,S)`. Returns a `(Bn,)` bool: `True` where every channel
    block of that batch row is finite and sums to `1.0` (`rtol=0`) -- the
    batched equivalent of `inference.sed._assert_block_sum_one`."""
    sums = x @ E  # (Bn,C) -- validate="all" only (not on the timed path); left as the one-hot matmul form
    finite = torch.isfinite(x).all(dim=1)
    close = (sums - 1.0).abs().max(dim=1).values <= atol
    return finite & close


def _first_failure(ok: "torch.Tensor", sequence_indices: list[int], t: int, active: "torch.Tensor | None" = None) -> tuple[int, int, int] | None:
    """`ok`: `(Bn,)` bool. Returns `(batch_local_idx, original_seq_idx, t)`
    for the first entry that is both `False` *and* (if `active` is given)
    genuinely active at this timestep for that sequence -- padded rows
    (past their own real length) are never reported as failures, matching
    the same masking `_check_block_sum_batched`'s own callers apply before
    calling this (see `_backward_sed_batch`'s module-level discussion of
    why a shared timestep index across a ragged batch needs this)."""
    bad_mask = ~ok if active is None else (~ok & active)
    bad = torch.nonzero(bad_mask, as_tuple=False)
    if bad.numel() == 0:
        return None
    local = int(bad[0, 0].item())
    return local, sequence_indices[local], t


def _forward_pass_batch(
    kernel: TorchKernel, Bemis: "torch.Tensor", sequence_indices: list[int], lengths: "torch.Tensor", check_all: bool
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """`inference.sed._forward_pass`, batched over a leading `(Bn,)` axis --
    `alpha[:, :, t] = abar @ W` (a batched matvec, `W` shared/cached, never
    reconstructed), `p_ot_c_cond_past` via the shared one-hot `E`. `lengths`
    (`(Bn,)` int64, each sequence's own true `T_i` within this padded chunk)
    gates `validate="all"`'s per-timestep checks: position `t` is only a
    *real* (non-padding) timestep for row `b` when `t < lengths[b]` -- a
    review round found the un-gated version would validate (and could, for
    a genuinely corrupted padded region, spuriously fail on) positions no
    caller will ever read back out of a trimmed `Posterior`."""
    Bn, S, T = Bemis.shape
    E = kernel.E
    _bc_states, broadcast_channels, block_sum_last, _bs_both = _onehot_ops(kernel)
    alpha = torch.zeros(Bn, S, T, dtype=Bemis.dtype, device=Bemis.device)
    scale = torch.zeros(Bn, S, T, dtype=Bemis.dtype, device=Bemis.device)
    alpha[:, :, 0] = kernel.pi_0[None, :]
    p0 = block_sum_last(alpha[:, :, 0] * Bemis[:, :, 0])  # (Bn,C)
    scale[:, :, 0] = Bemis[:, :, 0] / broadcast_channels(p0)
    loglik = torch.log(p0).sum(dim=1)  # (Bn,)

    if check_all:
        real_t0 = 0 < lengths  # every sequence has length >= 1, so this is always all-True; kept for symmetry/clarity
        ok = _check_block_sum_batched(alpha[:, :, 0], E)
        if not bool((ok | ~real_t0).all()):
            loc = _first_failure(ok, sequence_indices, 0, active=real_t0)
            raise ValidationError(f"[backend=torch] alpha at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    for t in range(1, T):
        abar = alpha[:, :, t - 1] * scale[:, :, t - 1]
        alpha[:, :, t] = abar @ kernel.W
        pc = block_sum_last(alpha[:, :, t] * Bemis[:, :, t])
        scale[:, :, t] = Bemis[:, :, t] / broadcast_channels(pc)
        loglik = loglik + torch.log(pc).sum(dim=1)

        if check_all:
            real_t = t < lengths
            ok = _check_block_sum_batched(alpha[:, :, t], E)
            if not bool((ok | ~real_t).all()):
                loc = _first_failure(ok, sequence_indices, t, active=real_t)
                raise ValidationError(f"[backend=torch] alpha at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    return alpha, scale, loglik


# ==========================================================================
# Batched SED backward (method="optimized" only -- see module docstring).
# ==========================================================================


_ONEHOT_IMPL = "indexed"
"""P8c: how the *one-hot* `E` contractions in the recursions are evaluated.

`E` is `(S,C)` one-hot channel membership, so `x @ E`, `E @ x` and
`einsum("ic,bij,jw->bcw", E, x, E)` compute no arithmetic of their own: they
broadcast a per-channel value to that channel's states, or sum a per-state axis
within each channel's block. Expressed as matmuls they nonetheless cost a factor
`C` more floating-point operations than the quantity being formed, and on
FP64-segmented GPUs (measured on an L40S: 1.25 TFLOPS FP64 versus 49.8 TFLOPS
FP32, while elementwise bandwidth is identical at 631 GB/s in both) those wasted
FLOPs dominate the SED backward recursion.

- `"indexed"` (default): `index_select` for the broadcasts and block-contiguous
  sums for the reductions -- memory-bound, no FP64 FLOP penalty.
- `"matmul"`: the original one-hot matmul/einsum forms, kept as the reference
  implementation the P8c parity tests compare against (set it by monkeypatching
  this module attribute; there is no public API or runtime switch).

The two differ in floating-point detail in two documented ways, both covered by
the parity tests: (1) block sums add only their own block's terms, while the
matmul adds all `S` terms (including exact zeros) in cuBLAS's order, so results
can differ by rounding, which the SED delta solve can amplify; (2) a matmul
propagates a `NaN` in one channel's entry across *every* state of the row
(`0 * NaN = NaN`), while a gather/segment sum confines it to that channel's own
states."""


def _poise_constants(kernel: "TorchKernel"):
    """POISE's `(S,C)` outgoing mixture weights, `(S,)` exponent and exact-zero mask,
    on `kernel`'s own device/dtype. Computed from the same `coupling` rows as
    `inference.poise.poise_backward_constants`, whose NumPy version is the reference the
    P12 tests compare every backend against."""
    theta_out_channel = kernel.coupling.sum(dim=1)  # (C,)
    nonzero = theta_out_channel > 0
    denom = torch.where(nonzero, theta_out_channel, torch.ones_like(theta_out_channel))
    rho_channel = torch.where(nonzero[:, None], kernel.coupling / denom[:, None], torch.zeros_like(kernel.coupling))
    rho = rho_channel[kernel.block_of_state]  # (S,C)
    theta_out = theta_out_channel[kernel.block_of_state]  # (S,)
    return rho, theta_out, theta_out == 0


def _onehot_ops(kernel: "TorchKernel"):
    """The four one-hot operations the recursions need, bound to `kernel`'s own
    block layout and to the active `_ONEHOT_IMPL`. Returned as closures so the
    timestep loops select an implementation once, never per iteration."""
    E, bos = kernel.E, kernel.block_of_state
    slices, block = kernel.channel_slices, kernel.equal_block_size
    C = kernel.structure.C

    if _ONEHOT_IMPL == "matmul":
        return (
            lambda x: torch.einsum("ic,bcj->bij", E, x),  # broadcast_states: (Bn,C,S) -> (Bn,S,S)
            lambda x: torch.matmul(E.unsqueeze(0), x) if x.dim() == 3 else x @ E.T,  # broadcast_channels
            lambda x: x @ E,  # block_sum_last: (...,S) -> (...,C)
            lambda x: torch.einsum("ic,bij,jw->bcw", E, x, E),  # block_sum_both: (Bn,S,S) -> (Bn,C,C)
        )
    if _ONEHOT_IMPL != "indexed":
        raise ValueError(f"_ONEHOT_IMPL must be 'indexed' or 'matmul', got {_ONEHOT_IMPL!r}")

    def broadcast(x):
        """(Bn,C,...) -> (Bn,S,...): every state takes its own channel's value."""
        return x.index_select(1, bos)

    if block is not None:
        def block_sum_last(x):
            """(...,S) -> (...,C), summing each channel's contiguous block."""
            return x.reshape(*x.shape[:-1], C, block).sum(-1)

        def block_sum_both(x):
            """(Bn,S,S) -> (Bn,C,C): block sums over the last axis, then the state axis."""
            y = x.reshape(*x.shape[:-1], C, block).sum(-1)  # (Bn,S,C)
            return y.reshape(y.shape[0], C, block, C).sum(2)
    else:
        def block_sum_last(x):
            return torch.stack([x[..., sl].sum(-1) for sl in slices], dim=-1)

        def block_sum_both(x):
            y = torch.stack([x[..., sl].sum(-1) for sl in slices], dim=-1)  # (Bn,S,C)
            return torch.stack([y[:, sl, :].sum(1) for sl in slices], dim=1)

    return broadcast, broadcast, block_sum_last, block_sum_both


def _gz_block(abar: "torch.Tensor", W: "torch.Tensor", channel_slices: list[slice]) -> "torch.Tensor":
    """`Gz[b,c,j] = sum_{i in block c} abar[b,i]*W[i,j]`, via one small
    matmul per channel (channels are always few) -- never materializes an
    `(Bn,S,S)` intermediate. The default form (`gz_form="block"`)."""
    return torch.stack([abar[:, sl] @ W[sl, :] for sl in channel_slices], dim=1)  # (Bn,C,S)


def _gz_masked(abar: "torch.Tensor", W: "torch.Tensor", E: "torch.Tensor") -> "torch.Tensor":
    """Same `Gz`, via a single masked-dense einsum instead of a per-channel
    loop -- materializes an `(Bn,S,S)` intermediate. Cross-checked against
    `_gz_block` in `tests/test_backends_torch.py`; kept as an explicit,
    independently-selectable alternative (`gz_form="masked"`), not the
    default -- P8b benchmarks both on CUDA before any default changes."""
    return torch.einsum("ic,bi,ij->bcj", E, abar, W)


def _solve_delta_batch(f: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
    """`inference.sed._solve_delta`, batched: row-wise Sherman-Morrison solve
    of `(diag(f_z-h_z) + h_z*ones(C,C)) @ x_z = f_z` for every channel `z`
    and every batch row at once. `f`: `(Bn,C,C)`, `h`: `(Bn,C)`."""
    d = f - h[..., None]
    s1 = (f / d).sum(dim=-1)
    s2 = (1.0 / d).sum(dim=-1)
    g = h * s1 / (1.0 + h * s2)
    return (f - g[..., None]) / d


def _backward_sed_batch(
    kernel: TorchKernel,
    alpha: "torch.Tensor",
    scale: "torch.Tensor",
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    check_all: bool,
    gz_form: str,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, "torch.Tensor"]]:
    """`inference.sed.forward_backward_sed`'s backward loop, `method=
    "optimized"` only, batched over a leading `(Bn,)` axis. `p_cond`
    (`(Bn,S,S)`) is the materialized reference `delta`/`f` implementation
    the P8a request calls for -- an algebraically-expanded, `(Bn,S,S)`-
    avoiding alternative is deliberately not attempted here (P8b, only if
    independently faster *and* verified to match).

    Returns per-*sequence* diagnostic counters (`(Bn,)` int64 tensors, one
    entry per batch row), not one chunk-wide total -- so
    `BatchedPosteriorChunk.extract` can hand back a `Diagnostics` scoped to
    exactly the one sequence it extracts, matching `Posterior.diagnostics`'s
    existing single-sequence contract on every other backend.

    **Ragged-batch masking, precisely.** The `tau` loop below is one shared
    Python loop over the whole chunk's `T_max`, but a shorter sequence `b`
    (`lengths[b] < T_max`) has no genuine backward-recursion step at every
    `tau` this loop visits: the standalone reference (`inference.sed`) only
    ever runs `for tau in range(T_i-2, -1, -1)`, i.e. `tau < lengths[b]-1` --
    anything past that is this module's own NaN-padding convention
    continuing to "run" mathematically (harmlessly; the recursion stays
    well-defined there) but does not correspond to any real computation the
    reference implementation performs for that sequence. A review round
    confirmed this is not merely a theoretical concern: batching a length-1
    sequence next to a much longer one made `zero_alpha_next_count` come
    back `3` instead of the correct `0` (a standalone length-1 sequence
    never runs the backward loop body *at all*, so it can never accumulate
    any of these counters) -- every one of `zero_alpha_next_count`/
    `beta_floor_hits`/`f_le_h_violations`/`nan_guard_hits`, and the `abar`/
    `V2`/`f`/`h` `validate="all"` checks, are gated by `active_bw = tau <
    lengths[b]-1` below, computed once per `tau`, so a padding-only row
    never contributes to another sequence's -- or its own -- reported
    diagnostics. `gamma`'s own block-sum check is gated by the *wider*
    `real_ts = pos < lengths[b]` instead (matching `gamma` being a genuine,
    checkable output at *every* real timestep, including the terminal one,
    unlike the four counters above, which are specifically about the
    recursion *step*, one fewer in count than the number of real timesteps)."""
    Bn, S, T = alpha.shape
    C = kernel.structure.C
    E = kernel.E
    W = kernel.W
    channel_slices = [kernel.structure.state_slice(c) for c in range(C)]
    broadcast_states, broadcast_channels, block_sum_last, block_sum_both = _onehot_ops(kernel)
    fused = _ONEHOT_IMPL == "indexed"  # reuse p_cond's buffer for `weighted`/`num` (same values, fewer (Bn,S,S) temporaries)
    # `p_cond` is built straight into its channel-blocked view when every channel has the same M_c
    # (the equal-block case a `(Bn,C,M,S)` view requires); otherwise via the gather above.
    blocked_p_cond = fused and kernel.equal_block_size is not None
    W_blocked = W.view(C, kernel.equal_block_size, S) if blocked_p_cond else None
    beta_floor, f_reg = fb_constants.beta_floor, fb_constants.f_reg

    beta = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    gamma = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    beta[:, :, T - 1] = scale[:, :, T - 1]
    gamma[:, :, T - 1] = alpha[:, :, T - 1] * beta[:, :, T - 1]

    # Accumulated on-device throughout the timestep loop -- transferred to
    # CPU exactly once, after the loop ends, not per-timestep (per-timestep
    # `.item()`/`.cpu()` calls would force a CPU/GPU synchronization at
    # every single step of the sequential recursion, exactly what P8a's own
    # "avoid CPU synchronization inside performance-critical timestep loops"
    # instruction rules out).
    zero_alpha_next_count = torch.zeros(Bn, dtype=torch.int64, device=alpha.device)
    beta_floor_hits = torch.zeros(Bn, dtype=torch.int64, device=alpha.device)
    f_le_h_violations = torch.zeros(Bn, dtype=torch.int64, device=alpha.device)
    nan_guard_hits = torch.zeros(Bn, dtype=torch.int64, device=alpha.device)

    if check_all:
        real_terminal = (T - 1) < lengths  # only True for rows whose OWN terminal really is at T-1 (i.e. lengths[b]==T)
        ok = _check_block_sum_batched(gamma[:, :, T - 1], E)
        if not bool((ok | ~real_terminal).all()):
            loc = _first_failure(ok, sequence_indices, T - 1, active=real_terminal)
            raise ValidationError(f"[backend=torch] gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    for tau in range(T - 2, -1, -1):
        active_bw = tau < (lengths - 1)  # (Bn,) -- see this function's own docstring for the exact condition and why
        abar = alpha[:, :, tau] * scale[:, :, tau]
        alpha_next = alpha[:, :, tau + 1]
        beta_next = beta[:, :, tau + 1]
        zero_alpha_next_count += torch.where(active_bw, (alpha_next == 0).sum(dim=1), torch.zeros(Bn, dtype=torch.int64, device=alpha.device))

        if check_all:
            ok = _check_block_sum_batched(abar, E)
            if not bool((ok | ~active_bw).all()):
                loc = _first_failure(ok, sequence_indices, tau, active=active_bw)
                raise ValidationError(f"[backend=torch] abar at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")
            v2 = block_sum_last(alpha_next * beta_next)  # (Bn,C)
            v2_ok = torch.isfinite(v2).all(dim=1) & ((v2 - 1.0).abs().max(dim=1).values <= _BLOCK_SUM_ATOL)
            if not bool((v2_ok | ~active_bw).all()):
                loc = _first_failure(v2_ok, sequence_indices, tau + 1, active=active_bw)
                raise ValidationError(f"[backend=torch] V2 at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-channel sums are not all 1.0 (or contain NaN/Inf)")

        Gz = _gz_block(abar, W, channel_slices) if gz_form == "block" else _gz_masked(abar, W, E)
        if blocked_p_cond:
            # Same three terms, same order, one fused broadcast: a (Bn,C,M,S) tensor IS the
            # (Bn,S,S) one (states are channel-contiguous), so `Gz` broadcasts along the
            # within-channel axis and the separate (Bn,S,S) `Gz_broadcast` never exists.
            p_cond = (W_blocked[None] + alpha_next[:, None, None, :] - Gz[:, :, None, :]).reshape(Bn, S, S)
        else:
            p_cond = W[None, :, :] + alpha_next[:, None, :] - broadcast_states(Gz)  # (Bn,S,S)

        beta_C = block_sum_last(p_cond * beta_next[:, None, :])  # (Bn,S,C)

        if C > 1:
            an = alpha_next[:, None, :]
            an_safe = torch.where(an != 0, an, torch.ones_like(an))
            if fused:
                # `p_cond` is dead after this point in the iteration, so `weighted`/`num` are
                # formed in its own storage: identical values and order of operations, two
                # fewer (Bn,S,S) allocations and their memory traffic per timestep.
                num = p_cond.mul_(abar[:, :, None]).pow_(2).div_(an_safe).masked_fill_(an == 0, 0.0)
            else:
                weighted = (p_cond * abar[:, :, None]) ** 2
                num = torch.where(an != 0, weighted / an_safe, torch.zeros_like(weighted))
            f = block_sum_both(num) + f_reg
            h = block_sum_last(abar**2)

            if check_all:
                finite_ok = torch.isfinite(f).all(dim=(1, 2)) & torch.isfinite(h).all(dim=1)
                if not bool((finite_ok | ~active_bw).all()):
                    loc = _first_failure(finite_ok, sequence_indices, tau, active=active_bw)
                    raise ValidationError(f"[backend=torch] f or h at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) contains NaN/Inf")
                f_gt_h = (f > h[:, :, None]).all(dim=(1, 2))
                if not bool((f_gt_h | ~active_bw).all()):
                    loc = _first_failure(f_gt_h, sequence_indices, tau, active=active_bw)
                    raise ValidationError(
                        f"[backend=torch] f<=h at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}): "
                        f"unreachable for valid parameters (§0.3.3) -- check coupling/transition stochasticity"
                    )

            f_le_h_violations += torch.where(active_bw, (f <= h[:, :, None]).sum(dim=(1, 2)), torch.zeros(Bn, dtype=torch.int64, device=alpha.device))
            delta = _solve_delta_batch(f, h)
            weights_subsystems = broadcast_channels(delta)  # (Bn,S,C)
        else:
            weights_subsystems = torch.ones(Bn, S, 1, dtype=alpha.dtype, device=alpha.device)

        raw = (beta_C * weights_subsystems).sum(dim=-1)
        beta_floor_hits += torch.where(active_bw, (raw < beta_floor).sum(dim=1), torch.zeros(Bn, dtype=torch.int64, device=alpha.device))
        raw = torch.clamp(raw, min=beta_floor)
        beta_tau = scale[:, :, tau] * raw

        if kernel.flag_nan:
            nan_mask = torch.isnan(beta_tau)
            nan_guard_hits += torch.where(active_bw, nan_mask.sum(dim=1), torch.zeros(Bn, dtype=torch.int64, device=alpha.device))
            beta_tau = torch.where(nan_mask, scale[:, :, tau], beta_tau)

        gamma_tau = beta_tau * alpha[:, :, tau]
        normalizer = broadcast_channels(block_sum_last(gamma_tau))
        beta[:, :, tau] = beta_tau / normalizer
        gamma[:, :, tau] = gamma_tau / normalizer

        if check_all:
            real_ts = tau < lengths
            ok = _check_block_sum_batched(gamma[:, :, tau], E)
            if not bool((ok | ~real_ts).all()):
                loc = _first_failure(ok, sequence_indices, tau, active=real_ts)
                raise ValidationError(f"[backend=torch] gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    diag_arrays = {
        "zero_alpha_next_count": zero_alpha_next_count.cpu(),
        "beta_floor_hits": beta_floor_hits.cpu(),
        "f_le_h_violations": f_le_h_violations.cpu(),
        "nan_guard_hits": nan_guard_hits.cpu(),
    }
    return beta, gamma, diag_arrays


# ==========================================================================
# Batched POISE backward (P12).
# ==========================================================================


def _backward_poise_batch(
    kernel: TorchKernel,
    alpha: "torch.Tensor",
    scale: "torch.Tensor",
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    check_all: bool,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, "torch.Tensor"]]:
    """`inference.poise.forward_backward_poise`'s backward loop and its optional
    one-shot refinement, batched over a leading `(Bn,)` axis.

    Like the fast branch, POISE's message needs only an `(Bn,S,C)` workspace --
    no `(Bn,S,S)` `p_cond`, no delta solve -- so this is structurally the cheaper
    of the two backward recursions on every device. The one-hot `E` contractions
    go through `_onehot_ops` (P8c), so POISE gets the same indexed/block-sum
    forms, and the same float64-on-Ada behaviour, as `sed`/`fast`.

    Ragged batches are masked exactly as `_backward_sed_batch` documents:
    `active_bw = tau < lengths-1` gates the recursion-step checks, the wider
    `real_ts = tau < lengths` gates `gamma`'s own block-sum check, and the
    refinement pass (a forward-direction loop) is gated by `real_ts` at its own
    `t`. POISE reports no diagnostic counters at all (no `beta_floor` clamp, no
    `f<=h` case, no NaN guard in the MATLAB POISE branch), so the returned
    counters are all zero -- `Diagnostics()`'s own defaults."""
    Bn, S, T = alpha.shape
    E = kernel.E
    broadcast_states, broadcast_channels, block_sum_last, _block_sum_both = _onehot_ops(kernel)

    rho, theta_out, zero_theta = _poise_constants(kernel)

    beta = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    gamma = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)

    # MATLAB's normalize_blocks guard (see inference/poise.check_channel_normalizer), which is
    # unconditional and so cannot cost a host synchronization per timestep here: the first
    # offending (sequence, channel) -> timestep is accumulated ON DEVICE and inspected exactly
    # once, below, before this function returns and therefore before any statistics reduction.
    # `-1` means "no failure"; the first write wins, and the loop runs backwards, so the
    # recorded timestep is the same one the NumPy reference raises at.
    C_ = kernel.E.shape[1] if kernel.E.dim() == 2 else int(rho.shape[1])
    no_fail = torch.full((Bn, C_), -1, dtype=torch.int64, device=alpha.device)
    raw_first_bad = no_fail.clone()
    refine_first_bad = no_fail.clone()

    def _raise_if_normalizer_failed():
        """One host transfer of both trackers, then pure host logic. Called once
        before returning; also called per timestep under `check_all`, where the
        surrounding invariant checks already synchronize every timestep anyway and
        reporting the *root cause* (the normalizer) beats reporting its downstream
        symptom (a gamma block-sum that is NaN because of it)."""
        both = torch.stack((raw_first_bad, refine_first_bad)).cpu().numpy()
        if not (both >= 0).any():
            return
        for which, stage in ((0, "backward recursion"), (1, "refinement")):
            hits = both[which]
            if (hits >= 0).any():
                b, c = (int(x) for x in np.argwhere(hits >= 0)[0])
                raise ValidationError(
                    f"[backend=torch] POISE {stage}: sequence {sequence_indices[b]} (batch index {b}), "
                    f"channel {c} normalizer at t={int(hits[b, c])} is nonpositive or non-finite; a "
                    f"channel's total smoothed mass must be finite and positive. Dividing by it would "
                    f"return NaN posteriors from a model whose forward log-likelihood is finite. This "
                    f"is MATLAB forward_backward_lsim_poise.m's own 'nonpositive or nonfinite channel "
                    f"normalizer' error."
                )
    beta[:, :, T - 1] = scale[:, :, T - 1]
    gamma[:, :, T - 1] = alpha[:, :, T - 1] * beta[:, :, T - 1]

    if check_all:
        real_terminal = (T - 1) < lengths
        ok = _check_block_sum_batched(gamma[:, :, T - 1], E)
        if not bool((ok | ~real_terminal).all()):
            loc = _first_failure(ok, sequence_indices, T - 1, active=real_terminal)
            raise ValidationError(f"[backend=torch] gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    for tau in range(T - 2, -1, -1):
        active_bw = tau < (lengths - 1)

        if check_all:
            abar = alpha[:, :, tau] * scale[:, :, tau]
            ok = _check_block_sum_batched(abar, E)
            if not bool((ok | ~active_bw).all()):
                loc = _first_failure(ok, sequence_indices, tau, active=active_bw)
                raise ValidationError(f"[backend=torch] abar at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

        beta_next = beta[:, :, tau + 1]
        child = block_sum_last(kernel.transition[None, :, :] * beta_next[:, None, :])  # (Bn,S,C)
        q = (child * rho[None, :, :]).sum(dim=-1)  # (Bn,S)
        beta_tau = scale[:, :, tau] * q ** theta_out[None, :]
        # the exact-zero outgoing-strength branch (a channel with no outgoing coupling)
        beta_tau = torch.where(zero_theta[None, :], scale[:, :, tau], beta_tau)

        # alpha_T from the UNNORMALIZED beta, then both divided by the same per-channel
        # normalizer -- the MATLAB POISE branch's own operation order (see inference/poise.py).
        alpha_T_tau = beta_tau * alpha[:, :, tau]
        normalizer_channel = block_sum_last(alpha_T_tau)  # (Bn,C)
        bad = (~torch.isfinite(normalizer_channel) | (normalizer_channel <= 0)) & (tau < lengths)[:, None]
        raw_first_bad = torch.where(bad & (raw_first_bad < 0), tau, raw_first_bad)
        if check_all:
            _raise_if_normalizer_failed()
        normalizer = broadcast_channels(normalizer_channel)
        beta[:, :, tau] = beta_tau / normalizer
        gamma[:, :, tau] = alpha_T_tau / normalizer

        if check_all:
            real_ts = tau < lengths
            ok = _check_block_sum_batched(gamma[:, :, tau], E)
            if not bool((ok | ~real_ts).all()):
                loc = _first_failure(ok, sequence_indices, tau, active=real_ts)
                raise ValidationError(f"[backend=torch] gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    if fb_constants.poise_refinement and T > 1:
        # `T` is the PADDED batch maximum, so `T > 1` only says *some* sequence is longer than one
        # timestep. Refinement is a per-sequence property: `inference/poise.py` (and MATLAB, whose
        # loop starts at t=2) skips the whole pass -- including the effective-beta reconstruction --
        # for a length-1 sequence, so a length-1 sequence sharing a chunk with longer ones must keep
        # its RAW beta. Without this gate its beta depended on the chunking, which is exactly what
        # test_cuda_f64_invariant_to_chunk_size_budget_order_and_repeats caught on `c3_sparse_l2r`.
        refine_seq = (lengths > 1)[:, None, None]  # (Bn,1,1)
        gamma_raw = gamma.clone()  # frozen: every propagated slice reads these, never a refined one
        w1 = torch.clamp(_POISE_REFINE_W0 - theta_out, min=0.0)  # (S,)
        for t in range(1, T):
            propagated = scale[:, :, t] * (gamma_raw[:, :, t - 1] @ kernel.W)
            propagated_norm = block_sum_last(propagated)  # (Bn,C)
            bad = (~torch.isfinite(propagated_norm) | (propagated_norm <= 0)) & (t < lengths)[:, None]
            refine_first_bad = torch.where(bad & (refine_first_bad < 0), t, refine_first_bad)
            if check_all:
                _raise_if_normalizer_failed()
            propagated = propagated / broadcast_channels(propagated_norm)
            gamma[:, :, t] = w1[None, :] * propagated + (1.0 - w1[None, :]) * gamma_raw[:, :, t]

            if check_all:
                real_ts = t < lengths
                ok = _check_block_sum_batched(gamma[:, :, t], E)
                if not bool((ok | ~real_ts).all()):
                    loc = _first_failure(ok, sequence_indices, t, active=real_ts)
                    raise ValidationError(f"[backend=torch] refined gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

        # effective message whose one-slice marginal is exactly the refined gamma -- only for the
        # sequences the refinement actually ran on (a length-1 sequence keeps its raw beta; its
        # t>=1 columns are padding, so its gamma needs no masking).
        positive = (alpha > 0) & refine_seq
        beta = torch.where(positive, gamma / torch.where(positive, alpha, torch.ones_like(alpha)),
                           torch.where(refine_seq, torch.zeros_like(gamma), beta))

    _raise_if_normalizer_failed()  # the single host synchronization in production mode

    zeros = torch.zeros(Bn, dtype=torch.int64)
    diag_arrays = {
        "zero_alpha_next_count": zeros.clone(),
        "beta_floor_hits": zeros.clone(),
        "f_le_h_violations": zeros.clone(),
        "nan_guard_hits": zeros.clone(),
    }
    return beta, gamma, diag_arrays


# ==========================================================================
# Batched fast/SVI backward.
# ==========================================================================


def _backward_fast_batch(
    kernel: TorchKernel,
    alpha: "torch.Tensor",
    scale: "torch.Tensor",
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    check_all: bool,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, "torch.Tensor"]]:
    """`inference.fast.forward_backward_fast`'s backward loop, batched.
    Returns per-*sequence* `nan_guard_hits` (see `_backward_sed_batch`'s own
    docstring for why -- including why `active_bw = tau < lengths[b]-1`
    gates it here too: a padding-only row must never contribute to another
    (or its own) sequence's reported `nan_guard_hits`).

    **`validate="all"` contract, matching `inference.fast`'s own exactly**
    -- a review round found this function previously accepted neither
    `sequence_indices` nor `check_all` and ran none of `fast`'s own
    `abar`/`gamma` invariant checks at all (only the shared `_forward_pass_
    batch` checked `alpha`), so a corrupted `abar`/`gamma` passed silently
    here while the `sed` backward correctly raised -- reproduced directly
    (a monkeypatched `_forward_pass_batch` returning a valid `alpha` but a
    doubled `scale` at one real timestep raises naming `abar` for
    `variant="sed"` but previously completed silently for `variant="fast"`).
    Fixed by mirroring `_backward_sed_batch`'s own masked checks exactly:
    terminal `gamma` is checked once before the loop (`real_terminal =
    (T-1) < lengths`); `abar` is checked at every real backward step
    (`active_bw = tau < lengths[b]-1` -- `fast`'s own NumPy reference
    computes `abar` *only* for this check, never for its own beta formula;
    the underlying block-sum-to-1 invariant holds regardless of which
    backward-message variant is in use, so the check is reproduced here
    verbatim for the same reason); `gamma` is checked at every real
    timestep (`real_ts = tau < lengths[b]`, the wider mask, since `gamma`
    is a meaningful, checkable output at the terminal timestep too, unlike
    `abar`, which is specifically about the backward *step*). Padding rows
    never raise and never contribute to another sequence's failure, exactly
    as `_backward_sed_batch`'s own masking does. `validate="params"`/`"off"`
    pass `check_all=False`, so none of this runs (matching `fast.py`'s own
    `if validate == "all":` gating)."""
    Bn, S, T = alpha.shape
    C = kernel.structure.C
    E = kernel.E
    _bc_states, broadcast_channels, block_sum_last, _bs_both = _onehot_ops(kernel)

    row_sum = kernel.coupling.sum(dim=1)
    weights = kernel.coupling / (fb_constants.svi_eps + row_sum[:, None])
    weights = torch.where(torch.isnan(weights), torch.full_like(weights, 1.0 / C), weights)
    w_opt = (fb_constants.svi_eps + kernel.coupling).sum(dim=1)
    weights_per_state = weights[kernel.block_of_state]  # (S,C)
    w_opt_per_state = w_opt[kernel.block_of_state]  # (S,)

    beta = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    gamma = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    beta[:, :, T - 1] = scale[:, :, T - 1]
    gamma[:, :, T - 1] = alpha[:, :, T - 1] * beta[:, :, T - 1]

    if check_all:
        real_terminal = (T - 1) < lengths
        ok = _check_block_sum_batched(gamma[:, :, T - 1], E)
        if not bool((ok | ~real_terminal).all()):
            loc = _first_failure(ok, sequence_indices, T - 1, active=real_terminal)
            raise ValidationError(f"[backend=torch] gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    nan_guard_hits = torch.zeros(Bn, dtype=torch.int64, device=alpha.device)
    for tau in range(T - 2, -1, -1):
        active_bw = tau < (lengths - 1)

        if check_all:
            abar = alpha[:, :, tau] * scale[:, :, tau]
            ok = _check_block_sum_batched(abar, E)
            if not bool((ok | ~active_bw).all()):
                loc = _first_failure(ok, sequence_indices, tau, active=active_bw)
                raise ValidationError(f"[backend=torch] abar at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

        beta_next = beta[:, :, tau + 1]
        beta_C = block_sum_last(kernel.transition[None, :, :] * beta_next[:, None, :])  # (Bn,S,C)
        mix = (beta_C * weights_per_state[None, :, :]).sum(dim=-1)
        beta_tau = scale[:, :, tau] * mix ** w_opt_per_state[None, :]

        nan_mask = torch.isnan(beta_tau)
        nan_guard_hits += torch.where(active_bw, nan_mask.sum(dim=1), torch.zeros(Bn, dtype=torch.int64, device=alpha.device))
        beta_tau = torch.where(nan_mask, scale[:, :, tau], beta_tau)

        alpha_T_temp = beta_tau * alpha[:, :, tau]
        normalizer = broadcast_channels(block_sum_last(alpha_T_temp))
        beta_tau = beta_tau / normalizer
        gamma_tau = beta_tau * alpha[:, :, tau]
        beta[:, :, tau] = beta_tau
        gamma[:, :, tau] = gamma_tau

        if check_all:
            real_ts = tau < lengths
            ok = _check_block_sum_batched(gamma[:, :, tau], E)
            if not bool((ok | ~real_ts).all()):
                loc = _first_failure(ok, sequence_indices, tau, active=real_ts)
                raise ValidationError(f"[backend=torch] gamma at t={loc[2]} (batch index {loc[0]}, original sequence index {loc[1]}) per-block sums are not all 1.0 (or contain NaN/Inf)")

    diag_arrays = {
        "zero_alpha_next_count": torch.zeros(Bn, dtype=torch.int64),
        "beta_floor_hits": torch.zeros(Bn, dtype=torch.int64),
        "f_le_h_violations": torch.zeros(Bn, dtype=torch.int64),
        "nan_guard_hits": nan_guard_hits.cpu(),
    }
    return beta, gamma, diag_arrays


# ==========================================================================
# One chunk, end to end: emissions -> forward -> backward -> CPU transfer.
# ==========================================================================


@dataclass
class _ChunkResult:
    alpha: NDArray[np.float64]  # (Bn,S,T) CPU float64
    beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    scale: NDArray[np.float64]
    B: NDArray[np.float64]
    B_comp: NDArray[np.float64]
    log_scale: NDArray[np.float64]  # (Bn,C,T)
    loglik: NDArray[np.float64]  # (Bn,)
    diag_arrays: dict[str, NDArray[np.int64]]  # each (Bn,)


def _check_nan_loglik(loglik: "torch.Tensor", sequence_indices: list[int]) -> None:
    if torch.isnan(loglik).any():
        bad = torch.nonzero(torch.isnan(loglik), as_tuple=False)
        local = int(bad[0, 0].item())
        raise ValidationError(
            f"[backend=torch] forward pass produced a NaN log-likelihood (batch index {local}, "
            f"original sequence index {sequence_indices[local]})"
        )


def _to_cpu_f64(x: "torch.Tensor") -> NDArray[np.float64]:
    # Two separate steps, not one combined `.to(device, dtype=...)` call:
    # MPS has no float64 support at all, including as a same-call conversion
    # target while data is still device-resident -- casting only *after*
    # the tensor has landed on CPU avoids ever asking MPS to materialize
    # float64 itself.
    return x.detach().to("cpu").to(dtype=torch.float64).numpy()


def _run_chunk_loglik_only(
    kernel: TorchKernel,
    Y_chunk: "torch.Tensor",  # (Bn,D,T_max), NaN-padded
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    *,
    validate: str,
) -> NDArray[np.float64]:
    """`mode="loglik"`'s own execution path -- forward pass only, genuinely:
    no backward recursion, no `alpha`/`beta`/`gamma`/`B`/`B_comp` transfer to
    CPU beyond the `(Bn,)` loglik values themselves. A review round found
    the previous version ran the *entire* posterior computation (backward
    pass included) for every mode, including this one, transferring and then
    discarding everything but `loglik` -- this function is the fix, not a
    thin wrapper around the full path."""
    check_all = validate == "all"
    with torch.inference_mode():
        B_resc, _B_comp_resc, log_scale = compute_emissions_torch(kernel, Y_chunk)
        _alpha, _scale, loglik_rescaled = _forward_pass_batch(kernel, B_resc, sequence_indices, lengths, check_all)
        loglik = loglik_rescaled + log_scale.sum(dim=(1, 2))
        _check_nan_loglik(loglik, sequence_indices)
        return _to_cpu_f64(loglik)


def _compute_forward_backward_device(
    kernel: TorchKernel,
    Y_chunk: "torch.Tensor",  # (Bn,D,T_max), NaN-padded
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    *,
    check_all: bool,
    gz_form: str,
):
    """The shared forward+backward computation, entirely on-device, shared
    by `_run_chunk` (`mode="posterior"`, transfers everything to CPU) and
    `_run_chunk_stats_only` (`mode="stats"`, reduces on-device first and
    transfers only the small, fixed-size reduced statistics -- never
    `alpha`/`beta`/`gamma`/`B`/`B_comp` themselves)."""
    B_resc, B_comp_resc, log_scale = compute_emissions_torch(kernel, Y_chunk)
    alpha, scale, loglik_rescaled = _forward_pass_batch(kernel, B_resc, sequence_indices, lengths, check_all)
    loglik_correction = log_scale.sum(dim=(1, 2))
    if fb_constants.variant == "poise":
        beta, gamma, diag_arrays_t = _backward_poise_batch(kernel, alpha, scale, fb_constants, sequence_indices, lengths, check_all)
    elif fb_constants.variant == "sed":
        beta, gamma, diag_arrays_t = _backward_sed_batch(kernel, alpha, scale, fb_constants, sequence_indices, lengths, check_all, gz_form)
    else:
        beta, gamma, diag_arrays_t = _backward_fast_batch(kernel, alpha, scale, fb_constants, sequence_indices, lengths, check_all)
    loglik = loglik_rescaled + loglik_correction
    _check_nan_loglik(loglik, sequence_indices)
    return alpha, beta, gamma, scale, B_resc, B_comp_resc, log_scale, loglik, diag_arrays_t


def _run_chunk(
    kernel: TorchKernel,
    Y_chunk: "torch.Tensor",  # (Bn,D,T_max), NaN-padded
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    *,
    validate: str,
    gz_form: str,
) -> _ChunkResult:
    """Full forward+backward path -- used by `mode="posterior"`, which
    genuinely needs `alpha`/`beta`/`gamma`/`B`/`B_comp` transferred to CPU
    (`mode="loglik"` uses `_run_chunk_loglik_only` instead, skipping the
    backward pass entirely; `mode="stats"` uses `_run_chunk_stats_only`,
    which reduces on-device before ever transferring anything this large)."""
    check_all = validate == "all"
    with torch.inference_mode():
        alpha, beta, gamma, scale, B_resc, B_comp_resc, log_scale, loglik, diag_arrays_t = _compute_forward_backward_device(
            kernel, Y_chunk, fb_constants, sequence_indices, lengths, check_all=check_all, gz_form=gz_form
        )

        alpha_np = _to_cpu_f64(alpha)
        beta_np = _to_cpu_f64(beta)
        gamma_np = _to_cpu_f64(gamma)
        scale_np = _to_cpu_f64(scale)
        B_np = _to_cpu_f64(B_resc)
        B_comp_np = _to_cpu_f64(B_comp_resc)
        log_scale_np = _to_cpu_f64(log_scale)
        loglik_np = _to_cpu_f64(loglik)
        diag_arrays = {k: v.numpy() for k, v in diag_arrays_t.items()}

    return _ChunkResult(
        alpha=alpha_np, beta=beta_np, gamma=gamma_np, scale=scale_np, B=B_np, B_comp=B_comp_np,
        log_scale=log_scale_np, loglik=loglik_np, diag_arrays=diag_arrays,
    )


def _run_chunk_stats_only(
    kernel: TorchKernel,
    Y_chunk: "torch.Tensor",
    fb_constants: FbConstants,
    sequence_indices: list[int],
    lengths: "torch.Tensor",
    *,
    validate: str,
    gz_form: str,
    sigma_diag: bool,
) -> tuple[dict, float]:
    """`mode="stats"`'s own execution path: forward+backward on device
    (batched, once for the whole chunk), then `_reduce_chunk_stats_device`
    reduces directly from the device tensors -- `alpha`/`beta`/`gamma`/`B`/
    `B_comp` are never transferred to CPU at all; only the small, fixed-size
    reduced dict (and the scalar `loglik`) cross back."""
    check_all = validate == "all"
    with torch.inference_mode():
        alpha, beta, gamma, scale, B_resc, B_comp_resc, log_scale, loglik, _diag = _compute_forward_backward_device(
            kernel, Y_chunk, fb_constants, sequence_indices, lengths, check_all=check_all, gz_form=gz_form
        )
        reduced = _reduce_chunk_stats_device(kernel, Y_chunk, alpha, beta, gamma, scale, B_resc, B_comp_resc, lengths, sigma_diag)
        loglik_total = float(loglik.detach().to("cpu").to(dtype=torch.float64).sum().item())
    return reduced, loglik_total


# ==========================================================================
# BatchedPosteriorChunk: the mode="posterior" public result type.
# ==========================================================================


@dataclass
class BatchedPosteriorChunk:
    """One chunk's worth of batched posterior results -- `mode="posterior"`
    yields a stream of these (`forward_backward_batch` is a generator in
    that mode), never concatenating all `N` sequences' posteriors into one
    array. `alpha`/`beta`/`gamma`/`scale`/`B`/`B_comp`/`log_scale` are
    `(chunk, S, T_max)` (or `(chunk, C, T_max)` for `log_scale`) CPU
    `float64` NumPy arrays, `T_max` this chunk's own longest sequence --
    shorter sequences are stored padded (per this module's own ragged-length
    convention, see the module docstring) and must be trimmed via
    `extract()`, not read directly past their own `lengths[i]`.

    `sequence_indices[i]` is sequence `i`'s index in the *original* `dataset`
    list passed to `forward_backward_batch` -- chunks are bucketed by length
    for efficiency (§5 of the P8a request), so this chunk's local order is
    *not* the original dataset order; callers that need original order
    consult this list explicitly (`mode="loglik"`'s own `(N,)` array is
    already restored to original order internally, precisely so its callers
    don't have to)."""

    structure: LSIMStructure
    Y: NDArray[np.float64]  # (chunk,D,T_max) original (NaN-padded) observations -- not part of Posterior's own contract, kept here for mode="stats"'s reduction, which needs the raw observations, not just the posterior derived from them
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    scale: NDArray[np.float64]
    B: NDArray[np.float64]
    B_comp: NDArray[np.float64]
    log_scale: NDArray[np.float64]
    loglik: NDArray[np.float64]
    diag_arrays: dict[str, NDArray[np.int64]]
    sequence_indices: list[int]
    lengths: list[int]

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def extract(self, local_index: int) -> Posterior:
        """The documented way to recover one sequence's ordinary, trimmed
        `Posterior` from this chunk -- `local_index` indexes into *this
        chunk* (`0 <= local_index < len(self)`), not the original dataset;
        use `self.sequence_indices[local_index]` to recover the original
        dataset index this corresponds to."""
        if not (0 <= local_index < len(self)):
            raise IndexError(f"local_index={local_index} out of range for chunk of size {len(self)}")
        Ti = self.lengths[local_index]
        diagnostics = Diagnostics(**{k: int(v[local_index]) for k, v in self.diag_arrays.items()})
        return Posterior(
            loglik=float(self.loglik[local_index]),
            alpha=self.alpha[local_index, :, :Ti],
            beta=self.beta[local_index, :, :Ti],
            gamma=self.gamma[local_index, :, :Ti],
            scale=self.scale[local_index, :, :Ti],
            B=self.B[local_index, :, :Ti],
            B_comp=self.B_comp[local_index, :, :, :Ti],  # B_comp is (chunk,S,max_K,T) -- 4 axes need 4 index terms, or :Ti trims max_K instead of T
            log_scale=self.log_scale[local_index, :, :Ti],
            diagnostics=diagnostics,
            structure=self.structure,
        )


# ==========================================================================
# Dataset validation, bucketing, and the memory estimator (§5).
# ==========================================================================


def _validate_dataset(structure: LSIMStructure, dataset: list[NDArray[np.float64]]) -> None:
    """Every sequence through `validate_observations`'s checks and messages
    (prefixed `dataset[i]: `), chunk-vectorized (P10f:
    `validate.validate_observation_list`) instead of one Python-level call per
    sequence -- which cost more than the CUDA forward pass itself at N=1e6. A
    `_ValidatedTrials` snapshot built for this same `structure` by the
    mini-batch training drivers was already validated once at their boundary
    and is not re-validated here."""
    if len(dataset) == 0:
        raise ValidationError("dataset is empty -- forward_backward_batch requires at least one sequence")
    if _is_prevalidated(structure, dataset):
        return
    validate_observation_list(structure, dataset, label="dataset")


def _bucket_by_length(dataset: list[NDArray[np.float64]], chunk_size: int) -> list[list[int]]:
    """Sorts original dataset indices by sequence length (so a chunk's own
    `T_max` -- and therefore its padding overhead -- stays small), then
    slices into `chunk_size`-sized buckets. Original order is restored by
    every public result type afterward (`mode="loglik"`'s `(N,)` array
    directly; `mode="posterior"`'s chunks carry `sequence_indices` for the
    caller to do so; `mode="stats"` is order-invariant by construction)."""
    order = sorted(range(len(dataset)), key=lambda i: dataset[i].shape[1])
    return [order[i : i + chunk_size] for i in range(0, len(order), chunk_size)]


_ALLOCATOR_MARGIN = 1.5  # extra headroom for allocator fragmentation/overhead, on top of the itemized estimate below


def _validate_max_memory_bytes(max_memory_bytes: object) -> None:
    """Shared boundary check for `max_memory_bytes`, used both by
    `estimate_chunk_size` (the automatic-`chunk_size` path) and directly by
    `forward_backward_batch` (so an *explicit* `chunk_size` no longer
    bypasses this check -- a review round found `max_memory_bytes=0`/`1.5`
    were silently ignored whenever `chunk_size` was also given, since
    `estimate_chunk_size`, the only place this used to be checked, was never
    even called in that case). `bool` is rejected even though it is
    technically an `int` subclass, matching `chunk_size`'s own validation."""
    if max_memory_bytes is not None and (
        not isinstance(max_memory_bytes, int) or isinstance(max_memory_bytes, bool) or max_memory_bytes <= 0
    ):
        raise ValidationError(f"max_memory_bytes={max_memory_bytes!r}, expected a positive integer (bool is not accepted, even though it is technically an int subclass)")


def _estimate_per_seq_bytes(S: int, T_max: int, max_K: int, D: int, max_d: int, dtype: "torch.dtype") -> int:
    """The itemized peak-concurrent-bytes-per-sequence estimate, factored
    out of `estimate_chunk_size` so `forward_backward_batch` can also use it
    to cross-check an *explicit* `chunk_size` against `max_memory_bytes`
    (see `estimate_chunk_size`'s own docstring for what each term below
    accounts for -- unchanged here, just shared)."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    per_seq_elems = (
        5 * S * T_max  # alpha, beta, gamma, scale, B (rescaled)
        + 2 * S * max_K * T_max  # log_B_comp (transient) + B_comp (rescaled) -- coexist, not sequential
        + S * T_max  # log_scale/max_log, broadcast to S (upper bound on the true (C,T) shape, C<=S)
        + D * T_max  # the raw input Y_chunk itself
        + 4 * S * S  # sed backward: p_cond, Gz_broadcast, weighted, num -- simultaneously alive per timestep
        + 2 * S * S  # beta_C/weights_subsystems-scale (S,C) terms, upper-bounded by (S,S) since C<=S
        + 3 * max_d * max_d * T_max  # mode="stats" full-covariance pairwise: Y_diff, Y_diff_partner, weight_full
    )
    return int(per_seq_elems * itemsize * _ALLOCATOR_MARGIN)


def estimate_chunk_size(
    N: int,
    S: int,
    T_max: int,
    max_K: int,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
    max_memory_bytes: int | None,
    D: int = 0,
    max_d: int = 0,
) -> int:
    """Overflow-safe chunk-size estimator, computed *before* any allocation
    -- a conservative *peak concurrent* estimate, not a tight bound. A
    review round found the previous version omitted several materially-
    sized concurrent tensors, making its CUDA branch (80% of
    `torch.cuda.mem_get_info()`'s free bytes) unsafe: the raw input
    `Y_chunk` (`(Bn,D,T)`); the log-domain emission intermediate
    `log_B_comp` (`(Bn,S,max_K,T)`), which briefly coexists with the final,
    exponentiated `B_comp` of the *same* shape during
    `compute_emissions_torch` (2x that term, not 1x); the *several*
    simultaneously-alive `(Bn,S,S)`-shaped intermediates in `sed`'s backward
    step (`p_cond`, `Gz_broadcast`, `weighted`, `num` -- 4x `S*S`, not the
    1x the previous version counted for `p_cond` alone); and `mode="stats"`'s
    own `_pairwise_batch_torch` intermediates (`Y_diff`/`Y_diff_partner`/
    `weight_full`, each `(max_d,max_d,T)` per sequence, for the
    full-covariance case) -- all itemized below now, plus a blanket
    `_ALLOCATOR_MARGIN` (50% extra) on top for allocator fragmentation and
    anything still not itemized by name.

    - `device="cuda"`: `min(0.8 * torch.cuda.mem_get_info()'s free bytes,
      max_memory_bytes)` when `max_memory_bytes` is given, else just the
      `80%`-of-free-bytes figure alone. A review round found the previous
      version used `80%` of free CUDA memory unconditionally and never
      applied a smaller user-supplied `max_memory_bytes` on this branch at
      all -- silently *not* enforcing a public, documented parameter on
      `device="cuda"` specifically, even though the exact same parameter
      was correctly enforced on `cpu`/`mps`. Fixed here (mocked CUDA test,
      via monkeypatching `torch.cuda.mem_get_info` -- no physical CUDA
      device is required to verify this specific calculation) rather than
      merely documented as a P8b blocker, since `max_memory_bytes` is a
      public parameter and this device-dependent gap in enforcing it would
      otherwise be silent. Real CUDA correctness/safety is still P8b-only in
      practice -- this fix means the *stated user limit* is honored on
      `cuda` too, not that the itemized estimate itself has been validated
      against real GPU allocator behavior yet.
    - `device="cpu"`/`"mps"`: `max_memory_bytes` if given, else
      `_DEFAULT_CPU_MEMORY_BUDGET_BYTES` (512 MiB) -- a documented,
      conservative default, not a measurement of actual available memory.

    Raises `ValidationError` if `max_memory_bytes` is given but is not a
    positive integer (`bool` included), or if the resulting chunk size would
    be `< 1` (before any allocation happens, not after)."""
    _validate_max_memory_bytes(max_memory_bytes)
    per_seq_bytes = _estimate_per_seq_bytes(S, T_max, max_K, D, max_d, dtype)

    if device.type == "cuda":
        free_bytes, _total = torch.cuda.mem_get_info()
        cuda_budget = int(free_bytes * _CUDA_MEMORY_HEADROOM)
        budget = min(cuda_budget, max_memory_bytes) if max_memory_bytes is not None else cuda_budget
    else:
        budget = max_memory_bytes if max_memory_bytes is not None else _DEFAULT_CPU_MEMORY_BUDGET_BYTES

    chunk_size = budget // max(per_seq_bytes, 1)
    if chunk_size < 1:
        raise ValidationError(
            f"estimate_chunk_size: even a single sequence's estimated peak memory ({per_seq_bytes} bytes, "
            f"S={S}, T_max={T_max}, max_K={max_K}, D={D}, max_d={max_d}) exceeds the available budget "
            f"({budget} bytes) -- refusing to compute a chunk_size < 1 before any allocation; reduce T_max, "
            f"raise max_memory_bytes, or pass chunk_size explicitly if you have verified it fits"
        )
    return int(min(chunk_size, N))


# ==========================================================================
# mode="stats": one chunk's raw ExpectedStats, via the GMM accumulator (emissions/gmm.py).
# ==========================================================================


def _weighted_batch_torch(weight: "torch.Tensor", value: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Direct Torch port of `emissions.gmm._welford_batch`, reducing over *every*
    trailing axis after the leading "quantity" axis at once (e.g. `(d_c,
    Bn, T)` -> `(d_c,)`) -- one call handles a whole chunk's worth of
    sequences and timesteps together, not one sequence at a time. Central,
    not raw, moments throughout (`Σw(y-mean)²`, never `Σwy²`), matching
    the NumPy accumulator's stability requirement exactly, just batched
    wider. `weight`/`value` must already agree in shape; `value` must be
    NaN-free at every position `weight` doesn't count (zero-filled)."""
    reduce_dims = tuple(range(1, weight.ndim))
    W = weight.sum(dim=reduce_dims)
    num = (weight * value).sum(dim=reduce_dims)
    mean = torch.where(W > 0, num / torch.where(W > 0, W, torch.ones_like(W)), torch.zeros_like(W))
    shape = [mean.shape[0]] + [1] * (weight.ndim - 1)
    diff = value - mean.reshape(shape)
    M2 = (weight * diff**2).sum(dim=reduce_dims)
    return W, mean, M2


def _pairwise_batch_torch(
    weight: "torch.Tensor", o: "torch.Tensor", Y_masked: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Direct Torch port of `emissions.gmm._pairwise_batch`: `weight` is `(Bn,T)`, `o`/
    `Y_masked` are `(d_c,Bn,T)` -- reduces over `(Bn,T)` jointly (flattened),
    for a whole chunk at once. `Q[i,j]=Σ w·o_i·o_j`, `M[i,j]` the weighted
    mean of `y_i` under that joint mask, `C[i,j]=Σ w·o_i·o_j·(y_i-M[i,j])·
    (y_j-M[j,i])` -- central, never a raw `y_i·y_j` moment."""
    d_c, Bn, T = o.shape
    o_f = o.reshape(d_c, Bn * T).to(Y_masked.dtype)
    Y_flat = Y_masked.reshape(d_c, Bn * T)
    w_flat = weight.reshape(1, Bn * T)
    wo = w_flat * o_f  # (d_c, N)
    Q = wo @ o_f.T  # (d_c,d_c)
    num_M = (wo * Y_flat) @ o_f.T
    M = torch.where(Q > 0, num_M / torch.where(Q > 0, Q, torch.ones_like(Q)), torch.zeros_like(Q))
    Y_diff = Y_flat[:, None, :] - M[:, :, None]  # (d_c,d_c,N): [i,j,n] = y_i(n) - M[i,j]
    Y_diff_partner = Y_diff.permute(1, 0, 2)  # [i,j,n] = y_j(n) - M[j,i]
    weight_full = w_flat[None, :, :] * o_f[:, None, :] * o_f[None, :, :]  # (d_c,d_c,N)
    C = (weight_full * Y_diff * Y_diff_partner).sum(dim=2)
    return Q, M, C


def _reduce_chunk_stats_device(
    kernel: TorchKernel,
    Y_chunk: "torch.Tensor",  # (Bn,D,T_max) raw, NaN-padded observations, still on device
    alpha: "torch.Tensor", beta: "torch.Tensor", gamma: "torch.Tensor", scale: "torch.Tensor",
    B: "torch.Tensor", B_comp: "torch.Tensor",  # rescaled (log-domain) -- their ratio is scale-invariant, see module docstring
    lengths: "torch.Tensor",  # (Bn,) int64
    sigma_diag: bool,
) -> dict:
    """Reduces one chunk's already-on-device forward-backward output into
    fixed-size raw `ExpectedStats`/`ChannelExpectedStats` fields, entirely
    with Torch tensor ops -- no per-sequence Python loop, no CPU transfer
    until the very end, when only the small, fixed-size *reduced* arrays
    (never `alpha`/`beta`/`gamma`/`B`/`B_comp` themselves) cross back to
    NumPy. Returns a plain dict of NumPy arrays/Python scalars, matching
    `ExpectedStats`/`ChannelExpectedStats`'s own field names, for the caller
    to assemble into those dataclasses directly (kept as a dict here rather
    than constructing the dataclasses on-device, since the dataclasses are
    plain NumPy containers with no Torch awareness of their own).

    Padding is *not* self-masking for every quantity here the way it is for
    the recursion itself (see the module docstring): a fully-missing
    (padded) channel-timestep gives `B_comp/B` a *finite*, nonzero ratio
    (`weight[s,k]`, not `NaN`), so `valid_mask = t < lengths[b]` is applied
    explicitly wherever a padded timestep could otherwise contribute a
    spurious nonzero amount -- everywhere except the plain `o = ~isnan(Y_c)`
    mask, which *is* already correctly zero at padded positions (they are
    genuinely `NaN` there, by this module's own padding convention)."""
    structure = kernel.structure
    Bn, S, T = alpha.shape
    C = structure.C
    device, dtype = alpha.device, alpha.dtype
    t_idx = torch.arange(T, device=device)
    valid_mask = t_idx[None, :] < lengths[:, None]  # (Bn,T)
    valid_f = valid_mask.to(dtype)

    T_all = int(lengths.sum().item())
    n_trials = Bn

    gamma_first = gamma[:, :, 0].sum(dim=0)  # t=0 is real for every sequence (length>=1 always)
    gamma_sum = (gamma * valid_f[:, None, :]).sum(dim=(0, 2))

    if T > 1:
        valid_transition = (t_idx[:-1][None, :] < (lengths[:, None] - 1)).to(dtype)  # (Bn,T-1): transition t->t+1 real iff t+1<lengths[b]
        abar = alpha[:, :, :-1] * scale[:, :, :-1] * valid_transition[:, None, :]
        beta_next = beta[:, :, 1:]
        zeta = torch.einsum("bit,bjt->ij", abar, beta_next)
        n_transitions = int(valid_transition.sum().item())
    else:
        zeta = torch.zeros(S, S, dtype=dtype, device=device)
        n_transitions = 0

    channels_out = []
    for c in range(C):
        M_c = structure.channel_state_num[c]
        K_c = structure.num_gmm_component[c]
        d_c = structure.channel_obs_dim[c]
        s0 = int(structure.state_offsets[c])
        Y_c = Y_chunk[:, structure.dim_slice(c), :]  # (Bn,d_c,T)
        o = ~torch.isnan(Y_c)  # (Bn,d_c,T) -- already correctly False at padded positions
        o_f = o.to(dtype)
        Y_masked = torch.where(o, Y_c, torch.zeros_like(Y_c))
        Y_masked_dcbt = Y_masked.permute(1, 0, 2)  # (d_c,Bn,T), matching _pairwise_batch_torch's own layout
        o_dcbt = o.permute(1, 0, 2)

        n_obs, welford_mean, welford_M2 = _weighted_batch_torch(o_f.permute(1, 0, 2), Y_masked_dcbt)
        allzero_weight = valid_f[None, :, :].expand(d_c, Bn, T)
        allzero_n, allzero_mean, allzero_M2 = _weighted_batch_torch(allzero_weight, Y_masked_dcbt)

        if not sigma_diag:
            u_pairwise_Q, u_pairwise_M, u_pairwise_C = _pairwise_batch_torch(valid_f, o_dcbt, Y_masked_dcbt)

        resp = torch.zeros(M_c, K_c, dtype=dtype, device=device)
        resp_pos = torch.zeros(M_c, K_c, dtype=torch.int64, device=device)
        n_w_pos_diag = torch.zeros(M_c, K_c, dtype=torch.int64, device=device)
        n_w_pos_full = torch.zeros(M_c, K_c, dtype=torch.int64, device=device)
        if sigma_diag:
            weighted_W = torch.zeros(M_c, K_c, d_c, dtype=dtype, device=device)
            weighted_mean = torch.zeros(M_c, K_c, d_c, dtype=dtype, device=device)
            weighted_M2 = torch.zeros(M_c, K_c, d_c, dtype=dtype, device=device)
        else:
            pairwise_Q = torch.zeros(M_c, K_c, d_c, d_c, dtype=dtype, device=device)
            pairwise_M = torch.zeros(M_c, K_c, d_c, d_c, dtype=dtype, device=device)
            pairwise_C = torch.zeros(M_c, K_c, d_c, d_c, dtype=dtype, device=device)

        for s in range(M_c):
            state_idx = s0 + s
            gamma_state = gamma[:, state_idx, :]  # (Bn,T)
            B_state = B[:, state_idx, :]
            for k in range(K_c):
                B_comp_sk = B_comp[:, state_idx, k, :]
                ratio = torch.where(B_state != 0, B_comp_sk / torch.where(B_state != 0, B_state, torch.ones_like(B_state)), torch.full_like(B_state, float("nan")))
                g = gamma_state * ratio
                g = torch.where(torch.isnan(g), torch.zeros_like(g), g)
                g = g * valid_f  # padded positions give a *finite*, nonzero ratio (weight[s,k]/1) -- must be masked explicitly, unlike o's own NaN-self-masking

                resp[s, k] = g.sum()
                resp_pos[s, k] = (g > _LOW_MASS_GAMMA_THRESHOLD).sum()
                n_w_pos_diag[s, k] = (o_dcbt & (g[None, :, :] > 0)).sum()
                n_w_pos_full[s, k] = d_c * (g > 0).sum()  # kept on device: `.item()` here forced one CPU/GPU sync per (state, component)

                if sigma_diag:
                    gW, gmean, gM2 = _weighted_batch_torch((g[None, :, :] * o_dcbt).to(dtype), Y_masked_dcbt)
                    weighted_W[s, k], weighted_mean[s, k], weighted_M2[s, k] = gW, gmean, gM2
                else:
                    gQ, gM, gC = _pairwise_batch_torch(g, o_dcbt, Y_masked_dcbt)
                    pairwise_Q[s, k], pairwise_M[s, k], pairwise_C[s, k] = gQ, gM, gC

        def _np(x):
            return x.detach().to("cpu").to(dtype=torch.float64).numpy()

        def _np_i(x):
            return x.detach().cpu().numpy()

        channels_out.append(
            dict(
                resp=_np(resp), resp_pos=_np_i(resp_pos), n_w_pos_diag=_np_i(n_w_pos_diag), n_w_pos_full=_np_i(n_w_pos_full),
                weighted_W=_np(weighted_W) if sigma_diag else None,
                weighted_mean=_np(weighted_mean) if sigma_diag else None,
                weighted_M2=_np(weighted_M2) if sigma_diag else None,
                pairwise_Q=None if sigma_diag else _np(pairwise_Q),
                pairwise_M=None if sigma_diag else _np(pairwise_M),
                pairwise_C=None if sigma_diag else _np(pairwise_C),
                u_pairwise_Q=None if sigma_diag else _np(u_pairwise_Q),
                u_pairwise_M=None if sigma_diag else _np(u_pairwise_M),
                u_pairwise_C=None if sigma_diag else _np(u_pairwise_C),
                n_obs=_np(n_obs), welford_mean=_np(welford_mean), welford_M2=_np(welford_M2),
                allzero_n=_np(allzero_n), allzero_mean=_np(allzero_mean), allzero_M2=_np(allzero_M2),
            )
        )

    return dict(
        zeta=zeta.detach().to("cpu").to(dtype=torch.float64).numpy(),
        gamma_first=gamma_first.detach().to("cpu").to(dtype=torch.float64).numpy(),
        gamma_sum=gamma_sum.detach().to("cpu").to(dtype=torch.float64).numpy(),
        n_trials=n_trials, n_transitions=n_transitions, T_all=T_all,
        channels=channels_out,
    )


def _chunk_dict_to_expected_stats(structure: LSIMStructure, sigma_diag: bool, W_np: NDArray[np.float64], d: dict, loglik: float) -> ExpectedStats:
    channels = [
        ChannelExpectedStats(
            resp=cd["resp"], resp_pos=cd["resp_pos"], n_w_pos_diag=cd["n_w_pos_diag"], n_w_pos_full=cd["n_w_pos_full"],
            weighted_W=cd["weighted_W"], weighted_mean=cd["weighted_mean"], weighted_M2=cd["weighted_M2"],
            pairwise_Q=cd["pairwise_Q"], pairwise_M=cd["pairwise_M"], pairwise_C=cd["pairwise_C"],
            u_pairwise_Q=cd["u_pairwise_Q"], u_pairwise_M=cd["u_pairwise_M"], u_pairwise_C=cd["u_pairwise_C"],
            n_obs=cd["n_obs"], welford_mean=cd["welford_mean"], welford_M2=cd["welford_M2"],
            allzero_n=cd["allzero_n"], allzero_mean=cd["allzero_mean"], allzero_M2=cd["allzero_M2"],
        )
        for cd in d["channels"]
    ]
    return ExpectedStats(
        zeta=d["zeta"] * W_np, gamma_first=d["gamma_first"], gamma_sum=d["gamma_sum"],
        n_trials=d["n_trials"], n_transitions=d["n_transitions"], T_all=d["T_all"], loglik=loglik, channels=channels,
    )


# ==========================================================================
# Public API: forward_backward_batch.
# ==========================================================================

_VALID_MODES = ("loglik", "posterior", "stats")
_VALID_GZ_FORMS = ("block", "masked")


def _pad_chunk(dataset: list[NDArray[np.float64]], indices: list[int], D: int) -> tuple[NDArray[np.float64], list[int]]:
    lengths = [dataset[i].shape[1] for i in indices]
    T_max = max(lengths)
    Y_chunk = np.full((len(indices), D, T_max), np.nan, dtype=np.float64)
    for local_i, orig_i in enumerate(indices):
        Ti = lengths[local_i]
        Y_chunk[local_i, :, :Ti] = dataset[orig_i]
    return Y_chunk, lengths


def _iter_chunks(
    structure: LSIMStructure,
    params: LSIMParams,
    dataset: list[NDArray[np.float64]],
    fb_constants: FbConstants,
    *,
    kernel: TorchKernel,
    chunk_size: int,
    validate: str,
    gz_form: str,
):
    """`mode="posterior"`/`"stats"`'s chunk iterator -- full forward+backward,
    via `_run_chunk`. `mode="loglik"` uses `_iter_loglik_chunks` instead,
    which never runs the backward pass at all."""
    buckets = _bucket_by_length(dataset, chunk_size)
    for bucket_indices in buckets:
        Y_chunk_np, lengths = _pad_chunk(dataset, bucket_indices, structure.D)
        Y_chunk_t = torch.as_tensor(Y_chunk_np, dtype=kernel.dtype, device=kernel.device)
        lengths_t = torch.as_tensor(lengths, dtype=torch.int64, device=kernel.device)
        result = _run_chunk(kernel, Y_chunk_t, fb_constants, bucket_indices, lengths_t, validate=validate, gz_form=gz_form)
        chunk = BatchedPosteriorChunk(
            structure=structure, Y=Y_chunk_np,
            alpha=result.alpha, beta=result.beta, gamma=result.gamma, scale=result.scale,
            B=result.B, B_comp=result.B_comp, log_scale=result.log_scale, loglik=result.loglik,
            diag_arrays=result.diag_arrays, sequence_indices=list(bucket_indices), lengths=lengths,
        )
        yield chunk


def _iter_loglik_chunks(
    structure: LSIMStructure,
    dataset: list[NDArray[np.float64]],
    fb_constants: FbConstants,
    *,
    kernel: TorchKernel,
    chunk_size: int,
    validate: str,
):
    """`mode="loglik"`'s own chunk iterator -- forward pass only, per chunk;
    yields `(bucket_indices, loglik_chunk_np)` pairs, never a
    `BatchedPosteriorChunk`. Precisely what this avoids, corrected wording
    (the emission tensors `B`/`B_comp` and the forward-pass `alpha`/`scale`
    are still computed on-device -- the forward pass genuinely needs them):
    the backward recursion, `beta`/`gamma` construction, and any transfer of
    `alpha`/`beta`/`gamma`/`B`/`B_comp` to CPU never happen in this mode --
    only the `(Bn,)` `loglik` values themselves ever leave the device."""
    buckets = _bucket_by_length(dataset, chunk_size)
    for bucket_indices in buckets:
        Y_chunk_np, lengths = _pad_chunk(dataset, bucket_indices, structure.D)
        Y_chunk_t = torch.as_tensor(Y_chunk_np, dtype=kernel.dtype, device=kernel.device)
        lengths_t = torch.as_tensor(lengths, dtype=torch.int64, device=kernel.device)
        loglik_np = _run_chunk_loglik_only(kernel, Y_chunk_t, fb_constants, bucket_indices, lengths_t, validate=validate)
        yield bucket_indices, loglik_np


def forward_backward_batch(
    structure: LSIMStructure,
    params: LSIMParams,
    dataset: list[NDArray[np.float64]],
    fb_constants: FbConstants | None = None,
    *,
    backend: str = "torch",
    mode: str = "loglik",
    device: str = "cpu",
    dtype: str = "float64",
    chunk_size: int | None = None,
    max_memory_bytes: int | None = None,
    validate: str = "params",
    gz_form: str = "block",
    method: str | None = None,
):
    """The batched public API (P8a §2). `dataset` is a list of `N` `(D,T_i)`
    floating arrays (`NaN` allowed, meaning "missing" -- §2.1.1, same
    convention as every other backend); `T_i` may differ across sequences
    (§5's ragged-length support, see the module docstring for how). Returns:

    - `mode="loglik"`: a `(N,)` `float64` NumPy array, in **original
      dataset order** (chunks are internally bucketed by length for
      efficiency, then un-bucketed here).
    - `mode="posterior"`: a **generator** of `BatchedPosteriorChunk` --
      never all `N` posteriors concatenated into one array; each chunk
      carries its own original sequence indices and lengths (`extract()`
      recovers one ordinary, trimmed `Posterior`).
    - `mode="stats"`: one raw (unfinalized) `ExpectedStats`, merged across
      every chunk via `em.merge_expected_stats` -- pass straight into
      `em.finalize_expected_stats`/`em.m_step`, same as any other backend's
      accumulated stats.

    `backend="torch"` is the only implemented batched backend (declared via
    `BackendCapabilities(batched=True)`); the parameter exists for symmetry
    with `forward_backward`'s own `backend=` and to fail clearly (via
    `backends.base.check_capability`) if anything else is ever requested.
    `method="reference"` (sed's non-default `P_cond` construction) is not
    implemented here (`sed_methods=("optimized",)`, same restriction as
    `backend="numba"`) and raises `UnsupportedCombination` before this
    module's own code ever runs. `gz_form` selects between the two Gz/V1
    constructions (§4): `"block"` (default -- per-channel small matmuls,
    never materializes an `(chunk,S,S)` intermediate) or `"masked"` (one
    masked-dense einsum) -- cross-checked against each other in
    `tests/test_backends_torch.py`; `"auto"` does not exist here, since P8a
    has no CUDA benchmark evidence yet to pick a winner from (P8b's job).

    `chunk_size`/`max_memory_bytes` **combination semantics, made explicit**
    (a review round found the previous version silently ignored an invalid
    `max_memory_bytes` whenever `chunk_size` was given explicitly, since it
    was only ever validated inside `estimate_chunk_size`, which an explicit
    `chunk_size` skips entirely): `max_memory_bytes`, whenever given, is
    always validated as a positive integer (`bool` rejected) regardless of
    whether `chunk_size` is automatic or explicit. If `chunk_size` is
    omitted, `max_memory_bytes` (or the CPU/MPS default, or CUDA's own
    runtime free-memory query, capped by `max_memory_bytes` if given --
    see `estimate_chunk_size`'s own docstring) drives `estimate_chunk_size`.
    If both are given, **both are enforced, not just one**: the explicit
    `chunk_size` is checked against `max_memory_bytes` using `min(chunk_size,
    N)` -- the largest chunk `_bucket_by_length` will actually ever build,
    since `chunk_size` is only an upper *cap* (a review round found a
    smaller `N` than `chunk_size`, e.g. `N=1, chunk_size=1000`, falsely
    rejected a call that would only ever allocate a one-sequence chunk) --
    times the same itemized peak-concurrency per-sequence estimate, and
    rejected with a `ValidationError` naming the requested cap, the
    effective chunk size, and `max_memory_bytes` if that would exceed the
    stated budget."""
    if fb_constants is None:
        fb_constants = FbConstants()
    if backend != "torch":
        raise UnsupportedCombination(f"forward_backward_batch only implements backend='torch', got {backend!r}")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if gz_form not in _VALID_GZ_FORMS:
        raise ValueError(f"gz_form must be one of {_VALID_GZ_FORMS}, got {gz_form!r}")
    if fb_constants.variant != "sed" and method is not None:
        raise UnsupportedCombination(f"method= is only meaningful for variant='sed', got variant={fb_constants.variant!r}")
    if fb_constants.variant == "sed" and method not in (None, "optimized"):
        raise UnsupportedCombination(f"backend='torch' does not support sed method={method!r} (supported: 'optimized')")
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")

    check_structure_matches(structure, params)
    # The torch kernels implement the built-in variants and Gaussian-mixture emissions only.
    check_capability("torch", variant=fb_constants.variant, mode=mode, batched=True, emission_model=emission_model_name(params))
    torch_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype)
    _check_device_dtype_combination(torch_device, torch_dtype)

    if validate != "off":
        params.validate()
    _validate_dataset(structure, dataset)

    N = len(dataset)
    T_max_overall = max(Y.shape[1] for Y in dataset)

    # `max_memory_bytes` is validated unconditionally here, regardless of
    # whether `chunk_size` is automatic or explicit -- a review round found
    # `max_memory_bytes=0`/`1.5`/`True` were silently ignored whenever an
    # explicit `chunk_size` was also given, since `estimate_chunk_size` (the
    # only place this used to be checked) is skipped entirely in that case.
    _validate_max_memory_bytes(max_memory_bytes)

    resolved_chunk_size = chunk_size
    if resolved_chunk_size is None:
        resolved_chunk_size = estimate_chunk_size(
            N, structure.S, T_max_overall, structure.max_K, device=torch_device, dtype=torch_dtype,
            max_memory_bytes=max_memory_bytes, D=structure.D, max_d=structure.max_d,
        )
    else:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValidationError(f"chunk_size={chunk_size!r}, expected a positive integer (bool is not accepted, even though it is technically an int subclass)")
        # Both an explicit `chunk_size` and `max_memory_bytes` were given --
        # enforce both genuinely rather than silently ignoring either: the
        # explicit chunk_size must itself fit the stated budget, using the
        # same itemized peak-concurrency estimate `estimate_chunk_size`
        # uses internally. `chunk_size` is only an upper *cap* -- the
        # largest chunk `_bucket_by_length` will ever actually build is
        # `min(chunk_size, N)` (a review round found this check used
        # `chunk_size` directly, so `chunk_size=1000` with `N=1` falsely
        # rejected a call that would only ever allocate a one-sequence
        # chunk); the effective size below fixes that.
        if max_memory_bytes is not None:
            effective_chunk_size = min(chunk_size, N)
            per_seq_bytes = _estimate_per_seq_bytes(structure.S, T_max_overall, structure.max_K, structure.D, structure.max_d, torch_dtype)
            required_bytes = effective_chunk_size * per_seq_bytes
            if required_bytes > max_memory_bytes:
                raise ValidationError(
                    f"chunk_size={chunk_size!r} (effective max chunk size {effective_chunk_size!r}, "
                    f"i.e. min(chunk_size, N={N!r})) is estimated to need ~{required_bytes} bytes "
                    f"({per_seq_bytes} bytes/sequence x {effective_chunk_size}), which exceeds the given "
                    f"max_memory_bytes={max_memory_bytes!r} -- lower chunk_size, raise max_memory_bytes, "
                    f"or omit chunk_size and let estimate_chunk_size derive one that fits"
                )

    with torch.inference_mode():
        kernel = TorchKernel.build(structure, params, device=torch_device, dtype=torch_dtype, pi0_reg=fb_constants.pi0_reg)

    if mode == "loglik":
        # Forward-pass-only path -- never builds a BatchedPosteriorChunk,
        # never runs the backward recursion. See _run_chunk_loglik_only's
        # own docstring for the gap this fixes.
        loglik_by_index = np.empty(N, dtype=np.float64)
        for bucket_indices, loglik_chunk in _iter_loglik_chunks(
            structure, dataset, fb_constants, kernel=kernel, chunk_size=resolved_chunk_size, validate=validate
        ):
            for local_i, orig_i in enumerate(bucket_indices):
                loglik_by_index[orig_i] = loglik_chunk[local_i]
        return loglik_by_index

    if mode == "posterior":
        return _iter_chunks(
            structure, params, dataset, fb_constants, kernel=kernel, chunk_size=resolved_chunk_size, validate=validate, gz_form=gz_form
        )

    # mode == "stats": on-device reduction per chunk (_run_chunk_stats_only)
    # -- never builds a BatchedPosteriorChunk, never transfers alpha/beta/
    # gamma/B/B_comp to CPU; only each chunk's small, fixed-size reduced
    # dict is kept, merged incrementally (not held as a growing list of
    # every chunk's full ExpectedStats before merging). `kernel.W` was
    # already built once by `TorchKernel.build` (which itself calls
    # `build_W`) -- reused here via `_to_cpu_f64` rather than importing
    # `build_W` again and recomputing it from `params` a second time.
    W_np = _to_cpu_f64(kernel.W)
    sigma_diag = params.sigma_diag
    buckets = _bucket_by_length(dataset, resolved_chunk_size)
    merged: ExpectedStats | None = None
    for bucket_indices in buckets:
        Y_chunk_np, lengths = _pad_chunk(dataset, bucket_indices, structure.D)
        Y_chunk_t = torch.as_tensor(Y_chunk_np, dtype=kernel.dtype, device=kernel.device)
        lengths_t = torch.as_tensor(lengths, dtype=torch.int64, device=kernel.device)
        reduced, loglik_total = _run_chunk_stats_only(
            kernel, Y_chunk_t, fb_constants, bucket_indices, lengths_t, validate=validate, gz_form=gz_form, sigma_diag=sigma_diag
        )
        chunk_stats = _chunk_dict_to_expected_stats(structure, sigma_diag, W_np, reduced, loglik_total)
        merged = chunk_stats if merged is None else merge_expected_stats([merged, chunk_stats])
    return merged


# ==========================================================================
# backends.base registration: forward_backward(..., backend="torch") is a
# batch-of-one adapter over forward_backward_batch, returning the same
# ordinary Posterior contract numpy/numba return.
# ==========================================================================


def _run_torch(variant, structure, params, Y, fb_constants, *, validate="params", dtype="float64", **kwargs):
    device = kwargs.pop("device", "cpu")
    gz_form = kwargs.pop("gz_form", "block")
    method = kwargs.pop("method", None)
    if kwargs:
        raise TypeError(f"forward_backward (backend='torch') got unexpected keyword arguments: {sorted(kwargs)}")
    fbc = fb_constants if fb_constants is not None else FbConstants(variant=variant)
    chunks = forward_backward_batch(
        structure, params, [Y], fbc, backend="torch", mode="posterior", device=device, dtype=dtype,
        validate=validate, gz_form=gz_form, method=method,
    )
    (chunk,) = list(chunks)
    return chunk.extract(0)


def _loglik_torch(variant, structure, params, Y, fb_constants, *, validate="params"):
    """Single-sequence forward-only log-likelihood: `forward_backward_batch`'s
    own genuinely forward-only `mode="loglik"`, on a batch of one (CPU, float64
    defaults). validate="all" never reaches this runner."""
    if fb_constants.variant != variant:
        raise ValueError(f"loglik runner requires fb_constants.variant={variant!r}, got {fb_constants.variant!r}")
    if validate not in ("params", "off"):
        raise ValueError(f"forward-only loglik supports validate='params' or 'off', got {validate!r}")
    return float(forward_backward_batch(structure, params, [Y], fb_constants, backend="torch", mode="loglik", validate=validate)[0])


register_backend(
    BackendCapabilities(
        name="torch",
        variants=("sed", "fast", "poise"),
        dtypes=("float64", "float32"),
        modes=("loglik", "posterior", "stats"),
        batched=True,
        gpu=True,
        sed_methods=("optimized",),
        emission_models=("gmm",),
    ),
    _run_torch,
    loglik_runner=_loglik_torch,
)
