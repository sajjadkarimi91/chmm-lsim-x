"""LSIM parameter initializer (`em_lsim.m`'s `else` branch, taken when no
`init_model` is supplied -- `em_lsim.m:176-256` -- plus its `fine_gmm_fitting`
subfunction, `em_lsim.m:551-599`).

**Structural parameters are exact uniform**, not fitted: `pi_0`'s per-channel
block is `1/M_c` (`em_lsim.m:248`); `coupling` is `ones(C,C)/C` (`:89`, shared
by both the `init_model`-given and no-`init_model` branches); every
source-to-target transition block `(c, zee)` is uniformly `1/M_zee`
regardless of the source channel `c`, including `c == zee` (`:250`) -- there
is no special self-transition treatment.

**Emissions are fitted per channel** with `sklearn.mixture.GaussianMixture`,
the direct analogue of MATLAB's `fitgmdist` (`covariance_type="diag"` ≈
`'CovarianceType','diagonal'`, `init_params="random_from_data"` ≈
`'Start','randSample'`, `n_init=20` ≈ `'Replicates',20`, `max_iter=5` ≈
`options.MaxIter`, `reg_covar` ≈ `'Regularize'`). `M_c*K_c` components are
fit flat (not per-state) and then grouped into `(M_c, K_c)` by MATLAB's own
greedy nearest-neighbour rule (`em_lsim.m:205-245`): the first remaining
component anchors a state, the nearest `K_c-1` remaining components (by
squared Euclidean distance between means, distance measured to the *fixed*
anchor throughout, not chained nearest-to-nearest) join it, and the pool
shrinks by `K_c` before the next state's anchor is picked. **Initialization
always fits diagonal covariances** (`em_lsim.m:194`'s `covariance_type =
'diagonal'` is unconditional, independent of the requested model's own
`sigma_diag`) -- a full-covariance request converts each fitted diagonal
variance vector into a diagonal matrix afterward (`em_lsim.m:214,233`), it
never fits a full covariance at init time.

**NaN handling is measured, not assumed** :
`fitgmdist` accepts NaN and performs exact listwise deletion per channel (a
timestep is dropped from channel `c` if *any* dimension of `c` is NaN at that
timestep -- measured bit-identical against dropping incomplete rows
directly, `max|Δμ|=0.0`), which `GaussianMixture` does not do itself, so this
module does it manually before ever calling `.fit()`. The `0.01*min(var(Y,
omitnan))` regularizer (`em_lsim.m:572`) is computed from the *original,
undeleted* channel data, using MATLAB's own `var(...,'omitnan')` convention:
`ddof=1` sample variance, except exactly one non-NaN observation gives `0`
(not NaN -- verified directly against MATLAB R2026a in P4's `em.py` work,
`nanvar_y`) and zero non-NaN observations gives NaN.

**Retries are bounded, seeded, and only under `matlab_compat=True`**
(`fine_gmm_fitting`'s `while(flag_check==0 && counter<10)` loop, `:569-593`,
which perturbs the data and retries on any fit failure, including MATLAB's
own `Ill-conditioned` covariance error). The numerics match MATLAB's own
formula, not an approximation of it: on failure, `counter*0.05*std(current
row, ddof=1)*randn(...)` is added to the *current* (already-jittered, if
this isn't the first failure) data -- cumulative, not fresh-from-original
each attempt -- where `counter` is the attempt number that just failed (the
first retry's jitter uses `counter=1`, not `2`); and the regularizer is
*recomputed* from that same current jittered data on every retry attempt,
exactly as `em_lsim.m:572,580` recomputes `0.01*min(var(inpute_features,...))`
from whatever `inpute_features` currently holds -- only the very first
attempt uses the fixed, undeleted-data regularizer described above. One
deliberate divergence, not a numeric one: MATLAB jitters the *raw*,
NaN-containing `inpute_features` with a plain (non-`omitnan`) `std`, which
means a channel with any missing data that ever reaches the jitter branch
has that dimension's *entire* row turn to NaN (`NaN + finite = NaN`,
elementwise) -- every subsequent attempt on that channel is then
unrecoverable by construction, an edge case in the original MATLAB code
rather than a behaviour worth reproducing. This module jitters the
already-listwise-deleted, NaN-free `Y_complete` instead (per the NaN
handling above), so that failure mode cannot arise here; the `counter`/`ddof`/
cumulative/regularizer-recomputation formula is otherwise applied exactly --
including MATLAB's singleton convention (a dimension with exactly one
complete observation has `std`/`var` of `0.0`, not NaN, so its jitter is
`0` and it never contributes the minimum to a NaN regularizer): both the
jitter noise scale and the recomputed regularizer go through
`_matlab_std_over_samples`/`_channel_variance`, the same MATLAB-compatible
helper the fixed, undeleted-data regularizer above already uses, rather than
a second, plain `ddof=1` formula that would return NaN for that case instead.
MATLAB's nested bare-retry special case for its own `Ill-conditioned` message
string has no `GaussianMixture` analogue and is not replicated -- any
exception falls straight through to the jitter branch. Bounded to 10 total
attempts (`_MAX_FIT_ATTEMPTS`); exhausting them raises `InitialisationError`
chained from the last underlying exception -- this function never returns
partial or uninitialized parameters. Under `matlab_compat=False`, any
failure (including "insufficient complete observations") raises
`InitialisationError` immediately, with no retry -- jittering existing rows
cannot manufacture new independent samples, so silently retrying an
insufficient-data channel would only hide the problem for nine more attempts
before failing anyway.

**Opt-in speed controls (not part of MATLAB parity).** Both default to the
MATLAB-compatible behaviour described above; with defaults the per-channel
computation is unchanged. `em_lsim.m`'s `fine_gmm_fitting` fits `fitgmdist` on
EVERY row of the concatenated channel data (`'Start','randSample'` only picks
each replicate's starting means) -- MATLAB does not subsample.

- `max_samples_per_channel` (default `None` = every complete row, the MATLAB
  path) fits each channel's GMM on a uniform random subset of at most that many
  complete (post-listwise-deletion) rows. The limit applies independently to
  every channel. The subset is drawn from a `SeedSequence` child that is
  independent of the stream seeding each `GaussianMixture` replicate, so the
  first fit attempt's `random_state` is identical with or without sampling.
  On the (rare) jitter-retry path the noise is drawn only for the sampled rows,
  so memory stays bounded by the sample size; retries are deterministic for a
  fixed seed and configuration but do not reproduce the unsampled retry noise.
  The `0.01*min(var)` regularizer is always computed from the
  complete, undeleted channel data. This is a deliberate departure from
  MATLAB: it changes the EM starting point, so it must be validated
  downstream before being relied on. `200_000` has been validated (five seeds,
  downstream AUROC/AUPRC and state occupancy) only for the ctx1 fusion
  workload with M=5, one-dimensional observations, diagonal covariance and
  one Gaussian per state; it is NOT validated for M=10/20, multidimensional
  observations, full covariance or K>1. The library default stays `None`.
- `n_jobs>1` fits channels concurrently in a `ProcessPoolExecutor`. The
  coordinator computes the regularizer, performs listwise deletion and the
  optional sampling BEFORE submitting, so a worker receives only the (possibly
  sampled) fit array, the regularizer, component counts and seed -- never the
  full multi-million-row channel. Workers are started with the `spawn` method
  (never `fork`), so no native thread-pool state is inherited. At most
  `min(n_jobs, C)` channel jobs are
  prepared/in flight at once (coordinator memory is O(n_jobs), not O(C)).
  Each worker's initializer sets the thread-count environment variables in
  that worker, loads scikit-learn, and only then applies a persistent
  one-thread `threadpoolctl` limit, so every BLAS/OpenMP pool a fit uses runs
  one thread and `n_jobs` processes never oversubscribe the machine, whatever
  the caller's `OMP_NUM_THREADS`. The coordinator's limits are untouched. Results are collected in channel order; if any
  channel fails, the lowest-numbered failing channel's error is raised (the
  serial loop's behaviour), wrapped as `InitialisationError` naming the
  channel when the worker raised something else.

**Reproducibility.** Output is a deterministic function of `seed`, the
arguments above, the ORDER of `Y_trials` (channel data are concatenated in
list order before deletion/sampling), and the numerical-library environment
(scikit-learn, NumPy, BLAS). Serial and channel-parallel runs agree exactly in
this project's measured environment (`tests/test_init.py`); bit-identity across
platforms or library versions is not claimed.

**Observations are validated before anything else touches them.** `Y_trials`
must be non-empty, and every trial is run through `chmm_lsim.
validate_observations` (2-D, floating dtype, exactly `structure.D` rows, at
least one timestep, no +/-Inf) before any per-channel slicing or
concatenation -- catching a wrong-shape, wrong-dtype, or non-finite trial
here, with `validate_observations`'s own clear message, rather than letting
it surface downstream as a confusing `IndexError`, a silently-wrong slice
(extra rows past `structure.D` are otherwise never read, not rejected), or a
misleading variance/regularizer failure several calls later.

**scikit-learn is imported lazily**, only inside this module's own fitting
path -- `import chmm_lsim` and every other module stay NumPy-only and importable
without scikit-learn installed; only calling the initializer itself requires
it, and does so with a clear `InitialisationError` (not an opaque
`ModuleNotFoundError`) if it is missing.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.process
import multiprocessing
import os
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.emissions.gmm import ChannelGMM


_MAX_FIT_ATTEMPTS = 10


_GMM_MAX_ITER = 5


_GMM_N_INIT = 20


_REG_SCALE = 0.01


_JITTER_SCALE = 0.05


# Process start method for n_jobs>1. "spawn", not the Linux default "fork": a
# forked worker inherits the coordinator's native BLAS/OpenMP thread-pool state,
# and once the coordinator has run any multi-threaded native code (a NumPy/
# scikit-learn fit, numba) the child can die mid-fit -- measured here as an
# intermittent BrokenProcessPool. Spawned workers start clean; the cost is one
# interpreter + scikit-learn import per worker, per initialization call.
_PROCESS_START_METHOD: str | None = "spawn"


class InitialisationError(Exception):
    """Raised when a channel's data cannot support GMM-based initialization:
    insufficient complete (post listwise-deletion) observations, an
    invalid/all-NaN per-dimension variance, an unusable (non-finite or
    negative) regularizer, or a `GaussianMixture` fit that never converges
    within the bounded retry budget. Always names the offending channel."""


def _import_gaussian_mixture():
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise InitialisationError(
            "scikit-learn is required for LSIM parameter initialization "
            "(sklearn.mixture.GaussianMixture) but is not installed -- "
            "install it via `pip install chmm-lsim[init]` or `pip install scikit-learn`"
        ) from exc
    return GaussianMixture


def _channel_variance(Y_c: NDArray[np.float64]) -> NDArray[np.float64]:
    """`var(Y_c, [], 2, 'omitnan')`'s MATLAB semantics, per dimension (row):
    `ddof=1` sample variance over non-NaN entries; exactly one non-NaN
    observation -> `0.0`, not NaN; zero non-NaN observations -> NaN."""
    n_obs = np.sum(~np.isnan(Y_c), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # n_obs<=1 warns ("Degrees of freedom <= 0"); handled explicitly below
        variance = np.nanvar(Y_c, axis=1, ddof=1)
    variance = np.where(n_obs == 1, 0.0, variance)
    variance = np.where(n_obs == 0, np.nan, variance)
    return variance


def _matlab_std_over_samples(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """`std(X, [], 1)`'s MATLAB singleton convention, per column (dimension):
    `ddof=1` sample std over rows (samples); exactly one row -> `0.0`, not
    NaN. `X` here is always NaN-free by construction (the retry loop's
    `fit_data` is derived from the already-listwise-deleted `Y_complete`), so
    this is `_channel_variance` reused on the transpose -- the `omitnan` part
    is a no-op, but the singleton convention is the same MATLAB behaviour
    either way, and reusing it (rather than a second, plain-`ddof=1` formula)
    is what keeps this rule in exactly one place."""
    return np.sqrt(_channel_variance(X.T))


def _compute_regularizer(variance: NDArray[np.float64], channel: int) -> float:
    """`0.01*min(variance)` (`em_lsim.m:572`), from the *undeleted* channel
    variance -- raises `InitialisationError` naming `channel` if any
    dimension's variance is NaN (a dimension with zero observed timesteps
    across every trial cannot support a regularizer at all), or if the
    resulting scalar is somehow non-finite or negative (variance is never
    negative in practice; this is a defensive invariant guard, not a path
    real data reaches -- see `test_compute_regularizer_rejects_bad_value`)."""
    if np.isnan(variance).any():
        bad_dims = np.nonzero(np.isnan(variance))[0].tolist()
        raise InitialisationError(
            f"channel {channel}: variance is NaN for dimension(s) {bad_dims} -- "
            f"that dimension has zero observed timesteps across every trial, "
            f"so 0.01*min(var(Y,omitnan)) cannot be computed"
        )
    reg_covar = _REG_SCALE * float(np.min(variance))
    if not np.isfinite(reg_covar) or reg_covar < 0:
        raise InitialisationError(f"channel {channel}: unusable regularizer {reg_covar!r} computed from 0.01*min(var(Y,omitnan))")
    return reg_covar


def _group_components_into_states(
    means: NDArray[np.float64], variances: NDArray[np.float64], weights: NDArray[np.float64], M_c: int, K_c: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MATLAB's greedy component-to-state grouping (`em_lsim.m:205-245`): for
    each of the `M_c` states in turn, the first still-unassigned component
    anchors the state, then the nearest `K_c-1` still-unassigned components
    (squared Euclidean distance between means, measured to the *fixed*
    anchor for every one of the `K_c-1` picks, not chained) join it, removed
    from the pool as they're picked so the next state's anchor is whatever
    is left. Weights are normalized within each state afterward
    (`em_lsim.m:243`). Returns `(mu, variance, weight)`, each `(M_c, K_c,
    ...)` -- caller reshapes into `ChannelGMM`'s own axis order."""
    d_c = means.shape[1]
    remaining = list(range(means.shape[0]))
    mu_out = np.empty((M_c, K_c, d_c))
    var_out = np.empty((M_c, K_c, d_c))
    w_out = np.empty((M_c, K_c))

    for i in range(M_c):
        anchor = remaining.pop(0)
        mu_out[i, 0] = means[anchor]
        var_out[i, 0] = variances[anchor]
        w_out[i, 0] = weights[anchor]
        anchor_mu = means[anchor]

        for k in range(1, K_c):
            candidates = means[remaining]
            dist = np.sum((candidates - anchor_mu) ** 2, axis=1)
            pos = int(np.argmin(dist))
            nearest = remaining.pop(pos)
            mu_out[i, k] = means[nearest]
            var_out[i, k] = variances[nearest]
            w_out[i, k] = weights[nearest]

        w_out[i] = w_out[i] / w_out[i].sum()

    return mu_out, var_out, w_out


@dataclass(frozen=True)
class _ChannelFitJob:
    """Everything one channel's GMM fit needs, already reduced on the
    coordinator: the complete (and optionally sampled) rows, the regularizer
    computed from the full undeleted channel data, and the seed. This is the
    only thing pickled to a worker process."""

    channel: int
    fit_data: NDArray[np.float64]  # (n_fit, d_c), NaN-free, C-contiguous
    reg_covar: float
    M_c: int
    K_c: int
    sigma_diag: bool
    seed: int
    matlab_compat: bool
    n_complete: int = 0  # complete rows available before any sampling (informational)


def _prepare_channel_job(
    Y_c: NDArray[np.float64],
    M_c: int,
    K_c: int,
    *,
    sigma_diag: bool,
    channel: int,
    seed: int,
    matlab_compat: bool,
    max_samples_per_channel: int | None = None,
) -> _ChannelFitJob:
    """Coordinator-side reduction of one channel: regularizer from the full
    undeleted data, channel-local listwise NaN deletion, the insufficient-
    observations check, then optional deterministic sampling."""
    n_components = M_c * K_c

    reg_covar = _compute_regularizer(_channel_variance(Y_c), channel)

    complete_mask = ~np.isnan(Y_c).any(axis=0)
    Y_complete = Y_c[:, complete_mask].T  # (n_complete, d_c)
    n_complete = Y_complete.shape[0]

    if n_complete < n_components:
        message = (
            f"channel {channel}: only {n_complete} complete (listwise NaN-deleted) observations "
            f"available, need at least {n_components} (= M_c*K_c = {M_c}*{K_c}) to fit {n_components} "
            f"GMM components"
        )
        if not matlab_compat:
            raise InitialisationError(message)
        # matlab_compat=True: em_lsim.m's fine_gmm_fitting doesn't pre-check this either -- it
        # just calls fitgmdist, which fails, and the retry loop jitters and tries again.
        # Jittering existing rows can never manufacture new independent samples, so every one
        # of the 10 attempts fails identically and the loop is exhausted deterministically.

    if max_samples_per_channel is not None and n_complete > max_samples_per_channel:
        # A SeedSequence child keeps this draw independent of the replicate/jitter stream
        # (`np.random.default_rng(seed)` in _fit_channel_job). Sorted to keep time order.
        sub_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        keep = np.sort(sub_rng.choice(n_complete, size=max_samples_per_channel, replace=False))
        Y_complete = Y_complete[keep]

    return _ChannelFitJob(
        channel=channel,
        fit_data=np.ascontiguousarray(Y_complete),
        reg_covar=reg_covar,
        M_c=M_c,
        K_c=K_c,
        sigma_diag=sigma_diag,
        seed=seed,
        matlab_compat=matlab_compat,
        n_complete=n_complete,
    )


def _fit_channel_job(job: _ChannelFitJob) -> ChannelGMM:
    """The GMM fit + MATLAB retry loop + greedy state grouping for one
    prepared channel. Identical arithmetic whether it runs in the
    coordinator (n_jobs=1) or in a worker process."""
    GaussianMixture = _import_gaussian_mixture()
    n_components = job.M_c * job.K_c
    d_c = job.fit_data.shape[1]
    channel = job.channel

    rng = np.random.default_rng(job.seed)
    last_exc: Exception | None = None
    fitted = None
    fit_data = job.fit_data  # attempt 1: unjittered, exactly em_lsim.m's first fitgmdist try
    current_reg_covar = job.reg_covar  # attempt 1: the fixed, undeleted-data regularizer

    for attempt in range(1, _MAX_FIT_ATTEMPTS + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="diag",
                init_params="random_from_data",
                n_init=_GMM_N_INIT,
                max_iter=_GMM_MAX_ITER,
                reg_covar=current_reg_covar,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            with warnings.catch_warnings():
                # em_lsim.m:195,197 brackets fine_gmm_fitting with warning('off')/warning('on') --
                # max_iter=5 is deliberately weak (matching MATLAB's own max_itr=5), so sklearn's
                # ConvergenceWarning on every call is expected, not a signal, and is suppressed here
                # the same way.
                warnings.simplefilter("ignore")
                gmm.fit(fit_data)
            fitted = gmm
            break
        except Exception as exc:  # sklearn raises ValueError / np.linalg.LinAlgError on ill-conditioned fits
            last_exc = exc
            if not job.matlab_compat:
                raise InitialisationError(f"channel {channel}: GaussianMixture fit failed on attempt {attempt}: {exc}") from exc
            if attempt == _MAX_FIT_ATTEMPTS:
                break
            # em_lsim.m:586-587: counter*0.05*std(current data, ddof=1)*randn(...), added to
            # (not replacing) the current data -- cumulative across attempts, and `attempt`
            # here IS `counter` (the attempt that just failed), matching MATLAB's first jitter
            # using counter=1, not 2. MATLAB's singleton convention (n==1 -> std/var of 0, not
            # NaN) is reused via _matlab_std_over_samples/_channel_variance, not np.std/np.var's
            # own ddof=1 (which gives NaN, not 0, for a single row) -- a single complete
            # observation is exactly the case n_complete<n_components always reaches once
            # n_components==1, and MATLAB's own jitter does not stall on it.
            noise_scale = attempt * _JITTER_SCALE * _matlab_std_over_samples(fit_data)
            # Noise is drawn for the rows actually being fitted, so a sampled channel's retry
            # allocation is bounded by the sample size, never by the full channel length.
            # (Without sampling this is exactly the previous full-data draw.)
            fit_data = fit_data + noise_scale[None, :] * rng.standard_normal(fit_data.shape)
            # em_lsim.m:572,580 recomputes the regularizer from whatever inpute_features
            # currently holds on every retry -- only the first attempt uses the fixed,
            # undeleted-data value computed on the coordinator.
            jittered_variance = _channel_variance(fit_data.T)
            jittered_reg_covar = _REG_SCALE * float(np.min(jittered_variance))
            if np.isfinite(jittered_reg_covar) and jittered_reg_covar >= 0:
                current_reg_covar = jittered_reg_covar

    if fitted is None:
        raise InitialisationError(
            f"channel {channel}: GaussianMixture fit did not succeed in {_MAX_FIT_ATTEMPTS} attempts "
            f"(matlab_compat=True jitter-retry budget exhausted)"
        ) from last_exc

    mu, variance, weight = _group_components_into_states(fitted.means_, fitted.covariances_, fitted.weights_, job.M_c, job.K_c)

    mu_out = np.transpose(mu, (2, 0, 1))  # (d_c, M_c, K_c)
    if job.sigma_diag:
        sigma_out = np.transpose(variance, (2, 0, 1))  # (d_c, M_c, K_c)
    else:
        sigma_out = np.zeros((d_c, d_c, job.M_c, job.K_c))
        for i in range(job.M_c):
            for k in range(job.K_c):
                sigma_out[:, :, i, k] = np.diag(variance[i, k])

    return ChannelGMM(weights=weight, mu=mu_out, sigma=sigma_out, diag=job.sigma_diag)


def _fit_channel(
    Y_c: NDArray[np.float64],
    M_c: int,
    K_c: int,
    *,
    sigma_diag: bool,
    channel: int,
    seed: int,
    matlab_compat: bool,
    max_samples_per_channel: int | None = None,
) -> ChannelGMM:
    """One channel end to end, in the calling process (the n_jobs=1 path)."""
    return _fit_channel_job(
        _prepare_channel_job(
            Y_c, M_c, K_c, sigma_diag=sigma_diag, channel=channel, seed=seed,
            matlab_compat=matlab_compat, max_samples_per_channel=max_samples_per_channel,
        )
    )


_WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


_WORKER_THREAD_LIMIT = None  # per worker process: the persistent threadpoolctl limit, once applied


def _init_channel_worker() -> None:
    """ProcessPoolExecutor initializer for channel-parallel initialization.

    Order matters: threadpoolctl can only limit native thread pools that are
    already loaded, and scikit-learn's OpenMP runtime is loaded when
    `sklearn.mixture` is imported. So each worker (1) sets the thread-count
    environment variables in its OWN environment, which a runtime loaded
    afterwards reads at load time, (2) imports scikit-learn/scipy exactly as a
    fit will, and only then (3) applies a persistent one-thread limit to every
    loaded pool, held for the worker's lifetime. The coordinator's environment
    and thread limits are never touched."""
    global _WORKER_THREAD_LIMIT
    if _WORKER_THREAD_LIMIT is not None:
        return
    for name in _WORKER_THREAD_ENV:
        os.environ[name] = "1"
    _import_gaussian_mixture()
    import scipy.linalg  # noqa: F401 -- loaded by a GMM fit; load it before limiting

    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover -- scikit-learn itself requires threadpoolctl
        raise InitialisationError(
            "channel-parallel initialization (n_jobs>1) requires threadpoolctl, which is installed "
            "with scikit-learn -- install chmm-lsim with the 'init' extra (pip install 'chmm-lsim[init]')"
        ) from exc
    _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)


def _fit_channel_job_in_worker(job: _ChannelFitJob) -> ChannelGMM:
    _init_channel_worker()  # no-op when the pool initializer already ran
    return _fit_channel_job(job)


def _channel_pool(max_workers: int) -> concurrent.futures.ProcessPoolExecutor:
    """The executor channel-parallel initialization uses (tests use this same factory)."""
    mp_context = multiprocessing.get_context(_PROCESS_START_METHOD) if _PROCESS_START_METHOD else None
    return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context, initializer=_init_channel_worker)


def _fit_channels_parallel(
    structure: LSIMStructure,
    Y_trials: list[NDArray[np.float64]],
    channel_seeds: list[int],
    *,
    n_jobs: int,
    sigma_diag: bool,
    matlab_compat: bool,
    max_samples_per_channel: int | None,
) -> list[ChannelGMM]:
    # Workers need scikit-learn and threadpoolctl. Check they are INSTALLED here,
    # without importing (no native thread pools loaded in the coordinator), so a
    # missing dependency raises the same clear error as the serial path instead of
    # an opaque broken-pool failure when every worker's initializer dies.
    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        _import_gaussian_mixture()  # raises the standard "scikit-learn is required" InitialisationError
    if importlib.util.find_spec("threadpoolctl") is None:
        raise InitialisationError(
            "channel-parallel initialization (n_jobs>1) requires threadpoolctl, which is installed "
            "with scikit-learn -- install chmm-lsim with the 'init' extra (pip install 'chmm-lsim[init]')"
        )

    C = structure.C
    max_workers = min(n_jobs, C)
    results: list[ChannelGMM | None] = [None] * C
    failures: dict[int, BaseException] = {}

    with _channel_pool(max_workers) as pool:
        in_flight: dict[concurrent.futures.Future, int] = {}
        next_channel = 0
        while True:
            # Prepare and submit lazily: never more than max_workers channel jobs
            # (and their fit arrays) exist on the coordinator at once.
            while not failures and next_channel < C and len(in_flight) < max_workers:
                c = next_channel
                next_channel += 1
                try:
                    Y_c = np.concatenate([Y[structure.dim_slice(c), :] for Y in Y_trials], axis=1)
                    job = _prepare_channel_job(
                        Y_c, structure.channel_state_num[c], structure.num_gmm_component[c],
                        sigma_diag=sigma_diag, channel=c, seed=channel_seeds[c],
                        matlab_compat=matlab_compat, max_samples_per_channel=max_samples_per_channel,
                    )
                    del Y_c
                except Exception as exc:  # noqa: BLE001 -- recorded, re-raised in channel order below
                    failures[c] = exc
                    break
                in_flight[pool.submit(_fit_channel_job_in_worker, job)] = c
                del job
            if not in_flight:
                break
            done, _ = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                c = in_flight.pop(fut)
                exc = fut.exception()
                if exc is None:
                    results[c] = fut.result()
                else:
                    failures[c] = exc

    if failures:
        # Every channel below the lowest failure was submitted and has finished, so raising
        # the lowest failing channel reproduces the serial loop's error.
        c = min(failures)
        exc = failures[c]
        if isinstance(exc, InitialisationError):
            raise exc
        if isinstance(exc, concurrent.futures.process.BrokenProcessPool):
            broken = sorted(ch for ch, e in failures.items() if isinstance(e, concurrent.futures.process.BrokenProcessPool))
            raise InitialisationError(
                f"channel {c}: initialization worker pool broke (a worker process was terminated, e.g. killed "
                f"for memory) while channels {broken} were unfinished; the failing channel cannot be "
                f"identified more precisely"
            ) from exc
        raise InitialisationError(f"channel {c}: initialization worker failed: {type(exc).__name__}: {exc}") from exc
    return results  # type: ignore[return-value]


def _validate_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)


def _check_init_options(structure: LSIMStructure, *, sigma_diag: bool, matlab_compat: bool = True,
                        max_samples_per_channel: int | None = None, n_jobs: int = 1) -> dict:
    """Validates the GMM initialization controls (before any observation is
    touched) and returns them normalized."""
    if max_samples_per_channel is not None:
        max_samples_per_channel = _validate_positive_int("max_samples_per_channel", max_samples_per_channel)
        need = max(m * k for m, k in zip(structure.channel_state_num, structure.num_gmm_component))
        if max_samples_per_channel < need:
            raise ValueError(
                f"max_samples_per_channel={max_samples_per_channel} is smaller than the largest "
                f"M_c*K_c={need}; a GMM cannot fit more components than rows"
            )
    n_jobs = _validate_positive_int("n_jobs", n_jobs)
    return {"sigma_diag": sigma_diag, "matlab_compat": matlab_compat, "max_samples_per_channel": max_samples_per_channel,
            "n_jobs": n_jobs}


def _initialize_gmm_emissions(structure: LSIMStructure, Y_trials: list[NDArray[np.float64]], channel_seeds: list[int], *,
                              sigma_diag: bool, matlab_compat: bool, max_samples_per_channel: int | None,
                              n_jobs: int) -> list[ChannelGMM]:
    """Fits every channel's GMM (serially, or in `n_jobs` worker processes);
    `channel_seeds[c]` seeds channel `c`'s fit. Observations must already be
    validated."""
    C = structure.C
    if n_jobs == 1 or C == 1:
        emissions: list[ChannelGMM] = []
        for c in range(C):
            Y_c = np.concatenate([Y[structure.dim_slice(c), :] for Y in Y_trials], axis=1)
            emissions.append(
                _fit_channel(
                    Y_c,
                    structure.channel_state_num[c],
                    structure.num_gmm_component[c],
                    sigma_diag=sigma_diag,
                    channel=c,
                    seed=channel_seeds[c],
                    matlab_compat=matlab_compat,
                    max_samples_per_channel=max_samples_per_channel,
                )
            )
        return emissions
    return _fit_channels_parallel(
        structure, Y_trials, channel_seeds, n_jobs=n_jobs, sigma_diag=sigma_diag,
        matlab_compat=matlab_compat, max_samples_per_channel=max_samples_per_channel,
    )
