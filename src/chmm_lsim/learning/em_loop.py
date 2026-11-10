"""Model-independent EM plumbing shared by every model family.

* `ExpectedStats`: the raw, mergeable E-step statistics (initial/occupancy
  posteriors, transition mass `zeta` in the model's own layout, counts,
  log-likelihood and one emission-statistics object per channel), with
  `merge_expected_stats` / `finalize_expected_stats`; the per-channel parts
  are merged and finalized by the emission model that produced them.
* `EmResult` and `_run_em_loop`: the iteration loop with its convergence
  rule and best-iterate selection (`em_lsim.m`/`em_chmm.m`).
* trial-level parallel execution helpers (`TrialExecutionError`, executor and
  start-method validation, read-only trial locking) and the AIC/BIC formula.

**Raw vs finalized.** An E-step returns raw statistics: every field is a sum
or a mergeable central-moment accumulator, so raw statistics from disjoint
chunks of trials can be combined with `merge_expected_stats` before anything
derived is computed. `finalize_expected_stats` computes the derived fields
once, after all merging, before the M-step.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import UnsupportedCombination
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.emissions.base import emission_model_for_stats

if TYPE_CHECKING:
    from chmm_lsim.models.chmm.params import CHMMParams
    from chmm_lsim.models.lsim.params import LSIMParams


_VALID_EXECUTORS = ("thread", "process")

_IN_FLIGHT_MULTIPLIER = 2  # bounded-scheduling window width, in units of n_jobs -- see _run_parallel_e_step


@dataclass
class ExpectedStats:
    """Raw, mergeable sufficient statistics from one E-step (or a merge of
    several). `gamma_p0` is `None` until `finalize_expected_stats` computes
    it from `gamma_first`/`n_trials` -- see the module docstring."""

    zeta: NDArray[np.float64]  # (S,S) W-weighted transition posterior mass, summed over t and trials; (J,J) joint pairwise mass for em_chmm
    gamma_first: NDArray[np.float64]  # (S,) Σ over trials of gamma[:, 0] -- NOT divided by n_trials
    gamma_sum: NDArray[np.float64]  # (S,) Σ_t gamma, all trials -- pi_steady, diagnostics
    n_trials: int
    n_transitions: int  # L_T_1 = sum_trials(T_r - 1)
    T_all: int
    loglik: float  # sum over trials of P_O_model
    channels: list  # one emission-statistics object per channel (e.g. ChannelExpectedStats)
    gamma_p0: NDArray[np.float64] | None = None  # (S,) derived: gamma_first / n_trials


def merge_expected_stats(chunks: list[ExpectedStats]) -> ExpectedStats:
    """Combines raw (unfinalized) `ExpectedStats` from disjoint chunks of
    trials -- e.g. from separate calls to `e_step` on different subsets of
    `Y_trials` (P8's chunked/streamed path). Raises if any chunk was already
    finalized."""
    if any(c.gamma_p0 is not None for c in chunks):
        raise ValueError("merge_expected_stats requires raw (unfinalized) stats -- finalize only after merging")
    n_channels = len(chunks[0].channels)
    model = emission_model_for_stats(chunks[0].channels) if n_channels else None
    merged_channels = [model.merge_stats([c.channels[ci] for c in chunks]) for ci in range(n_channels)]
    return ExpectedStats(
        zeta=sum(c.zeta for c in chunks),
        gamma_first=sum(c.gamma_first for c in chunks),
        gamma_sum=sum(c.gamma_sum for c in chunks),
        n_trials=sum(c.n_trials for c in chunks),
        n_transitions=sum(c.n_transitions for c in chunks),
        T_all=sum(c.T_all for c in chunks),
        loglik=sum(c.loglik for c in chunks),
        channels=merged_channels,
    )


def finalize_expected_stats(stats: ExpectedStats, *, sigma_diag: bool) -> ExpectedStats:
    """Computes `gamma_p0` and every channel's derived fields, once, after
    all merging. Returns a new object (see `finalize_channel_stats`)."""
    gamma_p0 = stats.gamma_first / stats.n_trials
    model = emission_model_for_stats(stats.channels) if stats.channels else None
    channels = [model.finalize_stats(c, sigma_diag=sigma_diag) for c in stats.channels]
    return dataclasses_replace(stats, gamma_p0=gamma_p0, channels=channels)


class TrialExecutionError(RuntimeError):
    """Raised by `e_step`'s parallel (`n_jobs>1`) path when a worker trial
    itself raises. Wraps the original exception (available as
    `original_exception`, chained via `raise ... from`) and records which
    trial -- by its 0-based position in the `Y_trials` list passed to
    `e_step`, not a completion-order position -- failed.

    Every not-yet-STARTED trial (queued in the pool but not yet picked up by
    a worker) is cancelled before this is raised (§P10b: "worker exceptions
    must cancel pending work and be re-raised with the original trial
    index") -- with the bounded in-flight scheduling `_run_parallel_e_step`
    uses (§P10b correction round), at most `_IN_FLIGHT_MULTIPLIER * n_jobs`
    trials are ever queued/running at once, so this failure mode wastes at
    most that many trials' worth of work, not the entire remaining dataset.
    A trial that is already RUNNING when the exception is detected cannot be
    interrupted by the stdlib executors -- and, since `_run_parallel_e_step`
    exits its executor's own `with` block on the way out (whether or not an
    exception is propagating), `Executor.__exit__` calls `shutdown(wait=
    True)`, which BLOCKS until every such already-running trial actually
    finishes. So this exception does **not** surface to the caller while
    other trials continue in the background: by the time `e_step` raises
    `TrialExecutionError`, every trial that was already running has already
    completed (or itself raised) -- only cancelled, not-yet-started trials
    are skipped, and every trial's *result* past the failing one, running or
    cancelled, is discarded either way."""

    def __init__(self, trial_index: int, original_exception: BaseException) -> None:
        super().__init__(f"trial index {trial_index} (0-based, into the Y_trials list passed to e_step) raised {type(original_exception).__name__}: {original_exception}")
        self.trial_index = trial_index
        self.original_exception = original_exception

def _validate_n_jobs_executor(n_jobs: int, executor: str | None, backend: str) -> None:
    """Strict, fail-loud validation for `e_step`'s `n_jobs`/`executor`
    axis -- never silently ignores an inconsistent combination (§P10b).
    `backend="torch"` owns its own batch/device parallelism inside
    `forward_backward_batch` (`chunk_size`/`max_memory_bytes`, an
    execution-memory control, never a trial-concurrency one); stacking an
    outer CPU executor on top of it would duplicate the model, oversubscribe
    CPU threads, or create multiple CUDA contexts, so it is rejected
    outright rather than downgraded to `n_jobs=1` behind the caller's back."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive int, got {n_jobs!r}")
    if executor is not None and executor not in _VALID_EXECUTORS:
        raise ValueError(f"executor must be one of {_VALID_EXECUTORS} or None, got {executor!r}")
    if backend == "torch":
        if n_jobs != 1 or executor is not None:
            raise UnsupportedCombination(
                f"backend='torch' does not support an outer CPU executor (n_jobs={n_jobs!r}, executor={executor!r}) -- "
                f"it already owns its own batch/device parallelism inside forward_backward_batch (chunk_size/"
                f"max_memory_bytes); pass n_jobs=1, executor=None for backend='torch', or use a CPU backend "
                f"(numpy/numba) for outer trial concurrency"
            )
        return
    if n_jobs == 1:
        if executor is not None:
            raise ValueError(
                f"executor={executor!r} was given but n_jobs=1 runs the serial reference path, which has no "
                f"executor to apply it to -- pass n_jobs>1 to use executor={executor!r}, or drop executor"
            )
    elif executor is None:
        raise ValueError(
            f"n_jobs={n_jobs} requires an explicit executor -- pass executor='thread' or executor='process' "
            f"(never inferred from backend={backend!r} or from n_jobs alone)"
        )

def _resolve_start_method(start_method: str | None, executor: str | None) -> str | None:
    """P10e: the multiprocessing start method for `executor="process"` pools.

    `None` means the safe default, "spawn", on every platform. Spawned workers
    are fresh interpreters: they share no memory, native thread-pool state,
    locks or libffi closure allocator with the parent or with each other.

    Measured on the development cluster (system libffi 3.1, which every tested
    interpreter's `_ctypes` links): a minimal ctypes-callback workload with no
    lsim or numba code segfaults under "fork" on Python 3.12, 3.13 and 3.14,
    and under "forkserver" on Python 3.14; numba's llvmlite creates exactly
    such callbacks when loading kernels, and the P10e test suite reproduced a
    forkserver worker segfault inside llvmlite on 3.14. "spawn" passed
    everywhere. "forkserver" and "fork" remain available as explicit opt-ins
    (they start pools faster), for environments where they are known to be
    safe.

    A start method is only meaningful with executor="process"."""
    if start_method is None:
        return "spawn" if executor == "process" else None
    if executor != "process":
        raise ValueError(f"start_method={start_method!r} was given but executor={executor!r}; it only applies to executor='process'")
    if start_method not in multiprocessing.get_all_start_methods():
        raise ValueError(f"start_method must be one of {multiprocessing.get_all_start_methods()} or None, got {start_method!r}")
    return start_method

def _lock_trials_read_only(Y_trials):
    """P10f: marks every trial array read-only for the duration of a training
    run and returns a callable restoring the original flags. With the
    driver's own `_ValidatedTrials` snapshot of the trial references, this
    means an observation validated at the boundary cannot be silently
    changed afterwards through the array object that was validated (e.g. from
    a hook): in-place writes raise `ValueError`, and replacing a list element
    does not affect the snapshot. Writes through another alias of the same
    memory (a different view, or the base array) are not detectable this way."""
    changed = []
    seen = set()
    for Y in Y_trials:
        if isinstance(Y, np.ndarray) and id(Y) not in seen:
            seen.add(id(Y))
            if Y.flags.writeable:
                Y.flags.writeable = False
                changed.append(Y)

    def restore() -> None:
        for Y in changed:
            try:
                Y.flags.writeable = True
            except ValueError:  # its base was made read-only in the meantime; leave it read-only
                pass

    return restore


def _aic_bic(loglik: float, n_params: int, T_all: int) -> tuple[float, float]:
    """`(AIC, BIC)` for a fit with `n_params` free parameters on `T_all` samples."""
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + np.log(T_all) * n_params
    return float(aic), float(bic)


def apply_left2right_pi0_smoothing(pi_0: NDArray[np.float64], structure: LSIMStructure, left2right: bool) -> NDArray[np.float64]:
    """`em_lsim.m:159-164`'s one-time conversion of a raw `init_model.pi_0`
    into the *effective* `pi_0` iteration 1's E-step runs on: `pi_0 <-
    (1e-4+pi_0)/sum(...)` per channel block when `left2right` is falsy, or a
    plain per-block renormalisation (no additive smoothing) when it isn't."""
    out = np.zeros_like(pi_0)
    for c in range(structure.C):
        sl = structure.state_slice(c)
        block = pi_0[sl] if left2right else 1e-4 + pi_0[sl]
        out[sl] = block / block.sum()
    return out

@dataclass
class EmResult:
    """Result of `em_lsim` and `em_chmm` (`params` is a `CHMMParams` for the latter)."""

    params: LSIMParams | CHMMParams  # best-iterate snapshot
    best_iterate: int  # 1-based iteration index the snapshot came from
    pi_steady: NDArray[np.float64] | None = None  # (S,) == gamma_sum/T_all at the best iterate (em_lsim.m's 8th output)
    loglik_trace: list[float] = field(default_factory=list)  # loglik_params_in per completed iteration
    aic_trace: list[float] = field(default_factory=list)
    bic_trace: list[float] = field(default_factory=list)
    n_iterations: int = 0

def _run_em_loop(params, iteration, *, max_iterations: int, check_convergence: bool, iter_hook) -> EmResult:
    """The iteration loop shared by `em_lsim` and `em_chmm` (`em_lsim.m` and
    `em_chmm.m` have the same loop). `iteration(params)` runs one E-step and
    M-step and returns `(new_params, loglik, aic, bic, pi_steady)`, where
    `loglik`/`aic`/`bic`/`pi_steady` are those of the `params` passed in."""
    result = EmResult(params=params, best_iterate=0)
    best_loglik = -np.inf

    for itr in range(1, max_iterations + 1):
        params, loglik, aic, bic, pi_steady = iteration(params)

        result.loglik_trace.append(loglik)
        result.aic_trace.append(aic)
        result.bic_trace.append(bic)
        result.n_iterations = itr

        if iter_hook is not None:
            iter_hook(itr, params, loglik, aic, bic)

        # Convergence is checked BEFORE the best-iterate snapshot (em_lsim.m's
        # order): a `break` here skips this iteration's snapshot entirely, so
        # a converged run's best-iterate never considers the iteration that
        # triggered the break.
        if check_convergence and itr > 9:
            recent = np.diff(result.loglik_trace[itr - 6 : itr])
            if np.sum(recent < 0.001) >= 2:
                break

        if itr > 2:
            if loglik > best_loglik:
                best_loglik = loglik
                result.params = params
                result.pi_steady = pi_steady
                result.best_iterate = itr
        else:
            best_loglik = loglik
            result.params = params
            result.pi_steady = pi_steady
            result.best_iterate = itr

    return result
