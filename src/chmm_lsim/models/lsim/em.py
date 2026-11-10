"""EM for LSIM (`em_lsim.m`): shared M-step, AIC/BIC, best-iterate tracking.

The E-step is one forward-backward call per trial, dispatched through
`backends.base.run_backend(backend, fb_constants.variant, ...)` (P3b's
variant axis and P7's execution-backend axis both apply here) --
Raw POISE is the default inference variant. Explicit `variant="sed"` and
`variant="fast"` also run end-to-end through the
same `e_step`/`em_lsim`, on either `backend="numpy"`
(the default) or `backend="numba"`, reduced into
`ExpectedStats` -- fixed-size sufficient statistics, independent of `T_all`
(§2.4.2), so a chunked/streamed caller (P8) can accumulate them without ever
materialising a `(S, T_all)` array. `left2right`'s only effect (§0.4.8) is a
one-time smoothing of a raw `init_model`'s `pi_0` into the *effective*
parameters iteration 1's E-step actually runs on; `em_lsim`'s own `pi_0`
update every iteration is unconditionally `mean(gamma_p0)` (MATLAB's
conditional preservation of it under `left2right` is commented out in
`em_lsim.m`, so nothing here reproduces it).

**Raw vs finalized, explicitly.** `e_step` (and its per-channel accumulator)
returns *raw* stats: every field is a plain sum or a mergeable central-moment
triple, so two raw `ExpectedStats` from disjoint chunks of trials can be
combined with `merge_expected_stats` before anything derived is computed.
`finalize_expected_stats` is the one place `nanvar_y`, the two fallback
flags, and `gamma_p0` get computed -- from the (possibly merged) raw
accumulators -- and must run exactly once, after all merging, before `m_step`.

**Central moments, not raw moments (§2.4.2, corrected).** MATLAB's own M-step
computes `mu` and `sigma` by *directly recentering* observed data around a
mean -- it never forms `Σy²` and subtracts. A naive port that accumulates
`Σy`, `Σy²` and later computes `Σy² - 2·μ·Σy + μ²·Σw` is algebraically
identical but catastrophically unstable once `y` sits far from zero (the
`Σy²` term and the correction terms are each `O(offset²)`, and subtracting
two such terms to recover an `O(variance)` result loses precision
proportional to `offset²/variance`). Every accumulator below is instead a
*central* moment -- a running `(weight, mean, M2)` triple (Welford/Chan) for
the diagonal case, or a running `(Q, M, C)` pairwise triple for the full
case -- combined via the mergeable parallel-variance formula, so precision
stays proportional to the data's own scale, not its offset from zero.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from collections import deque

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import UnsupportedCombination, run_backend
from chmm_lsim.backends.numpy_ import block_sum, build_W
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import _is_prevalidated, _ValidatedTrials
from chmm_lsim.emissions.base import (
    accumulate_emission_stats,
    emission_model_for,
    get_emission_model,
    m_step_emissions,
    new_emission_stats,
)
from chmm_lsim.learning.em_loop import (
    _IN_FLIGHT_MULTIPLIER,
    EmResult,
    ExpectedStats,
    TrialExecutionError,
    _aic_bic,
    _lock_trials_read_only,
    _resolve_start_method,
    _run_em_loop,
    _validate_n_jobs_executor,
    apply_left2right_pi0_smoothing,
    finalize_expected_stats,
    merge_expected_stats,
)
from chmm_lsim.learning.minibatch import (
    _UNSET,
    MinibatchEmResult,
    MinibatchSchedule,
    MinibatchTrainingResult,
    MinibatchUpdate,
    RobustConvergence,
    _legacy_convergence_stop,
    _resolve_convergence,
    _resolve_schedule,
    _RobustConvergenceMonitor,
    _Unset,
)
from chmm_lsim.models.lsim.params import LSIMParams


def _e_step_one_trial(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants,
    W: NDArray[np.float64],
    *,
    backend: str,
    validate: str,
) -> ExpectedStats:
    """One trial's own raw, unfinalized `ExpectedStats` (`n_trials=1`) --
    the parallel-execution counterpart of a single iteration of `e_step`'s
    own serial `for Y in Y_trials` loop body below, factored out to a plain
    module-level function (not a closure) so it is picklable and can run
    inside a `ProcessPoolExecutor` worker. `W` is passed in rather than
    recomputed per trial -- it depends only on `structure`/`params`, not on
    `Y`, so every trial (and every worker) shares the exact same array.

    Returns a value with the same `ExpectedStats.zeta` contract as the
    serial path (already `W`-weighted) -- `merge_expected_stats` sums this
    field in place across trials, and `(a+b)*W` and `a*W+b*W` agree only to
    within a few ULP in floating point, not bit-for-bit, so a parallel-path
    result is verified against the serial reference at a measured tolerance,
    never with `n_jobs=1`'s own exact reproducibility guarantee."""
    emission_model = emission_model_for(params.emissions)
    channels = new_emission_stats(structure, params.emissions, emission_model)
    zeta = np.zeros((structure.S, structure.S))
    gamma_first = np.zeros(structure.S)
    gamma_sum = np.zeros(structure.S)
    n_transitions = 0

    post = run_backend(backend, fb_constants.variant, structure, params, Y, fb_constants, validate=validate)
    T = Y.shape[1]
    gamma_first += post.gamma[:, 0]
    gamma_sum += post.gamma.sum(axis=1)
    if T > 1:
        abar = post.alpha[:, : T - 1] * post.scale[:, : T - 1]
        beta_next = post.beta[:, 1:]
        zeta += abar @ beta_next.T
        n_transitions = T - 1
    accumulate_emission_stats(channels, structure, post, Y, params.emissions, emission_model)

    return ExpectedStats(
        zeta=zeta * W,
        gamma_first=gamma_first,
        gamma_sum=gamma_sum,
        n_trials=1,
        n_transitions=n_transitions,
        T_all=T,
        loglik=post.loglik,
        channels=channels,
    )


# ProcessPoolExecutor worker-process state: populated once per worker
# process (via `initializer=`/`initargs=`, at pool startup) rather than
# re-pickled on every single submitted trial -- `structure`/`params`/`W`
# are the same for every trial in one e_step call, so paying their
# pickling/unpickling cost once per worker process (not once per trial) is
# what the plan doc's own "array serialization" benchmark axis is about.
# Deliberately a plain module-level dict, not a class instance: each worker
# is a genuinely separate process (its own copy of this module's globals),
# so there is no cross-worker or cross-thread sharing to guard against here
# (contrast the thread-executor path below, which passes everything
# directly through `submit(...)` instead, since threads share memory with
# zero copying cost and a shared mutable global would risk cross-talk
# between concurrent `e_step` calls from different application threads).
_PROCESS_WORKER_CTX: dict = {}


def _process_worker_init(
    structure: LSIMStructure, params: LSIMParams, fb_constants: FbConstants, W: NDArray[np.float64], backend: str, validate: str
) -> None:
    _PROCESS_WORKER_CTX.update(structure=structure, params=params, fb_constants=fb_constants, W=W, backend=backend, validate=validate)


def _process_worker_run(Y: NDArray[np.float64]) -> ExpectedStats:
    ctx = _PROCESS_WORKER_CTX
    return _e_step_one_trial(ctx["structure"], ctx["params"], Y, ctx["fb_constants"], ctx["W"], backend=ctx["backend"], validate=ctx["validate"])


def _process_loglik_worker_init(structure: LSIMStructure, params: LSIMParams, fb_constants: FbConstants, backend: str, validate: str) -> None:
    _PROCESS_WORKER_CTX.update(structure=structure, params=params, fb_constants=fb_constants, backend=backend, validate=validate)


def _process_loglik_worker_run(Y: NDArray[np.float64]) -> float:
    from chmm_lsim.backends.base import run_backend_loglik

    ctx = _PROCESS_WORKER_CTX
    return run_backend_loglik(ctx["backend"], ctx["fb_constants"].variant, ctx["structure"], ctx["params"], Y, ctx["fb_constants"], validate=ctx["validate"])


def _runner_validate(structure: LSIMStructure, params: LSIMParams, Y_trials, validate: str) -> str:
    """P10f: the `validate=` value the per-trial numpy/numba runners receive.
    For a private `_ValidatedTrials` snapshot (observations already validated
    once, at a training driver's boundary) with `validate="params"`, the
    parameters are validated here, once for this call, and the runners get
    `"off"` -- which skips only their per-trial `params.validate()` and
    `validate_observations` (for these runners `"params"` and `"off"` differ
    in nothing else). Any ordinary list, or `validate="all"`/`"off"`, is
    passed through unchanged."""
    if validate == "params" and _is_prevalidated(structure, Y_trials):
        params.validate()
        return "off"
    return validate


def _forward_only_loglik(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    fb_constants: FbConstants,
    *,
    backend: str,
    validate: str,
    n_jobs: int,
    executor: str | None,
    start_method: str | None = None,
) -> float:
    """Summed forward-only log-likelihood over `Y_trials` for `numpy`/`numba`.

    Equal, bit for bit, to `e_step(...).loglik` with the same arguments: every
    trial's value comes from the same forward recursion (`run_backend_loglik`),
    and trials are summed as a left fold from 0.0 in ORIGINAL trial order --
    the order `e_step`'s serial loop and its bounded parallel merge both use.
    Parallel execution reuses `_run_parallel_e_step`'s bounded in-flight
    scheduling and `TrialExecutionError` wrapping."""
    from chmm_lsim.backends.base import run_backend_loglik

    _validate_n_jobs_executor(n_jobs, executor, backend)
    n = len(Y_trials)
    if n == 0:
        raise ValueError("forward-only loglik requires at least one trial in Y_trials")
    validate = _runner_validate(structure, params, Y_trials, validate)
    total = 0.0
    if n_jobs == 1:
        for Y in Y_trials:
            total += run_backend_loglik(backend, fb_constants.variant, structure, params, Y, fb_constants, validate=validate)
        return total

    max_workers = min(n_jobs, n)
    max_in_flight = max_workers * _IN_FLIGHT_MULTIPLIER
    if executor == "thread":
        pool_cls = concurrent.futures.ThreadPoolExecutor
        pool_kwargs: dict = {}

        def _submit(pool, idx: int):
            return pool.submit(run_backend_loglik, backend, fb_constants.variant, structure, params, Y_trials[idx], fb_constants, validate=validate)
    else:
        pool_cls = concurrent.futures.ProcessPoolExecutor
        pool_kwargs = {"initializer": _process_loglik_worker_init, "initargs": (structure, params, fb_constants, backend, validate),
                       "mp_context": multiprocessing.get_context(_resolve_start_method(start_method, "process"))}

        def _submit(pool, idx: int):
            return pool.submit(_process_loglik_worker_run, Y_trials[idx])

    with pool_cls(max_workers=max_workers, **pool_kwargs) as pool:
        pending: deque = deque()
        next_to_submit = 0
        for _ in range(min(max_in_flight, n)):
            pending.append((next_to_submit, _submit(pool, next_to_submit)))
            next_to_submit += 1
        try:
            while pending:
                idx, fut = pending.popleft()
                try:
                    value = fut.result()
                except Exception as exc:  # noqa: BLE001 -- re-wrapped with the trial index, never swallowed
                    raise TrialExecutionError(idx, exc) from exc
                total += value
                if next_to_submit < n:
                    pending.append((next_to_submit, _submit(pool, next_to_submit)))
                    next_to_submit += 1
        except BaseException:
            for _, fut in pending:
                fut.cancel()
            raise
    return total


def _run_parallel_e_step(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    fb_constants: FbConstants,
    W: NDArray[np.float64],
    *,
    backend: str,
    validate: str,
    n_jobs: int,
    executor: str,
    start_method: str | None = None,
) -> ExpectedStats:
    """Runs one trial per task across a stdlib `ThreadPoolExecutor` or
    `ProcessPoolExecutor` (`_validate_n_jobs_executor` has already confirmed
    `executor` is one of these two), then merges every trial's own raw
    `ExpectedStats` into a running accumulator, in the trials' ORIGINAL
    `Y_trials` order -- never completion order, so the merged result (and
    therefore the fitted parameters downstream) does not depend on which
    worker happens to finish first, and repeated runs with the same inputs
    reproduce the same result.

    **Bounded in-flight scheduling (§P10b correction round).** At most
    `_IN_FLIGHT_MULTIPLIER * max_workers` trials are ever submitted-but-not-
    yet-merged at once: trials are submitted lazily, one more submitted each
    time the OLDEST still-outstanding one is retrieved and folded in, rather
    than submitting every one of `len(Y_trials)` up front and holding every
    `Future`/`ExpectedStats` until the very end. This is what makes this
    function usable at the intended ~100,000-trial scale: coordinator memory
    here is O(n_jobs), never O(len(Y_trials)) -- neither a list of every
    `Future` nor a list of every trial's own `ExpectedStats` is ever held at
    once (`tests/test_em_parallel.py`'s `test_bounded_in_flight_*` tests
    prove this directly, via an instrumented executor, as N grows).

    The merged result is IDENTICAL to submitting everything up front and
    merging once at the end, not merely close: `merge_expected_stats
    ([c0,...,cN-1])` already folds its chunks left-to-right internally
    (`_merge_field`'s own loop, and every top-level field's plain
    `sum(...)`), so incrementally calling `merge_expected_stats([running_
    merge, next_chunk])` one trial at a time performs the exact same
    sequence of arithmetic operations, in the exact same order, as one call
    over the whole list would -- this is a scheduling change, not a numeric
    one.

    A worker exception cancels every not-yet-STARTED trial and is re-raised
    as `TrialExecutionError`, naming the original (0-based) trial index --
    see that class's own docstring for exactly what does and doesn't finish
    running before this function returns."""
    n = len(Y_trials)
    max_workers = min(n_jobs, n)
    max_in_flight = max_workers * _IN_FLIGHT_MULTIPLIER

    if executor == "thread":
        pool_cls = concurrent.futures.ThreadPoolExecutor
        pool_kwargs: dict = {}

        def _submit(pool, idx: int):
            return pool.submit(_e_step_one_trial, structure, params, Y_trials[idx], fb_constants, W, backend=backend, validate=validate)
    else:
        pool_cls = concurrent.futures.ProcessPoolExecutor
        pool_kwargs = {"initializer": _process_worker_init, "initargs": (structure, params, fb_constants, W, backend, validate),
                       "mp_context": multiprocessing.get_context(_resolve_start_method(start_method, "process"))}

        def _submit(pool, idx: int):
            return pool.submit(_process_worker_run, Y_trials[idx])

    merged: ExpectedStats | None = None
    with pool_cls(max_workers=max_workers, **pool_kwargs) as pool:
        pending: deque = deque()  # (trial_index, Future) pairs, oldest-first -- never longer than max_in_flight
        next_to_submit = 0
        for _ in range(min(max_in_flight, n)):
            pending.append((next_to_submit, _submit(pool, next_to_submit)))
            next_to_submit += 1

        try:
            while pending:
                idx, fut = pending.popleft()
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001 -- deliberately re-wrapped with the trial index below, never swallowed
                    raise TrialExecutionError(idx, exc) from exc
                merged = result if merged is None else merge_expected_stats([merged, result])
                del result  # folded in already -- drop the reference now, not at the end of an O(N) list

                if next_to_submit < n:
                    pending.append((next_to_submit, _submit(pool, next_to_submit)))
                    next_to_submit += 1
        except BaseException:
            for _, fut in pending:
                fut.cancel()
            raise

    return merged


def e_step(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    fb_constants: FbConstants | None = None,
    *,
    backend: str = "numpy",
    backend_options: dict | None = None,
    validate: str = "params",
    n_jobs: int = 1,
    executor: str | None = None,
    start_method: str | None = None,
) -> ExpectedStats:
    """Runs the E-step over `Y_trials` and returns *raw* (unfinalized)
    `ExpectedStats` -- call `finalize_expected_stats` (directly, or after
    `merge_expected_stats` with other chunks) before `m_step`. `backend`
    (default `"numpy"`) is passed through `backends.base.run_backend` for
    *every* trial -- `backend="numba"` genuinely runs the compiled kernels
    here, not just when calling `forward_backward` directly (there is no
    separate `get_inference(...)`-based bypass of the backend axis).

    `backend_options` (P8a) is a plain `dict` of backend-specific keyword
    arguments -- `device=`/`dtype=`/`chunk_size=`/`max_memory_bytes=`/
    `gz_form=` for `backend="torch"` -- deliberately *not* individual named
    parameters on this signature, so `e_step`'s own contract doesn't grow a
    new parameter for every backend this project ever adds. Only meaningful
    for `backend="torch"`; a non-empty `backend_options` for `"numpy"`/
    `"numba"` raises rather than being silently ignored (those backends
    have no options to receive it). `backend="torch"` calls
    `backends.torch_batched.forward_backward_batch(..., mode="stats")`
    **once**, over the whole `Y_trials` dataset -- not once per trial -- so
    the batched recursion genuinely runs batched during EM, not merely when
    `forward_backward`/`run_backend` is called directly.

    `n_jobs`/`executor` (P10b) apply only to the per-trial `numpy`/`numba`
    CPU paths below -- they control how the trials *within this one E-step*
    are evaluated (concurrently or not), never how many optimization
    updates happen (that is P10c's mini-batch driver's job, not this
    function's). `n_jobs=1` (the default) is the deterministic serial
    reference path, byte-for-byte unchanged from before P10b -- it does not
    go through `ProcessPoolExecutor`/`ThreadPoolExecutor` or
    `merge_expected_stats` at all. `n_jobs>1` requires an explicit
    `executor='thread'` or `executor='process'` (never inferred) and merges
    every trial's own raw stats in `Y_trials`' original order, never
    completion order -- see `_run_parallel_e_step`. `backend='torch'`
    requires `n_jobs=1`, `executor=None` (`_validate_n_jobs_executor`
    raises `UnsupportedCombination` otherwise): it already owns its own
    batch/device parallelism via `chunk_size`/`max_memory_bytes` above, an
    execution-memory control, never a trial-concurrency one -- stacking an
    outer CPU executor on top of it would duplicate the model, oversubscribe
    CPU threads, or create multiple CUDA contexts.

    **`executor='process'` requires the standard `multiprocessing` entry-
    point guard on spawn-based platforms** (the default start method on
    Windows always, and on macOS since Python 3.8 -- i.e. this project's own
    supported platforms): calling code that creates a `ProcessPoolExecutor`
    (directly, or indirectly through this `executor='process'` path) must
    live inside `if __name__ == "__main__":` (or otherwise be safely
    re-importable with no top-level side effects), because `spawn` starts
    each worker as a fresh interpreter that re-imports the launching
    script -- without the guard, that reimport can re-run the whole calling
    script in every worker, including recursively re-entering `e_step`
    itself. This is standard Python `multiprocessing`/`concurrent.futures`
    behavior, not specific to this function; every one of this file's own
    process-executor tests already runs under pytest's own entry point,
    which already satisfies this."""
    backend_options = backend_options or {}
    if backend != "torch" and backend_options:
        raise ValueError(
            f"backend_options={backend_options!r} was given but backend={backend!r} has no options to receive "
            f"it -- backend_options is only meaningful for backend='torch'"
        )
    _validate_n_jobs_executor(n_jobs, executor, backend)
    _resolve_start_method(start_method, executor)  # validate early (P10e)
    if fb_constants is None:
        fb_constants = FbConstants()

    if backend == "torch":
        from chmm_lsim.backends.torch_batched import forward_backward_batch

        return forward_backward_batch(
            structure, params, Y_trials, fb_constants, backend="torch", mode="stats", validate=validate, **backend_options
        )

    validate = _runner_validate(structure, params, Y_trials, validate)
    W = build_W(params.transition, params.coupling, structure.block_of_state)

    if n_jobs > 1:
        if not Y_trials:
            raise ValueError("e_step requires at least one trial in Y_trials")
        return _run_parallel_e_step(
            structure, params, Y_trials, fb_constants, W, backend=backend, validate=validate, n_jobs=n_jobs, executor=executor,
            start_method=start_method,
        )

    emission_model = emission_model_for(params.emissions)
    channels = new_emission_stats(structure, params.emissions, emission_model)
    zeta = np.zeros((structure.S, structure.S))
    gamma_first = np.zeros(structure.S)
    gamma_sum = np.zeros(structure.S)
    n_transitions = 0
    n_trials = 0
    T_all = 0
    loglik = 0.0

    for Y in Y_trials:
        post = run_backend(backend, fb_constants.variant, structure, params, Y, fb_constants, validate=validate)
        T = Y.shape[1]
        T_all += T
        n_trials += 1
        loglik += post.loglik
        gamma_first += post.gamma[:, 0]
        gamma_sum += post.gamma.sum(axis=1)
        if T > 1:
            abar = post.alpha[:, : T - 1] * post.scale[:, : T - 1]
            beta_next = post.beta[:, 1:]
            zeta += abar @ beta_next.T
            n_transitions += T - 1
        accumulate_emission_stats(channels, structure, post, Y, params.emissions, emission_model)

    return ExpectedStats(
        zeta=zeta * W,
        gamma_first=gamma_first,
        gamma_sum=gamma_sum,
        n_trials=n_trials,
        n_transitions=n_transitions,
        T_all=T_all,
        loglik=loglik,
        channels=channels,
    )


def m_step(
    structure: LSIMStructure,
    stats: ExpectedStats,
    *,
    sigma_diag: bool,
    matlab_compat: bool = True,
    params_in: LSIMParams | None = None,
) -> LSIMParams:
    """`pi_0`/`transition`/`coupling`/every channel's emissions, from one
    (possibly mini-batch-scoped) finalized `ExpectedStats`.

    **Zero-transition mini-batches (P10c, §2.6).** `stats.n_transitions==0`
    (every trial in this E-step's own scope has `T=1`, e.g. a mini-batch
    drawn entirely from `T=1` trials) makes `transition`/`coupling`'s own
    defining formulas structurally undefined -- MATLAB's own `em_lsim_par.m`
    divides by its `L_T_1_batch=0` here, landing on `0/0=NaN` for the
    affected rows/columns, then its own NaN-replacement uniformly
    substitutes `1/M`/`1/C` -- silently manufacturing a UNIFORM structural
    parameter with no evidence behind it, which the design rules out
    ("must never... replace structural parameters with accidental uniform
    values merely because the batch contains no transition evidence").
    Python's policy is deliberately corrected, not reproduced: when `stats.
    n_transitions==0`, `transition`/`coupling` are copied UNCHANGED from
    `params_in` (required in that case -- raises `ValueError` if not given,
    rather than dividing by zero) instead of being recomputed at all; `pi_0`
    and every emission parameter are still updated normally, since both are
    well-defined from `gamma`/responsibilities alone, independent of
    transitions. Ordinary (`n_transitions>0`) callers are unaffected --
    `params_in` is optional and unused on that path, so this is not a
    behavior change for any existing caller."""
    if stats.gamma_p0 is None:
        raise ValueError("m_step requires finalized stats -- call finalize_expected_stats first")
    if stats.n_transitions == 0 and params_in is None:
        raise ValueError(
            "stats.n_transitions == 0 (every trial in this E-step's own scope has T=1) -- transition/coupling have "
            "no transition evidence to update from and must be preserved from the previous snapshot instead of "
            "dividing by zero; pass params_in= (P10c's zero-transition policy)"
        )
    emissions_new = m_step_emissions(structure, stats, sigma_diag=sigma_diag, matlab_compat=matlab_compat)
    pi_0_new, transition_new, coupling_new = structural_m_step(
        structure, stats.zeta, stats.gamma_p0, stats.n_transitions, params_in=params_in)
    return LSIMParams(structure=structure, pi_0=pi_0_new, transition=transition_new, coupling=coupling_new, emissions=emissions_new)


def structural_m_step(
    structure: LSIMStructure,
    zeta: NDArray[np.float64],
    gamma_p0: NDArray[np.float64],
    n_transitions: int,
    *,
    params_in=None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """`pi_0`, `transition`, `coupling` from the W-weighted source-selection mass
    `zeta` (S,S), the mean first-frame posterior `gamma_p0` (S,) and the
    transition count -- the structural half of `m_step` (`em_lsim.m:331-351`),
    independent of the emission model. With `n_transitions == 0` the
    structural blocks are copied from `params_in` (zero-transition policy)."""
    S, C = structure.S, structure.C
    pi_0_new = gamma_p0.copy()
    if n_transitions == 0:
        if params_in is None:
            raise ValueError("n_transitions == 0: transition/coupling must be preserved from params_in")
        if params_in.structure != structure:
            raise ValueError(
                "params_in.structure does not match the structure this m_step call is running under -- refusing to "
                "preserve transition/coupling from a different model's params_in (matching array shapes, e.g. same "
                "S and C with a different per-channel state partition, is not sufficient: the STATE MEANING differs)"
            )
        transition_new = params_in.transition.copy()
        coupling_new = params_in.coupling.copy()
    else:
        transition_new = np.zeros((S, S))
        coupling_new = np.zeros((C, C))
        for zee in range(C):
            zee_sl = structure.state_slice(zee)
            M_zee = structure.channel_state_num[zee]
            zeta_zee = zeta[:, zee_sl]  # (S, M_zee)
            row_sums = zeta_zee.sum(axis=1)  # (S,)

            trans_zee = np.divide(zeta_zee, row_sums[:, None], out=np.full_like(zeta_zee, np.nan), where=row_sums[:, None] != 0)
            trans_zee = np.where(np.isnan(trans_zee), 1.0 / M_zee, trans_zee)
            transition_new[:, zee_sl] = trans_zee

            coupling_new[:, zee] = block_sum(row_sums, structure.state_offsets) / n_transitions

        coupling_new = np.where(np.isnan(coupling_new), 1.0 / C, coupling_new)
        coupling_new = coupling_new / coupling_new.sum(axis=0, keepdims=True)

    return pi_0_new, transition_new, coupling_new


def compute_aic_bic(structure: LSIMStructure, sigma_diag: bool, loglik: float, T_all: int) -> tuple[float, float]:
    """AIC and BIC of an LSIM fit with Gaussian-mixture emissions (`em_lsim.m`'s
    parameter count: coupling, transition blocks and emission parameters)."""
    C = structure.C
    M = np.array(structure.channel_state_num)
    n_params = C * (C - 1) + int(M.sum()) ** 2 - C * int(M.sum()) + get_emission_model("gmm").n_free_parameters(structure, sigma_diag=sigma_diag)
    return _aic_bic(loglik, n_params, T_all)


def _reject_float32_training(backend: str, backend_options: dict | None) -> None:
    """P10f: EM training on `backend="torch"` with `dtype="float32"` is
    explicitly unsupported (it previously failed later, inside the first
    post-update E-step, with a `pi_0` block-sum `ValidationError`).

    Measured on CUDA (`benchmarks/p8b/float32_training_gate_study.py`, 14
    cases x 10 epochs): float32 statistics leave M-step stochastic blocks off
    by up to ~1.2e-7 (the reproduced `pi_0` sum 0.99999998 failure), which a
    bounded float64 renormalization does repair -- but for `variant="sed"`
    float32 statistics are themselves wrong: the float32 SED backward
    recursion hits `beta_floor` thousands of times where float64 hits it a
    handful, corrupting `gamma` (single entries off by up to 1.0), and trained
    models diverge from float64 from the first updates (parameter differences
    up to 1.4 on the C=18, M=5 workload). `variant="fast"` stayed within
    ~1e-5 of float64 in the same study but is not approved on that evidence
    alone. Forward-only float32 scoring (`forward_backward_batch(mode=
    "loglik")`) is unaffected by this guard."""
    if backend == "torch" and (backend_options or {}).get("dtype", "float64") == "float32":
        raise UnsupportedCombination(
            "EM training with backend='torch', dtype='float32' is not supported: float32 statistics violate the "
            "M-step's stochastic invariants and, for variant='sed', corrupt the smoothed posteriors (beta_floor "
            "saturation in the float32 backward recursion). Train with backend_options={'dtype': 'float64'} (CUDA "
            "float64 is fast enough); float32 remains available for forward-only scoring."
        )


def em_lsim(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    *,
    max_iterations: int,
    fb_constants: FbConstants | None = None,
    backend: str = "numpy",
    backend_options: dict | None = None,
    matlab_compat: bool = True,
    check_convergence: bool = True,
    validate: str = "params",
    left2right: bool = False,
    raw_init_model: bool = False,
    iter_hook=None,
) -> EmResult:
    """`em_lsim.m`'s iteration loop. `params` is taken as already-effective by
    default; pass `raw_init_model=True` to have this apply
    `apply_left2right_pi0_smoothing` itself first (`em_lsim.m:159-164`, gated
    by `left2right`), matching `em_lsim.m`'s own `nargin==6` branch rather
    than requiring the caller to pre-smooth. `backend`/`backend_options`
    (default `"numpy"`/`None`) are forwarded to every iteration's `e_step`
    call unchanged -- see `e_step`'s own docstring for what `backend_options`
    is (a plain dict, not individual named parameters on this signature).
    `backend="torch"` with `dtype="float32"` raises `UnsupportedCombination`
    (P10f, see `_reject_float32_training`)."""
    _reject_float32_training(backend, backend_options)
    if fb_constants is None:
        fb_constants = FbConstants()
    if raw_init_model:
        params = params.copy()
        params.pi_0 = apply_left2right_pi0_smoothing(params.pi_0, structure, left2right)
    T_all = sum(Y.shape[1] for Y in Y_trials)
    sigma_diag = params.sigma_diag

    def iteration(params: LSIMParams):
        raw_stats = e_step(structure, params, Y_trials, fb_constants, backend=backend, backend_options=backend_options, validate=validate)
        stats = finalize_expected_stats(raw_stats, sigma_diag=sigma_diag)
        aic, bic = compute_aic_bic(structure, sigma_diag, stats.loglik, T_all)
        params = m_step(structure, stats, sigma_diag=sigma_diag, matlab_compat=matlab_compat)
        return params, stats.loglik, aic, bic, stats.gamma_sum / stats.T_all

    return _run_em_loop(params, iteration, max_iterations=max_iterations, check_convergence=check_convergence, iter_hook=iter_hook)


def _run_one_minibatch_update(
    structure: LSIMStructure,
    params_in: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    trial_indices: tuple[int, ...],
    epoch_idx: int,
    minibatch_idx: int,
    fb_constants: FbConstants,
    global_nanvar: list[NDArray[np.float64]],
    *,
    backend: str,
    backend_options: dict | None,
    n_jobs: int,
    executor: str | None,
    matlab_compat: bool,
    validate: str,
    start_method: str | None = None,
) -> MinibatchUpdate:
    """One mini-batch's own complete E-step -> finalize -> global-`nanvar`-
    substitute -> M-step -> commit, given an ALREADY-COMPUTED `global_
    nanvar` (§P10c/P10d correction round: factored out of `em_lsim_
    minibatch`'s own loop body specifically so that neither it nor
    `em_lsim_minibatch_train` ever calls `compute_global_nanvar` more than
    ONCE per training run -- see both callers' own docstrings for why
    calling it once per epoch, as an earlier version of `em_lsim_minibatch_
    train` did, is a real scalability defect at the ~100,000-trial scale
    this whole phase targets, not merely an inefficiency). `params_in` is
    never mutated -- returned unchanged as `MinibatchUpdate.params_in`, safe
    to hold a reference to.

    Worker-exception trial indices are translated from LOCAL (`e_step`'s
    own `TrialExecutionError.trial_index`, an index into `trial_indices`
    itself) to GLOBAL (an index into the full `Y_trials` list) here -- the
    single place this translation needs to happen for either caller, so a
    caller debugging a large run learns which of ALL its trials failed,
    never merely which position within one mini-batch."""
    batch_Y = Y_trials.subset(trial_indices) if isinstance(Y_trials, _ValidatedTrials) else [Y_trials[i] for i in trial_indices]
    sigma_diag = params_in.sigma_diag

    try:
        raw = e_step(
            structure, params_in, batch_Y, fb_constants,
            backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor, validate=validate,
            start_method=start_method,
        )
    except TrialExecutionError as exc:
        global_index = trial_indices[exc.trial_index]
        raise TrialExecutionError(global_index, exc.original_exception) from exc.original_exception

    training_batch_loglik = raw.loglik
    stats = finalize_expected_stats(raw, sigma_diag=sigma_diag)
    stats = emission_model_for(params_in.emissions).apply_global_statistics(stats, global_nanvar)
    params_out = m_step(structure, stats, sigma_diag=sigma_diag, matlab_compat=matlab_compat, params_in=params_in)

    return MinibatchUpdate(
        epoch_idx=epoch_idx,
        minibatch_idx=minibatch_idx,
        trial_indices=trial_indices,
        params_in=params_in,
        params_out=params_out,
        n_trials=stats.n_trials,
        T_all=stats.T_all,
        n_transitions=stats.n_transitions,
        training_batch_loglik=training_batch_loglik,
    )


def em_lsim_minibatch(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    *,
    max_iterations: int | None = None,
    num_minibatches: int | _Unset = _UNSET,
    shuffle: bool | _Unset = _UNSET,
    random_state: int | np.random.Generator | None | _Unset = _UNSET,
    schedule: MinibatchSchedule | None = None,
    fb_constants: FbConstants | None = None,
    backend: str = "numpy",
    backend_options: dict | None = None,
    n_jobs: int = 1,
    executor: str | None = None,
    matlab_compat: bool = True,
    validate: str = "params",
    left2right: bool = False,
    raw_init_model: bool = False,
    start_method: str | None = None,
) -> MinibatchEmResult:
    """P10c's own incremental mini-batch EM scheduling/update driver
    -- backend-independent (reuses `e_step`
    unchanged for `numpy`/`numba`/`torch`, so P10b's `n_jobs`/`executor` axis
    and Torch's `chunk_size`/`max_memory_bytes` execution-memory controls
    apply exactly as documented there).

    **Two independent controls, never conflated** (§2.6): `num_minibatches`
    is a COUNT of disjoint mini-batches per epoch (never a batch size, and
    never Torch's own `chunk_size`); `n_jobs`/`executor`/`backend_options`
    control only how the trials WITHIN one mini-batch's own E-step are
    evaluated. `num_minibatches=1` reduces every epoch to one mini-batch
    containing every trial -- algebraically the same E-step/M-step `em_lsim`
    itself performs each iteration, and (with `shuffle=False`,
    `backend='numpy'`, `n_jobs=1`) numerically byte-identical PER ITERATION
    to the exact `params_out` `em_lsim`'s own E-step/M-step call produces at
    that iteration (confirmed against every intermediate iteration via
    `em_lsim`'s own `iter_hook`, not merely the last one) -- the exact same
    `e_step`/`m_step` calls run over the exact same trial order, so the
    UPDATE SEQUENCE matches exactly. **This is a narrower guarantee than
    "reproduces `em_lsim`'s own returned `EmResult`"** (§P10c correction
    round: an earlier version of this docstring overclaimed the broader
    one): `em_lsim`'s own `EmResult.params` is a BEST-ITERATE snapshot,
    which can be an EARLIER iteration than the last one, and `em_lsim` can
    stop early via its own convergence check -- neither best-iterate
    selection nor convergence exists in this driver at all (P10d's job).
    `em_lsim_minibatch`'s own `MinibatchEmResult.params` is always the LAST
    committed update's `params_out`, which only equals `em_lsim`'s own
    returned best-iterate params when the run happens to converge on (or
    never improve past) its final iteration.

    **Scheduling.** Either pass `schedule=` directly (an already-validated
    or independently-constructed `MinibatchSchedule` -- e.g. reconstructed
    from a P10a golden fixture's own `updates_trial_indices`/`updates_epoch_
    idx`/`updates_minibatch_idx`, for exact replay) or let this generate one
    via `generate_minibatch_schedule`, seeded by an explicit
    `np.random.Generator` (`random_state` -- an int seed, an existing
    `Generator`, or `None` for fresh, still-non-global entropy) -- this
    module never reads or writes NumPy's global random state.
    `num_minibatches > len(Y_trials)` warns and clamps to `len(Y_trials)`
    (`_resolve_num_minibatches`, matching `em_lsim_par.m`'s own policy).
    Every schedule -- generated or given -- is validated (`validate_
    minibatch_schedule`) before any E-step runs: exact epoch count (only
    when generating, since `max_iterations` is otherwise inferred FROM the
    given schedule), 0-based in-range indices, and every trial exactly once
    per epoch, no duplicates or drops, no empty mini-batches.

    **Per-mini-batch update (§2.6's own raw/merge/finalize contract)**: for
    each mini-batch, in schedule order, `params_in` is the CURRENT (already-
    committed) snapshot -- nothing here mutates it in place, so holding a
    reference to it in the returned trace is safe, not merely "frozen" by
    convention. One raw E-step runs over that mini-batch's own trials
    (`e_step`, reused unchanged from P10b/P8a); its raw `ExpectedStats` is
    finalized exactly once, has this run's precomputed global `nanvar_y`
    substituted in (`compute_global_nanvar`/`_substitute_global_nanvar`,
    §2.6's own covariance-ridge policy -- see that function's own docstring
    for the exact MATLAB citation), and is passed to exactly one `m_step`
    call (with `params_in=` for the zero-transition policy below); the
    resulting `params_out` is committed -- becomes the next mini-batch's own
    `params_in` -- before the loop continues. Never averages already-
    finalized parameters or statistics; never re-merges across mini-batches.

    **Zero-transition mini-batches** (`stats.n_transitions==0`, e.g. a
    mini-batch drawn entirely from `T=1` trials): `m_step`'s own
    `params_in=` argument makes it preserve `transition`/`coupling`
    UNCHANGED rather than dividing by zero or manufacturing uniform values
    -- see `m_step`'s own docstring. `pi_0` and every emission parameter
    still update normally.

    **Worker error trial indices are GLOBAL** (into `Y_trials`, not local to
    whichever mini-batch was running): `e_step`'s own `TrialExecutionError.
    trial_index` is local to the mini-batch's own trial list it was given,
    so a raised `TrialExecutionError` is caught here and re-raised with
    `trial_indices[local_index]` -- the actual position in the FULL dataset
    -- substituted in, so a caller debugging a large (e.g. ~100,000-trial)
    run learns which of ALL its trials failed, not merely which position
    within one mini-batch.

    **No persistent process pool.** Each mini-batch's `e_step` call creates
    and tears down its own `ProcessPoolExecutor` when `executor='process'`
    (P10b's own, already-bounded-memory, already-correct design) rather
    than this driver holding one open across mini-batches -- deliberately:
    `params`/`W` change every mini-batch (via `m_step`), and a
    `ProcessPoolExecutor`'s `initializer=` runs ONLY once, when a worker
    process starts, never again on a later `submit()` -- a persistent pool
    would need an explicit, verified per-mini-batch mechanism to refresh
    every already-running worker's cached `params`/`W` (e.g. deterministic
    worker-sized shards each re-initialized once per mini-batch, or
    resubmitting `initializer`-equivalent state to every worker and
    confirming every one of them picked it up) before it could safely claim
    any reuse-based speedup; that mechanism is not implemented here, so no
    such speedup is claimed for P10c -- `n_jobs=1` remains the default, and
    every worker in every mini-batch that DOES run in parallel is guaranteed
    to see the CURRENT `params`/`W` because it was started fresh for it.

    `backend="torch"` with `dtype="float32"` raises `UnsupportedCombination`
    (P10f, see `_reject_float32_training`)."""
    _reject_float32_training(backend, backend_options)
    if fb_constants is None:
        fb_constants = FbConstants()
    if raw_init_model:
        params = params.copy()
        params.pi_0 = apply_left2right_pi0_smoothing(params.pi_0, structure, left2right)

    num_trials = len(Y_trials)
    if num_trials == 0:
        raise ValueError("Y_trials must contain at least one trial")

    schedule = _resolve_schedule(
        Y_trials, max_iterations=max_iterations, num_minibatches=num_minibatches, shuffle=shuffle,
        random_state=random_state, schedule=schedule,
    )

    # Computed exactly ONCE, before any mini-batch's own E-step -- never
    # recomputed per mini-batch or per epoch (§P10c/P10d correction round;
    # see _run_one_minibatch_update's own docstring for why this matters at
    # the ~100,000-trial scale P10 targets).
    _resolve_start_method(start_method, executor)  # validate early (P10e)
    restore_trial_flags = _lock_trials_read_only(Y_trials)  # P10f: read-only while this run uses them
    try:
        global_nanvar = emission_model_for(params.emissions).global_statistics(structure, Y_trials)  # validates every observation, once
        trials = _ValidatedTrials(Y_trials, structure)  # P10f: snapshot; internal E-steps skip re-validation

        result = MinibatchEmResult(params=params, schedule=schedule)

        for e, mbs in enumerate(schedule.epochs):
            for mb, trial_indices in enumerate(mbs):
                update = _run_one_minibatch_update(
                    structure, params, trials, trial_indices, e + 1, mb + 1, fb_constants, global_nanvar,
                    backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor,
                    matlab_compat=matlab_compat, validate=validate, start_method=start_method,
                )
                result.updates.append(update)
                params = update.params_out  # commit atomically before the next mini-batch's own E-step

        result.params = params
        return result
    finally:
        restore_trial_flags()


# ==========================================================================
# P10d: coherent end-of-epoch re-score, convergence, best-epoch selection,
# coherent pi_steady, and callbacks/progress on top of P10c's mechanical
# mini-batch scheduling/update primitives. Deliberately does NOT modify `em_lsim_
# minibatch`, `m_step`, `e_step`, or any other P10a/P10b/P10c code -- all
# are CLOSED, signed off. `em_lsim_minibatch_train` (below) reuses the same
# private `_resolve_schedule`/`_run_one_minibatch_update` helpers `em_lsim_
# minibatch` itself uses (§P10c/P10d correction round: an earlier version
# instead called the already-public `em_lsim_minibatch` once per epoch,
# which made it recompute `compute_global_nanvar` once per epoch instead of
# once per run, and made `minibatch_hook` fire only after each whole epoch
# had already committed rather than live, per mini-batch -- see `em_lsim_
# minibatch_train`'s own docstring for both).
# ==========================================================================


def _coherent_epoch_loglik(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    fb_constants: FbConstants,
    *,
    backend: str,
    backend_options: dict | None,
    n_jobs: int,
    executor: str | None,
    validate: str,
    start_method: str | None = None,
) -> float:
    """The coherent (single-snapshot) log-likelihood under `params`, summed
    over every trial in `Y_trials` -- §2.6's own "Epoch metrics" paragraph:
    "run a forward-only evaluation of every trial under that one final
    parameter snapshot."

    `backend='torch'` genuinely does this forward-only, via `forward_
    backward_batch(..., mode="loglik")` -- confirmed directly in `torch_
    batched.py`'s own `mode="loglik"` execution path: no backward
    recursion, no `alpha`/`beta`/`gamma`/`B`/`B_comp` transfer beyond the
    loglik values themselves (a P8a correction round exists specifically
    because an earlier version ran the full posterior computation for this
    mode and discarded everything but loglik -- the same mistake this
    function is careful not to repeat for Torch).

    `numpy`/`numba` now also run forward-only, via `run_backend_loglik`
    (`_forward_only_loglik`): the same validation, emissions, pi_0
    regularization and forward recursion as the posterior runners, summed
    over trials in the same order, so the value is bit-identical to the
    previous full `e_step(...).loglik` while skipping the backward recursion
    and `ExpectedStats` construction. The one exception is `validate="all"`,
    which keeps the full `e_step` path because its backward-side invariant
    checks need a backward pass. `benchmarks/bench_em_lsim_minibatch_train.py`
    measures the per-epoch cost."""
    if backend == "torch":
        from chmm_lsim.backends.torch_batched import forward_backward_batch

        loglik_per_trial = forward_backward_batch(
            structure, params, Y_trials, fb_constants,
            backend="torch", mode="loglik", validate=validate, **(backend_options or {}),
        )
        return float(np.sum(loglik_per_trial))

    if validate == "all":
        # validate="all" also checks backward-side invariants (gamma/abar/V2/f<=h),
        # which only a full forward+backward pass can evaluate -- keep that path.
        raw = e_step(
            structure, params, Y_trials, fb_constants,
            backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor, validate=validate,
            start_method=start_method,
        )
        return raw.loglik

    return _forward_only_loglik(
        structure, params, Y_trials, fb_constants, backend=backend, validate=validate, n_jobs=n_jobs, executor=executor,
        start_method=start_method,
    )


def _coherent_pi_steady(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    fb_constants: FbConstants,
    *,
    backend: str,
    backend_options: dict | None,
    n_jobs: int,
    executor: str | None,
    validate: str,
    sigma_diag: bool,
    start_method: str | None = None,
) -> NDArray[np.float64]:
    """The coherent (gamma-based, single-snapshot) `pi_steady` under
    `params`. Unlike loglik/AIC/BIC, this needs SMOOTHED `gamma`, so it
    cannot truthfully come from a forward-only pass at all, for any
    backend (§2.6's own "Epoch metrics" paragraph is explicit about this).
    Runs one full E-step (forward AND backward -- via the unmodified
    `e_step`, which already dispatches to `forward_backward_batch(...,
    mode="stats")` for `backend='torch'`) and finalizes it, purely to reach
    `gamma_sum`/`T_all`, discarding everything else. Must be called exactly
    ONCE per training run, under the SELECTED best-epoch snapshot -- never
    once per epoch, which is exactly the extra-backward-pass-every-epoch
    cost §2.6's own design exists to avoid: "This avoids an extra backward
    pass every epoch while ensuring EmResult.params and EmResult.pi_steady
    describe the same model.\""""
    raw = e_step(
        structure, params, Y_trials, fb_constants,
        backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor, validate=validate,
        start_method=start_method,
    )
    stats = finalize_expected_stats(raw, sigma_diag=sigma_diag)
    return stats.gamma_sum / stats.T_all


def em_lsim_minibatch_train(
    structure: LSIMStructure,
    params: LSIMParams,
    Y_trials: list[NDArray[np.float64]],
    *,
    max_iterations: int | None = None,
    num_minibatches: int | _Unset = _UNSET,
    shuffle: bool | _Unset = _UNSET,
    random_state: int | np.random.Generator | None | _Unset = _UNSET,
    schedule: MinibatchSchedule | None = None,
    fb_constants: FbConstants | None = None,
    backend: str = "numpy",
    backend_options: dict | None = None,
    n_jobs: int = 1,
    executor: str | None = None,
    matlab_compat: bool = True,
    check_convergence: "bool | _Unset" = _UNSET,
    validate: str = "params",
    left2right: bool = False,
    raw_init_model: bool = False,
    epoch_hook=None,
    minibatch_hook=None,
    convergence: str | None = None,
    robust_convergence: RobustConvergence | None = None,
    coherent_eval_interval: "int | str" = 1,
    start_method: str | None = None,
) -> MinibatchTrainingResult:
    """Coherent mini-batch training driver -- runs its own mini-batch loop directly
    (via the same private `_run_one_minibatch_update` helper `em_lsim_
    minibatch` (P10c) uses internally), and adds coherent end-of-epoch
    log-likelihood/AIC/BIC, convergence, best-epoch selection, coherent
    `pi_steady`, and callbacks/progress reporting on top. Does NOT call
    `em_lsim_minibatch` itself (a prior version did, once per epoch --
    changed in the §P10c/P10d correction round specifically so `minibatch_
    hook` can fire immediately after each mini-batch commits, before the
    next one starts, rather than only after `em_lsim_minibatch` had already
    finished an entire epoch -- see "Callbacks/progress" below). `em_lsim`,
    `e_step`, `m_step`, and `em_lsim_minibatch` itself remain completely
    unmodified either way.

    **Scheduling is resolved exactly as `em_lsim_minibatch` itself would**
    -- both now call the same shared `_resolve_schedule` helper, so this
    logic exists in exactly one place: either pass `schedule=` directly
    (e.g. a complete P10a golden fixture's own multi-epoch schedule, for
    exact replay including its own stored coherent metrics), or let this
    generate one from `max_iterations`/`num_minibatches`/`shuffle`/`random_
    state` (the same `_UNSET`-sentinel contract as `em_lsim_minibatch`:
    supplying any of those three together with an explicit `schedule=`
    raises, rather than being silently ignored).

    **`compute_global_nanvar` runs exactly ONCE per training run** (§P10c/
    P10d correction round: an earlier version called `em_lsim_minibatch`
    once per epoch, and `em_lsim_minibatch` itself calls `compute_global_
    nanvar` once per call -- meaning `max_iterations` whole-dataset variance
    scans instead of one, a real scalability defect this function's own
    contract explicitly disclaims at the ~100,000-trial scale P10 targets).
    Computed once, up front, and threaded through every epoch's own mini-
    batch updates via `_run_one_minibatch_update`. `T_all` (needed for
    `compute_aic_bic`) is computed AFTER `compute_global_nanvar` has already
    validated every trial -- not before, which would let a malformed `Y`
    (wrong shape/dtype, `Inf`, `T=0`) escape as a raw, unattributed indexing
    error instead of `compute_global_nanvar`'s own clear, per-trial-indexed
    `ValidationError`.

    **Coherent per-epoch evaluation.** After each epoch's mini-batches have
    all committed, `_coherent_epoch_loglik` evaluates every trial in
    `Y_trials` under that epoch's own single final parameter snapshot
    (forward-only for `backend='torch'`; a full E-step, discarding the
    backward pass's other outputs, for `numpy`/`numba` -- see that
    function's own docstring for why), and `compute_aic_bic` derives
    AIC/BIC from it. This -- not `MinibatchEmResult.training_batch_loglik_
    by_epoch`, the MIXED pre-update diagnostic P10c already exposes -- is
    what populates `loglik_trace`/`aic_trace`/`bic_trace` here, convergence,
    and best-epoch selection: every entry describes one single, coherent
    parameter snapshot, never a mix of pre-update mini-batch scores.

    **Convergence and best-epoch selection reproduce `em_lsim`'s own exact
    rule, at epoch granularity** (confirmed identical to `em_lsim`'s own
    iteration-level logic in `tests/test_em_minibatch_train.py`, using the
    same monkeypatch methodology `tests/test_em.py` established for
    `em_lsim` itself): convergence is checked BEFORE the best-epoch
    snapshot, using the last 6 entries of `loglik_trace` (5 diffs) once
    `epoch_idx > 9`, breaking if at least 2 of those 5 diffs are `<0.001` --
    a `break` here means the loop stops with NO further epochs run (their
    updates never happen at all, unlike `em_lsim`'s own per-ITERATION break,
    which only skips that iteration's own best-iterate snapshot after
    having already computed it -- there is no equivalent "partially-run
    epoch" here since an epoch's own mini-batch updates always complete in
    full before its coherent evaluation runs) and excludes that final
    epoch's own snapshot from best-epoch consideration, exactly like
    `em_lsim`. Best-epoch selection is the same `epoch_idx>2` argmax rule
    (`epoch_idx` in `{1,2}` unconditionally overwrites the running best).

    **Coherent `pi_steady`** is evaluated exactly ONCE, via `_coherent_
    pi_steady`, under the SELECTED best-epoch's own final snapshot -- never
    once per epoch, avoiding an extra backward pass on every epoch just to
    track a metric only the winning snapshot ultimately needs.

    **Callbacks/progress -- genuinely live, not retrospective (§P10c/P10d
    correction round).** `minibatch_hook(epoch_idx, minibatch_idx, update:
    MinibatchUpdate)` fires IMMEDIATELY after each mini-batch commits --
    before the next mini-batch's own E-step begins, not merely after the
    whole epoch containing it has finished (a prior version of this
    function called `em_lsim_minibatch` for an entire epoch at once and
    fired every one of that epoch's own `minibatch_hook` calls only
    afterward, which is retrospective notification, not progress: on a
    large epoch it could stay silent for a long time, and if a LATER
    mini-batch in that epoch then failed, hooks for EARLIER, already-
    committed mini-batches would never fire at all). Now: if mini-batch `u`
    fails, every mini-batch before it in the same epoch has ALREADY had its
    own `minibatch_hook` call delivered (its update was already committed
    and its hook already fired, both before mini-batch `u` even started).
    If `minibatch_hook` itself raises, that exception propagates immediately
    -- no later mini-batch, and no epoch-level coherent evaluation for the
    epoch in progress, ever runs. `update.epoch_idx`/`minibatch_idx` are the
    TRUE global values (no renumbering needed -- unlike the prior
    implementation, this one never delegates to a one-epoch `em_lsim_
    minibatch` call that would otherwise always report `epoch_idx=1`).
    `epoch_hook(epoch_idx, params, coherent_loglik, aic, bic)` fires once
    per completed epoch, after its own coherent evaluation, mirroring
    `em_lsim`'s own `iter_hook(itr, params, loglik, aic, bic)` call
    signature (one fewer positional argument only because there is no
    epoch-level analogue of `em_lsim`'s per-iteration mixed-parameter
    concept to also pass).

    **Convergence policy (P10e).** `convergence=None` (default) keeps the
    previous behaviour exactly: `check_convergence=True` (its default) means
    `"legacy"`, `check_convergence=False` means `"off"`. Explicit choices:

    - `"legacy"`: `em_lsim.m`'s rule -- after epoch 9, stop when >=2 of the
      last 5 epoch deltas are < 0.001. It counts any decrease, of any size, as
      a convergence event, so on an oscillating mini-batch coherent trace it
      can stop while the trend is still improving. Kept for compatibility.
    - `"robust"`: stop after `patience` consecutive evaluated epochs whose
      change is small in either direction (`RobustConvergence`); a large
      decrease resets the count and is never treated as convergence. The
      current epoch is considered for best-epoch selection before the stop
      decision.
    - `"off"`: always run every scheduled epoch.

    `check_convergence` and `convergence` may be passed together only if they
    agree. `em_lsim` (full batch) is unchanged.

    **Coherent evaluation cadence (P10e, opt-in optimization).**
    `coherent_eval_interval=1` (default) evaluates every epoch -- exactly the
    previous behaviour. An int `k > 1` evaluates epochs `k, 2k, ...` plus the
    final scheduled epoch; `"final"` evaluates only the final scheduled epoch.
    Anything other than 1 requires `convergence="off"`. Mini-batch updates are
    identical whatever the cadence; only which snapshots are scored changes.
    `evaluated_epoch_indices` lists the evaluated epochs, and
    `loglik_trace`/`aic_trace`/`bic_trace` have one entry per EVALUATED epoch.
    Best-epoch selection only ever chooses an evaluated snapshot, and
    `epoch_hook` fires only for evaluated epochs (non-evaluated epochs have no
    coherent loglik to report; use `minibatch_hook` for per-update progress).
    Only the current best snapshot is retained, so memory does not grow with
    the number of skipped evaluations.

    **Process start method (P10e).** With `executor="process"`, pools use
    `start_method` (default `None`: "spawn"; see `_resolve_start_method` for
    the measured forkserver/fork crashes). Pass "forkserver" or "fork" only
    deliberately."""
    _reject_float32_training(backend, backend_options)  # P10f
    policy, coherent_eval_interval = _resolve_convergence(check_convergence, convergence, coherent_eval_interval)
    _resolve_start_method(start_method, executor)  # validate early (P10e)
    if robust_convergence is not None and policy != "robust":
        raise ValueError(f"robust_convergence settings were given but convergence={policy!r}")
    robust_monitor = _RobustConvergenceMonitor(robust_convergence or RobustConvergence()) if policy == "robust" else None
    if fb_constants is None:
        fb_constants = FbConstants()
    if raw_init_model:
        params = params.copy()
        params.pi_0 = apply_left2right_pi0_smoothing(params.pi_0, structure, left2right)

    num_trials = len(Y_trials)
    if num_trials == 0:
        raise ValueError("Y_trials must contain at least one trial")
    sigma_diag = params.sigma_diag

    full_schedule = _resolve_schedule(
        Y_trials, max_iterations=max_iterations, num_minibatches=num_minibatches, shuffle=shuffle,
        random_state=random_state, schedule=schedule,
    )

    # Computed exactly ONCE, before any epoch's own mini-batch updates --
    # never recomputed per epoch. Also validates every trial (raises a
    # clear, per-trial-indexed ValidationError for a malformed Y), so T_all
    # below is safe to compute with a plain, unvalidating sum().
    restore_trial_flags = _lock_trials_read_only(Y_trials)  # P10f: read-only while this run uses them
    try:
        global_nanvar = emission_model_for(params.emissions).global_statistics(structure, Y_trials)
        trials = _ValidatedTrials(Y_trials, structure)  # P10f: validated snapshot; internal E-steps/scoring skip re-validation
        T_all = sum(int(Y.shape[1]) for Y in Y_trials)

        all_updates: list[MinibatchUpdate] = []
        loglik_trace: list[float] = []
        aic_trace: list[float] = []
        bic_trace: list[float] = []
        evaluated_epochs: list[int] = []
        best_loglik = -np.inf
        best_params = params
        best_epoch = 0
        n_epochs_run = 0
        stop_reason = "max_iterations"
        n_scheduled = len(full_schedule.epochs)

        for e, mbs in enumerate(full_schedule.epochs):
            epoch_idx = e + 1
            for mb, trial_indices in enumerate(mbs):
                update = _run_one_minibatch_update(
                    structure, params, trials, trial_indices, epoch_idx, mb + 1, fb_constants, global_nanvar,
                    backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor,
                    matlab_compat=matlab_compat, validate=validate, start_method=start_method,
                )
                all_updates.append(update)
                params = update.params_out  # commit IMMEDIATELY -- before the hook, and before the next mini-batch

                if minibatch_hook is not None:
                    minibatch_hook(epoch_idx, update.minibatch_idx, update)

            n_epochs_run = epoch_idx

            if not (coherent_eval_interval == 1 or epoch_idx == n_scheduled
                    or (coherent_eval_interval != "final" and epoch_idx % coherent_eval_interval == 0)):
                continue  # not a scheduled evaluation: no coherent score, no hook, not a best-epoch candidate

            coherent_loglik = _coherent_epoch_loglik(
                structure, params, trials, fb_constants,
                backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor, validate=validate,
                start_method=start_method,
            )
            aic, bic = compute_aic_bic(structure, sigma_diag, coherent_loglik, T_all)
            loglik_trace.append(coherent_loglik)
            aic_trace.append(aic)
            bic_trace.append(bic)
            evaluated_epochs.append(epoch_idx)

            if epoch_hook is not None:
                epoch_hook(epoch_idx, params, coherent_loglik, aic, bic)

            # Legacy convergence is checked BEFORE the best-epoch snapshot (em_lsim.m's/
            # em_lsim's own order): a `break` here excludes this epoch's own
            # snapshot from best-epoch consideration entirely.
            if policy == "legacy" and _legacy_convergence_stop(loglik_trace, epoch_idx):
                stop_reason = "converged"
                break

            if epoch_idx > 2:
                if coherent_loglik > best_loglik:
                    best_loglik = coherent_loglik
                    best_params = params
                    best_epoch = epoch_idx
            else:
                best_loglik = coherent_loglik
                best_params = params
                best_epoch = epoch_idx

            if robust_monitor is not None and robust_monitor.update(epoch_idx, coherent_loglik):
                stop_reason = "converged"
                break

        best_pi_steady = _coherent_pi_steady(
            structure, best_params, trials, fb_constants,
            backend=backend, backend_options=backend_options, n_jobs=n_jobs, executor=executor, validate=validate,
            sigma_diag=sigma_diag, start_method=start_method,
        )

        executed_schedule = MinibatchSchedule(epochs=full_schedule.epochs[:n_epochs_run])
        minibatch_result = MinibatchEmResult(params=params, schedule=executed_schedule, updates=all_updates)

        return MinibatchTrainingResult(
            params=best_params,
            best_epoch=best_epoch,
            pi_steady=best_pi_steady,
            loglik_trace=loglik_trace,
            aic_trace=aic_trace,
            bic_trace=bic_trace,
            n_epochs=n_epochs_run,
            minibatch=minibatch_result,
            evaluated_epoch_indices=evaluated_epochs,
            convergence=policy,
            coherent_eval_interval=coherent_eval_interval,
            stop_reason=stop_reason,
        )
    finally:
        restore_trial_flags()
