"""P10b gate: deterministic CPU-parallel E-step execution (`em.e_step`'s
`n_jobs`/`executor` axis), added on top of P4/P8's existing raw/merge/
finalize contract without changing it.

Three independent things are verified here, matching docs/PYTHON_PORT_PLAN.md
section 2.6's own gate list:

1. **Contract/validation/determinism** -- `n_jobs=1` is byte-for-byte the
   pre-P10b serial path (never touches `concurrent.futures` or
   `merge_expected_stats`); `n_jobs`/`executor` are validated strictly, never
   silently coerced; `backend="torch"` rejects an outer executor outright; a
   worker exception cancels pending trials and is re-raised naming the
   original (0-based) trial index; repeated parallel runs reproduce the same
   merged result exactly (no randomness anywhere in this path).

2. **Field-by-field raw-stats equivalence** -- serial vs. parallel (thread
   and process) for both `numpy` and `numba`, across every P4 EM-trace
   fixture (diagonal/full covariance, missing data, unequal trial lengths,
   low-mass fallback triggers), plus hand-built T=1-only and large-offset
   scenarios no committed fixture covers. Also numpy-vs-numba agreement on
   the parallel path itself, and a "manual" stable-order merge (plain
   single-trial `e_step` calls folded through `merge_expected_stats` by
   hand, entirely independent of `_run_parallel_e_step`'s own internals)
   against `e_step(..., n_jobs>1)`.

3. **P10a golden-schedule replay** -- for every stored mini-batch update in
   every one of the six `p10a_scheduling__*.npz` fixtures (§P10a), rebuild
   that update's own `params_in`, select exactly its own stored ordered
   trial subset, run the new serial and parallel raw E-step, finalize, call
   the existing `m_step`, and compare all six returned parameter tensors
   against MATLAB's own stored `updates_*_out` -- the first time any Python
   E-step/M-step call is checked against `em_lsim_par.m`'s own mini-batch
   output, not merely against `em_lsim.m`'s full-batch trace (P4) or P10a's
   own internal MATLAB-side cross-checks. The SERIAL replay covers all six
   scenarios; the PARALLEL replay (both executors) covers all six scenarios
   for `backend='numpy'`, and a representative 3-scenario subset for
   `backend='numba'` (see `test_golden_schedule_replay_parallel_matches_
   updates_out_numba`'s own docstring for why that's sufficient rather than
   a coverage gap).

   Every trial in every one of the six current P10a fixtures has `T>1`
   (confirmed directly: `min(length_observation)` is 14-16 across all six),
   so no update in any of them is ever zero-transition -- the `if raw.
   n_transitions == 0: continue` branch below exists defensively (§2.6's own
   still-open coupling-denominator policy question, explicitly deferred to
   P10c) but is never actually exercised by this replay. The REAL zero-
   transition coverage is the dedicated synthetic `test_all_T1_trials_
   serial_vs_parallel_stay_finite_and_match` below, which builds an
   all-`T=1` mini-batch by hand specifically to hit it.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import P10A_SCENARIO_IDS, load_em_trace_fixture, load_p10a_scheduling_fixture  # noqa: E402

from chmm_lsim.backends.base import UnsupportedCombination
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.learning.em_loop import TrialExecutionError, finalize_expected_stats, merge_expected_stats
from chmm_lsim.models.lsim.em import e_step, m_step
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure

RTOL, ATOL = 1e-8, 1e-9  # same as test_em.py's P4 gate -- measured there against real MATLAB fixtures
EXECUTORS = ("thread", "process")


# --------------------------------------------------------------------------
# small, self-contained synthetic model -- used by every validation/
# determinism/exception test below, so those tests don't depend on any
# fixture file at all.
# --------------------------------------------------------------------------


def _toy_model(rng, *, C=2, M=(2, 3), d=(1, 2), K=(1, 2), sigma_diag=True):
    structure = LSIMStructure(channel_state_num=M, channel_obs_dim=d, num_gmm_component=K)
    S = structure.S
    pi_0 = np.zeros(S)
    transition = np.zeros((S, S))
    for c in range(C):
        sl = structure.state_slice(c)
        width = sl.stop - sl.start
        pi_0[sl] = 1.0 / width
        transition[:, sl] = 1.0 / width
    coupling = np.full((C, C), 1.0 / C)
    emissions = []
    for c in range(C):
        M_c, K_c, d_c = structure.channel_state_num[c], structure.num_gmm_component[c], structure.channel_obs_dim[c]
        weights = np.full((M_c, K_c), 1.0 / K_c)
        mu = rng.normal(size=(d_c, M_c, K_c))
        if sigma_diag:
            sigma = np.ones((d_c, M_c, K_c))
        else:
            sigma = np.tile(np.eye(d_c)[:, :, None, None], (1, 1, M_c, K_c))
        emissions.append(ChannelGMM(weights=weights, mu=mu, sigma=sigma, diag=sigma_diag))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    return structure, params


def _toy_trials(rng, structure, lengths):
    return [rng.normal(size=(structure.D, T)) for T in lengths]


class _CountingThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Test double, not a mock: behaves exactly like `ThreadPoolExecutor`
    (subclasses it, delegates everything) but records how many submitted
    tasks are outstanding -- submitted but not yet `.result()`ed -- at every
    point in time. `submit`/`.result()` are only ever called from the single
    coordinator thread inside `_run_parallel_e_step` (worker threads never
    call back into either), so `outstanding`/`max_outstanding` need no lock
    of their own despite the pool itself being multi-threaded."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.outstanding = 0
        self.max_outstanding = 0

    def submit(self, fn, *args, **kwargs):
        self.outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self.outstanding)
        fut = super().submit(fn, *args, **kwargs)
        original_result = fut.result

        def _tracked_result(timeout=None):
            try:
                return original_result(timeout)
            finally:
                self.outstanding -= 1

        fut.result = _tracked_result
        return fut


# ==========================================================================
# 1. Contract, validation, determinism, exception handling
# ==========================================================================


@pytest.mark.parametrize("n_jobs", [0, -1, 1.5, "2", True, False])
def test_invalid_n_jobs_raises(n_jobs):
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [5, 5])
    with pytest.raises(ValueError, match="n_jobs"):
        e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=n_jobs, executor="thread" if n_jobs not in (0, -1) else None)


def test_invalid_executor_name_raises():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [5, 5])
    with pytest.raises(ValueError, match="executor"):
        e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=2, executor="bogus")


def test_n_jobs_greater_than_one_requires_explicit_executor():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [5, 5])
    with pytest.raises(ValueError, match="requires an explicit executor"):
        e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=2, executor=None)


def test_n_jobs_one_rejects_explicit_executor():
    """n_jobs=1 has no executor to apply -- passing one anyway must raise,
    not be silently ignored (a caller who thinks they configured process-
    pool execution but actually got the serial path should find out)."""
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [5, 5])
    with pytest.raises(ValueError, match="n_jobs=1"):
        e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=1, executor="thread")


@pytest.mark.parametrize("n_jobs,executor", [(2, "thread"), (4, "process")])
def test_torch_backend_rejects_any_outer_executor_request(n_jobs, executor):
    """backend='torch' owns its own batch/device parallelism; n_jobs=1,
    executor=None is the only combination it accepts from this axis (NOT
    included in this parametrization -- it is the one non-error case, and
    asserting it "succeeds" here would require torch actually being
    installed to reach `forward_backward_batch` at all, which is unrelated
    to what this test checks). Everything else (including an otherwise-
    valid numpy/numba combination) must be rejected outright, not silently
    downgraded to n_jobs=1 -- `_validate_n_jobs_executor` raises this before
    `e_step` ever touches the `backend='torch'` dispatch branch, so this
    passes whether or not torch is installed."""
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [5, 5])
    with pytest.raises(UnsupportedCombination, match="torch"):
        e_step(structure, params, Y_trials, FbConstants(), backend="torch", n_jobs=n_jobs, executor=executor)


def test_empty_Y_trials_raises_for_parallel_path():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    with pytest.raises(ValueError, match="at least one trial"):
        e_step(structure, params, [], FbConstants(), backend="numpy", n_jobs=2, executor="thread")


def test_n_jobs_one_never_touches_the_parallel_machinery(monkeypatch):
    """Whitebox proof that n_jobs=1 (the default) runs the pre-P10b serial
    loop unchanged: monkeypatching concurrent.futures' executors to raise if
    ever constructed must not affect an n_jobs=1 call at all."""
    import concurrent.futures as cf

    import chmm_lsim.models.lsim.em as em_mod

    def _boom(*a, **k):
        raise AssertionError("n_jobs=1 must never construct an executor")

    monkeypatch.setattr(cf, "ThreadPoolExecutor", _boom)
    monkeypatch.setattr(cf, "ProcessPoolExecutor", _boom)
    monkeypatch.setattr(em_mod, "merge_expected_stats", _boom)

    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [5, 1, 8])
    stats = e_step(structure, params, Y_trials, FbConstants(), backend="numpy")  # n_jobs=1 default
    assert stats.n_trials == 3  # ran to completion, unaffected by the monkeypatches above


@pytest.mark.parametrize("executor", EXECUTORS)
def test_repeated_parallel_runs_reproduce_the_same_merged_result(executor):
    rng = np.random.default_rng(2)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 4, 1, 7, 13])
    results = [e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=3, executor=executor) for _ in range(3)]
    for r in results[1:]:
        np.testing.assert_array_equal(r.zeta, results[0].zeta)
        np.testing.assert_array_equal(r.gamma_sum, results[0].gamma_sum)
        assert r.loglik == results[0].loglik
        assert (r.n_trials, r.n_transitions, r.T_all) == (results[0].n_trials, results[0].n_transitions, results[0].T_all)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_worker_exception_reports_original_trial_index(executor):
    """A malformed trial (wrong observation-dimension count) at a known
    position must surface as TrialExecutionError naming that exact 0-based
    index, regardless of which other trials around it succeed."""
    rng = np.random.default_rng(3)
    structure, params = _toy_model(rng)
    bad_index = 2
    Y_trials = _toy_trials(rng, structure, [5, 6, 7, 8])
    Y_trials[bad_index] = rng.normal(size=(structure.D + 1, 7))  # wrong D -> ValidationError inside run_backend
    with pytest.raises(TrialExecutionError) as exc_info:
        e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=2, executor=executor)
    assert exc_info.value.trial_index == bad_index
    assert exc_info.value.original_exception is not None


def test_trial_execution_error_is_exported_from_top_level_package():
    """The public `chmm_lsim.e_step` can raise `TrialExecutionError` (the test
    above triggers exactly this), so a caller must be able to catch it via
    `chmm_lsim.TrialExecutionError`/`from chmm_lsim import TrialExecutionError`, not
    only via the private `chmm_lsim.em` module path."""
    import chmm_lsim

    assert chmm_lsim.TrialExecutionError is TrialExecutionError
    assert "TrialExecutionError" in chmm_lsim.__all__

    from chmm_lsim import TrialExecutionError as reimported

    assert reimported is TrialExecutionError


def test_worker_exception_cancels_pending_trials_thread():
    """n_jobs=2 with 7 trials (1 fast-failing + 6 individually expensive)
    means at most 2 of the 6 expensive trials can already be running when
    the main thread notices trial 0's exception and cancels the rest -- if
    cancellation did not happen at all, every one of the 6 expensive trials
    would run to completion in the background before e_step could return,
    making the failing call's wall time comparable to running all of them.
    This compares against exactly that (an all-succeeding baseline of the
    same shape, same n_jobs) rather than an absolute threshold, so the test
    self-calibrates to the machine's own speed."""
    rng = np.random.default_rng(4)
    structure, params = _toy_model(rng)
    n_slow = 6
    T_slow = 4000

    good_trials = _toy_trials(rng, structure, [T_slow] * n_slow)
    t0 = time.perf_counter()
    e_step(structure, params, good_trials, FbConstants(), backend="numpy", n_jobs=2, executor="thread")
    baseline_all_succeed = time.perf_counter() - t0

    bad_trials = [rng.normal(size=(structure.D + 1, 3))] + good_trials  # trial 0 fails instantly
    t0 = time.perf_counter()
    with pytest.raises(TrialExecutionError) as exc_info:
        e_step(structure, params, bad_trials, FbConstants(), backend="numpy", n_jobs=2, executor="thread")
    failing_call = time.perf_counter() - t0

    assert exc_info.value.trial_index == 0
    assert failing_call < baseline_all_succeed / 2, (
        f"failing call ({failing_call:.3f}s) was not meaningfully faster than running every trial to completion "
        f"({baseline_all_succeed:.3f}s) -- pending trials do not appear to have been cancelled"
    )


# --------------------------------------------------------------------------
# Bounded in-flight scheduling (§P10b correction round): _run_parallel_
# e_step must hold at most O(n_jobs) outstanding Futures/ExpectedStats at
# once, never O(len(Y_trials)) -- the whole point of the rewrite, for the
# intended ~100,000-trial workload.
# --------------------------------------------------------------------------


def test_bounded_in_flight_stays_bounded_as_n_grows():
    """Instrumented-executor proof, not an inference from reading the code:
    monkeypatch concurrent.futures.ThreadPoolExecutor with a subclass that
    records how many submitted tasks are outstanding at every point in
    time, run e_step over a LARGE number of trials, and confirm the
    observed maximum never exceeds em._IN_FLIGHT_MULTIPLIER * n_jobs --
    regardless of how large N is (parametrized across two very different
    N to rule out a bound that merely happens to hold for one size)."""
    import chmm_lsim.models.lsim.em as em_mod

    tracker: dict = {}
    OriginalCls = concurrent.futures.ThreadPoolExecutor

    class _Tracked(_CountingThreadPoolExecutor):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            tracker["instance"] = self

    for N in (200, 2000):
        tracker.clear()
        rng = np.random.default_rng(7)
        structure, params = _toy_model(rng)
        Y_trials = _toy_trials(rng, structure, [3] * N)
        n_jobs = 4

        with unittest.mock.patch.object(concurrent.futures, "ThreadPoolExecutor", _Tracked):
            stats = e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=n_jobs, executor="thread")

        assert stats.n_trials == N
        bound = n_jobs * em_mod._IN_FLIGHT_MULTIPLIER
        observed = tracker["instance"].max_outstanding
        assert 0 < observed <= bound, (
            f"N={N}: observed max outstanding tasks {observed} exceeds the documented bound "
            f"n_jobs*_IN_FLIGHT_MULTIPLIER={bound} -- coordinator memory is no longer O(n_jobs)"
        )

    assert concurrent.futures.ThreadPoolExecutor is OriginalCls  # patch cleanly reverted, not leaked to later tests


def test_bounded_in_flight_never_retains_full_result_list():
    """Whitebox proof of the OTHER half of the memory claim: even though
    `_run_parallel_e_step` merges `len(Y_trials)` results, it must never
    hold a Python list containing all of them at once -- monkeypatch
    `merge_expected_stats` to record the length of every list it is ever
    called with, and confirm every call site only ever passes it exactly 2
    chunks (the running merge plus the newest one), never a long list."""
    import chmm_lsim.models.lsim.em as em_mod

    call_lengths = []
    original_merge = em_mod.merge_expected_stats

    def _tracking_merge(chunks):
        call_lengths.append(len(chunks))
        return original_merge(chunks)

    rng = np.random.default_rng(8)
    structure, params = _toy_model(rng)
    N = 500
    Y_trials = _toy_trials(rng, structure, [3] * N)

    with unittest.mock.patch.object(em_mod, "merge_expected_stats", _tracking_merge):
        stats = e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=4, executor="thread")

    assert stats.n_trials == N
    assert len(call_lengths) == N - 1  # one merge call per trial after the first, never a single N-way call
    assert max(call_lengths) == 2, f"a merge call received {max(call_lengths)} chunks at once -- expected exactly 2 every time"


def test_large_N_bounded_parallel_matches_manual_stable_order_merge():
    """Functional correctness at the scale the bounded rewrite exists for:
    N=2000 small trials, compared against a manually-computed stable-order
    merge (plain single-trial e_step calls folded through merge_expected_
    stats by hand, over the WHOLE list in one call) -- proving the bounded-
    window scheduling change didn't change the actual numbers, only how
    much memory computing them requires.

    Tolerance is tight (`rtol=1e-9,atol=1e-12`, the same standard this file
    uses for every other serial-vs-parallel comparison) rather than exact:
    at this N, `loglik` was measured to differ from the manual/serial
    computation by up to ~2e-16 relative (~1 ULP) for at least one
    executor -- not from the merge arithmetic (`zeta`/`gamma_sum`/etc. all
    matched EXACTLY, `rtol=0,atol=0`, when this was first measured), but
    from the underlying per-trial `run_backend` call itself landing on a
    1-ULP-different result when executed inside a thread/process pool vs.
    plain sequential Python -- almost certainly BLAS-internal (Accelerate)
    reduction-order variance under concurrent CPU scheduling, a pre-existing
    characteristic of running any numpy computation under a pool, unrelated
    to this file's own merge-order-preservation claim (which concerns only
    the AGGREGATION step, not the per-trial kernel call)."""
    rng = np.random.default_rng(9)
    structure, params = _toy_model(rng)
    N = 2000
    Y_trials = _toy_trials(rng, structure, [4] * N)

    manual_chunks = [e_step(structure, params, [Y], FbConstants(), backend="numpy") for Y in Y_trials]
    manual_merged = merge_expected_stats(manual_chunks)

    for executor in EXECUTORS:
        parallel = e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=4, executor=executor)
        _assert_stats_close(manual_merged, parallel, rtol=1e-9, atol=1e-12, label=f"large-N-bounded-vs-manual/{executor}")


# ==========================================================================
# 2. Field-by-field raw-stats equivalence: serial vs. parallel
# ==========================================================================

_ALL_STATS_FIELDS = ("zeta", "gamma_first", "gamma_sum", "n_trials", "n_transitions", "T_all", "loglik")
_ALL_CHANNEL_FIELDS = (
    "resp", "resp_pos", "n_w_pos_diag", "n_w_pos_full",
    "weighted_W", "weighted_mean", "weighted_M2",
    "pairwise_Q", "pairwise_M", "pairwise_C",
    "u_pairwise_Q", "u_pairwise_M", "u_pairwise_C",
    "n_obs", "welford_mean", "welford_M2",
    "allzero_n", "allzero_mean", "allzero_M2",
)


def _assert_stats_close(a, b, *, rtol, atol, label):
    for f in _ALL_STATS_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if isinstance(va, (int, float)) and not isinstance(va, np.ndarray):
            np.testing.assert_allclose(vb, va, rtol=rtol, atol=atol, err_msg=f"{label}: {f}")
        else:
            np.testing.assert_allclose(vb, va, rtol=rtol, atol=atol, err_msg=f"{label}: {f}")
    assert len(a.channels) == len(b.channels)
    for ci, (ca, cb) in enumerate(zip(a.channels, b.channels)):
        for f in _ALL_CHANNEL_FIELDS:
            va, vb = getattr(ca, f), getattr(cb, f)
            if va is None:
                assert vb is None, f"{label}: channel {ci}.{f} expected None"
                continue
            np.testing.assert_allclose(vb, va, rtol=rtol, atol=atol, err_msg=f"{label}: channel {ci}.{f}")


def _build_em_trace(fx):
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    fbc = FbConstants(
        variant=fx.fb_constants.variant,
        beta_floor=fx.fb_constants.beta_floor,
        pi0_reg=fx.fb_constants.pi0_reg,
        f_reg=fx.fb_constants.f_reg,
        svi_eps=fx.fb_constants.svi_eps,
        poise_refinement=fx.fb_constants.poise_refinement,
    )
    return structure, fbc


def _params_from_packed(structure, sigma_diag, packed: dict) -> LSIMParams:
    return LSIMParams.from_packed(
        structure,
        pi_0=packed["pi_0"],
        transition=packed["transition_matrices_IM"],
        coupling=packed["coupling_tetha_IM"],
        P_all=packed["P_all"],
        mu_all=packed["mu_all"],
        sigma_all=packed["sigma_all"],
        sigma_diag=sigma_diag,
    )


@pytest.mark.parametrize("executor", EXECUTORS)
def test_serial_vs_parallel_numpy_matches_field_by_field(fixtures_dir, em_trace_scenario_id, executor):
    """Every P4 EM-trace fixture -- diagonal/full covariance, missing data
    (c3_het_full_nan), unequal trial lengths (c3_trials_unequal), low-mass
    fallback triggers (c3_lowmass_diag/c3_lowmass_full/c3_fallback_differs),
    heterogeneous K (the 'c3_het_*' fixtures) -- run through backend='numpy'
    at n_jobs=1 (existing serial path) and n_jobs>1 (thread/process), every
    raw ExpectedStats/ChannelExpectedStats field compared directly."""
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    structure, fbc = _build_em_trace(fx)
    sigma_diag = bool(fx.sigma_diag)
    params = _params_from_packed(structure, sigma_diag, fx.params_in(1))
    Y_trials = fx.Y_trials
    n_jobs = min(4, max(2, len(Y_trials)))

    serial = e_step(structure, params, Y_trials, fbc, backend="numpy", validate="off")
    parallel = e_step(structure, params, Y_trials, fbc, backend="numpy", validate="off", n_jobs=n_jobs, executor=executor)
    _assert_stats_close(serial, parallel, rtol=1e-9, atol=1e-12, label=f"{em_trace_scenario_id}/{executor}")

    m_serial = m_step(structure, finalize_expected_stats(serial, sigma_diag=sigma_diag), sigma_diag=sigma_diag)
    m_parallel = m_step(structure, finalize_expected_stats(parallel, sigma_diag=sigma_diag), sigma_diag=sigma_diag)
    np.testing.assert_allclose(m_parallel.pi_0, m_serial.pi_0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(m_parallel.transition, m_serial.transition, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_serial_vs_parallel_numba_matches_field_by_field(fixtures_dir, em_trace_scenario_id, executor):
    pytest.importorskip("numba")
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    structure, fbc = _build_em_trace(fx)
    sigma_diag = bool(fx.sigma_diag)
    params = _params_from_packed(structure, sigma_diag, fx.params_in(1))
    Y_trials = fx.Y_trials
    n_jobs = min(4, max(2, len(Y_trials)))

    serial = e_step(structure, params, Y_trials, fbc, backend="numba", validate="off")
    parallel = e_step(structure, params, Y_trials, fbc, backend="numba", validate="off", n_jobs=n_jobs, executor=executor)
    _assert_stats_close(serial, parallel, rtol=1e-9, atol=1e-12, label=f"{em_trace_scenario_id}/{executor}/numba")


def test_numpy_vs_numba_parallel_matches_within_measured_tolerance(fixtures_dir, em_trace_scenario_id):
    """NumPy and numba are two independently-implemented kernels (P7); this
    checks they agree not just per-trial (test_backends_numba.py) but on the
    parallel-executed, aggregated raw ExpectedStats -- confirming the P10b
    parallel path doesn't introduce a backend-specific discrepancy beyond
    P7's own already-measured per-trial tolerance."""
    pytest.importorskip("numba")
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    structure, fbc = _build_em_trace(fx)
    sigma_diag = bool(fx.sigma_diag)
    params = _params_from_packed(structure, sigma_diag, fx.params_in(1))
    Y_trials = fx.Y_trials
    n_jobs = min(4, max(2, len(Y_trials)))

    stats_np = e_step(structure, params, Y_trials, fbc, backend="numpy", validate="off", n_jobs=n_jobs, executor="thread")
    stats_nb = e_step(structure, params, Y_trials, fbc, backend="numba", validate="off", n_jobs=n_jobs, executor="thread")
    # measured (not guessed): worst observed relative error between numpy and
    # numba raw stats across the fixture set is on the order of the P7
    # per-trial tolerance (RTOL=1e-9 in test_backends_numba.py); aggregating
    # across trials before comparing can only add a small, bounded amount of
    # further rounding.
    _assert_stats_close(stats_np, stats_nb, rtol=1e-7, atol=1e-9, label=f"{em_trace_scenario_id}/numpy-vs-numba")


def test_manual_stable_order_merge_matches_parallel_implementation(fixtures_dir):
    """Independent of _run_parallel_e_step's own internals: call e_step once
    per single trial (plain Python loop, n_jobs=1 each time), fold the
    resulting per-trial ExpectedStats through merge_expected_stats by hand
    in Y_trials' own order, and compare against e_step(..., n_jobs>1)'s
    result on the same data -- both are, in the end, the same per-trial
    stats folded through the same merge function in the same order (the
    only difference is who calls merge_expected_stats), so this is checked
    at the same tight-but-not-exact tolerance as every other serial-vs-
    parallel comparison in this file (see `test_large_N_bounded_parallel_
    matches_manual_stable_order_merge`'s own docstring for why exact
    equality isn't guaranteed here -- measured, at larger N, to be a ~1-ULP
    per-trial `run_backend` effect, not a merge-arithmetic one)."""
    fx = load_em_trace_fixture(fixtures_dir, "c3_trials_unequal")
    structure, fbc = _build_em_trace(fx)
    sigma_diag = bool(fx.sigma_diag)
    params = _params_from_packed(structure, sigma_diag, fx.params_in(1))
    Y_trials = fx.Y_trials
    assert len(Y_trials) >= 3

    manual_chunks = [e_step(structure, params, [Y], fbc, backend="numpy", validate="off") for Y in Y_trials]
    manual_merged = merge_expected_stats(manual_chunks)

    parallel = e_step(structure, params, Y_trials, fbc, backend="numpy", validate="off", n_jobs=len(Y_trials), executor="thread")

    _assert_stats_close(manual_merged, parallel, rtol=1e-9, atol=1e-12, label="manual-merge-vs-parallel")


# --------------------------------------------------------------------------
# hand-built coverage no committed em-trace fixture provides: T=1-only
# trials (zero transitions dataset-wide) and large-offset numerical
# stability, both compared serial-vs-parallel.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("sigma_diag", [True, False])
def test_all_T1_trials_serial_vs_parallel_stay_finite_and_match(sigma_diag, executor):
    """A mini-batch of only T=1 trials has n_transitions=0 dataset-wide --
    §2.6's own zero-transition caveat is about what the M-step then does
    with that (explicitly deferred to P10c unless P10b itself hits a raw-
    statistics problem); this only checks the RAW E-step output itself
    (zeta all-zero, everything else finite) stays correct and identical
    between the serial and parallel paths."""
    rng = np.random.default_rng(5)
    structure, params = _toy_model(rng, sigma_diag=sigma_diag)
    Y_trials = _toy_trials(rng, structure, [1] * 6)

    serial = e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=1)
    parallel = e_step(structure, params, Y_trials, FbConstants(), backend="numpy", n_jobs=3, executor=executor)

    assert serial.n_transitions == 0
    assert np.all(serial.zeta == 0.0)
    for s in (serial, parallel):
        assert np.all(np.isfinite(s.zeta))
        assert np.all(np.isfinite(s.gamma_sum))
        assert np.isfinite(s.loglik)
        for ch in s.channels:
            for f in _ALL_CHANNEL_FIELDS:
                v = getattr(ch, f)
                if v is not None:
                    assert np.all(np.isfinite(v)), f"{f} not finite"

    _assert_stats_close(serial, parallel, rtol=1e-9, atol=1e-12, label=f"T1-only/{executor}/diag={sigma_diag}")

    fin = finalize_expected_stats(serial, sigma_diag=sigma_diag)
    assert np.all(np.isfinite(fin.gamma_p0))


@pytest.mark.parametrize("executor", EXECUTORS)
def test_large_offset_diagonal_serial_vs_parallel_match(executor):
    """Same construction as test_em.py's test_diagonal_m_step_stable_at_
    large_offsets, but multi-trial and comparing serial vs. parallel
    (rather than against a MATLAB-derived constant) -- proves the parallel
    path's per-trial central-moment computation is exactly as offset-stable
    as the serial one, not merely that both eventually agree with MATLAB in
    the single-trial case that test already covers."""
    offset = 1e10
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.array([[[offset]]]), sigma=np.array([[[1e8]]]), diag=True)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])
    Y_trials = [(offset + np.arange(100.0))[None, :], (offset + 50 + np.arange(60.0))[None, :], (offset + np.arange(30.0))[None, :]]

    serial = e_step(structure, params, Y_trials, FbConstants(), validate="off")
    parallel = e_step(structure, params, Y_trials, FbConstants(), validate="off", n_jobs=2, executor=executor)
    _assert_stats_close(serial, parallel, rtol=1e-9, atol=1e-12, label=f"large-offset-diag/{executor}")

    m_serial = m_step(structure, finalize_expected_stats(serial, sigma_diag=True), sigma_diag=True)
    m_parallel = m_step(structure, finalize_expected_stats(parallel, sigma_diag=True), sigma_diag=True)
    np.testing.assert_allclose(m_parallel.emissions[0].sigma, m_serial.emissions[0].sigma, rtol=1e-9)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_large_offset_full_covariance_serial_vs_parallel_match(executor):
    offset = 1e10
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(2,), num_gmm_component=(1,))
    sigma_init = (np.eye(2) * 1e8)[:, :, None, None]
    gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.array([[[offset]], [[offset]]]), sigma=sigma_init, diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])

    x = np.arange(100.0)
    Y_a = np.stack([offset + x, offset + 2 * x])
    y2 = np.arange(40.0)
    Y_b = np.stack([offset + 3 * y2, offset - y2])
    Y_trials = [Y_a, Y_b]

    serial = e_step(structure, params, Y_trials, FbConstants(), validate="off")
    parallel = e_step(structure, params, Y_trials, FbConstants(), validate="off", n_jobs=2, executor=executor)
    _assert_stats_close(serial, parallel, rtol=1e-9, atol=1e-12, label=f"large-offset-full/{executor}")


# ==========================================================================
# 3. P10a golden-schedule replay
# ==========================================================================

_PARAM_FIELDS = ("pi_0", "transition", "coupling", "P_all", "mu_all", "sigma_all")
_PACKED_KEY_OF = {
    "pi_0": "pi_0",
    "transition": "transitions_matrices",
    "coupling": "coupling_tetha_IM",
    "P_all": "P_all",
    "mu_all": "mu_all",
    "sigma_all": "sigma_all",
}


def _p10a_structure(fx):
    return LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )


def _global_nanvar_per_channel(structure, fbc, Y_trials_all, sigma_diag, any_params):
    """MATLAB's own `em_lsim_par.m` computes the covariance ridge
    (`0.001*var(...)`, §0.4.1) from `all_observation` -- the WHOLE dataset's
    concatenated observations, built once outside the mini-batch loop
    (`em_lsim_par.m:169`) -- even though every other M-step quantity for
    that update (`Recentering`, `weight_update`, the responsibility-weighted
    mean) uses only the CURRENT mini-batch's own `all_observation_batch`
    (`em_lsim_par.m:394-401`, consumed at `:511`/`:517`/`:520`; contrast the
    ridge itself at `:508`/`:517`/`:520`, which reads `all_observation`, not
    `all_observation_batch`). This is a known, already-flagged discrepancy,
    not a P10b bug: docs/PYTHON_PORT_PLAN.md section 2.6 anticipated it
    exactly ("MATLAB code uses whole-dataset nanvar for the covariance ridge
    while estimating other moments from the current mini-batch... either
    reproduces this through a precomputed immutable global variance or
    records and tests a deliberate corrected policy") and explicitly assigns
    resolving it to mini-batch TRAINING, not to this phase.

    `nanvar_y` depends only on which values were observed (`Y`/`structure`),
    never on `params`/`gamma` -- `_accumulate_channel_trial`'s `n_obs`/
    `welford_mean`/`welford_M2` accumulation reads only `Y`/its NaN mask --
    so it is safe (and exact, confirmed to `~1e-11` below) to compute this
    once per scenario, over every trial, and reuse it for every update's own
    M-step call in this replay -- exactly the "precomputed immutable global
    variance" the plan doc describes. Verifies the rest of the E-step/M-step
    math (weighted means, responsibilities, transition, coupling) is exactly
    right, without this test taking a position on P10c's own eventual
    mini-batch training policy for the ridge source."""
    global_stats = finalize_expected_stats(
        e_step(structure, any_params, Y_trials_all, fbc, backend="numpy", validate="off"), sigma_diag=sigma_diag
    )
    return [ch.nanvar_y for ch in global_stats.channels]


def _with_global_nanvar(stats, global_nanvar):
    from dataclasses import replace as dc_replace

    return dc_replace(stats, channels=[dc_replace(ch, nanvar_y=global_nanvar[c]) for c, ch in enumerate(stats.channels)])


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_golden_schedule_replay_serial_matches_updates_out(fixtures_dir, scenario_id):
    """For every stored update in every P10a scenario (including 'fast' and
    the non-uniform 'uneven' partition): rebuild params_in, select exactly
    that update's own ordered trial subset, run the new serial raw E-step,
    finalize, call the existing m_step, and compare every one of the six
    returned parameter tensors against em_lsim_par.m's own stored
    updates_*_out -- the missing MATLAB-to-Python raw-statistics mapping
    P10a's own writeup flagged as "P10b's job"."""
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = _p10a_structure(fx)
    sigma_diag = bool(fx.sigma_diag)
    fbc = FbConstants(variant=fx.variant)
    Y_trials_all = fx.Y_trials()

    any_params = LSIMParams.from_packed(
        structure, pi_0=fx.updates_pi_0_in[0], transition=fx.updates_transition_in[0], coupling=fx.updates_coupling_in[0],
        P_all=fx.updates_P_all_in[0], mu_all=fx.updates_mu_all_in[0], sigma_all=fx.updates_sigma_all_in[0], sigma_diag=sigma_diag,
    )
    global_nanvar = _global_nanvar_per_channel(structure, fbc, Y_trials_all, sigma_diag, any_params)

    skipped_zero_transition = 0
    for u in range(fx.n_updates):
        n_tr = int(fx.updates_n_trials[u])
        trial_idx0 = fx.updates_trial_indices[u, :n_tr] - 1
        Y_trials = [Y_trials_all[i] for i in trial_idx0]

        params_in = LSIMParams.from_packed(
            structure,
            pi_0=fx.updates_pi_0_in[u], transition=fx.updates_transition_in[u], coupling=fx.updates_coupling_in[u],
            P_all=fx.updates_P_all_in[u], mu_all=fx.updates_mu_all_in[u], sigma_all=fx.updates_sigma_all_in[u],
            sigma_diag=sigma_diag,
        )

        raw = e_step(structure, params_in, Y_trials, fbc, backend="numpy", validate="off")
        if raw.n_transitions == 0:
            # zero-transition mini-batch: m_step's coupling update divides by
            # n_transitions -- the finite policy for this is explicitly
            # deferred to P10c (§P10b task scope), so this update is skipped
            # here rather than guessing a policy. Defensive only: every trial
            # in every current P10a fixture has T>1 (module docstring), so
            # this branch is never actually taken here -- tracked below so a
            # future fixture regeneration that DID introduce a zero-
            # transition update couldn't silently fall through unnoticed.
            skipped_zero_transition += 1
            continue
        stats = _with_global_nanvar(finalize_expected_stats(raw, sigma_diag=sigma_diag), global_nanvar)
        params_out = m_step(structure, stats, sigma_diag=sigma_diag)
        packed_out = params_out.to_packed()

        for field in _PARAM_FIELDS:
            got = packed_out[_PACKED_KEY_OF[field]]
            expected = getattr(fx, f"updates_{field}_out")[u]
            np.testing.assert_allclose(
                got, expected, rtol=1e-6, atol=1e-8,
                err_msg=f"{scenario_id} update {u}: {field} != stored updates_{field}_out",
            )

    assert skipped_zero_transition == 0, (
        f"{scenario_id}: {skipped_zero_transition} update(s) were zero-transition -- this fixture no longer matches "
        f"the module docstring's claim that every trial has T>1; the zero-transition coupling-denominator policy "
        f"question (deferred to P10c) would need to be resolved before this update can be replayed"
    )


def _replay_scenario_parallel(fixtures_dir, scenario_id: str, executor: str, backend: str) -> None:
    """Shared body for the parallel golden-schedule replay: every update in
    the given scenario, running the new n_jobs>1 parallel raw E-step (both
    executors) instead of the serial path, compared against MATLAB's own
    stored `updates_*_out` -- see `test_golden_schedule_replay_serial_
    matches_updates_out`'s own docstring for what `_global_nanvar_per_
    channel`/`_with_global_nanvar` are compensating for and why."""
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = _p10a_structure(fx)
    sigma_diag = bool(fx.sigma_diag)
    fbc = FbConstants(variant=fx.variant)
    Y_trials_all = fx.Y_trials()

    any_params = LSIMParams.from_packed(
        structure, pi_0=fx.updates_pi_0_in[0], transition=fx.updates_transition_in[0], coupling=fx.updates_coupling_in[0],
        P_all=fx.updates_P_all_in[0], mu_all=fx.updates_mu_all_in[0], sigma_all=fx.updates_sigma_all_in[0], sigma_diag=sigma_diag,
    )
    global_nanvar = _global_nanvar_per_channel(structure, fbc, Y_trials_all, sigma_diag, any_params)

    exercised_any_update = False
    for u in range(fx.n_updates):
        n_tr = int(fx.updates_n_trials[u])
        trial_idx0 = fx.updates_trial_indices[u, :n_tr] - 1
        Y_trials = [Y_trials_all[i] for i in trial_idx0]
        if n_tr < 2:
            continue  # nothing to parallelize

        params_in = LSIMParams.from_packed(
            structure,
            pi_0=fx.updates_pi_0_in[u], transition=fx.updates_transition_in[u], coupling=fx.updates_coupling_in[u],
            P_all=fx.updates_P_all_in[u], mu_all=fx.updates_mu_all_in[u], sigma_all=fx.updates_sigma_all_in[u],
            sigma_diag=sigma_diag,
        )

        raw = e_step(structure, params_in, Y_trials, fbc, backend=backend, validate="off", n_jobs=min(4, n_tr), executor=executor)
        if raw.n_transitions == 0:
            continue  # not exercised by any current P10a fixture -- every trial has T>1 (see module docstring)
        exercised_any_update = True
        stats = _with_global_nanvar(finalize_expected_stats(raw, sigma_diag=sigma_diag), global_nanvar)
        params_out = m_step(structure, stats, sigma_diag=sigma_diag)
        packed_out = params_out.to_packed()

        for field in _PARAM_FIELDS:
            got = packed_out[_PACKED_KEY_OF[field]]
            expected = getattr(fx, f"updates_{field}_out")[u]
            np.testing.assert_allclose(
                got, expected, rtol=1e-6, atol=1e-8,
                err_msg=f"{scenario_id}/{executor}/{backend} update {u}: {field} != stored updates_{field}_out",
            )
    assert exercised_any_update, f"{scenario_id}/{executor}/{backend}: no update had >=2 trials -- nothing was actually parallelized"


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
@pytest.mark.parametrize("executor", EXECUTORS)
def test_golden_schedule_replay_parallel_matches_updates_out_numpy(fixtures_dir, scenario_id, executor):
    """Full P10b coverage requirement: every update in ALL SIX P10a
    scenarios (`diag`/`full`/`missing`/`unequal`/`uneven`/`fast`), both
    executors, `backend='numpy'` -- not a representative subset."""
    _replay_scenario_parallel(fixtures_dir, scenario_id, executor, backend="numpy")


@pytest.mark.parametrize("scenario_id", ["diag", "full", "fast"])
@pytest.mark.parametrize("executor", EXECUTORS)
def test_golden_schedule_replay_parallel_matches_updates_out_numba(fixtures_dir, scenario_id, executor):
    """A representative SUBSET of scenarios (diagonal covariance, full
    covariance, and the 'fast' variant -- deliberately not all six), rather
    than the full numpy matrix above: numba's own per-trial numerical
    parity with numpy is already exhaustively verified field-by-field,
    across every EM-trace fixture and both executors, by
    `test_serial_vs_parallel_numba_matches_field_by_field` and
    `test_numpy_vs_numba_parallel_matches_within_measured_tolerance` --
    running the full 6-scenario x 2-executor x 2-backend replay matrix would
    roughly double this file's own process-pool-creation cost (already the
    dominant cost of the numpy replay above) for coverage that would be
    almost entirely redundant with those two tests plus the numpy replay's
    own full six-scenario coverage."""
    pytest.importorskip("numba")
    _replay_scenario_parallel(fixtures_dir, scenario_id, executor, backend="numba")
