"""P4 gate: em.py's M-step against every EM-trace fixture (em_lsim.m) --
`c3_het_diag`'s 4-iteration diagonal trace, plus the single-step fixtures
added across P4's review rounds: `c3_het_full` (ordinary full covariance),
`c3_het_full_nan` (full covariance + missingness -- an ordinary M-step case,
NOT a fallback-trigger case; see `test_low_mass_fallback_genuinely_differs...`
below for the fixtures that actually exercise the two fallback branches),
`c3_lowmass_diag` (the weight+1 fallback, §0.4.1), `c3_trials_unequal`
(multi-trial bookkeeping: `zeta`/`gamma_p0` aggregation across 3 unequal-length
trials), `c3_lowmass_full` (the full-covariance low-mass fallback trigger),
and `c3_fallback_differs` (§0.4.3: the diag-vs-full low-mass trigger
disagreement -- `fallback_mu` fires while `fallback_sigma_full` does not, on
the same (state, component)).

Tolerances, measured against the real fixture set rather than guessed, same
methodology as P2/P3a: worst observed relative error across every scenario
and iteration is `1.04e-9` (`transition`, on `c3_het_diag` iter 4) and
`3.15e-10` (`sigma_all`, on `c3_het_full_nan`) -> `rtol=1e-8, atol=1e-9`
carries comfortable margin.

**Known gap, not silently covered**: no fixture isolates the §0.4.10
eigenvalue-repair threshold at exactly 1 vs exactly 2 non-positive
eigenvalues -- the unconditional `0.001*var` ridge that em_lsim.m always adds
tends to push a rank-deficient M-step estimate back to strictly positive
definite before the repair check ever sees it, which made this surprisingly
hard to hit with real data. `repair_covariance` (the isolated repair rule) is
tested directly against hand-built matrices with a known eigenvalue count
instead (see below) -- that verifies the *rule*, not that canonical MATLAB
data can actually reach the `==1` boundary in practice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_em_trace_fixture, load_posterior_fixture  # noqa: E402

from chmm_lsim.compat import FbConstants
from chmm_lsim.em import (
    apply_left2right_pi0_smoothing,
    compute_aic_bic,
    e_step,
    em_lsim,
    finalize_expected_stats,
    m_step,
)
from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.structure import LSIMStructure

RTOL, ATOL = 1e-8, 1e-9
SCENARIO = "c3_het_diag"


def _build(fx):
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


def _params_from(structure, sigma_diag, packed: dict) -> LSIMParams:
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


def test_left2right_smoothing_matches_effective_pi0_exactly(fixtures_dir, em_trace_scenario_id):
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    structure, _ = _build(fx)
    eff = apply_left2right_pi0_smoothing(fx.init_pi_0, structure, left2right=False)
    np.testing.assert_array_equal(eff, fx.effective_pi_0_iter1)


def test_effective_params_other_than_pi0_equal_init_exactly(fixtures_dir, em_trace_scenario_id):
    """SCHEMA.md: em_lsim.m:159-164 touches only pi_0; every other parameter
    passes through unchanged from init_model to iteration 1's effective params."""
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    np.testing.assert_array_equal(fx.init_transitions_matrices, fx.effective_transitions_matrices_iter1)
    np.testing.assert_array_equal(fx.init_coupling_tetha_IM, fx.effective_coupling_tetha_IM_iter1)
    np.testing.assert_array_equal(fx.init_P_all, fx.effective_P_all_iter1)
    np.testing.assert_array_equal(fx.init_mu_all, fx.effective_mu_all_iter1)
    np.testing.assert_array_equal(fx.init_sigma_all, fx.effective_sigma_all_iter1)


def test_one_e_step_m_step_reproduces_params_out(fixtures_dir, em_trace_scenario_id):
    """P4's primary gate: params_in[itr] through one E-step + M-step
    reproduces params_out[itr], and the E-step's own loglik/AIC/BIC match
    loglik_params_in[itr]/AIC_params_in[itr]/BIC_params_in[itr] -- not
    params_out[itr] against unqualified loglik[itr] (§3.A). Runs against
    every EM-trace fixture the manifest lists (`c3_het_diag`'s 4 diagonal
    iterations, plus P4's `c3_het_full`/`c3_het_full_nan`/`c3_lowmass_diag`/
    `c3_trials_unequal` single-step fixtures), not just one hardcoded scenario."""
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    structure, fbc = _build(fx)
    sigma_diag = bool(fx.sigma_diag)
    Y_trials = fx.Y_trials

    for itr in range(1, len(fx.iterations) + 1):
        params = _params_from(structure, sigma_diag, fx.params_in(itr))

        stats = finalize_expected_stats(e_step(structure, params, Y_trials, fbc, validate="off"), sigma_diag=sigma_diag)
        aic, bic = compute_aic_bic(structure, sigma_diag, stats.loglik, stats.T_all)
        new_params = m_step(structure, stats, sigma_diag=sigma_diag)

        it = fx.iterations[itr - 1]
        assert it.itr == itr
        np.testing.assert_allclose(stats.loglik, it.loglik_params_in, rtol=1e-9)
        np.testing.assert_allclose(aic, it.AIC_params_in, rtol=1e-9)
        np.testing.assert_allclose(bic, it.BIC_params_in, rtol=1e-9)

        np.testing.assert_allclose(new_params.pi_0, it.params_out_pi0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(new_params.transition, it.params_out_transition, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(new_params.coupling, it.params_out_coupling, rtol=RTOL, atol=ATOL)
        packed = new_params.to_packed()
        np.testing.assert_allclose(packed["P_all"], it.params_out_Pall, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(packed["mu_all"], it.params_out_muall, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(packed["sigma_all"], it.params_out_sigmaall, rtol=RTOL, atol=ATOL)


def test_full_replay_from_raw_init_model_reproduces_whole_trace(fixtures_dir):
    """The end-to-end em_lsim() driver, started from the fixture's raw
    init_model (not the precomputed effective params) via `raw_init_model=True`
    -- the driver applies `left2right`'s one-time smoothing itself, matching
    `em_lsim.m`'s own `nargin==6` branch rather than requiring a pre-smoothed
    caller -- must reproduce every iteration's params_out/loglik/AIC/BIC and
    land on the right best-iterate."""
    fx = load_em_trace_fixture(fixtures_dir, SCENARIO)
    structure, fbc = _build(fx)
    sigma_diag = bool(fx.sigma_diag)

    params = LSIMParams.from_packed(
        structure,
        pi_0=fx.init_pi_0,
        transition=fx.init_transitions_matrices,
        coupling=fx.init_coupling_tetha_IM,
        P_all=fx.init_P_all,
        mu_all=fx.init_mu_all,
        sigma_all=fx.init_sigma_all,
        sigma_diag=sigma_diag,
    )

    captured = []
    result = em_lsim(
        structure,
        params,
        [fx.Y],
        max_iterations=len(fx.iterations),
        fb_constants=fbc,
        check_convergence=False,
        validate="off",
        raw_init_model=True,
        left2right=False,
        iter_hook=lambda itr, p, ll, aic, bic: captured.append((itr, p, ll, aic, bic)),
    )

    assert result.n_iterations == len(fx.iterations)
    assert len(captured) == len(fx.iterations)

    for itr, p, ll, aic, bic in captured:
        it = fx.iterations[itr - 1]
        np.testing.assert_allclose(ll, it.loglik_params_in, rtol=1e-9)
        np.testing.assert_allclose(aic, it.AIC_params_in, rtol=1e-9)
        np.testing.assert_allclose(bic, it.BIC_params_in, rtol=1e-9)
        packed = p.to_packed()
        np.testing.assert_allclose(p.pi_0, it.params_out_pi0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(packed["sigma_all"], it.params_out_sigmaall, rtol=RTOL, atol=ATOL)

    # loglik is monotone increasing on this fixture, so argmax over i in {2..n} is the last iteration
    assert np.all(np.diff(result.loglik_trace) > 0)
    assert result.best_iterate == len(fx.iterations)
    np.testing.assert_allclose(result.params.pi_0, fx.iterations[-1].params_out_pi0, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# Best-iterate / convergence driver logic, isolated from E/M correctness via
# a monkeypatched e_step/m_step returning a controlled loglik sequence.
# --------------------------------------------------------------------------

def _make_tiny_params():
    structure = LSIMStructure((2,), (1,), (1,))
    gmm = ChannelGMM(weights=np.array([[1.0], [1.0]]), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 2, 1)), diag=True)
    return structure, LSIMParams(
        structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.array([[0.5, 0.5], [0.5, 0.5]]), coupling=np.ones((1, 1)), emissions=[gmm]
    )


def _fake_expected_stats(structure, params, loglik):
    """A *finalized* fake stats object (gamma_p0 already set): the driver
    calls `finalize_expected_stats` on whatever `e_step` returns, so a
    monkeypatched `e_step` must hand back something already in that state,
    or the driver's own finalize call (on an empty `channels` list, a no-op)
    would otherwise be the thing under test instead of the driver logic."""
    import chmm_lsim.em as em_module

    return em_module.ExpectedStats(
        zeta=np.zeros((structure.S, structure.S)),
        gamma_first=params.pi_0.copy(),
        gamma_p0=params.pi_0.copy(),
        gamma_sum=np.zeros(structure.S),
        n_trials=1,
        n_transitions=1,
        T_all=1,
        loglik=loglik,
        channels=[],
    )


def test_best_iterate_is_argmax_over_iterations_2_to_n(monkeypatch):
    """Iteration 1 is never the best-iterate even if it has the highest
    loglik: em_lsim.m's `if itr>2 ... else ...` unconditionally overwrites the
    snapshot at itr in {1,2}, so the argmax is effectively over {2..n}."""
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_params()
    loglik_seq = [100.0, 1.0, 2.0, 3.0]  # iter 1 is the largest value, must still lose

    calls = {"n": 0}

    def fake_e_step(structure, params, Y_trials, fb_constants, *, backend="numpy", backend_options=None, validate="params"):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return _fake_expected_stats(structure, params, ll)

    monkeypatch.setattr(em_module, "e_step", fake_e_step)
    monkeypatch.setattr(em_module, "m_step", lambda structure, stats, *, sigma_diag, matlab_compat=True: params.copy())

    result = em_module.em_lsim(structure, params, [np.zeros((1, 1))], max_iterations=4, check_convergence=False)
    assert result.loglik_trace == loglik_seq
    # candidates are iterations {2,3,4} with values [1.0, 2.0, 3.0] -- iter 1's
    # loglik of 100.0 is never a candidate, so the winner is iter 4, not iter 1
    assert result.best_iterate == 4


def test_convergence_breaks_after_five_flat_diffs(monkeypatch):
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_params()
    # 9 flat iterations then it would keep going; convergence should fire once
    # itr>9 and >=2 of the last 5 diffs are <0.001, which a flat sequence gives immediately.
    loglik_seq = [float(i) for i in range(9)] + [8.0005] * 5

    calls = {"n": 0}

    def fake_e_step(structure, params, Y_trials, fb_constants, *, backend="numpy", backend_options=None, validate="params"):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return _fake_expected_stats(structure, params, ll)

    monkeypatch.setattr(em_module, "e_step", fake_e_step)
    monkeypatch.setattr(em_module, "m_step", lambda structure, stats, *, sigma_diag, matlab_compat=True: params.copy())

    result = em_module.em_lsim(structure, params, [np.zeros((1, 1))], max_iterations=len(loglik_seq), check_convergence=True)
    assert result.n_iterations < len(loglik_seq)


def test_convergence_break_excludes_its_own_iteration_from_best_iterate(monkeypatch):
    """Regression test: em_lsim.m checks convergence BEFORE the best-iterate
    snapshot, so a `break` on the iteration that triggers convergence skips
    that iteration's own snapshot entirely -- even if its loglik would
    otherwise have been a new best. A prior version of this driver snapshotted
    first, so it returned iteration 11 here; the correct answer is 10."""
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_params()
    # iters 1-9: 0..8 (ordinary increase); iters 10-11: two consecutive small
    # diffs (<0.001) from iter 9 -- convergence's window at itr=11 is
    # trace[5:11] = [5,6,7,8.0,8.0004,8.0005], diffs = [1,1,1,0.0004,0.0001],
    # two of which are <0.001, so itr=11 is where the break fires. Iter 11's
    # own loglik (8.0005) is higher than iter 10's (8.0004) -- a new best if
    # its snapshot were taken, which it must not be.
    loglik_seq = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0004, 8.0005]

    calls = {"n": 0}

    def fake_e_step(structure, params, Y_trials, fb_constants, *, backend="numpy", backend_options=None, validate="params"):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return _fake_expected_stats(structure, params, ll)

    snapshots = {}

    def fake_m_step(structure, stats, *, sigma_diag, matlab_compat=True):
        p = params.copy()
        p.pi_0[0] = stats.loglik  # tag the snapshot with the iteration's own loglik, to identify it later
        snapshots[stats.loglik] = p
        return p

    monkeypatch.setattr(em_module, "e_step", fake_e_step)
    monkeypatch.setattr(em_module, "m_step", fake_m_step)

    result = em_module.em_lsim(structure, params, [np.zeros((1, 1))], max_iterations=len(loglik_seq), check_convergence=True)

    assert result.n_iterations == 11  # convergence fired at itr=11, ending the loop there
    assert result.best_iterate == 10  # NOT 11, even though iter 11's loglik (8.0005) is higher than iter 10's (8.0004)
    assert result.params.pi_0[0] == 8.0004


# --------------------------------------------------------------------------
# Low-mass fallback trigger: diag counts observed (dim,t) pairs; full counts
# every t regardless of Y's missingness (em_lsim.m:378 rebuilds an unmasked
# weight matrix). Regression test for the bug where a previous version
# reused the diag branch's NaN mask for the full branch's trigger too.
# --------------------------------------------------------------------------

def test_full_covariance_low_mass_trigger_counts_missing_cells():
    import chmm_lsim.em as em_module

    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(2,), num_gmm_component=(1,))
    gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.zeros((2, 1, 1)), sigma=np.eye(2)[:, :, None, None], diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])

    T = 10
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(2, T))
    Y[0, ::2] = np.nan  # dim 0 missing at half the timesteps; dim 1 fully observed

    fbc = FbConstants(variant="sed")
    stats = em_module.finalize_expected_stats(em_module.e_step(structure, params, [Y], fbc, validate="off"), sigma_diag=False)
    cs = stats.channels[0]

    # diag's trigger counts (dim,t) pairs actually observed: dim 0 contributes
    # T/2 (only where not NaN), dim 1 contributes T -- less than full's count,
    # which ignores Y's missingness entirely and counts d_c * T for every t
    # with positive responsibility (em_lsim.m's fresh, unmasked weight_update).
    assert cs.n_w_pos_full[0, 0] > cs.n_w_pos_diag[0, 0]
    assert cs.n_w_pos_full[0, 0] == 2 * T
    assert cs.n_w_pos_diag[0, 0] == T // 2 + T


def test_low_mass_fallback_genuinely_differs_between_mu_and_full_sigma(fixtures_dir):
    """§0.4.3: the same (state, component) can trigger the +1 fallback for mu
    but not for full sigma, because the two branches count differently --
    not just "both happen to be true" (clause 1, low responsibility), but
    clause 2 alone, with clause 1 false for both flags. `c3_fallback_differs`
    is a bespoke MATLAB fixture (not a build_scenario case, `export_fixtures.m`'s
    `local_build_fallback_differs_scenario`): a sticky 2-state transition keeps
    resp_pos >> 3 for both states via the Markov chain alone, while only 3 of
    60 (dim,t) cells in the whole sequence are ever observed, keeping
    n_w_pos_diag (masked) below 2*d_c=4 without affecting n_w_pos_full
    (unmasked, which ignores Y's missingness entirely). Confirmed against
    `em_lsim.m`'s own output, not just this module's own formulas."""
    fx = load_em_trace_fixture(fixtures_dir, "c3_fallback_differs")
    structure, fbc = _build(fx)
    params = _params_from(structure, bool(fx.sigma_diag), fx.params_in(1))

    stats = finalize_expected_stats(e_step(structure, params, fx.Y_trials, fbc, validate="off"), sigma_diag=False)
    cs = stats.channels[0]
    for i in (0, 1):  # both states: resp_pos >> 3, so clause 1 is false for both flags
        assert cs.resp_pos[i, 0] > 3
        assert cs.n_w_pos_diag[i, 0] < 4  # clause 2 fires for mu
        assert cs.n_w_pos_full[i, 0] >= 4  # clause 2 does NOT fire for full sigma
        assert bool(cs.fallback_mu[i, 0])
        assert not bool(cs.fallback_sigma_full[i, 0])

    # and the M-step built on top of these genuinely-differing flags still
    # matches em_lsim.m's own output -- not just the flags in isolation
    new_params = m_step(structure, stats, sigma_diag=False)
    it = fx.iterations[0]
    packed = new_params.to_packed()
    np.testing.assert_allclose(packed["mu_all"], it.params_out_muall, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(packed["sigma_all"], it.params_out_sigmaall, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# nanvar_y: mergeable Welford state, not a naive Σy²-(Σy)²/n formula.
# Regression test for exactly the catastrophic-cancellation repro that found
# the bug: large offsets blow up the naive formula; MATLAB's n_obs==1 -> 0
# convention (not NaN) is a separate, independently-verified special case.
# --------------------------------------------------------------------------

def test_nanvar_welford_stable_at_large_offsets():
    from chmm_lsim.em import _empty_channel_stats, _welford_batch, finalize_channel_stats

    for offset in (1e10, 1e12):
        x = offset + np.arange(100.0)
        cs = _empty_channel_stats(1, 1, 1, True)
        n, mean, M2 = _welford_batch(np.ones((1, 100)), x[None, :])
        cs.n_obs, cs.welford_mean, cs.welford_M2 = n, mean, M2
        fin = finalize_channel_stats(cs, sigma_diag=True)
        np.testing.assert_allclose(fin.nanvar_y[0], np.var(x, ddof=1), rtol=1e-10)


def test_nanvar_single_observation_is_zero_not_nan():
    """MATLAB R2026a: var([5], [], 2, 'omitnan') == 0, not NaN (verified
    directly against MATLAB, not assumed) -- var([], [], 2, 'omitnan') is
    empty/NaN for zero observations, which stays NaN here."""
    from chmm_lsim.em import _empty_channel_stats, _welford_batch, finalize_channel_stats

    cs = _empty_channel_stats(1, 1, 1, True)
    n, mean, M2 = _welford_batch(np.array([[1.0]]), np.array([[5.0]]))
    cs.n_obs, cs.welford_mean, cs.welford_M2 = n, mean, M2
    fin = finalize_channel_stats(cs, sigma_diag=True)
    assert fin.nanvar_y[0] == 0.0

    cs_empty = _empty_channel_stats(1, 1, 1, True)  # n_obs stays 0
    fin_empty = finalize_channel_stats(cs_empty, sigma_diag=True)
    assert np.isnan(fin_empty.nanvar_y[0])


# --------------------------------------------------------------------------
# Central component moments: the M-step's own mu/sigma, computed through the
# full E-step -> M-step path, must be stable at large offsets too -- not just
# the nanvar_y ridge term tested above. MATLAB-verified directly (not just
# asserted from the plan's formula): tools/matlab/check_large_offset_em.m
# runs em_lsim.m itself on these exact scenarios and prints the same numbers.
# --------------------------------------------------------------------------

def test_diagonal_m_step_stable_at_large_offsets():
    """C=1/M=1/K=1, Y = offset + arange(100), sigma wide enough that the
    emission likelihood itself never underflows. MATLAB's own em_lsim.m
    (tools/matlab/check_large_offset_em.m) gives sigma=834.0916667 at every
    one of these four offsets; a pre-fix raw-moment implementation gave
    -20970.68 at 1e10 and 0.84 at 1e12 (both nowhere near correct)."""
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    for offset in (0.0, 1e8, 1e10, 1e12):
        gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.array([[[offset]]]), sigma=np.array([[[1e8]]]), diag=True)
        params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])
        Y = (offset + np.arange(100.0))[None, :]

        fbc = FbConstants(variant="sed")
        stats = finalize_expected_stats(e_step(structure, params, [Y], fbc, validate="off"), sigma_diag=True)
        new_params = m_step(structure, stats, sigma_diag=True)

        np.testing.assert_allclose(new_params.emissions[0].mu[0, 0, 0], offset + 49.5, rtol=1e-10)
        np.testing.assert_allclose(new_params.emissions[0].sigma[0, 0, 0], 834.0916666667, rtol=1e-9)


def test_full_covariance_m_step_stable_at_large_offset():
    """C=1/M=1/K=1, d=2, Y=[offset+x; offset+2x] (exactly collinear, so the
    true covariance is singular) at offset=1e10. MATLAB's own em_lsim.m
    (tools/matlab/check_large_offset_em.m) gives eigenvalues
    [1.34642190719956, 4169.11191142613]; a pre-fix raw-moment implementation
    gave a minimum eigenvalue around -27320 -- not just inaccurate but the
    wrong sign, i.e. not even a valid covariance matrix."""
    offset = 1e10
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(2,), num_gmm_component=(1,))
    sigma_init = (np.eye(2) * 1e8)[:, :, None, None]
    gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.array([[[offset]], [[offset]]]), sigma=sigma_init, diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])

    x = np.arange(100.0)
    Y = np.stack([offset + x, offset + 2 * x])
    fbc = FbConstants(variant="sed")
    stats = finalize_expected_stats(e_step(structure, params, [Y], fbc, validate="off"), sigma_diag=False)
    new_params = m_step(structure, stats, sigma_diag=False)

    sigma = new_params.emissions[0].sigma[:, :, 0, 0]
    eigvals = np.sort(np.linalg.eigvalsh(sigma))
    np.testing.assert_allclose(eigvals, [1.34642190719956, 4169.11191142613], rtol=1e-6)
    assert np.all(eigvals > 0)  # a valid covariance matrix, not just "close to the right number"


# --------------------------------------------------------------------------
# merge/finalize: raw stats from disjoint chunks of trials must combine to
# the same result as one e_step call over all trials together -- the point
# of keeping ExpectedStats fixed-size and mergeable (§2.4.2, P8's chunked path).
# --------------------------------------------------------------------------

def test_merge_expected_stats_matches_single_call_over_all_trials(fixtures_dir):
    from chmm_lsim.em import merge_expected_stats

    fx = load_em_trace_fixture(fixtures_dir, "c3_trials_unequal")
    structure, fbc = _build(fx)
    sigma_diag = bool(fx.sigma_diag)
    params = _params_from(structure, sigma_diag, fx.params_in(1))
    Y_trials = fx.Y_trials
    assert len(Y_trials) == 3

    whole = finalize_expected_stats(e_step(structure, params, Y_trials, fbc, validate="off"), sigma_diag=sigma_diag)

    chunk_a = e_step(structure, params, Y_trials[:1], fbc, validate="off")
    chunk_b = e_step(structure, params, Y_trials[1:], fbc, validate="off")
    merged = finalize_expected_stats(merge_expected_stats([chunk_a, chunk_b]), sigma_diag=sigma_diag)

    np.testing.assert_allclose(merged.zeta, whole.zeta)
    np.testing.assert_allclose(merged.gamma_p0, whole.gamma_p0)
    np.testing.assert_allclose(merged.gamma_sum, whole.gamma_sum)
    assert merged.n_trials == whole.n_trials
    assert merged.T_all == whole.T_all
    np.testing.assert_allclose(merged.loglik, whole.loglik)
    for mc, wc in zip(merged.channels, whole.channels):
        np.testing.assert_allclose(mc.nanvar_y, wc.nanvar_y, equal_nan=True)
        np.testing.assert_array_equal(mc.fallback_mu, wc.fallback_mu)
        np.testing.assert_allclose(mc.resp, wc.resp)
        # every new central-moment field, not just the derived ones above
        np.testing.assert_allclose(mc.weighted_W, wc.weighted_W)
        np.testing.assert_allclose(mc.weighted_mean, wc.weighted_mean)
        np.testing.assert_allclose(mc.weighted_M2, wc.weighted_M2)
        np.testing.assert_allclose(mc.n_obs, wc.n_obs)
        np.testing.assert_allclose(mc.welford_mean, wc.welford_mean)
        np.testing.assert_allclose(mc.welford_M2, wc.welford_M2)
        np.testing.assert_allclose(mc.allzero_n, wc.allzero_n)
        np.testing.assert_allclose(mc.allzero_mean, wc.allzero_mean)
        np.testing.assert_allclose(mc.allzero_M2, wc.allzero_M2)

    params_from_merge = m_step(structure, merged, sigma_diag=sigma_diag)
    params_from_whole = m_step(structure, whole, sigma_diag=sigma_diag)
    np.testing.assert_allclose(params_from_merge.pi_0, params_from_whole.pi_0)
    np.testing.assert_allclose(params_from_merge.transition, params_from_whole.transition)


def test_merge_expected_stats_matches_single_call_full_covariance():
    """As above, but for the full-covariance pairwise fields specifically --
    no real fixture combines full covariance with multiple trials, so this
    uses a small synthetic multi-trial model (internal consistency between
    two equivalent computation paths, not a MATLAB parity claim)."""
    from chmm_lsim.em import merge_expected_stats

    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    gmm = ChannelGMM(
        weights=np.array([[1.0], [1.0]]),
        mu=np.array([[[0.0], [5.0]], [[0.0], [5.0]]]),
        sigma=np.stack([np.eye(2), np.eye(2)], axis=-1)[:, :, :, None],
        diag=False,
    )
    params = LSIMParams(
        structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.array([[0.7, 0.3], [0.2, 0.8]]), coupling=np.ones((1, 1)), emissions=[gmm]
    )
    rng = np.random.default_rng(3)
    Y_trials = [rng.normal(size=(2, 15)), rng.normal(size=(2, 22)), rng.normal(size=(2, 9))]
    for Y in Y_trials:
        Y[0, ::4] = np.nan  # scattered missingness, so the fallback machinery is exercised too

    fbc = FbConstants(variant="sed")
    whole = finalize_expected_stats(e_step(structure, params, Y_trials, fbc, validate="off"), sigma_diag=False)
    chunk_a = e_step(structure, params, Y_trials[:2], fbc, validate="off")
    chunk_b = e_step(structure, params, Y_trials[2:], fbc, validate="off")
    merged = finalize_expected_stats(merge_expected_stats([chunk_a, chunk_b]), sigma_diag=False)

    for mc, wc in zip(merged.channels, whole.channels):
        np.testing.assert_allclose(mc.pairwise_Q, wc.pairwise_Q)
        np.testing.assert_allclose(mc.pairwise_M, wc.pairwise_M)
        np.testing.assert_allclose(mc.pairwise_C, wc.pairwise_C)
        np.testing.assert_allclose(mc.u_pairwise_Q, wc.u_pairwise_Q)
        np.testing.assert_allclose(mc.u_pairwise_M, wc.u_pairwise_M)
        np.testing.assert_allclose(mc.u_pairwise_C, wc.u_pairwise_C)
        np.testing.assert_array_equal(mc.fallback_sigma_full, wc.fallback_sigma_full)

    params_from_merge = m_step(structure, merged, sigma_diag=False)
    params_from_whole = m_step(structure, whole, sigma_diag=False)
    np.testing.assert_allclose(params_from_merge.emissions[0].mu, params_from_whole.emissions[0].mu)
    np.testing.assert_allclose(params_from_merge.emissions[0].sigma, params_from_whole.emissions[0].sigma)


def test_merge_rejects_already_finalized_stats(fixtures_dir):
    from chmm_lsim.em import merge_channel_stats, merge_expected_stats

    fx = load_em_trace_fixture(fixtures_dir, "c3_het_diag")
    structure, fbc = _build(fx)
    params = _params_from(structure, True, fx.params_in(1))
    raw = e_step(structure, params, [fx.Y], fbc, validate="off")
    finalized = finalize_expected_stats(raw, sigma_diag=True)

    with pytest.raises(ValueError, match="raw"):
        merge_expected_stats([raw, finalized])
    with pytest.raises(ValueError, match="raw"):
        merge_channel_stats([raw.channels[0], finalized.channels[0]])


def test_m_step_rejects_unfinalized_stats(fixtures_dir):
    fx = load_em_trace_fixture(fixtures_dir, "c3_het_diag")
    structure, fbc = _build(fx)
    params = _params_from(structure, True, fx.params_in(1))
    raw = e_step(structure, params, [fx.Y], fbc, validate="off")
    with pytest.raises(ValueError, match="finalize"):
        m_step(structure, raw, sigma_diag=True)


# --------------------------------------------------------------------------
# EmResult.pi_steady: em_lsim.m's 8th output, snapshotted alongside params at
# the same best-iterate (gamma_sum/T_all from that iteration's own E-step).
# --------------------------------------------------------------------------

def test_pi_steady_is_snapshotted_alongside_best_iterate(fixtures_dir):
    fx = load_em_trace_fixture(fixtures_dir, "c3_het_diag")
    structure, fbc = _build(fx)
    sigma_diag = bool(fx.sigma_diag)
    params = LSIMParams.from_packed(
        structure,
        pi_0=fx.init_pi_0,
        transition=fx.init_transitions_matrices,
        coupling=fx.init_coupling_tetha_IM,
        P_all=fx.init_P_all,
        mu_all=fx.init_mu_all,
        sigma_all=fx.init_sigma_all,
        sigma_diag=sigma_diag,
    )
    result = em_lsim(
        structure, params, [fx.Y], max_iterations=len(fx.iterations), fb_constants=fbc,
        check_convergence=False, validate="off", raw_init_model=True, left2right=False,
    )
    assert result.pi_steady is not None
    assert result.pi_steady.shape == (structure.S,)
    for c in range(structure.C):
        block = result.pi_steady[structure.state_slice(c)]
        np.testing.assert_allclose(block.sum(), 1.0, rtol=0, atol=1e-8)

    # recompute pi_steady independently at the winning iteration's params_in
    # and confirm it matches what the driver snapshotted
    params_in = _params_from(structure, sigma_diag, fx.params_in(result.best_iterate)) if result.best_iterate > 1 else params
    stats = finalize_expected_stats(e_step(structure, params_in, [fx.Y], fbc, validate="off"), sigma_diag=sigma_diag)
    np.testing.assert_allclose(result.pi_steady, stats.gamma_sum / stats.T_all)


# --------------------------------------------------------------------------
# matlab_compat=False: quirk 1's corrected mu-bias (§0.4.1/§0.4.10). The
# fallback denominator becomes n_obs (observed timesteps), not T_all, so a
# missing timestep no longer contributes weight=1/value=0 to the average.
# Full covariance's fallback is deliberately left matlab_compat-invariant --
# no corrected formula for quirk 2 is specified anywhere in the plan, and
# guessing one here would be exactly the kind of unreviewed addition the P4
# review round has been correcting.
# --------------------------------------------------------------------------

def test_matlab_compat_false_removes_diag_mu_bias_toward_zero():
    import chmm_lsim.em as em_module

    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.zeros((1, 1, 1)), sigma=np.ones((1, 1, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])

    T = 3  # <= _LOW_MASS_STATE_COUNT: fallback always triggers regardless of masking
    Y = np.array([[10.0, np.nan, np.nan]])  # true signal is 10; 2 of 3 timesteps missing

    fbc = FbConstants(variant="sed")
    stats = finalize_expected_stats(em_module.e_step(structure, params, [Y], fbc, validate="off"), sigma_diag=True)
    assert bool(stats.channels[0].fallback_mu[0, 0])

    mu_true = em_module.m_step(structure, stats, sigma_diag=True, matlab_compat=True).emissions[0].mu[0, 0, 0]
    mu_false = em_module.m_step(structure, stats, sigma_diag=True, matlab_compat=False).emissions[0].mu[0, 0, 0]

    # w_sum=1, w_y=10 (the one observed point, weight 1); u_y=10 (same point,
    # zero-filled-then-summed is just that one value). The fallback numerator
    # is w_y+u_y=20 in both modes (u_y is already observed-only, unaffected by
    # T_all vs n_obs) -- only the denominator differs:
    #   matlab_compat=True:  denom = w_sum + T_all  = 1+3 = 4 -> mu = 20/4 = 5.0
    #   matlab_compat=False: denom = w_sum + n_obs  = 1+1 = 2 -> mu = 20/2 = 10.0
    # i.e. canonical MATLAB pulls the estimate toward 0 by counting the 2
    # unobserved timesteps in the denominator without adding anything to the
    # numerator for them; the corrected version recovers the true signal (10)
    # exactly, since n_obs=1 makes the fallback a no-op here (denom=2*w_sum).
    np.testing.assert_allclose(mu_true, 5.0)
    np.testing.assert_allclose(mu_false, 10.0)
    assert abs(mu_false - 10.0) < abs(mu_true - 10.0)


def test_matlab_compat_full_covariance_sigma_fallback_is_invariant():
    """Full covariance's low-mass fallback has no matlab_compat-specific
    correction anywhere in the plan (quirk 2, deliberately left as-is): the
    `resp + T_all` denominator and the unweighted-pairwise fallback term are
    applied identically in both modes -- `sigma_denom` never becomes some
    masked, matlab_compat=False-only alternative.

    `mu` still differs between modes here (quirk 1's correction reaches every
    caller of `_fallback_univariate`, including the diagonal-of-the-pairwise
    path full covariance uses for its own mean) and `sigma` is recentered
    around that mu, so asserting the two modes' sigma values are literally
    equal would be mathematically wrong, not a faithful regression check.
    Instead this reproduces each mode's sigma directly from the documented
    formula (§2.4.2) using that mode's own mu and confirms it matches m_step's
    output bit-for-bit in *both* modes -- proving the formula itself (in
    particular the denominator) is unbranched, which is the actual quirk-2
    invariance the plan asks for."""
    import chmm_lsim.em as em_module

    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(2,), num_gmm_component=(1,))
    gmm = ChannelGMM(
        weights=np.array([[1.0]]), mu=np.zeros((2, 1, 1)), sigma=(np.eye(2) * 1e4)[:, :, None, None], diag=False
    )
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])

    T = 3  # <= _LOW_MASS_STATE_COUNT: fallback triggers via clause 1, regardless of masking
    Y = np.array([[10.0, 20.0, np.nan], [100.0, 200.0, np.nan]])  # real spread, not a single repeated point

    fbc = FbConstants(variant="sed")
    stats = finalize_expected_stats(em_module.e_step(structure, params, [Y], fbc, validate="off"), sigma_diag=False)
    cs = stats.channels[0]
    assert bool(cs.fallback_mu[0, 0])
    assert bool(cs.fallback_sigma_full[0, 0])

    for matlab_compat in (True, False):
        out = em_module.m_step(structure, stats, sigma_diag=False, matlab_compat=matlab_compat)
        mu_ik = out.emissions[0].mu[:, 0, 0]

        Q, M, C = cs.pairwise_Q[0, 0], cs.pairwise_M[0, 0], cs.pairwise_C[0, 0]
        delta_i, delta_j = M - mu_ik[:, None], M.T - mu_ik[None, :]
        numerator = C + Q * delta_i * delta_j
        u_delta_i, u_delta_j = cs.u_pairwise_M - mu_ik[:, None], cs.u_pairwise_M.T - mu_ik[None, :]
        numerator = numerator + cs.u_pairwise_C + cs.u_pairwise_Q * u_delta_i * u_delta_j
        sigma_denom = cs.resp[0, 0] + stats.T_all  # unmasked, matlab_compat-invariant by design
        expected_sigma = np.diag(0.001 * cs.nanvar_y) + numerator / sigma_denom

        np.testing.assert_allclose(out.emissions[0].sigma[:, :, 0, 0], expected_sigma, rtol=1e-12)

    params_true = em_module.m_step(structure, stats, sigma_diag=False, matlab_compat=True)
    params_false = em_module.m_step(structure, stats, sigma_diag=False, matlab_compat=False)
    assert not np.allclose(params_true.emissions[0].mu, params_false.emissions[0].mu)


# --------------------------------------------------------------------------
# matlab_compat threading into the eigenvalue-repair threshold (§0.4.10):
# MATLAB's own `sum(eig<=0)>1` under matlab_compat=True; corrected `>0` under False.
# --------------------------------------------------------------------------

def test_eigenvalue_repair_threshold_follows_matlab_compat():
    """`repair_covariance` in isolation, on a matrix with a *known* eigenvalue
    count -- constructing an M-step scenario that lands exactly on the
    boundary (1 vs 2 non-positive eigenvalues) is numerically fragile (the
    unconditional ridge tends to push a rank-deficient sample PD again), so
    this tests the repair rule directly rather than reverse-engineering data."""
    from chmm_lsim.em import repair_covariance

    ridge = np.array([1e-6, 1e-6])

    one_nonpositive = np.diag([2.0, 0.0])  # exactly one non-positive eigenvalue (0.0)
    repaired_true = repair_covariance(one_nonpositive, ridge, matlab_compat=True)
    repaired_false = repair_covariance(one_nonpositive, ridge, matlab_compat=False)
    np.testing.assert_array_equal(repaired_true, one_nonpositive)  # MATLAB's >1 threshold: untouched
    assert np.all(np.linalg.eigvalsh(repaired_false) > 0)  # corrected >0 threshold: repaired

    two_nonpositive = np.diag([0.0, -1.0])  # two non-positive eigenvalues
    repaired_true_2 = repair_covariance(two_nonpositive, ridge, matlab_compat=True)
    repaired_false_2 = repair_covariance(two_nonpositive, ridge, matlab_compat=False)
    assert np.all(np.linalg.eigvalsh(repaired_true_2) > 0)  # >=2 non-positive: MATLAB repairs too
    assert np.all(np.linalg.eigvalsh(repaired_false_2) > 0)


# --------------------------------------------------------------------------
# Full covariance: shape/sanity only (no MATLAB fixture, see module docstring)
# --------------------------------------------------------------------------

def test_full_covariance_m_step_runs_and_produces_valid_params():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    gmm = ChannelGMM(
        weights=np.array([[1.0], [1.0]]),
        mu=np.zeros((2, 2, 1)),
        sigma=np.stack([np.eye(2), np.eye(2)], axis=-1)[:, :, :, None],
        diag=False,
    )
    params = LSIMParams(
        structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.array([[0.6, 0.4], [0.3, 0.7]]), coupling=np.ones((1, 1)), emissions=[gmm]
    )
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(2, 20))
    fbc = FbConstants(variant="sed")

    stats = finalize_expected_stats(e_step(structure, params, [Y], fbc, validate="off"), sigma_diag=False)
    new_params = m_step(structure, stats, sigma_diag=False)
    new_params.validate()  # shape + stochasticity + PD checks must all pass

    assert new_params.sigma_diag is False
    assert new_params.emissions[0].sigma.shape == (2, 2, 2, 1)


# --------------------------------------------------------------------------
# P3b: e_step/em_lsim dispatch on fb_constants.variant (inference/registry.py)
# rather than always calling forward_backward_sed -- FbConstants(variant="fast")
# must actually run the fast kernel end to end, not silently fall back to sed.
# --------------------------------------------------------------------------

def test_e_step_dispatches_to_fast_variant(fixtures_dir):
    from chmm_lsim.inference.fast import forward_backward_fast

    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__fast")
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    params = LSIMParams.from_packed(
        structure,
        pi_0=fx.pi_0,
        transition=fx.transitions_matrices,
        coupling=fx.coupling_tetha_IM,
        P_all=fx.P_all,
        mu_all=fx.mu_all,
        sigma_all=fx.sigma_all,
        sigma_diag=bool(fx.sigma_diag),
    )
    Y = fx.trials[0].Y
    fbc_fast = FbConstants(variant="fast")

    stats_fast = finalize_expected_stats(
        e_step(structure, params, [Y], fbc_fast, validate="off"), sigma_diag=bool(fx.sigma_diag)
    )
    post_fast = forward_backward_fast(structure, params, Y, fbc_fast, validate="off")
    np.testing.assert_allclose(stats_fast.gamma_first, post_fast.gamma[:, 0])
    np.testing.assert_allclose(stats_fast.gamma_sum, post_fast.gamma.sum(axis=1))
    np.testing.assert_allclose(stats_fast.loglik, post_fast.loglik)

    # and genuinely not what sed's own e_step call would have produced on the
    # same params/Y -- proves this dispatched to fast rather than a no-op
    fbc_sed = FbConstants(variant="sed")
    stats_sed = finalize_expected_stats(
        e_step(structure, params, [Y], fbc_sed, validate="off"), sigma_diag=bool(fx.sigma_diag)
    )
    assert not np.allclose(stats_fast.gamma_sum, stats_sed.gamma_sum)
    assert stats_fast.loglik == stats_sed.loglik  # forward pass (and therefore loglik) is shared


def test_em_lsim_runs_through_fast_kernel(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__fast")
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    params0 = LSIMParams.from_packed(
        structure,
        pi_0=fx.pi_0,
        transition=fx.transitions_matrices,
        coupling=fx.coupling_tetha_IM,
        P_all=fx.P_all,
        mu_all=fx.mu_all,
        sigma_all=fx.sigma_all,
        sigma_diag=bool(fx.sigma_diag),
    )
    Y = fx.trials[0].Y

    result_fast = em_lsim(
        structure, params0.copy(), [Y], max_iterations=2, fb_constants=FbConstants(variant="fast"), check_convergence=False, validate="off"
    )
    result_sed = em_lsim(
        structure, params0.copy(), [Y], max_iterations=2, fb_constants=FbConstants(variant="sed"), check_convergence=False, validate="off"
    )

    # iteration 1's E-step runs at the same (unmodified) initial params for both
    # variants, and loglik depends only on the shared forward pass -- identical.
    assert result_fast.loglik_trace[0] == result_sed.loglik_trace[0]
    # the M-step update between iterations 1 and 2 uses gamma, which differs
    # between fast's and sed's backward message -- by iteration 2 the params
    # (and therefore the loglik) have diverged, proving fast really ran its
    # own kernel rather than silently reusing sed's.
    assert result_fast.loglik_trace[1] != result_sed.loglik_trace[1]
