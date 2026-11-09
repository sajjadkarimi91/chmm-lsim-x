"""P10d gate: coherent end-of-epoch re-score, convergence, best-epoch
selection, coherent `pi_steady`, and callbacks/progress
(`em.em_lsim_minibatch_train`, docs/PYTHON_PORT_PLAN.md section 2.6's own
"Epoch metrics" paragraph) -- built strictly ON TOP of the already-CLOSED
`em_lsim_minibatch` (P10c), `e_step` (P10b), and `em_lsim` (P4): none of
those three are modified by this phase.

Verified here:

1. **Golden-fixture ground truth (the strongest check)**: replaying all six
   P10a scenarios' own exact schedules end-to-end, `loglik_trace`/`aic_
   trace`/`bic_trace` match MATLAB's own independently-computed `epoch_
   coherent_loglik`/derived AIC/BIC exactly; an independently-rederived
   best-epoch selection (applying `em_lsim`'s own exact rule to that SAME
   coherent trace, entirely in the test, not reusing any driver code) is
   compared against `result.best_epoch`; `result.pi_steady` matches
   MATLAB's own `epoch_coherent_pi_steady` at that epoch. Confirms P10d's
   own new coherent-evaluation code is not merely self-consistent but
   agrees with an independent, MATLAB-derived source of truth -- something
   MATLAB itself never actually computes a best-epoch/convergence decision
   from (`em_lsim_par.m`'s own `AIC_trace`/`BIC_trace`/`returned_best_
   epoch_idx` are ALL based on the MIXED pre-update diagnostic, matching
   `MinibatchEmResult.training_batch_loglik_by_epoch`, not the coherent
   metric P10d introduces -- so this is a genuine Python-side improvement,
   not a MATLAB-parity claim, and is tested as such).
2. Convergence and best-epoch selection reproduce `em_lsim`'s own exact
   rule at epoch granularity, via the same monkeypatch methodology
   `tests/test_em.py` established for `em_lsim` itself (a controlled
   coherent-loglik sequence, `_run_one_minibatch_update` itself mocked out
   so the driver logic is isolated from the full scheduling/E-step/M-step
   pipeline -- `em_lsim_minibatch_train` calls this private helper
   directly, once per mini-batch, rather than delegating to the public
   `em_lsim_minibatch` §P10c/P10d correction round).
3. The `num_minibatches=1` relationship to `em_lsim`'s own trace: P10d's
   own `loglik_trace[i]` (epoch `i+1`'s own POST-update coherent
   evaluation, per §2.6's explicit "under that ONE FINAL parameter
   snapshot" language) is NOT the same quantity as `em_lsim.loglik_trace[i]`
   (iteration `i+1`'s own PRE-update evaluation) -- they are off by exactly
   one index, both being well-defined, intentionally different
   conventions. Confirmed directly, not assumed.
4. `epoch_hook`/`minibatch_hook` callbacks: correct arguments, correct
   call count and order, `minibatch_hook`'s own `epoch_idx` correctly
   renumbered to the TRUE global epoch (not always `1`, which is what the
   underlying single-epoch `em_lsim_minibatch` calls themselves report).
5. Torch coherent-loglik parity (genuinely forward-only `mode="loglik"`)
   against NumPy's own full-E-step-derived value.
6. Schedule/argument validation is inherited correctly (empty `Y_trials`,
   `schedule=` combined with generation-only controls).
7. Zero-transition mini-batches interact correctly with the coherent layer
   (no NaN/Inf anywhere in the resulting trace).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import P10A_SCENARIO_IDS, load_p10a_scheduling_fixture  # noqa: E402

from chmm_lsim.compat import FbConstants
from chmm_lsim.em import (
    MinibatchEmResult,
    MinibatchSchedule,
    MinibatchUpdate,
    compute_aic_bic,
    em_lsim,
    em_lsim_minibatch_train,
)
from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.structure import LSIMStructure


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
        sigma = np.ones((d_c, M_c, K_c))
        emissions.append(ChannelGMM(weights=weights, mu=mu, sigma=sigma, diag=sigma_diag))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    return structure, params


def _toy_trials(rng, structure, lengths):
    return [rng.normal(size=(structure.D, T)) for T in lengths]


def _schedule_from_p10a_fixture(fx) -> MinibatchSchedule:
    epochs = []
    for itr in range(fx.max_itration):
        mbs = []
        for mb in range(fx.mini_batch):
            u = fx.update_index(itr + 1, mb + 1)
            n_tr = int(fx.updates_n_trials[u])
            idx0 = tuple(int(i) - 1 for i in fx.updates_trial_indices[u, :n_tr])
            mbs.append(idx0)
        epochs.append(tuple(mbs))
    return MinibatchSchedule(epochs=tuple(epochs))


def _rederive_best_epoch(loglik_trace: list[float]) -> int:
    """em_lsim's/em_lsim_par.m's own exact (and deliberately unusual)
    selection rule, applied here entirely independently of `em_lsim_
    minibatch_train`'s own implementation: iteration/epoch 2 unconditionally
    overwrites epoch 1's snapshot; only epoch 3+ requires strict
    improvement over the running max."""
    best_idx = 1
    max_ll = loglik_trace[0]
    for itr in range(2, len(loglik_trace) + 1):
        ll = loglik_trace[itr - 1]
        if itr > 2:
            if max_ll < ll:
                max_ll = ll
                best_idx = itr
        else:
            max_ll = ll
            best_idx = itr
    return best_idx


# ==========================================================================
# 1. Golden-fixture ground truth: all six P10a scenarios
# ==========================================================================


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_coherent_trace_matches_p10a_epoch_coherent_loglik(fixtures_dir, scenario_id):
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()), channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    sigma_diag = bool(fx.sigma_diag)
    fbc = FbConstants(variant=fx.variant)
    Y_trials = fx.Y_trials()
    schedule = _schedule_from_p10a_fixture(fx)
    params0 = LSIMParams.from_packed(
        structure, pi_0=fx.updates_pi_0_in[0], transition=fx.updates_transition_in[0], coupling=fx.updates_coupling_in[0],
        P_all=fx.updates_P_all_in[0], mu_all=fx.updates_mu_all_in[0], sigma_all=fx.updates_sigma_all_in[0], sigma_diag=sigma_diag,
    )

    result = em_lsim_minibatch_train(
        structure, params0, Y_trials, fb_constants=fbc, schedule=schedule, backend="numpy", validate="off",
        check_convergence=False,
    )

    expected_coherent_loglik = fx.epoch_coherent_loglik.sum(axis=1)
    np.testing.assert_allclose(result.loglik_trace, expected_coherent_loglik, rtol=1e-9, atol=1e-9)

    T_all = int(fx.length_observation.sum())
    expected_aic, expected_bic = [], []
    for itr in range(fx.max_itration):
        a, b = compute_aic_bic(structure, sigma_diag, float(expected_coherent_loglik[itr]), T_all)
        expected_aic.append(a)
        expected_bic.append(b)
    np.testing.assert_allclose(result.aic_trace, expected_aic, rtol=1e-9)
    np.testing.assert_allclose(result.bic_trace, expected_bic, rtol=1e-9)

    rederived_best = _rederive_best_epoch(list(expected_coherent_loglik))
    assert result.best_epoch == rederived_best

    np.testing.assert_allclose(
        result.pi_steady, fx.epoch_coherent_pi_steady[result.best_epoch - 1], rtol=1e-6, atol=1e-8
    )
    assert result.n_epochs == fx.max_itration
    assert len(result.minibatch.updates) == fx.n_updates


@pytest.mark.parametrize("scenario_id", ["diag", "uneven", "fast"])
def test_coherent_trace_matches_p10a_with_convergence_enabled(fixtures_dir, scenario_id):
    """Same as above but with check_convergence=True (the default) -- these
    short (3-epoch) P10a fixtures never actually trigger convergence
    (itr>9 is never reached), so the result must be identical either way;
    this just confirms enabling convergence checking doesn't change
    anything for a run too short to ever trip it."""
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    assert fx.max_itration < 10
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()), channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    sigma_diag = bool(fx.sigma_diag)
    fbc = FbConstants(variant=fx.variant)
    Y_trials = fx.Y_trials()
    schedule = _schedule_from_p10a_fixture(fx)
    params0 = LSIMParams.from_packed(
        structure, pi_0=fx.updates_pi_0_in[0], transition=fx.updates_transition_in[0], coupling=fx.updates_coupling_in[0],
        P_all=fx.updates_P_all_in[0], mu_all=fx.updates_mu_all_in[0], sigma_all=fx.updates_sigma_all_in[0], sigma_diag=sigma_diag,
    )

    result = em_lsim_minibatch_train(
        structure, params0, Y_trials, fb_constants=fbc, schedule=schedule, backend="numpy", validate="off",
        check_convergence=True,
    )
    assert result.n_epochs == fx.max_itration
    np.testing.assert_allclose(result.loglik_trace, fx.epoch_coherent_loglik.sum(axis=1), rtol=1e-9, atol=1e-9)


# ==========================================================================
# 2. num_minibatches=1's own exact relationship to em_lsim's trace
# ==========================================================================


def test_num_minibatches_1_trace_is_em_lsim_trace_shifted_by_one():
    """P10d's own loglik_trace[i] (epoch i+1's POST-update coherent
    evaluation) equals em_lsim's own loglik_trace[i+1] (iteration i+2's
    PRE-update evaluation) -- NOT loglik_trace[i] itself. Both are well-
    defined, deliberately different conventions (em_lsim's is "loglik
    BEFORE this iteration's own M-step"; P10d's is "loglik AFTER this
    epoch's own final mini-batch commit", per section 2.6's explicit "under
    that ONE FINAL parameter snapshot" language) -- confirmed directly,
    not assumed, since the two are easy to conflate."""
    rng = np.random.default_rng(70)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])
    fbc = FbConstants(variant="sed")
    max_iterations = 6

    result = em_lsim_minibatch_train(
        structure, params.copy(), Y_trials, fb_constants=fbc, max_iterations=max_iterations, num_minibatches=1,
        shuffle=False, backend="numpy", n_jobs=1, validate="off", check_convergence=False,
    )
    ref = em_lsim(
        structure, params.copy(), Y_trials, max_iterations=max_iterations, fb_constants=fbc, backend="numpy",
        check_convergence=False, validate="off",
    )

    np.testing.assert_allclose(result.loglik_trace[:-1], ref.loglik_trace[1:], rtol=0, atol=0)
    np.testing.assert_allclose(result.aic_trace[:-1], ref.aic_trace[1:], rtol=0, atol=0)
    np.testing.assert_allclose(result.bic_trace[:-1], ref.bic_trace[1:], rtol=0, atol=0)


# ==========================================================================
# 3. Convergence and best-epoch selection: exact em_lsim rule, at epoch
#    granularity, via the same monkeypatch methodology as tests/test_em.py
# ==========================================================================


def _make_tiny_structure_and_params():
    structure = LSIMStructure((2,), (1,), (1,))
    gmm = ChannelGMM(weights=np.array([[1.0], [1.0]]), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 2, 1)), diag=True)
    return structure, LSIMParams(
        structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.array([[0.5, 0.5], [0.5, 0.5]]), coupling=np.ones((1, 1)), emissions=[gmm]
    )


def test_best_epoch_is_argmax_over_epochs_2_to_n(monkeypatch):
    """Epoch 1 is never the best-epoch even if it has the highest coherent
    loglik -- the exact same rule em_lsim.m/em_lsim itself follows,
    verified here at epoch granularity via a controlled loglik sequence."""
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_structure_and_params()
    loglik_seq = [100.0, 1.0, 2.0, 3.0]  # epoch 1 is the largest value, must still lose
    calls = {"n": 0}

    def fake_run_one_minibatch_update(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        p = params_in.copy()
        p.pi_0 = np.array([float(epoch_idx), 0.0])  # tag with the 1-based epoch about to be evaluated
        return MinibatchUpdate(
            epoch_idx=epoch_idx, minibatch_idx=minibatch_idx, trial_indices=trial_indices, params_in=params_in,
            params_out=p, n_trials=1, T_all=1, n_transitions=0, training_batch_loglik=0.0,
        )

    def fake_coherent_epoch_loglik(structure, params, Y_trials, fb_constants, **kwargs):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return ll

    def fake_coherent_pi_steady(structure, params, Y_trials, fb_constants, **kwargs):
        return np.zeros(structure.S)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", fake_run_one_minibatch_update)
    monkeypatch.setattr(em_module, "_coherent_epoch_loglik", fake_coherent_epoch_loglik)
    monkeypatch.setattr(em_module, "_coherent_pi_steady", fake_coherent_pi_steady)

    result = em_module.em_lsim_minibatch_train(
        structure, params, [np.zeros((1, 1))], max_iterations=4, num_minibatches=1, shuffle=False,
        check_convergence=False,
    )
    assert result.loglik_trace == loglik_seq
    # candidates are epochs {2,3,4} with values [1.0, 2.0, 3.0] -- epoch 1's
    # loglik of 100.0 is never a candidate, so the winner is epoch 4
    assert result.best_epoch == 4
    assert result.params.pi_0[0] == 4.0  # tagged with the winning epoch's own identity


def test_convergence_breaks_after_five_flat_diffs(monkeypatch):
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_structure_and_params()
    loglik_seq = [float(i) for i in range(9)] + [8.0005] * 5
    calls = {"n": 0}

    def fake_run_one_minibatch_update(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        return MinibatchUpdate(
            epoch_idx=epoch_idx, minibatch_idx=minibatch_idx, trial_indices=trial_indices, params_in=params_in,
            params_out=params_in.copy(), n_trials=1, T_all=1, n_transitions=0, training_batch_loglik=0.0,
        )

    def fake_coherent_epoch_loglik(structure, params, Y_trials, fb_constants, **kwargs):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return ll

    def fake_coherent_pi_steady(structure, params, Y_trials, fb_constants, **kwargs):
        return np.zeros(structure.S)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", fake_run_one_minibatch_update)
    monkeypatch.setattr(em_module, "_coherent_epoch_loglik", fake_coherent_epoch_loglik)
    monkeypatch.setattr(em_module, "_coherent_pi_steady", fake_coherent_pi_steady)

    result = em_module.em_lsim_minibatch_train(
        structure, params, [np.zeros((1, 1))], max_iterations=len(loglik_seq), num_minibatches=1, shuffle=False,
        check_convergence=True,
    )
    assert result.n_epochs < len(loglik_seq)


def test_convergence_break_excludes_its_own_epoch_from_best_epoch(monkeypatch):
    """Regression case mirroring test_em.py's own analogous test: convergence
    is checked BEFORE the best-epoch snapshot, so a break on the epoch that
    triggers convergence must exclude that epoch's own snapshot from best-
    epoch consideration, even if its own coherent loglik is higher than the
    previous epoch's."""
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_structure_and_params()
    loglik_seq = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0004, 8.0005]
    calls = {"n": 0}

    def fake_run_one_minibatch_update(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        p = params_in.copy()
        p.pi_0 = np.array([float(epoch_idx), 0.0])
        return MinibatchUpdate(
            epoch_idx=epoch_idx, minibatch_idx=minibatch_idx, trial_indices=trial_indices, params_in=params_in,
            params_out=p, n_trials=1, T_all=1, n_transitions=0, training_batch_loglik=0.0,
        )

    def fake_coherent_epoch_loglik(structure, params, Y_trials, fb_constants, **kwargs):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return ll

    def fake_coherent_pi_steady(structure, params, Y_trials, fb_constants, **kwargs):
        return np.zeros(structure.S)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", fake_run_one_minibatch_update)
    monkeypatch.setattr(em_module, "_coherent_epoch_loglik", fake_coherent_epoch_loglik)
    monkeypatch.setattr(em_module, "_coherent_pi_steady", fake_coherent_pi_steady)

    result = em_module.em_lsim_minibatch_train(
        structure, params, [np.zeros((1, 1))], max_iterations=len(loglik_seq), num_minibatches=1, shuffle=False,
        check_convergence=True,
    )
    assert result.n_epochs == 11
    assert result.best_epoch == 10  # NOT 11, even though epoch 11's loglik (8.0005) > epoch 10's (8.0004)
    assert result.params.pi_0[0] == 10.0


# ==========================================================================
# 4. Callbacks/progress
# ==========================================================================


def test_epoch_hook_called_once_per_epoch_with_correct_arguments():
    rng = np.random.default_rng(71)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    fbc = FbConstants(variant="sed")

    calls = []
    result = em_lsim_minibatch_train(
        structure, params, Y_trials, fb_constants=fbc, max_iterations=3, num_minibatches=2, shuffle=False,
        backend="numpy", validate="off", check_convergence=False,
        epoch_hook=lambda epoch_idx, p, loglik, aic, bic: calls.append((epoch_idx, p, loglik, aic, bic)),
    )
    assert [c[0] for c in calls] == [1, 2, 3]
    for i, (epoch_idx, p, loglik, aic, bic) in enumerate(calls):
        assert loglik == result.loglik_trace[i]
        assert aic == result.aic_trace[i]
        assert bic == result.bic_trace[i]
        assert isinstance(p, LSIMParams)


def test_minibatch_hook_called_once_per_minibatch_in_order_with_global_epoch_index():
    rng = np.random.default_rng(72)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])
    fbc = FbConstants(variant="sed")

    calls = []
    result = em_lsim_minibatch_train(
        structure, params, Y_trials, fb_constants=fbc, max_iterations=3, num_minibatches=2, shuffle=True,
        random_state=0, backend="numpy", validate="off", check_convergence=False,
        minibatch_hook=lambda epoch_idx, minibatch_idx, update: calls.append((epoch_idx, minibatch_idx, update)),
    )
    assert len(calls) == len(result.minibatch.updates) == 3 * 2
    # every (epoch_idx, minibatch_idx) pair appears exactly once, epoch_idx
    # spans 1..3 (NOT always 1, which is what em_lsim_minibatch itself would
    # report for a one-epoch schedule with no visibility into the true epoch)
    seen = [(c[0], c[1]) for c in calls]
    assert seen == [(e, mb) for e in (1, 2, 3) for mb in (1, 2)]
    for (epoch_idx, minibatch_idx, update), recorded in zip(calls, result.minibatch.updates):
        assert update is recorded
        assert update.epoch_idx == epoch_idx
        assert update.minibatch_idx == minibatch_idx


def test_convergence_break_stops_remaining_epoch_and_minibatch_hooks(monkeypatch):
    """When convergence fires at epoch e, no hook (epoch or minibatch) for
    any LATER epoch may fire -- those epochs never run at all."""
    import chmm_lsim.em as em_module

    structure, params = _make_tiny_structure_and_params()
    loglik_seq = [float(i) for i in range(9)] + [8.0005] * 5
    calls = {"n": 0}

    def fake_run_one_minibatch_update(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        p = params_in.copy()
        return MinibatchUpdate(
            epoch_idx=epoch_idx, minibatch_idx=minibatch_idx, trial_indices=trial_indices, params_in=params_in,
            params_out=p, n_trials=1, T_all=1, n_transitions=0, training_batch_loglik=0.0,
        )

    def fake_coherent_epoch_loglik(structure, params, Y_trials, fb_constants, **kwargs):
        ll = loglik_seq[calls["n"]]
        calls["n"] += 1
        return ll

    def fake_coherent_pi_steady(structure, params, Y_trials, fb_constants, **kwargs):
        return np.zeros(structure.S)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", fake_run_one_minibatch_update)
    monkeypatch.setattr(em_module, "_coherent_epoch_loglik", fake_coherent_epoch_loglik)
    monkeypatch.setattr(em_module, "_coherent_pi_steady", fake_coherent_pi_steady)

    epoch_calls, minibatch_calls = [], []
    result = em_module.em_lsim_minibatch_train(
        structure, params, [np.zeros((1, 1))], max_iterations=len(loglik_seq), num_minibatches=1, shuffle=False,
        check_convergence=True,
        epoch_hook=lambda epoch_idx, p, loglik, aic, bic: epoch_calls.append(epoch_idx),
        minibatch_hook=lambda epoch_idx, minibatch_idx, update: minibatch_calls.append(epoch_idx),
    )
    assert result.n_epochs == 11
    assert epoch_calls == list(range(1, 12))
    assert minibatch_calls == list(range(1, 12))


# ==========================================================================
# 4b. compute_global_nanvar runs exactly ONCE per training run (§P10c/P10d
#     correction round): the original bug delegated each epoch to
#     em_lsim_minibatch, which itself calls compute_global_nanvar once per
#     call -- max_iterations whole-dataset variance scans instead of one.
# ==========================================================================


def test_compute_global_nanvar_called_exactly_once_for_multi_epoch_training(monkeypatch):
    import chmm_lsim.em as em_module

    rng = np.random.default_rng(80)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    fbc = FbConstants(variant="sed")

    real_compute_global_nanvar = em_module.compute_global_nanvar
    calls = {"n": 0}

    def counting_compute_global_nanvar(structure, Y_trials):
        calls["n"] += 1
        return real_compute_global_nanvar(structure, Y_trials)

    monkeypatch.setattr(em_module, "compute_global_nanvar", counting_compute_global_nanvar)

    em_module.em_lsim_minibatch_train(
        structure, params, Y_trials, fb_constants=fbc, max_iterations=5, num_minibatches=2, shuffle=False,
        backend="numpy", validate="off", check_convergence=False,
    )
    assert calls["n"] == 1


# ==========================================================================
# 4c. minibatch_hook is a genuinely LIVE progress callback (§P10c/P10d
#     correction round): fires immediately after each mini-batch commits,
#     before the next mini-batch's own E-step begins -- not only after the
#     whole epoch containing it has finished.
# ==========================================================================


def test_minibatch_hook_fires_before_next_minibatch_update_begins(monkeypatch):
    import chmm_lsim.em as em_module

    rng = np.random.default_rng(90)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 10, 10, 10])
    schedule = MinibatchSchedule(epochs=(((0,), (1,), (2,), (3,)),))

    real_run_one = em_module._run_one_minibatch_update
    events = []

    def tracking_run_one(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        events.append(("update_start", minibatch_idx))
        return real_run_one(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", tracking_run_one)

    def hook(epoch_idx, minibatch_idx, update):
        events.append(("hook", minibatch_idx))

    em_module.em_lsim_minibatch_train(
        structure, params, Y_trials, schedule=schedule, backend="numpy", validate="off",
        check_convergence=False, minibatch_hook=hook,
    )

    update_positions = [i for i, e in enumerate(events) if e[0] == "update_start"]
    hook_positions = [i for i, e in enumerate(events) if e[0] == "hook"]
    assert len(update_positions) == len(hook_positions) == 4
    # each hook fires immediately after its own update, nothing else interleaved
    for u_pos, h_pos in zip(update_positions, hook_positions):
        assert h_pos == u_pos + 1
    # hook u fires strictly before update u+1's own E-step begins
    for i in range(3):
        assert hook_positions[i] < update_positions[i + 1]


def test_hooks_for_earlier_minibatches_already_delivered_when_a_later_one_fails(monkeypatch):
    import chmm_lsim.em as em_module

    rng = np.random.default_rng(91)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 10, 10])
    schedule = MinibatchSchedule(epochs=(((0,), (1,), (2,)),))

    real_run_one = em_module._run_one_minibatch_update
    calls = {"n": 0}

    def failing_on_third(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated mini-batch 3 failure")
        return real_run_one(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", failing_on_third)

    delivered = []
    with pytest.raises(RuntimeError, match="simulated mini-batch 3 failure"):
        em_module.em_lsim_minibatch_train(
            structure, params, Y_trials, schedule=schedule, backend="numpy", validate="off",
            check_convergence=False, minibatch_hook=lambda e, mb, u: delivered.append((e, mb)),
        )
    # mini-batches 1 and 2 already committed and had their own hook delivered
    # before mini-batch 3 ever started (let alone failed)
    assert delivered == [(1, 1), (1, 2)]


def test_minibatch_hook_exception_stops_subsequent_minibatches(monkeypatch):
    import chmm_lsim.em as em_module

    rng = np.random.default_rng(92)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 10, 10])
    schedule = MinibatchSchedule(epochs=(((0,), (1,), (2,)),))

    real_run_one = em_module._run_one_minibatch_update
    ran = []

    def tracking_run_one(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs):
        ran.append(minibatch_idx)
        return real_run_one(structure, params_in, Y_trials, trial_indices, epoch_idx, minibatch_idx, fb_constants, global_nanvar, **kwargs)

    monkeypatch.setattr(em_module, "_run_one_minibatch_update", tracking_run_one)

    def hook(epoch_idx, minibatch_idx, update):
        if minibatch_idx == 2:
            raise RuntimeError("hook boom")

    with pytest.raises(RuntimeError, match="hook boom"):
        em_module.em_lsim_minibatch_train(
            structure, params, Y_trials, schedule=schedule, backend="numpy", validate="off",
            check_convergence=False, minibatch_hook=hook,
        )
    assert ran == [1, 2]  # mini-batch 3 never ran -- the hook's own exception stopped it


# ==========================================================================
# 5. Torch coherent-loglik parity (genuinely forward-only mode="loglik")
# ==========================================================================


def test_torch_coherent_loglik_matches_numpy():
    pytest.importorskip("torch")
    rng = np.random.default_rng(73)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    fbc = FbConstants(variant="sed")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numpy_result = em_lsim_minibatch_train(
            structure, params, Y_trials, fb_constants=fbc, max_iterations=3, num_minibatches=2, shuffle=True,
            random_state=0, backend="numpy", validate="off", check_convergence=False,
        )
        torch_result = em_lsim_minibatch_train(
            structure, params, Y_trials, fb_constants=fbc, max_iterations=3, num_minibatches=2, shuffle=True,
            random_state=0, backend="torch", validate="off", check_convergence=False,
        )

    np.testing.assert_allclose(torch_result.loglik_trace, numpy_result.loglik_trace, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(torch_result.aic_trace, numpy_result.aic_trace, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(torch_result.bic_trace, numpy_result.bic_trace, rtol=1e-6, atol=1e-6)
    assert torch_result.best_epoch == numpy_result.best_epoch
    np.testing.assert_allclose(torch_result.pi_steady, numpy_result.pi_steady, rtol=1e-6, atol=1e-8)


def test_torch_rejects_outer_executor():
    pytest.importorskip("torch")
    from chmm_lsim.backends.base import UnsupportedCombination

    rng = np.random.default_rng(74)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8])
    with pytest.raises(UnsupportedCombination, match="torch"):
        em_lsim_minibatch_train(
            structure, params, Y_trials, max_iterations=1, num_minibatches=1, backend="torch", n_jobs=2,
            executor="thread", validate="off",
        )


# ==========================================================================
# 6. Validation contract inherited correctly
# ==========================================================================


def test_empty_Y_trials_rejected_cleanly():
    rng = np.random.default_rng(75)
    structure, _ = _toy_model(rng)
    _, params = _toy_model(rng)
    with pytest.raises(ValueError, match="at least one trial"):
        em_lsim_minibatch_train(structure, params, [], max_iterations=1, num_minibatches=1, backend="numpy", validate="off")


def test_schedule_with_generation_controls_raises():
    rng = np.random.default_rng(76)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    sched = MinibatchSchedule(epochs=((tuple(range(4)),),))
    with pytest.raises(ValueError, match="num_minibatches"):
        em_lsim_minibatch_train(structure, params, Y_trials, schedule=sched, num_minibatches=2, backend="numpy", validate="off")


def test_no_schedule_and_no_max_iterations_raises():
    rng = np.random.default_rng(77)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    with pytest.raises(ValueError, match="max_iterations"):
        em_lsim_minibatch_train(structure, params, Y_trials, backend="numpy", validate="off")


# --------------------------------------------------------------------------
# 6b. P10d boundary tests (§P10c/P10d correction round): an earlier version
# of em_lsim_minibatch_train reimplemented P10c's own already-hardened
# schedule/observation validation and reintroduced closed bugs -- accessing
# schedule.epochs before confirming schedule is a MinibatchSchedule,
# max_iterations=True/1.0 silently comparing equal to a one-epoch schedule
# via a bare !=, and T_all computed from raw Y.shape[1] before any
# observation validation. These are planted here at the em_lsim_minibatch_
# train (driver) level specifically -- not merely re-testing validate_
# minibatch_schedule/compute_global_nanvar themselves, which tests/
# test_em_minibatch.py already covers -- to confirm the driver actually
# routes through those already-hardened primitives instead of its own,
# separately-maintained (and previously separately-broken) copy.
# --------------------------------------------------------------------------


def test_non_minibatchschedule_object_rejected_with_clear_error():
    rng = np.random.default_rng(81)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    with pytest.raises(ValueError, match="MinibatchSchedule"):
        em_lsim_minibatch_train(structure, params, Y_trials, schedule=[[[0, 1]]], backend="numpy", validate="off")


@pytest.mark.parametrize("bad_max_iterations", [True, 1.0])
def test_max_iterations_bool_or_float_rejected_even_when_it_equals_schedule_epoch_count(bad_max_iterations):
    """With a genuine one-epoch schedule, max_iterations=True or 1.0 both
    compare equal to the int 1 via a bare != check and would otherwise
    silently pass."""
    rng = np.random.default_rng(82)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    schedule = MinibatchSchedule(epochs=(((0, 1),),))
    with pytest.raises(ValueError, match="max_iterations"):
        em_lsim_minibatch_train(
            structure, params, Y_trials, schedule=schedule, max_iterations=bad_max_iterations, backend="numpy",
            validate="off",
        )


@pytest.mark.parametrize(
    "bad_Y, match",
    [
        (np.array(5.0), "2-D"),
        (np.zeros(10), "2-D"),
        (np.zeros((2, 10)), "structure.D"),  # too many channels (tiny structure has D=1)
        (np.zeros((0, 10)), "structure.D"),  # too few channels
        (np.zeros((1, 10), dtype=np.int64), "floating dtype"),
    ],
)
def test_em_lsim_minibatch_train_rejects_malformed_observations(bad_Y, match):
    structure, params = _make_tiny_structure_and_params()
    with pytest.raises(ValueError, match=match):
        em_lsim_minibatch_train(
            structure, params, [bad_Y], max_iterations=1, num_minibatches=1, backend="numpy", validate="off",
        )


def test_em_lsim_minibatch_train_rejects_inf_observation():
    structure, params = _make_tiny_structure_and_params()
    Y = np.zeros((1, 10))
    Y[0, 0] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        em_lsim_minibatch_train(structure, params, [Y], max_iterations=1, num_minibatches=1, backend="numpy", validate="off")


def test_em_lsim_minibatch_train_rejects_T_zero_observation():
    structure, params = _make_tiny_structure_and_params()
    with pytest.raises(ValueError, match="T=0"):
        em_lsim_minibatch_train(
            structure, params, [np.zeros((1, 0))], max_iterations=1, num_minibatches=1, backend="numpy", validate="off",
        )


def test_em_lsim_minibatch_train_rejects_malformed_second_trial_naming_its_index():
    """The FIRST trial is valid; the SECOND is malformed -- confirms
    compute_global_nanvar's own per-trial validation loop (routed through
    directly, not re-implemented) checks every trial, not just the first,
    and names the offending GLOBAL index."""
    structure, params = _make_tiny_structure_and_params()
    good = np.zeros((1, 10))
    bad = np.zeros((1, 10), dtype=np.int64)
    with pytest.raises(ValueError, match=r"Y_trials\[1\]"):
        em_lsim_minibatch_train(
            structure, params, [good, bad], max_iterations=1, num_minibatches=1, backend="numpy", validate="off",
        )


# ==========================================================================
# 7. Zero-transition mini-batches under the coherent layer
# ==========================================================================


def test_zero_transition_epoch_stays_finite_in_coherent_trace():
    """A mini-batch (and here, a whole epoch) drawn entirely from T=1
    trials must not introduce any NaN/Inf into the coherent loglik/AIC/BIC
    trace or pi_steady."""
    rng = np.random.default_rng(78)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [1, 1, 1, 1])
    schedule = MinibatchSchedule(epochs=((tuple(range(4)),),))

    result = em_lsim_minibatch_train(
        structure, params, Y_trials, schedule=schedule, backend="numpy", validate="off", check_convergence=False,
    )
    assert np.all(np.isfinite(result.loglik_trace))
    assert np.all(np.isfinite(result.aic_trace))
    assert np.all(np.isfinite(result.bic_trace))
    assert np.all(np.isfinite(result.pi_steady))
    np.testing.assert_array_equal(result.params.transition, params.transition)
    np.testing.assert_array_equal(result.params.coupling, params.coupling)


# ==========================================================================
# 8. Top-level package export contract
# ==========================================================================


def test_p10d_types_exported_from_top_level_package():
    import chmm_lsim
    from chmm_lsim.em import MinibatchTrainingResult as _MinibatchTrainingResult

    assert chmm_lsim.em_lsim_minibatch_train is em_lsim_minibatch_train
    assert "em_lsim_minibatch_train" in chmm_lsim.__all__
    assert chmm_lsim.MinibatchTrainingResult is _MinibatchTrainingResult
    assert "MinibatchTrainingResult" in chmm_lsim.__all__

    from chmm_lsim import em_lsim_minibatch_train as reimported

    assert reimported is em_lsim_minibatch_train
