"""P10e: mini-batch convergence policies, opt-in coherent-evaluation cadence,
and the pi_0 / pi_steady distinction.

Policy tests are deterministic and trace-based. Driver-level tests replace
the coherent scorer with a scripted trace where a specific stopping epoch is
needed, and otherwise compare real runs bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

import chmm_lsim.em as em_mod
from chmm_lsim import (
    ChannelGMM,
    FbConstants,
    LSIMParams,
    LSIMStructure,
    RobustConvergence,
    em_lsim_minibatch_train,
    load_lsim_params,
    save_lsim_params,
)
from chmm_lsim.em import _legacy_convergence_stop, _RobustConvergenceMonitor

# The real ctx1 M=5 coherent trace shape (s0full, epoch-to-epoch deltas): still
# improving on net, but epochs 12 and 14 decrease.
REAL_DELTAS = [11465.8e3, 2187.8e3, 652.7e3, 220.9e3, 117.1e3, 128.9e3, 43.6e3, 80.8e3, 52.3e3, 63.9e3, -25.3e3, 55.2e3, -31.0e3]


def _trace(start, deltas):
    return list(np.cumsum([start] + list(deltas)))


def _legacy_stop_epoch(trace):
    for e in range(1, len(trace) + 1):
        if _legacy_convergence_stop(trace[:e], e):
            return e
    return None


def _robust_stop_epoch(trace, cfg=RobustConvergence()):
    m = _RobustConvergenceMonitor(cfg)
    for e, v in enumerate(trace, start=1):
        if m.update(e, v):
            return e
    return None


# ---------------- policy unit tests ----------------

def test_legacy_rule_counts_large_decreases_as_convergence_on_the_real_trace_shape():
    trace = _trace(-5.4e7, REAL_DELTAS)
    assert _legacy_stop_epoch(trace) == 14  # what every real Python run did


def test_robust_rule_does_not_stop_on_the_real_trace_shape():
    trace = _trace(-5.4e7, REAL_DELTAS)
    assert _robust_stop_epoch(trace) is None


def test_robust_single_large_decrease_never_stops_training():
    trace = _trace(-1e7, [1e5] * 12 + [-5e6] + [1e5] * 7)
    assert _robust_stop_epoch(trace) is None


def test_robust_oscillation_with_net_improvement_never_stops():
    trace = _trace(-4e7, [6e4, -3e4] * 20)
    assert _robust_stop_epoch(trace) is None


def test_robust_plateau_stops_after_patience():
    cfg = RobustConvergence(rel_tol=1e-5, patience=3, min_epochs=5)
    # strong gains for 6 epochs, then relative gains of ~1e-8 (plateau)
    trace = _trace(-1e7, [1e5] * 6 + [0.1] * 10)
    assert _robust_stop_epoch(trace, cfg) == 10  # epochs 8, 9, 10 are the 3 non-improving evaluations


def test_robust_genuine_improvement_never_stops():
    trace = _trace(-1e7, [2e3] * 60)  # 2e-4 relative per epoch >= 1e-5
    assert _robust_stop_epoch(trace) is None


def test_robust_sustained_large_decline_is_not_convergence():
    cfg = RobustConvergence(rel_tol=1e-5, patience=3, min_epochs=1)
    trace = _trace(-1e7, [1e5, 1e5] + [-1e5] * 10)
    assert _robust_stop_epoch(trace, cfg) is None


def test_robust_recovery_after_a_drop_keeps_training():
    cfg = RobustConvergence(rel_tol=1e-5, patience=3, min_epochs=1)
    trace = _trace(-1e7, [1e5] * 3 + [-2e6] + [2e5] * 10)
    assert _robust_stop_epoch(trace, cfg) is None


def test_robust_tiny_oscillation_on_a_plateau_stops():
    cfg = RobustConvergence(rel_tol=1e-5, patience=3, min_epochs=1)
    trace = _trace(-1e7, [1e5] * 3 + [5.0, -5.0, 5.0, -5.0, 5.0])  # |delta| = 5e-7 relative
    assert _robust_stop_epoch(trace, cfg) == 7


def test_robust_zero_tolerance_never_stops():
    assert _robust_stop_epoch(_trace(-1e7, [0.0] * 30), RobustConvergence(rel_tol=0.0, patience=1, min_epochs=1)) is None


def test_robust_respects_min_epochs():
    cfg = RobustConvergence(rel_tol=1e-5, patience=2, min_epochs=12)
    trace = _trace(-1e7, [0.0] * 20)
    assert _robust_stop_epoch(trace, cfg) == 12


def test_legacy_rule_on_plateau_and_improvement_is_unchanged():
    assert _legacy_stop_epoch(_trace(-1e7, [0.0] * 20)) == 10
    assert _legacy_stop_epoch(_trace(-1e7, [1.0] * 30)) is None


@pytest.mark.parametrize("kw", [dict(rel_tol=-1.0), dict(rel_tol=float("nan")), dict(patience=0), dict(min_epochs=0),
                                dict(patience=True), dict(patience=2.5)])
def test_robust_settings_are_validated(kw):
    with pytest.raises(ValueError):
        RobustConvergence(**kw)


# ---------------- driver-level ----------------

def _problem(seed=0):
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 1))
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for z in range(structure.C):
        sl = structure.state_slice(z)
        transition[:, sl] = rng.dirichlet(np.ones(sl.stop - sl.start), size=structure.S)
    coupling = rng.dirichlet(np.ones(structure.C), size=structure.C).T
    emissions = [ChannelGMM(np.ones((m, 1)), rng.normal(size=(d, m, 1)), rng.uniform(0.5, 2, size=(d, m, 1)), diag=True)
                 for m, d in zip(structure.channel_state_num, structure.channel_obs_dim)]
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    Y = [rng.normal(size=(structure.D, T)) for T in (40, 25, 60, 33, 18, 50)]
    return structure, params, Y


_COMMON = dict(num_minibatches=3, shuffle=True, random_state=4, fb_constants=FbConstants(), backend="numpy")


def _same_updates(a, b):
    ua, ub = a.minibatch.updates, b.minibatch.updates
    assert len(ua) == len(ub)
    for x, y in zip(ua, ub):
        assert (x.epoch_idx, x.minibatch_idx, x.trial_indices) == (y.epoch_idx, y.minibatch_idx, y.trial_indices)
        assert x.training_batch_loglik == y.training_batch_loglik
        np.testing.assert_array_equal(x.params_out.transition, y.params_out.transition)
        np.testing.assert_array_equal(x.params_out.pi_0, y.params_out.pi_0)
        for g, h in zip(x.params_out.emissions, y.params_out.emissions):
            np.testing.assert_array_equal(g.mu, h.mu)
            np.testing.assert_array_equal(g.sigma, h.sigma)


def test_default_is_exactly_legacy_and_check_convergence_false_is_off():
    structure, params, Y = _problem()
    default = em_lsim_minibatch_train(structure, params, Y, max_iterations=12, **_COMMON)
    legacy = em_lsim_minibatch_train(structure, params, Y, max_iterations=12, convergence="legacy", **_COMMON)
    legacy2 = em_lsim_minibatch_train(structure, params, Y, max_iterations=12, check_convergence=True, **_COMMON)
    off = em_lsim_minibatch_train(structure, params, Y, max_iterations=12, check_convergence=False, **_COMMON)
    off2 = em_lsim_minibatch_train(structure, params, Y, max_iterations=12, convergence="off", **_COMMON)
    for r in (legacy, legacy2):
        assert r.loglik_trace == default.loglik_trace and r.best_epoch == default.best_epoch
    assert default.convergence == "legacy" and off.convergence == "off"
    assert off.loglik_trace == off2.loglik_trace and off.n_epochs == 12
    assert default.evaluated_epoch_indices == list(range(1, default.n_epochs + 1))


@pytest.mark.parametrize("kwargs,match", [
    (dict(check_convergence=False, convergence="legacy"), "contradicts"),
    (dict(check_convergence=True, convergence="off"), "contradicts"),
    (dict(convergence="sometimes"), "convergence must be one of"),
    (dict(check_convergence=1), "check_convergence must be a bool"),
    (dict(convergence="legacy", robust_convergence=RobustConvergence()), "robust_convergence settings were given"),
])
def test_contradictory_convergence_arguments_are_rejected(kwargs, match):
    structure, params, Y = _problem()
    with pytest.raises(ValueError, match=match):
        em_lsim_minibatch_train(structure, params, Y, max_iterations=2, **_COMMON, **kwargs)


def _scripted_scorer(monkeypatch, values):
    calls = []

    def scorer(structure, params, Y_trials, fb_constants, **kw):
        calls.append(len(calls))
        return values[len(calls) - 1]

    monkeypatch.setattr(em_mod, "_coherent_epoch_loglik", scorer)
    return calls


def test_policies_on_a_scripted_real_shaped_trace(monkeypatch):
    structure, params, Y = _problem()
    trace = _trace(-5.4e7, REAL_DELTAS + [40e3, -20e3, 35e3, 10e3, 5e3, 2e3])  # 20 epochs
    runs = {}
    for policy in ("legacy", "robust", "off"):
        _scripted_scorer(monkeypatch, trace)
        runs[policy] = em_lsim_minibatch_train(structure, params, Y, max_iterations=20, convergence=policy, **_COMMON)
    assert runs["legacy"].n_epochs == 14 and runs["legacy"].stop_reason == "converged"
    assert runs["robust"].n_epochs == 20 and runs["robust"].stop_reason == "max_iterations"
    assert runs["off"].n_epochs == 20
    assert runs["off"].best_epoch == int(np.argmax(trace)) + 1
    assert runs["legacy"].best_epoch == 13  # legacy breaks before snapshotting epoch 14


@pytest.mark.parametrize("interval,expected", [(1, list(range(1, 9))), (3, [3, 6, 8]), (5, [5, 8]), (8, [8]), ("final", [8])])
def test_cadence_evaluates_only_scheduled_epochs_with_identical_updates(monkeypatch, interval, expected):
    structure, params, Y = _problem(seed=1)
    every = em_lsim_minibatch_train(structure, params, Y, max_iterations=8, convergence="off", **_COMMON)

    real = em_mod._coherent_epoch_loglik
    scored_epochs = []
    hooked = []

    def counting(*a, **k):
        scored_epochs.append(len(scored_epochs))
        return real(*a, **k)

    monkeypatch.setattr(em_mod, "_coherent_epoch_loglik", counting)
    sparse = em_lsim_minibatch_train(structure, params, Y, max_iterations=8, convergence="off", coherent_eval_interval=interval,
                                     epoch_hook=lambda e, p, ll, aic, bic: hooked.append(e), **_COMMON)
    assert sparse.evaluated_epoch_indices == expected
    assert len(scored_epochs) == len(expected)  # skipped epochs are never scored, now or later
    assert hooked == expected
    assert len(sparse.loglik_trace) == len(sparse.aic_trace) == len(sparse.bic_trace) == len(expected)
    for e, ll in zip(expected, sparse.loglik_trace):
        assert ll == every.loglik_trace[e - 1]  # same snapshot, same score
    assert sparse.best_epoch in expected
    _same_updates(every, sparse)
    np.testing.assert_array_equal(sparse.minibatch.params.transition, every.minibatch.params.transition)
    assert sparse.n_epochs == 8 and sparse.coherent_eval_interval == interval


def test_cadence_best_epoch_is_best_among_evaluated_snapshots(monkeypatch):
    structure, params, Y = _problem(seed=2)
    _scripted_scorer(monkeypatch, [-10.0, -1.0, -5.0])  # evaluations at epochs 3, 6, 8
    r = em_lsim_minibatch_train(structure, params, Y, max_iterations=8, convergence="off", coherent_eval_interval=3, **_COMMON)
    assert r.evaluated_epoch_indices == [3, 6, 8]
    assert r.best_epoch == 6
    np.testing.assert_array_equal(r.params.transition, r.minibatch.updates[6 * 3 - 1].params_out.transition)


@pytest.mark.parametrize("kwargs", [dict(), dict(check_convergence=True), dict(convergence="legacy"), dict(convergence="robust")])
def test_sparse_cadence_requires_convergence_off(kwargs):
    structure, params, Y = _problem()
    with pytest.raises(ValueError, match="pass convergence='off'"):
        em_lsim_minibatch_train(structure, params, Y, max_iterations=4, coherent_eval_interval=2, **_COMMON, **kwargs)


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "every", None])
def test_invalid_cadence_values_are_rejected(bad):
    structure, params, Y = _problem()
    with pytest.raises(ValueError, match="coherent_eval_interval must be"):
        em_lsim_minibatch_train(structure, params, Y, max_iterations=2, convergence="off", coherent_eval_interval=bad, **_COMMON)


def test_robust_policy_end_to_end_on_a_real_run():
    structure, params, Y = _problem(seed=5)
    r = em_lsim_minibatch_train(structure, params, Y, max_iterations=40, convergence="robust",
                                robust_convergence=RobustConvergence(rel_tol=1e-4, patience=2, min_epochs=3), **_COMMON)
    assert r.convergence == "robust" and r.stop_reason in ("converged", "max_iterations")
    assert r.evaluated_epoch_indices == list(range(1, r.n_epochs + 1))
    assert r.best_epoch in r.evaluated_epoch_indices


# ---------------- pi_0 vs pi_steady (P10e item 7) ----------------

def test_result_params_keep_the_selected_snapshot_pi_0_not_pi_steady(tmp_path):
    structure, params, Y = _problem(seed=6)
    r = em_lsim_minibatch_train(structure, params, Y, max_iterations=6, convergence="off", **_COMMON)
    last_update_of_best = [u for u in r.minibatch.updates if u.epoch_idx == r.best_epoch][-1]
    np.testing.assert_array_equal(r.params.pi_0, last_update_of_best.params_out.pi_0)
    assert not np.allclose(r.params.pi_0, r.pi_steady)  # the two are genuinely different vectors here
    np.testing.assert_array_equal(r.params.to_packed()["pi_0"], r.params.pi_0)

    save_lsim_params(tmp_path / "best.npz", r.params)
    loaded, _ = load_lsim_params(tmp_path / "best.npz")
    np.testing.assert_array_equal(loaded.pi_0, r.params.pi_0)

    fbc = FbConstants()
    score_best = em_mod._coherent_epoch_loglik(structure, r.params, Y, fbc, backend="numpy", backend_options=None, n_jobs=1, executor=None, validate="params")
    with_steady = r.params.copy()
    with_steady.pi_0 = r.pi_steady.copy()
    score_steady = em_mod._coherent_epoch_loglik(structure, with_steady, Y, fbc, backend="numpy", backend_options=None, n_jobs=1, executor=None, validate="params")
    assert score_best == r.loglik_trace[r.evaluated_epoch_indices.index(r.best_epoch)]
    assert score_steady != score_best
    reloaded_score = em_mod._coherent_epoch_loglik(structure, loaded, Y, fbc, backend="numpy", backend_options=None, n_jobs=1, executor=None, validate="params")
    assert reloaded_score == score_best  # save/reload never substitutes pi_steady
