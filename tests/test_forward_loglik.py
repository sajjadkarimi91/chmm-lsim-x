"""Forward-only log-likelihood (`run_backend_loglik`, `mode="loglik"`) for the
NumPy and Numba backends, and its use in coherent epoch scoring.

The contract is exact equality, not closeness: the forward-only value runs
the posterior runner's own validation, emissions, pi_0 regularization and
forward recursion, so it must equal `run_backend(...).loglik` bit for bit on
every MATLAB posterior fixture (sed and fast), and a coherent epoch score must
equal `e_step(...).loglik` bit for bit for serial, thread and process
execution -- which is what keeps convergence and best-epoch selection
unchanged in `em_lsim_minibatch_train`.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

import chmm_lsim.models.lsim.em as em_mod
from chmm_lsim import ChannelGMM, FbConstants, LSIMParams, LSIMStructure, e_step, em_lsim_minibatch_train
from chmm_lsim.backends.base import UnsupportedCombination, get_backend, run_backend, run_backend_loglik
from chmm_lsim.core.validate import ValidationError

try:
    import numba  # noqa: F401

    BACKENDS = ["numpy", "numba"]
except ImportError:  # pragma: no cover
    BACKENDS = ["numpy"]


def _build(fx):
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    params = LSIMParams.from_packed(
        structure, pi_0=fx.pi_0, transition=fx.transitions_matrices, coupling=fx.coupling_tetha_IM,
        P_all=fx.P_all, mu_all=fx.mu_all, sigma_all=fx.sigma_all, sigma_diag=bool(fx.sigma_diag),
    )
    fbc = FbConstants(
        variant=fx.fb_constants.variant, beta_floor=fx.fb_constants.beta_floor, pi0_reg=fx.fb_constants.pi0_reg,
        f_reg=fx.fb_constants.f_reg, svi_eps=fx.fb_constants.svi_eps,
        poise_refinement=fx.fb_constants.poise_refinement,
    )
    return structure, params, fbc


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_forward_only_equals_posterior_loglik_on_every_matlab_fixture(fixtures_dir, posterior_fixture_id, backend):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    for trial in fx.trials:
        post = run_backend(backend, fbc.variant, structure, params, trial.Y, fbc)
        fwd = run_backend_loglik(backend, fbc.variant, structure, params, trial.Y, fbc)
        assert fwd == post.loglik  # bit-identical
        np.testing.assert_allclose(fwd, trial.P_O_model, rtol=1e-9, atol=1e-9)  # and still matches MATLAB


def _single_channel_three_state():
    structure = LSIMStructure(channel_state_num=(3,), channel_obs_dim=(1,), num_gmm_component=(1,))
    gmm = ChannelGMM(weights=np.ones((3, 1)), mu=np.array([0.0, 5.0, 10.0]).reshape(1, 3, 1), sigma=np.ones((1, 3, 1)), diag=True)
    params = LSIMParams(
        structure=structure, pi_0=np.array([0.5, 0.3, 0.2]),
        transition=np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
        coupling=np.array([[1.0]]), emissions=[gmm],
    )
    params.validate()
    return structure, params


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("position", ["first", "interior", "final"])
@pytest.mark.parametrize("variant", ["sed", "fast"])
@pytest.mark.parametrize("validate", ["params", "off"])
def test_zero_emission_outcome_matches_posterior_path(backend, position, variant, validate):
    """Same outcome as the posterior runner when every emission density
    underflows: the same exception type, or the same (-inf) loglik."""
    structure, params = _single_channel_three_state()
    Y = {"first": [[1000.0, 0.5, 0.5]], "interior": [[0.5, 1000.0, 0.5]], "final": [[0.5, 0.6, 1000.0]]}[position]
    Y = np.array(Y)
    fbc = FbConstants(variant=variant)

    def outcome(fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return ("ok", fn())
            except Exception as exc:  # noqa: BLE001
                return ("raised", type(exc))

    post = outcome(lambda: run_backend(backend, variant, structure, params, Y, fbc, validate=validate).loglik)
    fwd = outcome(lambda: run_backend_loglik(backend, variant, structure, params, Y, fbc, validate=validate))
    assert post[0] == fwd[0]
    if post[0] == "ok":
        assert (post[1] == fwd[1]) or (np.isnan(post[1]) and np.isnan(fwd[1]))
    else:
        assert post[1] is fwd[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_validation_is_preserved(backend):
    structure, params = _single_channel_three_state()
    fbc = FbConstants(variant="sed")
    with pytest.raises(ValidationError):
        run_backend_loglik(backend, "sed", structure, params, np.array([[0.1, np.inf, 0.3]]), fbc)
    bad = params.copy()
    bad.pi_0 = np.array([0.9, 0.9, 0.9])
    with pytest.raises(ValidationError):
        run_backend_loglik(backend, "sed", structure, bad, np.array([[0.1, 0.2, 0.3]]), fbc)
    with pytest.raises(ValueError, match="conflicting inference variant"):
        run_backend_loglik(backend, "fast", structure, params, np.array([[0.1, 0.2, 0.3]]), FbConstants(variant="sed"))
    with pytest.raises(ValueError, match="validate must be one of"):
        run_backend_loglik(backend, "sed", structure, params, np.array([[0.1, 0.2, 0.3]]), fbc, validate="some")


def test_capabilities_and_unsupported_backends():
    assert "loglik" in get_backend("numpy").modes
    structure, params = _single_channel_three_state()
    with pytest.raises(UnsupportedCombination, match="unknown execution backend"):
        run_backend_loglik("tensorflow", "sed", structure, params, np.zeros((1, 3)), FbConstants(variant="sed"))
    # run_backend itself stays posterior-only
    with pytest.raises(UnsupportedCombination, match="always mode='posterior'"):
        run_backend("numpy", "sed", structure, params, np.zeros((1, 3)), FbConstants(variant="sed"), mode="loglik")


def _multi_trial_problem(seed=0):
    structure = LSIMStructure(channel_state_num=(2, 3, 2), channel_obs_dim=(1, 2, 1), num_gmm_component=(1, 2, 1))
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for z in range(structure.C):
        sl = structure.state_slice(z)
        transition[:, sl] = rng.dirichlet(np.ones(sl.stop - sl.start), size=structure.S)
    coupling = rng.dirichlet(np.ones(structure.C), size=structure.C).T
    emissions = []
    for c in range(structure.C):
        M, d, K = structure.channel_state_num[c], structure.channel_obs_dim[c], structure.num_gmm_component[c]
        emissions.append(ChannelGMM(rng.dirichlet(np.ones(K), size=M), rng.normal(size=(d, M, K)), rng.uniform(0.5, 2, size=(d, M, K)), diag=True))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    Y = [rng.normal(size=(structure.D, T)) for T in (40, 1, 75, 13, 60, 22, 9)]
    Y[2][1, 7] = np.nan
    return structure, params, Y


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", ["sed", "fast"])
@pytest.mark.parametrize("n_jobs,executor", [(1, None), (3, "thread"), (3, "process")])
def test_coherent_score_is_bit_identical_to_e_step(backend, variant, n_jobs, executor):
    structure, params, Y = _multi_trial_problem()
    fbc = FbConstants(variant=variant)
    ref = e_step(structure, params, Y, fbc, backend=backend, n_jobs=n_jobs, executor=executor).loglik
    got = em_mod._coherent_epoch_loglik(
        structure, params, Y, fbc, backend=backend, backend_options=None, n_jobs=n_jobs, executor=executor, validate="params",
    )
    assert got == ref
    serial_ref = e_step(structure, params, Y, fbc, backend=backend).loglik
    assert got == serial_ref  # a left fold in trial order, independent of executor


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_forward_only_worker_error_names_the_trial(executor):
    structure, params, Y = _multi_trial_problem()
    Y = list(Y)
    Y[4] = np.ones((structure.D + 1, 5))  # wrong row count -> ValidationError in the worker
    with pytest.raises(em_mod.TrialExecutionError) as info:
        em_mod._forward_only_loglik(structure, params, Y, FbConstants(), backend="numpy", validate="params", n_jobs=2, executor=executor)
    assert info.value.trial_index == 4


def test_validate_all_keeps_the_full_e_step_path(monkeypatch):
    structure, params, Y = _multi_trial_problem()
    calls = {"e_step": 0, "forward": 0}
    real_e_step, real_fwd = em_mod.e_step, em_mod._forward_only_loglik

    def count_e_step(*a, **k):
        calls["e_step"] += 1
        return real_e_step(*a, **k)

    def count_fwd(*a, **k):
        calls["forward"] += 1
        return real_fwd(*a, **k)

    monkeypatch.setattr(em_mod, "e_step", count_e_step)
    monkeypatch.setattr(em_mod, "_forward_only_loglik", count_fwd)
    for validate, expect in (("all", {"e_step": 1, "forward": 0}), ("params", {"e_step": 0, "forward": 1})):
        calls.update(e_step=0, forward=0)
        em_mod._coherent_epoch_loglik(structure, params, Y, FbConstants(), backend="numpy", backend_options=None,
                                      n_jobs=1, executor=None, validate=validate)
        assert calls == expect, validate


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_jobs,executor", [(1, None), (2, "process")])
def test_training_trace_and_best_epoch_unchanged_by_forward_only_scoring(monkeypatch, backend, n_jobs, executor):
    """em_lsim_minibatch_train with forward-only scoring vs the previous full
    e_step scoring: identical loglik/AIC/BIC traces, best epoch, params and pi_steady."""
    structure, params, Y = _multi_trial_problem(seed=3)
    kwargs = dict(max_iterations=12, num_minibatches=3, shuffle=True, random_state=1, fb_constants=FbConstants(),
                  backend=backend, n_jobs=n_jobs, executor=executor, check_convergence=True)
    new = em_lsim_minibatch_train(structure, params, Y, **kwargs)

    def old_scoring(structure, params, Y_trials, fb_constants, *, backend, backend_options, n_jobs, executor, validate, start_method=None):
        return e_step(structure, params, Y_trials, fb_constants, backend=backend, backend_options=backend_options,
                      n_jobs=n_jobs, executor=executor, validate=validate, start_method=start_method).loglik

    monkeypatch.setattr(em_mod, "_coherent_epoch_loglik", old_scoring)
    old = em_lsim_minibatch_train(structure, params, Y, **kwargs)

    assert new.loglik_trace == old.loglik_trace
    assert new.aic_trace == old.aic_trace and new.bic_trace == old.bic_trace
    assert (new.best_epoch, new.n_epochs) == (old.best_epoch, old.n_epochs)
    np.testing.assert_array_equal(new.pi_steady, old.pi_steady)
    np.testing.assert_array_equal(new.params.transition, old.params.transition)
    for a, b in zip(new.params.emissions, old.params.emissions):
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.sigma, b.sigma)


# --------------------------------------------------------------------------
# P10e item 4: validate="all" keeps its full meaning, and the capability /
# loglik-runner registries cannot disagree.
# --------------------------------------------------------------------------

import chmm_lsim.backends.base as base_mod  # noqa: E402


def _backends_loaded():
    names = list(BACKENDS)
    try:
        import torch  # noqa: F401

        names.append("torch")
    except ImportError:  # pragma: no cover
        pass
    for n in names:
        get_backend(n)
    return names


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_all_routes_to_the_full_posterior_runner(backend, monkeypatch):
    structure, params, Y = _multi_trial_problem(seed=11)
    fbc = FbConstants(variant="sed")
    expected = run_backend(backend, "sed", structure, params, Y[0], fbc, validate="all").loglik

    def forbidden(*a, **k):
        raise AssertionError("validate='all' must not use the forward-only runner")

    monkeypatch.setitem(base_mod._LOGLIK_RUNNERS, backend, forbidden)
    assert run_backend_loglik(backend, "sed", structure, params, Y[0], fbc, validate="all") == expected
    with pytest.raises(AssertionError, match="must not use the forward-only runner"):
        run_backend_loglik(backend, "sed", structure, params, Y[0], fbc, validate="params")


def test_validate_all_enforces_backward_side_invariants(monkeypatch):
    """A planted backward-only invariant failure (gamma block sums) must raise
    under validate='all' exactly as the posterior path does; forward-only
    validate='params' never evaluates it."""
    import chmm_lsim.models.lsim.inference.sed as sed_mod

    structure, params, Y = _multi_trial_problem(seed=12)
    real = sed_mod._assert_block_sum_one

    def planted(name, x, offsets, t):
        if name == "gamma":
            raise ValidationError(f"planted backward-side failure at t={t}")
        return real(name, x, offsets, t)

    monkeypatch.setattr(sed_mod, "_assert_block_sum_one", planted)
    with pytest.raises(ValidationError, match="planted backward-side failure"):
        run_backend("numpy", "sed", structure, params, Y[0], FbConstants(variant="sed"), validate="all")
    with pytest.raises(ValidationError, match="planted backward-side failure"):
        run_backend_loglik("numpy", "sed", structure, params, Y[0], FbConstants(variant="sed"), validate="all")
    run_backend_loglik("numpy", "sed", structure, params, Y[0], FbConstants(variant="sed"), validate="params")  # does not raise


@pytest.mark.parametrize("backend", BACKENDS)
def test_forward_only_runners_refuse_validate_all_directly(backend):
    structure, params, Y = _multi_trial_problem(seed=13)
    runner = base_mod._LOGLIK_RUNNERS[backend]
    with pytest.raises(ValueError, match="supports validate='params' or 'off'"):
        runner("sed", structure, params, Y[0], FbConstants(variant="sed"), validate="all")


def test_every_backend_declares_loglik_iff_it_registers_a_runner():
    for name in _backends_loaded():
        assert ("loglik" in get_backend(name).modes) == (name in base_mod._LOGLIK_RUNNERS), name


@pytest.mark.parametrize("modes,with_runner,match", [
    (("posterior", "loglik"), False, "declares mode 'loglik' but registers no loglik_runner"),
    (("posterior",), True, "registers a loglik_runner but does not declare mode 'loglik'"),
])
def test_register_backend_rejects_inconsistent_loglik_declarations_atomically(monkeypatch, modes, with_runner, match):
    from chmm_lsim.backends.base import BackendCapabilities, register_backend

    for attr in ("_BACKENDS", "_RUNNERS", "_LOGLIK_RUNNERS"):
        monkeypatch.setattr(base_mod, attr, dict(getattr(base_mod, attr)))
    snapshot = {a: dict(getattr(base_mod, a)) for a in ("_BACKENDS", "_RUNNERS", "_LOGLIK_RUNNERS")}
    caps = BackendCapabilities(name="p10e_fake", variants=("sed",), modes=modes)
    with pytest.raises(ValueError, match=match):
        register_backend(caps, lambda *a, **k: None, loglik_runner=(lambda *a, **k: 0.0) if with_runner else None)
    for a, before in snapshot.items():
        assert getattr(base_mod, a) == before, f"{a} changed after a rejected registration"


def test_torch_single_sequence_loglik_runner_matches_its_posterior():
    pytest.importorskip("torch")
    structure, params, Y = _multi_trial_problem(seed=14)
    for variant in ("sed", "fast", "poise"):
        fbc = FbConstants(variant=variant)
        for y in Y:
            post = run_backend("torch", variant, structure, params, y, fbc).loglik
            fwd = run_backend_loglik("torch", variant, structure, params, y, fbc)
            np.testing.assert_allclose(fwd, post, rtol=1e-9, atol=1e-9)
