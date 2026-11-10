"""P12 gate: LEARNING with POISE -- the M-step, AIC/BIC, driver bookkeeping and
the mini-batch training path consuming POISE statistics, on every backend.

`em_lsim_poise.m` is a thin wrapper over `em_lsim` precisely because SED and
POISE share the whole EM apparatus and differ only in the E-step recursion. The
Python port needs no new driver either: `variant="poise"` flows through
`e_step`/`em_lsim`/`em_lsim_minibatch`/`em_lsim_minibatch_train` unchanged. These
tests pin that, against MATLAB's own per-iteration `params_out` where a golden
trace exists.

The P12 export added POISE EM-trace fixtures for `c3_het_diag` (4 iterations),
`c3_het_full`, `c3_trials_unequal` and `c3_coupling_zero_row`, each in both
refinement modes. `tests/test_em.py`'s own generic trace tests already replay
them on the NumPy backend (the manifest's `em_trace` ids are shared); this module
adds the backend axis, the multi-iteration driver replay, and the mini-batch
training integration.

**Why the refined mode matters here specifically.** With refinement on, `beta` is
the reconstructed effective message (`gamma/alpha` where `alpha > 0`, else 0) --
and EM's transition statistic `zeta` is built from `beta`. A port that returned
the raw recursion's `beta` alongside a refined `gamma` would still pass every
posterior check yet produce a different M-step. The refined traces below are what
catch that.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_em_trace_fixture, load_posterior_fixture  # noqa: E402

from chmm_lsim.core.constants import FbConstants  # noqa: E402
from chmm_lsim.models.lsim.em import compute_aic_bic, e_step, em_lsim, em_lsim_minibatch_train, m_step  # noqa: E402
from chmm_lsim.learning.em_loop import finalize_expected_stats  # noqa: E402
from chmm_lsim.models.lsim.params import LSIMParams  # noqa: E402
from chmm_lsim.core.structure import LSIMStructure  # noqa: E402

RTOL, ATOL = 1e-8, 1e-9

POISE_TRACES = [
    f"{scenario}__{variant}"
    for scenario in ("c3_het_diag", "c3_het_full", "c3_trials_unequal", "c3_coupling_zero_row")
    for variant in ("poise", "poise_refined")
]

try:
    import numba  # noqa: F401

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

BACKENDS = [
    "numpy",
    pytest.param("numba", marks=pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")),
    pytest.param("torch", marks=pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")),
]


def _build(fx):
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    fbc = FbConstants(
        variant=fx.fb_constants.variant, beta_floor=fx.fb_constants.beta_floor, pi0_reg=fx.fb_constants.pi0_reg,
        f_reg=fx.fb_constants.f_reg, svi_eps=fx.fb_constants.svi_eps,
        poise_refinement=fx.fb_constants.poise_refinement,
    )
    return structure, fbc


def _params_from(structure, sigma_diag, packed):
    return LSIMParams.from_packed(
        structure, pi_0=packed["pi_0"], transition=packed["transition_matrices_IM"],
        coupling=packed["coupling_tetha_IM"], P_all=packed["P_all"], mu_all=packed["mu_all"],
        sigma_all=packed["sigma_all"], sigma_diag=sigma_diag,
    )


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("trace_id", POISE_TRACES)
def test_one_e_step_m_step_reproduces_matlab_params_out(fixtures_dir, trace_id, backend):
    """The P12 learning gate: `params_in[itr]` through one POISE E-step + the
    shared M-step reproduces MATLAB's `params_out[itr]`, and the E-step's own
    loglik/AIC/BIC match `*_params_in[itr]` -- for every stored iteration, both
    refinement modes, on every backend."""
    fx = load_em_trace_fixture(fixtures_dir, trace_id)
    structure, fbc = _build(fx)
    assert fbc.variant == "poise"
    assert fbc.poise_refinement == trace_id.endswith("__poise_refined")
    sigma_diag = bool(fx.sigma_diag)

    for itr in range(1, len(fx.iterations) + 1):
        params = _params_from(structure, sigma_diag, fx.params_in(itr))
        raw = e_step(structure, params, fx.Y_trials, fbc, backend=backend, validate="off")
        stats = finalize_expected_stats(raw, sigma_diag=sigma_diag)
        aic, bic = compute_aic_bic(structure, sigma_diag, stats.loglik, stats.T_all)
        new_params = m_step(structure, stats, sigma_diag=sigma_diag)

        it = fx.iterations[itr - 1]
        np.testing.assert_allclose(stats.loglik, it.loglik_params_in, rtol=1e-9, err_msg=f"{trace_id} itr {itr} loglik")
        np.testing.assert_allclose(aic, it.AIC_params_in, rtol=1e-9)
        np.testing.assert_allclose(bic, it.BIC_params_in, rtol=1e-9)

        packed = new_params.to_packed()
        np.testing.assert_allclose(new_params.pi_0, it.params_out_pi0, rtol=RTOL, atol=ATOL, err_msg=f"{trace_id} itr {itr} pi_0")
        np.testing.assert_allclose(new_params.transition, it.params_out_transition, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(new_params.coupling, it.params_out_coupling, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(packed["P_all"], it.params_out_Pall, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(packed["mu_all"], it.params_out_muall, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(packed["sigma_all"], it.params_out_sigmaall, rtol=RTOL, atol=ATOL)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("refinement", [False, True])
def test_multi_iteration_em_lsim_replay_matches_matlab(fixtures_dir, refinement):
    """The end-to-end driver, four POISE iterations from the raw init_model, must
    reproduce every iteration's params_out/loglik/AIC/BIC and the best-iterate --
    the same gate `test_em.py` applies to sed, on the POISE trace."""
    trace_id = "c3_het_diag__poise_refined" if refinement else "c3_het_diag__poise"
    fx = load_em_trace_fixture(fixtures_dir, trace_id)
    structure, fbc = _build(fx)
    assert fbc.poise_refinement is refinement
    params = LSIMParams.from_packed(
        structure, pi_0=fx.init_pi_0, transition=fx.init_transitions_matrices, coupling=fx.init_coupling_tetha_IM,
        P_all=fx.init_P_all, mu_all=fx.init_mu_all, sigma_all=fx.init_sigma_all, sigma_diag=bool(fx.sigma_diag),
    )

    captured = []
    result = em_lsim(
        structure, params, [fx.Y], max_iterations=len(fx.iterations), fb_constants=fbc,
        check_convergence=False, validate="off", raw_init_model=True, left2right=False,
        iter_hook=lambda itr, p, ll, aic, bic: captured.append((itr, p, ll, aic, bic)),
    )
    assert result.n_iterations == len(fx.iterations) == 4
    for itr, p, ll, aic, bic in captured:
        it = fx.iterations[itr - 1]
        np.testing.assert_allclose(ll, it.loglik_params_in, rtol=1e-9)
        np.testing.assert_allclose(aic, it.AIC_params_in, rtol=1e-9)
        np.testing.assert_allclose(bic, it.BIC_params_in, rtol=1e-9)
        np.testing.assert_allclose(p.pi_0, it.params_out_pi0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(p.to_packed()["sigma_all"], it.params_out_sigmaall, rtol=RTOL, atol=ATOL)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("trace_id", ["c3_trials_unequal__poise", "c3_trials_unequal__poise_refined"])
def test_expected_stats_agree_across_backends(fixtures_dir, trace_id, backend):
    """Every raw `ExpectedStats` field a POISE E-step produces must agree with the
    NumPy reference -- multi-trial and ragged, where the batched Torch path also
    exercises its padding."""
    fx = load_em_trace_fixture(fixtures_dir, trace_id)
    structure, fbc = _build(fx)
    params = _params_from(structure, bool(fx.sigma_diag), fx.params_in(1))
    ref = e_step(structure, params, fx.Y_trials, fbc, backend="numpy", validate="off")
    got = e_step(structure, params, fx.Y_trials, fbc, backend=backend, validate="off")

    for name in ("zeta", "gamma_first", "gamma_sum"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=0, atol=1e-11, err_msg=name)
    np.testing.assert_allclose(got.loglik, ref.loglik, rtol=1e-11, atol=0)
    assert (got.n_trials, got.n_transitions, got.T_all) == (ref.n_trials, ref.n_transitions, ref.T_all)
    for c, (a, b) in enumerate(zip(got.channels, ref.channels)):
        for name in ("resp", "weighted_W", "weighted_mean", "weighted_M2", "pairwise_Q", "pairwise_M", "pairwise_C",
                     "n_obs", "welford_mean", "welford_M2", "allzero_n", "allzero_mean", "allzero_M2"):
            x, y = getattr(a, name), getattr(b, name)
            assert (x is None) == (y is None)
            if x is not None:
                np.testing.assert_allclose(x, y, rtol=0, atol=1e-11, err_msg=f"c={c} {name}")
        for name in ("resp_pos", "n_w_pos_diag", "n_w_pos_full"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=f"c={c} {name}")


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("refinement", [False, True])
def test_minibatch_training_runs_with_poise_and_agrees_across_backends(fixtures_dir, backend, refinement):
    """`em_lsim_minibatch_train` needs no POISE-specific code: the variant flows
    through `backend_options`/`fb_constants` unchanged. Same schedule and same
    initial parameters must give the same trace on every backend."""
    fx = load_em_trace_fixture(fixtures_dir, "c3_trials_unequal__poise")
    structure, _fbc = _build(fx)
    fbc = FbConstants(variant="poise", poise_refinement=refinement)
    params = _params_from(structure, bool(fx.sigma_diag), fx.params_in(1))

    def run(which):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return em_lsim_minibatch_train(
                structure, params.copy(), fx.Y_trials, max_iterations=3, num_minibatches=2, shuffle=True,
                random_state=0, fb_constants=fbc, convergence="off", coherent_eval_interval=1,
                backend=which, n_jobs=1,
            )

    ref = run("numpy")
    got = run(backend)
    assert len(got.loglik_trace) == 3 and np.all(np.isfinite(got.loglik_trace))
    np.testing.assert_allclose(got.loglik_trace, ref.loglik_trace, rtol=1e-10, atol=0)
    for k, v in got.params.to_packed().items():
        if k == "sigma_diag":
            continue
        np.testing.assert_allclose(np.asarray(v, float), np.asarray(ref.params.to_packed()[k], float),
                                    rtol=0, atol=1e-10, err_msg=k)
    np.testing.assert_allclose(got.pi_steady, ref.pi_steady, rtol=0, atol=1e-10)
    for c in range(structure.C):
        np.testing.assert_allclose(got.pi_steady[structure.state_slice(c)].sum(), 1.0, rtol=0, atol=1e-10)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_refined_and_raw_training_diverge_but_both_stay_valid(fixtures_dir):
    """The refinement is not cosmetic: it changes the statistics EM consumes, so
    the two modes follow different trajectories from the same start -- while both
    keep every stochastic invariant."""
    fx = load_em_trace_fixture(fixtures_dir, "c3_het_diag__poise")
    structure, _fbc = _build(fx)
    params = _params_from(structure, bool(fx.sigma_diag), fx.params_in(1))

    traces = {}
    for refinement in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = em_lsim(structure, params.copy(), [fx.Y], max_iterations=3,
                          fb_constants=FbConstants(variant="poise", poise_refinement=refinement),
                          check_convergence=False, validate="off")
        traces[refinement] = res
        packed = res.params.to_packed()
        for c in range(structure.C):
            np.testing.assert_allclose(res.params.pi_0[structure.state_slice(c)].sum(), 1.0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(packed["coupling_tetha_IM"].sum(axis=0), 1.0, rtol=0, atol=1e-10)
        assert np.all(np.isfinite(packed["mu_all"])) and np.all(packed["sigma_all"] > 0)

    assert not np.allclose(traces[False].loglik_trace, traces[True].loglik_trace, rtol=1e-6), (
        "refinement must change the EM trajectory, not just the reported marginals")


def _tiny_poise_model():
    """A small, healthy 2-channel model for the driver test below -- built here
    rather than loaded from a fixture so the test states its own inputs."""
    from chmm_lsim.emissions.gmm import ChannelGMM
    from chmm_lsim.models.lsim.params import LSIMParams
    from chmm_lsim.core.structure import LSIMStructure

    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    rng = np.random.default_rng(0)
    transition = np.zeros((4, 4))
    for block in (slice(0, 2), slice(2, 4)):
        rows = rng.uniform(0.2, 0.8, size=(4, 2))
        transition[:, block] = rows / rows.sum(axis=1, keepdims=True)
    coupling = np.array([[0.6, 0.4], [0.4, 0.6]])

    def gmm(m0, m1):
        return ChannelGMM(np.ones((2, 1)), np.array([[[m0], [m1]]], dtype=float), np.ones((1, 2, 1)), diag=True)

    params = LSIMParams(structure=structure, pi_0=np.full(4, 0.5), transition=transition,
                        coupling=coupling, emissions=[gmm(-1.0, 1.0), gmm(-1.0, 1.0)])
    params.validate()
    return structure, params, rng.normal(size=(2, 25))


def test_poise_needs_no_dedicated_em_driver():
    """The MATLAB side ships `em_lsim_poise.m` only as an argument-routing wrapper;
    the Python port exposes the variant through `fb_constants` instead, so there is
    deliberately no `em_lsim_poise` symbol to import."""
    import chmm_lsim

    assert not hasattr(chmm_lsim, "em_lsim_poise")

    # The route that replaces it must actually work end to end, and the flag must be
    # documented where a caller would look for it -- `or True` asserted neither.
    assert "poise_refinement" in chmm_lsim.FbConstants.__doc__
    fbc = chmm_lsim.FbConstants(variant="poise")
    assert fbc.poise_refinement is False, "the documented default"
    structure, params, Y = _tiny_poise_model()
    result = em_lsim(structure, params.copy(), [Y], max_iterations=2, fb_constants=fbc,
                     check_convergence=False, validate="off")
    assert len(result.loglik_trace) >= 1 and np.all(np.isfinite(result.loglik_trace))


def test_full_batch_em_default_is_explicit_raw_poise():
    """The package-wide default is a product contract, not only an inference
    dispatcher detail: omitting `fb_constants` from EM must mean raw POISE."""
    structure, params, Y = _tiny_poise_model()
    default = em_lsim(
        structure, params.copy(), [Y], max_iterations=2,
        check_convergence=False, validate="off",
    )
    explicit = em_lsim(
        structure, params.copy(), [Y], max_iterations=2,
        fb_constants=FbConstants(variant="poise", poise_refinement=False),
        check_convergence=False, validate="off",
    )
    np.testing.assert_array_equal(default.loglik_trace, explicit.loglik_trace)
    np.testing.assert_array_equal(default.params.pi_0, explicit.params.pi_0)
    np.testing.assert_array_equal(default.params.transition, explicit.params.transition)
    np.testing.assert_array_equal(default.params.coupling, explicit.params.coupling)
