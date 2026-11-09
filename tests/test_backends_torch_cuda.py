"""P8b gate: the batched Torch backend on a real CUDA device, float64 (skipped without CUDA).

**Tolerances are calibrated from measurement, not inherited from the CPU suite.** Measured on NVIDIA
RTX 6000 Ada (driver 610.43.02, CUDA runtime 13.0.96, torch 2.14.0+cu130) by
`benchmarks/p8b/cuda_parity_study.py` and `benchmarks/p8b/cuda_invariance_study.py` over all 48 posterior
fixtures (52 trials), 7 EM-trace fixtures (10 iterations) and 5 multi-iteration EM runs, after the P8b
`pi0_reg` fix (see tests/test_backends_torch.py). Each bound below is the measured worst case times
roughly 10, rounded up:

| quantity (CUDA f64 vs NumPy f64 unless noted) | measured worst | bound here |
|---|---|---|
| alpha / scale (abs) | 3.3e-16 / 1.1e-14 | 1e-14 / 1e-13 |
| beta / gamma (abs; c3_sparse_l2r -- Numba and Torch-CPU show the same order) | 7.3e-10 / 2.7e-10 | 1e-8 / 3e-9 |
| loglik (rel) | 8.5e-12 | 1e-10 |
| B, B_comp (rel only; zero pattern exact) | 6.0e-9 | 1e-7 |
| vs MATLAB: alpha / beta / gamma / scale (abs) | 1.6e-14 / 1.3e-9 / 2.6e-10 / 1.1e-12 | 1e-13 / 1e-8 / 3e-9 / 1e-11 |
| stats: mass fields (zeta, gamma_*, resp, weighted_W, pairwise_Q) (abs) | 2.3e-10 | 3e-9 |
| stats: responsibility-weighted moments (abs) | 3.0e-10 | 3e-9 |
| stats: observation-only accumulators (abs) | 4.5e-13 | 1e-11 |
| EM-trace one-step params vs MATLAB (abs) / loglik (rel) | 1.3e-10 / 1.2e-14 | 2e-9 / 1e-12 |
| 5-iteration em_lsim / mini-batch training params (abs) / loglik (rel) | 3.2e-11 / 2.5e-13 | 3e-10 / 3e-12 |
| chunk_size / max_memory_bytes / ordering invariance (rel) | 2.9e-13 | 3e-12 |
| repeated warm runs | bit-identical | exact |

Integer statistics, fallback flags and Diagnostics counters are compared exactly.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("torch.cuda.is_available() is False -- P8b CUDA tests need a CUDA device", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_em_trace_fixture, load_posterior_fixture  # noqa: E402

from chmm_lsim.backends.base import run_backend  # noqa: E402
from chmm_lsim.backends.torch_batched import _estimate_per_seq_bytes, forward_backward_batch  # noqa: E402
from chmm_lsim.compat import FbConstants  # noqa: E402
from chmm_lsim.em import e_step, em_lsim, em_lsim_minibatch_train, finalize_expected_stats, m_step  # noqa: E402
from chmm_lsim.params import LSIMParams  # noqa: E402
from chmm_lsim.structure import LSIMStructure  # noqa: E402

CUDA64 = {"device": "cuda", "dtype": "float64"}

POST_ABS = {"alpha": 1e-14, "beta": 1e-8, "gamma": 3e-9, "scale": 1e-13}
MATLAB_ABS = {"alpha": 1e-13, "beta": 1e-8, "gamma": 3e-9, "scale": 1e-11}
LOGLIK_RTOL = 1e-10
B_RTOL = 1e-7
STATS_MASS = ("zeta", "gamma_first", "gamma_sum")
CH_MASS_ABS = {"resp": 3e-9, "weighted_W": 3e-9, "pairwise_Q": 3e-9, "weighted_mean": 3e-9, "weighted_M2": 3e-9,
               "pairwise_M": 3e-9, "pairwise_C": 3e-9}
CH_OBS_ABS = {"n_obs": 1e-11, "welford_mean": 1e-11, "welford_M2": 1e-11, "allzero_n": 1e-11, "allzero_mean": 1e-11,
              "allzero_M2": 1e-11, "u_pairwise_Q": 1e-11, "u_pairwise_M": 1e-11, "u_pairwise_C": 1e-11}
CH_INT = ("resp_pos", "n_w_pos_diag", "n_w_pos_full")
INVARIANCE_RTOL, INVARIANCE_ATOL = 3e-12, 1e-14


def _build(fx):
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    fbc = FbConstants(variant=fx.fb_constants.variant, beta_floor=fx.fb_constants.beta_floor, pi0_reg=fx.fb_constants.pi0_reg,
                      f_reg=fx.fb_constants.f_reg, svi_eps=fx.fb_constants.svi_eps,
                      poise_refinement=fx.fb_constants.poise_refinement)
    return structure, fbc


def _posterior_model(fx):
    structure, fbc = _build(fx)
    params = LSIMParams.from_packed(structure, pi_0=fx.pi_0, transition=fx.transitions_matrices, coupling=fx.coupling_tetha_IM,
                                    P_all=fx.P_all, mu_all=fx.mu_all, sigma_all=fx.sigma_all, sigma_diag=bool(fx.sigma_diag))
    return structure, params, fbc


def _em_params(structure, sigma_diag, packed):
    return LSIMParams.from_packed(structure, pi_0=packed["pi_0"], transition=packed["transition_matrices_IM"],
                                  coupling=packed["coupling_tetha_IM"], P_all=packed["P_all"], mu_all=packed["mu_all"],
                                  sigma_all=packed["sigma_all"], sigma_diag=sigma_diag)


def _assert_posterior_close(post_cu, post_np, label):
    for name, atol in POST_ABS.items():
        np.testing.assert_allclose(getattr(post_cu, name), getattr(post_np, name), rtol=0, atol=atol, err_msg=f"{label}: {name}")
    np.testing.assert_allclose(post_cu.loglik, post_np.loglik, rtol=LOGLIK_RTOL, atol=0, err_msg=f"{label}: loglik")
    for name, got, ref in (("B", post_cu.B_raw(), post_np.B), ("B_comp", post_cu.B_comp_raw(), post_np.B_comp)):
        np.testing.assert_array_equal(got == 0, ref == 0, err_msg=f"{label}: {name} zero pattern")
        np.testing.assert_allclose(got, ref, rtol=B_RTOL, atol=0, err_msg=f"{label}: {name}")
    assert post_cu.diagnostics == post_np.diagnostics, f"{label}: diagnostics"


# --------------------------------------------------------------------------
# 1. Every fixture x variant x trial: posteriors (single and ragged-batched) vs NumPy and MATLAB.
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_cuda_f64_posteriors_match_numpy_and_matlab(fixtures_dir, posterior_fixture_id):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _posterior_model(fx)
    posts_np = []
    for i, trial in enumerate(fx.trials):
        post_np = run_backend("numpy", fbc.variant, structure, params, trial.Y, fbc)
        posts_np.append(post_np)
        post_cu = run_backend("torch", fbc.variant, structure, params, trial.Y, fbc, device="cuda", dtype="float64")
        label = f"{posterior_fixture_id} trial {i}"
        _assert_posterior_close(post_cu, post_np, label)
        for name, matlab in (("alpha", trial.alpha), ("beta", trial.beta), ("gamma", trial.alpha_T), ("scale", trial.b_c_ot_nc)):
            np.testing.assert_allclose(getattr(post_cu, name), matlab, rtol=0, atol=MATLAB_ABS[name], err_msg=f"{label}: {name} vs MATLAB")
        np.testing.assert_allclose(post_cu.loglik, trial.P_O_model, rtol=LOGLIK_RTOL, atol=0)
        np.testing.assert_allclose(post_cu.B_raw(), trial.B, rtol=B_RTOL, atol=0)
        np.testing.assert_allclose(post_cu.B_comp_raw(), trial.B_comp, rtol=B_RTOL, atol=0)
    Ys = [t.Y for t in fx.trials]
    for chunk in forward_backward_batch(structure, params, Ys, fbc, mode="posterior", device="cuda", dtype="float64", chunk_size=len(Ys)):
        for li, oi in enumerate(chunk.sequence_indices):
            _assert_posterior_close(chunk.extract(li), posts_np[oi], f"{posterior_fixture_id} trial {oi} (batched)")


# --------------------------------------------------------------------------
# 2. mode="stats": every ExpectedStats / ChannelExpectedStats field, and finalization flags.
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_cuda_f64_stats_every_field_matches_numpy(fixtures_dir, posterior_fixture_id):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _posterior_model(fx)
    Ys = [t.Y for t in fx.trials]
    s_np = e_step(structure, params, Ys, fbc, backend="numpy")
    s_cu = e_step(structure, params, Ys, fbc, backend="torch", backend_options=CUDA64)
    for name in STATS_MASS:
        np.testing.assert_allclose(getattr(s_cu, name), getattr(s_np, name), rtol=0, atol=3e-9, err_msg=name)
    np.testing.assert_allclose(s_cu.loglik, s_np.loglik, rtol=LOGLIK_RTOL, atol=0)
    assert (s_cu.n_trials, s_cu.n_transitions, s_cu.T_all) == (s_np.n_trials, s_np.n_transitions, s_np.T_all)
    for c, (a, b) in enumerate(zip(s_cu.channels, s_np.channels)):
        for name, atol in {**CH_MASS_ABS, **CH_OBS_ABS}.items():
            x, y = getattr(a, name), getattr(b, name)
            assert (x is None) == (y is None), f"c={c} {name}: None mismatch"
            if x is not None:
                np.testing.assert_allclose(x, y, rtol=0, atol=atol, err_msg=f"c={c} {name}")
        for name in CH_INT:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=f"c={c} {name}")
    f_cu = finalize_expected_stats(s_cu, sigma_diag=params.sigma_diag)
    f_np = finalize_expected_stats(s_np, sigma_diag=params.sigma_diag)
    for c in range(structure.C):
        np.testing.assert_array_equal(f_cu.channels[c].fallback_mu, f_np.channels[c].fallback_mu)
        if not params.sigma_diag:
            np.testing.assert_array_equal(f_cu.channels[c].fallback_sigma_full, f_np.channels[c].fallback_sigma_full)


# --------------------------------------------------------------------------
# 3. EM traces: one CUDA E-step + M-step per stored MATLAB iteration.
# --------------------------------------------------------------------------


def test_cuda_f64_em_trace_step_matches_matlab(fixtures_dir, em_trace_scenario_id):
    fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
    structure, fbc = _build(fx)
    sd = bool(fx.sigma_diag)
    for itr in range(1, len(fx.iterations) + 1):
        p_in = _em_params(structure, sd, fx.params_in(itr))
        stats = finalize_expected_stats(e_step(structure, p_in, fx.Y_trials, fbc, backend="torch", backend_options=CUDA64, validate="off"), sigma_diag=sd)
        p_out = m_step(structure, stats, sigma_diag=sd)
        it = fx.iterations[itr - 1]
        np.testing.assert_allclose(stats.loglik, it.loglik_params_in, rtol=1e-12, atol=0)
        packed = p_out.to_packed()
        for got, ref, name in ((p_out.pi_0, it.params_out_pi0, "pi_0"), (p_out.transition, it.params_out_transition, "transition"),
                               (p_out.coupling, it.params_out_coupling, "coupling"), (packed["P_all"], it.params_out_Pall, "P_all"),
                               (packed["mu_all"], it.params_out_muall, "mu_all"), (packed["sigma_all"], it.params_out_sigmaall, "sigma_all")):
            np.testing.assert_allclose(got, ref, rtol=0, atol=2e-9, err_msg=f"itr {itr} {name}")


# --------------------------------------------------------------------------
# 4. Multi-iteration EM (diagonal, full covariance, ragged multi-trial, missing, low-mass): CUDA vs NumPy.
# --------------------------------------------------------------------------


def _assert_params_close(p, q, atol, label):
    a, b = p.to_packed(), q.to_packed()
    for k in ("pi_0", "transitions_matrices", "coupling_tetha_IM", "P_all", "mu_all", "sigma_all"):
        np.testing.assert_allclose(a[k], b[k], rtol=0, atol=atol, err_msg=f"{label}: {k}")


@pytest.mark.parametrize("scenario", ["c3_het_diag", "c3_het_full", "c3_trials_unequal", "c3_het_full_nan", "c3_lowmass_full"])
def test_cuda_f64_multi_iteration_em_matches_numpy(fixtures_dir, scenario):
    fx = load_em_trace_fixture(fixtures_dir, scenario)
    structure, fbc = _build(fx)
    sd = bool(fx.sigma_diag)
    p0 = _em_params(structure, sd, fx.params_in(1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_np = em_lsim(structure, p0.copy(), fx.Y_trials, fb_constants=fbc, max_iterations=5, backend="numpy", check_convergence=False)
        r_cu = em_lsim(structure, p0.copy(), fx.Y_trials, fb_constants=fbc, max_iterations=5, backend="torch", backend_options=CUDA64, check_convergence=False)
    np.testing.assert_allclose(r_cu.loglik_trace, r_np.loglik_trace, rtol=3e-12, atol=0)
    _assert_params_close(r_cu.params, r_np.params, 3e-10, f"{scenario} em_lsim")

    kw = dict(fb_constants=fbc, max_iterations=5, num_minibatches=min(2, len(fx.Y_trials)), shuffle=True, random_state=0, convergence="off")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_np = em_lsim_minibatch_train(structure, p0.copy(), fx.Y_trials, backend="numpy", **kw)
        m_cu = em_lsim_minibatch_train(structure, p0.copy(), fx.Y_trials, backend="torch", backend_options=CUDA64, **kw)
    np.testing.assert_allclose(m_cu.loglik_trace, m_np.loglik_trace, rtol=3e-12, atol=0)
    for u_cu, u_np in zip(m_cu.minibatch.updates, m_np.minibatch.updates):
        _assert_params_close(u_cu.params_out, u_np.params_out, 3e-10, f"{scenario} e{u_cu.epoch_idx} mb{u_cu.minibatch_idx}")
    np.testing.assert_allclose(m_cu.pi_steady, m_np.pi_steady, rtol=0, atol=2e-12)
    assert m_cu.best_epoch == m_np.best_epoch


# --------------------------------------------------------------------------
# 5. Invariance: chunk_size, max_memory_bytes, input ordering, repeated warm runs, mixed-length batches.
# --------------------------------------------------------------------------


def _ragged(fx):
    base = max((t.Y for t in fx.trials), key=lambda y: y.shape[1])
    T = base.shape[1]
    lengths = sorted({1, 2, 3, 5, max(1, T // 3), max(1, T // 2), max(1, T - 1), T})
    rng = np.random.default_rng(0)
    data = [np.ascontiguousarray(base[:, :L]) for L in lengths]
    return [data[i] for i in rng.permutation(len(data))]


def _all_modes(structure, params, data, fbc, **opts):
    ll = forward_backward_batch(structure, params, data, fbc, mode="loglik", device="cuda", dtype="float64", **opts)
    posts = [None] * len(data)
    for chunk in forward_backward_batch(structure, params, data, fbc, mode="posterior", device="cuda", dtype="float64", **opts):
        for li, oi in enumerate(chunk.sequence_indices):
            posts[oi] = chunk.extract(li)
    stats = e_step(structure, params, data, fbc, backend="torch", backend_options={**CUDA64, **opts})
    return ll, posts, stats


def _assert_invariant(a, b, label, *, exact=False):
    ll_a, posts_a, s_a = a
    ll_b, posts_b, s_b = b
    close = (lambda x, y, m: np.testing.assert_array_equal(x, y, err_msg=m)) if exact else (
        lambda x, y, m: np.testing.assert_allclose(x, y, rtol=INVARIANCE_RTOL, atol=INVARIANCE_ATOL, err_msg=m))
    close(ll_b, ll_a, f"{label}: loglik")
    for i, (pa, pb) in enumerate(zip(posts_a, posts_b)):
        for q in ("alpha", "beta", "gamma", "scale"):
            close(getattr(pb, q), getattr(pa, q), f"{label}: trial {i} {q}")
        assert pb.diagnostics == pa.diagnostics
    for q in STATS_MASS:
        close(getattr(s_b, q), getattr(s_a, q), f"{label}: {q}")
    close(s_b.loglik, s_a.loglik, f"{label}: stats loglik")
    assert (s_b.n_trials, s_b.n_transitions, s_b.T_all) == (s_a.n_trials, s_a.n_transitions, s_a.T_all)
    for c, (ca, cb) in enumerate(zip(s_a.channels, s_b.channels)):
        for q in (*CH_MASS_ABS, *CH_OBS_ABS):
            if getattr(ca, q) is not None:
                close(getattr(cb, q), getattr(ca, q), f"{label}: c={c} {q}")
        for q in CH_INT:
            np.testing.assert_array_equal(getattr(cb, q), getattr(ca, q))


INVARIANCE_CASES = [(s, v) for s in ("c3_het_diag", "c3_het_full", "c3_het_full_nan", "c5_K_mixed", "c3_sparse_l2r")
                    for v in ("sed", "fast", "poise", "poise_refined")]  # P12 added the two poise modes


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("scenario,variant", INVARIANCE_CASES)
def test_cuda_f64_invariant_to_chunk_size_budget_order_and_repeats(fixtures_dir, scenario, variant):
    fx = load_posterior_fixture(fixtures_dir, f"{scenario}__{variant}")
    structure, params, fbc = _posterior_model(fx)
    data = _ragged(fx)
    N = len(data)
    ref = _all_modes(structure, params, data, fbc, chunk_size=N)

    for cs in (1, 2, 3, 7):
        _assert_invariant(ref, _all_modes(structure, params, data, fbc, chunk_size=cs), f"chunk_size={cs}")
    _assert_invariant(ref, _all_modes(structure, params, data, fbc), "chunk_size=auto")

    per_seq = _estimate_per_seq_bytes(structure.S, max(Y.shape[1] for Y in data), structure.max_K, structure.D, structure.max_d, torch.float64)
    for fit in (1, 2, 5, N):
        _assert_invariant(ref, _all_modes(structure, params, data, fbc, max_memory_bytes=int(per_seq * fit)), f"budget fits {fit}")

    rng = np.random.default_rng(1)
    for rep in range(2):
        perm = rng.permutation(N)
        ll_p, posts_p, s_p = _all_modes(structure, params, [data[j] for j in perm], fbc, chunk_size=3)
        inv = np.argsort(perm)
        _assert_invariant(ref, (ll_p[inv], [posts_p[j] for j in inv], s_p), f"permutation {rep}")

    for rep in range(2):
        _assert_invariant(ref, _all_modes(structure, params, data, fbc, chunk_size=N), f"repeat {rep}", exact=True)

    for i, Y in enumerate(data):  # the mixed-length batch vs each trial alone on NumPy float64
        # `variant` is the FIXTURE LABEL, which is not a variant name for the two poise modes
        # (`poise_refined` is `variant="poise"` with the refinement flag on) -- the constants are
        # authoritative, and run_backend rightly rejects a label that disagrees with them.
        post_np = run_backend("numpy", fbc.variant, structure, params, Y, fbc)
        _assert_posterior_close(ref[1][i], post_np, f"mixed batch trial T={Y.shape[1]}")
        np.testing.assert_allclose(ref[0][i], post_np.loglik, rtol=LOGLIK_RTOL, atol=0)
