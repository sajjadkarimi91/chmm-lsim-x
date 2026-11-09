"""Full coupled HMM (CHMM): MATLAB parity and independent checks.

MATLAB references come from `tools/matlab/export_chmm_fixtures.m` (R2024b):
`fb` is `forward_backward_chmm` on the first sequence, `vit` the exact
4-argument `viterbi_chmm`, and `em` six iterations of `em_chmm` on all
sequences from the stored model passed as `init_model` (so its `pi_0` is
smoothed first, `raw_init_model=True` here).
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.io

import chmm_lsim
from chmm_lsim import (
    ChannelGMM,
    CHMMParams,
    EmResult,
    EquivalentModelTooLargeError,
    LSIMParams,
    LSIMStructure,
    ValidationError,
    build_equivalent_hmm,
    chmm_exact_forward_backward,
    chmm_joint_model,
    compute_chmm_aic_bic,
    em_chmm,
    em_lsim,
    exact_posterior,
    forward_backward_chmm,
    lsim_to_chmm,
    simulate_chmm,
    viterbi_chmm,
    viterbi_exact,
)
from chmm_lsim.emissions import compute_emissions

FIXTURE = Path(__file__).parent / "fixtures" / "chmm" / "chmm_fixtures.mat"
CASES = ["c2_diag", "c2_diag_nan", "c2_full_nan", "c2_mix_diag", "c3_diag", "c3_full_nan", "c2_T1"]
EM_CASES = [c for c in CASES if c != "c2_T1"]

# Observed maximum elementwise relative difference against MATLAB is ~4e-14
# (every quantity, every case), so 1e-10 leaves four orders of margin.
RTOL = 1e-10


# ---------------------------------------------------------------------------
# fixture loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return scipy.io.loadmat(FIXTURE, simplify_cells=True)


def _ints(x) -> tuple[int, ...]:
    return tuple(int(v) for v in np.atleast_1d(x))


def _params_from_mat(structure: LSIMStructure, p: dict, sigma_diag: bool) -> CHMMParams:
    """`simplify_cells` drops singleton dimensions; restore every array's shape."""
    M, d, K = structure.channel_state_num, structure.channel_obs_dim, structure.num_gmm_component
    J = int(np.prod(M))
    tables = p["transition"]
    transition = [np.asarray(tables[z], dtype=float).reshape(J, M[z]) for z in range(structure.C)]
    emissions = []
    for c, e in enumerate(p["emissions"]):
        sigma_shape = (d[c], M[c], K[c]) if sigma_diag else (d[c], d[c], M[c], K[c])
        emissions.append(
            ChannelGMM(
                weights=np.asarray(e["weights"], dtype=float).reshape(M[c], K[c]),
                mu=np.asarray(e["mu"], dtype=float).reshape(d[c], M[c], K[c]),
                sigma=np.asarray(e["sigma"], dtype=float).reshape(sigma_shape),
                diag=sigma_diag,
            )
        )
    return CHMMParams(structure=structure, pi_0=np.atleast_1d(np.asarray(p["pi_0"], dtype=float)), transition=transition, emissions=emissions)


def _load_case(fixtures: dict, name: str):
    fx = fixtures[name]
    structure = LSIMStructure(_ints(fx["M"]), _ints(fx["d"]), _ints(fx["K"]))
    sigma_diag = bool(fx["sigma_diag"])
    params = _params_from_mat(structure, fx["model"], sigma_diag)
    Y = fx["Y"]
    if not (isinstance(Y, np.ndarray) and Y.dtype == object):
        Y = [Y]
    Y_trials = [np.asarray(y, dtype=float).reshape(structure.D, -1) for y in Y]
    return fx, structure, params, Y_trials, sigma_diag


def _assert_params_close(actual: CHMMParams, expected: CHMMParams, rtol: float) -> None:
    np.testing.assert_allclose(actual.pi_0, expected.pi_0, rtol=rtol, atol=0)
    for a, b in zip(actual.transition, expected.transition):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=0)
    for a, b in zip(actual.emissions, expected.emissions):
        np.testing.assert_allclose(a.weights, b.weights, rtol=rtol, atol=0)
        np.testing.assert_allclose(a.mu, b.mu, rtol=rtol, atol=0)
        np.testing.assert_allclose(a.sigma, b.sigma, rtol=rtol, atol=0)


# ---------------------------------------------------------------------------
# MATLAB parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_forward_backward_matches_matlab(fixtures, name):
    fx, structure, params, Y_trials, _ = _load_case(fixtures, name)
    post = forward_backward_chmm(structure, params, Y_trials[0])
    T = Y_trials[0].shape[1]
    ref = fx["fb"]
    np.testing.assert_allclose(post.loglik, ref["loglik"], rtol=RTOL, atol=0)
    for key in ("gamma", "alpha", "B"):
        np.testing.assert_allclose(getattr(post, key), np.asarray(ref[key], dtype=float).reshape(structure.S, T), rtol=RTOL, atol=0)


@pytest.mark.parametrize("name", CASES)
def test_viterbi_matches_matlab(fixtures, name):
    fx, structure, params, Y_trials, _ = _load_case(fixtures, name)
    result = viterbi_chmm(structure, params, Y_trials[0])
    T = Y_trials[0].shape[1]
    expected_states = np.asarray(fx["vit"]["states"], dtype=np.int64).reshape(structure.C, T) - 1
    np.testing.assert_array_equal(result.states, expected_states)
    np.testing.assert_allclose(result.log_score, fx["vit"]["log_prob"], rtol=RTOL, atol=0)


@pytest.mark.parametrize("name", EM_CASES)
def test_em_matches_matlab(fixtures, name):
    fx, structure, params, Y_trials, sigma_diag = _load_case(fixtures, name)
    ref = fx["em"]
    result = em_chmm(structure, params, Y_trials, int(ref["max_iterations"]), check_convergence=False, raw_init_model=True)
    assert result.n_iterations == int(ref["max_iterations"])
    np.testing.assert_allclose(result.aic_trace, ref["aic"], rtol=RTOL, atol=0)
    np.testing.assert_allclose(result.loglik_trace, ref["loglik"], rtol=RTOL, atol=0)
    np.testing.assert_allclose(result.bic_trace, ref["bic"], rtol=RTOL, atol=0)
    _assert_params_close(result.params, _params_from_mat(structure, ref["params"], sigma_diag), RTOL)
    np.testing.assert_allclose(result.pi_steady, ref["pi_steady"], rtol=RTOL, atol=0)


# ---------------------------------------------------------------------------
# random models
# ---------------------------------------------------------------------------


def _random_gmms(rng, structure: LSIMStructure, sigma_diag: bool) -> list[ChannelGMM]:
    out = []
    for c in range(structure.C):
        M, d, K = structure.channel_state_num[c], structure.channel_obs_dim[c], structure.num_gmm_component[c]
        w = rng.random((M, K)) + 0.3
        mu = 1.5 * np.arange(1, M + 1)[None, :, None] + 0.4 * rng.standard_normal((d, M, K))
        if sigma_diag:
            sigma = 0.4 + rng.random((d, M, K))
        else:
            sigma = np.zeros((d, d, M, K))
            for i in range(M):
                for k in range(K):
                    R = 0.2 * rng.standard_normal((d, d))
                    sigma[:, :, i, k] = np.diag(0.4 + rng.random(d)) + R @ R.T
        out.append(ChannelGMM(weights=w / w.sum(axis=1, keepdims=True), mu=mu, sigma=sigma, diag=sigma_diag))
    return out


def _random_chmm(rng, structure: LSIMStructure, sigma_diag: bool = True) -> CHMMParams:
    J = int(np.prod(structure.channel_state_num))
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m) * 2) for m in structure.channel_state_num])
    transition = [rng.dirichlet(np.ones(m), size=J) for m in structure.channel_state_num]
    return CHMMParams(structure=structure, pi_0=pi_0, transition=transition, emissions=_random_gmms(rng, structure, sigma_diag))


def _random_lsim(rng, structure: LSIMStructure, sigma_diag: bool = True) -> LSIMParams:
    S, C = structure.S, structure.C
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m) * 2) for m in structure.channel_state_num])
    transition = np.zeros((S, S))
    for z in range(C):
        transition[:, structure.state_slice(z)] = rng.dirichlet(np.ones(structure.channel_state_num[z]), size=S)
    coupling = rng.dirichlet(np.ones(C), size=C).T
    return LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=_random_gmms(rng, structure, sigma_diag))


# ---------------------------------------------------------------------------
# independent checks
# ---------------------------------------------------------------------------


def _brute_force(structure: LSIMStructure, params: CHMMParams, Y):
    """Enumerates every joint path: (loglik, per-channel smoothed marginals, best path, its log-probability)."""
    dims = structure.channel_state_num
    J = int(np.prod(dims))
    T = Y.shape[1]
    B, _ = compute_emissions(structure, params, Y)
    joint_states = list(itertools.product(*[range(m) for m in dims]))

    def p_init(x):
        return np.prod([params.pi_0[structure.state_offsets[c] + x[c]] for c in range(structure.C)])

    def p_trans(j_prev, x):
        return np.prod([params.transition[c][j_prev, x[c]] for c in range(structure.C)])

    def p_emit(x, t):
        return np.prod([B[structure.state_offsets[c] + x[c], t] for c in range(structure.C)])

    total = 0.0
    marg = np.zeros((structure.S, T))
    best, best_path = -1.0, None
    for path in itertools.product(range(J), repeat=T):
        p = p_init(joint_states[path[0]]) * p_emit(joint_states[path[0]], 0)
        for t in range(1, T):
            p *= p_trans(path[t - 1], joint_states[path[t]]) * p_emit(joint_states[path[t]], t)
        total += p
        for t in range(T):
            for c in range(structure.C):
                marg[structure.state_offsets[c] + joint_states[path[t]][c], t] += p
        if p > best:
            best, best_path = p, path
    states = np.array([joint_states[j] for j in best_path]).T
    return np.log(total), marg / total, states, np.log(best)


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_brute_force_enumeration(sigma_diag):
    rng = np.random.default_rng(3)
    structure = LSIMStructure((2, 3), (1, 2), (2, 1))
    params = _random_chmm(rng, structure, sigma_diag)
    Y, _ = simulate_chmm(structure, params, 5, seed=11)
    Y[1, 2] = np.nan
    loglik, marg, states, log_best = _brute_force(structure, params, Y)

    post = forward_backward_chmm(structure, params, Y, return_joint=True)
    np.testing.assert_allclose(post.loglik, loglik, rtol=1e-12)
    np.testing.assert_allclose(post.gamma, marg, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(post.joint_gamma.sum(axis=0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(post.xi_sum.sum(), Y.shape[1] - 1, rtol=1e-12)

    vit = viterbi_chmm(structure, params, Y)
    np.testing.assert_array_equal(vit.states, states)
    np.testing.assert_allclose(vit.log_score, log_best, rtol=1e-12)


def test_predictive_alpha_is_one_step_prediction():
    rng = np.random.default_rng(5)
    structure = LSIMStructure((2, 2), (1, 1), (1, 1))
    params = _random_chmm(rng, structure)
    Y, _ = simulate_chmm(structure, params, 6, seed=2)
    post = forward_backward_chmm(structure, params, Y, return_joint=True)
    np.testing.assert_allclose(post.alpha[:, 0], params.pi_0, rtol=1e-14)
    # alpha[:, t] = P(x_t | y_1..y_{t-1}): log P(y_1..y_t) - log P(y_1..y_{t-1}) = log sum_j alpha_j(t) B_j(t)
    model = chmm_joint_model(structure, params, Y)
    for t in range(1, 6):
        ll_t = forward_backward_chmm(structure, params, Y[:, : t + 1]).loglik
        ll_prev = forward_backward_chmm(structure, params, Y[:, :t]).loglik
        np.testing.assert_allclose(ll_t - ll_prev, np.log(post.joint_alpha[:, t] @ np.exp(model.log_B[:, t])), rtol=1e-10)


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_lsim_to_chmm_matches_exact_lsim_posterior(sigma_diag):
    rng = np.random.default_rng(7)
    structure = LSIMStructure((2, 3, 2), (1, 2, 1), (1, 2, 1))
    lsim_params = _random_lsim(rng, structure, sigma_diag)
    Y, _ = chmm_lsim.simulate_lsim(structure, lsim_params, 40, seed=4)
    Y[0, 3] = np.nan
    Y[:, 7] = np.nan

    chmm = lsim_to_chmm(structure, lsim_params)
    chmm.validate()
    equiv = build_equivalent_hmm(structure, lsim_params)
    for z in range(structure.C):
        np.testing.assert_array_equal(chmm.transition[z], equiv.cond_transitions[z])

    post = forward_backward_chmm(structure, chmm, Y, return_joint=True)
    gamma_ref, loglik_ref = exact_posterior(equiv, Y)
    np.testing.assert_allclose(post.loglik, loglik_ref, rtol=1e-12)
    np.testing.assert_allclose(post.joint_gamma, gamma_ref, rtol=1e-9, atol=1e-14)

    model = chmm_joint_model(structure, chmm, Y)
    np.testing.assert_allclose(model.transition, equiv.transition, rtol=1e-13, atol=0)
    np.testing.assert_allclose(model.pi, equiv.pi0, rtol=1e-14, atol=0)

    vit = viterbi_chmm(structure, chmm, Y)
    vit_ref = viterbi_exact(structure, lsim_params, Y)
    np.testing.assert_array_equal(vit.states, vit_ref.states)
    np.testing.assert_allclose(vit.log_score, vit_ref.log_score, rtol=1e-12)

    # one em_chmm iteration scores the LSIM-equivalent CHMM with the exact LSIM likelihood
    Y2, _ = chmm_lsim.simulate_lsim(structure, lsim_params, 25, seed=5)
    result = em_chmm(structure, chmm, [Y, Y2], 1)
    exact = loglik_ref + exact_posterior(equiv, Y2)[1]
    np.testing.assert_allclose(result.loglik_trace[0], exact, rtol=1e-12)


def test_chmm_exact_forward_backward_rejects_impossible_observation():
    with pytest.raises(ValidationError, match="t=1"):
        chmm_exact_forward_backward(np.array([0.5, 0.5]), np.eye(2), np.array([[0.0, -np.inf], [0.0, -np.inf]]))


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_em_loglik_non_decreasing(sigma_diag):
    rng = np.random.default_rng(11)
    structure = LSIMStructure((2, 3), (2, 1), (1, 2))
    truth = _random_chmm(rng, structure, sigma_diag)
    Y_trials = [simulate_chmm(structure, truth, 150, seed=s)[0] for s in range(3)]
    Y_trials[0][0, 5] = np.nan
    start = _random_chmm(np.random.default_rng(99), structure, sigma_diag)
    result = em_chmm(structure, start, Y_trials, 15)
    ll = np.array(result.loglik_trace)
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[1:])), ll
    assert ll[-1] > ll[0]
    result.params.validate()


def test_em_chmm_and_em_lsim_return_the_same_result_type():
    rng = np.random.default_rng(13)
    structure = LSIMStructure((2, 2), (1, 1), (1, 1))
    lsim_params = _random_lsim(rng, structure)
    Y_trials = [chmm_lsim.simulate_lsim(structure, lsim_params, 30, seed=s)[0] for s in range(2)]
    hooks = []
    r_lsim = em_lsim(structure, lsim_params, Y_trials, max_iterations=4, check_convergence=False, raw_init_model=True)
    r_chmm = em_chmm(
        structure, lsim_to_chmm(structure, lsim_params), Y_trials, max_iterations=4, check_convergence=False,
        raw_init_model=True, iter_hook=lambda *args: hooks.append(args),
    )
    assert type(r_lsim) is type(r_chmm) is EmResult
    fields = [f.name for f in dataclasses.fields(EmResult)]
    for name in fields:
        a, b = getattr(r_lsim, name), getattr(r_chmm, name)
        assert type(a) is type(b) or name == "params"
    assert isinstance(r_chmm.params, CHMMParams)
    assert len(r_chmm.loglik_trace) == len(r_chmm.aic_trace) == len(r_chmm.bic_trace) == r_chmm.n_iterations == 4
    assert r_chmm.pi_steady.shape == (structure.S,)
    assert 1 <= r_chmm.best_iterate <= 4
    expected_best = 2  # iterations 1 and 2 are always taken, later ones only when strictly better
    for itr in range(3, 5):
        if r_chmm.loglik_trace[itr - 1] > r_chmm.loglik_trace[expected_best - 1]:
            expected_best = itr
    assert r_chmm.best_iterate == expected_best
    assert [h[0] for h in hooks] == [1, 2, 3, 4]
    assert all(isinstance(h[1], CHMMParams) for h in hooks)
    # same starting model: the first log-likelihood is the LSIM's exact one, which the
    # approximate LSIM E-step need not reproduce, but both are finite and close here
    assert np.isfinite(r_lsim.loglik_trace[0]) and np.isfinite(r_chmm.loglik_trace[0])


def test_aic_bic_parameter_count():
    structure = LSIMStructure((2, 3), (1, 2), (2, 1))
    J = 6
    emis_diag = 2 * 2 * 2 * 1 + 3 * 1 * 2 * 2 + 2 * 1 + 0
    emis_full = 2 * 2 * (1 + 1) + 3 * 1 * (2 + 4) + 2 * 1
    for sigma_diag, emis in ((True, emis_diag), (False, emis_full)):
        n = (1 + 2) * J + emis
        aic, bic = compute_chmm_aic_bic(structure, sigma_diag, -10.0, 50)
        assert aic == pytest.approx(20.0 + 2 * n)
        assert bic == pytest.approx(20.0 + np.log(50) * n)


def test_em_convergence_check_stops_early():
    structure = LSIMStructure((2,), (1,), (1,))
    params = CHMMParams(
        structure,
        np.array([0.5, 0.5]),
        [np.array([[0.9, 0.1], [0.1, 0.9]])],
        [ChannelGMM(np.ones((2, 1)), np.array([[[0.0], [5.0]]]), np.ones((1, 2, 1)), True)],
    )
    Y, _ = simulate_chmm(structure, params, 200, seed=0)
    full = em_chmm(structure, params, [Y], 40)
    stopped = em_chmm(structure, params, [Y], 40, check_convergence=True)
    assert full.n_iterations == 40
    assert 10 <= stopped.n_iterations < 40
    np.testing.assert_array_equal(stopped.loglik_trace, full.loglik_trace[: stopped.n_iterations])


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def test_simulate_shapes_ranges_and_seed():
    rng = np.random.default_rng(17)
    structure = LSIMStructure((2, 3, 2), (1, 2, 3), (2, 1, 1))
    params = _random_chmm(rng, structure, sigma_diag=False)
    Y, X = simulate_chmm(structure, params, 60, seed=3)
    assert Y.shape == (structure.D, 60) and Y.dtype == np.float64 and np.all(np.isfinite(Y))
    assert X.shape == (structure.C, 60) and X.dtype == np.int64
    for c in range(structure.C):
        assert X[c].min() >= 0 and X[c].max() < structure.channel_state_num[c]
    Y2, X2 = simulate_chmm(structure, params, 60, seed=3)
    np.testing.assert_array_equal(Y, Y2)
    np.testing.assert_array_equal(X, X2)
    with pytest.raises(ValidationError):
        simulate_chmm(structure, params, 0, seed=0)


def test_simulate_follows_joint_conditional_transitions():
    # deterministic tables: channel 0 copies channel 1's previous state, channel 1 flips its own
    structure = LSIMStructure((2, 2), (1, 1), (1, 1))
    J = 4
    states = np.array(np.unravel_index(np.arange(J), (2, 2)))
    t0 = np.eye(2)[states[1]]
    t1 = np.eye(2)[1 - states[1]]
    gmm = ChannelGMM(np.ones((2, 1)), np.array([[[0.0], [1.0]]]), np.full((1, 2, 1), 1e-4), True)
    params = CHMMParams(structure, np.array([1.0, 0.0, 1.0, 0.0]), [t0, t1], [gmm, gmm.copy()])
    _, X = simulate_chmm(structure, params, 20, seed=1)
    np.testing.assert_array_equal(X[1], np.arange(20) % 2)
    np.testing.assert_array_equal(X[0, 1:], X[1, :-1])


# ---------------------------------------------------------------------------
# CHMMParams validation
# ---------------------------------------------------------------------------


def _valid_params() -> CHMMParams:
    return _random_chmm(np.random.default_rng(23), LSIMStructure((2, 3), (1, 1), (1, 2)))


def test_params_validate_accepts_valid_and_copy_is_deep():
    p = _valid_params()
    p.validate()
    assert p.sigma_diag and p.n_joint_states == 6
    q = p.copy()
    q.transition[0][0, 0] = 99.0
    q.emissions[0].mu[0, 0, 0] = 99.0
    assert p.transition[0][0, 0] != 99.0 and p.emissions[0].mu[0, 0, 0] != 99.0


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p.pi_0.__setitem__(0, p.pi_0[0] + 0.1), "pi_0 block for channel 0"),
        (lambda p: p.pi_0.__setitem__(slice(0, 2), [1.5, -0.5]), "pi_0 has negative"),
        (lambda p: p.transition[1].__setitem__((2, 0), np.nan), r"transition\[1\] contains NaN"),
        (lambda p: p.transition[1].__setitem__((4, 0), p.transition[1][4, 0] + 0.2), r"transition\[1\] row 4"),
        (lambda p: p.transition[0].__setitem__((1, slice(None)), [1.2, -0.2]), r"transition\[0\] has negative"),
        (lambda p: p.emissions[1].weights.__setitem__((0, 0), 2.0), r"emissions\[1\].weights"),
        (lambda p: p.emissions[0].sigma.__setitem__((0, 0, 0), -1.0), r"emissions\[0\].sigma"),
    ],
)
def test_params_validate_rejects(mutate, match):
    p = _valid_params()
    mutate(p)
    with pytest.raises(ValidationError, match=match):
        p.validate()


def test_params_shape_errors():
    p = _valid_params()
    s = p.structure
    with pytest.raises(ValueError, match="pi_0.shape"):
        CHMMParams(s, p.pi_0[:-1], p.transition, p.emissions)
    with pytest.raises(ValueError, match="len\\(transition\\)"):
        CHMMParams(s, p.pi_0, p.transition[:1], p.emissions)
    with pytest.raises(ValueError, match=r"transition\[1\].shape"):
        CHMMParams(s, p.pi_0, [p.transition[0], p.transition[1][:, :2]], p.emissions)
    with pytest.raises(ValueError, match="len\\(emissions\\)"):
        CHMMParams(s, p.pi_0, p.transition, p.emissions[:1])
    with pytest.raises(ValueError, match="M_c"):
        CHMMParams(s, p.pi_0, p.transition, [p.emissions[1], p.emissions[1]])


def test_joint_state_guard():
    structure = LSIMStructure((65, 64), (1, 1), (1, 1))
    with pytest.raises(EquivalentModelTooLargeError):
        CHMMParams(structure, np.zeros(structure.S), [], [])


def test_structure_mismatch_rejected():
    p = _valid_params()
    other = LSIMStructure((3, 2), (1, 1), (2, 1))
    Y = np.zeros((2, 5))
    with pytest.raises(ValidationError, match="structure does not match"):
        forward_backward_chmm(other, p, Y)
    with pytest.raises(ValidationError, match="structure does not match"):
        em_chmm(other, p, [Y], 1)


def test_public_exports():
    for name in (
        "CHMMParams", "CHMMPosterior", "CHMMJointModel", "lsim_to_chmm", "chmm_joint_model",
        "chmm_exact_forward_backward", "forward_backward_chmm", "viterbi_chmm", "compute_chmm_aic_bic",
        "em_chmm", "initialize_chmm_params", "simulate_chmm",
    ):
        assert name in chmm_lsim.__all__ and hasattr(chmm_lsim, name)


# ---------------------------------------------------------------------------
# initialization (scikit-learn)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_initialize_chmm_params(sigma_diag):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(29)
    structure = LSIMStructure((2, 3), (1, 2), (1, 1))
    truth = _random_chmm(rng, structure, sigma_diag)
    Y_trials = [simulate_chmm(structure, truth, 120, seed=s)[0] for s in range(2)]
    init = chmm_lsim.initialize_chmm_params(structure, Y_trials, sigma_diag=sigma_diag, seed=4)
    ref = chmm_lsim.initialize_lsim_params(structure, Y_trials, sigma_diag=sigma_diag, seed=4)
    np.testing.assert_array_equal(init.pi_0, np.r_[np.full(2, 0.5), np.full(3, 1 / 3)])
    for z, m in enumerate(structure.channel_state_num):
        np.testing.assert_array_equal(init.transition[z], np.full((6, m), 1.0 / m))
    for a, b in zip(init.emissions, ref.emissions):
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.sigma, b.sigma)
        np.testing.assert_array_equal(a.weights, b.weights)
    result = em_chmm(structure, init, Y_trials, 5)
    assert np.all(np.isfinite(result.loglik_trace))
