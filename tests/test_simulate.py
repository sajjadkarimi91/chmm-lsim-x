"""P6 gate: `simulate.py` (`chmm/generate_chmm_time_series.m` + `select_p`/
`gmm_gen`), from `LSIMParams` directly.

No cross-language sample identity is claimed (see `simulate.py`'s own
docstring for why) -- what's tested here is: seeded reproducibility, shapes,
state ranges, the transition law's own probability-sum invariant, and that a
large simulated sample's empirical frequencies land close to the model's own
stationary/marginal expectations (a statistical check, not exact equality).
"""

from __future__ import annotations

import numpy as np
import pytest

from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.simulate import select_p, simulate_lsim
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError


def _heterogeneous_model():
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 2))
    pi_0 = np.array([0.5, 0.5, 0.3, 0.3, 0.4])
    transition = np.zeros((5, 5))
    rng = np.random.default_rng(1)
    transition[0:2, 0:2] = rng.dirichlet(np.ones(2), 2)
    transition[0:2, 2:5] = rng.dirichlet(np.ones(3), 2)
    transition[2:5, 0:2] = rng.dirichlet(np.ones(2), 3)
    transition[2:5, 2:5] = rng.dirichlet(np.ones(3), 3)
    coupling = np.array([[0.6, 0.4], [0.4, 0.6]])
    gmm0 = ChannelGMM(weights=np.ones((2, 1)), mu=rng.normal(size=(1, 2, 1)) * 5, sigma=np.ones((1, 2, 1)), diag=True)
    gmm1 = ChannelGMM(weights=np.full((3, 2), 0.5), mu=rng.normal(size=(2, 3, 2)) * 5, sigma=np.ones((2, 3, 2)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])
    params.validate()
    return structure, params


def test_select_p_respects_explicit_generator_not_global_state():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    p = np.array([0.2, 0.3, 0.5])
    draws_a = [select_p(rng_a, p) for _ in range(20)]
    draws_b = [select_p(rng_b, p) for _ in range(20)]
    assert draws_a == draws_b


def test_select_p_never_returns_out_of_range_even_at_the_boundary():
    p = np.array([0.5, 0.5])
    # a Generator whose .random() always returns just under 1.0 -- the edge
    # case the defensive clip in select_p exists for
    class _AlwaysNearOne:
        def random(self):
            return 1.0 - 1e-16

    idx = select_p(_AlwaysNearOne(), p)
    assert 0 <= idx < len(p)


def test_simulate_shapes_and_state_ranges():
    structure, params = _heterogeneous_model()
    Y, X = simulate_lsim(structure, params, 30, seed=0)
    assert Y.shape == (structure.D, 30)
    assert X.shape == (structure.C, 30)
    for c in range(structure.C):
        assert np.all(X[c] >= 0)
        assert np.all(X[c] < structure.channel_state_num[c])


def test_same_seed_reproduces_observations_and_states_exactly():
    structure, params = _heterogeneous_model()
    Y1, X1 = simulate_lsim(structure, params, 25, seed=7)
    Y2, X2 = simulate_lsim(structure, params, 25, seed=7)
    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(X1, X2)


def test_different_seed_gives_different_output():
    structure, params = _heterogeneous_model()
    Y1, X1 = simulate_lsim(structure, params, 25, seed=7)
    Y2, X2 = simulate_lsim(structure, params, 25, seed=8)
    assert not np.array_equal(Y1, Y2)
    assert not np.array_equal(X1, X2)


def test_c1_reduction_runs_and_matches_a_plain_single_chain():
    structure = LSIMStructure(channel_state_num=(3,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi_0 = np.array([0.2, 0.3, 0.5])
    transition = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((3, 1)), mu=np.array([0.0, 5.0, 10.0]).reshape(1, 3, 1), sigma=np.ones((1, 3, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()

    Y, X = simulate_lsim(structure, params, 40, seed=3)
    assert Y.shape == (1, 40)
    assert X.shape == (1, 40)
    assert np.all((X >= 0) & (X < 3))


def test_t_must_be_at_least_one():
    structure, params = _heterogeneous_model()
    with pytest.raises(ValidationError, match="T"):
        simulate_lsim(structure, params, 0, seed=0)


def test_invalid_params_rejected_before_simulating():
    structure, params = _heterogeneous_model()
    params.pi_0 = params.pi_0.copy()
    params.pi_0[0] = -1.0  # negative probability
    with pytest.raises(ValidationError):
        simulate_lsim(structure, params, 5, seed=0)


def test_empirical_frequencies_approach_pi0_at_t0_over_many_trials():
    """Not exact equality (this is a statistical claim): repeat the T=1 draw
    many times with different seeds and check the empirical state
    frequencies at t=0 land close to pi_0 -- select_p's own distribution,
    exercised the same way generate_chmm_time_series.m's does it."""
    structure, params = _heterogeneous_model()
    n_trials = 4000
    counts = np.zeros(structure.channel_state_num[0])
    for seed in range(n_trials):
        _, X = simulate_lsim(structure, params, 1, seed=seed)
        counts[X[0, 0]] += 1
    empirical = counts / n_trials
    true_pi0 = params.pi_0[structure.state_slice(0)]
    # measured: with n=4000 draws from a 2-outcome distribution, the binomial
    # standard error is well under 0.01 for probabilities in [0.3,0.7] -- 0.05
    # carries comfortable margin without making this test flaky.
    np.testing.assert_allclose(empirical, true_pi0, atol=0.05)


def test_full_covariance_emissions_supported():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    pi_0 = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    coupling = np.array([[1.0]])
    mu = np.stack([np.zeros(2), np.full(2, 5.0)], axis=1)[:, :, None]
    sigma = np.stack([np.eye(2) * 0.1, np.eye(2) * 0.1], axis=-1)[:, :, :, None]
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=mu, sigma=sigma, diag=False)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()

    Y, X = simulate_lsim(structure, params, 200, seed=0)
    # low noise (sigma=0.1I) relative to the mu separation (5.0): every sample
    # should land near its own state's mean -- a coarse sanity check that the
    # full-covariance sampling path actually uses the right mean/cov, not a
    # statistical tolerance calibration.
    for t in range(200):
        expected_mu = mu[:, X[0, t], 0]
        assert np.linalg.norm(Y[:, t] - expected_mu) < 2.0


def test_rank_one_singular_covariance_does_not_crash():
    """A rank-1 covariance (one nonzero eigenvalue, the rest exactly 0) is
    numerically singular PSD -- accepted by `params.validate(matlab_compat=
    True)` (§0.4.10) but fatal to `np.linalg.cholesky`, which requires strict
    positive-definiteness. `_symmetric_sqrt`'s eigendecomposition handles it
    directly; every sample must land exactly on the rank-1 subspace (no noise
    at all in the zero-eigenvalue direction)."""
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(2,), num_gmm_component=(1,))
    v = np.array([3.0, 4.0])  # not unit length, exercises the eigenvalue scale too
    sigma_rank1 = np.outer(v, v)  # rank-1 PSD: eigenvalues {25, 0}
    assert np.linalg.matrix_rank(sigma_rank1) == 1
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(sigma_rank1)  # confirms this case would crash without the fix

    mu = np.zeros((2, 1, 1))
    gmm = ChannelGMM(weights=np.ones((1, 1)), mu=mu, sigma=sigma_rank1[:, :, None, None], diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.array([[1.0]]), emissions=[gmm])
    params.validate()  # matlab_compat=True default: numerically singular PSD is accepted

    Y, X = simulate_lsim(structure, params, 50, seed=0)
    assert not np.any(np.isnan(Y))
    # every sample lies on the span of v: the component orthogonal to v must be ~0
    v_hat = v / np.linalg.norm(v)
    orthogonal_component = Y - np.outer(v_hat, v_hat @ Y)
    np.testing.assert_allclose(orthogonal_component, 0.0, atol=1e-6)  # eigendecomposition floating-point noise


def test_structure_mismatch_rejected():
    structure, params = _heterogeneous_model()
    other_structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 1))  # K differs
    with pytest.raises(ValidationError, match="structure"):
        simulate_lsim(other_structure, params, 5, seed=0)
