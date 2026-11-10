"""P6 gate: `viterbi.py` -- the approximate per-channel algorithm actually in
`viterbi_chmm.m` (`viterbi_lsim`), and the independent exact joint-state
Viterbi built on `equivalent.py` (`viterbi_exact`).

`viterbi_exact` is checked against brute-force path enumeration (it claims
to be exact, so it must match exactly). `viterbi_lsim` is checked against a
real, seeded MATLAB run (`tests/fixtures/p6_fixture.npz`) for both the chosen
states *and* `log_score`'s exact numeric value -- and, for `C=1` specifically,
against `viterbi_exact` too: with no other channels to approximate away,
`viterbi_lsim` must choose the *same states* as exact Viterbi (confirmed
below), even though its `log_score` is not numerically the same quantity
(see the module docstring: `viterbi_lsim` accumulates `log(b_c_ot_nc)`, the
forward pass's already-*rescaled* emission ratio, not raw `log(B)` -- for
`C=1` the two differ by exactly the model's own forward-pass `loglik`,
itself an exact, hand-checkable relationship, not a coincidence)."""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_p6_fixture  # noqa: E402

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.equivalent import build_equivalent_hmm
from chmm_lsim.models.lsim.inference.sed import forward_backward_sed
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError
from chmm_lsim.models.lsim.viterbi import viterbi_exact, viterbi_exact_from_equiv, viterbi_lsim


def _small_coupled_model():
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    pi0 = np.array([0.6, 0.4, 0.3, 0.7])
    transition = np.zeros((4, 4))
    transition[0:2, 0:2] = [[0.9, 0.1], [0.2, 0.8]]
    transition[0:2, 2:4] = [[0.5, 0.5], [0.5, 0.5]]
    transition[2:4, 0:2] = [[0.3, 0.7], [0.6, 0.4]]
    transition[2:4, 2:4] = [[0.7, 0.3], [0.4, 0.6]]
    coupling = np.array([[0.7, 0.3], [0.3, 0.7]])
    gmm0 = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([0.0, 5.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    gmm1 = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([-3.0, 3.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])
    params.validate()
    return structure, params


def _single_channel_model():
    structure = LSIMStructure(channel_state_num=(3,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi_0 = np.array([0.5, 0.3, 0.2])
    transition = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((3, 1)), mu=np.array([0.0, 5.0, 10.0]).reshape(1, 3, 1), sigma=np.ones((1, 3, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()
    return structure, params


# --------------------------------------------------------------------------
# viterbi_exact vs brute-force path enumeration.
# --------------------------------------------------------------------------

def test_viterbi_exact_matches_brute_force():
    structure, params = _small_coupled_model()
    equiv = build_equivalent_hmm(structure, params)
    rng = np.random.default_rng(0)
    T = 5
    Y = rng.normal(size=(2, T))

    result = viterbi_exact(structure, params, Y)

    joint_structure, joint_params = equiv.as_single_channel_params()
    B, _ = compute_emissions(joint_structure, joint_params, Y)

    best_score, best_path = -np.inf, None
    for path in itertools.product(range(equiv.J), repeat=T):
        lp = np.log(equiv.pi0[path[0]]) + np.log(B[path[0], 0])
        for t in range(1, T):
            lp += np.log(equiv.transition[path[t - 1], path[t]]) + np.log(B[path[t], t])
        if lp > best_score:
            best_score, best_path = lp, path

    expected_states = np.array(np.unravel_index(best_path, equiv.dims))
    np.testing.assert_allclose(result.log_score, best_score, rtol=1e-10)
    np.testing.assert_array_equal(result.states, expected_states)


def test_viterbi_exact_builds_its_own_equivalent_hmm_if_not_given():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(1).normal(size=(2, 4))
    r1 = viterbi_exact(structure, params, Y)
    equiv = build_equivalent_hmm(structure, params)
    r2 = viterbi_exact_from_equiv(equiv, Y)
    np.testing.assert_array_equal(r1.states, r2.states)
    np.testing.assert_allclose(r1.log_score, r2.log_score)


def test_viterbi_exact_no_longer_accepts_a_conflicting_equiv():
    """Regression test for a real bug: the old `viterbi_exact(structure,
    params, Y, equiv=equiv)` API silently ignored `params` whenever `equiv`
    was also given, even if `equiv` was built from a *different* model with
    the same `structure` -- `params_B` + `equiv_A` returned model A's path,
    not model B's, with no error. `viterbi_exact` no longer accepts `equiv`
    at all (a `TypeError` on the old keyword proves it), so `params` is
    always the only source of truth; a caller who wants to reuse a
    pre-built `equiv` must call `viterbi_exact_from_equiv`, which takes no
    `params` and therefore has no second source of truth to go stale."""
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi0 = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    coupling = np.array([[1.0]])

    gmm_a = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([0.0, 10.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    params_a = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm_a])
    params_a.validate()

    gmm_b = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([10.0, 0.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    params_b = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm_b])
    params_b.validate()

    Y = np.array([[0.0, 0.0, 0.0]])  # near state 0 under model A's emissions, near state 1 under model B's

    equiv_a = build_equivalent_hmm(structure, params_a)
    r_a = viterbi_exact_from_equiv(equiv_a, Y)
    r_b = viterbi_exact(structure, params_b, Y)

    # The two models genuinely disagree on this Y -- a meaningful regression
    # test, not a vacuous one.
    assert not np.array_equal(r_a.states, r_b.states)
    np.testing.assert_array_equal(r_a.states, np.zeros_like(r_a.states))
    np.testing.assert_array_equal(r_b.states, np.ones_like(r_b.states))

    with pytest.raises(TypeError):
        viterbi_exact(structure, params_b, Y, equiv=equiv_a)


# --------------------------------------------------------------------------
# viterbi_lsim vs a real, seeded MATLAB run (viterbi_chmm.m).
# --------------------------------------------------------------------------

def test_viterbi_lsim_matches_matlab_fixture(fixtures_dir):
    fx = load_p6_fixture(fixtures_dir)
    structure = LSIMStructure(tuple(int(x) for x in fx.channel_num_states), tuple(int(x) for x in fx.dim_observation), tuple(int(x) for x in fx.num_gmm_component))
    params = LSIMParams.from_packed(
        structure,
        pi_0=fx.pi_0_packed,
        transition=fx.transition_packed,
        coupling=fx.coupling_packed,
        P_all=fx.P_all_packed,
        mu_all=fx.mu_all_packed,
        sigma_all=fx.sigma_all_packed,
        sigma_diag=fx.sigma_diag,
    )
    params.validate()

    result = viterbi_lsim(structure, params, fx.Y)

    np.testing.assert_array_equal(result.states, fx.X_star)
    np.testing.assert_allclose(result.log_score, fx.P_star_model, rtol=1e-8)


# --------------------------------------------------------------------------
# C=1 reduction: same *states* as exact Viterbi (not the same log_score --
# see module docstring); the exact score relationship IS hand-checkable.
# --------------------------------------------------------------------------

def test_c1_reduction_same_states_as_exact():
    structure, params = _single_channel_model()
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(1, 8)) * 2 + 5

    r_approx = viterbi_lsim(structure, params, Y)
    r_exact = viterbi_exact(structure, params, Y)
    np.testing.assert_array_equal(r_approx.states, r_exact.states)


def test_c1_log_score_differs_from_exact_by_exactly_the_forward_pass_loglik():
    """viterbi_lsim accumulates log(b_c_ot_nc) = log(B) - log(P_ot_c_cond_past);
    for C=1 the per-timestep divisor is state-independent, so it factors out
    of the max entirely and the accumulated total is exactly the model's own
    forward-pass loglik (sed.py's `post.loglik`, with pi0_reg=0 to match
    viterbi_chmm.m's own unregularized pi_0) -- checked here as an exact
    numeric identity, not an approximation."""
    structure, params = _single_channel_model()
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(1, 8)) * 2 + 5

    r_approx = viterbi_lsim(structure, params, Y)
    r_exact = viterbi_exact(structure, params, Y)
    post = forward_backward_sed(structure, params, Y, FbConstants(variant="sed", pi0_reg=0.0))

    np.testing.assert_allclose(r_approx.log_score, r_exact.log_score - post.loglik, rtol=1e-10)


# --------------------------------------------------------------------------
# T=1, heterogeneous states, full covariance, NaN-marginalized emissions,
# zero-probability -> -inf without NaN.
# --------------------------------------------------------------------------

def test_t_equals_one():
    structure, params = _single_channel_model()
    Y = np.array([[5.2]])
    r = viterbi_lsim(structure, params, Y)
    assert r.states.shape == (1, 1)
    assert np.isfinite(r.log_score)

    r_exact = viterbi_exact(structure, params, Y)
    assert r_exact.states.shape == (1, 1)
    np.testing.assert_array_equal(r.states, r_exact.states)


def test_heterogeneous_states_and_dims():
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 2))
    rng = np.random.default_rng(2)
    pi0 = np.array([0.5, 0.5, 0.3, 0.3, 0.4])
    transition = np.zeros((5, 5))
    transition[0:2, 0:2] = rng.dirichlet(np.ones(2), 2)
    transition[0:2, 2:5] = rng.dirichlet(np.ones(3), 2)
    transition[2:5, 0:2] = rng.dirichlet(np.ones(2), 3)
    transition[2:5, 2:5] = rng.dirichlet(np.ones(3), 3)
    coupling = np.array([[0.6, 0.4], [0.4, 0.6]])
    gmm0 = ChannelGMM(weights=np.ones((2, 1)), mu=rng.normal(size=(1, 2, 1)) * 5, sigma=np.ones((1, 2, 1)), diag=True)
    gmm1 = ChannelGMM(weights=np.full((3, 2), 0.5), mu=rng.normal(size=(2, 3, 2)) * 5, sigma=np.ones((2, 3, 2)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])
    params.validate()

    Y = rng.normal(size=(3, 10))
    r = viterbi_lsim(structure, params, Y)
    assert r.states.shape == (2, 10)
    assert np.all(r.states[0] < 2) and np.all(r.states[1] < 3)
    assert np.isfinite(r.log_score)


def test_full_covariance():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    mu = np.stack([np.zeros(2), np.full(2, 5.0)], axis=1)[:, :, None]
    sigma = np.stack([np.eye(2), np.eye(2)], axis=-1)[:, :, :, None]
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=mu, sigma=sigma, diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.array([[0.9, 0.1], [0.1, 0.9]]), coupling=np.array([[1.0]]), emissions=[gmm])
    params.validate()

    Y = np.random.default_rng(3).normal(size=(2, 6))
    r = viterbi_lsim(structure, params, Y)
    r_exact = viterbi_exact(structure, params, Y)
    assert not np.isnan(r.log_score)
    np.testing.assert_array_equal(r.states, r_exact.states)


def test_nan_marginalized_emission_does_not_produce_nan():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(0).normal(size=(2, 6))
    Y[0, 2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = viterbi_lsim(structure, params, Y)
    assert not np.isnan(r.log_score)
    assert np.all(np.isfinite(r.states))


def test_zero_probability_transition_gives_finite_or_minus_inf_never_nan():
    structure, params = _small_coupled_model()
    params = params.copy()
    params.transition[0:2, 0:2] = [[1.0, 0.0], [0.2, 0.8]]
    Y = np.random.default_rng(0).normal(size=(2, 6))
    r = viterbi_lsim(structure, params, Y)
    assert not np.isnan(r.log_score)
    assert np.all(~np.isnan(r.states))


def test_invalid_validate_argument_rejected():
    structure, params = _single_channel_model()
    with pytest.raises(ValueError, match="validate"):
        viterbi_lsim(structure, params, np.zeros((1, 3)), validate="bogus")


# --------------------------------------------------------------------------
# Whole-block emission underflow: a clear ValidationError, never a silent
# NaN -- including the edge case _forward_pass's own NaN-loglik check misses
# (underflow only at the very last timestep, which gives -inf, not NaN, in
# loglik, but still 0/0=NaN in that timestep's own b_c_ot_nc).
# --------------------------------------------------------------------------

def test_underflow_only_at_final_timestep_raises_not_nan():
    structure, params = _single_channel_model()
    Y = np.array([[0.5, 0.6, 1000.0]])  # every state's density underflows to 0 only at t=2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValidationError, match=r"channel 0 at t=2"):
            viterbi_lsim(structure, params, Y)


def test_underflow_at_an_interior_timestep_also_raises_not_nan():
    """Also confirms the error names the exact channel/timestep -- before
    this correction round, an interior-timestep underflow was instead
    caught by `_forward_pass`'s own generic NaN check, whose message names
    neither."""
    structure, params = _single_channel_model()
    Y = np.array([[0.5, 1000.0, 0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValidationError, match=r"channel 0 at t=1"):
            viterbi_lsim(structure, params, Y)


def test_underflow_at_the_first_timestep_also_raises_not_nan():
    structure, params = _single_channel_model()
    Y = np.array([[1000.0, 0.5, 0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValidationError, match=r"channel 0 at t=0"):
            viterbi_lsim(structure, params, Y)


# --------------------------------------------------------------------------
# Structure/params mismatch, and validate_observations coverage
# (scalar/wrong-D/integer/Inf/T=0) for both viterbi_lsim and viterbi_exact.
# --------------------------------------------------------------------------

def test_viterbi_lsim_rejects_structure_mismatch():
    structure, params = _small_coupled_model()
    other = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(2, 1), num_gmm_component=(1, 1))  # d differs
    with pytest.raises(ValidationError, match="structure"):
        viterbi_lsim(other, params, np.zeros((2, 3)))


def test_viterbi_exact_rejects_structure_mismatch():
    structure, params = _small_coupled_model()
    other = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(2, 1), num_gmm_component=(1, 1))
    with pytest.raises(ValidationError, match="structure"):
        viterbi_exact(other, params, np.zeros((2, 3)))


@pytest.mark.parametrize(
    "bad_Y,match",
    [
        (np.array(5.0), "2-D"),
        (np.zeros((3, 4)), "structure.D"),
        (np.zeros((2, 4), dtype=np.int64), "dtype"),
        (np.array([[1.0, np.inf, 0.5, 0.2, 0.1, 0.3]]).reshape(2, 3), "Inf"),
        (np.zeros((2, 0)), "T"),
    ],
)
def test_viterbi_lsim_rejects_malformed_observations(bad_Y, match):
    structure, params = _small_coupled_model()
    with pytest.raises(ValidationError, match=match):
        viterbi_lsim(structure, params, bad_Y)


@pytest.mark.parametrize(
    "bad_Y,match",
    [
        (np.array(5.0), "2-D"),
        (np.zeros((3, 4)), "structure.D"),
        (np.zeros((2, 4), dtype=np.int64), "dtype"),
        (np.zeros((2, 0)), "T"),
    ],
)
def test_viterbi_exact_rejects_malformed_observations(bad_Y, match):
    structure, params = _small_coupled_model()
    with pytest.raises(ValidationError, match=match):
        viterbi_exact(structure, params, bad_Y)
