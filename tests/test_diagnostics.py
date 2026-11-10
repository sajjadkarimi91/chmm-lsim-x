"""P6 gate: `diagnostics.py` -- `joint_log_probability` (`common function/
compute_joint_state_observation_prob.m`, from `LSIMParams` directly), plus
the reusable marginalization/total-variation helpers Layer-B verification
needs (`tests/test_layer_b.py` is where those get exercised together with
`equivalent.py` and `forward_backward_sed`).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_p6_fixture  # noqa: E402

from chmm_lsim.models.lsim.diagnostics import joint_log_probability
from chmm_lsim.core.diagnostics import marginalize_joint_to_channels, per_channel_total_variation, total_variation_distance
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.equivalent import build_equivalent_hmm, exact_posterior
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError


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


def test_joint_log_probability_matches_manual_calculation():
    structure, params = _small_coupled_model()
    rng = np.random.default_rng(3)
    T = 5
    Y = rng.normal(size=(2, T))
    X = rng.integers(0, 2, size=(2, T))

    B, _ = compute_emissions(structure, params, Y)
    lp = np.log(params.pi_0[0 + X[0, 0]]) + np.log(B[0 + X[0, 0], 0])
    lp += np.log(params.pi_0[2 + X[1, 0]]) + np.log(B[2 + X[1, 0], 0])
    for t in range(1, T):
        prev0, prev1 = X[0, t - 1], X[1, t - 1]
        for c, off in [(0, 0), (1, 2)]:
            nxt = off + X[c, t]
            p = params.coupling[0, c] * params.transition[0 + prev0, nxt] + params.coupling[1, c] * params.transition[2 + prev1, nxt]
            lp += np.log(p) + np.log(B[nxt, t])

    ported = joint_log_probability(structure, params, Y, X)
    np.testing.assert_allclose(lp, ported, rtol=1e-12)


def test_joint_log_probability_matches_matlab_fixture(fixtures_dir):
    fx = load_p6_fixture(fixtures_dir)
    structure = LSIMStructure(tuple(int(x) for x in fx.channel_num_states), tuple(int(x) for x in fx.dim_observation), tuple(int(x) for x in fx.num_gmm_component))
    params = LSIMParams.from_packed(
        structure, pi_0=fx.pi_0_packed, transition=fx.transition_packed, coupling=fx.coupling_packed,
        P_all=fx.P_all_packed, mu_all=fx.mu_all_packed, sigma_all=fx.sigma_all_packed, sigma_diag=fx.sigma_diag,
    )
    params.validate()

    lp_star = joint_log_probability(structure, params, fx.Y, fx.X_star)
    np.testing.assert_allclose(lp_star, fx.log_joint_prob_viterbi_path, rtol=1e-8)

    lp_alt = joint_log_probability(structure, params, fx.Y, fx.X_alt)
    np.testing.assert_allclose(lp_alt, fx.log_joint_prob_alt_path, rtol=1e-8)


def test_impossible_path_returns_minus_inf_not_nan():
    structure, params = _small_coupled_model()
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(2, 4))
    X = np.zeros((2, 4), dtype=np.int64)

    params_bad = params.copy()
    params_bad.pi_0 = params_bad.pi_0.copy()
    params_bad.pi_0[0] = 0.0  # X[0,0]=0 is now structurally impossible

    # deliberately-invalid pi_0 block (no longer sums to 1) -- validate=False,
    # since this is testing -inf propagation through a broken *model*, not
    # asking joint_log_probability to accept a valid one.
    lp = joint_log_probability(structure, params_bad, Y, X, validate=False)
    assert lp == -np.inf
    assert not np.isnan(lp)


def test_impossible_transition_returns_minus_inf():
    structure, params = _small_coupled_model()
    params_bad = params.copy()
    # decouple first, so channel 0's *only* transition contribution is its own
    # self-block -- otherwise channel 1's nonzero contribution (via coupling)
    # would keep p(x_t^0=1|prev=0) > 0 even with the self-block entry zeroed.
    params_bad.coupling = np.eye(2)
    params_bad.transition = params_bad.transition.copy()
    params_bad.transition[0:2, 0:2] = [[1.0, 0.0], [0.2, 0.8]]  # (state0->state1 in block(0,0)) impossible

    Y = np.random.default_rng(3).normal(size=(2, 3))
    X = np.zeros((2, 3), dtype=np.int64)
    X[0, :] = [0, 1, 0]  # forces the impossible 0->1 transition at t=1

    lp = joint_log_probability(structure, params_bad, Y, X)
    assert lp == -np.inf


def test_hidden_path_wrong_shape_raises():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(0).normal(size=(2, 5))
    X = np.zeros((2, 4), dtype=np.int64)  # T mismatch
    with pytest.raises(ValidationError, match="shape"):
        joint_log_probability(structure, params, Y, X)


def test_hidden_path_out_of_range_state_raises():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(0).normal(size=(2, 3))
    X = np.zeros((2, 3), dtype=np.int64)
    X[0, 0] = 2  # channel 0 only has states {0,1}
    with pytest.raises(ValidationError, match="outside"):
        joint_log_probability(structure, params, Y, X)


def test_marginalize_joint_to_channels_shape_and_block_sums():
    structure, params = _small_coupled_model()
    equiv = build_equivalent_hmm(structure, params)
    Y = np.random.default_rng(4).normal(size=(2, 5))
    gamma, _ = exact_posterior(equiv, Y)

    marg = marginalize_joint_to_channels(equiv.dims, gamma)
    assert marg.shape == (structure.S, 5)
    for c in range(structure.C):
        block_sum = marg[structure.state_slice(c)].sum(axis=0)
        np.testing.assert_allclose(block_sum, 1.0, atol=1e-10)


def test_marginalize_rejects_dims_j_mismatch():
    with pytest.raises(ValidationError, match="J"):
        marginalize_joint_to_channels((2, 3), np.zeros((5, 4)))


def test_total_variation_distance_properties():
    p = np.array([[0.5, 0.5], [0.3, 0.7]]).T  # (2,2): 2 outcomes, 2 timesteps
    q = p.copy()
    np.testing.assert_allclose(total_variation_distance(p, q), 0.0)

    q2 = np.array([[1.0, 0.0], [0.3, 0.7]]).T
    tv = total_variation_distance(p, q2)
    np.testing.assert_allclose(tv[0], 0.5)  # |0.5-1.0|/2 + |0.5-0.0|/2 = 0.5
    np.testing.assert_allclose(tv[1], 0.0)

    # a valid TV distance is always in [0,1]
    assert np.all((tv >= 0) & (tv <= 1))


def test_total_variation_shape_mismatch_raises():
    with pytest.raises(ValidationError, match="shape"):
        total_variation_distance(np.zeros((2, 3)), np.zeros((3, 3)))


def test_per_channel_total_variation_self_distance_is_zero():
    structure, params = _small_coupled_model()
    equiv = build_equivalent_hmm(structure, params)
    Y = np.random.default_rng(5).normal(size=(2, 6))
    gamma, _ = exact_posterior(equiv, Y)
    marg = marginalize_joint_to_channels(equiv.dims, gamma)

    tv = per_channel_total_variation(structure, marg, marg)
    np.testing.assert_allclose(tv, 0.0, atol=1e-12)
    assert tv.shape == (structure.C, 6)


def test_full_covariance_supported():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    mu = np.stack([np.zeros(2), np.full(2, 5.0)], axis=1)[:, :, None]
    sigma = np.stack([np.eye(2), np.eye(2)], axis=-1)[:, :, :, None]
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=mu, sigma=sigma, diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.array([[0.9, 0.1], [0.1, 0.9]]), coupling=np.array([[1.0]]), emissions=[gmm])
    params.validate()

    Y = np.random.default_rng(6).normal(size=(2, 5))
    X = np.zeros((1, 5), dtype=np.int64)
    lp = joint_log_probability(structure, params, Y, X)
    assert np.isfinite(lp)


def test_nan_marginalized_dimension_does_not_break_joint_log_probability():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(0).normal(size=(2, 4))
    Y[0, 1] = np.nan
    X = np.zeros((2, 4), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lp = joint_log_probability(structure, params, Y, X)
    assert np.isfinite(lp)


# --------------------------------------------------------------------------
# X/Y dtype and shape: a floating X previously escaped as a raw IndexError;
# an integer Y was previously silently accepted.
# --------------------------------------------------------------------------

def test_float_x_rejected_not_raw_indexerror():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(0).normal(size=(2, 4))
    X = np.zeros((2, 4), dtype=np.float64)
    with pytest.raises(ValidationError, match="dtype"):
        joint_log_probability(structure, params, Y, X)


def test_integer_y_rejected():
    structure, params = _small_coupled_model()
    Y = np.zeros((2, 4), dtype=np.int64)
    X = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ValidationError, match="dtype"):
        joint_log_probability(structure, params, Y, X)


def test_x_wrong_ndim_rejected():
    structure, params = _small_coupled_model()
    Y = np.random.default_rng(0).normal(size=(2, 4))
    X = np.zeros(4, dtype=np.int64)  # 1-D, not (C,T)
    with pytest.raises(ValidationError, match="2-D"):
        joint_log_probability(structure, params, Y, X)


def test_structure_mismatch_rejected():
    structure, params = _small_coupled_model()
    other = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(2, 1), num_gmm_component=(1, 1))
    Y = np.random.default_rng(0).normal(size=(2, 4))
    X = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ValidationError, match="structure"):
        joint_log_probability(other, params, Y, X)


@pytest.mark.parametrize(
    "bad_Y,match",
    [
        (np.array(5.0), "2-D"),
        (np.zeros((3, 4)), "structure.D"),
        (np.zeros((2, 4), dtype=np.int64), "dtype"),
        (np.array([1.0, np.inf, 0.5, 0.2, 0.1, 0.3]).reshape(2, 3), "Inf"),
        (np.zeros((2, 0)), "T"),
    ],
)
def test_joint_log_probability_rejects_malformed_observations(bad_Y, match):
    structure, params = _small_coupled_model()
    T = bad_Y.shape[1] if bad_Y.ndim == 2 else 4
    X = np.zeros((2, max(T, 1)), dtype=np.int64)
    with pytest.raises(ValidationError, match=match):
        joint_log_probability(structure, params, bad_Y, X)
