"""Unit tests for chmm_lsim.params (ChannelGMM, LSIMParams, packed <-> per-channel,
cells <-> per-channel conversions) (P2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure


def _make_diag_params() -> LSIMParams:
    """C=2, M=(2,3), d=(1,2), K=(1,2), diagonal covariance -- every constraint
    (pi_0 blocks, transition block rows, coupling columns, mixture weights)
    exactly satisfied."""
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 2))
    pi_0 = np.array([0.4, 0.6, 0.2, 0.3, 0.5])
    transition = np.array(
        [
            [0.5, 0.5, 0.2, 0.3, 0.5],
            [0.3, 0.7, 0.4, 0.4, 0.2],
            [0.6, 0.4, 0.1, 0.4, 0.5],
            [0.2, 0.8, 0.3, 0.3, 0.4],
            [0.9, 0.1, 0.2, 0.2, 0.6],
        ]
    )
    coupling = np.array([[0.7, 0.4], [0.3, 0.6]])
    gmm0 = ChannelGMM(
        weights=np.array([[1.0], [1.0]]),
        mu=np.zeros((1, 2, 1)),
        sigma=np.ones((1, 2, 1)),
        diag=True,
    )
    gmm1 = ChannelGMM(
        weights=np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]),
        mu=np.zeros((2, 3, 2)),
        sigma=np.ones((2, 3, 2)),
        diag=True,
    )
    return LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])


def test_valid_params_construct_and_validate():
    params = _make_diag_params()
    params.validate()  # must not raise
    assert params.sigma_diag is True


def test_from_packed_to_packed_round_trip():
    params = _make_diag_params()
    packed = params.to_packed()
    roundtripped = LSIMParams.from_packed(
        params.structure,
        pi_0=packed["pi_0"],
        transition=packed["transitions_matrices"],
        coupling=packed["coupling_tetha_IM"],
        P_all=packed["P_all"],
        mu_all=packed["mu_all"],
        sigma_all=packed["sigma_all"],
        sigma_diag=packed["sigma_diag"],
    )
    np.testing.assert_array_equal(roundtripped.pi_0, params.pi_0)
    np.testing.assert_array_equal(roundtripped.transition, params.transition)
    np.testing.assert_array_equal(roundtripped.coupling, params.coupling)
    for a, b in zip(roundtripped.emissions, params.emissions):
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.sigma, b.sigma)


def test_to_packed_pads_to_max_M_and_max_K():
    params = _make_diag_params()
    packed = params.to_packed()
    s = params.structure
    assert packed["P_all"].shape == (s.C, s.max_M, s.max_K)
    assert packed["mu_all"].shape == (s.D, s.max_M, s.max_K)
    assert packed["sigma_all"].shape == (s.D, s.max_M, s.max_K)
    # channel 0 has only K_c=1 < max_K=2: P_all/mu_all pad with zero, sigma_all with one
    # (lsim_pack_para.m:50-51 vs :64,67 -- a padded covariance entry is not a real 0 variance)
    np.testing.assert_array_equal(packed["P_all"][0, :, 1], 0.0)
    np.testing.assert_array_equal(packed["mu_all"][0, :, 1], 0.0)
    np.testing.assert_array_equal(packed["sigma_all"][0, :, 1], 1.0)


def test_from_cells_to_cells_round_trip():
    params = _make_diag_params()
    cells = params.to_cells()
    roundtripped = LSIMParams.from_cells(
        params.structure,
        pi_0_cells=cells["pi_0_cells"],
        transition_cells=cells["transition_cells"],
        coupling=cells["coupling"],
        emissions=cells["emissions"],
    )
    np.testing.assert_array_equal(roundtripped.pi_0, params.pi_0)
    np.testing.assert_array_equal(roundtripped.transition, params.transition)


def test_copy_is_deep():
    params = _make_diag_params()
    copied = params.copy()
    copied.pi_0[0] = 999.0
    copied.emissions[0].weights[0, 0] = 999.0
    assert params.pi_0[0] != 999.0
    assert params.emissions[0].weights[0, 0] != 999.0


def test_from_packed_to_packed_exact_round_trip_against_real_fixture(fixtures_dir, posterior_fixture_id):
    """packed -> from_packed -> to_packed must reproduce the fixture's own
    packed arrays bit-for-bit, padding included -- this is what actually
    exercises `to_packed`'s ones-vs-zeros fill, which a synthetic (already
    padding-free) params object cannot."""
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
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
    packed = params.to_packed()
    np.testing.assert_array_equal(packed["pi_0"], fx.pi_0)
    np.testing.assert_array_equal(packed["transitions_matrices"], fx.transitions_matrices)
    np.testing.assert_array_equal(packed["coupling_tetha_IM"], fx.coupling_tetha_IM)
    np.testing.assert_array_equal(packed["P_all"], fx.P_all)
    np.testing.assert_array_equal(packed["mu_all"], fx.mu_all)
    np.testing.assert_array_equal(packed["sigma_all"], fx.sigma_all)


def test_channel_gmm_mu_shape_mismatch_raises():
    with pytest.raises(ValueError, match="mu.shape"):
        ChannelGMM(weights=np.ones((2, 1)), mu=np.zeros((1, 3, 1)), sigma=np.ones((1, 2, 1)), diag=True)


def test_channel_gmm_sigma_shape_mismatch_raises():
    with pytest.raises(ValueError, match="sigma.shape"):
        ChannelGMM(weights=np.ones((2, 1)), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 2, 2)), diag=True)


def test_channel_gmm_full_sigma_shape():
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=np.zeros((3, 2, 1)), sigma=np.zeros((3, 3, 2, 1)), diag=False)
    assert gmm.d_c == 3
    assert gmm.M_c == 2
    assert gmm.K_c == 1


def test_lsim_params_pi0_shape_mismatch_raises():
    params = _make_diag_params()
    with pytest.raises(ValueError, match="pi_0.shape"):
        LSIMParams(structure=params.structure, pi_0=params.pi_0[:-1], transition=params.transition, coupling=params.coupling, emissions=params.emissions)


def test_lsim_params_wrong_emissions_count_raises():
    params = _make_diag_params()
    with pytest.raises(ValueError, match="len\\(emissions\\)"):
        LSIMParams(structure=params.structure, pi_0=params.pi_0, transition=params.transition, coupling=params.coupling, emissions=params.emissions[:1])


def test_lsim_params_emission_M_c_mismatch_raises():
    structure = LSIMStructure((2, 3), (1, 2), (1, 2))
    bad_gmm0 = ChannelGMM(weights=np.ones((3, 1)), mu=np.zeros((1, 3, 1)), sigma=np.ones((1, 3, 1)), diag=True)
    gmm1 = ChannelGMM(weights=np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]), mu=np.zeros((2, 3, 2)), sigma=np.ones((2, 3, 2)), diag=True)
    with pytest.raises(ValueError, match="M_c"):
        LSIMParams(structure=structure, pi_0=np.zeros(5), transition=np.zeros((5, 5)), coupling=np.eye(2), emissions=[bad_gmm0, gmm1])


def test_lsim_params_mixed_diag_and_full_raises():
    structure = LSIMStructure((2, 2), (1, 1), (1, 1))
    diag_gmm = ChannelGMM(weights=np.ones((2, 1)), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 2, 1)), diag=True)
    full_gmm = ChannelGMM(weights=np.ones((2, 1)), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 1, 2, 1)), diag=False)
    with pytest.raises(ValueError, match="diagonal-vs-full"):
        LSIMParams(
            structure=structure,
            pi_0=np.array([0.5, 0.5, 0.5, 0.5]),
            transition=np.full((4, 4), 0.25),
            coupling=np.array([[0.5, 0.5], [0.5, 0.5]]),
            emissions=[diag_gmm, full_gmm],
        )
