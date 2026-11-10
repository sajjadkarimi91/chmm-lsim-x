"""Negative-test table for chmm_lsim.validate (§2.1.1): every malformed-input
category must raise ValidationError, never pass silently or crash with a raw
exception (P2 gate: "validation raises on each malformed input in a
negative-test table")."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, validate_observations
from chmm_lsim.models.lsim.params import validate_params


def _make_diag_params() -> LSIMParams:
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
    gmm0 = ChannelGMM(weights=np.array([[1.0], [1.0]]), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 2, 1)), diag=True)
    gmm1 = ChannelGMM(
        weights=np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]),
        mu=np.zeros((2, 3, 2)),
        sigma=np.ones((2, 3, 2)),
        diag=True,
    )
    return LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])


def _make_full_params() -> LSIMParams:
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    pi_0 = np.array([0.4, 0.6])
    transition = np.array([[0.5, 0.5], [0.3, 0.7]])
    coupling = np.array([[1.0]])
    sigma = np.stack([np.eye(2), np.eye(2)], axis=-1)[:, :, :, None]  # (2,2,2,1), one PD matrix per state
    gmm0 = ChannelGMM(weights=np.array([[1.0], [1.0]]), mu=np.zeros((2, 2, 1)), sigma=sigma, diag=False)
    return LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm0])


def test_valid_diag_params_pass():
    validate_params(_make_diag_params())


def test_valid_full_params_pass():
    validate_params(_make_full_params())


def test_pi0_block_sum_not_one_raises():
    p = _make_diag_params()
    p.pi_0[0] = 0.9  # block [0.9, 0.6] now sums to 1.5, not 1
    with pytest.raises(ValidationError, match="pi_0 block"):
        validate_params(p)


def test_pi0_negative_raises():
    p = _make_diag_params()
    p.pi_0[0] = -0.1
    with pytest.raises(ValidationError, match="pi_0 has negative"):
        validate_params(p)


def test_pi0_nan_raises():
    p = _make_diag_params()
    p.pi_0[0] = np.nan
    with pytest.raises(ValidationError, match="pi_0 contains NaN"):
        validate_params(p)


def test_pi0_inf_raises():
    p = _make_diag_params()
    p.pi_0[0] = np.inf
    with pytest.raises(ValidationError, match="pi_0 contains Inf"):
        validate_params(p)


def test_transition_block_row_not_stochastic_raises():
    p = _make_diag_params()
    p.transition[0, 0] = 5.0  # channel-0 block, row 0 no longer sums to 1
    with pytest.raises(ValidationError, match=r"transition block \(c=0, zee=0\)"):
        validate_params(p)


def test_transition_negative_entry_raises_even_when_row_sums_to_one():
    p = _make_diag_params()
    p.transition[0, 0] = -0.1
    p.transition[0, 1] = 1.1  # row still sums to 1.0, but one entry is not a valid probability
    with pytest.raises(ValidationError, match="transition has negative"):
        validate_params(p)


def test_coupling_negative_entry_raises_even_when_column_sums_to_one():
    p = _make_diag_params()
    p.coupling[0, 0] = -0.1
    p.coupling[1, 0] = 1.1  # column still sums to 1.0
    with pytest.raises(ValidationError, match="coupling has negative"):
        validate_params(p)


def test_pi0_sum_rejected_at_rtol_zero_even_though_within_default_isclose_rtol():
    """1.000009 would pass np.isclose's *default* rtol=1e-5 even with a tight
    atol; the sum check must use rtol=0 so only atol sets the tolerance."""
    p = _make_diag_params()
    p.pi_0[0] = p.pi_0[0] + 9e-6
    with pytest.raises(ValidationError, match="pi_0 block"):
        validate_params(p)


def test_transition_nan_raises():
    p = _make_diag_params()
    p.transition[0, 0] = np.nan
    with pytest.raises(ValidationError, match="transition contains NaN"):
        validate_params(p)


def test_coupling_columns_not_stochastic_raises():
    p = _make_diag_params()
    p.coupling[0, 0] = 5.0
    with pytest.raises(ValidationError, match="coupling columns"):
        validate_params(p)


def test_coupling_inf_raises():
    p = _make_diag_params()
    p.coupling[0, 0] = np.inf
    with pytest.raises(ValidationError, match="coupling contains Inf"):
        validate_params(p)


def test_mixture_weights_not_summing_to_one_raises():
    p = _make_diag_params()
    p.emissions[1].weights[0, 0] = 0.9  # row was [0.5, 0.5] -> now [0.9, 0.5]
    with pytest.raises(ValidationError, match=r"emissions\[1\]\.weights rows"):
        validate_params(p)


def test_mixture_weights_negative_raises():
    p = _make_diag_params()
    p.emissions[1].weights[0, 0] = -0.5
    with pytest.raises(ValidationError, match=r"emissions\[1\]\.weights has negative"):
        validate_params(p)


def test_mixture_weights_nan_raises():
    p = _make_diag_params()
    p.emissions[0].weights[0, 0] = np.nan
    with pytest.raises(ValidationError, match=r"emissions\[0\]\.weights contains NaN"):
        validate_params(p)


def test_mu_inf_raises():
    p = _make_diag_params()
    p.emissions[0].mu[0, 0, 0] = np.inf
    with pytest.raises(ValidationError, match=r"emissions\[0\]\.mu contains Inf"):
        validate_params(p)


def test_diagonal_sigma_zero_raises():
    p = _make_diag_params()
    p.emissions[0].sigma[0, 0, 0] = 0.0
    with pytest.raises(ValidationError, match=r"emissions\[0\]\.sigma \(diagonal\) must be strictly positive"):
        validate_params(p)


def test_diagonal_sigma_negative_raises():
    p = _make_diag_params()
    p.emissions[0].sigma[0, 0, 0] = -1.0
    with pytest.raises(ValidationError, match=r"emissions\[0\]\.sigma \(diagonal\) must be strictly positive"):
        validate_params(p)


def test_diagonal_sigma_nan_raises():
    p = _make_diag_params()
    p.emissions[0].sigma[0, 0, 0] = np.nan
    with pytest.raises(ValidationError, match=r"emissions\[0\]\.sigma contains NaN"):
        validate_params(p)


def test_full_sigma_not_symmetric_raises():
    p = _make_full_params()
    p.emissions[0].sigma[0, 1, 0, 0] = 5.0  # break symmetry without breaking PD-ness of the diagonal
    with pytest.raises(ValidationError, match="is not symmetric"):
        validate_params(p)


def test_full_sigma_singular_psd_matlab_compat_true_tolerated():
    p = _make_full_params()
    # eigenvalues 0 and 2: rank-1, numerically singular, but PSD -- not indefinite
    # (indefinite means a genuinely negative eigenvalue, which this matrix does not have)
    p.emissions[0].sigma[:, :, 0, 0] = np.array([[1.0, 1.0], [1.0, 1.0]])
    validate_params(p, matlab_compat=True)  # must not raise


def test_full_sigma_doubly_singular_still_tolerated_under_matlab_compat_true():
    """Every eigenvalue at (numerically) zero is still PSD, just maximally
    degenerate -- there is no negative eigenvalue, so matlab_compat=True
    tolerates it (docs/PYTHON_PORT_PLAN.md §0.4.10's `eig(...)<=0` *count*
    threshold is the M-step's own repair decision, not this validator's)."""
    p = _make_full_params()
    p.emissions[0].sigma[:, :, 0, 0] = np.zeros((2, 2))  # both eigenvalues 0
    validate_params(p, matlab_compat=True)  # must not raise


def test_full_sigma_negative_eigenvalue_raises_regardless_of_matlab_compat():
    p = _make_full_params()
    p.emissions[0].sigma[:, :, 0, 0] = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    with pytest.raises(ValidationError, match="negative eigenvalue"):
        validate_params(p, matlab_compat=True)
    with pytest.raises(ValidationError, match="negative eigenvalue"):
        validate_params(p, matlab_compat=False)


def test_full_sigma_singular_psd_matlab_compat_false_raises():
    p = _make_full_params()
    p.emissions[0].sigma[:, :, 0, 0] = np.array([[1.0, 1.0], [1.0, 1.0]])  # PSD but not strictly PD
    with pytest.raises(ValidationError, match="matlab_compat=False requires strict positive-definiteness"):
        validate_params(p, matlab_compat=False)


def test_full_sigma_nan_raises():
    p = _make_full_params()
    p.emissions[0].sigma[0, 0, 0, 0] = np.nan
    with pytest.raises(ValidationError, match=r"emissions\[0\]\.sigma contains NaN"):
        validate_params(p)


# --------------------------------------------------------------------------
# validate_observations
# --------------------------------------------------------------------------

def _structure():
    return LSIMStructure((2, 3), (1, 2), (1, 2))


def test_valid_observations_pass():
    s = _structure()
    Y = np.random.default_rng(0).normal(size=(s.D, 10))
    validate_observations(s, Y)


def test_wrong_D_raises():
    s = _structure()
    Y = np.zeros((s.D + 1, 10))
    with pytest.raises(ValidationError, match="structure.D"):
        validate_observations(s, Y)


def test_wrong_ndim_raises():
    s = _structure()
    Y = np.zeros((s.D, 10, 1))
    with pytest.raises(ValidationError, match="expected 2-D"):
        validate_observations(s, Y)


def test_scalar_observation_raises_validation_error_not_indexerror():
    """0-D input has no shape[0] at all -- checking ndim before shape[0] is
    what turns this into a clean ValidationError instead of a raw IndexError."""
    s = _structure()
    Y = np.array(1.0)
    with pytest.raises(ValidationError, match="expected 2-D"):
        validate_observations(s, Y)


def test_integer_dtype_observations_rejected():
    s = _structure()
    Y = np.zeros((s.D, 5), dtype=np.int64)
    with pytest.raises(ValidationError, match="floating dtype"):
        validate_observations(s, Y)


def test_zero_timesteps_raises():
    s = _structure()
    Y = np.zeros((s.D, 0))
    with pytest.raises(ValidationError, match="at least 1 timestep"):
        validate_observations(s, Y)


def test_inf_in_observations_raises():
    s = _structure()
    Y = np.zeros((s.D, 5))
    Y[0, 0] = np.inf
    with pytest.raises(ValidationError, match=r"\+/-Inf"):
        validate_observations(s, Y)


def test_nan_in_observations_is_legal():
    s = _structure()
    Y = np.zeros((s.D, 5))
    Y[0, 0] = np.nan
    validate_observations(s, Y)  # must not raise


def test_fully_missing_channel_warns_but_does_not_raise():
    s = _structure()
    Y = np.zeros((s.D, 5))
    Y[s.dim_slice(0), :] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_observations(s, Y)
    assert any("channel 0 is NaN for every timestep" in str(w.message) for w in caught)
