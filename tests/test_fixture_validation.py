"""Planted-corruption negative tests for tests/fixtures/loader.py's validators.

Each test loads a real, valid fixture, mutates exactly one array or scalar on
the returned dataclass, and asserts validate_*_fixture raises
FixtureValidationError mentioning the corrupted field -- proving the checker
actually discriminates, not just that it exists. Complements
test_fixture_schema.py, which only ever sees valid fixtures and so cannot by
itself prove a validator would catch a bad one.
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import (  # noqa: E402
    P10A_SCENARIO_IDS,
    FixtureValidationError,
    load_em_trace_fixture,
    load_init_reference_fixture,
    load_p6_fixture,
    load_p10a_scheduling_fixture,
    load_posterior_fixture,
    validate_em_trace_fixture,
    validate_init_reference_fixture,
    validate_p6_fixture,
    validate_p10a_scheduling_fixture,
    validate_posterior_fixture,
)

POSTERIOR_WITH_DIAG = "c3_sparse_l2r__sed"
POSTERIOR_PLAIN = "c3_het_diag__sed"
POSTERIOR_MULTI_TRIAL = "c3_trials_unequal__sed"


def _load(fixtures_dir, fixture_id):
    return load_posterior_fixture(fixtures_dir, fixture_id)


# --------------------------------------------------------------------------
# Positive control: a freshly-loaded, unmutated fixture must validate clean.
# Without this, a test asserting "corruption raises" proves nothing -- the
# validator could be raising on everything, corrupted or not.
# --------------------------------------------------------------------------

def test_positive_control_unmutated_fixture_validates(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    validate_posterior_fixture(fx)  # must not raise
    fx2 = _load(fixtures_dir, POSTERIOR_PLAIN)
    validate_posterior_fixture(fx2)  # must not raise


def test_positive_control_nan_in_Y_is_allowed(fixtures_dir):
    """Y is the one array allowed to contain NaN (missing data by design);
    confirm the validator doesn't reject a fixture just for having NaN there,
    or every c2_diag_d1_nan-style fixture would fail for the wrong reason."""
    fx = _load(fixtures_dir, "c2_diag_d1_nan__sed")
    assert np.any(np.isnan(fx.trials[0].Y))
    validate_posterior_fixture(fx)  # must not raise despite the NaN in Y


# --------------------------------------------------------------------------
# Posterior fixture: packed parameters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["pi_0", "transitions_matrices", "coupling_tetha_IM", "P_all", "mu_all", "sigma_all"])
def test_packed_param_wrong_shape_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    setattr(fx, field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=field):
        validate_posterior_fixture(fx)


@pytest.mark.parametrize("field", ["pi_0", "transitions_matrices", "coupling_tetha_IM", "P_all", "mu_all", "sigma_all"])
def test_packed_param_nan_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    arr = getattr(fx, field).copy()
    arr.flat[0] = np.nan
    setattr(fx, field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_posterior_fixture(fx)


def test_packed_param_inf_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    arr = fx.P_all.copy()
    arr.flat[0] = np.inf
    fx.P_all = arr
    with pytest.raises(FixtureValidationError, match="P_all"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# Posterior fixture: per-trial outputs, including B_comp
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["alpha", "beta", "alpha_T", "b_c_ot_nc", "B", "B_comp"])
def test_trial_output_wrong_shape_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    setattr(fx.trials[0], field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=field):
        validate_posterior_fixture(fx)


@pytest.mark.parametrize("field", ["alpha", "beta", "alpha_T", "b_c_ot_nc", "B", "B_comp"])
def test_trial_output_nan_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    arr = getattr(fx.trials[0], field).copy()
    arr.flat[0] = np.nan
    setattr(fx.trials[0], field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_posterior_fixture(fx)


def test_trial_output_inf_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    arr = fx.trials[0].B_comp.copy()
    arr.flat[0] = np.inf
    fx.trials[0].B_comp = arr
    with pytest.raises(FixtureValidationError, match="B_comp"):
        validate_posterior_fixture(fx)


def test_trial_Y_nan_does_not_raise_but_inf_does(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    y = fx.trials[0].Y.copy()
    y.flat[0] = np.nan
    fx.trials[0].Y = y
    validate_posterior_fixture(fx)  # NaN in Y is fine

    y2 = fx.trials[0].Y.copy()
    y2.flat[0] = np.inf
    fx.trials[0].Y = y2
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_posterior_fixture(fx)  # Inf in Y is not


def test_P_O_model_nan_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.trials[0].P_O_model = float("nan")
    with pytest.raises(FixtureValidationError, match="P_O_model"):
        validate_posterior_fixture(fx)


def test_P_O_model_inf_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.trials[0].P_O_model = float("inf")
    with pytest.raises(FixtureValidationError, match="P_O_model"):
        validate_posterior_fixture(fx)


def test_multi_trial_second_trial_corruption_is_caught(fixtures_dir):
    """Corruption isolated to a trial other than trial 0 must still be caught
    -- proves the loop actually visits every trial, not just the first."""
    fx = _load(fixtures_dir, POSTERIOR_MULTI_TRIAL)
    assert fx.num_trials >= 2
    arr = fx.trials[1].alpha.copy()
    arr.flat[0] = np.nan
    fx.trials[1].alpha = arr
    with pytest.raises(FixtureValidationError, match="trial1"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# Posterior fixture: SED diagnostics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["f", "h", "delta", "Gz"])
def test_sed_diagnostic_wrong_shape_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    setattr(fx.sed_diagnostics, field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=f"diag_{field}"):
        validate_posterior_fixture(fx)


@pytest.mark.parametrize("field", ["f", "h", "delta", "Gz"])
def test_sed_diagnostic_nan_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    arr = getattr(fx.sed_diagnostics, field).copy()
    arr.flat[0] = np.nan
    setattr(fx.sed_diagnostics, field, arr)
    with pytest.raises(FixtureValidationError, match=f"diag_{field}"):
        validate_posterior_fixture(fx)


def test_sed_diagnostics_on_fast_variant_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    fx.variant = "fast"  # sed_diagnostics is not None, so this must be flagged
    with pytest.raises(FixtureValidationError, match="fast"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# Posterior fixture: malformed offsets/lengths must raise FixtureValidationError,
# never IndexError -- the specific requirement this correction pass targets.
# --------------------------------------------------------------------------

def test_malformed_state_numbers_index_raises_validation_error_not_indexerror(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.state_numbers_index = np.array([0])  # far too short for C+1
    with pytest.raises(FixtureValidationError, match="state_numbers_index"):
        validate_posterior_fixture(fx)


def test_empty_state_numbers_index_raises_validation_error_not_indexerror(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.state_numbers_index = np.array([], dtype=np.int64)
    with pytest.raises(FixtureValidationError):
        validate_posterior_fixture(fx)


def test_malformed_dimension_numbers_index_raises_validation_error(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.dimension_numbers_index = np.array([0, 1])  # wrong length for this C
    with pytest.raises(FixtureValidationError, match="dimension_numbers_index"):
        validate_posterior_fixture(fx)


def test_empty_length_observation_raises_validation_error_not_indexerror(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.length_observation = np.array([], dtype=np.int64)
    with pytest.raises(FixtureValidationError, match="length_observation"):
        validate_posterior_fixture(fx)


def test_wrong_length_length_observation_for_multi_trial_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_MULTI_TRIAL)
    fx.length_observation = fx.length_observation[:1]  # drop trials 1, 2
    with pytest.raises(FixtureValidationError, match="length_observation"):
        validate_posterior_fixture(fx)


def test_malformed_channel_num_states_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.channel_num_states = np.array([1])  # wrong length for this C
    with pytest.raises(FixtureValidationError, match="channel_num_states"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# Posterior fixture: row/column vectors. The prior validator used
# arr.reshape(-1).size, which accepts a (C,1) column or (1,C) row silently --
# these must now fail against SCHEMA.md's exact (C,)/(C+1,)/(num_trials,).
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_vector_field_as_column_vector_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    arr = getattr(fx, field)
    setattr(fx, field, arr.reshape(-1, 1))  # (C,1) instead of (C,)
    with pytest.raises(FixtureValidationError, match=field):
        validate_posterior_fixture(fx)


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_vector_field_as_row_vector_in_2d_raises(fixtures_dir, field):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    arr = getattr(fx, field)
    setattr(fx, field, arr.reshape(1, -1))  # (1,C) instead of (C,)
    with pytest.raises(FixtureValidationError, match=field):
        validate_posterior_fixture(fx)


def test_state_numbers_index_as_column_vector_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.state_numbers_index = fx.state_numbers_index.reshape(-1, 1)
    with pytest.raises(FixtureValidationError, match="state_numbers_index"):
        validate_posterior_fixture(fx)


def test_length_observation_as_column_vector_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_MULTI_TRIAL)
    fx.length_observation = fx.length_observation.reshape(-1, 1)
    with pytest.raises(FixtureValidationError, match="length_observation"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# Posterior fixture: inconsistent offsets -- right length, right endpoints,
# but the increments don't match the per-channel counts, or the offset
# doesn't start at 0. The prior validator only checked state_numbers_index[-1]
# == S; it would have missed all of these.
# --------------------------------------------------------------------------

def test_state_numbers_index_not_starting_at_zero_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    # shift every offset by +1: diffs (and hence channel_num_states) are
    # unchanged, isolating the "starts at 0" check from the diff check
    fx.state_numbers_index = fx.state_numbers_index + 1
    with pytest.raises(FixtureValidationError, match=r"state_numbers_index\[0\]"):
        validate_posterior_fixture(fx)


def test_dimension_numbers_index_not_starting_at_zero_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    fx.dimension_numbers_index = fx.dimension_numbers_index + 1
    with pytest.raises(FixtureValidationError, match=r"dimension_numbers_index\[0\]"):
        validate_posterior_fixture(fx)


def test_state_numbers_index_diff_mismatch_raises(fixtures_dir):
    """Same start (0), same end (S), same length -- but a different partition
    of S across channels than channel_num_states declares."""
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    assert fx.C >= 2
    rotated = np.roll(fx.channel_num_states, 1)
    assert not np.array_equal(rotated, fx.channel_num_states), "rotation must actually change the partition"
    fx.state_numbers_index = np.concatenate([[0], np.cumsum(rotated)])
    assert fx.state_numbers_index[-1] == fx.S  # end untouched, isolating the diff check
    with pytest.raises(FixtureValidationError, match="diff\\(state_numbers_index\\)"):
        validate_posterior_fixture(fx)


def test_dimension_numbers_index_diff_mismatch_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_PLAIN)
    assert fx.C >= 2
    rotated = np.roll(fx.dim_observation, 1)
    assert not np.array_equal(rotated, fx.dim_observation), "rotation must actually change the partition"
    fx.dimension_numbers_index = np.concatenate([[0], np.cumsum(rotated)])
    assert fx.dimension_numbers_index[-1] == fx.D
    with pytest.raises(FixtureValidationError, match="diff\\(dimension_numbers_index\\)"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# Posterior fixture: diag_t_samples exact shape and in-range requirement
# --------------------------------------------------------------------------

def test_diag_t_samples_as_column_vector_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    fx.sed_diagnostics.t_samples = fx.sed_diagnostics.t_samples.reshape(-1, 1)
    with pytest.raises(FixtureValidationError, match="diag_t_samples"):
        validate_posterior_fixture(fx)


def test_diag_t_samples_negative_index_raises(fixtures_dir):
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    fx.sed_diagnostics.t_samples = fx.sed_diagnostics.t_samples.copy()
    fx.sed_diagnostics.t_samples[0] = -1
    with pytest.raises(FixtureValidationError, match="diag_t_samples out of range"):
        validate_posterior_fixture(fx)


def test_diag_t_samples_too_large_for_trial_0_raises(fixtures_dir):
    """Valid range is [0, T0-2] (the reconstruction reads alpha[:,t] AND
    alpha[:,t+1]); T0-1 is one past the valid maximum and must be rejected."""
    fx = _load(fixtures_dir, POSTERIOR_WITH_DIAG)
    T0 = int(fx.length_observation[0])
    fx.sed_diagnostics.t_samples = fx.sed_diagnostics.t_samples.copy()
    fx.sed_diagnostics.t_samples[0] = T0 - 1
    with pytest.raises(FixtureValidationError, match="diag_t_samples out of range"):
        validate_posterior_fixture(fx)


# --------------------------------------------------------------------------
# EM-trace fixture
# --------------------------------------------------------------------------

def _load_em(fixtures_dir):
    return load_em_trace_fixture(fixtures_dir, "c3_het_diag")


def test_em_trace_positive_control_validates(fixtures_dir):
    fx = _load_em(fixtures_dir)
    validate_em_trace_fixture(fx)  # must not raise


@pytest.mark.parametrize("field", [
    "init_pi_0", "init_transitions_matrices", "init_coupling_tetha_IM",
    "init_P_all", "init_mu_all", "init_sigma_all",
    "effective_pi_0_iter1", "effective_transitions_matrices_iter1",
    "effective_coupling_tetha_IM_iter1", "effective_P_all_iter1",
    "effective_mu_all_iter1", "effective_sigma_all_iter1",
])
def test_em_trace_raw_or_effective_param_wrong_shape_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    setattr(fx, field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


@pytest.mark.parametrize("field", ["init_pi_0", "effective_pi_0_iter1", "effective_sigma_all_iter1"])
def test_em_trace_raw_or_effective_param_nan_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    arr = getattr(fx, field).copy()
    arr.flat[0] = np.nan
    setattr(fx, field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


def test_em_trace_Y_wrong_shape_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.Y = np.zeros((1, 1))
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_em_trace_fixture(fx)


def test_em_trace_Y_inf_raises_but_nan_does_not(fixtures_dir):
    fx = _load_em(fixtures_dir)
    y = fx.Y.copy()
    y.flat[0] = np.nan
    fx.Y = y
    validate_em_trace_fixture(fx)  # NaN in Y (missing data) is fine

    y2 = fx.Y.copy()
    y2.flat[0] = np.inf
    fx.Y = y2
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_em_trace_fixture(fx)


@pytest.mark.parametrize("field", [
    "params_out_pi0", "params_out_transition", "params_out_coupling",
    "params_out_Pall", "params_out_muall", "params_out_sigmaall",
])
def test_em_trace_iteration_output_wrong_shape_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    setattr(fx.iterations[0], field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


def test_em_trace_iteration_output_nan_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    arr = fx.iterations[0].params_out_sigmaall.copy()
    arr.flat[0] = np.nan
    fx.iterations[0].params_out_sigmaall = arr
    with pytest.raises(FixtureValidationError, match="params_out_sigmaall"):
        validate_em_trace_fixture(fx)


def test_em_trace_second_iteration_corruption_is_caught(fixtures_dir):
    fx = _load_em(fixtures_dir)
    assert len(fx.iterations) >= 2
    fx.iterations[1].params_out_pi0 = fx.iterations[1].params_out_pi0.copy()
    fx.iterations[1].params_out_pi0[0] = np.nan
    with pytest.raises(FixtureValidationError, match="iter2"):
        validate_em_trace_fixture(fx)


@pytest.mark.parametrize("field", ["loglik_params_in", "AIC_params_in", "BIC_params_in"])
def test_em_trace_ll_aic_bic_nan_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    setattr(fx.iterations[0], field, float("nan"))
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


@pytest.mark.parametrize("field", ["loglik_params_in", "AIC_params_in", "BIC_params_in"])
def test_em_trace_ll_aic_bic_inf_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    setattr(fx.iterations[0], field, float("inf"))
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


def test_em_trace_no_iterations_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.iterations = []
    with pytest.raises(FixtureValidationError, match="no iterations"):
        validate_em_trace_fixture(fx)


# --------------------------------------------------------------------------
# EM-trace fixture: malformed offsets/lengths -> FixtureValidationError, not
# IndexError -- same requirement as the posterior-fixture tests above.
# --------------------------------------------------------------------------

def test_em_trace_malformed_state_numbers_index_raises_validation_error(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.state_numbers_index = np.array([0])
    with pytest.raises(FixtureValidationError, match="state_numbers_index"):
        validate_em_trace_fixture(fx)


def test_em_trace_empty_length_observation_raises_validation_error_not_indexerror(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.length_observation = np.array([], dtype=np.int64)
    with pytest.raises(FixtureValidationError, match="length_observation"):
        validate_em_trace_fixture(fx)


def test_em_trace_malformed_channel_num_states_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.channel_num_states = np.array([1])
    with pytest.raises(FixtureValidationError, match="channel_num_states"):
        validate_em_trace_fixture(fx)


# --------------------------------------------------------------------------
# EM-trace fixture: row/column vectors -- same requirement as the posterior
# fixture, the prior flatten-and-count check would have missed these.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_em_trace_vector_field_as_column_vector_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    arr = getattr(fx, field)
    setattr(fx, field, arr.reshape(-1, 1))
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_em_trace_vector_field_as_row_vector_in_2d_raises(fixtures_dir, field):
    fx = _load_em(fixtures_dir)
    arr = getattr(fx, field)
    setattr(fx, field, arr.reshape(1, -1))
    with pytest.raises(FixtureValidationError, match=field):
        validate_em_trace_fixture(fx)


def test_em_trace_state_numbers_index_as_column_vector_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.state_numbers_index = fx.state_numbers_index.reshape(-1, 1)
    with pytest.raises(FixtureValidationError, match="state_numbers_index"):
        validate_em_trace_fixture(fx)


def test_em_trace_length_observation_as_column_vector_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.length_observation = fx.length_observation.reshape(-1, 1)
    with pytest.raises(FixtureValidationError, match="length_observation"):
        validate_em_trace_fixture(fx)


# --------------------------------------------------------------------------
# EM-trace fixture: inconsistent offsets -- right length and endpoints, but
# increments that don't match the per-channel counts, or a nonzero start.
# --------------------------------------------------------------------------

def test_em_trace_state_numbers_index_not_starting_at_zero_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.state_numbers_index = fx.state_numbers_index + 1
    with pytest.raises(FixtureValidationError, match=r"state_numbers_index\[0\]"):
        validate_em_trace_fixture(fx)


def test_em_trace_dimension_numbers_index_not_starting_at_zero_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    fx.dimension_numbers_index = fx.dimension_numbers_index + 1
    with pytest.raises(FixtureValidationError, match=r"dimension_numbers_index\[0\]"):
        validate_em_trace_fixture(fx)


def test_em_trace_state_numbers_index_diff_mismatch_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    assert fx.C >= 2
    rotated = np.roll(fx.channel_num_states, 1)
    assert not np.array_equal(rotated, fx.channel_num_states), "rotation must actually change the partition"
    S = int(fx.state_numbers_index[-1])
    fx.state_numbers_index = np.concatenate([[0], np.cumsum(rotated)])
    assert fx.state_numbers_index[-1] == S
    with pytest.raises(FixtureValidationError, match="diff\\(state_numbers_index\\)"):
        validate_em_trace_fixture(fx)


def test_em_trace_dimension_numbers_index_diff_mismatch_raises(fixtures_dir):
    fx = _load_em(fixtures_dir)
    assert fx.C >= 2
    rotated = np.roll(fx.dim_observation, 1)
    assert not np.array_equal(rotated, fx.dim_observation), "rotation must actually change the partition"
    fx.dimension_numbers_index = np.concatenate([[0], np.cumsum(rotated)])
    assert fx.dimension_numbers_index[-1] == fx.D
    with pytest.raises(FixtureValidationError, match="diff\\(dimension_numbers_index\\)"):
        validate_em_trace_fixture(fx)


# --------------------------------------------------------------------------
# Init-reference fixture (init_reference.npz, SCHEMA.md's "Init-reference
# fixture" section) -- a single bespoke file, not manifest-driven, but its
# loader/validator follows the exact same contract as the two families above.
# --------------------------------------------------------------------------

def _load_init_ref(fixtures_dir):
    return load_init_reference_fixture(fixtures_dir)


def test_positive_control_init_reference_validates(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    validate_init_reference_fixture(fx)  # must not raise


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_init_reference_per_channel_field_wrong_shape_raises(fixtures_dir, field):
    fx = _load_init_ref(fixtures_dir)
    setattr(fx, field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=field):
        validate_init_reference_fixture(fx)


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_init_reference_per_channel_field_nan_raises(fixtures_dir, field):
    fx = _load_init_ref(fixtures_dir)
    arr = getattr(fx, field).astype(np.float64)
    arr[0] = np.nan
    setattr(fx, field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_init_reference_fixture(fx)


@pytest.mark.parametrize(
    "field,bad_value",
    [("channel_num_states", 0), ("dim_observation", 0), ("num_gmm_component", -1)],
)
def test_init_reference_per_channel_field_non_positive_raises(fixtures_dir, field, bad_value):
    """channel_num_states[0]=0 and num_gmm_component[0]=-1 were both
    previously silently accepted -- neither shape nor NaN/Inf catches a
    non-positive count, only a dedicated positivity check does."""
    fx = _load_init_ref(fixtures_dir)
    arr = getattr(fx, field).copy()
    arr[0] = bad_value
    setattr(fx, field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_init_reference_fixture(fx)


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_init_reference_per_channel_field_wrong_dtype_raises(fixtures_dir, field):
    fx = _load_init_ref(fixtures_dir)
    setattr(fx, field, getattr(fx, field).astype(np.float64))
    with pytest.raises(FixtureValidationError, match=field):
        validate_init_reference_fixture(fx)


def test_init_reference_sigma_diag_false_raises(fixtures_dir):
    """This fixture is diagonal-only by construction (init.py always fits
    diagonal covariances) -- sigma_diag=False was previously silently
    accepted despite that contract."""
    fx = _load_init_ref(fixtures_dir)
    fx.sigma_diag = False
    with pytest.raises(FixtureValidationError, match="sigma_diag"):
        validate_init_reference_fixture(fx)


def test_init_reference_sigma_diag_wrong_type_raises(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    fx.sigma_diag = 1  # truthy, but not a bool
    with pytest.raises(FixtureValidationError, match="sigma_diag"):
        validate_init_reference_fixture(fx)


def test_init_reference_Y_wrong_shape_raises(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    fx.Y = np.zeros((1, 1))
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_init_reference_fixture(fx)


def test_init_reference_Y_nan_raises(fixtures_dir):
    """Unlike the posterior/EM-trace fixtures' Y, this one is built with no
    missingness at all -- NaN here must be rejected, not tolerated."""
    fx = _load_init_ref(fixtures_dir)
    fx.Y = fx.Y.copy()
    fx.Y[0, 0] = np.nan
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_init_reference_fixture(fx)


def test_init_reference_Y_inf_raises(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    fx.Y = fx.Y.copy()
    fx.Y[0, 0] = np.inf
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_init_reference_fixture(fx)


def test_init_reference_Y_integer_dtype_raises(fixtures_dir):
    """Integer Y was previously silently accepted."""
    fx = _load_init_ref(fixtures_dir)
    fx.Y = fx.Y.astype(np.int64)
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_init_reference_fixture(fx)


def test_init_reference_loglik_wrong_shape_raises(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    fx.loglik_params_in = fx.loglik_params_in[:-1]
    with pytest.raises(FixtureValidationError, match="loglik_params_in"):
        validate_init_reference_fixture(fx)


def test_init_reference_loglik_nan_raises(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    fx.loglik_params_in = fx.loglik_params_in.copy()
    fx.loglik_params_in[0] = np.nan
    with pytest.raises(FixtureValidationError, match="loglik_params_in"):
        validate_init_reference_fixture(fx)


def test_init_reference_loglik_integer_dtype_raises(fixtures_dir):
    """Integer loglik_params_in was previously silently accepted."""
    fx = _load_init_ref(fixtures_dir)
    fx.loglik_params_in = fx.loglik_params_in.astype(np.int64)
    with pytest.raises(FixtureValidationError, match="loglik_params_in"):
        validate_init_reference_fixture(fx)


def test_init_reference_zero_num_iters_raises(fixtures_dir):
    fx = _load_init_ref(fixtures_dir)
    fx.num_iters = 0
    fx.loglik_params_in = np.zeros((0,))
    with pytest.raises(FixtureValidationError, match="num_iters"):
        validate_init_reference_fixture(fx)


# --------------------------------------------------------------------------
# P6 fixture (p6_fixture.npz, SCHEMA.md's "P6 fixture" section) -- a single
# bespoke file, not manifest-driven, same loader/validator contract as the
# other three families.
# --------------------------------------------------------------------------

def _load_p6(fixtures_dir):
    return load_p6_fixture(fixtures_dir)


def test_positive_control_p6_fixture_validates(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    validate_p6_fixture(fx)  # must not raise


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_p6_per_channel_field_wrong_shape_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    setattr(fx, field, np.zeros((1,)))
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_p6_per_channel_field_non_positive_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    arr = getattr(fx, field).copy()
    arr[0] = 0
    setattr(fx, field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_p6_per_channel_field_wrong_dtype_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    setattr(fx, field, getattr(fx, field).astype(np.float64))
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


def test_p6_joint_pi0_not_summing_to_one_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.joint_pi0 = fx.joint_pi0.copy()
    fx.joint_pi0[0] += 1.0
    with pytest.raises(FixtureValidationError, match="joint_pi0"):
        validate_p6_fixture(fx)


def test_p6_joint_transition_not_row_stochastic_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.joint_transition = fx.joint_transition.copy()
    fx.joint_transition[0, 0] += 1.0
    with pytest.raises(FixtureValidationError, match="joint_transition"):
        validate_p6_fixture(fx)


def test_p6_joint_transition_wrong_shape_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.joint_transition = np.eye(2)
    with pytest.raises(FixtureValidationError, match="joint_transition"):
        validate_p6_fixture(fx)


def test_p6_Y_nan_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.Y = fx.Y.copy()
    fx.Y[0, 0] = np.nan
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_p6_fixture(fx)


def test_p6_Y_integer_dtype_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.Y = fx.Y.astype(np.int64)
    with pytest.raises(FixtureValidationError, match="Y"):
        validate_p6_fixture(fx)


def test_p6_X_star_out_of_range_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.X_star = fx.X_star.copy()
    fx.X_star[0, 0] = fx.channel_num_states[0]  # one past the valid range
    with pytest.raises(FixtureValidationError, match="X_star"):
        validate_p6_fixture(fx)


def test_p6_X_star_wrong_shape_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.X_star = fx.X_star[:, :-1]
    with pytest.raises(FixtureValidationError, match="X_star"):
        validate_p6_fixture(fx)


def test_p6_P_star_model_non_finite_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.P_star_model = float("nan")
    with pytest.raises(FixtureValidationError, match="P_star_model"):
        validate_p6_fixture(fx)


def test_p6_log_joint_prob_non_finite_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.log_joint_prob_viterbi_path = float("-inf")
    with pytest.raises(FixtureValidationError, match="log_joint_prob_viterbi_path"):
        validate_p6_fixture(fx)


def test_p6_packed_param_wrong_shape_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.transition_packed = np.eye(2)
    with pytest.raises(FixtureValidationError, match="transition_packed"):
        validate_p6_fixture(fx)


def test_p6_packed_param_nan_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.mu_all_packed = fx.mu_all_packed.copy()
    fx.mu_all_packed[0, 0, 0] = np.nan
    with pytest.raises(FixtureValidationError, match="mu_all_packed"):
        validate_p6_fixture(fx)


def test_p6_sigma_diag_false_raises(fixtures_dir):
    """SCHEMA.md's P6 fixture is diagonal-only by construction -- `bool`
    typed but *not* `True` must still be rejected, not just non-bool types."""
    fx = _load_p6(fixtures_dir)
    fx.sigma_diag = False
    with pytest.raises(FixtureValidationError, match="sigma_diag"):
        validate_p6_fixture(fx)


def test_p6_sigma_diag_non_bool_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.sigma_diag = 1
    with pytest.raises(FixtureValidationError, match="sigma_diag"):
        validate_p6_fixture(fx)


@pytest.mark.parametrize("field", ["T", "C"])
def test_p6_non_positive_scalar_metadata_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    setattr(fx, field, 0)
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


@pytest.mark.parametrize("field", ["X_star", "X_alt"])
def test_p6_hidden_state_float_dtype_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    setattr(fx, field, getattr(fx, field).astype(np.float64))
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


@pytest.mark.parametrize(
    "field",
    [
        "joint_pi0", "joint_transition", "joint_P_all", "joint_mu_all", "joint_sigma_all",
        "cond_trans_1", "cond_trans_2", "pi_0_packed", "coupling_packed", "transition_packed",
        "P_all_packed", "mu_all_packed", "sigma_all_packed",
    ],
)
def test_p6_float_field_integer_dtype_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    setattr(fx, field, getattr(fx, field).astype(np.int64))
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


def test_p6_joint_pi0_sum_uses_rtol_zero(fixtures_dir):
    """A sum of `1 + 3e-6` would pass the old `np.isclose(..., atol=1e-8)`
    call's default `rtol=1e-5` (threshold ~1.00001e-5) but must fail once
    `rtol=0` is used (threshold 1e-8)."""
    fx = _load_p6(fixtures_dir)
    fx.joint_pi0 = fx.joint_pi0.copy()
    fx.joint_pi0[0] += 3e-6
    with pytest.raises(FixtureValidationError, match="joint_pi0"):
        validate_p6_fixture(fx)


def test_p6_joint_transition_sum_uses_rtol_zero(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.joint_transition = fx.joint_transition.copy()
    fx.joint_transition[0, 0] += 3e-6
    with pytest.raises(FixtureValidationError, match="joint_transition"):
        validate_p6_fixture(fx)


def test_p6_joint_pi0_negative_entry_with_sum_preserved_raises(fixtures_dir):
    """A negative entry compensated elsewhere still sums to 1.0 -- the sum
    check alone would miss it; non-negativity must be checked explicitly."""
    fx = _load_p6(fixtures_dir)
    fx.joint_pi0 = fx.joint_pi0.copy()
    delta = fx.joint_pi0[0] + 0.5
    fx.joint_pi0[0] -= delta
    fx.joint_pi0[1] += delta
    assert fx.joint_pi0[0] < 0
    np.testing.assert_allclose(fx.joint_pi0.sum(), 1.0, atol=1e-8)
    with pytest.raises(FixtureValidationError, match="joint_pi0"):
        validate_p6_fixture(fx)


def test_p6_joint_transition_negative_entry_with_row_sum_preserved_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.joint_transition = fx.joint_transition.copy()
    row = fx.joint_transition[0]
    delta = row[0] + 0.5
    row[0] -= delta
    row[1] += delta
    assert fx.joint_transition[0, 0] < 0
    np.testing.assert_allclose(fx.joint_transition[0].sum(), 1.0, atol=1e-8)
    with pytest.raises(FixtureValidationError, match="joint_transition"):
        validate_p6_fixture(fx)


def test_p6_joint_P_all_not_row_stochastic_raises(fixtures_dir):
    fx = _load_p6(fixtures_dir)
    fx.joint_P_all = fx.joint_P_all.copy()
    fx.joint_P_all[0, 0] += 1.0
    with pytest.raises(FixtureValidationError, match="joint_P_all"):
        validate_p6_fixture(fx)


@pytest.mark.parametrize("field", ["cond_trans_1", "cond_trans_2"])
def test_p6_cond_trans_not_row_stochastic_raises(fixtures_dir, field):
    fx = _load_p6(fixtures_dir)
    arr = getattr(fx, field).copy()
    arr[0, 0] += 1.0
    setattr(fx, field, arr)
    with pytest.raises(FixtureValidationError, match=field):
        validate_p6_fixture(fx)


def test_p6_packed_params_fail_lsim_params_validate(fixtures_dir):
    """A corruption none of the shape/dtype/NaN checks above would catch --
    the first channel's `pi_0_packed` block no longer sums to 1 -- must
    still be caught, via reusing `LSIMParams.from_packed(...).validate()`."""
    fx = _load_p6(fixtures_dir)
    fx.pi_0_packed = fx.pi_0_packed.copy()
    fx.pi_0_packed[0] += 1.0
    with pytest.raises(FixtureValidationError, match="packed parameters failed"):
        validate_p6_fixture(fx)


# --------------------------------------------------------------------------
# P10a scheduling fixtures (p10a_scheduling__<scenario_id>.npz, SCHEMA.md's
# "P10a scheduling fixtures" section) -- six single, bespoke files (one per
# scenario in P10A_SCENARIO_IDS), same loader/validator contract as the
# other fixture families. Ground truth for P10a's own audit/instrumentation
# deliverable -- no Python mini-batch training exists yet to consume it
# (P10b, explicitly out of scope here), so these tests only prove the
# fixtures load and validate correctly, the same discipline the P6/init-
# reference fixtures were held to before any consuming algorithm existed.
#
# A correction round found the FIRST version of this validator (and this
# test file's own coverage of it) accepted several planted corruptions it
# should have rejected: Inf in Y, NaN in AIC_trace, a scenario_id mismatch,
# a float-dtype schedule array, sigma_diag=2 (silently coerced to True
# rather than rejected), and NaN in unused P_all padding. Every one of
# those is now a dedicated negative-control test below, alongside broader,
# systematic coverage (shape/dtype/finiteness/range/identity) across every
# field group the schema declares: raw/effective/per-update/per-epoch/
# returned-best parameters, the ordered schedule, every trace array, and
# both pi_steady variants.
# --------------------------------------------------------------------------


def _load_p10a(fixtures_dir, scenario_id):
    return load_p10a_scheduling_fixture(fixtures_dir, scenario_id)


def _assert_rejects(fx, match):
    with pytest.raises(FixtureValidationError, match=match):
        validate_p10a_scheduling_fixture(fx)


def _copy_p10a_npz_as(fixtures_dir, src_scenario_id, dst_scenario_id, tmp_path):
    """Copies p10a_scheduling__<src>.npz's raw bytes, unmodified, into a file
    named p10a_scheduling__<dst>.npz under tmp_path -- used to test load_
    p10a_scheduling_fixture's own filename/content binding check (item 1 of
    the final P10a correction round), which a plain fx.scenario_id mutation
    on an already-loaded fixture cannot exercise (that only reaches
    validate_p10a_scheduling_fixture's membership-in-P10A_SCENARIO_IDS
    check, not the loader's requested-vs-actual binding)."""
    src_path = Path(fixtures_dir) / f"p10a_scheduling__{src_scenario_id}.npz"
    dst_path = Path(tmp_path) / f"p10a_scheduling__{dst_scenario_id}.npz"
    shutil.copyfile(src_path, dst_path)
    return Path(tmp_path)


def _write_p10a_npz_with_override(fixtures_dir, scenario_id, tmp_path, **overrides):
    """Loads a real p10a_scheduling__<scenario_id>.npz, overrides exactly the
    given top-level keys, and re-saves it under tmp_path with the same
    filename -- used for load-time corruptions (a malformed scalar dtype, a
    variant/sigma_diag value that no longer matches the known scenario
    mapping) that must be caught by load_p10a_scheduling_fixture itself,
    before validate_p10a_scheduling_fixture ever sees the loaded fixture."""
    src_path = Path(fixtures_dir) / f"p10a_scheduling__{scenario_id}.npz"
    with np.load(src_path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data.update(overrides)
    dst_path = Path(tmp_path) / f"p10a_scheduling__{scenario_id}.npz"
    np.savez(dst_path, **data)
    return Path(tmp_path)


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_positive_control_p10a_scheduling_fixture_validates(fixtures_dir, scenario_id):
    fx = _load_p10a(fixtures_dir, scenario_id)
    validate_p10a_scheduling_fixture(fx)  # must not raise
    assert fx.scenario_id == scenario_id
    assert fx.sigma_diag == (scenario_id != "full")
    assert fx.variant == ("fast" if scenario_id == "fast" else "sed")
    assert fx.n_updates == fx.max_itration * fx.mini_batch


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_epoch_permutation_covers_every_trial_exactly_once_per_epoch(fixtures_dir, scenario_id):
    """Independent of validate_p10a_scheduling_fixture's own check: for every
    epoch, epoch_permutation is a genuine permutation of 1..num_trials --
    the §2.6 gate this fixture exists to support ("every trial occurs once
    per epoch, with no drops or duplicates"), now checked on the ORDERED
    representation, not a membership matrix that could not distinguish
    order from a merely-correct multiset."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    for itr in range(fx.max_itration):
        row = fx.epoch_permutation[itr]
        assert sorted(row.tolist()) == list(range(1, fx.num_trials + 1))


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_updates_trial_indices_match_epoch_permutation_slice(fixtures_dir, scenario_id):
    """The flattened per-update axis and the per-epoch permutation must
    agree with each other, not merely each look well-formed independently
    -- exercises the `uneven` scenario's genuinely non-uniform minibatch
    sizes specifically (a bug here would most likely hide on a uniform
    partition, where every slice happens to be the same length)."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    for u in range(fx.n_updates):
        itr0 = int(fx.updates_epoch_idx[u]) - 1
        mb0 = int(fx.updates_minibatch_idx[u]) - 1
        n_tr = int(fx.updates_n_trials[u])
        lo, hi = int(fx.batch_offsets[itr0, mb0]), int(fx.batch_offsets[itr0, mb0 + 1])
        expected = fx.epoch_permutation[itr0, lo:hi]
        got = fx.updates_trial_indices[u, :n_tr]
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_mixed_loglik_sums_to_loglik_trace_two_ways(fixtures_dir, scenario_id):
    """The legacy batch-training diagnostic's own defining property (§2.6),
    checked via BOTH the already-assembled per-epoch row (epoch_mixed_
    loglik) AND independently via the ragged per-update breakdown
    (updates_pretrial_loglik, active region only) -- both must equal
    log_likelyhood_trace exactly, since that IS how em_lsim_par.m computes
    log_likelihood(itr) (sum(epoch_P_O_model)), confirmed on the actual
    exported numbers, not assumed from the export code's own logic."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    np.testing.assert_allclose(fx.epoch_mixed_loglik.sum(axis=1), fx.log_likelyhood_trace, rtol=1e-9, atol=1e-9)

    per_epoch = np.zeros(fx.max_itration)
    for u in range(fx.n_updates):
        itr0 = int(fx.updates_epoch_idx[u]) - 1
        n_tr = int(fx.updates_n_trials[u])
        per_epoch[itr0] += float(fx.updates_pretrial_loglik[u, :n_tr].sum())
    np.testing.assert_allclose(per_epoch, fx.log_likelyhood_trace, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_epoch_final_equals_last_update_out(fixtures_dir, scenario_id):
    """Each epoch's own final packed snapshot must equal that epoch's LAST
    update's own params_out exactly -- the epoch-final snapshot IS the last
    minibatch's own post-update commit, not a separately-computed value
    that merely happens to agree."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    for itr in range(fx.max_itration):
        last_u = itr * fx.mini_batch + (fx.mini_batch - 1)
        np.testing.assert_array_equal(fx.epoch_pi_0_final[itr], fx.updates_pi_0_out[last_u])
        np.testing.assert_array_equal(fx.epoch_transition_final[itr], fx.updates_transition_out[last_u])


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_returned_best_equals_epoch_final_at_best_index(fixtures_dir, scenario_id):
    """em_lsim_par.m's own returned output must equal the epoch-final
    snapshot at the recorded best epoch exactly -- proving the exporter's
    own best-epoch bookkeeping genuinely tracks what em_lsim_par.m did,
    not merely a plausible-looking independent guess."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    best_i = fx.returned_best_epoch_idx - 1
    np.testing.assert_array_equal(fx.returned_best_pi_0, fx.epoch_pi_0_final[best_i])
    np.testing.assert_allclose(fx.returned_best_pi_steady, fx.epoch_legacy_mixed_pi_steady[best_i], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_raw_init_differs_from_effective_first_update(fixtures_dir, scenario_id):
    """The single most important distinction this correction round exists
    to enforce: raw_init_pi_0 (caller-provided, pre-smoothing) must NOT
    equal updates_pi_0_in[0] (the model the first E-step actually used,
    post-smoothing) -- if a future regeneration ever made these equal, it
    would mean the smoothing step silently stopped happening, or the
    exporter regressed to exporting the same snapshot for both fields
    again."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    diff = np.max(np.abs(fx.raw_init_pi_0 - fx.updates_pi_0_in[0]))
    assert diff > 1e-8, f"raw_init_pi_0 and updates_pi_0_in[0] are suspiciously close (diff={diff:.3e}) -- expected a measurable pi_0-smoothing difference"


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_coherent_and_legacy_pi_steady_are_distinct_arrays(fixtures_dir, scenario_id):
    """Not a strict inequality requirement (they could coincide numerically
    in principle), but they must be independently-computed, differently-
    shaped-in-purpose quantities -- this test fails loudly if a future
    regression makes the exporter alias one to the other by construction
    (e.g. accidentally writing epoch_legacy_mixed_pi_steady into both
    fields), by checking they are not byte-identical across every epoch
    simultaneously, which would be the signature of aliasing rather than
    of two genuinely different computations that happen to agree."""
    fx = _load_p10a(fixtures_dir, scenario_id)
    identical_every_epoch = np.array_equal(fx.epoch_coherent_pi_steady, fx.epoch_legacy_mixed_pi_steady)
    assert not identical_every_epoch, "epoch_coherent_pi_steady and epoch_legacy_mixed_pi_steady are byte-identical across every epoch -- looks aliased, not independently computed"


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_p10a_coherent_loglik_independently_recomputed_by_python_lsim(fixtures_dir, scenario_id):
    """The strongest available cross-check for this fixture family: build a
    real LSIMStructure/LSIMParams from each epoch's OWN epoch_*_final
    packed snapshot, run the already-P3a/P3b-verified Python `chmm_lsim` kernel
    (backend='numpy') on the fixture's own Y_trials(), and confirm the
    recomputed per-trial loglik matches epoch_coherent_loglik to machine
    precision -- not merely that the MATLAB exporter's own internal
    bookkeeping is self-consistent, but that an independent, differently-
    implemented (Python, not MATLAB) kernel reproduces the same numbers
    from the same stored parameters. Confirmed directly: max abs diff
    ~1e-14-6e-14 across every scenario including 'fast', during the P10a
    correction round."""
    fx = _load_p10a(fixtures_dir, scenario_id)

    from chmm_lsim.backends.base import run_backend
    from chmm_lsim.compat import FbConstants
    from chmm_lsim.params import LSIMParams
    from chmm_lsim.structure import LSIMStructure

    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    Y_trials = fx.Y_trials()
    fbc = FbConstants(variant=fx.variant)

    total_T = int(fx.length_observation.sum())

    for itr in range(fx.max_itration):
        params = LSIMParams.from_packed(
            structure,
            pi_0=fx.epoch_pi_0_final[itr], transition=fx.epoch_transition_final[itr],
            coupling=fx.epoch_coupling_final[itr], P_all=fx.epoch_P_all_final[itr],
            mu_all=fx.epoch_mu_all_final[itr], sigma_all=fx.epoch_sigma_all_final[itr],
            sigma_diag=bool(fx.sigma_diag),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posteriors = [run_backend("numpy", fx.variant, structure, params, Y, fbc) for Y in Y_trials]
        recomputed = np.array([p.loglik for p in posteriors])
        np.testing.assert_allclose(
            recomputed, fx.epoch_coherent_loglik[itr], rtol=1e-9, atol=1e-9,
            err_msg=f"{scenario_id} epoch {itr}: Python-recomputed coherent loglik != epoch_coherent_loglik",
        )

        # epoch_coherent_pi_steady is genuinely gamma-based (§2.6's own
        # description: "a full forward-BACKWARD pass ... coherent_pi_steady
        # = coherent_alpha_T_sum / T_all"), so recompute it the same way:
        # accumulate each trial's own Posterior.gamma (== alpha_T, the
        # smoothed per-channel marginals) across every trial's own T, then
        # divide by the TOTAL T across the whole dataset -- not merely
        # replay loglik, which would miss a genuine bug in how gamma itself
        # (as opposed to the scalar loglik) is exported.
        gamma_sum = np.zeros(fx.S)
        for p in posteriors:
            gamma_sum += p.gamma.sum(axis=1)
        recomputed_pi_steady = gamma_sum / total_T
        np.testing.assert_allclose(
            recomputed_pi_steady, fx.epoch_coherent_pi_steady[itr], rtol=1e-7, atol=1e-9,
            err_msg=f"{scenario_id} epoch {itr}: Python-recomputed gamma-based pi_steady != epoch_coherent_pi_steady",
        )


# ---- negative controls: the specific bugs a review round confirmed the
# first version of this validator wrongly accepted -------------------------


def test_p10a_inf_in_Y_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.Y = fx.Y.copy()
    fx.Y[0, 0] = np.inf
    _assert_rejects(fx, "Y.*Inf")


def test_p10a_nan_in_AIC_trace_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.AIC_trace = fx.AIC_trace.copy()
    fx.AIC_trace[0] = np.nan
    _assert_rejects(fx, "AIC_trace.*NaN")


def test_p10a_nan_in_BIC_trace_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.BIC_trace = fx.BIC_trace.copy()
    fx.BIC_trace[-1] = np.nan
    _assert_rejects(fx, "BIC_trace.*NaN")


def test_p10a_inf_in_log_likelyhood_trace_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.log_likelyhood_trace = fx.log_likelyhood_trace.copy()
    fx.log_likelyhood_trace[0] = -np.inf
    _assert_rejects(fx, "log_likelyhood_trace.*Inf")


def test_p10a_scenario_id_mismatch_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.scenario_id = "wrong"
    _assert_rejects(fx, "scenario_id")


def test_p10a_invalid_variant_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.variant = "svi"
    _assert_rejects(fx, "variant")


def test_p10a_float_dtype_schedule_array_raises(fixtures_dir):
    """A float-dtype epoch_permutation (even with integer-VALUED entries)
    must be rejected on dtype alone, not merely accepted because its
    contents happen to still look like a valid permutation."""
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_permutation = fx.epoch_permutation.astype(np.float64)
    _assert_rejects(fx, "epoch_permutation.dtype")


def test_p10a_float_dtype_updates_trial_indices_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_trial_indices = fx.updates_trial_indices.astype(np.float64)
    _assert_rejects(fx, "updates_trial_indices.dtype")


@pytest.mark.parametrize("bad_value", [2, 0, 1, "True"])
def test_p10a_non_bool_sigma_diag_raises(fixtures_dir, bad_value):
    """sigma_diag=2 (or any other non-bool, including the plain ints 0/1,
    which are valid-looking but still not actually bool instances) must be
    rejected outright, never coerced via bool(x) -- bool(2) is True, which
    would silently accept a corrupted value."""
    fx = _load_p10a(fixtures_dir, "diag")
    fx.sigma_diag = bad_value
    _assert_rejects(fx, "sigma_diag")


def test_p10a_nan_in_unused_P_all_padding_raises(fixtures_dir):
    """`diag`'s channel 1 has num_gmm_component=1 (heterogeneous K is
    channel 2's own K=2), so P_all[0, :, 1:] is unused padding, not an
    active value -- NaN there must still be rejected."""
    fx = _load_p10a(fixtures_dir, "diag")
    fx.raw_init_P_all = fx.raw_init_P_all.copy()
    assert fx.raw_init_P_all.shape[-1] >= 2, "this test needs K>1 padding to exist -- scenario changed?"
    fx.raw_init_P_all[0, 0, -1] = np.nan
    _assert_rejects(fx, "raw_init_P_all")


def test_p10a_nan_in_unused_updates_P_all_padding_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_P_all_in = fx.updates_P_all_in.copy()
    fx.updates_P_all_in[0, 0, 0, -1] = np.nan
    _assert_rejects(fx, "updates_P_all_in")


def test_p10a_nan_in_unused_epoch_final_padding_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_mu_all_final = fx.epoch_mu_all_final.copy()
    fx.epoch_mu_all_final[0, 0, 0, -1] = np.nan
    _assert_rejects(fx, "epoch_mu_all_final")


# ---- broader systematic coverage: shape, dtype, range, identities --------


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_p10a_per_channel_field_wrong_shape_raises(fixtures_dir, field):
    fx = _load_p10a(fixtures_dir, "diag")
    setattr(fx, field, np.zeros((1,), dtype=np.int64))
    _assert_rejects(fx, field)


@pytest.mark.parametrize("field", ["max_itration", "mini_batch", "num_trials", "C", "max_batch_size"])
def test_p10a_non_positive_scalar_raises(fixtures_dir, field):
    fx = _load_p10a(fixtures_dir, "diag")
    setattr(fx, field, 0)
    _assert_rejects(fx, field)


def test_p10a_n_updates_inconsistent_with_max_itration_mini_batch_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.n_updates = fx.n_updates + 1
    _assert_rejects(fx, "n_updates")


def test_p10a_returned_best_epoch_idx_out_of_range_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.returned_best_epoch_idx = fx.max_itration + 1
    _assert_rejects(fx, "returned_best_epoch_idx")


def test_p10a_updates_n_transitions_inconsistent_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_n_transitions = fx.updates_n_transitions.copy()
    fx.updates_n_transitions[0] += 1
    _assert_rejects(fx, "updates_n_transitions")


def test_p10a_updates_trial_indices_duplicate_in_active_region_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_trial_indices = fx.updates_trial_indices.copy()
    n_tr = int(fx.updates_n_trials[0])
    if n_tr >= 2:
        fx.updates_trial_indices[0, 1] = fx.updates_trial_indices[0, 0]  # duplicate the first trial index
        _assert_rejects(fx, "updates_trial_indices")


def test_p10a_updates_trial_indices_nonzero_padding_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_trial_indices = fx.updates_trial_indices.copy()
    n_tr = int(fx.updates_n_trials[0])
    assert n_tr < fx.max_batch_size, "this test needs real padding to exist -- scenario changed?"
    fx.updates_trial_indices[0, n_tr] = 1  # write into what should be zero-padding
    _assert_rejects(fx, "updates_trial_indices")


def test_p10a_updates_pretrial_loglik_nonnan_padding_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_pretrial_loglik = fx.updates_pretrial_loglik.copy()
    n_tr = int(fx.updates_n_trials[0])
    assert n_tr < fx.max_batch_size
    fx.updates_pretrial_loglik[0, n_tr] = 0.0  # write a real number into what should be NaN-padding
    _assert_rejects(fx, "updates_pretrial_loglik")


def test_p10a_batch_offsets_empty_minibatch_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.batch_offsets = fx.batch_offsets.copy()
    fx.batch_offsets[0, 1] = fx.batch_offsets[0, 0]  # collapse minibatch 1 of epoch 0 to zero trials
    _assert_rejects(fx, "batch_offsets")


def test_p10a_epoch_permutation_not_a_permutation_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_permutation = fx.epoch_permutation.copy()
    fx.epoch_permutation[0, 0] = fx.epoch_permutation[0, 1]  # duplicate an entry, breaking the permutation
    _assert_rejects(fx, "epoch_permutation")


def test_p10a_raw_init_packed_params_fail_lsim_params_validate(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.raw_init_pi_0 = fx.raw_init_pi_0.copy()
    fx.raw_init_pi_0[0] += 1.0
    _assert_rejects(fx, "packed parameters .raw_init. failed")


def test_p10a_returned_best_packed_params_fail_lsim_params_validate(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.returned_best_coupling = fx.returned_best_coupling.copy()
    fx.returned_best_coupling[0, 0] += 1.0
    _assert_rejects(fx, "packed parameters .returned_best. failed")


def test_p10a_updates_in_packed_params_fail_lsim_params_validate(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_pi_0_in = fx.updates_pi_0_in.copy()
    fx.updates_pi_0_in[0, 0] += 1.0
    _assert_rejects(fx, r"packed parameters \(updates\[0\]\.in\) failed")


def test_p10a_epoch_final_packed_params_fail_lsim_params_validate(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_coupling_final = fx.epoch_coupling_final.copy()
    fx.epoch_coupling_final[0, 0, 0] += 1.0
    _assert_rejects(fx, r"packed parameters \(epoch\[0\]\.final\) failed")


@pytest.mark.parametrize("field,shape", [
    ("updates_pi_0_in", None), ("updates_transition_out", None), ("updates_P_all_in", None),
    ("epoch_pi_0_final", None), ("epoch_sigma_all_final", None),
    ("returned_best_mu_all", None), ("raw_init_sigma_all", None),
])
def test_p10a_wrong_shape_raises(fixtures_dir, field, shape):
    fx = _load_p10a(fixtures_dir, "diag")
    arr = getattr(fx, field)
    setattr(fx, field, arr.reshape(-1)[: max(1, arr.size - 1)])  # drop one element -> guaranteed wrong shape
    _assert_rejects(fx, field)


# ==========================================================================
# Final P10a fixture-contract correction round: the validator did not
# enforce several schema-declared identities -- filename/scenario binding,
# the exact flattened update sequence, complete parameter-chain continuity
# across all six packed tensors (not only pi_0/transition), independent
# gamma-based pi_steady verification, exact dtype + positivity + anti-
# laundering, and AIC/BIC identities. Every item below has a dedicated
# planted-corruption test, including corruption of each of the six
# parameter families where relevant.
# ==========================================================================

_P10A_PARAM_FIELDS = ("pi_0", "transition", "coupling", "P_all", "mu_all", "sigma_all")


# ---- item 1: bind filename/requested scenario to fixture metadata --------


def test_p10a_load_rejects_scenario_id_filename_mismatch(fixtures_dir, tmp_path):
    """The bug this item exists to catch: a file literally named
    p10a_scheduling__diag.npz but whose internal scenario_id says "fast" (a
    valid scenario_id, unlike the pre-existing "wrong" negative control,
    which only exercises validate_p10a_scheduling_fixture's membership
    check) must be rejected by load_p10a_scheduling_fixture itself."""
    tmp_dir = _copy_p10a_npz_as(fixtures_dir, "fast", "diag", tmp_path)
    with pytest.raises(FixtureValidationError, match="scenario_id"):
        load_p10a_scheduling_fixture(tmp_dir, "diag")


def test_p10a_load_rejects_variant_scenario_mapping_violation(fixtures_dir, tmp_path):
    """scenario_id="diag" must have variant='sed' -- a file that (correctly,
    per its own filename/scenario_id) says "diag" but internally has
    variant='fast' violates the known scenario->variant mapping and must be
    rejected, independent of the filename-binding check above."""
    tmp_dir = _write_p10a_npz_with_override(fixtures_dir, "diag", tmp_path, variant=np.array("fast"))
    with pytest.raises(FixtureValidationError, match="variant"):
        load_p10a_scheduling_fixture(tmp_dir, "diag")


def test_p10a_load_rejects_sigma_diag_scenario_mapping_violation_full(fixtures_dir, tmp_path):
    """scenario_id="full" must have sigma_diag=False."""
    tmp_dir = _write_p10a_npz_with_override(fixtures_dir, "full", tmp_path, sigma_diag=np.array(True))
    with pytest.raises(FixtureValidationError, match="sigma_diag"):
        load_p10a_scheduling_fixture(tmp_dir, "full")


def test_p10a_load_rejects_sigma_diag_scenario_mapping_violation_diag(fixtures_dir, tmp_path):
    """Every scenario other than "full" must have sigma_diag=True."""
    tmp_dir = _write_p10a_npz_with_override(fixtures_dir, "diag", tmp_path, sigma_diag=np.array(False))
    with pytest.raises(FixtureValidationError, match="sigma_diag"):
        load_p10a_scheduling_fixture(tmp_dir, "diag")


def test_p10a_load_accepts_genuine_fixture_for_every_scenario(fixtures_dir):
    """Positive control for the new binding check: every real, un-corrupted
    fixture must still load cleanly under its own scenario_id."""
    for scenario_id in P10A_SCENARIO_IDS:
        load_p10a_scheduling_fixture(fixtures_dir, scenario_id)  # must not raise


# ---- item 2: exact flattened update sequence ------------------------------


def test_p10a_updates_epoch_idx_wrong_sequence_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    assert fx.n_updates > 1
    fx.updates_epoch_idx = fx.updates_epoch_idx.copy()
    fx.updates_epoch_idx[0] = fx.updates_epoch_idx[-1]
    _assert_rejects(fx, "updates_epoch_idx")


def test_p10a_updates_minibatch_idx_wrong_sequence_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "uneven")
    assert fx.n_updates > 1
    fx.updates_minibatch_idx = fx.updates_minibatch_idx.copy()
    fx.updates_minibatch_idx[0] = fx.updates_minibatch_idx[-1]
    _assert_rejects(fx, "updates_minibatch_idx")


def test_p10a_updates_n_trials_inconsistent_with_batch_offsets_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "uneven")
    fx.updates_n_trials = fx.updates_n_trials.copy()
    fx.updates_n_trials[0] += 1
    _assert_rejects(fx, "updates_n_trials")


def test_p10a_updates_T_all_inconsistent_with_length_observation_raises(fixtures_dir):
    """Bumping T_all[0] while keeping n_transitions == T_all - n_trials
    consistent isolates this specific identity (T_all vs. the sum of
    length_observation over the update's own ordered trial indices) from
    the pre-existing n_transitions identity, which would otherwise fire
    first and mask this one."""
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_T_all = fx.updates_T_all.copy()
    fx.updates_n_transitions = fx.updates_n_transitions.copy()
    fx.updates_T_all[0] += 5
    fx.updates_n_transitions[0] += 5
    _assert_rejects(fx, "updates_T_all")


# ---- item 3: complete parameter-chain continuity, all six tensors --------


@pytest.mark.parametrize("field", _P10A_PARAM_FIELDS)
def test_p10a_updates_chain_continuity_broken_raises(fixtures_dir, field):
    """updates_<field>_in[1] must equal updates_<field>_out[0] exactly, for
    EVERY one of the six packed tensors -- a prior version of this
    validator checked this identity for pi_0 only."""
    fx = _load_p10a(fixtures_dir, "diag")
    assert fx.n_updates > 1
    arr = getattr(fx, f"updates_{field}_in").copy()
    arr[1] = arr[1] + 1.0
    setattr(fx, f"updates_{field}_in", arr)
    _assert_rejects(fx, f"updates_{field}_in")


@pytest.mark.parametrize("field", ("transition", "coupling", "P_all", "mu_all", "sigma_all"))
def test_p10a_updates_in0_differs_from_raw_init_raises(fixtures_dir, field):
    """At u=0, every tensor except pi_0 (which is smoothed once before the
    first E-step) must equal raw_init_<field> exactly."""
    fx = _load_p10a(fixtures_dir, "diag")
    arr = getattr(fx, f"updates_{field}_in").copy()
    arr[0] = arr[0] + 1.0
    setattr(fx, f"updates_{field}_in", arr)
    _assert_rejects(fx, f"updates_{field}_in")


def test_p10a_pi_0_smoothing_recomputation_mismatch_raises(fixtures_dir):
    """updates_pi_0_in[0] must equal the independently recomputed effective
    pi_0 (apply_left2right_pi0_smoothing(raw_init_pi_0, ..., left2right=
    False)) -- reversing the (still valid, still sums-to-1-per-channel)
    vector isolates this check from LSIMParams.validate(), which would not
    otherwise notice anything wrong."""
    fx = _load_p10a(fixtures_dir, "diag")
    fx.updates_pi_0_in = fx.updates_pi_0_in.copy()
    fx.updates_pi_0_in[0] = fx.updates_pi_0_in[0][::-1]
    _assert_rejects(fx, "independently recomputed effective pi_0")


@pytest.mark.parametrize("field", _P10A_PARAM_FIELDS)
def test_p10a_epoch_final_field_differs_from_last_update_out_raises(fixtures_dir, field):
    """epoch_<field>_final[itr] must equal updates_<field>_out[last update
    of that epoch] exactly, for EVERY packed tensor -- a prior version of
    this validator checked pi_0 only."""
    fx = _load_p10a(fixtures_dir, "diag")
    arr = getattr(fx, f"epoch_{field}_final").copy()
    arr[0] = arr[0] + 1.0
    setattr(fx, f"epoch_{field}_final", arr)
    _assert_rejects(fx, f"epoch_{field}_final")


@pytest.mark.parametrize("field", _P10A_PARAM_FIELDS)
def test_p10a_returned_best_field_differs_from_epoch_final_raises(fixtures_dir, field):
    """returned_best_<field> must equal epoch_<field>_final at the recorded
    best epoch exactly, for EVERY packed tensor -- a prior version of this
    validator checked pi_0 only."""
    fx = _load_p10a(fixtures_dir, "diag")
    arr = getattr(fx, f"returned_best_{field}").copy()
    arr = arr + 1.0
    setattr(fx, f"returned_best_{field}", arr)
    _assert_rejects(fx, f"returned_best_{field}")


def test_p10a_returned_best_epoch_idx_disagrees_with_rederivation_raises(fixtures_dir):
    """returned_best_epoch_idx must equal the independently re-derived best
    epoch, using em_lsim_par.m's own exact (and deliberately unusual)
    selection rule: iteration 2 unconditionally overwrites iteration 1's
    snapshot, and only iteration 3+ requires strict improvement."""
    fx = _load_p10a(fixtures_dir, "diag")
    assert fx.max_itration >= 3
    wrong_idx = (fx.returned_best_epoch_idx % fx.max_itration) + 1
    assert wrong_idx != fx.returned_best_epoch_idx
    fx.returned_best_epoch_idx = wrong_idx
    _assert_rejects(fx, "independently re-derived best epoch")


# ---- item 4: independently verify coherent pi_steady ----------------------
# (the gamma-accumulation replay itself is extended directly in
# test_p10a_coherent_loglik_independently_recomputed_by_python_lsim above;
# these cover the validator's own finiteness/non-negativity/block-sum
# checks on both pi_steady variants.)


def test_p10a_epoch_coherent_pi_steady_negative_entry_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_coherent_pi_steady = fx.epoch_coherent_pi_steady.copy()
    fx.epoch_coherent_pi_steady[0, 0] = -0.01
    _assert_rejects(fx, "epoch_coherent_pi_steady")


def test_p10a_epoch_coherent_pi_steady_nan_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_coherent_pi_steady = fx.epoch_coherent_pi_steady.copy()
    fx.epoch_coherent_pi_steady[0, 0] = np.nan
    _assert_rejects(fx, "epoch_coherent_pi_steady")


def test_p10a_epoch_legacy_mixed_pi_steady_block_sum_not_one_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_legacy_mixed_pi_steady = fx.epoch_legacy_mixed_pi_steady.copy()
    fx.epoch_legacy_mixed_pi_steady[0, 0] += 0.3
    _assert_rejects(fx, "does not sum to 1")


def test_p10a_epoch_coherent_pi_steady_block_sum_not_one_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.epoch_coherent_pi_steady = fx.epoch_coherent_pi_steady.copy()
    fx.epoch_coherent_pi_steady[0, 0] += 0.3
    _assert_rejects(fx, "does not sum to 1")


# ---- item 5: exact dtype, positivity, anti-laundering ---------------------


@pytest.mark.parametrize("field", ["Y", "raw_init_pi_0", "epoch_pi_0_final", "log_likelyhood_trace", "updates_pretrial_loglik"])
def test_p10a_float32_dtype_raises(fixtures_dir, field):
    """SCHEMA.md declares float64 specifically -- a float32 fixture (even
    with numerically-identical values) must be rejected on dtype alone;
    _check_dtype's kind-only check would otherwise accept it."""
    fx = _load_p10a(fixtures_dir, "diag")
    setattr(fx, field, getattr(fx, field).astype(np.float32))
    _assert_rejects(fx, f"{field}.dtype")


@pytest.mark.parametrize("field", ["channel_num_states", "updates_epoch_idx", "batch_offsets", "length_observation"])
def test_p10a_int32_dtype_raises(fixtures_dir, field):
    """SCHEMA.md declares int64 specifically -- an int32 fixture must be
    rejected on dtype alone."""
    fx = _load_p10a(fixtures_dir, "diag")
    setattr(fx, field, getattr(fx, field).astype(np.int32))
    _assert_rejects(fx, f"{field}.dtype")


def test_p10a_length_observation_zero_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.length_observation = fx.length_observation.copy()
    fx.length_observation[0] = 0
    _assert_rejects(fx, "length_observation")


@pytest.mark.parametrize("field", ["channel_num_states", "dim_observation", "num_gmm_component"])
def test_p10a_per_channel_field_zero_raises(fixtures_dir, field):
    fx = _load_p10a(fixtures_dir, "diag")
    arr = getattr(fx, field).copy()
    arr[0] = 0
    setattr(fx, field, arr)
    _assert_rejects(fx, field)


def test_p10a_load_rejects_float_dtype_int_scalar(fixtures_dir, tmp_path):
    """A malformed num_trials scalar stored as a float (even an integer-
    valued one) must be rejected outright by the loader itself, not
    silently laundered through int(...) into a plausible-looking value."""
    src_path = Path(fixtures_dir) / "p10a_scheduling__diag.npz"
    with np.load(src_path, allow_pickle=False) as d:
        real_num_trials = float(d["num_trials"])
    tmp_dir = _write_p10a_npz_with_override(fixtures_dir, "diag", tmp_path, num_trials=np.array(real_num_trials))
    with pytest.raises(FixtureValidationError, match="num_trials"):
        load_p10a_scheduling_fixture(tmp_dir, "diag")


def test_p10a_load_rejects_float_dtype_returned_best_epoch_idx_scalar(fixtures_dir, tmp_path):
    src_path = Path(fixtures_dir) / "p10a_scheduling__diag.npz"
    with np.load(src_path, allow_pickle=False) as d:
        real_idx = float(d["returned_best_epoch_idx"])
    tmp_dir = _write_p10a_npz_with_override(fixtures_dir, "diag", tmp_path, returned_best_epoch_idx=np.array(real_idx))
    with pytest.raises(FixtureValidationError, match="returned_best_epoch_idx"):
        load_p10a_scheduling_fixture(tmp_dir, "diag")


# ---- item 6: AIC/BIC identities --------------------------------------------


def test_p10a_AIC_trace_inconsistent_with_recomputed_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.AIC_trace = fx.AIC_trace.copy()
    fx.AIC_trace[0] += 100.0
    _assert_rejects(fx, "AIC_trace")


def test_p10a_BIC_trace_inconsistent_with_recomputed_raises(fixtures_dir):
    fx = _load_p10a(fixtures_dir, "diag")
    fx.BIC_trace = fx.BIC_trace.copy()
    fx.BIC_trace[0] += 100.0
    _assert_rejects(fx, "BIC_trace")
