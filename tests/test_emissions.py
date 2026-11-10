"""Emission parity tests against every real MATLAB fixture (P2 gate): B and
B_comp must match `P_observ_cond_to_state[_comp]` on every one of the 24x2
posterior fixtures, for every trial.

Tolerances are relative with atol=0 (docs/PYTHON_PORT_PLAN.md §3.A's
"Tolerances" table), measured against the real fixture set rather than
guessed. **atol must be 0, not a small floor**: B/B_comp values are ordinary
(often large) density values, not quantities that sit near a beta/eps floor,
so a nonzero atol swamps the rtol it's supposedly just "absorbing near-zero
values for" -- e.g. atol=1e-10 against a density of ~1 makes the effective
tolerance ~1e-10, two orders of magnitude looser than the rtol=1e-12 the test
claims to enforce. Exact-zero reference positions (there are real ones -- a
structurally-impossible state, or an underflowed density) are asserted exactly
equal separately, since rtol=0/atol=0 can never pass a nonzero-vs-zero pair:
  - diagonal covariance: worst observed *relative* error (atol=0) is 2.09e-12
    (c3_lowmass_diag) -> rtol=1e-11, roughly 5x margin. (A prior version of
    this file measured 1.3e-13 with atol=1e-10 in the comparison and mislabeled
    that as "relative error" -- it was the absolute error, hidden below the
    atol floor; rtol=1e-12 with atol=0 actually fails on c3_lowmass_diag.)
  - full covariance: worst observed relative error (atol=0) is 4.59e-9, on
    c3_sigma_repair_2eig (cond(sigma) ~2.8e7 by construction -- see SCHEMA.md
    and §0.4.10) -> rtol=1e-7, about 20x margin. This replaces §3.A's
    placeholder "1e-10, to be calibrated": the measured worst case is ~46x
    looser than that placeholder (4.59e-9 / 1e-10), not the four-orders-of-
    magnitude gap an earlier version of this note claimed -- that earlier
    number came from comparing the wrong pair of quantities. The placeholder
    was still wrong (assumed a closed form with no accumulation, which full
    covariance does not have: scipy's Cholesky/eigendecomposition order
    differs from MATLAB's mvnpdf, and the error is dominated by cond(sigma),
    not by d_c) -- just wrong by ~46x, not ~10000x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.emissions import compute_emissions, gmm_pdf_diag, gmm_pdf_full
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure

DIAG_RTOL = 1e-11
FULL_RTOL = 1e-7


def _build_structure_and_params(fx):
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
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
    return structure, params


def test_emissions_match_fixture(fixtures_dir, posterior_fixture_id):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params = _build_structure_and_params(fx)
    params.validate()

    rtol = DIAG_RTOL if fx.sigma_diag else FULL_RTOL
    for i, trial in enumerate(fx.trials):
        B, B_comp = compute_emissions(structure, params, trial.Y)
        assert B.shape == trial.B.shape
        assert B_comp.shape == trial.B_comp.shape

        # rtol/atol=0 can never pass a nonzero-vs-exact-zero pair, so check
        # exact-zero reference positions separately, then rtol-only (atol=0)
        # everywhere else -- an atol floor here would silently loosen the rtol
        # this test claims to enforce (see module docstring).
        for mine, matlab, label in [(B, trial.B, "B"), (B_comp, trial.B_comp, "B_comp")]:
            zero_mask = matlab == 0
            np.testing.assert_array_equal(
                mine[zero_mask], 0.0, err_msg=f"{posterior_fixture_id} trial {i}: {label} nonzero where MATLAB is exactly 0"
            )
            nonzero_mask = ~zero_mask
            np.testing.assert_allclose(
                mine[nonzero_mask],
                matlab[nonzero_mask],
                rtol=rtol,
                atol=0,
                err_msg=f"{posterior_fixture_id} trial {i}: {label} mismatch",
            )


def test_compute_emissions_dispatches_diag(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params = _build_structure_and_params(fx)
    Y = fx.trials[0].Y
    dispatched_B, dispatched_Bc = compute_emissions(structure, params, Y)
    direct_B, direct_Bc = gmm_pdf_diag(structure, params, Y)
    np.testing.assert_array_equal(dispatched_B, direct_B)
    np.testing.assert_array_equal(dispatched_Bc, direct_Bc)


def test_compute_emissions_dispatches_full(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_full__sed")
    structure, params = _build_structure_and_params(fx)
    Y = fx.trials[0].Y
    dispatched_B, dispatched_Bc = compute_emissions(structure, params, Y)
    direct_B, direct_Bc = gmm_pdf_full(structure, params, Y)
    np.testing.assert_array_equal(dispatched_B, direct_B)
    np.testing.assert_array_equal(dispatched_Bc, direct_Bc)


def test_gmm_pdf_diag_rejects_full_covariance_params(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_full__sed")
    structure, params = _build_structure_and_params(fx)
    with pytest.raises(ValueError, match="sigma_diag=True"):
        gmm_pdf_diag(structure, params, fx.trials[0].Y)


def test_gmm_pdf_full_rejects_diag_covariance_params(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params = _build_structure_and_params(fx)
    with pytest.raises(ValueError, match="sigma_diag=False"):
        gmm_pdf_full(structure, params, fx.trials[0].Y)


def test_fully_missing_channel_timestep_contributes_likelihood_one():
    """§0.4.9 / §3bis.2: a channel missing in every dimension at a timestep
    contributes B=1 for every state at that timestep, not 0 or NaN."""
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    gmm = ChannelGMM(
        weights=np.array([[1.0], [1.0]]),
        mu=np.array([[[1.0], [2.0]], [[1.0], [2.0]]]).reshape(2, 2, 1),
        sigma=np.ones((2, 2, 1)),
        diag=True,
    )
    params = LSIMParams(structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.eye(2), coupling=np.ones((1, 1)), emissions=[gmm])
    Y = np.array([[np.nan, 0.0], [np.nan, 0.0]])
    B, _ = gmm_pdf_diag(structure, params, Y)
    np.testing.assert_allclose(B[:, 0], [1.0, 1.0])


def test_fully_missing_channel_timestep_full_covariance_contributes_likelihood_one():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    sigma = np.stack([np.eye(2), np.eye(2)], axis=-1)[:, :, :, None]
    gmm = ChannelGMM(weights=np.array([[1.0], [1.0]]), mu=np.zeros((2, 2, 1)), sigma=sigma, diag=False)
    params = LSIMParams(structure=structure, pi_0=np.array([0.5, 0.5]), transition=np.eye(2), coupling=np.ones((1, 1)), emissions=[gmm])
    Y = np.array([[np.nan, 0.0], [np.nan, 0.0]])
    B, _ = gmm_pdf_full(structure, params, Y)
    np.testing.assert_allclose(B[:, 0], [1.0, 1.0])
