"""P3a gate: NumPy SED forward-backward against the sed half of the 48
posterior fixtures, `reference` vs `optimized` agreement, SED diagnostics, and
the §C invariants (f>h, per-block P_cond sums, V2=1).

Tolerances, measured against the real fixture set (not guessed), same
methodology as P2's emissions calibration (docs/PYTHON_PORT_PLAN.md §3.A):
`rtol=1e-8, atol=1e-9` for `alpha`/`beta`/`gamma`/`scale` passes every sed
fixture x trial with zero failures; the worst case is on `c3_sparse_l2r`
(the fixture built specifically to exercise `alpha_next==0`, §0.3.3(4)) and
`c3_lowmass_diag`/`c3_sigma_repair_2eig` (the near-zero-mass / near-singular
fixtures) -- exactly where a bigger error is expected and honest, not a sign
of a bug.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.backends.numpy_ import block_sum, broadcast_block, build_W
from chmm_lsim.compat import FbConstants
from chmm_lsim.inference.sed import forward_backward_sed, reconstruct_sed_diagnostics
from chmm_lsim.params import LSIMParams
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError

RTOL, ATOL = 1e-8, 1e-9


def _build(fx):
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
    fbc = FbConstants(
        variant=fx.fb_constants.variant,
        beta_floor=fx.fb_constants.beta_floor,
        pi0_reg=fx.fb_constants.pi0_reg,
        f_reg=fx.fb_constants.f_reg,
        svi_eps=fx.fb_constants.svi_eps,
        poise_refinement=fx.fb_constants.poise_refinement,
    )
    return structure, params, fbc


def _sed_only(posterior_fixture_id):
    if not posterior_fixture_id.endswith("__sed"):
        pytest.skip("fast variant is P3b, not P3a")


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_forward_backward_sed_matches_fixture(fixtures_dir, posterior_fixture_id):
    _sed_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for i, trial in enumerate(fx.trials):
        post = forward_backward_sed(structure, params, trial.Y, fbc)
        for name, mine, matlab in [
            ("alpha", post.alpha, trial.alpha),
            ("beta", post.beta, trial.beta),
            ("gamma", post.gamma, trial.alpha_T),
            ("scale", post.scale, trial.b_c_ot_nc),
        ]:
            np.testing.assert_allclose(
                mine, matlab, rtol=RTOL, atol=ATOL, err_msg=f"{posterior_fixture_id} trial {i}: {name} mismatch"
            )
        np.testing.assert_allclose(
            post.loglik, trial.P_O_model, rtol=1e-10, atol=1e-8, err_msg=f"{posterior_fixture_id} trial {i}: loglik mismatch"
        )
        # B/B_comp parity against MATLAB is P2's gate (tests/test_emissions.py, with its
        # own calibrated tolerance) -- here just confirm forward_backward_sed's internal
        # call to compute_emissions is deterministic and reproduces the same B/B_comp it
        # would if called standalone, not a second MATLAB comparison.
        from chmm_lsim.emissions import compute_emissions

        expected_B, expected_B_comp = compute_emissions(structure, params, trial.Y)
        np.testing.assert_array_equal(post.B, expected_B)
        np.testing.assert_array_equal(post.B_comp, expected_B_comp)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_reference_and_optimized_methods_agree(fixtures_dir, posterior_fixture_id):
    _sed_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for trial in fx.trials:
        post_opt = forward_backward_sed(structure, params, trial.Y, fbc, method="optimized")
        post_ref = forward_backward_sed(structure, params, trial.Y, fbc, method="reference")
        np.testing.assert_allclose(post_opt.alpha, post_ref.alpha, rtol=0, atol=1e-9)
        np.testing.assert_allclose(post_opt.beta, post_ref.beta, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(post_opt.gamma, post_ref.gamma, rtol=1e-9, atol=1e-9)
        assert post_opt.loglik == post_ref.loglik


def test_unknown_method_raises(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    with pytest.raises(ValueError, match="method"):
        forward_backward_sed(structure, params, fx.trials[0].Y, fbc, method="bogus")


def test_fast_fb_constants_rejected(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, _ = _build(fx)
    with pytest.raises(ValueError, match="variant='sed'"):
        forward_backward_sed(structure, params, fx.trials[0].Y, FbConstants(variant="fast"))


def test_validate_all_does_not_spuriously_raise_and_off_skips_checks(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__sed")
    structure, params, fbc = _build(fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_all = forward_backward_sed(structure, params, fx.trials[0].Y, fbc, validate="all")
        post_off = forward_backward_sed(structure, params, fx.trials[0].Y, fbc, validate="off")
    assert post_all.diagnostics.f_le_h_violations == 0
    np.testing.assert_allclose(post_all.alpha, post_off.alpha)


def test_posterior_channel_view_and_b_raw(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    post = forward_backward_sed(structure, params, fx.trials[0].Y, fbc)

    ch1 = post.channel(1)
    sl = structure.state_slice(1)
    np.testing.assert_array_equal(ch1.alpha, post.alpha[sl])
    np.testing.assert_array_equal(ch1.beta, post.beta[sl])
    np.testing.assert_array_equal(ch1.gamma, post.gamma[sl])
    assert ch1.loglik == post.loglik

    # no emission rescaling in P3a -- B_raw is a no-op
    np.testing.assert_array_equal(post.B_raw(), post.B)


def test_validate_all_catches_corrupted_alpha_block_sum(fixtures_dir):
    """Regression test: validate="all" previously checked only f<=h, so a
    corrupted alpha (per-block sum != 1, §0.2.1(i)) passed silently -- alpha
    itself, plus abar/gamma/V2 derived from it, are now checked at every t."""
    import chmm_lsim.inference.sed as sed_module

    fx = load_posterior_fixture(fixtures_dir, "c1_minimal__sed")
    structure, params, fbc = _build(fx)
    real_forward_pass = sed_module._forward_pass

    def corrupted_forward_pass(pi_0, W, B, structure, **kwargs):
        alpha, scale, loglik = real_forward_pass(pi_0, W, B, structure, validate="off")
        alpha = alpha.copy()
        alpha[:, 0] *= 2.0  # break the per-block-sum-1 invariant
        return alpha, scale, loglik

    with mock.patch("chmm_lsim.inference.sed._forward_pass", corrupted_forward_pass):
        with pytest.raises(ValidationError, match="per-block sums"):
            sed_module.forward_backward_sed(structure, params, fx.trials[0].Y, fbc, validate="all")


def test_zero_alpha_next_diagnostic_is_nonzero_on_the_fixture_built_for_it(fixtures_dir):
    """c3_sparse_l2r is the sparse/left-to-right, decoupled, pi0_reg=0 fixture
    added specifically to exercise alpha_next==0 (§0.3.3(4))."""
    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__sed")
    structure, params, fbc = _build(fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = forward_backward_sed(structure, params, fx.trials[0].Y, fbc)
    assert post.diagnostics.zero_alpha_next_count > 0


# --------------------------------------------------------------------------
# SED diagnostics: reconstructed f/h/delta/Gz vs the fixture's diag_* arrays
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id", ["c3_het_diag", "c3_sparse_l2r"])
def test_sed_diagnostics_match_fixture(fixtures_dir, scenario_id):
    fx = load_posterior_fixture(fixtures_dir, f"{scenario_id}__sed")
    structure, params, fbc = _build(fx)
    trial = fx.trials[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = forward_backward_sed(structure, params, trial.Y, fbc)
    diag = fx.sed_diagnostics
    f, h, delta, Gz = reconstruct_sed_diagnostics(structure, params, post.alpha, post.scale, fbc, diag.t_samples)
    np.testing.assert_allclose(f, diag.f, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(h, diag.h, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(delta, diag.delta, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(Gz, diag.Gz, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# §C invariants -- proven in docs/PYTHON_PORT_PLAN.md §0.2.1/§0.3.3, checked
# here against every real sed fixture rather than assumed.
# --------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_f_greater_than_h_invariant(fixtures_dir, posterior_fixture_id):
    """§0.3.3: f[z,w] >= h[z] + f_reg strictly, for every t,z,w -- proven from
    theta's columns summing to 1 and every transition block's rows summing to 1."""
    _sed_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    if structure.C < 2:
        pytest.skip("the delta solver only runs for C>1")

    for trial in fx.trials:
        post = forward_backward_sed(structure, params, trial.Y, fbc)
        T = trial.Y.shape[1]
        if T < 2:
            continue
        f, h, _, _ = reconstruct_sed_diagnostics(structure, params, post.alpha, post.scale, fbc, np.arange(T - 1))
        assert np.all(f > h[:, :, None])  # f is (n,C,C) rows=z cols=w; h is (n,C) indexed by z


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_v2_identity(fixtures_dir, posterior_fixture_id):
    """§0.2.1(ii): V2[w] = sum_{j in w} alpha_next[j]*beta_next[j] == 1 identically."""
    _sed_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for trial in fx.trials:
        post = forward_backward_sed(structure, params, trial.Y, fbc)
        T = trial.Y.shape[1]
        for t in range(T - 1):
            v2 = block_sum(post.alpha[:, t + 1] * post.beta[:, t + 1], structure.state_offsets)
            np.testing.assert_allclose(v2, 1.0, rtol=0, atol=1e-8)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_pcond_target_block_sums_to_one(fixtures_dir, posterior_fixture_id):
    """§0.3.3 step 3: sum_{j in w} P_cond[i,j] == 1 for every row i, every
    target block w -- where theta's columns summing to 1 is load-bearing."""
    _sed_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for trial in fx.trials:
        post = forward_backward_sed(structure, params, trial.Y, fbc)
        T = trial.Y.shape[1]
        if T < 2:
            continue
        W = build_W(params.transition, params.coupling, structure.block_of_state)
        for t in range(T - 1):
            abar = post.alpha[:, t] * post.scale[:, t]
            alpha_next = post.alpha[:, t + 1]
            Gz = block_sum(abar[:, None] * W, structure.state_offsets, axis=0)
            p_cond = W + alpha_next[None, :] - broadcast_block(Gz, structure.block_of_state, axis=0)
            row_block_sums = block_sum(p_cond, structure.state_offsets, axis=1)  # (S,C)
            np.testing.assert_allclose(row_block_sums, 1.0, rtol=0, atol=1e-8)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_per_block_alpha_and_gamma_sums_to_one(fixtures_dir, posterior_fixture_id):
    """§0.2.1(i): per-block sums of alpha and gamma (alpha_T) are 1."""
    _sed_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for trial in fx.trials:
        post = forward_backward_sed(structure, params, trial.Y, fbc)
        alpha_block_sums = block_sum(post.alpha, structure.state_offsets, axis=0)
        gamma_block_sums = block_sum(post.gamma, structure.state_offsets, axis=0)
        np.testing.assert_allclose(alpha_block_sums, 1.0, rtol=0, atol=1e-8)
        np.testing.assert_allclose(gamma_block_sums, 1.0, rtol=0, atol=1e-8)
