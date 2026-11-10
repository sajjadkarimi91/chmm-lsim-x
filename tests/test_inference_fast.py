"""P3b gate: NumPy fast/SVI forward-backward against the fast half of the 48
posterior fixtures, shared-forward-pass loglik parity with sed, and the
registry/dispatcher (`inference/registry.py`).

Tolerances, measured against the real fixture set (not guessed), same
methodology as P3a: `rtol=1e-8, atol=1e-9` for `alpha`/`beta`/`gamma`/`scale`
and `rtol=1e-10, atol=1e-8` for `loglik` pass every one of the 24 `fast`
fixtures x trial with zero failures (worst observed relative error `2.1e-12`
on `beta`/`gamma`/`scale`, `4.2e-11` on `loglik`) -- the same tolerances P3a
used for `sed`, not a separately loosened bound.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.models.lsim.inference.fast import forward_backward_fast
from chmm_lsim.models.lsim.inference.registry import forward_backward, get_inference, register_inference
from chmm_lsim.models.lsim.inference.sed import forward_backward_sed
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure

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


def _fast_only(posterior_fixture_id):
    if not posterior_fixture_id.endswith("__fast"):
        pytest.skip("sed variant is P3a, covered by test_inference_sed.py")


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_forward_backward_fast_matches_fixture(fixtures_dir, posterior_fixture_id):
    """Every committed `__fast.npz` fixture, every trial -- covers C=1
    (`c1_minimal`, `c1_T1`), T=1 (`c1_T1`), diagonal and full covariance,
    heterogeneous M/d/K (`c3_het_*`, `c5_*`), NaNs (`c2_diag_d1_nan`,
    `c3_nan_timestep`, `c3_channel_all_nan`, `c3_het_full_nan`, `c3_fast_nan`),
    sparse/left-to-right transitions (`c3_sparse_l2r`), zero/degenerate
    coupling (`c3_coupling_zero_row`, `c3_coupling_degenerate`), and unequal
    multi-trial bookkeeping (`c3_trials_unequal`) -- by construction, since
    `posterior_fixture_id` iterates every fixture in the manifest."""
    _fast_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for i, trial in enumerate(fx.trials):
        post = forward_backward_fast(structure, params, trial.Y, fbc)
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
        from chmm_lsim.emissions import compute_emissions

        expected_B, expected_B_comp = compute_emissions(structure, params, trial.Y)
        np.testing.assert_array_equal(post.B, expected_B)
        np.testing.assert_array_equal(post.B_comp, expected_B_comp)


def test_sed_fb_constants_rejected_by_fast(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__fast")
    structure, params, _ = _build(fx)
    with pytest.raises(ValueError, match="variant='fast'"):
        forward_backward_fast(structure, params, fx.trials[0].Y, FbConstants(variant="sed"))


def test_fast_fb_constants_rejected_by_sed(fixtures_dir):
    """The reverse of P3a's own `test_fast_fb_constants_rejected` -- both
    kernels are strict about which `FbConstants.variant` they accept."""
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__fast")
    structure, params, fbc = _build(fx)
    with pytest.raises(ValueError, match="variant='sed'"):
        forward_backward_sed(structure, params, fx.trials[0].Y, fbc)


def test_validate_all_does_not_spuriously_raise_and_off_skips_checks(fixtures_dir):
    """The block-normalization and non-finite checks applicable to fast:
    `validate="all"` checks per-block sums of `alpha` (inside the shared,
    imported `_forward_pass`), `abar` (fast's own backward loop, computed
    purely for this invariant -- fast's beta formula itself never needs
    `abar`; only sed's `P_cond` does), and `gamma` -- fast has no `f`/`h`/
    `P_cond`/delta solve to additionally check, unlike sed."""
    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__fast")
    structure, params, fbc = _build(fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_all = forward_backward_fast(structure, params, fx.trials[0].Y, fbc, validate="all")
        post_off = forward_backward_fast(structure, params, fx.trials[0].Y, fbc, validate="off")
    np.testing.assert_allclose(post_all.alpha, post_off.alpha)
    np.testing.assert_allclose(post_all.gamma, post_off.gamma)


def test_validate_all_catches_corrupted_alpha_block_sum(fixtures_dir):
    """Same regression shape as sed's, but corrupting the *last* timestep's
    `alpha` rather than the first: fast's backward loop never *uses* `abar`
    (that's sed-only, feeding `P_cond`) in its own beta formula, and
    self-normalizes every `gamma[:,tau]` it derives except the terminal,
    unnormalized `gamma[:,T-1] = alpha[:,T-1] * beta[:,T-1]`. A corruption at
    `t=0` is now caught by the `abar` check below instead
    (`test_validate_all_catches_corrupted_abar_block_sum`), since `abar`
    still reads `alpha[:,tau]` directly with no normalization in between --
    so *this* test specifically targets `t=T-1`, the one point `abar` is
    never computed (the backward loop only runs `tau` from `T-2` to `0`) and
    `gamma` is not self-normalized either, isolating the terminal-`gamma`
    check instead."""
    from unittest import mock

    import chmm_lsim.models.lsim.inference.fast as fast_module
    import chmm_lsim.models.lsim.inference.sed as sed_module
    from chmm_lsim.core.validate import ValidationError

    fx = load_posterior_fixture(fixtures_dir, "c1_minimal__fast")
    structure, params, fbc = _build(fx)
    real_forward_pass = sed_module._forward_pass

    def corrupted_forward_pass(pi_0, W, B, structure, **kwargs):
        alpha, scale, loglik = real_forward_pass(pi_0, W, B, structure, validate="off")
        alpha = alpha.copy()
        alpha[:, -1] *= 2.0
        return alpha, scale, loglik

    with mock.patch("chmm_lsim.models.lsim.inference.fast._forward_pass", corrupted_forward_pass):
        with pytest.raises(ValidationError, match="per-block sums"):
            fast_module.forward_backward_fast(structure, params, fx.trials[0].Y, fbc, validate="all")


def test_validate_all_catches_corrupted_abar_block_sum(fixtures_dir):
    """§0.2.1(i)'s `abar` invariant, checked at every backward-relevant
    timestep (`tau=T-2..0`), matching P3a's contract -- previously missing
    from fast.py entirely (fast's own beta formula has no use for `abar`, but
    the generic invariant still holds and P3a checks it regardless of
    variant). Corrupting `scale` rather than `alpha` isolates it precisely:
    `alpha`'s own per-block-sum-1 invariant is untouched (it never reads
    `scale`), but `abar = alpha * scale` is not, since `scale` feeds directly
    into it."""
    from unittest import mock

    import chmm_lsim.models.lsim.inference.fast as fast_module
    import chmm_lsim.models.lsim.inference.sed as sed_module
    from chmm_lsim.core.validate import ValidationError

    fx = load_posterior_fixture(fixtures_dir, "c1_minimal__fast")
    structure, params, fbc = _build(fx)
    real_forward_pass = sed_module._forward_pass

    def corrupted_forward_pass(pi_0, W, B, structure, **kwargs):
        alpha, scale, loglik = real_forward_pass(pi_0, W, B, structure, validate="off")
        scale = scale.copy()
        scale[:, 0] *= 2.0  # alpha's own block-sum-1 invariant is untouched by this
        return alpha, scale, loglik

    with mock.patch("chmm_lsim.models.lsim.inference.fast._forward_pass", corrupted_forward_pass):
        with pytest.raises(ValidationError, match="abar"):
            fast_module.forward_backward_fast(structure, params, fx.trials[0].Y, fbc, validate="all")


def test_posterior_channel_view(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__fast")
    structure, params, fbc = _build(fx)
    post = forward_backward_fast(structure, params, fx.trials[0].Y, fbc)

    ch1 = post.channel(1)
    sl = structure.state_slice(1)
    np.testing.assert_array_equal(ch1.alpha, post.alpha[sl])
    np.testing.assert_array_equal(ch1.beta, post.beta[sl])
    np.testing.assert_array_equal(ch1.gamma, post.gamma[sl])
    assert ch1.loglik == post.loglik
    np.testing.assert_array_equal(post.B_raw(), post.B)  # no emission rescaling here either


# --------------------------------------------------------------------------
# fast vs sed: identical forward pass -> identical loglik at fixed params
# (docs/PYTHON_PORT_PLAN.md §3bis.3), even though beta/gamma legitimately
# differ (different backward message). Not a MATLAB fixture comparison --
# both kernels are run here on the *same* structure/params/Y, only the
# fb_constants.variant differs, isolating the claim to "shared forward pass".
# --------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_fast_and_sed_loglik_identical_at_fixed_params(fixtures_dir, posterior_fixture_id):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    fbc_sed = FbConstants(variant="sed", beta_floor=fbc.beta_floor, pi0_reg=fbc.pi0_reg, f_reg=fbc.f_reg, svi_eps=fbc.svi_eps)
    fbc_fast = FbConstants(variant="fast", beta_floor=fbc.beta_floor, pi0_reg=fbc.pi0_reg, f_reg=fbc.f_reg, svi_eps=fbc.svi_eps)

    for trial in fx.trials:
        post_sed = forward_backward_sed(structure, params, trial.Y, fbc_sed)
        post_fast = forward_backward_fast(structure, params, trial.Y, fbc_fast)
        assert post_sed.loglik == post_fast.loglik


# --------------------------------------------------------------------------
# registry.py: registration, lookup, and the forward_backward(...) dispatcher.
# --------------------------------------------------------------------------

def test_get_inference_returns_registered_kernels():
    assert get_inference("sed") is forward_backward_sed
    assert get_inference("fast") is forward_backward_fast


def test_get_inference_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown inference variant"):
        get_inference("svi")


def test_register_inference_duplicate_raises():
    @register_inference("__test_dummy_variant__")
    def _dummy(*args, **kwargs):
        raise NotImplementedError

    try:
        with pytest.raises(ValueError, match="already registered"):
            @register_inference("__test_dummy_variant__")
            def _dummy2(*args, **kwargs):
                raise NotImplementedError
    finally:
        from chmm_lsim.models.lsim.inference.registry import _REGISTRY

        del _REGISTRY["__test_dummy_variant__"]


def test_forward_backward_dispatches_by_variant(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc_sed = _build(fx)
    fx_fast = load_posterior_fixture(fixtures_dir, "c3_het_diag__fast")
    _, params_fast, fbc_fast = _build(fx_fast)

    post_sed = forward_backward(structure, params, fx.trials[0].Y, fbc_sed)
    direct_sed = forward_backward_sed(structure, params, fx.trials[0].Y, fbc_sed)
    np.testing.assert_array_equal(post_sed.alpha, direct_sed.alpha)
    np.testing.assert_array_equal(post_sed.beta, direct_sed.beta)

    post_fast = forward_backward(structure, params_fast, fx_fast.trials[0].Y, fbc_fast)
    direct_fast = forward_backward_fast(structure, params_fast, fx_fast.trials[0].Y, fbc_fast)
    np.testing.assert_array_equal(post_fast.alpha, direct_fast.alpha)
    np.testing.assert_array_equal(post_fast.beta, direct_fast.beta)


def test_forward_backward_default_variant_is_raw_poise(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise")
    structure, params, _ = _build(fx)
    post_default = forward_backward(structure, params, fx.trials[0].Y)  # fb_constants=None
    from chmm_lsim.models.lsim.inference.poise import forward_backward_poise

    post_explicit = forward_backward_poise(structure, params, fx.trials[0].Y, FbConstants())
    np.testing.assert_array_equal(post_default.alpha, post_explicit.alpha)


def test_forward_backward_passes_through_kwargs(fixtures_dir):
    """`method=` is sed-only, but the dispatcher must still forward it."""
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    post_ref = forward_backward(structure, params, fx.trials[0].Y, fbc, method="reference")
    post_opt = forward_backward_sed(structure, params, fx.trials[0].Y, fbc, method="reference")
    np.testing.assert_array_equal(post_ref.alpha, post_opt.alpha)


def test_forward_backward_rejects_unsupported_backend(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    with pytest.raises(ValueError, match="backend"):
        forward_backward(structure, params, fx.trials[0].Y, fbc, backend="tensorflow")
