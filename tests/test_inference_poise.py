"""P12 gate: the POISE variant (`inference/poise.py`) against MATLAB's own
`forward_backward_lsim_poise` / `forward_backward_lsim_kernel(variant='poise')`
output, in both refinement modes.

**Fixtures.** The P12 export added `<scenario>__poise` (raw) and
`<scenario>__poise_refined` fixtures for all 24 canonical scenarios plus the
POISE-only `c3_coupling_zero_row_exact` (50 files in total), exported by the same
`tools/matlab/export_fixtures.m` as the sed/fast grid (MATLAB R2025b; the
pre-existing sed/fast files keep their own recorded R2026a release, and were
neither regenerated nor renamed). Each file records its own
`fbc_poise_refinement`, so no test depends on either language's default.

**Tolerances, measured over all 50 POISE fixtures x 54 trials, not guessed**
(same methodology and the same values as P3a/P3b): `rtol=1e-8, atol=1e-9` for
`alpha`/`beta`/`gamma`/`scale` (worst measured `1.6e-14`, `1.0e-12`, `2.4e-13`,
`1.1e-12` -- the largest on `c3_lowmass_diag`, where the smoothed mass is tiny)
and `rtol=1e-10, atol=1e-8` for `loglik` (worst `1.1e-9` absolute, `3.0e-12`
relative, on `c3_sigma_repair_1eig`). `B`/`B_comp` come from the shared emission
code and are compared relatively, exactly as P3a does.

**What POISE shares and what it changes.** The forward recursion, emissions and
`pi_0` regularization are literally `sed.py`'s, so `loglik` is variant-invariant
at fixed parameters -- asserted here across all three variants. Only the
backward message differs, plus the optional one-shot refinement.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.core.constants import FbConstants  # noqa: E402
from chmm_lsim.models.lsim.inference.registry import forward_backward, get_inference  # noqa: E402
from chmm_lsim.models.lsim.inference.poise import _REFINE_W0, forward_backward_poise, poise_backward_constants  # noqa: E402
from chmm_lsim.models.lsim.inference.sed import forward_backward_sed  # noqa: E402
from chmm_lsim.models.lsim.params import LSIMParams  # noqa: E402
from chmm_lsim.core.structure import LSIMStructure  # noqa: E402

RTOL, ATOL = 1e-8, 1e-9
LOGLIK_RTOL, LOGLIK_ATOL = 1e-10, 1e-8
B_RTOL = 1e-6  # relative-only, as in P3a/P3b: B is an unnormalized density spanning orders of magnitude


def _build(fx):
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    params = LSIMParams.from_packed(
        structure, pi_0=fx.pi_0, transition=fx.transitions_matrices, coupling=fx.coupling_tetha_IM,
        P_all=fx.P_all, mu_all=fx.mu_all, sigma_all=fx.sigma_all, sigma_diag=bool(fx.sigma_diag),
    )
    fbc = FbConstants(
        variant=fx.fb_constants.variant, beta_floor=fx.fb_constants.beta_floor, pi0_reg=fx.fb_constants.pi0_reg,
        f_reg=fx.fb_constants.f_reg, svi_eps=fx.fb_constants.svi_eps,
        poise_refinement=fx.fb_constants.poise_refinement,
    )
    return structure, params, fbc


def _poise_only(posterior_fixture_id):
    if "__poise" not in posterior_fixture_id:
        pytest.skip("sed/fast variants are P3a/P3b, covered by their own modules")


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_forward_backward_poise_matches_fixture(fixtures_dir, posterior_fixture_id):
    """The P12 golden gate: every POISE fixture, every trial, both modes."""
    _poise_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    assert fbc.variant == "poise"
    assert fbc.poise_refinement == posterior_fixture_id.endswith("__poise_refined")

    for i, trial in enumerate(fx.trials):
        post = forward_backward_poise(structure, params, trial.Y, fbc)
        for name, mine, matlab in (
            ("alpha", post.alpha, trial.alpha),
            ("beta", post.beta, trial.beta),
            ("gamma", post.gamma, trial.alpha_T),
            ("scale", post.scale, trial.b_c_ot_nc),
        ):
            np.testing.assert_allclose(
                mine, matlab, rtol=RTOL, atol=ATOL, err_msg=f"{posterior_fixture_id} trial {i}: {name} mismatch")
        np.testing.assert_allclose(
            post.loglik, trial.P_O_model, rtol=LOGLIK_RTOL, atol=LOGLIK_ATOL,
            err_msg=f"{posterior_fixture_id} trial {i}: loglik mismatch")
        np.testing.assert_allclose(post.B, trial.B, rtol=B_RTOL, atol=0)
        np.testing.assert_allclose(post.B_comp, trial.B_comp, rtol=B_RTOL, atol=0)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_poise_reports_no_diagnostic_counters(fixtures_dir, posterior_fixture_id):
    """POISE has no `beta_floor` clamp, no delta solve and no NaN guard, so every
    counter stays zero -- the counters exist on `Diagnostics` for sed/fast."""
    _poise_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    for trial in fx.trials:
        d = forward_backward_poise(structure, params, trial.Y, fbc).diagnostics
        assert (d.zero_alpha_next_count, d.beta_floor_hits, d.f_le_h_violations, d.nan_guard_hits) == (0, 0, 0, 0)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_gamma_blocks_sum_to_one(fixtures_dir, posterior_fixture_id):
    """The smoothed marginals are per-channel distributions at every timestep, in
    both modes -- the refinement mixes two block-normalized vectors with weights
    summing to 1, so it preserves this."""
    _poise_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    for trial in fx.trials:
        gamma = forward_backward_poise(structure, params, trial.Y, fbc).gamma
        for c in range(structure.C):
            block = gamma[structure.state_slice(c), :]
            np.testing.assert_allclose(block.sum(axis=0), 1.0, rtol=0, atol=1e-10)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_loglik_is_variant_invariant_at_fixed_params(fixtures_dir, posterior_fixture_id):
    """POISE shares sed's forward pass verbatim, and `loglik` is a forward-only
    quantity, so all three variants agree on it for the same parameters and data.
    Checked here against `sed` on POISE's own fixtures (the MATLAB numbers back
    this too: a `__poise` and its `__poise_refined` twin record the same
    `P_O_model`)."""
    _poise_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    fbc_sed = FbConstants(variant="sed", beta_floor=fbc.beta_floor, pi0_reg=fbc.pi0_reg, f_reg=fbc.f_reg, svi_eps=fbc.svi_eps)
    for trial in fx.trials:
        poise = forward_backward_poise(structure, params, trial.Y, fbc)
        sed = forward_backward_sed(structure, params, trial.Y, fbc_sed)
        np.testing.assert_allclose(poise.loglik, sed.loglik, rtol=1e-12, atol=0)
        np.testing.assert_allclose(poise.alpha, sed.alpha, rtol=0, atol=1e-12)
        np.testing.assert_allclose(poise.scale, sed.scale, rtol=0, atol=1e-12)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("scenario", ["c3_het_diag", "c3_coupling_zero_row", "c3_het_full_nan", "c3_trials_unequal"])
def test_refinement_changes_only_the_smoothed_quantities(fixtures_dir, scenario):
    """Raw vs refined on the same model and data: the forward-only quantities
    (`alpha`, `scale`, `loglik`, `B`) are untouched, while `gamma`/`beta` differ
    wherever the weight rule is active. The two fixtures are independent MATLAB
    exports, so this also cross-checks the exporter."""
    raw_fx = load_posterior_fixture(fixtures_dir, f"{scenario}__poise")
    ref_fx = load_posterior_fixture(fixtures_dir, f"{scenario}__poise_refined")
    structure, params, fbc_raw = _build(raw_fx)
    _s, _p, fbc_ref = _build(ref_fx)
    assert fbc_raw.poise_refinement is False and fbc_ref.poise_refinement is True

    _rho, theta_out, _zero = poise_backward_constants(structure, params.coupling)
    w1 = np.maximum(0.0, _REFINE_W0 - theta_out)

    for trial_raw, trial_ref in zip(raw_fx.trials, ref_fx.trials):
        raw = forward_backward_poise(structure, params, trial_raw.Y, fbc_raw)
        ref = forward_backward_poise(structure, params, trial_ref.Y, fbc_ref)
        np.testing.assert_allclose(ref.alpha, raw.alpha, rtol=0, atol=0)
        np.testing.assert_allclose(ref.scale, raw.scale, rtol=0, atol=0)
        assert ref.loglik == raw.loglik
        np.testing.assert_allclose(ref.B, raw.B, rtol=0, atol=0)
        # t=0 keeps its raw marginal (the MATLAB refinement loop starts at t=2)
        np.testing.assert_allclose(ref.gamma[:, 0], raw.gamma[:, 0], rtol=0, atol=0)
        if trial_raw.Y.shape[1] > 1 and np.any(w1 > 0):
            assert not np.allclose(ref.gamma[:, 1:], raw.gamma[:, 1:]), "refinement must change the smoothed marginals"


def test_refinement_is_skipped_for_single_timestep_sequences(fixtures_dir):
    """`T == 1` has nothing to propagate, so both modes agree exactly -- the
    MATLAB kernel guards the pass with `T > 1` and so does this port."""
    raw_fx = load_posterior_fixture(fixtures_dir, "c1_T1__poise")
    ref_fx = load_posterior_fixture(fixtures_dir, "c1_T1__poise_refined")
    structure, params, fbc_raw = _build(raw_fx)
    _s, _p, fbc_ref = _build(ref_fx)
    Y = raw_fx.trials[0].Y
    assert Y.shape[1] == 1
    raw = forward_backward_poise(structure, params, Y, fbc_raw)
    ref = forward_backward_poise(structure, params, Y, fbc_ref)
    for name in ("alpha", "beta", "gamma", "scale"):
        np.testing.assert_array_equal(getattr(ref, name), getattr(raw, name), err_msg=name)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("scenario", ["c3_het_diag", "c3_coupling_zero_row"])
def test_refinement_is_one_shot_from_frozen_raw_marginals(fixtures_dir, scenario):
    """Every propagated slice must read the RAW marginals, never an
    already-refined one. Reproducing the pass by hand from the raw run's own
    `gamma` gives exactly the refined `gamma`; a sequential variant that fed each
    refined slice into the next would not."""
    raw_fx = load_posterior_fixture(fixtures_dir, f"{scenario}__poise")
    ref_fx = load_posterior_fixture(fixtures_dir, f"{scenario}__poise_refined")
    structure, params, fbc_raw = _build(raw_fx)
    _s, _p, fbc_ref = _build(ref_fx)
    from chmm_lsim.backends.numpy_ import block_sum, broadcast_block, build_W

    Y = raw_fx.trials[0].Y
    raw = forward_backward_poise(structure, params, Y, fbc_raw)
    ref = forward_backward_poise(structure, params, Y, fbc_ref)

    W = build_W(params.transition, params.coupling, structure.block_of_state)
    _rho, theta_out, _zero = poise_backward_constants(structure, params.coupling)
    w1 = np.maximum(0.0, _REFINE_W0 - theta_out)
    expected = raw.gamma.copy()
    for t in range(1, Y.shape[1]):
        propagated = raw.scale[:, t] * (raw.gamma[:, t - 1] @ W)  # raw, frozen -- not expected[:, t-1]
        propagated = propagated / broadcast_block(block_sum(propagated, structure.state_offsets), structure.block_of_state)
        expected[:, t] = w1 * propagated + (1.0 - w1) * raw.gamma[:, t]
    np.testing.assert_allclose(ref.gamma, expected, rtol=0, atol=1e-15)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_effective_beta_reproduces_gamma_where_alpha_is_positive(fixtures_dir, posterior_fixture_id):
    """With refinement on, `beta` is the reconstructed effective message
    (`gamma/alpha` where `alpha > 0`, else 0), so `gamma == alpha*beta` holds
    exactly where `alpha > 0` and `beta` is 0 elsewhere. With refinement off the
    raw recursion satisfies the identity everywhere."""
    _poise_only(posterior_fixture_id)
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    for trial in fx.trials:
        post = forward_backward_poise(structure, params, trial.Y, fbc)
        positive = post.alpha > 0
        np.testing.assert_allclose(
            (post.alpha * post.beta)[positive], post.gamma[positive], rtol=0, atol=1e-12)
        if fbc.poise_refinement:
            assert np.all(post.beta[~positive] == 0.0)


def test_zero_outgoing_strength_channel_takes_the_observation_term(fixtures_dir):
    """`theta_out == 0` (a channel with no outgoing coupling at all) must use the
    bare observation term, and `rho` for that channel must be all-zero rather
    than `0/0`. `c3_coupling_zero_row_exact` is a P12 addition to the exported
    grid specifically for this branch: the pre-existing `c3_coupling_zero_row`
    uses a 1e-12 row (it was built for sed's `flag_nan` threshold), which only
    approaches the exact-zero case. Both are checked here."""
    fx = load_posterior_fixture(fixtures_dir, "c3_coupling_zero_row_exact__poise")
    structure, params, fbc = _build(fx)
    rho, theta_out, zero_theta = poise_backward_constants(structure, params.coupling)
    assert zero_theta.any(), "the exact-zero scenario must contain a zero-outgoing-strength channel"
    assert np.all(np.isfinite(rho)) and np.all(rho[zero_theta] == 0.0)
    np.testing.assert_allclose(rho[~zero_theta].sum(axis=1), 1.0, rtol=0, atol=1e-15)

    Y = fx.trials[0].Y
    post = forward_backward_poise(structure, params, Y, fbc)
    assert np.all(np.isfinite(post.gamma)) and np.all(np.isfinite(post.beta))
    # the golden MATLAB arrays for this scenario already pin the values; here we pin the
    # branch itself: for a zero-theta_out channel beta_t is the observation term, up to the
    # per-channel normalizer both are divided by.
    for t in range(Y.shape[1] - 1):
        got, expected = post.beta[zero_theta, t], post.scale[zero_theta, t]
        np.testing.assert_allclose(got / got.sum(), expected / expected.sum(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(post.gamma, fx.trials[0].alpha_T, rtol=RTOL, atol=ATOL)


def test_near_zero_outgoing_strength_matches_the_exact_zero_limit(fixtures_dir):
    """`c3_coupling_zero_row`'s 1e-12 row does NOT take the exact-zero branch, and
    must not need to: `q ** 2.7e-12` is 1 to within float64 resolution, so the
    message is the observation term anyway. Both fixtures are golden-verified
    elsewhere; this pins that the two paths agree in the limit."""
    fx = load_posterior_fixture(fixtures_dir, "c3_coupling_zero_row__poise")
    structure, params, fbc = _build(fx)
    _rho, theta_out, zero_theta = poise_backward_constants(structure, params.coupling)
    assert not zero_theta.any(), "this scenario is the NEAR-zero case"
    assert theta_out.min() < 1e-10
    post = forward_backward_poise(structure, params, fx.trials[0].Y, fbc)
    tiny = theta_out < 1e-10
    for t in range(fx.trials[0].Y.shape[1] - 1):
        got, expected = post.beta[tiny, t], post.scale[tiny, t]
        np.testing.assert_allclose(got / got.sum(), expected / expected.sum(), rtol=1e-9, atol=0)


# --------------------------------------------------------------------------
# API strictness and registry wiring.
# --------------------------------------------------------------------------


def test_poise_kernel_rejects_other_variants(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise")
    structure, params, _fbc = _build(fx)
    for other in ("sed", "fast"):
        with pytest.raises(ValueError, match="variant='poise'"):
            forward_backward_poise(structure, params, fx.trials[0].Y, FbConstants(variant=other))


def test_other_kernels_reject_poise_constants(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise")
    structure, params, fbc = _build(fx)
    from chmm_lsim.models.lsim.inference.fast import forward_backward_fast

    with pytest.raises(ValueError, match="variant='sed'"):
        forward_backward_sed(structure, params, fx.trials[0].Y, fbc)
    with pytest.raises(ValueError, match="variant='fast'"):
        forward_backward_fast(structure, params, fx.trials[0].Y, fbc)


def test_poise_is_registered_and_dispatched_by_variant(fixtures_dir):
    assert get_inference("poise") is forward_backward_poise
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise_refined")
    structure, params, fbc = _build(fx)
    via_registry = forward_backward(structure, params, fx.trials[0].Y, fbc)
    direct = forward_backward_poise(structure, params, fx.trials[0].Y, fbc)
    np.testing.assert_array_equal(via_registry.gamma, direct.gamma)


def test_refinement_default_is_off_and_matches_matlabs_default(fixtures_dir):
    """Both sides default `poise_refinement` to False (the P12 correction round
    aligned MATLAB's `lsim_fb_constants.m`, `forward_backward_lsim_poise.m` and
    `em_lsim_poise.m` on the same value, and `tools/matlab/check_em_lsim_poise.m`
    asserts it there). The default must therefore reproduce the RAW fixture, not
    the refined one."""
    assert FbConstants().variant == "poise"
    assert FbConstants().poise_refinement is False
    raw_fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise")
    ref_fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise_refined")
    structure, params, _fbc = _build(raw_fx)
    default = forward_backward_poise(structure, params, raw_fx.trials[0].Y, FbConstants())
    np.testing.assert_allclose(default.gamma, raw_fx.trials[0].alpha_T, rtol=RTOL, atol=ATOL)
    assert not np.allclose(default.gamma, ref_fx.trials[0].alpha_T, rtol=RTOL, atol=ATOL)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_validate_all_accepts_valid_fixtures_and_off_skips_checks(fixtures_dir):
    for fid in ("c3_het_diag__poise", "c3_het_diag__poise_refined", "c3_coupling_zero_row__poise_refined"):
        fx = load_posterior_fixture(fixtures_dir, fid)
        structure, params, fbc = _build(fx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strict = forward_backward_poise(structure, params, fx.trials[0].Y, fbc, validate="all")
            off = forward_backward_poise(structure, params, fx.trials[0].Y, fbc, validate="off")
        np.testing.assert_array_equal(strict.gamma, off.gamma)


def test_validate_all_catches_a_corrupted_refined_gamma(fixtures_dir, monkeypatch):
    """The refinement pass runs its own block-sum check under `validate="all"`:
    corrupting the propagated slice must be reported, not silently returned."""
    import chmm_lsim.models.lsim.inference.poise as poise_mod

    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__poise_refined")
    structure, params, fbc = _build(fx)
    real_block_sum = poise_mod.block_sum
    calls = {"n": 0}

    def corrupt(values, offsets, axis=0):
        out = real_block_sum(values, offsets, axis=axis)
        calls["n"] += 1
        if calls["n"] > 1 and out.ndim == 1:  # a normalizer inside the refinement pass
            return out * 2.0
        return out

    monkeypatch.setattr(poise_mod, "block_sum", corrupt)
    from chmm_lsim.core.validate import ValidationError

    with pytest.raises(ValidationError, match="gamma"):
        forward_backward_poise(structure, params, fx.trials[0].Y, fbc, validate="all")
