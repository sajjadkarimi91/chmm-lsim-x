"""P7 gate: NumPy vs Numba parity on every one of the 48 posterior fixtures
(both `sed` and `fast`, every trial) -- `loglik`, `alpha`/`beta`/`gamma`/
`scale`, `B`/`B_comp`, and every `Diagnostics` counter. Also: `validate="all"`
genuinely checks something on the numba path (not a no-op), `backend="numba"`
propagates through `em.e_step`/`em.em_lsim` (not just `forward_backward`
directly), and representative diagonal/full-covariance/multi-trial `em_lsim`
runs match NumPy, proving EM actually uses the compiled kernels end to end.

**Tolerances, measured against every fixture x trial, not guessed --
one ordinary `rtol=1e-9, atol=1e-9` gate, applied uniformly, no per-fixture
carve-out.** A review round found an earlier version of this file used a
special `atol=1e-4` for `c3_sparse_l2r__sed`'s `beta`/`gamma`, reasoning from
a *relative*-error measurement (`~5.3e-5`) as if it were an absolute one --
`np.allclose`'s formula is `abs(a-b) <= atol + rtol*abs(b)`, so a `~5.3e-5`
*relative* error at a tiny value (measured: `~4.05e-8`) is an *absolute*
difference of only `~4.53e-10` -- comfortably inside the ordinary
`atol=1e-9` gate already (confirmed directly: `np.allclose(..., rtol=1e-9,
atol=1e-9)` returns `True` for this exact fixture, with no special-casing).
`c3_sparse_l2r`'s backward pass does hit a genuinely small `f-h` gap
(`~1e-6`, `f_reg`-dominated) at one timestep, which does amplify the
`~1e-16` floating-point-order difference between NumPy's vectorized
block-summing and numba's explicit loop nest into that `~5.3e-5` *relative*
outlier -- worth noting as the honest explanation for where the largest
measured discrepancy comes from, but it never needed a loosened *absolute*
tolerance to pass. `alpha`/`scale`/`loglik`/`B`/`B_comp`/`beta`/`gamma`
(worst observed relative error `~9e-16` for the forward-pass-only
quantities, `~5.3e-5` relative / `~4.53e-10` absolute for `beta`/`gamma` on
`c3_sparse_l2r__sed`) all pass the same `rtol=1e-9, atol=1e-9` gate on every
one of the 48 fixtures x every trial, no exceptions.

**Numba is also checked directly against the committed MATLAB posterior
arrays**, not only indirectly through NumPy -- reusing P3a/P3b's own
`rtol=1e-8, atol=1e-9` gate (`test_inference_sed.py`/`test_inference_fast.py`)
on `trial.alpha`/`trial.beta`/`trial.alpha_T`/`trial.b_c_ot_nc`, so a bug
that happened to move *both* NumPy and numba away from MATLAB by the same
amount (which a NumPy-vs-numba-only comparison could never catch) is still
caught here.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip("numba")

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.backends.base import UnsupportedCombination, run_backend
from chmm_lsim.compat import FbConstants
from chmm_lsim.em import e_step, em_lsim, finalize_expected_stats, m_step
from chmm_lsim.inference.sed import forward_backward_sed
from chmm_lsim.init import initialize_lsim_params
from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.simulate import simulate_lsim
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError

RTOL, ATOL = 1e-9, 1e-9
# P3a/P3b's own NumPy-vs-MATLAB tolerance, reused below for the direct
# numba-vs-MATLAB parity check.
MATLAB_RTOL, MATLAB_ATOL = 1e-8, 1e-9


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


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_numpy_vs_numba_matches_on_every_fixture_and_trial(fixtures_dir, posterior_fixture_id):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for i, trial in enumerate(fx.trials):
        post_np = run_backend("numpy", fbc.variant, structure, params, trial.Y, fbc)
        post_nb = run_backend("numba", fbc.variant, structure, params, trial.Y, fbc)

        for name, a, b in [
            ("alpha", post_np.alpha, post_nb.alpha),
            ("scale", post_np.scale, post_nb.scale),
            ("beta", post_np.beta, post_nb.beta),
            ("gamma", post_np.gamma, post_nb.gamma),
        ]:
            np.testing.assert_allclose(b, a, rtol=RTOL, atol=ATOL, err_msg=f"{posterior_fixture_id} trial {i}: {name} mismatch")

        np.testing.assert_allclose(
            post_nb.loglik, post_np.loglik, rtol=RTOL, atol=ATOL, err_msg=f"{posterior_fixture_id} trial {i}: loglik mismatch"
        )
        np.testing.assert_array_equal(post_nb.B, post_np.B, err_msg=f"{posterior_fixture_id} trial {i}: B mismatch")
        np.testing.assert_array_equal(post_nb.B_comp, post_np.B_comp, err_msg=f"{posterior_fixture_id} trial {i}: B_comp mismatch")

        d_np, d_nb = post_np.diagnostics, post_nb.diagnostics
        assert (d_np.zero_alpha_next_count, d_np.beta_floor_hits, d_np.f_le_h_violations, d_np.nan_guard_hits) == (
            d_nb.zero_alpha_next_count,
            d_nb.beta_floor_hits,
            d_nb.f_le_h_violations,
            d_nb.nan_guard_hits,
        ), f"{posterior_fixture_id} trial {i}: diagnostics mismatch"


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_numba_matches_matlab_directly_on_every_fixture_and_trial(fixtures_dir, posterior_fixture_id):
    """Not merely NumPy-vs-numba: this checks numba against the *committed
    MATLAB* posterior arrays directly, at P3a/P3b's own original
    `rtol=1e-8, atol=1e-9` tolerance -- a bug that shifted NumPy and numba
    away from MATLAB by the same amount (invisible to a NumPy-vs-numba-only
    comparison) would still be caught here."""
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for i, trial in enumerate(fx.trials):
        post_nb = run_backend("numba", fbc.variant, structure, params, trial.Y, fbc)
        for name, mine, matlab in [
            ("alpha", post_nb.alpha, trial.alpha),
            ("beta", post_nb.beta, trial.beta),
            ("gamma", post_nb.gamma, trial.alpha_T),
            ("scale", post_nb.scale, trial.b_c_ot_nc),
        ]:
            np.testing.assert_allclose(
                mine, matlab, rtol=MATLAB_RTOL, atol=MATLAB_ATOL, err_msg=f"{posterior_fixture_id} trial {i}: {name} vs MATLAB mismatch"
            )
        np.testing.assert_allclose(
            post_nb.loglik, trial.P_O_model, rtol=1e-10, atol=1e-8, err_msg=f"{posterior_fixture_id} trial {i}: loglik vs MATLAB mismatch"
        )


# --------------------------------------------------------------------------
# Zero-emission mass: NumPy and numba must agree, including on validate="off".
# A review round found numba raised a raw ZeroDivisionError from a compiled
# scalar division (numba's *default* error model matches Python's own
# division-by-zero-raises semantics) where NumPy's IEEE-754 float division
# silently produces `nan`/`inf` -- fixed via `error_model="numpy"` on every
# `@njit` in the module (not just `_forward_pass_kernel`; `_solve_delta_kernel`
# and both backward kernels divide too). Every position (first/interior/
# final timestep) is covered for both variants and both `validate=` modes
# actually reachable here (`params.validate()` requires a valid `params`
# regardless, so only "params"/"off" apply to a model this deliberately
# extreme observation is layered onto -- "all" would additionally exercise
# the block-sum checks, covered separately above).
# --------------------------------------------------------------------------


def _single_channel_three_state():
    structure = LSIMStructure(channel_state_num=(3,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi_0 = np.array([0.5, 0.3, 0.2])
    transition = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((3, 1)), mu=np.array([0.0, 5.0, 10.0]).reshape(1, 3, 1), sigma=np.ones((1, 3, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()
    return structure, params


@pytest.mark.parametrize("position", ["first", "interior", "final"])
@pytest.mark.parametrize("variant", ["sed", "fast"])
@pytest.mark.parametrize("validate", ["params", "off"])
def test_zero_emission_mass_numpy_and_numba_agree(position, variant, validate):
    """Reproduces the exact regression: a valid model whose final
    observation underflows every state's emission density to `0`. NumPy
    returns `loglik=-inf` with a non-finite `scale` there (established,
    documented behavior -- not something to "fix"); at the first/interior
    positions, `_forward_pass`'s own unconditional NaN-loglik check raises
    `ValidationError` in both backends regardless of `validate=`. Either
    way, NumPy and numba must land on the *same* outcome -- both raising the
    same exception type, or both returning the same (non-)finite result --
    never one silently succeeding while the other crashes with an unrelated
    raw exception."""
    structure, params = _single_channel_three_state()
    Y = {
        "first": np.array([[1000.0, 0.5, 0.5]]),
        "interior": np.array([[0.5, 1000.0, 0.5]]),
        "final": np.array([[0.5, 0.6, 1000.0]]),
    }[position]
    fbc = FbConstants(variant=variant)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            post_np = run_backend("numpy", variant, structure, params, Y, fbc, validate=validate)
            np_outcome = ("ok", bool(np.isfinite(post_np.loglik)))
        except Exception as exc:
            np_outcome = ("raised", type(exc))
        try:
            post_nb = run_backend("numba", variant, structure, params, Y, fbc, validate=validate)
            nb_outcome = ("ok", bool(np.isfinite(post_nb.loglik)))
        except Exception as exc:
            nb_outcome = ("raised", type(exc))

    assert nb_outcome == np_outcome, f"position={position} variant={variant} validate={validate}: numpy={np_outcome} numba={nb_outcome}"
    if np_outcome[0] == "ok":
        assert position == "final"
        # Full comparison, not just "loglik matches and scale has *some*
        # non-finite entry": alpha/scale/beta/gamma must agree element-for-
        # element, including *which* entries are NaN/Inf and where
        # (`equal_nan=True`, `assert_allclose`'s own default, so a NaN in
        # the same position on both sides compares equal rather than
        # unconditionally failing) -- independently confirmed these already
        # agree; this only closes the regression-test gap, not a new fix.
        for name in ("alpha", "scale", "beta", "gamma"):
            np.testing.assert_allclose(
                getattr(post_nb, name), getattr(post_np, name), equal_nan=True, err_msg=f"{position}/{variant}/{validate}: {name}"
            )
        np.testing.assert_allclose(post_nb.loglik, post_np.loglik, equal_nan=True)
        np.testing.assert_array_equal(post_nb.B, post_np.B)
        np.testing.assert_array_equal(post_nb.B_comp, post_np.B_comp)
        d_np, d_nb = post_np.diagnostics, post_nb.diagnostics
        assert (d_np.zero_alpha_next_count, d_np.beta_floor_hits, d_np.f_le_h_violations, d_np.nan_guard_hits) == (
            d_nb.zero_alpha_next_count,
            d_nb.beta_floor_hits,
            d_nb.f_le_h_violations,
            d_nb.nan_guard_hits,
        )
        assert not np.any(np.isfinite(post_np.scale[:, -1]))
        assert not np.any(np.isfinite(post_nb.scale[:, -1]))


def test_solve_delta_kernel_zero_denominator_does_not_raise():
    """`_solve_delta_kernel`'s own division (`f/d`, `1/d`) -- a `d=f-h==0`
    is structurally unreachable for valid parameters, but the kernel itself
    must not crash with a raw `ZeroDivisionError` if it's ever handed one
    (e.g. via a pathological `f_reg`, see
    `test_backward_sed_kernel_validate_all_catches_f_le_h_violation`);
    `error_model="numpy"` makes this `nan`, matching what `inference.sed
    ._solve_delta`'s own NumPy division would produce for the same input."""
    from chmm_lsim.backends.numba_ import _solve_delta_kernel

    f = np.array([[1.0, 2.0], [3.0, 4.0]])
    h = np.array([1.0, 3.0])  # d = f - h[:,None] is exactly 0 in column 0 for both rows
    delta = np.zeros((2, 2))
    _solve_delta_kernel(f, h, delta)  # must not raise
    assert np.all(np.isnan(delta))


# --------------------------------------------------------------------------
# validate="all" on the numba path: not a no-op.
# --------------------------------------------------------------------------


def _single_channel_two_state():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi_0 = np.array([0.6, 0.4])
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([0.0, 5.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    return structure, params


@pytest.mark.parametrize("variant", ["sed", "fast"])
def test_numba_validate_all_does_not_spuriously_raise_on_real_fixtures(fixtures_dir, variant):
    fx = load_posterior_fixture(fixtures_dir, f"c3_sparse_l2r__{variant}")
    structure, params, fbc = _build(fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_all = run_backend("numba", variant, structure, params, fx.trials[0].Y, fbc, validate="all")
        post_off = run_backend("numba", variant, structure, params, fx.trials[0].Y, fbc, validate="off")
    np.testing.assert_allclose(post_all.alpha, post_off.alpha)
    np.testing.assert_allclose(post_all.gamma, post_off.gamma)


def test_numba_wrapper_validate_all_catches_invalid_pi0_before_the_kernel_even_runs():
    """`params.validate()` itself (run whenever `validate != "off"`, both
    `"params"` and `"all"`) already rejects a `pi_0` summing to `0.6` -- so
    this specific corruption never reaches the compiled kernel's own
    block-sum check at all. Confirms the *wrapper*'s existing validation
    still applies unchanged on the numba path; the kernel's own internal
    checks are tested directly below, since a probability input invalid
    enough to violate the block-sum invariant is, by construction, always
    caught earlier by `params.validate()` for any input actually reachable
    through the public API."""
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi_0 = np.array([0.3, 0.3])  # sums to 0.6
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([0.0, 5.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    Y = np.array([[0.1, 0.2, 0.3]])

    with pytest.raises(ValidationError, match="pi_0"):
        run_backend("numba", "sed", structure, params, Y, FbConstants(variant="sed"), validate="all")
    run_backend("numba", "sed", structure, params, Y, FbConstants(variant="sed"), validate="off")


def test_forward_pass_kernel_validate_all_catches_corrupted_alpha_block_sum():
    """Directly exercises the compiled kernel's own `validate="all"` check,
    below the wrapper's `params.validate()` gate -- a hand-built `pi_0` that
    doesn't sum to 1 (which never reaches this deep through the public API,
    since `params.validate()` always catches it first, per the test above)
    still makes the kernel itself detect and report the violation, proving
    the check is real code, not a no-op that merely never gets exercised."""
    from chmm_lsim.backends.numba_ import _FAIL_ALPHA_BLOCK_SUM, _OK, _forward_pass_kernel

    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    W = np.array([[0.8, 0.2], [0.3, 0.7]])
    B = np.ones((2, 3))

    good_pi_0 = np.array([0.6, 0.4])
    _, _, _, fail_code, fail_t = _forward_pass_kernel(good_pi_0, W, B, structure.state_offsets, structure.block_of_state, True)
    assert fail_code == _OK

    bad_pi_0 = np.array([0.3, 0.3])  # sums to 0.6
    _, _, _, fail_code, fail_t = _forward_pass_kernel(bad_pi_0, W, B, structure.state_offsets, structure.block_of_state, True)
    assert fail_code == _FAIL_ALPHA_BLOCK_SUM
    assert fail_t == 0


def test_backward_fast_kernel_validate_all_catches_corrupted_abar_block_sum():
    from chmm_lsim.backends.numba_ import _FAIL_ABAR_BLOCK_SUM, _OK, _backward_fast_kernel

    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    T = 3
    alpha = np.tile(np.array([[0.6], [0.4]]), (1, T))
    scale = np.ones((2, T))
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    weights_per_state = np.ones((2, 1))
    w_opt_per_state = np.ones(2)

    _, _, _, fail_code, fail_t = _backward_fast_kernel(
        alpha, scale, transition, structure.state_offsets, weights_per_state, w_opt_per_state, True
    )
    assert fail_code == _OK

    alpha_bad = alpha.copy()
    alpha_bad[:, 1] *= 2.0  # abar[:,1] = alpha[:,1]*scale[:,1] now sums to 2.0, not 1.0
    _, _, _, fail_code, fail_t = _backward_fast_kernel(
        alpha_bad, scale, transition, structure.state_offsets, weights_per_state, w_opt_per_state, True
    )
    assert fail_code == _FAIL_ABAR_BLOCK_SUM
    assert fail_t == 1


def test_backward_sed_kernel_validate_all_catches_corrupted_abar_block_sum():
    from chmm_lsim.backends.numba_ import _FAIL_ABAR_BLOCK_SUM, _OK, _backward_sed_kernel

    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    T = 3
    S = structure.S
    alpha = np.tile(np.array([[0.3], [0.7], [0.5], [0.5]]), (1, T))
    scale = np.ones((S, T))
    W = np.full((S, S), 1.0 / S)
    W = W / W.sum(axis=1, keepdims=True)

    _, _, _, _, _, _, fail_code, fail_t = _backward_sed_kernel(
        alpha, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, 1e-6, False, True
    )
    assert fail_code == _OK

    alpha_bad = alpha.copy()
    alpha_bad[:, 1] *= 2.0
    _, _, _, _, _, _, fail_code, fail_t = _backward_sed_kernel(
        alpha_bad, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, 1e-6, False, True
    )
    assert fail_code == _FAIL_ABAR_BLOCK_SUM
    assert fail_t == 1


def test_backward_sed_kernel_validate_all_catches_corrupted_gamma_at_final_timestep():
    """`gamma[:,T-1] = alpha[:,T-1]*beta[:,T-1]` is checked once, before the
    `tau` loop even starts -- corrupting `alpha[:,T-1]` (not read by any
    other check before this one) isolates it."""
    from chmm_lsim.backends.numba_ import _FAIL_GAMMA_BLOCK_SUM, _backward_sed_kernel

    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    T = 3
    S = structure.S
    alpha = np.tile(np.array([[0.3], [0.7], [0.5], [0.5]]), (1, T))
    alpha[:, T - 1] *= 2.0  # gamma[:,T-1] = alpha[:,T-1]*scale[:,T-1] now sums to 2.0 per block
    scale = np.ones((S, T))
    W = np.full((S, S), 1.0 / S)
    W = W / W.sum(axis=1, keepdims=True)

    _, _, _, _, _, _, fail_code, fail_t = _backward_sed_kernel(
        alpha, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, 1e-6, False, True
    )
    assert fail_code == _FAIL_GAMMA_BLOCK_SUM
    assert fail_t == T - 1


def test_check_all_close_to_one_helper_detects_v2_style_violations():
    """`V2` is already a per-channel scalar (`block_sum(alpha_next*beta_next,
    ...)`) by the time `sed.py`'s own `_assert_close_to_one` -- and this
    kernel's `_check_all_close_to_one` -- ever sees it, so it's checked with
    a plain "every entry close to 1.0" rule, not another block-sum reduction
    (that's `_check_block_sum`'s job, for a still-per-*state* array like
    `alpha`/`abar`/`gamma`). In the real `_backward_sed_kernel`, `V2` at the
    first backward step (`tau=T-2`) is mathematically the *same* product as
    the pre-loop `gamma[:,T-1]` check (both reduce `alpha[:,T-1]*beta[:,T-1]`
    over the same blocks), so any input that breaks one always breaks the
    other identically and the pre-loop gamma check fires first -- exactly
    mirroring `inference.sed.forward_backward_sed`'s own check ordering, not
    a numba-specific gap. The helper itself is tested directly here instead,
    since that's the one piece actually specific to how `V2` is checked."""
    from chmm_lsim.backends.numba_ import _check_all_close_to_one

    assert _check_all_close_to_one(np.array([1.0, 1.0, 1.0])) is True
    assert _check_all_close_to_one(np.array([1.0, 1.0 + 1e-9])) is True  # within _BLOCK_SUM_ATOL
    assert _check_all_close_to_one(np.array([1.0, 0.9])) is False
    assert _check_all_close_to_one(np.array([1.0, np.nan])) is False
    assert _check_all_close_to_one(np.array([1.0, np.inf])) is False


def test_backward_sed_kernel_v2_workspace_is_not_reallocated_and_does_not_leak_state():
    """A review round found `v2` allocated fresh (`np.zeros(C)`) on every
    `tau` iteration inside `_backward_sed_kernel`, contradicting the module's
    own "no allocations inside the timestep loop" guarantee -- fixed by
    hoisting it next to the other per-call workspaces. This doesn't change
    behavior (every element of `v2` is unconditionally overwritten each time
    it's used, whether freshly allocated or reused), but confirms it
    directly: two independent calls on the same valid input, with
    `validate="all"` exercising the `v2` workspace at every backward step,
    must return bit-identical results -- proving reuse doesn't leak stale
    values from one call into the next."""
    from chmm_lsim.backends.numba_ import _OK, _backward_sed_kernel

    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    T = 5
    S = structure.S
    alpha = np.tile(np.array([[0.3], [0.7], [0.5], [0.5]]), (1, T))
    scale = np.ones((S, T))
    W = np.full((S, S), 1.0 / S)
    W = W / W.sum(axis=1, keepdims=True)

    result_1 = _backward_sed_kernel(alpha, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, 1e-6, False, True)
    result_2 = _backward_sed_kernel(alpha, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, 1e-6, False, True)

    assert result_1[-2] == _OK  # fail_code
    np.testing.assert_array_equal(result_1[0], result_2[0])  # beta
    np.testing.assert_array_equal(result_1[1], result_2[1])  # gamma
    assert result_1[2:] == result_2[2:]  # scalar diagnostics + fail_code/fail_t


def test_backward_sed_kernel_validate_all_catches_f_le_h_violation():
    """`f<=h` is structurally unreachable for valid parameters (§0.3.3) --
    reproduced here via a deliberately, unrealistically negative `f_reg`
    (not part of any real `FbConstants`), which forces `f = raw + f_reg`
    below `h` even though the underlying `raw`/`h` themselves come from a
    perfectly ordinary, valid two-channel model."""
    from chmm_lsim.backends.numba_ import _FAIL_F_LE_H, _OK, _backward_sed_kernel

    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    T = 3
    S = structure.S
    alpha = np.tile(np.array([[0.3], [0.7], [0.5], [0.5]]), (1, T))
    scale = np.ones((S, T))
    W = np.full((S, S), 1.0 / S)
    W = W / W.sum(axis=1, keepdims=True)

    _, _, _, _, _, _, fail_code, fail_t = _backward_sed_kernel(
        alpha, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, 1e-6, False, True
    )
    assert fail_code == _OK

    _, _, _, _, _, _, fail_code, fail_t = _backward_sed_kernel(
        alpha, scale, W, structure.state_offsets, structure.block_of_state, 1e-9, -1000.0, False, True
    )
    assert fail_code == _FAIL_F_LE_H


# --------------------------------------------------------------------------
# Registry/capability behavior specific to the numba backend (in addition to
# test_backends_base.py's generic capability-layer tests).
# --------------------------------------------------------------------------


def test_forward_backward_backend_numba_dispatches(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    from chmm_lsim.inference.registry import forward_backward

    post = forward_backward(structure, params, fx.trials[0].Y, fbc, backend="numba")
    direct = run_backend("numba", "sed", structure, params, fx.trials[0].Y, fbc)
    np.testing.assert_array_equal(post.alpha, direct.alpha)


def test_numba_rejects_sed_method_reference():
    structure, params = _single_channel_two_state()
    params.validate()
    Y = np.array([[0.1, 0.2, 0.3]])
    with pytest.raises(UnsupportedCombination, match="sed method"):
        run_backend("numba", "sed", structure, params, Y, FbConstants(variant="sed"), method="reference")


# --------------------------------------------------------------------------
# backend="numba" genuinely runs during e_step/em_lsim -- not just when
# forward_backward is called directly.
# --------------------------------------------------------------------------


def test_e_step_backend_numba_matches_direct_run_backend_call(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y_trials = [t.Y for t in fx.trials]

    raw_numba = e_step(structure, params, Y_trials, fbc, backend="numba")
    raw_numpy = e_step(structure, params, Y_trials, fbc, backend="numpy")

    np.testing.assert_allclose(raw_numba.gamma_first, raw_numpy.gamma_first, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw_numba.loglik, raw_numpy.loglik, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw_numba.zeta, raw_numpy.zeta, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "structure_kwargs,sigma_diag,seed",
    [
        (dict(channel_state_num=(3,), channel_obs_dim=(2,), num_gmm_component=(1,)), True, 0),  # diagonal
        (dict(channel_state_num=(3,), channel_obs_dim=(2,), num_gmm_component=(1,)), False, 1),  # full covariance
    ],
)
def test_em_lsim_backend_numba_matches_numpy_diag_and_full(structure_kwargs, sigma_diag, seed):
    """Representative diagonal and full-covariance single-trial em_lsim runs:
    identical Y, identical initial params, `backend="numba"` vs
    `backend="numpy"` for every E-step, must land on the same params trace."""
    structure = LSIMStructure(**structure_kwargs)
    rng = np.random.default_rng(seed)
    pi_0 = rng.dirichlet(np.ones(structure.channel_state_num[0]))
    transition = rng.dirichlet(np.ones(structure.channel_state_num[0]) * 3, structure.channel_state_num[0])
    coupling = np.array([[1.0]])
    M, d = structure.channel_state_num[0], structure.channel_obs_dim[0]
    mu = rng.normal(scale=5.0, size=(d, M, 1))
    if sigma_diag:
        sigma = np.ones((d, M, 1))
    else:
        sigma = np.stack([np.eye(d) for _ in range(M)], axis=-1)[:, :, None, :].reshape(d, d, M, 1)
    gmm = ChannelGMM(weights=np.ones((M, 1)), mu=mu, sigma=sigma, diag=sigma_diag)
    true_params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    true_params.validate()

    Y, _ = simulate_lsim(structure, true_params, 150, seed=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params0 = initialize_lsim_params(structure, [Y], sigma_diag=sigma_diag, seed=seed)
        result_numpy = em_lsim(structure, params0.copy(), [Y], max_iterations=4, backend="numpy", check_convergence=False, validate="off")
        result_numba = em_lsim(structure, params0.copy(), [Y], max_iterations=4, backend="numba", check_convergence=False, validate="off")

    np.testing.assert_allclose(result_numba.loglik_trace, result_numpy.loglik_trace, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result_numba.params.pi_0, result_numpy.params.pi_0, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result_numba.params.transition, result_numpy.params.transition, rtol=1e-8, atol=1e-8)


def test_em_lsim_backend_numba_matches_numpy_multi_trial(fixtures_dir):
    """Multi-trial em_lsim (c3_trials_unequal, 3 unequal-length trials, real
    MATLAB-derived structure/params) -- confirms `backend="numba"` reduces
    correctly across multiple trials in one E-step, not just a single trial."""
    fx = load_posterior_fixture(fixtures_dir, "c3_trials_unequal__sed")
    structure, params, fbc = _build(fx)
    Y_trials = [t.Y for t in fx.trials]
    assert len(Y_trials) > 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_numpy = em_lsim(structure, params.copy(), Y_trials, fb_constants=fbc, max_iterations=3, backend="numpy", check_convergence=False)
        result_numba = em_lsim(structure, params.copy(), Y_trials, fb_constants=fbc, max_iterations=3, backend="numba", check_convergence=False)

    np.testing.assert_allclose(result_numba.loglik_trace, result_numpy.loglik_trace, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result_numba.params.transition, result_numpy.params.transition, rtol=1e-8, atol=1e-8)
