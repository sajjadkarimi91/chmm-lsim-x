"""P8a gate: the batched Torch backend, CPU-fp64 verification only (§9 of
the P8a request) -- CUDA correctness/error-study/`N=100,000` benchmarks are
P8b, deferred until run on real GPU hardware (`benchmarks/cuda_validation_cli.py`,
written but never executed here). A handful of `device="mps"` smoke tests
run for real on this development machine (`torch.backends.mps.is_available()`
is `True` here) -- optional, not a sign-off requirement, and not a
substitute for P8b's CUDA validation.

**Tolerances, measured against every fixture x trial, not guessed -- same
methodology, and the same lesson, as P7's correction rounds; re-measured
after the P8a correction round's `_eig_logpdf` rewrite (SciPy-compatible
eigenvalue cutoff + null-space support masking) and found unchanged --
none of the 48 fixtures happen to exercise the null-space/off-support
branch itself (their "repaired" covariances are repaired to be non-singular
before reaching this stage), so that branch's own correctness is covered
separately by the dedicated unit tests below (`test_eig_logpdf_*`,
`test_full_covariance_batch_emissions_off_support_gives_zero_not_nan`), not
by this fixture battery.** `alpha`/`beta`/`gamma`/`scale`/`loglik` match
NumPy at `rtol=1e-9, atol=1e-9` uniformly (measured worst absolute
difference `~1.9e-10`, on `c3_sigma_repair_2eig` -- previously reported as
`~1.5e-9` on `c3_sparse_l2r`/`c3_lowmass_full`; both figures are well inside
tolerance, the exact worst-case fixture shifts run to run with LAPACK/MKL
thread-count and isn't itself meaningful). `B`/`B_comp` (recovered via
`Posterior.B_raw()`/`B_comp_raw()`, since this backend returns the
*rescaled*, log-domain values -- see `backends.torch_batched`'s own module
docstring) are compared at `rtol=1e-6` (**relative only, no absolute
term**): `B`'s own dynamic range spans orders of magnitude by construction
(it's an unnormalized density, not a probability), so an absolute tolerance
is the wrong tool here -- confirmed directly: the single largest *absolute*
`B_raw()`/`B_comp_raw()` difference measured (`~6.2e-3`, identically for
both, on `c3_sigma_repair_2eig`, a covariance with two near-degenerate
repaired eigenvalues) sits on a raw value of `~3.4e6`, a *relative*
difference of only `~3.2e-9`, i.e. two different LAPACK-backed
eigendecomposition routines (`numpy`/`scipy` vs `torch`) picking a slightly
different but equally valid eigenvector basis within a near-degenerate
subspace -- not a bug, and not something an absolute tolerance would have
described honestly.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_posterior_fixture  # noqa: E402

from chmm_lsim.backends.base import MissingBackendDependency, UnsupportedCombination, run_backend
from chmm_lsim.compat import FbConstants
from chmm_lsim.em import e_step, em_lsim, finalize_expected_stats
from chmm_lsim.init import initialize_lsim_params
from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.simulate import simulate_lsim
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError

RTOL, ATOL = 1e-9, 1e-9
B_RTOL = 1e-6  # relative-only, see module docstring
MATLAB_RTOL, MATLAB_ATOL = 1e-7, 1e-8  # slightly looser than P3a/P3b's own 1e-8/1e-9 -- see the dedicated test's own docstring for why

_MPS_AVAILABLE = torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not _MPS_AVAILABLE, reason="torch.backends.mps.is_available() is False on this machine")

_CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="torch.cuda.is_available() is False on this machine -- P8b runs these on a real GPU cluster")


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


def _tiny_sed_model(C=1, M=(2,)):
    structure = LSIMStructure(channel_state_num=M, channel_obs_dim=(1,) * C, num_gmm_component=(1,) * C)
    rng = np.random.default_rng(0)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in M])
    transition = np.zeros((structure.S, structure.S))
    for c in range(C):
        for z in range(C):
            m_z = M[z]
            transition[structure.state_slice(c), structure.state_slice(z)] = rng.dirichlet(np.ones(m_z) * 3, M[c])
    coupling = rng.dirichlet(np.ones(C) * 2, C).T if C > 1 else np.array([[1.0]])
    emissions = []
    for c in range(C):
        m_c = M[c]
        weights = np.ones((m_c, 1))
        mu = rng.normal(scale=4.0, size=(1, m_c, 1))
        sigma = np.ones((1, m_c, 1))
        emissions.append(ChannelGMM(weights=weights, mu=mu, sigma=sigma, diag=True))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    return structure, params


# --------------------------------------------------------------------------
# Full 48-fixture parity, both variants, every trial (via run_backend, the
# batch-of-one single-sequence contract).
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_torch_matches_numpy_on_every_fixture_and_trial(fixtures_dir, posterior_fixture_id):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)

    for i, trial in enumerate(fx.trials):
        post_np = run_backend("numpy", fbc.variant, structure, params, trial.Y, fbc)
        post_torch = run_backend("torch", fbc.variant, structure, params, trial.Y, fbc)

        for name in ("alpha", "beta", "gamma", "scale"):
            np.testing.assert_allclose(
                getattr(post_torch, name), getattr(post_np, name), rtol=RTOL, atol=ATOL,
                err_msg=f"{posterior_fixture_id} trial {i}: {name} mismatch",
            )
        np.testing.assert_allclose(
            post_torch.loglik, post_np.loglik, rtol=RTOL, atol=ATOL, err_msg=f"{posterior_fixture_id} trial {i}: loglik mismatch"
        )
        np.testing.assert_allclose(
            post_torch.B_raw(), post_np.B, rtol=B_RTOL, atol=0, err_msg=f"{posterior_fixture_id} trial {i}: B mismatch"
        )
        np.testing.assert_allclose(
            post_torch.B_comp_raw(), post_np.B_comp, rtol=B_RTOL, atol=0,
            err_msg=f"{posterior_fixture_id} trial {i}: B_comp mismatch",
        )

        d_np, d_t = post_np.diagnostics, post_torch.diagnostics
        assert (d_np.zero_alpha_next_count, d_np.beta_floor_hits, d_np.f_le_h_violations, d_np.nan_guard_hits) == (
            d_t.zero_alpha_next_count, d_t.beta_floor_hits, d_t.f_le_h_violations, d_t.nan_guard_hits,
        ), f"{posterior_fixture_id} trial {i}: diagnostics mismatch"


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
def test_torch_matches_matlab_directly_on_every_fixture_and_trial(fixtures_dir, posterior_fixture_id):
    """Direct numba-style check against the committed MATLAB arrays, not
    only indirectly through NumPy -- a slightly looser tolerance than P3a/
    P3b's own `1e-8/1e-9` (`1e-7/1e-8`) since this path also carries the
    (already-measured, ~1e-9-scale) NumPy-vs-Torch gap on top of NumPy's own
    (already-verified) MATLAB gap -- still tight, still measured."""
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    for i, trial in enumerate(fx.trials):
        post_torch = run_backend("torch", fbc.variant, structure, params, trial.Y, fbc)
        for name, mine, matlab in [
            ("alpha", post_torch.alpha, trial.alpha),
            ("beta", post_torch.beta, trial.beta),
            ("gamma", post_torch.gamma, trial.alpha_T),
            ("scale", post_torch.scale, trial.b_c_ot_nc),
        ]:
            np.testing.assert_allclose(
                mine, matlab, rtol=MATLAB_RTOL, atol=MATLAB_ATOL, err_msg=f"{posterior_fixture_id} trial {i}: {name} vs MATLAB"
            )


# --------------------------------------------------------------------------
# gz_form: block (default) vs masked cross-check.
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("scenario", ["c3_het_diag", "c3_het_full", "c5_K_mixed", "c3_sparse_l2r"])
def test_gz_form_block_and_masked_agree(fixtures_dir, scenario):
    fx = load_posterior_fixture(fixtures_dir, f"{scenario}__sed")
    structure, params, fbc = _build(fx)
    Y = fx.trials[0].Y
    post_block = run_backend("torch", "sed", structure, params, Y, fbc, gz_form="block")
    post_masked = run_backend("torch", "sed", structure, params, Y, fbc, gz_form="masked")
    np.testing.assert_allclose(post_block.alpha, post_masked.alpha, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(post_block.beta, post_masked.beta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(post_block.loglik, post_masked.loglik, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# Ragged batches: T=1, unequal lengths, widely separated lengths, checked
# against per-sequence NumPy directly (§5 of the P8a request).
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("scenario,variant", [("c3_het_diag", "sed"), ("c3_het_diag", "fast"), ("c3_het_full", "sed"), ("c5_K_mixed", "sed"),
                                              # P12: both poise modes, the refined one because its
                                              # refinement is per-sequence (see the T=1 test below).
                                              ("c3_het_diag", "poise"), ("c3_het_diag", "poise_refined"),
                                              ("c3_het_full", "poise_refined"), ("c5_K_mixed", "poise_refined")])
def test_ragged_batch_matches_per_sequence_numpy(fixtures_dir, scenario, variant):
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    fx = load_posterior_fixture(fixtures_dir, f"{scenario}__{variant}")
    structure, params, fbc = _build(fx)
    Y_real = fx.trials[0].Y
    T_real = Y_real.shape[1]
    seqs = [Y_real, Y_real[:, :1], Y_real[:, : max(1, T_real // 2)], Y_real[:, : max(1, T_real // 5)]]

    chunks = list(forward_backward_batch(structure, params, seqs, fbc, mode="posterior", chunk_size=2))
    seen = set()
    for chunk in chunks:
        for local_i, orig_i in enumerate(chunk.sequence_indices):
            seen.add(orig_i)
            post_torch = chunk.extract(local_i)
            # `variant` is the fixture LABEL; `poise_refined` is variant="poise" with the flag on
            post_np = run_backend("numpy", fbc.variant, structure, params, seqs[orig_i], fbc)
            assert post_torch.alpha.shape == (structure.S, seqs[orig_i].shape[1])
            np.testing.assert_allclose(post_torch.alpha, post_np.alpha, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(post_torch.gamma, post_np.gamma, rtol=RTOL, atol=ATOL)
            # `beta` was not compared here before P12. It is now, though note that this test's
            # scenarios would NOT have caught the padded-`T` POISE refinement bug even so: that
            # bug only shows where `alpha` has exact zeros (sparse left-to-right transitions), so
            # it has its own test on `c3_sparse_l2r` below.
            np.testing.assert_allclose(post_torch.beta, post_np.beta, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(post_torch.loglik, post_np.loglik, rtol=RTOL, atol=ATOL)
    assert seen == set(range(len(seqs)))  # every sequence appeared in exactly one chunk


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_poise_refinement_is_per_sequence_not_per_padded_batch(fixtures_dir, chunk_size):
    """P12 regression, a real bug the CUDA chunk-invariance gate caught: POISE's
    refinement is a *per-sequence* property. `inference/poise.py` skips the whole
    pass -- refinement loop *and* effective-`beta` reconstruction -- when `T == 1`
    (MATLAB's loop starts at `t=2`), but the batched backward tested `T > 1` on
    the **padded batch maximum**, so a `T=1` sequence sharing a chunk with longer
    companions got the reconstruction applied anyway and came back with a
    `beta` that was zero wherever `alpha == 0` instead of its raw `scale`.

    That made a length-1 sequence's `beta` depend on what it was batched with and
    on `chunk_size` -- 0.001 absolute on `c3_sparse_l2r`, not a rounding
    difference. Every sequence here must match its own standalone NumPy
    posterior at every chunk size, including when the `T=1` sequence lands alone
    in its chunk (`chunk_size=1`, the case that accidentally behaved correctly).
    """
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__poise_refined")
    structure, params, fbc = _build(fx)
    assert fbc.variant == "poise" and fbc.poise_refinement is True
    Y_real = fx.trials[0].Y
    seqs = [Y_real, Y_real[:, :1], Y_real[:, :2], Y_real[:, : Y_real.shape[1] // 2]]
    standalone = [run_backend("numpy", "poise", structure, params, Y, fbc) for Y in seqs]

    for chunk in forward_backward_batch(structure, params, seqs, fbc, mode="posterior", chunk_size=chunk_size):
        for local_i, orig_i in enumerate(chunk.sequence_indices):
            post = chunk.extract(local_i)
            ref = standalone[orig_i]
            for name in ("alpha", "beta", "gamma", "scale"):
                np.testing.assert_allclose(
                    getattr(post, name), getattr(ref, name), rtol=RTOL, atol=ATOL,
                    err_msg=f"chunk_size={chunk_size}, sequence {orig_i} (T={seqs[orig_i].shape[1]}): {name}")

    # and the T=1 sequence's own invariant, stated directly rather than only relatively:
    # with no t>=1 slice to refine, its beta is the raw observation term.
    np.testing.assert_allclose(standalone[1].beta, standalone[1].scale, rtol=0, atol=0)


@pytest.mark.parametrize("t_short", [1, 2, 3, 5, 10])
def test_ragged_batch_diagnostics_match_standalone_sparse_model(fixtures_dir, t_short):
    """A review round found a real bug: a short sequence batched beside a
    much longer one counted the longer sequence's own padding-region
    backward-recursion steps toward the short sequence's `Diagnostics` --
    confirmed directly with `c3_sparse_l2r` (a sparse/left-to-right model
    whose `zero_alpha_next_count` is genuinely nonzero): a standalone
    `T=1` sequence has `zero_alpha_next_count=0` (its own backward loop
    never runs at all -- `range(T-2,-1,-1)` is empty for `T=1`), but the
    same sequence batched next to a `T=150` companion previously came back
    with `zero_alpha_next_count=3`, silently borrowed from the companion's
    own padding continuation. `alpha`/`gamma`/`loglik` must also match
    exactly, not only `Diagnostics` -- confirming the fix didn't only
    suppress the *symptom* (wrong counters) while leaving the underlying
    per-timestep masking wrong in some other, array-affecting way (it
    doesn't: the arrays were already correct before this fix too, since the
    NaN-padding trick alone already made them right; only the *diagnostics*
    needed the additional masking)."""
    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__sed")
    structure, params, fbc = _build(fx)
    Y_full = fx.trials[0].Y
    Y_short = Y_full[:, :t_short]
    Y_long = np.tile(Y_full, (1, 5))  # a much longer companion, forcing heavy padding on the short one

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_standalone = run_backend("torch", "sed", structure, params, Y_short, fbc)
        from chmm_lsim.backends.torch_batched import forward_backward_batch

        chunks = list(forward_backward_batch(structure, params, [Y_short, Y_long], fbc, mode="posterior", chunk_size=2))

    (post_batched,) = [
        chunk.extract(local_i) for chunk in chunks for local_i, orig_i in enumerate(chunk.sequence_indices) if orig_i == 0
    ]
    assert post_standalone.diagnostics == post_batched.diagnostics
    np.testing.assert_allclose(post_batched.alpha, post_standalone.alpha, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(post_batched.gamma, post_standalone.gamma, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(post_batched.loglik, post_standalone.loglik, rtol=RTOL, atol=ATOL)


def test_mode_loglik_matches_and_restores_original_order(fixtures_dir):
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y_real = fx.trials[0].Y
    T_real = Y_real.shape[1]
    seqs = [Y_real[:, : max(1, T_real // 3)], Y_real, Y_real[:, :1]]  # deliberately NOT length-sorted

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logliks = forward_backward_batch(structure, params, seqs, fbc, mode="loglik", chunk_size=2)
        ref = [run_backend("numpy", "sed", structure, params, s, fbc).loglik for s in seqs]
    assert logliks.shape == (3,)
    np.testing.assert_allclose(logliks, ref, rtol=RTOL, atol=ATOL)


def test_mode_loglik_never_runs_the_backward_pass(fixtures_dir, monkeypatch):
    """A review round found `mode="loglik"` ran the *entire* posterior
    computation (backward recursion included), transferred every array to
    CPU, and simply discarded everything but `loglik`. Confirmed directly
    here: monkeypatching both backward-pass functions to raise if called
    proves `mode="loglik"` never reaches them, while `mode="posterior"` (on
    the same inputs) still does."""
    import chmm_lsim.backends.torch_batched as tb

    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y = fx.trials[0].Y

    def _boom(*args, **kwargs):
        raise AssertionError("backward pass must not run for mode='loglik'")

    monkeypatch.setattr(tb, "_backward_sed_batch", _boom)
    monkeypatch.setattr(tb, "_backward_fast_batch", _boom)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logliks = tb.forward_backward_batch(structure, params, [Y], fbc, mode="loglik")
    assert logliks.shape == (1,)
    assert np.isfinite(logliks[0])

    with pytest.raises(AssertionError, match="backward pass must not run"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            list(tb.forward_backward_batch(structure, params, [Y], fbc, mode="posterior"))


# --------------------------------------------------------------------------
# mode="stats": every ExpectedStats/ChannelExpectedStats field, ragged
# multi-trial, against e_step(backend="numpy").
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("scenario", ["c3_het_diag", "c3_het_full", "c3_trials_unequal", "c3_het_full_nan"])
def test_mode_stats_matches_e_step_every_field(fixtures_dir, scenario):
    fx = load_posterior_fixture(fixtures_dir, f"{scenario}__sed")
    structure, params, fbc = _build(fx)
    Y_trials = [t.Y for t in fx.trials]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats_torch = e_step(structure, params, Y_trials, fbc, backend="torch")
        stats_np = e_step(structure, params, Y_trials, fbc, backend="numpy")

    np.testing.assert_allclose(stats_torch.zeta, stats_np.zeta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(stats_torch.gamma_first, stats_np.gamma_first, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(stats_torch.gamma_sum, stats_np.gamma_sum, rtol=1e-6, atol=1e-8)
    assert stats_torch.n_trials == stats_np.n_trials
    assert stats_torch.n_transitions == stats_np.n_transitions
    assert stats_torch.T_all == stats_np.T_all
    assert isinstance(stats_torch.n_trials, int) and isinstance(stats_torch.T_all, int)
    np.testing.assert_allclose(stats_torch.loglik, stats_np.loglik, rtol=1e-6, atol=1e-8)

    for c in range(structure.C):
        ct, cr = stats_torch.channels[c], stats_np.channels[c]
        np.testing.assert_allclose(ct.resp, cr.resp, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(ct.resp_pos, cr.resp_pos)
        np.testing.assert_array_equal(ct.n_w_pos_diag, cr.n_w_pos_diag)
        np.testing.assert_array_equal(ct.n_w_pos_full, cr.n_w_pos_full)
        if ct.weighted_W is not None:
            np.testing.assert_allclose(ct.weighted_W, cr.weighted_W, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ct.weighted_mean, cr.weighted_mean, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ct.weighted_M2, cr.weighted_M2, rtol=1e-5, atol=1e-7)
        else:
            np.testing.assert_allclose(ct.pairwise_Q, cr.pairwise_Q, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ct.pairwise_M, cr.pairwise_M, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ct.pairwise_C, cr.pairwise_C, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(ct.u_pairwise_Q, cr.u_pairwise_Q, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ct.u_pairwise_M, cr.u_pairwise_M, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ct.u_pairwise_C, cr.u_pairwise_C, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(ct.n_obs, cr.n_obs)
        np.testing.assert_allclose(ct.welford_mean, cr.welford_mean, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(ct.welford_M2, cr.welford_M2, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(ct.allzero_mean, cr.allzero_mean, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(ct.allzero_M2, cr.allzero_M2, rtol=1e-5, atol=1e-7)

    # Finalization (fallback flags, nanvar_y) reached through the *same*
    # already-verified finalize_expected_stats -- confirms the raw stats
    # this backend produces are compatible with the existing pipeline, not
    # merely numerically close in isolation.
    finalized_t = finalize_expected_stats(stats_torch, sigma_diag=params.sigma_diag)
    finalized_r = finalize_expected_stats(stats_np, sigma_diag=params.sigma_diag)
    for c in range(structure.C):
        np.testing.assert_array_equal(finalized_t.channels[c].fallback_mu, finalized_r.channels[c].fallback_mu)
        if not params.sigma_diag:
            np.testing.assert_array_equal(finalized_t.channels[c].fallback_sigma_full, finalized_r.channels[c].fallback_sigma_full)


def test_mode_stats_chunk_size_1_matches_one_large_chunk(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_trials_unequal__sed")
    structure, params, fbc = _build(fx)
    Y_trials = [t.Y for t in fx.trials]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats_chunked = e_step(structure, params, Y_trials, fbc, backend="torch", backend_options={"chunk_size": 1})
        stats_whole = e_step(structure, params, Y_trials, fbc, backend="torch", backend_options={"chunk_size": 100})
    np.testing.assert_allclose(stats_chunked.zeta, stats_whole.zeta, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats_chunked.loglik, stats_whole.loglik, rtol=1e-9, atol=1e-9)
    assert stats_chunked.T_all == stats_whole.T_all


def test_mode_stats_is_invariant_to_sequence_order(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_trials_unequal__sed")
    structure, params, fbc = _build(fx)
    Y_trials = [t.Y for t in fx.trials]
    shuffled = list(reversed(Y_trials))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats_orig = e_step(structure, params, Y_trials, fbc, backend="torch")
        stats_shuffled = e_step(structure, params, shuffled, fbc, backend="torch")
    np.testing.assert_allclose(stats_orig.zeta, stats_shuffled.zeta, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats_orig.gamma_first, stats_shuffled.gamma_first, rtol=1e-9, atol=1e-9)
    assert stats_orig.T_all == stats_shuffled.T_all


# --------------------------------------------------------------------------
# EM integration.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_diag,seed", [(True, 0), (False, 1)])
def test_em_lsim_backend_torch_matches_numpy_diag_and_full(sigma_diag, seed):
    pytest.importorskip("sklearn")  # initialize_lsim_params needs the init extra
    structure = LSIMStructure(channel_state_num=(3,), channel_obs_dim=(2,), num_gmm_component=(1,))
    rng = np.random.default_rng(seed)
    pi_0 = rng.dirichlet(np.ones(3))
    transition = rng.dirichlet(np.ones(3) * 3, 3)
    coupling = np.array([[1.0]])
    mu = rng.normal(scale=5.0, size=(2, 3, 1))
    sigma = np.ones((2, 3, 1)) if sigma_diag else np.stack([np.eye(2) for _ in range(3)], axis=-1)[:, :, None, :].reshape(2, 2, 3, 1)
    gmm = ChannelGMM(weights=np.ones((3, 1)), mu=mu, sigma=sigma, diag=sigma_diag)
    true_params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    true_params.validate()
    Y, _ = simulate_lsim(structure, true_params, 150, seed=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params0 = initialize_lsim_params(structure, [Y], sigma_diag=sigma_diag, seed=seed)
        result_np = em_lsim(structure, params0.copy(), [Y], max_iterations=4, backend="numpy", check_convergence=False, validate="off")
        result_torch = em_lsim(structure, params0.copy(), [Y], max_iterations=4, backend="torch", check_convergence=False, validate="off")

    np.testing.assert_allclose(result_torch.loglik_trace, result_np.loglik_trace, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result_torch.params.transition, result_np.params.transition, rtol=1e-5, atol=1e-6)


def test_em_lsim_backend_options_rejected_for_numpy_and_numba():
    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(ValueError, match="backend_options"):
        e_step(structure, params, [Y], FbConstants(), backend="numpy", backend_options={"chunk_size": 4})


# --------------------------------------------------------------------------
# Capability layer / device / dtype.
# --------------------------------------------------------------------------


def test_torch_registered_backend_capabilities():
    from chmm_lsim.backends.base import get_backend

    caps = get_backend("torch")
    assert set(caps.variants) == {"sed", "fast", "poise"}
    assert set(caps.dtypes) == {"float64", "float32"}
    assert set(caps.modes) == {"loglik", "posterior", "stats"}
    assert caps.batched is True
    assert caps.sed_methods == ("optimized",)


@pytest.mark.parametrize("bad_kwargs", [{"mode": "loglik"}, {"mode": "stats"}, {"batched": True}])
def test_run_backend_rejects_non_posterior_mode_and_batched_for_torch(fixtures_dir, bad_kwargs):
    """A review round found `run_backend("torch", ..., mode="loglik")` (and
    `mode="stats"`, and `batched=True`) were silently *accepted* and
    returned an ordinary single-sequence `Posterior` anyway -- `torch`'s own
    `BackendCapabilities` genuinely declares support for those axes (for
    `forward_backward_batch`), so `check_capability` passed them through,
    and `_run_torch` then just ignored them, always hardcoding
    `mode="posterior"`. `run_backend`/`forward_backward` is now always the
    single-sequence, unbatched, `"posterior"`-mode contract, unconditionally
    -- these must raise `UnsupportedCombination`, not silently substitute a
    different mode/batching than what was actually requested."""
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y = fx.trials[0].Y
    with pytest.raises(UnsupportedCombination):
        run_backend("torch", "sed", structure, params, Y, fbc, **bad_kwargs)


def test_run_backend_still_works_ordinarily_for_torch(fixtures_dir):
    """Companion to the test above: confirms the fix doesn't overreach and
    also break the ordinary, correct single-sequence call."""
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y = fx.trials[0].Y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = run_backend("torch", "sed", structure, params, Y, fbc)
    assert np.all(np.isfinite(post.alpha))


def test_torch_cuda_unavailable_raises_not_silent_cpu_fallback():
    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    if torch.cuda.is_available():
        pytest.skip("this machine genuinely has CUDA -- the unavailable-device path isn't exercised here")
    with pytest.raises(UnsupportedCombination, match="cuda"):
        run_backend("torch", "sed", structure, params, Y, FbConstants(variant="sed"), device="cuda")


def test_torch_rejects_unsupported_dtype():
    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(UnsupportedCombination, match="dtype"):
        run_backend("torch", "sed", structure, params, Y, FbConstants(variant="sed"), dtype="float16")


def test_torch_rejects_sed_method_reference():
    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(UnsupportedCombination, match="sed method"):
        run_backend("torch", "sed", structure, params, Y, FbConstants(variant="sed"), method="reference")


@requires_mps
def test_mps_float64_unsupported_combination():
    """MPS has no float64 support in this torch build -- requesting it must
    raise `UnsupportedCombination` naming both axes, never silently
    downcast to float32 or run on CPU instead."""
    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(UnsupportedCombination, match="float64"):
        run_backend("torch", "sed", structure, params, Y, FbConstants(variant="sed"), device="mps", dtype="float64")


@requires_mps
def test_mps_float32_smoke(fixtures_dir):
    """Runtime-detected smoke coverage only (P8a explicitly does not require
    MPS for sign-off) -- confirms the device path executes and produces
    finite output, not tight numerical parity with CPU float64 (float32 on
    a different device backend is not expected to match to 1e-9)."""
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y = fx.trials[0].Y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = run_backend("torch", "sed", structure, params, Y, fbc, device="mps", dtype="float32")
    assert np.all(np.isfinite(post.alpha))
    assert np.all(np.isfinite(post.gamma))
    assert np.isfinite(post.loglik)


def test_missing_torch_dependency_raises_actionable_error(monkeypatch):
    """Mirrors `test_backends_base.py`'s numba equivalent exactly -- see
    that test's own docstring for the full explanation of the technique."""
    from chmm_lsim.backends import base as base_module

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "chmm_lsim.backends.torch_batched", raising=False)
    monkeypatch.delitem(base_module._BACKENDS, "torch", raising=False)
    monkeypatch.delitem(base_module._RUNNERS, "torch", raising=False)

    with pytest.raises(MissingBackendDependency, match=r"pip install 'chmm-lsim\[torch\]'"):
        base_module.get_backend("torch")


def test_forward_backward_batch_is_genuinely_public_via_lsim_namespace():
    """`forward_backward_batch`/`BatchedPosteriorChunk`/`TorchKernel`/
    `estimate_chunk_size` must be reachable as ordinary `chmm_lsim.<name>`
    attributes (and via `from chmm_lsim import <name>`), not only via the
    internal `chmm_lsim.backends.torch_batched` submodule path -- resolved
    lazily (PEP 562 module `__getattr__`), so `import chmm_lsim` itself still
    never imports `torch` (confirmed by the dedicated core-only-venv check;
    see docs/PYTHON_PORT_PLAN.md's P8a row)."""
    import chmm_lsim
    from chmm_lsim.backends.torch_batched import BatchedPosteriorChunk, TorchKernel, estimate_chunk_size, forward_backward_batch

    assert chmm_lsim.forward_backward_batch is forward_backward_batch
    assert chmm_lsim.BatchedPosteriorChunk is BatchedPosteriorChunk
    assert chmm_lsim.TorchKernel is TorchKernel
    assert chmm_lsim.estimate_chunk_size is estimate_chunk_size

    from chmm_lsim import forward_backward_batch as fbb_reimport

    assert fbb_reimport is forward_backward_batch
    assert "forward_backward_batch" in dir(chmm_lsim)

    with pytest.raises(AttributeError):
        chmm_lsim.this_attribute_does_not_exist_anywhere


def test_lsim_forward_backward_batch_attribute_raises_missing_dependency_when_torch_absent(monkeypatch):
    """Simulates `torch` genuinely not being importable (same technique as
    `test_missing_torch_dependency_raises_actionable_error` above) and
    confirms accessing `chmm_lsim.forward_backward_batch` -- the lazy `__getattr__`
    path, not `backends.base.get_backend` directly -- raises the same clear
    `MissingBackendDependency`, not a confusing raw `ImportError` from
    inside `lsim/__init__.py`'s own attribute resolution."""
    import chmm_lsim
    import chmm_lsim.backends as backends_pkg
    from chmm_lsim.backends import base as base_module

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "chmm_lsim.backends.torch_batched", raising=False)
    monkeypatch.delitem(base_module._BACKENDS, "torch", raising=False)
    monkeypatch.delitem(base_module._RUNNERS, "torch", raising=False)
    # A successful `import chmm_lsim.backends.torch_batched` earlier in this
    # session also cached it as an *attribute* of the `chmm_lsim.backends`
    # package object, separately from `sys.modules` -- `from chmm_lsim.backends
    # import torch_batched` (what `chmm_lsim.__getattr__` does) resolves via that
    # attribute if present, bypassing the import machinery's own
    # `sys.modules` check entirely; it must be cleared too for this
    # simulated-absence test to actually re-trigger the (now-failing) import.
    if hasattr(backends_pkg, "torch_batched"):
        monkeypatch.delattr(backends_pkg, "torch_batched")

    with pytest.raises(MissingBackendDependency, match=r"pip install 'chmm-lsim\[torch\]'"):
        chmm_lsim.forward_backward_batch


# --------------------------------------------------------------------------
# Full-covariance log-domain emissions: large-offset / extreme-observation,
# demonstrating scaling avoids float32 underflow without changing results.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1e4, 1e6])
def test_full_covariance_large_offset_matches_numpy(offset):
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    rng = np.random.default_rng(3)
    pi_0 = rng.dirichlet(np.ones(2))
    transition = rng.dirichlet(np.ones(2) * 3, 2)
    coupling = np.array([[1.0]])
    A = rng.normal(size=(2, 2, 2))
    sigma = np.stack([A[:, :, s] @ A[:, :, s].T + np.eye(2) * 0.3 for s in range(2)], axis=-1)[:, :, :, None].reshape(2, 2, 2, 1)
    mu = (offset + rng.normal(scale=2.0, size=(2, 2, 1)))
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=mu, sigma=sigma, diag=False)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()

    Y, _ = simulate_lsim(structure, params, 30, seed=1)
    fbc = FbConstants(variant="sed")
    post_np = run_backend("numpy", "sed", structure, params, Y, fbc)
    post_torch = run_backend("torch", "sed", structure, params, Y, fbc)
    np.testing.assert_allclose(post_torch.alpha, post_np.alpha, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(post_torch.gamma, post_np.gamma, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(post_torch.loglik, post_np.loglik, rtol=1e-8, atol=1e-8)


def test_eig_logpdf_matches_scipy_well_conditioned_and_singular():
    """Direct unit test of `_eig_logpdf` against `scipy.stats.multivariate_normal`
    for both a well-conditioned and a rank-1 singular-but-PSD covariance --
    the exact two cases the P8a request calls out by name."""
    from scipy.stats import multivariate_normal

    from chmm_lsim.backends.torch_batched import _eig_logpdf

    rng = np.random.default_rng(1)
    d = 3
    A = rng.normal(size=(d, d))
    sigma_full = A @ A.T + np.eye(d) * 0.1
    mu = rng.normal(size=d)
    x = rng.normal(size=(5, d))
    ref = multivariate_normal.logpdf(x, mean=mu, cov=sigma_full, allow_singular=True)
    got = _eig_logpdf(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(sigma_full)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)

    v = rng.normal(size=d)
    sigma_singular = np.outer(v, v)
    x2 = mu[None, :] + rng.normal(size=(5, 1)) * v[None, :]
    ref2 = multivariate_normal.logpdf(x2, mean=mu, cov=sigma_singular, allow_singular=True)
    got2 = _eig_logpdf(torch.from_numpy(x2), torch.from_numpy(mu), torch.from_numpy(sigma_singular)).numpy()
    np.testing.assert_allclose(got2, ref2, rtol=1e-9, atol=1e-9)

    # Off-support, same rank-1 singular sigma: a generic direction is off the
    # 1-D support subspace almost surely -- scipy returns -inf there, and a
    # tolerance-based comparison must not hide a finite-vs--inf mismatch (no
    # finite atol/rtol can ever make a real number "close to" -inf, so a
    # plain assert_allclose is already a strict, non-hiding check here).
    x2_off = mu[None, :] + rng.normal(size=(5, d))
    ref2_off = multivariate_normal.logpdf(x2_off, mean=mu, cov=sigma_singular, allow_singular=True)
    got2_off = _eig_logpdf(torch.from_numpy(x2_off), torch.from_numpy(mu), torch.from_numpy(sigma_singular)).numpy()
    assert np.all(np.isneginf(ref2_off)), "test setup: expected every off-support point to be -inf under scipy"
    np.testing.assert_allclose(got2_off, ref2_off)


def test_eig_logpdf_on_and_off_support_literal_case():
    """The exact literal case a review round reproduced the bug with:
    `sigma=[[1,1],[1,1]]` (rank-1 singular), `x=[1,1]` lies exactly on the
    1-D support subspace (scipy: finite), `x=[1,0]` does not (scipy: -inf).
    The previous `_eig_logpdf` silently dropped the null-space contribution
    to the quadratic form without ever checking whether the deviation
    actually had a nonzero component there, so it returned a *finite* value
    off-support instead of `-inf` -- confirmed as a real, not hypothetical,
    discrepancy before this fix."""
    from scipy.stats import multivariate_normal

    from chmm_lsim.backends.torch_batched import _eig_logpdf

    sigma = np.array([[1.0, 1.0], [1.0, 1.0]])
    mu = np.zeros(2)
    x_on = np.array([[1.0, 1.0]])
    x_off = np.array([[1.0, 0.0]])

    ref_on = multivariate_normal.logpdf(x_on, mean=mu, cov=sigma, allow_singular=True)
    ref_off = multivariate_normal.logpdf(x_off, mean=mu, cov=sigma, allow_singular=True)
    assert np.isfinite(ref_on) and np.isneginf(ref_off), "test setup sanity check"

    got_on = _eig_logpdf(torch.from_numpy(x_on), torch.from_numpy(mu), torch.from_numpy(sigma)).numpy()
    got_off = _eig_logpdf(torch.from_numpy(x_off), torch.from_numpy(mu), torch.from_numpy(sigma)).numpy()
    np.testing.assert_allclose(got_on, ref_on, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got_off, ref_off)  # must be exactly -inf, not merely "close"


@pytest.mark.parametrize("e", [1e-7, 1e-8, 1e-9, 1e-10])
def test_eig_logpdf_near_threshold_matches_scipy_dtype_dependent_cutoff(e):
    """`sigma=diag(1,e)`, `x=[0,1e-4]` -- the literal near-threshold case a
    review round reproduced: a *fixed* `rel_tol=1e-8` disagreed with scipy's
    own dtype-dependent cutoff (`1e6*finfo(float64).eps*max_eigenvalue`,
    `~2.2e-10` here) by many orders of magnitude in both directions --
    wrongly zeroing a real eigenvalue at `e=1e-8` (scipy keeps it) and
    wrongly keeping one at `e=1e-10` (scipy -- and this fixed function, now
    -- correctly treats the point as off-support, `-inf`)."""
    from scipy.stats import multivariate_normal

    from chmm_lsim.backends.torch_batched import _eig_logpdf

    sigma = np.diag([1.0, e])
    mu = np.zeros(2)
    x = np.array([[0.0, 1e-4]])
    ref = multivariate_normal.logpdf(x, mean=mu, cov=sigma, allow_singular=True)
    got = _eig_logpdf(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(sigma)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-8)


def test_full_covariance_batch_emissions_off_support_gives_zero_not_nan():
    """A batched full-covariance emission computation where one state's
    covariance is singular and the observation is off-support for it: that
    state's `B`/`B_comp` must come out exactly `0` (matching `-inf`
    log-density, per the fix above), and -- critically -- the *other*
    states in the same channel (on-support / well-conditioned) must be
    completely unaffected, with no `NaN` anywhere in the batch."""
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    pi_0 = np.array([0.5, 0.5])
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    coupling = np.array([[1.0]])
    sigma_singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    sigma_regular = np.eye(2) * 0.5
    mu = np.zeros((2, 2, 1))
    sigma = np.stack([sigma_singular, sigma_regular], axis=-1)[:, :, None, :].reshape(2, 2, 2, 1)
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=mu, sigma=sigma, diag=False)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()

    from chmm_lsim.backends.torch_batched import TorchKernel, compute_emissions_torch

    Y = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)  # (1,D=2,T=1): x=[1,0] -- off-support for state 0, on for state 1
    kernel = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64)
    B, B_comp, max_log = compute_emissions_torch(kernel, Y)
    assert not torch.isnan(B).any()
    assert not torch.isnan(B_comp).any()
    assert not torch.isnan(max_log).any()
    assert float(B[0, 0, 0]) == 0.0  # state 0: off-support singular covariance -> exactly 0
    assert float(B[0, 1, 0]) > 0.0  # state 1: regular covariance, unaffected


def test_diagonal_batch_emissions_no_nan_regression(fixtures_dir):
    """Companion to the full-covariance test above: confirms the (unrelated,
    always-full-rank-by-construction) diagonal path was never affected by
    this fix and still produces no `NaN` on a real fixture."""
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    from chmm_lsim.backends.torch_batched import TorchKernel, compute_emissions_torch

    Y = torch.from_numpy(fx.trials[0].Y[None, :, :])
    kernel = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B, B_comp, max_log = compute_emissions_torch(kernel, Y)
    assert not torch.isnan(B).any()
    assert not torch.isnan(B_comp).any()


# --------------------------------------------------------------------------
# Dataset validation: empty, malformed, wrong D, integer, Inf, T=0.
# --------------------------------------------------------------------------


def test_forward_backward_batch_rejects_empty_dataset():
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    with pytest.raises(ValidationError, match="empty"):
        forward_backward_batch(structure, params, [], FbConstants())


# --------------------------------------------------------------------------
# estimate_chunk_size: validation and a conservative, itemized peak-memory
# estimate (a review round found the previous version omitted several
# materially-sized concurrent tensors -- see the function's own docstring).
# --------------------------------------------------------------------------


def test_estimate_chunk_size_rejects_non_positive_max_memory_bytes():
    from chmm_lsim.backends.torch_batched import estimate_chunk_size

    for bad in (0, -1, -1000):
        with pytest.raises(ValidationError, match="max_memory_bytes"):
            estimate_chunk_size(10, 7, 20, 2, device=torch.device("cpu"), dtype=torch.float64, max_memory_bytes=bad)


def test_estimate_chunk_size_rejects_non_integer_max_memory_bytes():
    from chmm_lsim.backends.torch_batched import estimate_chunk_size

    with pytest.raises(ValidationError, match="max_memory_bytes"):
        estimate_chunk_size(10, 7, 20, 2, device=torch.device("cpu"), dtype=torch.float64, max_memory_bytes=1024.5)


def test_estimate_chunk_size_raises_before_allocation_when_budget_too_small():
    """A budget too small for even one sequence must raise a clear error --
    computed *before* any allocation, never silently clamped to `chunk_size=1`
    and then left to fail (or worse, succeed while quietly exceeding the
    stated budget) once real tensors are allocated."""
    from chmm_lsim.backends.torch_batched import estimate_chunk_size

    with pytest.raises(ValidationError, match="chunk_size < 1|exceeds the available budget"):
        estimate_chunk_size(1000, 500, 2000, 10, device=torch.device("cpu"), dtype=torch.float64, max_memory_bytes=1024, D=50, max_d=50)


def test_estimate_chunk_size_accounts_for_itemized_concurrent_tensors():
    """Direct regression for the specific gaps a review round found: the
    estimate must scale with `D` and `max_d` (both previously ignored
    entirely) and must be more conservative (smaller, for the same budget)
    than a naive "5 (S,T) arrays + one (S,S) scratch" estimate would give,
    since real concurrent SED-backward/full-covariance-stats intermediates
    are wider than that."""
    from chmm_lsim.backends.torch_batched import estimate_chunk_size

    S, T_max, max_K, N = 20, 100, 3, 10_000
    budget = 200_000_000  # 200 MB
    naive_per_seq_bytes = (5 * S * T_max + S * max_K * T_max + S * T_max + S * S) * 8
    naive_chunk_size = budget // naive_per_seq_bytes

    got_no_d = estimate_chunk_size(N, S, T_max, max_K, device=torch.device("cpu"), dtype=torch.float64, max_memory_bytes=budget)
    got_with_d = estimate_chunk_size(N, S, T_max, max_K, device=torch.device("cpu"), dtype=torch.float64, max_memory_bytes=budget, D=15, max_d=8)

    assert got_no_d < naive_chunk_size, "the itemized estimate must be strictly more conservative than the old, incomplete one"
    assert got_with_d <= got_no_d, "accounting for D/max_d (both nonzero here) must not increase the estimate"
    assert got_with_d >= 1


def test_forward_backward_batch_rejects_non_integer_chunk_size():
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(ValidationError, match="chunk_size"):
        forward_backward_batch(structure, params, [Y], FbConstants(), chunk_size=2.5)
    with pytest.raises(ValidationError, match="chunk_size"):
        forward_backward_batch(structure, params, [Y], FbConstants(), chunk_size=0)
    with pytest.raises(ValidationError, match="chunk_size"):
        forward_backward_batch(structure, params, [Y], FbConstants(), chunk_size=-3)


# --------------------------------------------------------------------------
# max_memory_bytes must be validated unconditionally -- a review round found
# it was silently ignored whenever an explicit chunk_size was also given
# (forward_backward_batch(..., chunk_size=1, max_memory_bytes=0) and
# max_memory_bytes=1.5 both silently "succeeded"), since the only place this
# used to be checked (estimate_chunk_size) is skipped entirely in that case.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_mmb", [0, -1, -1000, 1.5, "512", True, False])
def test_forward_backward_batch_rejects_invalid_max_memory_bytes_with_explicit_chunk_size(bad_mmb):
    """The exact planted reproduction from the correction request: an
    explicit chunk_size must not bypass max_memory_bytes validation."""
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(ValidationError, match="max_memory_bytes"):
        forward_backward_batch(structure, params, [Y], FbConstants(), chunk_size=1, max_memory_bytes=bad_mmb)


@pytest.mark.parametrize("bad_mmb", [0, -1, -1000, 1.5, "512", True, False])
def test_forward_backward_batch_rejects_invalid_max_memory_bytes_with_automatic_chunk_size(bad_mmb):
    """Same invalid values, but with chunk_size left automatic -- this path
    already routed through estimate_chunk_size before this correction round,
    kept here as an explicit regression alongside its explicit-chunk_size
    counterpart above so both paths are pinned down together."""
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with pytest.raises(ValidationError, match="max_memory_bytes"):
        forward_backward_batch(structure, params, [Y], FbConstants(), max_memory_bytes=bad_mmb)


def test_forward_backward_batch_valid_max_memory_bytes_accepted_with_explicit_chunk_size():
    """A large-enough, valid max_memory_bytes alongside an explicit
    chunk_size must not raise -- confirms the fix rejects only genuinely
    invalid values, not the combination itself."""
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        list(forward_backward_batch(structure, params, [Y], FbConstants(), mode="posterior", chunk_size=1, max_memory_bytes=1_000_000_000))


def test_forward_backward_batch_explicit_chunk_size_exceeding_max_memory_bytes_raises():
    """Combination semantics, made explicit: when both chunk_size and
    max_memory_bytes are given, both are enforced -- here even the single
    (`N=1`) sequence's own itemized peak-memory estimate (a long `T=500`
    sequence) alone exceeds the tiny stated budget, so this must raise
    regardless of `chunk_size`'s own value -- not silently proceed."""
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 500))
    with pytest.raises(ValidationError, match="max_memory_bytes"):
        forward_backward_batch(structure, params, [Y], FbConstants(), chunk_size=10_000, max_memory_bytes=1024)


# --------------------------------------------------------------------------
# chunk_size is only an upper *cap* -- the largest chunk ever actually built
# is min(chunk_size, N). A review round found the max_memory_bytes cross-
# check above used chunk_size directly (not min(chunk_size, N)), so
# N=1, chunk_size=1000 falsely rejected a call that would only ever
# allocate a one-sequence chunk.
# --------------------------------------------------------------------------


def test_forward_backward_batch_large_chunk_size_cap_with_single_sequence_succeeds():
    """The exact planted reproduction from the correction request: N=1, a
    very large chunk_size cap (1000), and a budget that comfortably fits
    one sequence but not 1000 -- must succeed, since _bucket_by_length can
    only ever build a one-sequence chunk here regardless of the cap."""
    from chmm_lsim.backends.torch_batched import _estimate_per_seq_bytes, forward_backward_batch

    structure, params = _tiny_sed_model()
    T = 5
    Y = np.zeros((1, T))
    per_seq_bytes = _estimate_per_seq_bytes(structure.S, T, structure.max_K, structure.D, structure.max_d, torch.float64)
    budget = per_seq_bytes + 500  # comfortably fits one sequence
    assert 1000 * per_seq_bytes > budget, "sanity: 1000 sequences' worth must NOT fit, or this test proves nothing"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loglik = forward_backward_batch(structure, params, [Y], FbConstants(), chunk_size=1000, max_memory_bytes=budget)
    assert loglik.shape == (1,)


def test_forward_backward_batch_min_chunk_size_and_n_genuinely_exceeding_budget_raises():
    """A dataset where min(chunk_size, N) -- not chunk_size alone -- still
    genuinely exceeds the budget must still raise: N=5 short sequences,
    chunk_size=5 (==N, so the cap is never the binding constraint here),
    budget sized to fit exactly one sequence but not five."""
    from chmm_lsim.backends.torch_batched import _estimate_per_seq_bytes, forward_backward_batch

    structure, params = _tiny_sed_model()
    T = 5
    dataset = [np.zeros((1, T)) for _ in range(5)]
    per_seq_bytes = _estimate_per_seq_bytes(structure.S, T, structure.max_K, structure.D, structure.max_d, torch.float64)
    budget = per_seq_bytes + 500  # fits 1 sequence, not 5
    assert 5 * per_seq_bytes > budget

    with pytest.raises(ValidationError, match="max_memory_bytes"):
        forward_backward_batch(structure, params, dataset, FbConstants(), chunk_size=5, max_memory_bytes=budget)


def test_forward_backward_batch_chunk_size_equal_to_n_boundary():
    """chunk_size == N: the effective chunk size (min(chunk_size, N) == N)
    must be used exactly, neither off-by-one-larger nor smaller -- a budget
    that fits exactly N sequences' worth succeeds, one byte less raises."""
    from chmm_lsim.backends.torch_batched import _estimate_per_seq_bytes, forward_backward_batch

    structure, params = _tiny_sed_model()
    T, N = 5, 3
    dataset = [np.zeros((1, T)) for _ in range(N)]
    per_seq_bytes = _estimate_per_seq_bytes(structure.S, T, structure.max_K, structure.D, structure.max_d, torch.float64)
    exact_budget = N * per_seq_bytes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loglik = forward_backward_batch(structure, params, dataset, FbConstants(), chunk_size=N, max_memory_bytes=exact_budget)
    assert loglik.shape == (N,)

    with pytest.raises(ValidationError, match="max_memory_bytes"):
        forward_backward_batch(structure, params, dataset, FbConstants(), chunk_size=N, max_memory_bytes=exact_budget - 1)


# --------------------------------------------------------------------------
# CUDA budget must honor a user-supplied max_memory_bytes too, not just the
# 80%-of-free-memory figure -- mocked, no physical CUDA device required.
# --------------------------------------------------------------------------


def test_estimate_chunk_size_cuda_budget_capped_by_max_memory_bytes(monkeypatch):
    """`device="cuda"` previously used 80% of `torch.cuda.mem_get_info()`'s
    free bytes unconditionally, never applying a smaller user-supplied
    `max_memory_bytes` on this branch at all -- a public parameter silently
    not enforced on `cuda` specifically, even though the exact same
    parameter was already correctly enforced on `cpu`/`mps`. Mocks
    `torch.cuda.mem_get_info` to report abundant free memory, then confirms
    a tight `max_memory_bytes` still binds (a smaller resulting chunk_size
    than the free-memory figure alone would ever produce)."""
    from chmm_lsim.backends.torch_batched import _estimate_per_seq_bytes, estimate_chunk_size

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100 * 1024**3, 100 * 1024**3))  # 100 GiB "free"
    cuda_device = torch.device("cuda")
    S, T_max, max_K = 7, 50, 2

    unconstrained = estimate_chunk_size(10_000, S, T_max, max_K, device=cuda_device, dtype=torch.float64, max_memory_bytes=None)
    per_seq_bytes = _estimate_per_seq_bytes(S, T_max, max_K, 0, 0, torch.float64)
    tight_budget = 10 * per_seq_bytes
    constrained = estimate_chunk_size(10_000, S, T_max, max_K, device=cuda_device, dtype=torch.float64, max_memory_bytes=tight_budget)

    assert constrained <= 10, "a tight max_memory_bytes must cap the chunk size well below the 80%-of-100GiB figure"
    assert constrained < unconstrained


@pytest.mark.parametrize(
    "bad_Y,match",
    [
        (np.zeros(5), "2-D"),
        (np.zeros((99, 3)), "structure.D"),
        (np.zeros((1, 3), dtype=np.int64), "dtype"),
        (np.array([[1.0, np.inf, 2.0]]), "Inf"),
        (np.zeros((1, 0)), "T"),
    ],
)
def test_forward_backward_batch_validates_every_sequence(bad_Y, match):
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    good_Y = np.zeros((1, 5))
    with pytest.raises(ValidationError, match=match):
        forward_backward_batch(structure, params, [good_Y, bad_Y], FbConstants())  # second entry, not only the first


def test_forward_backward_batch_rejects_structure_mismatch():
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params = _tiny_sed_model()
    other = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    Y = np.zeros((2, 5))
    with pytest.raises(ValidationError, match="structure"):
        forward_backward_batch(other, params, [Y], FbConstants())


# --------------------------------------------------------------------------
# validate="all": not a no-op on the torch path.
# --------------------------------------------------------------------------


def test_torch_validate_all_does_not_spuriously_raise_on_real_fixture(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__sed")
    structure, params, fbc = _build(fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_all = run_backend("torch", "sed", structure, params, fx.trials[0].Y, fbc, validate="all")
        post_off = run_backend("torch", "sed", structure, params, fx.trials[0].Y, fbc, validate="off")
    np.testing.assert_allclose(post_all.alpha, post_off.alpha)
    np.testing.assert_allclose(post_all.gamma, post_off.gamma)


def test_torch_validate_all_catches_corrupted_alpha_block_sum():
    """Direct kernel-level test (mirrors `backends.numba_`'s own pattern):
    a hand-built `pi_0` that doesn't sum to 1 never reaches this deep
    through the public API (`params.validate()` always catches it first),
    but the compiled batched check itself must still detect and report it,
    naming the batch/original-sequence index and timestep."""
    from chmm_lsim.backends.torch_batched import TorchKernel, _forward_pass_batch

    structure, params = _tiny_sed_model()
    kernel = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64)
    B = torch.ones(1, structure.S, 3, dtype=torch.float64)
    lengths = torch.tensor([3], dtype=torch.int64)

    _forward_pass_batch(kernel, B, [0], lengths, False)  # must not raise when check_all=False

    bad_kernel = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64)
    bad_kernel.pi_0 = torch.tensor([0.3, 0.3], dtype=torch.float64)  # sums to 0.6
    with pytest.raises(ValidationError, match="alpha"):
        _forward_pass_batch(bad_kernel, B, [0], lengths, True)


# --------------------------------------------------------------------------
# validate="all" for variant="fast": completing the contract. A review round
# found `_backward_fast_batch` accepted neither `check_all` nor
# `sequence_indices` and ran none of `inference.fast`'s own `abar`/`gamma`
# invariant checks -- so a corrupted `abar`/`gamma` passed silently on the
# `fast` path while `sed` correctly raised. Fixed to mirror
# `_backward_sed_batch`'s own masked checks exactly (see
# `_backward_fast_batch`'s own docstring).
# --------------------------------------------------------------------------


def test_torch_fast_validate_all_completes_the_backward_invariant_contract(fixtures_dir, monkeypatch):
    """Exact reproduction from the correction request: `_forward_pass_batch`
    is monkeypatched to return an otherwise-valid `alpha` but a `scale`
    doubled at one real, non-terminal timestep, so `abar = alpha*scale` no
    longer block-sums to 1 there. `variant="sed"` already caught this
    (pre-existing coverage); `variant="fast"` previously completed silently
    -- both must now raise naming `abar`, via both `forward_backward_batch`
    directly and the ordinary batch-of-one `run_backend("torch", ...)`."""
    import chmm_lsim.backends.torch_batched as tb

    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__sed")
    structure, params, fbc_sed = _build(fx)
    fbc_fast = FbConstants(variant="fast", beta_floor=fbc_sed.beta_floor, pi0_reg=fbc_sed.pi0_reg, f_reg=fbc_sed.f_reg, svi_eps=fbc_sed.svi_eps)
    Y = fx.trials[0].Y
    assert Y.shape[1] >= 3, "need a real, non-terminal timestep (index 1) to corrupt"

    real_forward = tb._forward_pass_batch

    def _corrupt_scale_at_t1(kernel, Bemis, sequence_indices, lengths, check_all):
        alpha, scale, loglik = real_forward(kernel, Bemis, sequence_indices, lengths, check_all)
        scale = scale.clone()
        scale[:, :, 1] *= 2.0
        return alpha, scale, loglik

    monkeypatch.setattr(tb, "_forward_pass_batch", _corrupt_scale_at_t1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValidationError, match="abar"):
            list(tb.forward_backward_batch(structure, params, [Y], fbc_sed, mode="posterior", validate="all"))
        with pytest.raises(ValidationError, match="abar"):
            list(tb.forward_backward_batch(structure, params, [Y], fbc_fast, mode="posterior", validate="all"))
        with pytest.raises(ValidationError, match="abar"):
            run_backend("torch", "sed", structure, params, Y, fbc_sed, validate="all")
        with pytest.raises(ValidationError, match="abar"):
            run_backend("torch", "fast", structure, params, Y, fbc_fast, validate="all")


def _fast_backward_inputs(lengths_list, T):
    from chmm_lsim.backends.torch_batched import TorchKernel, _forward_pass_batch

    structure, params = _tiny_sed_model(C=1, M=(3,))
    kernel = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64)
    Bn = len(lengths_list)
    B = torch.ones(Bn, structure.S, T, dtype=torch.float64)
    lengths = torch.tensor(lengths_list, dtype=torch.int64)
    alpha, scale, _ = _forward_pass_batch(kernel, B, list(range(Bn)), lengths, False)
    fbc = FbConstants(variant="fast")
    return kernel, alpha, scale, lengths, fbc


def test_torch_fast_validate_all_does_not_raise_on_uncorrupted_inputs():
    from chmm_lsim.backends.torch_batched import _backward_fast_batch

    kernel, alpha, scale, lengths, fbc = _fast_backward_inputs([4], T=4)
    _backward_fast_batch(kernel, alpha, scale, fbc, [0], lengths, True)  # must not raise


def test_torch_fast_validate_all_catches_corrupted_abar_single_length_batch():
    """`abar = alpha*scale` corrupted at one real, non-terminal timestep,
    with `alpha` itself left untouched -- isolates `abar`'s own check from
    `alpha`'s (already covered by `_forward_pass_batch`'s own test above)."""
    from chmm_lsim.backends.torch_batched import _backward_fast_batch

    kernel, alpha, scale, lengths, fbc = _fast_backward_inputs([4], T=4)
    bad_scale = scale.clone()
    bad_scale[:, :, 1] *= 2.0
    with pytest.raises(ValidationError, match="abar"):
        _backward_fast_batch(kernel, alpha, bad_scale, fbc, [0], lengths, True)


def test_torch_fast_validate_all_catches_corrupted_terminal_gamma():
    """Terminal `gamma = alpha[T-1]*beta[T-1] = alpha[T-1]*scale[T-1]` is
    not itself re-normalized by the loop (unlike every non-terminal `gamma`,
    which the recursion's own division by `normalizer` forces to block-sum
    to 1 by construction) -- corrupting only the terminal `alpha` column
    leaves every `abar`/non-terminal-`gamma` check untouched and isolates
    the pre-loop terminal check added by this correction round."""
    from chmm_lsim.backends.torch_batched import _backward_fast_batch

    kernel, alpha, scale, lengths, fbc = _fast_backward_inputs([4], T=4)
    bad_alpha = alpha.clone()
    bad_alpha[:, :, -1] *= 2.0
    with pytest.raises(ValidationError, match="gamma"):
        _backward_fast_batch(kernel, bad_alpha, scale, fbc, [0], lengths, True)


def test_torch_fast_validate_all_catches_corrupted_non_terminal_gamma(monkeypatch):
    """A non-terminal `gamma`'s own block-sum-to-1 is mathematically
    guaranteed by the recursion's own explicit re-normalization step, so
    there is no finite, naturally-arising corruption that breaks it without
    first breaking `abar`'s own check at the same `tau` (both would derive
    from the same corrupted `alpha`/`scale` entry, and `abar` is checked
    first). To exercise the non-terminal `gamma` check's own wiring,
    message, and masking directly, `_check_block_sum_batched` is wrapped to
    force exactly one call to report failure -- by call order, since every
    other call (the terminal check, every `abar` check, every other `gamma`
    check) is left passing through to the real implementation unchanged.
    For `T=4`, `check_all=True`, the call sequence is: 0=terminal gamma;
    1=abar@tau=2, 2=gamma@tau=2; 3=abar@tau=1, 4=gamma@tau=1 (forced to
    fail here); 5=abar@tau=0, 6=gamma@tau=0 (never reached)."""
    import chmm_lsim.backends.torch_batched as tb

    kernel, alpha, scale, lengths, fbc = _fast_backward_inputs([4], T=4)
    real_check = tb._check_block_sum_batched
    calls = {"n": 0}
    TARGET_CALL = 4

    def _spy(x, E, atol=1e-8):
        ok = real_check(x, E, atol)
        idx = calls["n"]
        calls["n"] += 1
        return torch.zeros_like(ok) if idx == TARGET_CALL else ok

    monkeypatch.setattr(tb, "_check_block_sum_batched", _spy)
    with pytest.raises(ValidationError, match="gamma"):
        tb._backward_fast_batch(kernel, alpha, scale, fbc, [0], lengths, True)
    assert calls["n"] == TARGET_CALL + 1, "must have reached exactly the targeted call, not stopped earlier or later"


def test_torch_fast_validate_all_ignores_padding_only_corruption_in_ragged_batch():
    """A short sequence batched beside a longer one: corrupting `alpha`/
    `scale` *only* in the short sequence's padded region (`t >=
    lengths[0]`) must never raise -- `active_bw`/`real_ts` mask every check
    by each row's own real length, exactly as `_backward_sed_batch`'s own
    ragged-batch masking does."""
    from chmm_lsim.backends.torch_batched import _backward_fast_batch

    T, short_len = 6, 2
    kernel, alpha, scale, lengths, fbc = _fast_backward_inputs([short_len, T], T=T)

    bad_alpha, bad_scale = alpha.clone(), scale.clone()
    bad_alpha[0, :, short_len:] *= 3.0  # padding-only region for the short sequence
    bad_scale[0, :, short_len:] *= 5.0

    _backward_fast_batch(kernel, bad_alpha, bad_scale, fbc, [0, 1], lengths, True)  # must not raise


# --------------------------------------------------------------------------
# P8b infrastructure: CUDA-marked tests, skip cleanly here (no CUDA device
# on this development machine), run for real once P8b executes on a GPU
# cluster. Never fabricate or assume their result in the meantime -- these
# either run and pass/fail for real, or are skipped and reported as such.
# --------------------------------------------------------------------------


@requires_cuda
def test_cuda_matches_numpy_fp64_reference(fixtures_dir):
    fx = load_posterior_fixture(fixtures_dir, "c3_het_diag__sed")
    structure, params, fbc = _build(fx)
    Y = fx.trials[0].Y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post_np = run_backend("numpy", "sed", structure, params, Y, fbc)
        post_cuda = run_backend("torch", "sed", structure, params, Y, fbc, device="cuda", dtype="float32")
    rel = np.abs(post_cuda.loglik - post_np.loglik) / (abs(post_np.loglik) + 1e-300)
    assert rel <= 1e-4, "P8b's own fp32-CUDA-vs-NumPy-fp64 target tolerance"


@requires_cuda
def test_cuda_device_selected_when_requested():
    structure, params = _tiny_sed_model()
    Y = np.zeros((1, 5))
    post = run_backend("torch", "sed", structure, params, Y, FbConstants(variant="sed"), device="cuda", dtype="float32", validate="off")
    assert np.all(np.isfinite(post.alpha))


# --------------------------------------------------------------------------
# P8b finding (reproduced on CPU and CUDA): the batched Torch kernel must apply
# the same per-channel `pi0_reg` smoothing of `pi_0` as the NumPy/Numba kernels
# (`inference.sed._regularize_pi0`) before the forward pass. Every committed
# fixture has a strictly positive `pi_0`, where the ~`pi0_reg` absolute effect
# hides inside `atol=1e-9`; an EM-updated `pi_0` with an exactly-zero entry
# (e.g. c3_lowmass_full after one iteration) turned that into exact-zero
# alpha/gamma on Torch versus ~1e-10 on NumPy, and a different transition row.
# --------------------------------------------------------------------------


def _zero_pi0_model(variant):
    structure, params = _tiny_sed_model(C=2, M=(3, 2))
    params = params.copy()
    params.pi_0 = np.array([0.0, 0.25, 0.75, 1.0, 0.0])  # exact zeros in both channel blocks
    params.validate()
    rng = np.random.default_rng(11)
    Y = rng.normal(scale=3.0, size=(structure.D, 12))
    fbc = FbConstants(variant=variant)
    assert fbc.pi0_reg > 0, "test premise: the canonical constants regularize pi_0"
    return structure, params, Y, fbc


@pytest.mark.parametrize("variant", ["sed", "fast"])
def test_torch_applies_pi0_regularization_like_numpy(variant):
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params, Y, fbc = _zero_pi0_model(variant)
    post_np = run_backend("numpy", variant, structure, params, Y, fbc)
    post_t = run_backend("torch", variant, structure, params, Y, fbc)
    assert post_np.alpha[0, 0] > 0 and post_np.alpha[4, 0] > 0, "numpy smooths the zero pi_0 entries"
    for name in ("alpha", "beta", "gamma", "scale"):
        a, b = getattr(post_t, name), getattr(post_np, name)
        np.testing.assert_array_equal(a == 0, b == 0, err_msg=f"{variant} {name}: zero pattern differs")
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-300, err_msg=f"{variant} {name}")
    np.testing.assert_allclose(post_t.loglik, post_np.loglik, rtol=1e-12)
    ll = forward_backward_batch(structure, params, [Y], fbc, mode="loglik")
    np.testing.assert_allclose(ll[0], post_np.loglik, rtol=1e-12)
    s_t = e_step(structure, params, [Y], fbc, backend="torch")
    s_np = e_step(structure, params, [Y], fbc, backend="numpy")
    np.testing.assert_allclose(s_t.gamma_first, s_np.gamma_first, rtol=1e-9, atol=1e-300)


def test_torch_kernel_build_default_keeps_raw_pi0():
    """`TorchKernel.build`'s own default (`pi0_reg=0.0`) stores `params.pi_0`
    unchanged; `forward_backward_batch` passes `fb_constants.pi0_reg`."""
    from chmm_lsim.backends.torch_batched import TorchKernel

    structure, params, _Y, fbc = _zero_pi0_model("sed")
    raw = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64)
    np.testing.assert_array_equal(raw.pi_0.numpy(), params.pi_0)
    reg = TorchKernel.build(structure, params, device=torch.device("cpu"), dtype=torch.float64, pi0_reg=fbc.pi0_reg)
    from chmm_lsim.inference.sed import _regularize_pi0

    np.testing.assert_array_equal(reg.pi_0.numpy(), _regularize_pi0(params.pi_0, structure, fbc.pi0_reg))
