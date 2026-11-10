"""P8c gate: the indexed one-hot forms in the batched Torch recursions must agree with the
original one-hot matmul/einsum forms, which are kept as the reference implementation
(`chmm_lsim.backends.torch_batched._ONEHOT_IMPL = "matmul"`).

`E` is one-hot, so `x @ E`, `E @ x` and `einsum("ic,bij,jw->bcw", E, x, E)` only broadcast a
per-channel value to that channel's states or sum a per-state axis within each channel's block --
but as matmuls they cost a factor `C` more floating-point operations, which dominates the SED
backward recursion on FP64-segmented GPUs. The indexed forms (`index_select`, block-contiguous
sums) compute the same quantities with no FP64 FLOP penalty.

Two documented floating-point differences are pinned here rather than assumed away:

1. **Reduction order.** A block sum adds only its own block's terms; the matmul adds all `S` terms
   (including exact zeros) in BLAS/cuBLAS order. Measured over all 48 committed posterior fixtures
   (both variants, every trial, float64), worst |difference| per quantity, CPU / CUDA (L40S):
   `B`, `B_comp`, `loglik` 0 / 0 (exact); `alpha` 1.1e-16 / 2.2e-16; `scale` 1.8e-15 / 1.8e-15;
   `gamma` 2.6e-11 / 3.8e-10; `beta` 1.6e-10 / 1.0e-9; `ExpectedStats` fields <= 1.3e-10 / 6.3e-10.
   The largest are all on `c3_sparse_l2r`, where the SED delta solve amplifies rounding -- the same
   order as the pre-existing Torch-vs-NumPy gap there. Every existing parity gate against
   NumPy/MATLAB passes unchanged, with unchanged tolerances. Nothing is bit-identical across
   machines here (BLAS reduction order differs between CPUs too), so these are tolerances, not
   equalities.
2. **NaN propagation.** `0 * NaN = NaN`, so a one-hot matmul spreads a NaN in one channel's entry
   across every state of the row, while a gather/segment sum confines it to that channel's own
   states. No committed fixture reaches that state (finite/NaN patterns match exactly on every
   fixture, both devices), but the difference is real and is pinned directly on the helpers below.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_em_trace_fixture, load_posterior_fixture  # noqa: E402

import chmm_lsim.backends.torch_batched as tb  # noqa: E402
from chmm_lsim.backends.base import run_backend  # noqa: E402
from chmm_lsim.core.constants import FbConstants  # noqa: E402
from chmm_lsim.backends.torch_batched import forward_backward_batch  # noqa: E402
from chmm_lsim.models.lsim.em import e_step, em_lsim  # noqa: E402
from chmm_lsim.learning.em_loop import finalize_expected_stats  # noqa: E402
from chmm_lsim.emissions.gmm import ChannelGMM  # noqa: E402
from chmm_lsim.models.lsim.params import LSIMParams  # noqa: E402
from chmm_lsim.core.structure import LSIMStructure  # noqa: E402
from chmm_lsim.core.validate import ValidationError  # noqa: E402

# Calibrated from the measurements in this module's docstring, with ~10x headroom over the worst
# (CUDA) value: beta 1.0e-9 -> 1e-8; alpha 2.2e-16 -> 1e-14; scale 1.8e-15 -> 1e-13.
ATOL_AMPLIFIED = 1e-8
ATOL_TIGHT = {"alpha": 1e-14, "scale": 1e-13}

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")),
]


def _opts(device):
    """`backend_options` for `e_step`/`em_lsim`, with the device stated explicitly."""
    return {"device": device, "dtype": "float64"}


@pytest.fixture
def impl_matmul():
    """Runs the body with the reference one-hot matmul forms, restoring the default afterwards."""
    previous = tb._ONEHOT_IMPL
    tb._ONEHOT_IMPL = "matmul"
    try:
        yield
    finally:
        tb._ONEHOT_IMPL = previous


def _both_impls(fn):
    """`fn()` under the indexed default and under the matmul reference (same device, same inputs)."""
    previous = tb._ONEHOT_IMPL
    try:
        tb._ONEHOT_IMPL = "indexed"
        indexed = fn()
        tb._ONEHOT_IMPL = "matmul"
        reference = fn()
    finally:
        tb._ONEHOT_IMPL = previous
    return indexed, reference


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
    fbc = FbConstants(variant=fx.fb_constants.variant, beta_floor=fx.fb_constants.beta_floor, pi0_reg=fx.fb_constants.pi0_reg,
                      f_reg=fx.fb_constants.f_reg, svi_eps=fx.fb_constants.svi_eps,
                      poise_refinement=fx.fb_constants.poise_refinement)
    return structure, params, fbc


def _kernel(M=(3, 2), device="cpu"):
    structure = LSIMStructure(channel_state_num=M, channel_obs_dim=(1,) * len(M), num_gmm_component=(1,) * len(M))
    rng = np.random.default_rng(0)
    S, C = structure.S, structure.C
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in M])
    transition = np.zeros((S, S))
    for z in range(C):
        transition[:, structure.state_slice(z)] = rng.dirichlet(np.ones(M[z]) * 3, size=S)
    coupling = rng.dirichlet(np.ones(C) * 2, size=C).T
    emissions = [ChannelGMM(np.ones((M[c], 1)), rng.normal(size=(1, M[c], 1)), np.ones((1, M[c], 1)), diag=True) for c in range(C)]
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    return tb.TorchKernel.build(structure, params, device=torch.device(device), dtype=torch.float64, pi0_reg=0.0), structure


def test_default_is_the_indexed_form():
    assert tb._ONEHOT_IMPL == "indexed"


def test_unknown_impl_is_rejected(monkeypatch):
    kernel, _structure = _kernel((2, 2))
    monkeypatch.setattr(tb, "_ONEHOT_IMPL", "nonsense")
    with pytest.raises(ValueError, match="_ONEHOT_IMPL"):
        tb._onehot_ops(kernel)


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("device", DEVICES)
def test_indexed_matches_matmul_on_every_fixture_and_trial(fixtures_dir, posterior_fixture_id, device):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    for i, trial in enumerate(fx.trials):
        got, ref = _both_impls(
            lambda: run_backend("torch", fbc.variant, structure, params, trial.Y, fbc, device=device, dtype="float64"))
        label = f"{posterior_fixture_id} trial {i} on {device}"
        for name, atol in ATOL_TIGHT.items():
            np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=0, atol=atol, err_msg=f"{label}: {name}")
        np.testing.assert_array_equal(got.B_raw(), ref.B_raw(), err_msg=f"{label}: B (emissions are untouched by this change)")
        np.testing.assert_array_equal(got.B_comp_raw(), ref.B_comp_raw(), err_msg=f"{label}: B_comp")
        np.testing.assert_allclose(got.loglik, ref.loglik, rtol=1e-12, atol=0, err_msg=f"{label}: loglik")
        assert got.diagnostics == ref.diagnostics, f"{label}: diagnostics"
        for name in ("beta", "gamma"):
            a, b = getattr(got, name), getattr(ref, name)
            np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=f"{label}: {name} finite pattern")
            np.testing.assert_allclose(a, b, rtol=0, atol=ATOL_AMPLIFIED, err_msg=f"{label}: {name}")


@pytest.mark.filterwarnings("ignore:channel .* is NaN for every timestep")
@pytest.mark.parametrize("device", DEVICES)
def test_indexed_matches_matmul_for_every_expected_stats_field(fixtures_dir, posterior_fixture_id, device):
    fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
    structure, params, fbc = _build(fx)
    Ys = [t.Y for t in fx.trials]
    got, ref = _both_impls(lambda: e_step(structure, params, Ys, fbc, backend="torch", backend_options=_opts(device)))
    for name in ("zeta", "gamma_first", "gamma_sum"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=0, atol=ATOL_AMPLIFIED, err_msg=name)
    np.testing.assert_allclose(got.loglik, ref.loglik, rtol=0, atol=ATOL_AMPLIFIED)
    assert (got.n_trials, got.n_transitions, got.T_all) == (ref.n_trials, ref.n_transitions, ref.T_all)
    for c, (a, b) in enumerate(zip(got.channels, ref.channels)):
        for name in ("resp", "weighted_W", "weighted_mean", "weighted_M2", "pairwise_Q", "pairwise_M", "pairwise_C",
                     "u_pairwise_Q", "u_pairwise_M", "u_pairwise_C", "n_obs", "welford_mean", "welford_M2",
                     "allzero_n", "allzero_mean", "allzero_M2"):
            x, y = getattr(a, name), getattr(b, name)
            assert (x is None) == (y is None)
            if x is not None:
                np.testing.assert_allclose(x, y, rtol=0, atol=ATOL_AMPLIFIED, err_msg=f"c={c} {name}")
        for name in ("resp_pos", "n_w_pos_diag", "n_w_pos_full"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=f"c={c} {name}")
    fin_got = finalize_expected_stats(got, sigma_diag=params.sigma_diag)
    fin_ref = finalize_expected_stats(ref, sigma_diag=params.sigma_diag)
    for c in range(structure.C):
        np.testing.assert_array_equal(fin_got.channels[c].fallback_mu, fin_ref.channels[c].fallback_mu)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scenario", ["c3_het_diag", "c3_het_full", "c3_trials_unequal", "c3_lowmass_full"])
def test_indexed_matches_matmul_over_multiple_em_iterations(fixtures_dir, scenario, device):
    """Rounding differences that the delta solve amplifies must not accumulate into a different
    EM trajectory: five full iterations, comparing every iteration's log-likelihood and the
    final parameters."""
    fx = load_em_trace_fixture(fixtures_dir, scenario)
    structure = LSIMStructure(tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist()))
    fbc = FbConstants(variant=fx.fb_constants.variant, beta_floor=fx.fb_constants.beta_floor, pi0_reg=fx.fb_constants.pi0_reg,
                      f_reg=fx.fb_constants.f_reg, svi_eps=fx.fb_constants.svi_eps,
                      poise_refinement=fx.fb_constants.poise_refinement)
    packed = fx.params_in(1)
    p0 = LSIMParams.from_packed(structure, pi_0=packed["pi_0"], transition=packed["transition_matrices_IM"],
                                coupling=packed["coupling_tetha_IM"], P_all=packed["P_all"], mu_all=packed["mu_all"],
                                sigma_all=packed["sigma_all"], sigma_diag=bool(fx.sigma_diag))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got, ref = _both_impls(lambda: em_lsim(structure, p0.copy(), fx.Y_trials, fb_constants=fbc, max_iterations=5,
                                               backend="torch", backend_options=_opts(device), check_convergence=False))
    np.testing.assert_allclose(got.loglik_trace, ref.loglik_trace, rtol=1e-11, atol=0)
    a, b = got.params.to_packed(), ref.params.to_packed()
    for k in ("pi_0", "transitions_matrices", "coupling_tetha_IM", "P_all", "mu_all", "sigma_all"):
        np.testing.assert_allclose(a[k], b[k], rtol=0, atol=ATOL_AMPLIFIED, err_msg=k)


# --------------------------------------------------------------------------
# Public-API regression for the documented non-finite propagation difference.
# --------------------------------------------------------------------------


def _unreachable_state_model(mu_far=100.0):
    """A model whose second state per channel is unreachable after t=0 (zero transition column).
    Placing an observation on that unreachable state's mean underflows the reachable state's
    rescaled emission to exactly 0, so the channel's `p_ot_c` is 0 and `scale` becomes NaN/Inf --
    a non-finite value that enters the backward recursion's one-hot operands."""
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    S = structure.S
    transition = np.zeros((S, S))
    for z in range(structure.C):
        block = np.zeros((S, 2))
        block[:, 0] = 1.0  # all mass to each block's first state; the second column is exactly zero
        transition[:, structure.state_slice(z)] = block
    emissions = [ChannelGMM(np.ones((2, 1)), np.array([[[0.0], [mu_far]]]), np.ones((1, 2, 1)), diag=True),
                 ChannelGMM(np.ones((2, 1)), np.array([[[0.0], [3.0]]]), np.ones((1, 2, 1)), diag=True)]
    params = LSIMParams(structure=structure, pi_0=np.full(S, 0.5), transition=transition, coupling=np.eye(2),
                        emissions=emissions)
    params.validate()
    return structure, params


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variant", ["sed", "fast"])
def test_public_api_non_finite_cases_agree_between_implementations(device, variant):
    """The helper-level NaN difference (pinned below) must not change what the PUBLIC API returns.

    Reaching a non-finite value in the one-hot operands through the public API is possible (the
    case built here returns `loglik = -inf` and non-finite posteriors), but the SED backward sums
    `beta_C * weights_subsystems` over channels, so a non-finite in one channel reaches every
    channel regardless of how the one-hot operations are evaluated. Both implementations therefore
    agree exactly -- including which entries are non-finite -- and that is what is asserted here,
    on the public entry points (`run_backend`, `forward_backward_batch` in both `posterior` and
    `loglik` mode, and `e_step`), on every device."""
    structure, params = _unreachable_state_model()
    fbc = FbConstants(variant=variant)
    Y_last = np.zeros((2, 5))
    Y_last[0, -1] = 100.0  # only the final timestep, so the forward pass still produces a non-NaN loglik
    Y_clean = np.zeros((2, 5))

    got, ref = _both_impls(lambda: run_backend("torch", variant, structure, params, Y_last, fbc, device=device, dtype="float64"))
    assert not np.isfinite(got.gamma).all(), "test premise: this case must produce non-finite posteriors"
    assert got.loglik == ref.loglik or (np.isnan(got.loglik) and np.isnan(ref.loglik))
    for name in ("alpha", "beta", "gamma", "scale"):
        a, b = getattr(got, name), getattr(ref, name)
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=f"{name}: non-finite pattern differs")
        np.testing.assert_allclose(a, b, rtol=0, atol=ATOL_AMPLIFIED, equal_nan=True, err_msg=name)
    assert got.diagnostics == ref.diagnostics

    data = [Y_last, Y_clean]
    got_ll, ref_ll = _both_impls(
        lambda: forward_backward_batch(structure, params, data, fbc, mode="loglik", device=device, dtype="float64", chunk_size=2))
    np.testing.assert_array_equal(np.isfinite(got_ll), np.isfinite(ref_ll))
    np.testing.assert_allclose(got_ll, ref_ll, rtol=0, atol=ATOL_AMPLIFIED, equal_nan=True)

    def posteriors():
        out = []
        for chunk in forward_backward_batch(structure, params, data, fbc, mode="posterior", device=device,
                                            dtype="float64", chunk_size=2):
            for local in range(len(chunk)):
                out.append((chunk.sequence_indices[local], chunk.extract(local)))
        return sorted(out, key=lambda t: t[0])

    got_posts, ref_posts = _both_impls(posteriors)
    for (_i, a), (_j, b) in zip(got_posts, ref_posts):
        for name in ("alpha", "beta", "gamma", "scale"):
            np.testing.assert_array_equal(np.isfinite(getattr(a, name)), np.isfinite(getattr(b, name)), err_msg=name)
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0, atol=ATOL_AMPLIFIED, equal_nan=True, err_msg=name)
        assert a.diagnostics == b.diagnostics

    got_stats, ref_stats = _both_impls(lambda: e_step(structure, params, data, fbc, backend="torch", backend_options=_opts(device)))
    for name in ("zeta", "gamma_first", "gamma_sum"):
        x, y = getattr(got_stats, name), getattr(ref_stats, name)
        np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y), err_msg=name)
        np.testing.assert_allclose(x, y, rtol=0, atol=ATOL_AMPLIFIED, equal_nan=True, err_msg=name)
    assert (got_stats.n_trials, got_stats.n_transitions, got_stats.T_all) == (ref_stats.n_trials, ref_stats.n_transitions, ref_stats.T_all)


@pytest.mark.parametrize("device", DEVICES)
def test_public_api_rejects_the_nan_loglik_case_identically(device):
    """The same construction with the extreme observation in the middle of the sequence makes the
    forward pass produce a NaN log-likelihood, which the backend rejects before any caller sees it
    -- identically under both implementations."""
    structure, params = _unreachable_state_model()
    fbc = FbConstants(variant="sed")
    Y = np.zeros((2, 5))
    Y[0, 2] = 100.0
    for impl in ("indexed", "matmul"):
        previous = tb._ONEHOT_IMPL
        tb._ONEHOT_IMPL = impl
        try:
            with pytest.raises(ValidationError, match="NaN log-likelihood"):
                run_backend("torch", "sed", structure, params, Y, fbc, device=device, dtype="float64")
        finally:
            tb._ONEHOT_IMPL = previous


# --------------------------------------------------------------------------
# The helpers themselves: broadcasting/reduction semantics, including the NaN
# difference the matmul form has and the indexed form deliberately does not.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("M", [(3, 2), (4, 4), (2, 3, 5)])
def test_indexed_helpers_match_matmul_forms_on_finite_input(M, device):
    kernel, structure = _kernel(M, device)
    S, C = structure.S, structure.C
    rng = np.random.default_rng(1)
    x_cs = torch.as_tensor(rng.normal(size=(3, C, S)), device=device)
    x_cc = torch.as_tensor(rng.normal(size=(3, C, C)), device=device)
    x_ss = torch.as_tensor(rng.normal(size=(3, S, S)), device=device)
    x_s = torch.as_tensor(rng.normal(size=(3, S)), device=device)

    tb._ONEHOT_IMPL = "indexed"
    bs_i, bc_i, sum_last_i, sum_both_i = tb._onehot_ops(kernel)
    tb._ONEHOT_IMPL = "matmul"
    bs_m, bc_m, sum_last_m, sum_both_m = tb._onehot_ops(kernel)
    tb._ONEHOT_IMPL = "indexed"

    np.testing.assert_array_equal(bs_i(x_cs).cpu().numpy(), bs_m(x_cs).cpu().numpy())  # exact: a gather vs a 0/1 matmul of the same values
    np.testing.assert_allclose(bc_i(x_cc).cpu().numpy(), bc_m(x_cc).cpu().numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(bc_i(x_s.reshape(3, S)[:, :C]).cpu().numpy(), bc_m(x_s.reshape(3, S)[:, :C]).cpu().numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(sum_last_i(x_ss).cpu().numpy(), sum_last_m(x_ss).cpu().numpy(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(sum_last_i(x_s).cpu().numpy(), sum_last_m(x_s).cpu().numpy(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(sum_both_i(x_ss).cpu().numpy(), sum_both_m(x_ss).cpu().numpy(), rtol=0, atol=1e-13)


def test_heterogeneous_and_equal_block_paths_agree():
    """Equal `M_c` takes the reshape path, heterogeneous `M_c` the per-channel slice path."""
    kernel_eq, structure_eq = _kernel((4, 4))
    kernel_het, structure_het = _kernel((3, 5))
    assert kernel_eq.equal_block_size == 4 and kernel_het.equal_block_size is None
    for kernel, structure in ((kernel_eq, structure_eq), (kernel_het, structure_het)):
        S = structure.S
        x = torch.as_tensor(np.random.default_rng(2).normal(size=(2, S, S)))
        tb._ONEHOT_IMPL = "indexed"
        _b, _bc, sum_last_i, sum_both_i = tb._onehot_ops(kernel)
        tb._ONEHOT_IMPL = "matmul"
        _b, _bc, sum_last_m, sum_both_m = tb._onehot_ops(kernel)
        tb._ONEHOT_IMPL = "indexed"
        np.testing.assert_allclose(sum_last_i(x).cpu().numpy(), sum_last_m(x).cpu().numpy(), rtol=0, atol=1e-13)
        np.testing.assert_allclose(sum_both_i(x).cpu().numpy(), sum_both_m(x).cpu().numpy(), rtol=0, atol=1e-13)


def test_nan_confined_to_its_own_channel_block_unlike_the_matmul_form():
    """The documented semantic difference, pinned: a NaN in one channel's entry stays inside that
    channel's states with the indexed form (`0 * NaN` never arises), while the one-hot matmul
    spreads it across the whole row. No committed fixture reaches this state -- the finite/NaN
    patterns match exactly on all of them -- but the difference is real, so it is asserted here
    rather than left to be discovered later."""
    kernel, structure = _kernel((3, 2))
    S, C = structure.S, structure.C
    x_cs = torch.zeros(1, C, S, dtype=torch.float64)
    x_cs[0, 0, :] = float("nan")  # channel 0's value is NaN

    tb._ONEHOT_IMPL = "indexed"
    bs_i, *_ = tb._onehot_ops(kernel)
    tb._ONEHOT_IMPL = "matmul"
    bs_m, *_ = tb._onehot_ops(kernel)
    tb._ONEHOT_IMPL = "indexed"

    indexed = bs_i(x_cs).cpu().numpy()
    reference = bs_m(x_cs).cpu().numpy()
    block0 = structure.state_slice(0)
    block1 = structure.state_slice(1)
    assert np.isnan(indexed[0, block0, :]).all(), "the affected channel's own states still carry the NaN"
    assert np.isfinite(indexed[0, block1, :]).all(), "the indexed form confines it to that channel"
    assert np.isnan(reference[0, block1, :]).all(), "the one-hot matmul spreads it (0 * NaN = NaN)"
