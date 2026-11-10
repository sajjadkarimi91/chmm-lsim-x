"""Unit tests for chmm_lsim.inference.base.Posterior (P3a): .channel() and
.B_raw() with a synthetic emission-rescaled (non-None log_scale) Posterior --
no forward_backward_sed run needed, and no reliance on P3a's `log_scale=None`
path leaving this uncovered."""

from __future__ import annotations

import numpy as np
import pytest

from chmm_lsim.models.lsim.inference.base import Diagnostics, Posterior
from chmm_lsim.core.structure import LSIMStructure


def _make_posterior(log_scale: np.ndarray) -> Posterior:
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 2))
    S, T = structure.S, log_scale.shape[1]
    rng = np.random.default_rng(0)
    B = rng.uniform(0.1, 1.0, size=(S, T))
    return Posterior(
        loglik=-12.3,
        alpha=rng.uniform(size=(S, T)),
        beta=rng.uniform(size=(S, T)),
        gamma=rng.uniform(size=(S, T)),
        scale=rng.uniform(size=(S, T)),
        B=B,
        B_comp=rng.uniform(size=(S, 2, T)),
        log_scale=log_scale,
        diagnostics=Diagnostics(),
        structure=structure,
    )


def test_b_raw_with_rescaling_on_full_posterior():
    structure_C, T = 2, 4
    log_scale = np.array([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
    post = _make_posterior(log_scale)
    raw = post.B_raw()
    expected = post.B * np.exp(log_scale[post.structure.block_of_state])
    np.testing.assert_array_equal(raw, expected)


def test_channel_view_b_raw_with_rescaling_does_not_raise_indexerror():
    """Regression test: .channel(c).B_raw() used to index the *sliced*
    log_scale (shape (1,T)) with the *full* structure's block_of_state
    (values 0..C-1), raising IndexError for every channel c>0 whose sliced
    view has states beyond the first. c=1 here (channel_state_num=(2,3), so
    channel 1 has 3 states) is exactly that case."""
    log_scale = np.array([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
    post = _make_posterior(log_scale)

    ch1 = post.channel(1)
    raw = ch1.B_raw()  # must not raise IndexError

    assert ch1.log_scale.shape == (1, 4)
    assert ch1.structure.C == 1
    expected = ch1.B * np.exp(log_scale[1:2, :])  # channel 1's own row, broadcast over its 3 states
    np.testing.assert_array_equal(raw, expected)


def test_channel_view_without_rescaling_b_raw_is_noop():
    structure = LSIMStructure((2, 3), (1, 2), (1, 2))
    S, T = structure.S, 4
    rng = np.random.default_rng(1)
    B = rng.uniform(size=(S, T))
    post = Posterior(
        loglik=0.0,
        alpha=rng.uniform(size=(S, T)),
        beta=rng.uniform(size=(S, T)),
        gamma=rng.uniform(size=(S, T)),
        scale=rng.uniform(size=(S, T)),
        B=B,
        B_comp=rng.uniform(size=(S, 2, T)),
        log_scale=None,
        structure=structure,
    )
    ch1 = post.channel(1)
    np.testing.assert_array_equal(ch1.B_raw(), ch1.B)


# --------------------------------------------------------------------------
# Heterogeneous K_c: .channel(c).B_comp must slice the K axis down to that
# channel's own num_gmm_component, not the global max across all channels
# (forward_backward_lsim.m:86's `P_observ_cond_to_state_comp(zee_index,
# 1:num_gmm_component(zee), :)` -- c5_K_mixed's own K_c is (1,2,3,2,1)).
# --------------------------------------------------------------------------

def _make_posterior_with_num_gmm_component(num_gmm_component: tuple[int, ...]) -> Posterior:
    structure = LSIMStructure(
        channel_state_num=tuple([2] * len(num_gmm_component)),
        channel_obs_dim=tuple([1] * len(num_gmm_component)),
        num_gmm_component=num_gmm_component,
    )
    S, T, Kmax = structure.S, 4, structure.max_K
    rng = np.random.default_rng(2)
    return Posterior(
        loglik=0.0,
        alpha=rng.uniform(size=(S, T)),
        beta=rng.uniform(size=(S, T)),
        gamma=rng.uniform(size=(S, T)),
        scale=rng.uniform(size=(S, T)),
        B=rng.uniform(size=(S, T)),
        B_comp=rng.uniform(size=(S, Kmax, T)),
        log_scale=None,
        structure=structure,
    )


def test_channel_view_b_comp_slices_to_own_k_c_not_global_max_k():
    num_gmm_component = (1, 2, 3, 2, 1)  # c5_K_mixed's own K_c
    post = _make_posterior_with_num_gmm_component(num_gmm_component)
    assert post.B_comp.shape[1] == 3  # global max_K

    for c, K_c in enumerate(num_gmm_component):
        ch = post.channel(c)
        assert ch.B_comp.shape[1] == K_c, f"channel {c}: B_comp.shape[1]={ch.B_comp.shape[1]} != K_c={K_c}"
        assert ch.structure.max_K == K_c
        sl = post.structure.state_slice(c)
        np.testing.assert_array_equal(ch.B_comp, post.B_comp[sl][:, :K_c, :])


def test_channel_view_b_comp_matches_real_heterogeneous_k_fixture(fixtures_dir):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(fixtures_dir)))
    from loader import load_posterior_fixture

    from chmm_lsim.core.constants import FbConstants
    from chmm_lsim.models.lsim.inference.sed import forward_backward_sed
    from chmm_lsim.models.lsim.params import LSIMParams

    fx = load_posterior_fixture(fixtures_dir, "c5_K_mixed__sed")
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
    fbc = FbConstants(
        variant=fx.fb_constants.variant,
        beta_floor=fx.fb_constants.beta_floor,
        pi0_reg=fx.fb_constants.pi0_reg,
        f_reg=fx.fb_constants.f_reg,
        svi_eps=fx.fb_constants.svi_eps,
        poise_refinement=fx.fb_constants.poise_refinement,
    )
    post = forward_backward_sed(structure, params, fx.trials[0].Y, fbc)

    assert tuple(structure.num_gmm_component) == (1, 2, 3, 2, 1)
    for c, K_c in enumerate(structure.num_gmm_component):
        ch = post.channel(c)
        assert ch.B_comp.shape == (structure.channel_state_num[c], K_c, fx.trials[0].Y.shape[1])
        # padding beyond this channel's own K_c must never be silently included
        assert ch.B_comp.shape[1] <= post.B_comp.shape[1]


# --------------------------------------------------------------------------
# channel() index bounds
# --------------------------------------------------------------------------

def test_channel_negative_index_rejected():
    post = _make_posterior_with_num_gmm_component((1, 2, 3))
    with pytest.raises(IndexError, match="out of range"):
        post.channel(-1)


def test_channel_index_too_large_rejected():
    post = _make_posterior_with_num_gmm_component((1, 2, 3))
    with pytest.raises(IndexError, match="out of range"):
        post.channel(3)
