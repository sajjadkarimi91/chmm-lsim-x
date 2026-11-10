"""Extension points: the inference-variant registry (FbConstants validation, backend
dispatch) and the emission-model registry."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import chmm_lsim
from chmm_lsim import (
    CHMMParams,
    ChannelGMM,
    FbConstants,
    LSIMParams,
    LSIMStructure,
    UnsupportedCombination,
    forward_backward,
    forward_backward_poise,
    get_backend,
    register_inference,
)
from chmm_lsim.emissions import (
    EmissionModel,
    GaussianMixtureEmissions,
    emission_model_for,
    emission_model_for_stats,
    get_emission_model,
    registered_emission_models,
)
from chmm_lsim.models.lsim.inference import registry as inference_registry

DUMMY = "__test_registered_variant__"


def _model(seed=0):
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 2))
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for c in range(structure.C):
        for z in range(structure.C):
            transition[structure.state_slice(c), structure.state_slice(z)] = rng.dirichlet(
                np.ones(structure.channel_state_num[z]), structure.channel_state_num[c])
    coupling = rng.dirichlet(np.ones(structure.C), structure.C).T
    emissions = [
        ChannelGMM(weights=rng.dirichlet(np.ones(K), M), mu=rng.standard_normal((d, M, K)), sigma=0.5 + rng.random((d, M, K)), diag=True)
        for M, d, K in zip(structure.channel_state_num, structure.channel_obs_dim, structure.num_gmm_component)
    ]
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    Y, _ = chmm_lsim.simulate_lsim(structure, params, T=40, seed=seed)
    return structure, params, Y


@pytest.fixture
def dummy_variant():
    """A registered variant that runs POISE under another name."""
    calls = []

    @register_inference(DUMMY)
    def forward_backward_dummy(structure, params, Y, fb_constants=None, *, validate="params"):
        calls.append(fb_constants.variant)
        return forward_backward_poise(structure, params, Y, dataclasses.replace(fb_constants, variant="poise"), validate=validate)

    try:
        yield calls
    finally:
        del inference_registry._REGISTRY[DUMMY]
        inference_registry._LOGLIK_REGISTRY.pop(DUMMY, None)


# --------------------------------------------------------------------------------------------- inference
def test_builtin_variants_registered_in_order():
    assert inference_registry.registered_variants()[:3] == ("sed", "fast", "poise")


def test_unknown_variant_rejected_with_registered_list():
    with pytest.raises(ValueError, match=r"variant must be one of \('sed', 'fast', 'poise'"):
        FbConstants(variant="no_such_variant")


def test_registered_variant_is_accepted_and_runs_on_numpy(dummy_variant):
    structure, params, Y = _model()
    fbc = FbConstants(variant=DUMMY)
    assert DUMMY in get_backend("numpy").variants
    post = forward_backward(structure, params, Y, fbc)
    ref = forward_backward(structure, params, Y, FbConstants(variant="poise"))
    assert dummy_variant == [DUMMY]
    np.testing.assert_array_equal(post.gamma, ref.gamma)
    assert post.loglik == ref.loglik
    # EM's E-step dispatches through the same registry
    stats = chmm_lsim.e_step(structure, params, [Y], fbc)
    assert stats.loglik == ref.loglik
    # forward-only scoring uses the shared LSIM forward recursion
    from chmm_lsim.backends.base import run_backend_loglik

    assert run_backend_loglik("numpy", DUMMY, structure, params, Y, fbc) == ref.loglik


def test_registered_variant_custom_loglik_kernel():
    name = DUMMY + "_loglik"

    def loglik(variant, structure, params, Y, fb_constants, *, validate="params"):
        return -1.25

    register_inference(name, loglik=loglik)(lambda *a, **k: None)
    try:
        from chmm_lsim.backends.base import run_backend_loglik

        structure, params, Y = _model()
        assert run_backend_loglik("numpy", name, structure, params, Y, FbConstants(variant=name)) == -1.25
    finally:
        del inference_registry._REGISTRY[name]
        del inference_registry._LOGLIK_REGISTRY[name]


@pytest.mark.parametrize("backend", ["numba", "torch"])
def test_compiled_backends_reject_unported_variant(dummy_variant, backend):
    pytest.importorskip(backend)
    structure, params, Y = _model()
    assert DUMMY not in get_backend(backend).variants
    with pytest.raises(UnsupportedCombination, match=DUMMY):
        forward_backward(structure, params, Y, FbConstants(variant=DUMMY), backend=backend)


def test_torch_batched_rejects_unported_variant(dummy_variant):
    pytest.importorskip("torch")
    structure, params, Y = _model()
    with pytest.raises(UnsupportedCombination, match=DUMMY):
        chmm_lsim.forward_backward_batch(structure, params, [Y], FbConstants(variant=DUMMY), mode="loglik")


def test_duplicate_or_invalid_variant_name_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_inference("sed")(lambda *a, **k: None)
    with pytest.raises(ValueError, match="non-empty str"):
        register_inference("")


# --------------------------------------------------------------------------------------------- emissions
def test_gmm_is_found_for_lsim_and_chmm_params():
    structure, params, _ = _model()
    gmm = get_emission_model("gmm")
    assert isinstance(gmm, GaussianMixtureEmissions) and isinstance(gmm, EmissionModel)
    assert "gmm" in registered_emission_models()
    assert emission_model_for(params) is gmm
    assert emission_model_for(params.emissions) is gmm
    chmm = chmm_lsim.lsim_to_chmm(structure, params)
    assert isinstance(chmm, CHMMParams) and emission_model_for(chmm) is gmm
    raw = chmm_lsim.e_step(structure, params, [chmm_lsim.simulate_lsim(structure, params, T=10, seed=1)[0]])
    assert emission_model_for_stats(raw.channels) is gmm


def test_unknown_emission_model_raises_clear_error():
    with pytest.raises(ValueError, match=r"unknown emission model 'ar'; registered emission models are \[.*'gmm'.*\]"):
        get_emission_model("ar")


def test_unregistered_emission_type_raises_clear_error():
    @dataclasses.dataclass
    class NotRegistered:
        pass

    with pytest.raises(ValueError, match="no registered emission model handles .*NotRegistered"):
        emission_model_for([NotRegistered(), NotRegistered()])


def test_mixed_emission_types_rejected():
    structure, params, _ = _model()

    @dataclasses.dataclass
    class Other:
        pass

    with pytest.raises(ValueError, match=r"emissions\[1\]"):
        emission_model_for([params.emissions[0], Other()])


def test_gmm_registration_is_unique():
    from chmm_lsim.emissions import register_emission_model

    with pytest.raises(ValueError, match="already registered"):
        register_emission_model(GaussianMixtureEmissions())


def test_backend_emission_model_capabilities():
    assert get_backend("numpy").emission_models == registered_emission_models()
    for backend in ("numba", "torch"):
        try:
            caps = get_backend(backend)
        except ImportError:
            continue
        assert caps.emission_models == ("gmm",)
    with pytest.raises(UnsupportedCombination, match="emission model 'ar'"):
        chmm_lsim.check_capability("numpy", variant="poise", emission_model="ar")


def test_registered_emission_model_reaches_backend_check():
    """A second emission model (here a GMM subclass registered under another name) is
    run by NumPy and rejected by the GMM-specialised compiled backends."""
    from chmm_lsim.emissions import register_emission_model, unregister_emission_model

    @dataclasses.dataclass
    class OtherChannel(ChannelGMM):
        def copy(self):
            return OtherChannel(self.weights.copy(), self.mu.copy(), self.sigma.copy(), self.diag)

    @dataclasses.dataclass
    class OtherStats(chmm_lsim.ChannelExpectedStats):
        pass

    class Other(GaussianMixtureEmissions):
        name = "__test_other__"
        channel_type = OtherChannel
        stats_type = OtherStats

    register_emission_model(Other())
    try:
        structure, params, Y = _model()
        other = dataclasses.replace(params, emissions=[OtherChannel(g.weights, g.mu, g.sigma, g.diag) for g in params.emissions])
        assert emission_model_for(other).name == "__test_other__"
        assert "__test_other__" in get_backend("numpy").emission_models
        np.testing.assert_array_equal(forward_backward(structure, other, Y).gamma, forward_backward(structure, params, Y).gamma)
        try:
            get_backend("numba")
        except ImportError:
            return
        with pytest.raises(UnsupportedCombination, match="__test_other__"):
            forward_backward(structure, other, Y, backend="numba")
    finally:
        unregister_emission_model("__test_other__")


# --------------------------------------------------------------------------- inference on precomputed emissions
@pytest.mark.parametrize("variant,refine", [("sed", False), ("fast", False), ("poise", False), ("poise", True)])
def test_from_emissions_is_bit_identical_to_gmm_entry_points(variant, refine):
    structure, params, Y = _model(3)
    Y[0, 5] = np.nan
    fbc = FbConstants(variant=variant, poise_refinement=refine)
    ref = forward_backward(structure, params, Y, fbc)
    B, B_comp = chmm_lsim.compute_emissions(structure, params, Y)
    post = chmm_lsim.forward_backward_from_emissions(structure, params.pi_0, params.transition, params.coupling, B, B_comp, fbc)
    for name in ("alpha", "beta", "gamma", "scale", "B", "B_comp"):
        np.testing.assert_array_equal(getattr(post, name), getattr(ref, name))
    assert post.loglik == ref.loglik and post.log_scale is None
    assert chmm_lsim.forward_loglik_from_emissions(structure, params.pi_0, params.transition, params.coupling, B, fbc) == ref.loglik


def test_from_emissions_log_scale_and_options():
    structure, params, Y = _model(4)
    B, B_comp = chmm_lsim.compute_emissions(structure, params, Y)
    log_scale = np.full((structure.C, Y.shape[1]), 0.5)
    args = (structure, params.pi_0, params.transition, params.coupling, B, B_comp)
    base = chmm_lsim.forward_backward_from_emissions(*args, FbConstants(variant="sed"))
    scaled = chmm_lsim.forward_backward_from_emissions(*args, FbConstants(variant="sed"), log_scale=log_scale, method="reference")
    assert scaled.loglik == pytest.approx(base.loglik + log_scale.sum(), rel=1e-12)
    assert scaled.log_scale is log_scale
    with pytest.raises(ValueError, match="unexpected options"):
        chmm_lsim.forward_backward_from_emissions(*args, FbConstants(variant="poise"), method="reference")


def test_registered_variant_runs_from_emissions(dummy_variant):
    from chmm_lsim.models.lsim.inference.poise import poise_from_emissions
    from chmm_lsim.models.lsim.inference.registry import _FROM_EMISSIONS_REGISTRY, register_from_emissions

    structure, params, Y = _model(5)
    B, B_comp = chmm_lsim.compute_emissions(structure, params, Y)
    args = (structure, params.pi_0, params.transition, params.coupling, B, B_comp)
    with pytest.raises(ValueError, match="no registered from-emissions kernel"):
        chmm_lsim.forward_backward_from_emissions(*args, FbConstants(variant=DUMMY))

    @register_from_emissions(DUMMY)
    def dummy_from_emissions(structure, pi_0, transition, coupling, B, B_comp, fb_constants, *, log_scale=None, validate="params"):
        return poise_from_emissions(structure, pi_0, transition, coupling, B, B_comp,
                                    dataclasses.replace(fb_constants, variant="poise"), log_scale=log_scale, validate=validate)

    try:
        post = chmm_lsim.forward_backward_from_emissions(*args, FbConstants(variant=DUMMY))
        np.testing.assert_array_equal(post.gamma, forward_backward(structure, params, Y, FbConstants(variant="poise")).gamma)
    finally:
        del _FROM_EMISSIONS_REGISTRY[DUMMY]


def test_structural_m_step_is_the_structural_half_of_m_step():
    from chmm_lsim.models.lsim.em import structural_m_step

    structure, params, Y = _model(6)
    stats = chmm_lsim.finalize_expected_stats(chmm_lsim.e_step(structure, params, [Y]), sigma_diag=True)
    new = chmm_lsim.m_step(structure, stats, sigma_diag=True)
    pi_0, transition, coupling = structural_m_step(structure, stats.zeta, stats.gamma_p0, stats.n_transitions)
    np.testing.assert_array_equal(pi_0, new.pi_0)
    np.testing.assert_array_equal(transition, new.transition)
    np.testing.assert_array_equal(coupling, new.coupling)
    with pytest.raises(ValueError, match="params_in"):
        structural_m_step(structure, stats.zeta, stats.gamma_p0, 0)
    kept = structural_m_step(structure, stats.zeta, stats.gamma_p0, 0, params_in=params)
    np.testing.assert_array_equal(kept[1], params.transition)
