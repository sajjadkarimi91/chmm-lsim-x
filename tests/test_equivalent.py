"""P6 gate: `equivalent.py` (`common function/im_para_eqhmm.m` +
`chmm/chmm_cartesian_product.m`, plus an independent textbook scaled
forward-backward).

Exact deterministic tests throughout: Cartesian index ordering on
hand-computable cases, joint initial/transition probabilities against a
hand-derived formula, row-stochasticity of every conditional/equivalent
transition, equivalent emissions for both covariance kinds, and the exact
forward-backward against brute-force path enumeration on tiny models --
`equivalent.py`'s whole point is that none of this is approximate.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.lsim.equivalent import (
    MAX_JOINT_MIXTURE_COMPONENTS,
    MAX_JOINT_STATES,
    build_equivalent_hmm,
    exact_posterior,
)
from chmm_lsim.core.joint_state import (
    EquivalentModelTooLargeError,
    joint_state_from_index,
    joint_state_index,
    scaled_forward_backward,
)
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError


# --------------------------------------------------------------------------
# Cartesian index ordering: hand-computable, the last channel fastest.
# --------------------------------------------------------------------------

def test_joint_state_index_last_channel_fastest():
    dims = (2, 3)
    for m0 in range(2):
        for m1 in range(3):
            j = joint_state_index(np.array([m0, m1]), dims)
            assert j == m0 * 3 + m1
            assert joint_state_from_index(j, dims) == (m0, m1)


def test_joint_state_index_three_channels():
    dims = (2, 2, 3)
    # channel 2 (last) fastest, then channel 1, then channel 0 slowest
    assert joint_state_index(np.array([0, 0, 0]), dims) == 0
    assert joint_state_index(np.array([0, 0, 1]), dims) == 1
    assert joint_state_index(np.array([0, 1, 0]), dims) == 3
    assert joint_state_index(np.array([1, 0, 0]), dims) == 6
    assert joint_state_index(np.array([1, 1, 2]), dims) == 6 + 3 + 2


def test_joint_state_index_and_from_index_are_inverses_over_every_j():
    dims = (2, 3, 2)
    J = int(np.prod(dims))
    for j in range(J):
        states = joint_state_from_index(j, dims)
        assert joint_state_index(np.array(states), dims) == j


# --------------------------------------------------------------------------
# A small, hand-built C=2 model reused across most tests below.
# --------------------------------------------------------------------------

def _small_model(sigma_diag=True):
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    pi0 = np.array([0.6, 0.4, 0.3, 0.7])
    transition = np.zeros((4, 4))
    transition[0:2, 0:2] = [[0.9, 0.1], [0.2, 0.8]]
    transition[0:2, 2:4] = [[0.5, 0.5], [0.5, 0.5]]
    transition[2:4, 0:2] = [[0.3, 0.7], [0.6, 0.4]]
    transition[2:4, 2:4] = [[0.7, 0.3], [0.4, 0.6]]
    coupling = np.array([[0.7, 0.3], [0.3, 0.7]])
    if sigma_diag:
        gmm0 = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([0.0, 5.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
        gmm1 = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([-3.0, 3.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    else:
        structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(2, 2), num_gmm_component=(1, 1))
        mu0 = np.stack([np.zeros(2), np.full(2, 5.0)], axis=1)[:, :, None]
        mu1 = np.stack([np.full(2, -3.0), np.full(2, 3.0)], axis=1)[:, :, None]
        sig0 = np.stack([np.eye(2), np.array([[1.5, 0.3], [0.3, 1.2]])], axis=-1)[:, :, :, None]
        sig1 = np.stack([np.eye(2) * 0.8, np.eye(2) * 1.1], axis=-1)[:, :, :, None]
        gmm0 = ChannelGMM(weights=np.ones((2, 1)), mu=mu0, sigma=sig0, diag=False)
        gmm1 = ChannelGMM(weights=np.ones((2, 1)), mu=mu1, sigma=sig1, diag=False)
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])
    params.validate()
    return structure, params


def test_joint_pi0_hand_derived():
    structure, params = _small_model()
    equiv = build_equivalent_hmm(structure, params)
    pi0_0 = params.pi_0[0:2]
    pi0_1 = params.pi_0[2:4]
    expected = np.array([pi0_0[m0] * pi0_1[m1] for m0 in range(2) for m1 in range(2)])
    np.testing.assert_allclose(equiv.pi0, expected)
    np.testing.assert_allclose(equiv.pi0.sum(), 1.0)


def test_joint_transition_hand_derived_entry():
    structure, params = _small_model()
    equiv = build_equivalent_hmm(structure, params)
    theta = params.coupling
    A = params.transition
    # joint row j=(0,0) [index 0], joint col (1,0) [index 2]:
    # p(m0'=1|prev=(0,0)) = theta[0,0]*A00[0,1] + theta[1,0]*A10[0,1]
    p_m0_1 = theta[0, 0] * A[0, 1] + theta[1, 0] * A[2, 1]
    p_m1_0 = theta[0, 1] * A[0, 2] + theta[1, 1] * A[2, 2]
    np.testing.assert_allclose(equiv.transition[0, 2], p_m0_1 * p_m1_0)


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_row_stochasticity_of_conditional_and_joint_transitions(sigma_diag):
    structure, params = _small_model(sigma_diag)
    equiv = build_equivalent_hmm(structure, params)
    for cond in equiv.cond_transitions:
        np.testing.assert_allclose(cond.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(equiv.transition.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(equiv.pi0.sum(), 1.0, atol=1e-10)


def test_decoupled_model_gives_kron_of_self_blocks():
    structure, params = _small_model()
    params = params.copy()
    params.coupling = np.eye(2)  # fully decoupled: theta[c,z]=0 unless c==z
    equiv = build_equivalent_hmm(structure, params)
    expected = np.kron(params.transition[0:2, 0:2], params.transition[2:4, 2:4])
    np.testing.assert_allclose(equiv.transition, expected)


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_equivalent_emissions_match_gathered_per_channel_values(sigma_diag):
    structure, params = _small_model(sigma_diag)
    equiv = build_equivalent_hmm(structure, params)
    assert equiv.J == 4
    assert equiv.K_joint == 1
    for j in range(equiv.J):
        m0, m1 = joint_state_from_index(j, equiv.dims)
        expected_mu = np.concatenate([params.emissions[0].mu[:, m0, 0], params.emissions[1].mu[:, m1, 0]])
        np.testing.assert_allclose(equiv.mu[:, j, 0], expected_mu)
        expected_weight = params.emissions[0].weights[m0, 0] * params.emissions[1].weights[m1, 0]
        np.testing.assert_allclose(equiv.weights[j, 0], expected_weight)
        if sigma_diag:
            expected_sigma = np.concatenate([params.emissions[0].sigma[:, m0, 0], params.emissions[1].sigma[:, m1, 0]])
            np.testing.assert_allclose(equiv.sigma[:, j, 0], expected_sigma)
        else:
            d0 = params.emissions[0].d_c
            np.testing.assert_allclose(equiv.sigma[:d0, :d0, j, 0], params.emissions[0].sigma[:, :, m0, 0])
            np.testing.assert_allclose(equiv.sigma[d0:, d0:, j, 0], params.emissions[1].sigma[:, :, m1, 0])
            np.testing.assert_allclose(equiv.sigma[:d0, d0:, j, 0], 0.0)  # block-diagonal: no cross term
            np.testing.assert_allclose(equiv.sigma[d0:, :d0, j, 0], 0.0)


def test_heterogeneous_K_c_joint_mixture():
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(2, 3))
    pi0 = np.array([0.5, 0.5, 0.5, 0.5])
    transition = np.zeros((4, 4))
    transition[0:2, 0:2] = 0.5
    transition[0:2, 2:4] = 0.5
    transition[2:4, 0:2] = 0.5
    transition[2:4, 2:4] = 0.5
    coupling = np.eye(2)
    rng = np.random.default_rng(0)
    gmm0 = ChannelGMM(weights=rng.dirichlet(np.ones(2), 2), mu=rng.normal(size=(1, 2, 2)), sigma=np.ones((1, 2, 2)), diag=True)
    gmm1 = ChannelGMM(weights=rng.dirichlet(np.ones(3), 2), mu=rng.normal(size=(1, 2, 3)), sigma=np.ones((1, 2, 3)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm0, gmm1])
    params.validate()

    equiv = build_equivalent_hmm(structure, params)
    assert equiv.K_joint == 6
    np.testing.assert_allclose(equiv.weights.sum(axis=1), 1.0, atol=1e-10)  # a real GMM per joint state
    # spot check one (joint state, joint component)
    j, k = 1, 4
    m0, m1 = joint_state_from_index(j, equiv.dims)
    k0, k1 = joint_state_from_index(k, (2, 3))
    expected = gmm0.weights[m0, k0] * gmm1.weights[m1, k1]
    np.testing.assert_allclose(equiv.weights[j, k], expected)


# --------------------------------------------------------------------------
# Guards: two independent Cartesian explosions.
# --------------------------------------------------------------------------

def test_max_joint_states_guard():
    huge_M = MAX_JOINT_STATES  # a single channel already at the ceiling+
    structure = LSIMStructure(channel_state_num=(huge_M + 1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi0 = np.full(huge_M + 1, 1.0 / (huge_M + 1))
    transition = np.full((huge_M + 1, huge_M + 1), 1.0 / (huge_M + 1))
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((huge_M + 1, 1)), mu=np.zeros((1, huge_M + 1, 1)), sigma=np.ones((1, huge_M + 1, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm])
    with pytest.raises(EquivalentModelTooLargeError, match="MAX_JOINT_STATES"):
        build_equivalent_hmm(structure, params)


def test_max_joint_mixture_components_guard():
    huge_K = MAX_JOINT_MIXTURE_COMPONENTS
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(huge_K + 1,))
    pi0 = np.array([1.0])
    transition = np.array([[1.0]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(
        weights=np.full((1, huge_K + 1), 1.0 / (huge_K + 1)), mu=np.zeros((1, 1, huge_K + 1)), sigma=np.ones((1, 1, huge_K + 1)), diag=True
    )
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=[gmm])
    with pytest.raises(EquivalentModelTooLargeError, match="MAX_JOINT_MIXTURE_COMPONENTS"):
        build_equivalent_hmm(structure, params)


def test_max_joint_states_guard_uses_math_prod_not_np_prod():
    """64 channels of 2 states each is J=2**64 -- np.prod on a fixed-width
    int64 array silently overflows/wraps this to 0, which would sail straight
    past `J > MAX_JOINT_STATES` and attempt to build a 0-or-tiny "equivalent
    HMM" instead of rejecting it. `math.prod` (arbitrary-precision) must not
    have this problem; the actual `LSIMParams` here is fully valid and a
    realistic size (S=128 states total), so this exercises the real
    `build_equivalent_hmm` code path, not a synthetic J value."""
    C = 64
    structure = LSIMStructure(channel_state_num=tuple([2] * C), channel_obs_dim=tuple([1] * C), num_gmm_component=tuple([1] * C))
    pi0 = np.tile([0.5, 0.5], C)
    transition = np.zeros((2 * C, 2 * C))
    for c in range(C):
        for z in range(C):
            transition[2 * c : 2 * c + 2, 2 * z : 2 * z + 2] = [[0.5, 0.5], [0.5, 0.5]]
    coupling = np.full((C, C), 1.0 / C)
    emissions = [ChannelGMM(weights=np.ones((2, 1)), mu=np.zeros((1, 2, 1)), sigma=np.ones((1, 2, 1)), diag=True) for _ in range(C)]
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()

    with pytest.raises(EquivalentModelTooLargeError, match=r"J=Π channel_state_num=18446744073709551616"):
        build_equivalent_hmm(structure, params)


def test_combined_memory_guard_catches_j_and_k_joint_both_individually_legal():
    """C=6, M_c=K_c=4: J=4**6=4096 and K_joint=4**6=4096 both individually
    clear MAX_JOINT_STATES/MAX_JOINT_MIXTURE_COMPONENTS -- but mu/sigma/
    weights/transition together would need an estimated ~1.75 GiB (measured
    below), which the combined guard must catch on its own, before any of
    those arrays is allocated."""
    C, M, K = 6, 4, 4
    structure = LSIMStructure(channel_state_num=tuple([M] * C), channel_obs_dim=tuple([1] * C), num_gmm_component=tuple([K] * C))
    assert M**C <= MAX_JOINT_STATES
    assert K**C <= MAX_JOINT_MIXTURE_COMPONENTS

    pi0 = np.tile(np.full(M, 1.0 / M), C)
    transition = np.zeros((M * C, M * C))
    for c in range(C):
        for z in range(C):
            transition[M * c : M * c + M, M * z : M * z + M] = np.full((M, M), 1.0 / M)
    coupling = np.full((C, C), 1.0 / C)
    emissions = [ChannelGMM(weights=np.full((M, K), 1.0 / K), mu=np.zeros((1, M, K)), sigma=np.ones((1, M, K)), diag=True) for _ in range(C)]
    params = LSIMParams(structure=structure, pi_0=pi0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()

    with pytest.raises(EquivalentModelTooLargeError, match="MAX_ESTIMATED_BYTES"):
        build_equivalent_hmm(structure, params)


def test_structure_mismatch_rejected():
    structure, params = _small_model()
    other_structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    with pytest.raises(ValidationError, match="structure"):
        build_equivalent_hmm(other_structure, params)


# --------------------------------------------------------------------------
# Independent scaled forward-backward vs brute-force enumeration.
# --------------------------------------------------------------------------

def test_exact_posterior_matches_brute_force_enumeration():
    structure, params = _small_model()
    equiv = build_equivalent_hmm(structure, params)
    rng = np.random.default_rng(0)
    T = 6
    Y = rng.normal(size=(2, T))

    gamma, loglik = exact_posterior(equiv, Y)
    np.testing.assert_allclose(gamma.sum(axis=0), 1.0, atol=1e-10)

    joint_structure, joint_params = equiv.as_single_channel_params()
    B, _ = compute_emissions(joint_structure, joint_params, Y)

    paths = list(itertools.product(range(equiv.J), repeat=T))
    probs = np.zeros(len(paths))
    for pi, path in enumerate(paths):
        p = equiv.pi0[path[0]] * B[path[0], 0]
        for t in range(1, T):
            p *= equiv.transition[path[t - 1], path[t]] * B[path[t], t]
        probs[pi] = p
    Z = probs.sum()

    np.testing.assert_allclose(loglik, np.log(Z), rtol=1e-10)

    gamma_bf = np.zeros((equiv.J, T))
    for pi, path in enumerate(paths):
        for t in range(T):
            gamma_bf[path[t], t] += probs[pi]
    gamma_bf /= Z
    np.testing.assert_allclose(gamma, gamma_bf, atol=1e-10)


def test_scaled_forward_backward_is_generic_not_tied_to_equivalent_hmm():
    """A plain 3-state chain, no LSIM/CHMM structure involved at all --
    confirms scaled_forward_backward works as a standalone textbook routine,
    not only ever called through build_equivalent_hmm's output."""
    pi0 = np.array([0.5, 0.3, 0.2])
    transition = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    rng = np.random.default_rng(1)
    T = 5
    B = rng.uniform(0.1, 1.0, size=(3, T))

    gamma, loglik = scaled_forward_backward(pi0, transition, B)
    np.testing.assert_allclose(gamma.sum(axis=0), 1.0, atol=1e-10)

    paths = list(itertools.product(range(3), repeat=T))
    probs = np.array(
        [pi0[p[0]] * B[p[0], 0] * np.prod([transition[p[t - 1], p[t]] * B[p[t], t] for t in range(1, T)]) for p in paths]
    )
    np.testing.assert_allclose(loglik, np.log(probs.sum()), rtol=1e-10)


# --------------------------------------------------------------------------
# scaled_forward_backward: malformed shapes and zero/non-finite scaling mass
# rejected clearly, never returned as NaN.
# --------------------------------------------------------------------------

def test_scaled_forward_backward_rejects_wrong_pi0_shape():
    B = np.ones((3, 4))
    with pytest.raises(ValidationError, match="pi0"):
        scaled_forward_backward(np.array([0.5, 0.5]), np.eye(3), B)


def test_scaled_forward_backward_rejects_wrong_transition_shape():
    B = np.ones((3, 4))
    with pytest.raises(ValidationError, match="transition"):
        scaled_forward_backward(np.full(3, 1 / 3), np.eye(2), B)


def test_scaled_forward_backward_rejects_zero_scaling_mass_not_nan():
    """Every state's emission likelihood is 0 at t=1 -- no state can explain
    that observation at all, so the forward pass's own normalizing mass
    c[1]=alpha_hat[:,1].sum() is exactly 0. Dividing by it would silently
    produce NaN; this must raise instead."""
    pi0 = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.array([[1.0, 0.0], [1.0, 0.0]])  # column 1 is all-zero
    with pytest.raises(ValidationError, match="scaling mass"):
        scaled_forward_backward(pi0, transition, B)


def test_scaled_forward_backward_rejects_zero_scaling_mass_at_t0():
    pi0 = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.array([[0.0, 1.0], [0.0, 1.0]])  # column 0 is all-zero
    with pytest.raises(ValidationError, match="scaling mass"):
        scaled_forward_backward(pi0, transition, B)


def test_scaled_forward_backward_rejects_1d_B_not_raw_valueerror():
    """`N, T = B.shape` on a 1-D array is a raw `ValueError` ("not enough
    values to unpack") if reached -- must be caught before that unpacking."""
    pi0 = np.array([0.5, 0.5])
    transition = np.eye(2)
    with pytest.raises(ValidationError, match="2-D"):
        scaled_forward_backward(pi0, transition, np.ones(4))


def test_scaled_forward_backward_rejects_zero_states_not_raw_error():
    with pytest.raises(ValidationError, match="N=0"):
        scaled_forward_backward(np.zeros(0), np.zeros((0, 0)), np.zeros((0, 4)))


def test_scaled_forward_backward_rejects_zero_timesteps_not_raw_indexerror():
    """`B.shape=(N,0)` unpacks fine but `B[:, 0]` is a raw `IndexError` a few
    lines later -- must be caught before that indexing."""
    pi0 = np.array([0.5, 0.5])
    transition = np.eye(2)
    with pytest.raises(ValidationError, match="T=0"):
        scaled_forward_backward(pi0, transition, np.zeros((2, 0)))


@pytest.mark.parametrize("bad_value,match", [(np.nan, "finite"), (np.inf, "finite"), (-1.0, "negative")])
def test_scaled_forward_backward_rejects_bad_B_entry(bad_value, match):
    pi0 = np.array([0.5, 0.5])
    transition = np.eye(2)
    B = np.ones((2, 3))
    B[0, 1] = bad_value
    with pytest.raises(ValidationError, match=match):
        scaled_forward_backward(pi0, transition, B)


@pytest.mark.parametrize("bad_value,match", [(np.nan, "finite"), (np.inf, "finite"), (-0.1, "negative")])
def test_scaled_forward_backward_rejects_bad_pi0_entry(bad_value, match):
    pi0 = np.array([0.5 + 0.1, 0.4 - 0.1])
    pi0[0] = bad_value
    transition = np.eye(2)
    B = np.ones((2, 3))
    with pytest.raises(ValidationError, match=match):
        scaled_forward_backward(pi0, transition, B)


@pytest.mark.parametrize("bad_value,match", [(np.nan, "finite"), (np.inf, "finite"), (-0.1, "negative")])
def test_scaled_forward_backward_rejects_bad_transition_entry(bad_value, match):
    pi0 = np.array([0.5, 0.5])
    transition = np.eye(2)
    transition[0, 1] = bad_value
    B = np.ones((2, 3))
    with pytest.raises(ValidationError, match=match):
        scaled_forward_backward(pi0, transition, B)


# --------------------------------------------------------------------------
# exact_posterior: validate_observations against equiv.structure.
# --------------------------------------------------------------------------

def test_exact_posterior_validates_observations():
    structure, params = _small_model()
    equiv = build_equivalent_hmm(structure, params)
    wrong_D = np.zeros((3, 5))  # equiv.structure.D == 2
    with pytest.raises(ValidationError, match="structure.D"):
        exact_posterior(equiv, wrong_D)


def test_exact_posterior_validate_false_skips_the_check():
    structure, params = _small_model()
    equiv = build_equivalent_hmm(structure, params)
    Y = np.random.default_rng(0).normal(size=(2, 4))
    gamma, loglik = exact_posterior(equiv, Y, validate=False)
    assert gamma.shape == (equiv.J, 4)
