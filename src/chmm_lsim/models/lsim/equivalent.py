"""LSIM/CHMM to equivalent-HMM conversion (`common function/im_para_eqhmm.m`,
`chmm/chmm_cartesian_product.m`), plus an independent, textbook scaled HMM
forward-backward -- the exact-inference reference, deliberately *not*
built on `inference/sed.py`/`inference/fast.py` (those are exactly the
approximate recursions this reference has to be independent of).

**Cartesian-product joint-state indexing, one convention, used everywhere
below**: joint state `j` for per-channel states `(m_0, ..., m_{C-1})` is the
row-major ("C"-order) flattening of `(M_0, ..., M_{C-1})` -- the *last*
channel varies fastest. This is not a Python-side choice; it is what MATLAB's
`kron`-based construction already does (`im_para_eqhmm.m:20-34`,
`chmm_cartesian_product.m:61-80`): `kron(kron(A, B), C)` puts `C` (the last,
rightmost operand) fastest, and `index_matrix`'s `kron(1:N_c, ones(1,
prod(tail)))`-then-`repmat` construction produces bit-for-bit the same column
order `np.unravel_index(np.arange(J), dims)` does with NumPy's default
`order="C"`. `np.ravel_multi_index`/`np.unravel_index` are used directly
below rather than hand-rolled, precisely because they already implement this
convention.

**The LSIM transition law** (`im_para_eqhmm.m:40-54`): for target channel
`z`, the per-target conditional transition `cond[z]` has shape `(J, M_z)`
(joint previous state -> this channel's next state) and
`cond[z][j, :] = Σ_c θ[c,z]·A[c,z][m_c(j), :]`, `m_c(j)` being channel `c`'s
own state within joint state `j`. `simulate.py` applies this same formula
directly, without ever forming the full `(J, M_z)` table (see its own
docstring for why); this module forms the whole table because
`chmm_cartesian_product.m`'s joint transition needs every row of it at once.

**Guarded, separately, against two independent Cartesian explosions**:
`J = Π M_c` (`MAX_JOINT_STATES`) and `K_joint = Π K_c` (
`MAX_JOINT_MIXTURE_COMPONENTS`) -- a model with few states per channel but
many mixture components (or vice versa) can blow up one without the other,
so one guard cannot stand in for both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.joint_state import (
    MAX_ESTIMATED_BYTES,
    MAX_JOINT_MIXTURE_COMPONENTS,
    MAX_JOINT_STATES,
    EquivalentModelTooLargeError,
    _cartesian_combine,
    scaled_forward_backward,
)
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import check_structure_matches
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams


def _conditional_transitions(
    structure: LSIMStructure, params: LSIMParams, states_per_dim: NDArray[np.int64]
) -> list[NDArray[np.float64]]:
    """The LSIM's per-target conditional transitions `cond[z]` `(J, M_z)`
    (the transition law in the module docstring), evaluated for every joint
    previous state at once via fancy-indexing gather (im_para_eqhmm.m:40-54's
    explicit double loop over states/joint indices, vectorized).
    `states_per_dim` is `(C, J)`, each joint state's per-channel states."""
    C = structure.C
    J = states_per_dim.shape[1]
    cond_transitions: list[NDArray[np.float64]] = []
    for z in range(C):
        M_z = structure.channel_state_num[z]
        cond_z = np.zeros((J, M_z))
        for c in range(C):
            theta_cz = params.coupling[c, z]
            if theta_cz == 0.0:
                continue
            A_cz = params.transition[structure.state_slice(c), structure.state_slice(z)]  # (M_c, M_z)
            cond_z += theta_cz * A_cz[states_per_dim[c], :]  # (J, M_z)
        cond_transitions.append(cond_z)
    return cond_transitions

@dataclass
class EquivalentHMM:
    """A flat, single-chain HMM equivalent to an LSIM/CHMM model -- exact for
    the LSIM generative process (`chmm_cartesian_product.m`'s whole point):
    conditional on the *full* previous joint state, channels transition and
    emit independently, so the joint transition/emission factorize into a
    per-channel Kronecker/Cartesian product, with no approximation."""

    structure: LSIMStructure
    dims: tuple[int, ...]  # channel_state_num, i.e. (M_0, ..., M_{C-1})
    pi0: NDArray[np.float64]  # (J,)
    transition: NDArray[np.float64]  # (J, J)
    weights: NDArray[np.float64]  # (J, K_joint)
    mu: NDArray[np.float64]  # (D, J, K_joint)
    sigma: NDArray[np.float64]  # (D, J, K_joint) diag, or (D, D, J, K_joint) full
    sigma_diag: bool
    cond_transitions: list[NDArray[np.float64]]  # cond_transitions[z]: (J, M_z)

    @property
    def J(self) -> int:
        return self.pi0.shape[0]

    @property
    def K_joint(self) -> int:
        return self.weights.shape[1]

    def as_single_channel_params(self) -> tuple[LSIMStructure, LSIMParams]:
        """Wraps the joint GMM as a `C=1` `LSIMStructure`/`LSIMParams`, so the
        already-verified P2 `compute_emissions` can evaluate the joint
        emission likelihood `(J, T)` directly -- reusing a separately-gated
        numerical primitive, not "the LSIM inference kernel" this module's
        forward-backward must stay independent of (see module docstring)."""
        joint_structure = LSIMStructure(channel_state_num=(self.J,), channel_obs_dim=(self.structure.D,), num_gmm_component=(self.K_joint,))
        joint_gmm = ChannelGMM(weights=self.weights, mu=self.mu, sigma=self.sigma, diag=self.sigma_diag)
        joint_params = LSIMParams(
            structure=joint_structure,
            pi_0=self.pi0,
            transition=self.transition,
            coupling=np.ones((1, 1)),
            emissions=[joint_gmm],
        )
        return joint_structure, joint_params

def build_equivalent_hmm(structure: LSIMStructure, params: LSIMParams, *, validate: bool = True) -> EquivalentHMM:
    """`im_para_eqhmm.m` + `chmm_cartesian_product.m`, from `LSIMParams`
    directly rather than the cell-array CHMM API. `validate=True` (the
    default) checks `structure == params.structure` (always -- see
    `check_structure_matches`) and runs `params.validate()` -- deferred until
    *after* the size guards below, so a model that's about to be rejected
    for being too large doesn't first pay for validating multi-gigabyte
    arrays it's never going to use."""
    check_structure_matches(structure, params)

    C = structure.C
    dims = structure.channel_state_num
    # math.prod, not np.prod: np.prod computes in a fixed-width integer dtype
    # and silently *overflows* for a large enough product (e.g. 64 channels of
    # 2 states each is 2**64, which wraps a signed int64 to 0 and would sail
    # straight past the J>MAX_JOINT_STATES check below). math.prod uses
    # Python's arbitrary-precision ints, so it can never silently wrap.
    J = math.prod(dims)
    if J > MAX_JOINT_STATES:
        raise EquivalentModelTooLargeError(
            f"equivalent HMM would have J=Π channel_state_num={J} joint states, "
            f"exceeding MAX_JOINT_STATES={MAX_JOINT_STATES} -- this is a Layer-B "
            f"verification model, not a general inference path"
        )
    k_dims = structure.num_gmm_component
    K_joint = math.prod(k_dims)
    if K_joint > MAX_JOINT_MIXTURE_COMPONENTS:
        raise EquivalentModelTooLargeError(
            f"equivalent HMM would have K_joint=Π num_gmm_component={K_joint} joint mixture "
            f"components, exceeding MAX_JOINT_MIXTURE_COMPONENTS={MAX_JOINT_MIXTURE_COMPONENTS}"
        )

    # Combined guard: J and K_joint can each individually clear the ceilings
    # above and still make mu/sigma/weights/transition multi-gigabyte together
    # (e.g. J=K_joint=4096 is J*K_joint=16.7M -- fine as a *count*, not fine as
    # an array of D*that float64s). Estimated from shapes alone, before any of
    # these arrays is actually allocated.
    D = structure.D
    sigma_factor = D if not params.sigma_diag else 1  # sigma is (D,J,K) diag, (D,D,J,K) full
    mu_and_sigma_elems = D * J * K_joint * (1 + sigma_factor)
    estimated_bytes = 8 * (J * J + J + mu_and_sigma_elems + J * K_joint)
    if estimated_bytes > MAX_ESTIMATED_BYTES:
        raise EquivalentModelTooLargeError(
            f"equivalent HMM's mu/sigma/weights/transition would need an estimated "
            f"{estimated_bytes / 1024**2:.0f} MiB (J={J}, K_joint={K_joint}, D={D}), exceeding "
            f"MAX_ESTIMATED_BYTES={MAX_ESTIMATED_BYTES / 1024**2:.0f} MiB -- J<={MAX_JOINT_STATES} and "
            f"K_joint<={MAX_JOINT_MIXTURE_COMPONENTS} individually are not sufficient guards on their own"
        )

    if validate:
        params.validate()

    states_per_dim = np.array(np.unravel_index(np.arange(J), dims))  # (C, J)
    components_per_dim = np.array(np.unravel_index(np.arange(K_joint), k_dims))  # (C, K_joint)

    # joint pi_0: Cartesian product of per-channel pi_0 vectors (im_para_eqhmm.m
    # delegates this to chmm_cartesian_product.m:17-22's kron chain).
    per_channel_pi0 = [params.pi_0[structure.state_slice(c)][None, :] for c in range(C)]
    pi0 = _cartesian_combine(per_channel_pi0)[0]

    cond_transitions = _conditional_transitions(structure, params, states_per_dim)

    # joint transition (J, J): row j's next-joint-state distribution is the
    # Cartesian product, across channels, of that channel's own (J,M_z) row j
    # (chmm_cartesian_product.m:45-56's per-row kron, batched over every row
    # at once via _cartesian_combine).
    transition = _cartesian_combine(cond_transitions)

    # joint GMM emissions: gather each channel's own (state, component) slice
    # at every (joint state, joint component) pair, then concatenate means
    # (block layout, same as every other packed array) and place covariances
    # block-diagonally (full) or stacked (diag) -- channels' emission noise is
    # independent given the joint state, so there is no cross-channel term.
    weights_per_channel = []
    D = structure.D
    mu = np.zeros((D, J, K_joint))
    sigma = np.zeros((D, J, K_joint)) if params.sigma_diag else np.zeros((D, D, J, K_joint))
    for c in range(C):
        gmm = params.emissions[c]
        state_idx = states_per_dim[c][:, None]  # (J, 1)
        comp_idx = components_per_dim[c][None, :]  # (1, K_joint)
        weights_per_channel.append(gmm.weights[state_idx, comp_idx])  # (J, K_joint)
        mu[structure.dim_slice(c), :, :] = gmm.mu[:, state_idx, comp_idx]  # (d_c, J, K_joint)
        if params.sigma_diag:
            sigma[structure.dim_slice(c), :, :] = gmm.sigma[:, state_idx, comp_idx]
        else:
            sigma[structure.dim_slice(c), structure.dim_slice(c), :, :] = gmm.sigma[:, :, state_idx, comp_idx]

    # each weights_per_channel[c] is already (J, K_joint) -- gathered with comp_idx
    # spanning the *joint* component range, so channel c's own local component at
    # joint index k is already picked out at column k. Combining across channels is
    # therefore a plain elementwise product (each factor already lives on the same
    # (J, K_joint) grid), not another Cartesian expansion -- unlike pi0/transition
    # above, where each channel contributes its *own* independent axis and
    # _cartesian_combine's expansion is exactly what's needed.
    weights = np.prod(np.stack(weights_per_channel, axis=0), axis=0)

    return EquivalentHMM(
        structure=structure,
        dims=dims,
        pi0=pi0,
        transition=transition,
        weights=weights,
        mu=mu,
        sigma=sigma,
        sigma_diag=params.sigma_diag,
        cond_transitions=cond_transitions,
    )

def exact_posterior(equiv: EquivalentHMM, Y: NDArray[np.float64], *, validate: bool = True) -> tuple[NDArray[np.float64], float]:
    """Convenience wrapper: joint emission likelihoods via the shared P2
    `compute_emissions` primitive (see `EquivalentHMM.as_single_channel_params`
    for why that reuse is fine here), then the independent
    `scaled_forward_backward` above. Returns `(gamma, loglik)`, `gamma`
    `(J, T)`. `validate=True` (the default) runs `validate_observations`
    against `equiv.structure` before touching `Y`."""
    from chmm_lsim.core.validate import validate_observations
    from chmm_lsim.emissions import compute_emissions

    if validate:
        validate_observations(equiv.structure, Y)

    joint_structure, joint_params = equiv.as_single_channel_params()
    B, _ = compute_emissions(joint_structure, joint_params, Y)
    return scaled_forward_backward(equiv.pi0, equiv.transition, B)
