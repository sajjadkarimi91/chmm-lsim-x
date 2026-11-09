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

from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError, check_structure_matches

MAX_JOINT_STATES = 4096
MAX_JOINT_MIXTURE_COMPONENTS = 4096
MAX_ESTIMATED_BYTES = 512 * 1024 * 1024  # 512 MiB, before any of mu/sigma/weights/transition is allocated


class EquivalentModelTooLargeError(ValueError):
    """Raised by `build_equivalent_hmm` when `J = Π channel_state_num` or
    `K_joint = Π num_gmm_component` would exceed its own guard, or when the
    two together would still make `mu`/`sigma`/`weights`/`transition`
    multi-gigabyte (`MAX_ESTIMATED_BYTES`) even with both individually under
    their ceiling -- the equivalent HMM is only ever meant for small Layer-B
    verification models, not as a general inference path (that is exactly
    what P3a/P3b's approximate kernels are for)."""


def joint_state_index(states: NDArray[np.int64], dims: tuple[int, ...]) -> int:
    """The joint index of per-channel (0-based) states `states`, under the
    row-major/last-channel-fastest convention (see module docstring)."""
    return int(np.ravel_multi_index(tuple(int(s) for s in states), dims))


def joint_state_from_index(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of `joint_state_index`: the per-channel (0-based) states for
    one joint index."""
    return tuple(int(x) for x in np.unravel_index(index, dims))


def _cartesian_combine(arrays: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """`arrays[c]` is `(N, K_c)`; returns `(N, Π K_c)` where
    `result[n, j] = Π_c arrays[c][n, m_c(j)]`, `j`'s per-`c` component
    `m_c(j)` under the same row-major/last-fastest convention as everywhere
    else in this module (`np.ravel_multi_index`'s default order). Used both
    for a genuine batch (`N` joint states, combining per-channel *mixture*
    weights already gathered at each state) and, with `N=1`, for the single
    joint `pi_0` (a pure Cartesian product of per-channel vectors)."""
    result = arrays[0]
    for arr in arrays[1:]:
        result = (result[:, :, None] * arr[:, None, :]).reshape(result.shape[0], -1)
    return result


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


def scaled_forward_backward(
    pi0: NDArray[np.float64], transition: NDArray[np.float64], B: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Independent, textbook scaled HMM forward-backward (Rabiner's
    `alpha_hat`/`beta_hat` scaling) on a *flat* `N`-state chain -- no channel
    blocks, no coupling, no `inference/sed.py`/`inference/fast.py` code or
    helpers, by design (this is the Layer-B reference those must be checked
    against, so it must not share their recursion). `pi0` is `(N,)`,
    `transition` is `(N,N)` row-stochastic, `B` is `(N,T)` emission
    likelihoods (not necessarily normalized). Returns `(gamma, loglik)`,
    `gamma` `(N,T)` the smoothed posterior.

    Raises `ValidationError` -- never returns NaN, and never escapes as a
    raw `ValueError`/`IndexError` from unpacking or indexing a malformed
    array -- if: `B` isn't 2-D; `B` has `N=0` or `T=0` states/timesteps;
    `B`/`pi0`/`transition` contain a negative or non-finite entry; `pi0`/
    `transition` don't have the shapes `B` implies; or the scaling mass
    `c[t]` (`alpha_hat[:,t]`'s own sum, before it's normalized away) is ever
    zero or non-finite: that happens exactly when *no* state can explain the
    observations up to `t` at all (e.g. every state's emission density has
    underflowed to `0`) -- `alpha_hat[:,t] /= c[t]` would silently divide `0`
    by `0` into NaN otherwise, and every later `t` would inherit it silently.
    All of this is checked *before* `B.shape` is even unpacked, since a 1-D
    `B` can't be unpacked into `(N, T)` at all, and an `(N, 0)` `B` unpacks
    fine but then fails on `B[:, 0]` with a raw `IndexError` further down."""
    if B.ndim != 2:
        raise ValidationError(f"B.shape={B.shape}, expected 2-D (N, T)")
    N, T = B.shape
    if N < 1:
        raise ValidationError(f"B.shape={B.shape}: N=0, expected at least 1 state")
    if T < 1:
        raise ValidationError(f"B.shape={B.shape}: T=0, expected at least 1 timestep")

    def _check_probability_array(name: str, arr: NDArray[np.float64]) -> None:
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"{name} contains a non-finite (NaN/Inf) entry")
        if np.any(arr < 0):
            raise ValidationError(f"{name} contains a negative entry")

    _check_probability_array("B", B)

    if pi0.shape != (N,):
        raise ValidationError(f"pi0.shape={pi0.shape}, expected (N,)=({N},) to match B.shape={B.shape}")
    if transition.shape != (N, N):
        raise ValidationError(f"transition.shape={transition.shape}, expected (N,N)=({N},{N}) to match B.shape={B.shape}")
    _check_probability_array("pi0", pi0)
    _check_probability_array("transition", transition)

    alpha_hat = np.zeros((N, T))
    c = np.zeros(T)

    def _check_scaling_mass(t: int) -> None:
        if not np.isfinite(c[t]) or c[t] <= 0:
            raise ValidationError(
                f"scaled_forward_backward: scaling mass c[{t}]={c[t]} is zero or non-finite -- no state "
                f"can explain the observations up to t={t} (e.g. every emission density has underflowed "
                f"to 0); refusing to divide by it rather than silently producing NaN"
            )

    alpha_hat[:, 0] = pi0 * B[:, 0]
    c[0] = alpha_hat[:, 0].sum()
    _check_scaling_mass(0)
    alpha_hat[:, 0] /= c[0]
    for t in range(1, T):
        alpha_hat[:, t] = (transition.T @ alpha_hat[:, t - 1]) * B[:, t]
        c[t] = alpha_hat[:, t].sum()
        _check_scaling_mass(t)
        alpha_hat[:, t] /= c[t]

    beta_hat = np.zeros((N, T))
    beta_hat[:, T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta_hat[:, t] = transition @ (B[:, t + 1] * beta_hat[:, t + 1]) / c[t + 1]

    gamma = alpha_hat * beta_hat
    loglik = float(np.sum(np.log(c)))
    return gamma, loglik


def exact_posterior(equiv: EquivalentHMM, Y: NDArray[np.float64], *, validate: bool = True) -> tuple[NDArray[np.float64], float]:
    """Convenience wrapper: joint emission likelihoods via the shared P2
    `compute_emissions` primitive (see `EquivalentHMM.as_single_channel_params`
    for why that reuse is fine here), then the independent
    `scaled_forward_backward` above. Returns `(gamma, loglik)`, `gamma`
    `(J, T)`. `validate=True` (the default) runs `validate_observations`
    against `equiv.structure` before touching `Y`."""
    from chmm_lsim.emissions import compute_emissions
    from chmm_lsim.validate import validate_observations

    if validate:
        validate_observations(equiv.structure, Y)

    joint_structure, joint_params = equiv.as_single_channel_params()
    B, _ = compute_emissions(joint_structure, joint_params, Y)
    return scaled_forward_backward(equiv.pi0, equiv.transition, B)
