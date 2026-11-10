"""Full coupled hidden Markov models (CHMMs) with exact joint-state inference.

A CHMM conditions every channel's next state on the *joint* previous state of
all channels: channel `z` has a transition table of shape `(J, M_z)`, with
`J = prod(M)` and row `j` the distribution `P(x^z_t | joint state j at t-1)`.
Given the previous joint state, channels transition and emit independently,
so the model is exactly a single-chain HMM over the `J` joint states. The
joint initial distribution is the product of the per-channel `pi_0` blocks,
and emissions are per-channel Gaussian mixtures, exactly as in an LSIM
(`ChannelGMM`, evaluated by `compute_emissions`). An LSIM is the special case
whose tables are coupling-weighted mixtures of pairwise transitions
(`lsim_to_chmm`).

Inference here is exact and costs `O(J^2 T)`, so it is meant for small
models; the parameter count also grows with `J`. The joint-state order is the
one used throughout the package (`equivalent.joint_state_index`): row-major,
the last channel varies fastest.

This module mirrors the MATLAB functions `em_chmm.m`, `forward_backward_chmm.m`,
`viterbi_chmm.m` (4-argument form), `chmm_joint_model.m`,
`chmm_exact_forward_backward.m` and `generate_chmm_time_series.m`. `em_chmm`
shares the LSIM EM machinery: the emission statistics and M-step, the
initial-distribution update, the iteration loop with its convergence check
and best-iterate selection, and the `EmResult` it returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, check_structure_matches, validate_observations
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.models.chmm.params import CHMMParams, _joint_states


# ---------------------------------------------------------------------------
# exact joint-state inference
# ---------------------------------------------------------------------------


@dataclass
class CHMMJointModel:
    """A CHMM written as a single-chain HMM over its `J` joint states, for one
    observation sequence (`chmm_joint_model`).

    Attributes
    ----------
    pi : ndarray, shape (J,)
        Joint initial distribution.
    transition : ndarray, shape (J, J)
        `transition[j, j'] = prod_z params.transition[z][j, m_z(j')]`.
    log_B : ndarray, shape (J, T)
        Log emission density of every joint state: the sum over channels of
        the log density of that channel's state.
    states : ndarray of int, shape (C, J)
        Zero-based per-channel states of every joint state.
    B, B_comp : ndarray, shapes (S, T) and (S, max_K, T)
        Per-channel state and mixture-component densities from
        `compute_emissions`.
    """

    pi: NDArray[np.float64]
    transition: NDArray[np.float64]
    log_B: NDArray[np.float64]
    states: NDArray[np.int64]
    B: NDArray[np.float64]
    B_comp: NDArray[np.float64]

def _joint_chain(params: CHMMParams, states: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Joint `pi` (J,) and transition (J, J) (the parts of the joint model that
    do not depend on the observations)."""
    structure = params.structure
    J = states.shape[1]
    pi = np.ones(1)
    for c in range(structure.C):
        pi = np.kron(pi, params.pi_0[structure.state_slice(c)])
    transition = np.ones((J, J))
    for z in range(structure.C):
        transition = transition * params.transition[z][:, states[z]]
    return pi, transition

def _joint_log_emissions(structure: LSIMStructure, states: NDArray[np.int64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    log_B = np.zeros((states.shape[1], B.shape[1]))
    with np.errstate(divide="ignore"):
        for c in range(structure.C):
            log_B = log_B + np.log(B[structure.state_offsets[c] + states[c], :])
    return log_B

def chmm_joint_model(structure: LSIMStructure, params: CHMMParams, Y: NDArray[np.float64], *, validate: bool = True) -> CHMMJointModel:
    """The joint-state HMM of a CHMM for observations `Y` `(D, T)` (NaN =
    missing; a missing dimension is marginalized and a fully missing channel
    has density 1). `validate=True` checks `params` and `Y` first."""
    check_structure_matches(structure, params)
    if validate:
        params.validate()
        validate_observations(structure, Y)
    states = _joint_states(structure)
    pi, transition = _joint_chain(params, states)
    B, B_comp = compute_emissions(structure, params, Y)
    log_B = _joint_log_emissions(structure, states, B)
    return CHMMJointModel(pi=pi, transition=transition, log_B=log_B, states=states, B=B, B_comp=B_comp)

def chmm_exact_forward_backward(
    pi: NDArray[np.float64],
    transition: NDArray[np.float64],
    log_B: NDArray[np.float64],
    *,
    return_xi: bool = False,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Scaled forward-backward on a single chain of `J` states.

    Parameters
    ----------
    pi : ndarray, shape (J,)
        Initial distribution.
    transition : ndarray, shape (J, J)
        Row-stochastic transition matrix (rows = previous state).
    log_B : ndarray, shape (J, T)
        Log emission densities. They are shifted by their per-time maximum
        before exponentiation and the shift is added back to the
        log-likelihood, so very small densities do not underflow.
    return_xi : bool
        Also return the summed pairwise posteriors.

    Returns
    -------
    loglik : float
        Exact log-likelihood.
    gamma : ndarray, shape (J, T)
        Smoothed posteriors.
    alpha : ndarray, shape (J, T)
        One-step predictive distributions `P(x_t | y_1..y_{t-1})`, with
        `alpha[:, 0] = pi`.
    xi_sum : ndarray, shape (J, J), or None
        `xi_sum[j, j'] = sum_t P(x_t = j, x_{t+1} = j' | y)` if `return_xi`.

    Raises
    ------
    ValidationError
        If no state can explain the observation at some time step.
    """
    J, T = log_B.shape
    shift = np.max(log_B, axis=0)
    with np.errstate(invalid="ignore"):
        B = np.exp(log_B - shift)

    alpha = np.zeros((J, T))
    filtered = np.zeros((J, T))
    c = np.zeros(T)
    alpha[:, 0] = pi
    for t in range(T):
        if t > 0:
            alpha[:, t] = transition.T @ filtered[:, t - 1]
        a = alpha[:, t] * B[:, t]
        c[t] = a.sum()
        if not (c[t] > 0) or not np.isfinite(c[t]):
            raise ValidationError(f"chmm_exact_forward_backward: no joint state can explain the observation at t={t}")
        filtered[:, t] = a / c[t]

    beta = np.ones((J, T))
    for t in range(T - 2, -1, -1):
        beta[:, t] = transition @ (B[:, t + 1] * beta[:, t + 1]) / c[t + 1]
    gamma = filtered * beta
    gamma = gamma / gamma.sum(axis=0)
    loglik = float(np.sum(np.log(c)) + np.sum(shift))

    xi_sum = None
    if return_xi:
        xi_sum = ((filtered[:, : T - 1] / c[1:]) @ (B[:, 1:] * beta[:, 1:]).T) * transition
    return loglik, gamma, alpha, xi_sum

def _channel_marginals(structure: LSIMStructure, joint: NDArray[np.float64]) -> NDArray[np.float64]:
    """`(J, T)` joint distribution -> stacked per-channel marginals `(S, T)`."""
    T = joint.shape[1]
    dims = structure.channel_state_num
    grid = joint.reshape(*dims, T)
    out = np.empty((structure.S, T))
    for c in range(structure.C):
        other = tuple(a for a in range(structure.C) if a != c)
        out[structure.state_slice(c), :] = grid.sum(axis=other) if other else grid
    return out

@dataclass
class CHMMPosterior:
    """Result of `forward_backward_chmm`.

    Attributes
    ----------
    loglik : float
        Exact log-likelihood of the sequence.
    alpha : ndarray, shape (S, T)
        Per-channel one-step predictive marginals `P(x^c_t | y_1..y_{t-1})`
        (`alpha[:, 0]` is `pi_0`).
    gamma : ndarray, shape (S, T)
        Per-channel smoothed marginals `P(x^c_t | y_1..y_T)`.
    B, B_comp : ndarray, shapes (S, T) and (S, max_K, T)
        Per-channel state and mixture-component emission densities.
    joint_gamma, joint_alpha : ndarray, shape (J, T), or None
        Smoothed and predictive joint-state distributions
        (only with `return_joint=True`).
    xi_sum : ndarray, shape (J, J), or None
        Summed joint pairwise posteriors `sum_t P(x_t = j, x_{t+1} = j' | y)`
        (only with `return_joint=True`).
    """

    loglik: float
    alpha: NDArray[np.float64]
    gamma: NDArray[np.float64]
    B: NDArray[np.float64]
    B_comp: NDArray[np.float64]
    joint_gamma: NDArray[np.float64] | None = None
    joint_alpha: NDArray[np.float64] | None = None
    xi_sum: NDArray[np.float64] | None = None
    structure: LSIMStructure | None = None

def _posterior_from_joint(
    structure: LSIMStructure,
    states: NDArray[np.int64],
    pi: NDArray[np.float64],
    transition: NDArray[np.float64],
    Y: NDArray[np.float64],
    params: CHMMParams,
    *,
    return_joint: bool,
) -> CHMMPosterior:
    B, B_comp = compute_emissions(structure, params, Y)
    log_B = _joint_log_emissions(structure, states, B)
    loglik, joint_gamma, joint_alpha, xi_sum = chmm_exact_forward_backward(pi, transition, log_B, return_xi=return_joint)
    return CHMMPosterior(
        loglik=loglik,
        alpha=_channel_marginals(structure, joint_alpha),
        gamma=_channel_marginals(structure, joint_gamma),
        B=B,
        B_comp=B_comp,
        joint_gamma=joint_gamma if return_joint else None,
        joint_alpha=joint_alpha if return_joint else None,
        xi_sum=xi_sum,
        structure=structure,
    )

def forward_backward_chmm(
    structure: LSIMStructure,
    params: CHMMParams,
    Y: NDArray[np.float64],
    *,
    validate: bool = True,
    return_joint: bool = False,
) -> CHMMPosterior:
    """Exact forward-backward inference for a CHMM on one sequence.

    Parameters
    ----------
    structure : LSIMStructure
    params : CHMMParams
    Y : ndarray, shape (D, T)
        Observations, channels stacked by dimension; NaN marks missing values.
    validate : bool
        Check `params` and `Y` first.
    return_joint : bool
        Also return the joint-state posteriors and summed pairwise posteriors.

    Returns
    -------
    CHMMPosterior
    """
    check_structure_matches(structure, params)
    if validate:
        params.validate()
        validate_observations(structure, Y)
    states = _joint_states(structure)
    pi, transition = _joint_chain(params, states)
    return _posterior_from_joint(structure, states, pi, transition, Y, params, return_joint=return_joint)
