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

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.em import (
    EmResult,
    ExpectedStats,
    _accumulate_channel_trial,
    _emission_param_count,
    _empty_channel_stats,
    _m_step_emissions,
    _run_em_loop,
    apply_left2right_pi0_smoothing,
    finalize_expected_stats,
)
from chmm_lsim.emissions import compute_emissions
from chmm_lsim.equivalent import MAX_JOINT_STATES, EquivalentModelTooLargeError, _conditional_transitions
from chmm_lsim.params import ChannelGMM, LSIMParams, check_emissions_match
from chmm_lsim.simulate import _gmm_sample, select_p
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import (
    _SUM_ATOL,
    ValidationError,
    _check_no_nan_inf,
    check_structure_matches,
    validate_emissions,
    validate_observation_list,
    validate_observations,
)
from chmm_lsim.viterbi import ViterbiResult

__all__ = [
    "CHMMParams",
    "CHMMJointModel",
    "CHMMPosterior",
    "lsim_to_chmm",
    "chmm_joint_model",
    "chmm_exact_forward_backward",
    "forward_backward_chmm",
    "viterbi_chmm",
    "compute_chmm_aic_bic",
    "em_chmm",
    "initialize_chmm_params",
    "simulate_chmm",
]

_VALIDATE_MODES = ("params", "all", "off")


def _num_joint_states(structure: LSIMStructure) -> int:
    """`J = prod(M)`, rejected above `MAX_JOINT_STATES` (exact Python integer
    product, so it cannot overflow)."""
    J = math.prod(structure.channel_state_num)
    if J > MAX_JOINT_STATES:
        raise EquivalentModelTooLargeError(
            f"a CHMM with channel_state_num={structure.channel_state_num} has J={J} joint states, exceeding "
            f"MAX_JOINT_STATES={MAX_JOINT_STATES}; exact CHMM inference is meant for small models"
        )
    return J


def _joint_states(structure: LSIMStructure) -> NDArray[np.int64]:
    """`(C, J)` zero-based per-channel states of every joint state
    (MATLAB's `index_matrix - 1`)."""
    J = _num_joint_states(structure)
    return np.array(np.unravel_index(np.arange(J), structure.channel_state_num), dtype=np.int64).reshape(structure.C, J)


@dataclass
class CHMMParams:
    """Parameters of a full CHMM.

    Attributes
    ----------
    structure : LSIMStructure
        Per-channel numbers of states, observation dimensions and mixture
        components.
    pi_0 : ndarray, shape (S,)
        Per-channel initial distributions, stacked channel by channel; each
        block sums to one. The joint initial distribution is their product.
    transition : list of ndarray
        `transition[z]` has shape `(J, M_z)`; row `j` is
        `P(x^z_t | joint state j at t-1)` and sums to one.
    emissions : list of ChannelGMM
        Per-channel Gaussian mixtures (all diagonal or all full covariance).
    """

    structure: LSIMStructure
    pi_0: NDArray[np.float64]
    transition: list[NDArray[np.float64]]
    emissions: list[ChannelGMM] = field(default_factory=list)

    def __post_init__(self) -> None:
        structure = self.structure
        J = _num_joint_states(structure)
        self.pi_0 = np.asarray(self.pi_0, dtype=np.float64)
        if self.pi_0.shape != (structure.S,):
            raise ValueError(f"pi_0.shape={self.pi_0.shape} != (S,)=({structure.S},)")
        if len(self.transition) != structure.C:
            raise ValueError(f"len(transition)={len(self.transition)} != C={structure.C}")
        self.transition = [np.asarray(t, dtype=np.float64) for t in self.transition]
        for z, table in enumerate(self.transition):
            expected = (J, structure.channel_state_num[z])
            if table.shape != expected:
                raise ValueError(f"transition[{z}].shape={table.shape} != (J, M_z)={expected}")
        check_emissions_match(structure, self.emissions)

    @property
    def sigma_diag(self) -> bool:
        """True for diagonal covariances, False for full covariances."""
        return self.emissions[0].diag

    @property
    def n_joint_states(self) -> int:
        """`J = prod(channel_state_num)`."""
        return math.prod(self.structure.channel_state_num)

    def validate(self, *, matlab_compat: bool = True) -> None:
        """Raises `ValidationError` unless every distribution is finite,
        non-negative and sums to one (within 1e-8), and every emission model
        is valid (see `chmm_lsim.validate.validate_emissions`)."""
        structure = self.structure
        _check_no_nan_inf("pi_0", self.pi_0)
        if (self.pi_0 < 0).any():
            raise ValidationError("pi_0 has negative entries")
        for c in range(structure.C):
            total = float(self.pi_0[structure.state_slice(c)].sum())
            if not np.isclose(total, 1.0, rtol=0, atol=_SUM_ATOL):
                raise ValidationError(f"pi_0 block for channel {c} sums to {total}, expected 1.0")
        for z, table in enumerate(self.transition):
            _check_no_nan_inf(f"transition[{z}]", table)
            if (table < 0).any():
                raise ValidationError(f"transition[{z}] has negative entries")
            row_sums = table.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=0, atol=_SUM_ATOL):
                bad = int(np.argmax(np.abs(row_sums - 1.0)))
                raise ValidationError(f"transition[{z}] row {bad} sums to {row_sums[bad]}, expected 1.0")
        validate_emissions(self.emissions, matlab_compat=matlab_compat)

    def copy(self) -> CHMMParams:
        return CHMMParams(
            structure=self.structure,
            pi_0=self.pi_0.copy(),
            transition=[t.copy() for t in self.transition],
            emissions=[gmm.copy() for gmm in self.emissions],
        )


def lsim_to_chmm(structure: LSIMStructure, params: LSIMParams) -> CHMMParams:
    """The CHMM an LSIM is equivalent to: `transition[z][j, :] =
    sum_c coupling[c, z] * A[c, z][m_c(j), :]`, with the LSIM's own `pi_0`
    and emissions (copied)."""
    check_structure_matches(structure, params)
    states = _joint_states(structure)
    return CHMMParams(
        structure=structure,
        pi_0=params.pi_0.copy(),
        transition=_conditional_transitions(structure, params, states),
        emissions=[gmm.copy() for gmm in params.emissions],
    )


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


def viterbi_chmm(structure: LSIMStructure, params: CHMMParams, Y: NDArray[np.float64], *, validate: bool = True) -> ViterbiResult:
    """Exact Viterbi decoding of a CHMM on its joint state space.

    Returns a `ViterbiResult` whose `states` `(C, T)` are zero-based and whose
    `log_score` is the log-probability of the decoded joint path together with
    the observations, `log P(x_1..x_T, y_1..y_T)` -- a true log-probability,
    unlike the per-channel score of the approximate `viterbi_lsim`. Ties are
    broken towards the lowest joint-state index.
    """
    model = chmm_joint_model(structure, params, Y, validate=validate)
    J, T = model.log_B.shape
    with np.errstate(divide="ignore"):
        log_A = np.log(model.transition)
        delta = np.log(model.pi) + model.log_B[:, 0]
    psi = np.zeros((J, T), dtype=np.int64)
    for t in range(1, T):
        candidate = delta[:, None] + log_A  # (previous, next)
        psi[:, t] = np.argmax(candidate, axis=0)
        delta = candidate[psi[:, t], np.arange(J)] + model.log_B[:, t]

    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(delta))
    log_prob = float(delta[path[T - 1]])
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return ViterbiResult(states=model.states[:, path].copy(), log_score=log_prob)


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------


def compute_chmm_aic_bic(structure: LSIMStructure, sigma_diag: bool, loglik: float, T_all: int) -> tuple[float, float]:
    """AIC and BIC of a CHMM fit, with `em_chmm.m`'s parameter count:
    `sum(M-1)*J` transition parameters plus the LSIM emission parameters
    (see `compute_aic_bic`)."""
    M = np.array(structure.channel_state_num)
    n_params = int(np.sum(M - 1)) * math.prod(structure.channel_state_num) + _emission_param_count(structure, sigma_diag)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + np.log(T_all) * n_params
    return float(aic), float(bic)


def _chmm_e_step(structure: LSIMStructure, params: CHMMParams, Y_trials: list[NDArray[np.float64]]) -> ExpectedStats:
    """Raw (unfinalized) expected statistics over all trials; `zeta` holds the
    `(J, J)` joint pairwise posterior mass summed over time and trials. The
    per-channel emission statistics are the LSIM ones."""
    sigma_diag = params.sigma_diag
    states = _joint_states(structure)
    J = states.shape[1]
    pi, transition = _joint_chain(params, states)
    channels = [
        _empty_channel_stats(structure.channel_state_num[c], structure.num_gmm_component[c], structure.channel_obs_dim[c], sigma_diag)
        for c in range(structure.C)
    ]
    xi_total = np.zeros((J, J))
    gamma_first = np.zeros(structure.S)
    gamma_sum = np.zeros(structure.S)
    n_transitions = 0
    T_all = 0
    loglik = 0.0
    for Y in Y_trials:
        post = _posterior_from_joint(structure, states, pi, transition, Y, params, return_joint=True)
        T = Y.shape[1]
        T_all += T
        n_transitions += T - 1
        loglik += post.loglik
        gamma_first += post.gamma[:, 0]
        gamma_sum += post.gamma.sum(axis=1)
        xi_total += post.xi_sum
        for c in range(structure.C):
            _accumulate_channel_trial(channels[c], structure, c, post, Y, sigma_diag)
    return ExpectedStats(
        zeta=xi_total,
        gamma_first=gamma_first,
        gamma_sum=gamma_sum,
        n_trials=len(Y_trials),
        n_transitions=n_transitions,
        T_all=T_all,
        loglik=loglik,
        channels=channels,
    )


def _chmm_m_step(structure: LSIMStructure, stats: ExpectedStats, *, sigma_diag: bool, matlab_compat: bool) -> CHMMParams:
    """`em_chmm.m`'s M-step from finalized stats. The joint pairwise mass is
    normalized per previous joint state (a state with no mass gets the
    uniform `1/J`), then marginalized onto each channel's next state."""
    states = _joint_states(structure)
    J = states.shape[1]
    xi = stats.zeta
    with np.errstate(invalid="ignore", divide="ignore"):
        joint_next = xi / xi.sum(axis=1, keepdims=True)
    joint_next = np.where(np.isnan(joint_next), 1.0 / J, joint_next)
    transition = []
    for z in range(structure.C):
        table = np.zeros((J, structure.channel_state_num[z]))
        for s in range(structure.channel_state_num[z]):
            table[:, s] = joint_next[:, states[z] == s].sum(axis=1)
        transition.append(table)
    return CHMMParams(
        structure=structure,
        pi_0=stats.gamma_p0.copy(),
        transition=transition,
        emissions=_m_step_emissions(structure, stats, sigma_diag=sigma_diag, matlab_compat=matlab_compat),
    )


def em_chmm(
    structure: LSIMStructure,
    params: CHMMParams,
    Y_trials: list[NDArray[np.float64]],
    max_iterations: int,
    *,
    matlab_compat: bool = True,
    check_convergence: bool = False,
    validate: str = "params",
    left2right: bool = False,
    raw_init_model: bool = False,
    iter_hook=None,
) -> EmResult:
    """Maximum-likelihood EM for a CHMM with exact joint-state E-steps
    (`em_chmm.m`).

    The emission update, the initial-distribution update (mean posterior at
    the first time step), the convergence rule and the choice of the returned
    iterate are those of `em_lsim`; only the transition model differs. Each
    previous joint state's next-joint-state distribution is the normalized
    expected pairwise count, marginalized onto each channel; a previous joint
    state with no posterior mass (for instance when every sequence has
    `T = 1`) gets uniform rows, as in MATLAB.

    Parameters
    ----------
    structure : LSIMStructure
    params : CHMMParams
        Starting parameters, e.g. from `initialize_chmm_params`.
    Y_trials : list of ndarray
        Observation sequences, each `(D, T_n)`; NaN marks missing values.
    max_iterations : int
        Maximum number of EM iterations.
    matlab_compat : bool
        Emission M-step conventions (see `em_lsim`); True reproduces MATLAB.
    check_convergence : bool
        Stop once at least two of the last five log-likelihood increments are
        below 1e-3 (checked from iteration 10). MATLAB's default is off.
    validate : {"params", "all", "off"}
        "params"/"all" check the observations once and the parameters before
        every E-step; "off" skips both.
    left2right, raw_init_model : bool
        With `raw_init_model=True`, `params.pi_0` is first smoothed as MATLAB
        does for a supplied `init_model` (`(1e-4 + pi_0)` renormalized per
        channel, or only renormalized when `left2right=True`).
    iter_hook : callable, optional
        Called as `iter_hook(itr, params, loglik, aic, bic)` after every
        iteration, with the updated parameters and the scores of the
        parameters before the update.

    Returns
    -------
    EmResult
        `params` is the `CHMMParams` iterate with the highest log-likelihood
        (selected as in `em_lsim`), with its `pi_steady` (mean posterior
        state occupancy); the traces hold the log-likelihood, AIC and BIC of
        the parameters each iteration started from.
    """
    if validate not in _VALIDATE_MODES:
        raise ValueError(f"validate must be one of {_VALIDATE_MODES}, got {validate!r}")
    check_structure_matches(structure, params)
    if len(Y_trials) == 0:
        raise ValueError("em_chmm requires at least one trial in Y_trials")
    if validate != "off":
        validate_observation_list(structure, Y_trials)
    if raw_init_model:
        params = params.copy()
        params.pi_0 = apply_left2right_pi0_smoothing(params.pi_0, structure, left2right)
    T_all = sum(Y.shape[1] for Y in Y_trials)
    sigma_diag = params.sigma_diag

    def iteration(params: CHMMParams):
        if validate != "off":
            params.validate(matlab_compat=matlab_compat)
        stats = finalize_expected_stats(_chmm_e_step(structure, params, Y_trials), sigma_diag=sigma_diag)
        aic, bic = compute_chmm_aic_bic(structure, sigma_diag, stats.loglik, T_all)
        params = _chmm_m_step(structure, stats, sigma_diag=sigma_diag, matlab_compat=matlab_compat)
        return params, stats.loglik, aic, bic, stats.gamma_sum / stats.T_all

    return _run_em_loop(params, iteration, max_iterations=max_iterations, check_convergence=check_convergence, iter_hook=iter_hook)


def initialize_chmm_params(
    structure: LSIMStructure,
    Y_trials: list[NDArray[np.float64]],
    *,
    sigma_diag: bool = True,
    seed: int = 0,
    matlab_compat: bool = True,
    max_samples_per_channel: int | None = None,
    n_jobs: int = 1,
) -> CHMMParams:
    """Starting parameters for `em_chmm` (`em_chmm.m` without `init_model`):
    uniform `pi_0` and transition tables, and per-channel GMM emissions fitted
    exactly as `initialize_lsim_params` does (requires scikit-learn; the
    keyword arguments are passed through)."""
    from chmm_lsim.init import initialize_lsim_params

    J = _num_joint_states(structure)
    lsim_params = initialize_lsim_params(
        structure,
        Y_trials,
        sigma_diag=sigma_diag,
        seed=seed,
        matlab_compat=matlab_compat,
        max_samples_per_channel=max_samples_per_channel,
        n_jobs=n_jobs,
    )
    M = structure.channel_state_num
    params = CHMMParams(
        structure=structure,
        pi_0=np.concatenate([np.full(m, 1.0 / m) for m in M]),
        transition=[np.full((J, m), 1.0 / m) for m in M],
        emissions=lsim_params.emissions,
    )
    params.validate(matlab_compat=matlab_compat)
    return params


def simulate_chmm(
    structure: LSIMStructure,
    params: CHMMParams,
    T: int,
    *,
    seed: int,
    validate: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Samples one sequence of length `T` from a CHMM
    (`generate_chmm_time_series.m`).

    Returns `(Y, X)`: observations `(D, T)` and zero-based hidden states
    `(C, T)`. Draws use `np.random.default_rng(seed)` only, so equal seeds
    give equal samples (not the MATLAB random stream)."""
    if T < 1:
        raise ValidationError(f"simulate_chmm requires T >= 1, got T={T}")
    check_structure_matches(structure, params)
    if validate:
        params.validate()
    rng = np.random.default_rng(seed)
    C = structure.C
    dims = structure.channel_state_num
    X = np.zeros((C, T), dtype=np.int64)
    Y = np.zeros((structure.D, T))
    for c in range(C):
        X[c, 0] = select_p(rng, params.pi_0[structure.state_slice(c)])
        Y[structure.dim_slice(c), 0] = _gmm_sample(rng, params.emissions[c], X[c, 0])
    for t in range(1, T):
        j = int(np.ravel_multi_index(tuple(int(x) for x in X[:, t - 1]), dims))
        for c in range(C):
            X[c, t] = select_p(rng, params.transition[c][j])
            Y[structure.dim_slice(c), t] = _gmm_sample(rng, params.emissions[c], X[c, t])
    return Y, X
