"""Maximum-likelihood EM for a CHMM with exact joint-state E-steps
(`em_chmm.m`). Shares the emission statistics and M-step, the initial
distribution update, the iteration loop and `EmResult` with `em_lsim`."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import check_structure_matches, validate_observation_list
from chmm_lsim.emissions.base import (
    accumulate_emission_stats,
    emission_model_for,
    get_emission_model,
    m_step_emissions,
    new_emission_stats,
)
from chmm_lsim.learning.em_loop import EmResult, ExpectedStats, _aic_bic, _run_em_loop, apply_left2right_pi0_smoothing, finalize_expected_stats
from chmm_lsim.models.chmm.inference import _joint_chain, _posterior_from_joint
from chmm_lsim.models.chmm.params import CHMMParams, _joint_states


_VALIDATE_MODES = ("params", "all", "off")

# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------


def compute_chmm_aic_bic(structure: LSIMStructure, sigma_diag: bool, loglik: float, T_all: int) -> tuple[float, float]:
    """AIC and BIC of a CHMM fit, with `em_chmm.m`'s parameter count:
    `sum(M-1)*J` transition parameters plus the LSIM emission parameters
    (see `compute_aic_bic`)."""
    M = np.array(structure.channel_state_num)
    n_params = int(np.sum(M - 1)) * math.prod(structure.channel_state_num) + get_emission_model("gmm").n_free_parameters(
        structure, sigma_diag=sigma_diag
    )
    return _aic_bic(loglik, n_params, T_all)

def _chmm_e_step(structure: LSIMStructure, params: CHMMParams, Y_trials: list[NDArray[np.float64]]) -> ExpectedStats:
    """Raw (unfinalized) expected statistics over all trials; `zeta` holds the
    `(J, J)` joint pairwise posterior mass summed over time and trials. The
    per-channel emission statistics are the LSIM ones."""
    emission_model = emission_model_for(params.emissions)
    states = _joint_states(structure)
    J = states.shape[1]
    pi, transition = _joint_chain(params, states)
    channels = new_emission_stats(structure, params.emissions, emission_model)
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
        accumulate_emission_stats(channels, structure, post, Y, params.emissions, emission_model)
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
        emissions=m_step_emissions(structure, stats, sigma_diag=sigma_diag, matlab_compat=matlab_compat),
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
