"""Starting parameters for `em_chmm`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.emissions.base import initialize_emissions
from chmm_lsim.models.chmm.params import CHMMParams, _num_joint_states


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
    keyword arguments are the same)."""
    J = _num_joint_states(structure)
    emissions = initialize_emissions(
        structure, Y_trials, label="initialize_chmm_params", seed=seed, model="gmm",
        sigma_diag=sigma_diag, matlab_compat=matlab_compat, max_samples_per_channel=max_samples_per_channel, n_jobs=n_jobs,
    )
    M = structure.channel_state_num
    params = CHMMParams(
        structure=structure,
        pi_0=np.concatenate([np.full(m, 1.0 / m) for m in M]),
        transition=[np.full((J, m), 1.0 / m) for m in M],
        emissions=emissions,
    )
    params.validate(matlab_compat=matlab_compat)
    return params
