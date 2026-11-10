"""LSIM parameter initialization (`em_lsim.m` without `init_model`): uniform
initial, transition and coupling parameters, and emissions initialized by the
emission model (for Gaussian mixtures, see `chmm_lsim.emissions.gmm_init`)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.emissions.base import initialize_emissions
from chmm_lsim.emissions.gmm_init import InitialisationError  # noqa: F401  (raised by the GMM initializer)
from chmm_lsim.models.lsim.params import LSIMParams


def initialize_lsim_params(
    structure: LSIMStructure,
    Y_trials: list[NDArray[np.float64]],
    *,
    sigma_diag: bool,
    seed: int = 0,
    matlab_compat: bool = True,
    max_samples_per_channel: int | None = None,
    n_jobs: int = 1,
) -> LSIMParams:
    """`em_lsim.m`'s no-`init_model` branch (`:176-256`): exact uniform
    `pi_0`/`coupling`/`transition`, and per-channel `GaussianMixture`-fitted
    emissions, grouped into states by MATLAB's greedy nearest-neighbour rule.
    `sigma_diag` selects the *requested* model's covariance shape --
    initialization itself always fits diagonal covariances (see module
    docstring) and converts to a diagonal matrix per component when
    `sigma_diag=False`. `matlab_compat=True` (the default, matching every
    other `matlab_compat` knob in this port) replicates `fine_gmm_fitting`'s
    bounded, seeded jitter-and-retry loop on fit failure; `matlab_compat=False`
    raises `InitialisationError` immediately on the first failure instead.
    Returns an already-`.validate()`-passed `LSIMParams` -- never partial or
    uninitialized. Raises `ValidationError` if `Y_trials` is empty or any
    trial fails `validate_observations` (wrong shape/dtype, non-finite, or
    too few timesteps) -- checked before any per-channel slicing.

    Opt-in speed controls (module docstring has the full contract):
    `max_samples_per_channel` (default `None` = every complete row, as MATLAB)
    caps the rows each channel's GMM is fitted on; `n_jobs` (default 1) fits
    channels in that many single-threaded worker processes. Raises
    `ValueError` if `max_samples_per_channel` is not a positive int or is
    smaller than some channel's `M_c*K_c`, or if `n_jobs` is not a positive
    int. `n_jobs>1` starts processes, so call it from under an
    `if __name__ == "__main__":` guard on spawn-based platforms. To reuse one
    initialization across experiments, see `chmm_lsim.save_lsim_params` /
    `chmm_lsim.load_lsim_params`."""
    emissions = initialize_emissions(
        structure, Y_trials, label="initialize_lsim_params", seed=seed, model="gmm",
        sigma_diag=sigma_diag, matlab_compat=matlab_compat, max_samples_per_channel=max_samples_per_channel, n_jobs=n_jobs,
    )

    C = structure.C

    pi_0 = np.concatenate([np.full(structure.channel_state_num[c], 1.0 / structure.channel_state_num[c]) for c in range(C)])
    coupling = np.full((C, C), 1.0 / C)

    transition = np.zeros((structure.S, structure.S))
    for zee in range(C):
        transition[:, structure.state_slice(zee)] = 1.0 / structure.channel_state_num[zee]

    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate(matlab_compat=matlab_compat)
    return params
