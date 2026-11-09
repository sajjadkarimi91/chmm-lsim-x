"""Full coupled HMM (CHMM) versus LSIM on the same data (Python twin of matlab/demos/demo_run_chmm.m).

A CHMM conditions every channel's next state on the JOINT state of all channels, so its
parameters and its exact inference grow with prod(M). An LSIM replaces the joint conditional
by a theta-weighted mixture of pairwise transitions, so it grows with sum(M)^2. This demo
simulates data from a small 2-channel LSIM, fits both models by EM (same initial emissions,
same result type) and decodes the most likely joint state path of the CHMM exactly.

Needs the `init` extra for the initializers:  pip install "chmm-lsim[init]"
"""
from __future__ import annotations

import time

import numpy as np

from chmm_lsim import (
    LSIMStructure, em_chmm, em_lsim, initialize_chmm_params, initialize_lsim_params, simulate_lsim,
    viterbi_chmm,
)

from demo_run_lsim import random_lsim


def main() -> None:
    t_start = time.time()
    structure = LSIMStructure(channel_state_num=(3, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 1))
    truth = random_lsim(structure, np.random.default_rng(2))
    Y_list = [simulate_lsim(structure, truth, T=200, seed=s)[0] for s in range(5)]

    chmm = em_chmm(structure, initialize_chmm_params(structure, Y_list, seed=0), Y_list, max_iterations=50)
    lsim = em_lsim(structure, initialize_lsim_params(structure, Y_list, sigma_diag=True, seed=0), Y_list,
                   max_iterations=50, check_convergence=False)

    print(f"{'':6s}{'final log-likelihood':>22s}{'AIC':>12s}{'BIC':>12s}")
    for name, res in (("CHMM", chmm), ("LSIM", lsim)):
        print(f"{name:6s}{res.loglik_trace[-1]:22.2f}{res.aic_trace[-1]:12.1f}{res.bic_trace[-1]:12.1f}")

    path = viterbi_chmm(structure, chmm.params, Y_list[0])
    print(f"\nExact CHMM Viterbi path: log P(path, Y) = {path.log_score:.2f}; first 10 samples:")
    for c in range(structure.C):
        print(f"  channel {c}: {path.states[c, :10].tolist()}")
    print(f"\nDone in {time.time() - t_start:.2f}s total.")


if __name__ == "__main__":
    main()
