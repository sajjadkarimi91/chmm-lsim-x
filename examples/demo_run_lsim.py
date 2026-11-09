"""LSIM inference and EM learning in a few steps (Python twin of matlab/demos/demo_run_lsim.m).

1. Build a random 3-channel LSIM and simulate one observation sequence with missing samples.
2. Smooth it with the three LSIM recursions (POISE, the default; SED; fast/SVI) and compare
   each with exact inference on the equivalent joint HMM.
3. Initialize and learn an LSIM from the data by EM (POISE E-step, the default), then decode
   the hidden states with Viterbi.

Needs the `init` extra for the initializer:  pip install "chmm-lsim[init]"
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from chmm_lsim import (
    ChannelGMM, FbConstants, LSIMParams, LSIMStructure, build_equivalent_hmm, em_lsim,
    exact_posterior, forward_backward, initialize_lsim_params, marginalize_joint_to_channels,
    simulate_lsim, viterbi_lsim,
)

SEED = 0


def random_lsim(structure: LSIMStructure, rng: np.random.Generator) -> LSIMParams:
    """Random parameters: rows of every transition block and columns of the coupling sum to one."""
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for c in range(structure.C):
        for z in range(structure.C):
            transition[structure.state_slice(c), structure.state_slice(z)] = rng.dirichlet(
                np.ones(structure.channel_state_num[z]), structure.channel_state_num[c])
    coupling = rng.dirichlet(np.ones(structure.C), structure.C).T      # theta[c, z]: influence of c on z
    emissions = []
    for c in range(structure.C):
        M, d, K = structure.channel_state_num[c], structure.channel_obs_dim[c], structure.num_gmm_component[c]
        mu = 1.5 * np.arange(1, M + 1)[None, :, None] + rng.standard_normal((d, M, K))
        emissions.append(ChannelGMM(weights=np.ones((M, K)), mu=mu, sigma=0.5 + rng.random((d, M, K)), diag=True))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    return params


def main() -> None:
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # 1. model and data: Y is (D, T), channels stacked along D, NaN = missing
    structure = LSIMStructure(channel_state_num=(3, 2, 4), channel_obs_dim=(2, 1, 3), num_gmm_component=(1, 1, 1))
    true_params = random_lsim(structure, rng)
    Y, _ = simulate_lsim(structure, true_params, T=400, seed=SEED)
    Y[structure.dim_slice(1), [5, 50, 200]] = np.nan
    print(f"LSIM with C={structure.C} channels, states {structure.channel_state_num}, T={Y.shape[1]}")

    # 2. approximate smoothing vs exact inference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        equivalent = build_equivalent_hmm(structure, true_params)
        gamma_joint, _ = exact_posterior(equivalent, Y)
        exact = marginalize_joint_to_channels(equivalent.dims, gamma_joint)
    print("Mean Hellinger distance to the exact smoothed marginals:")
    for variant in ("poise", "sed", "fast"):
        t0 = time.time()
        post = forward_backward(structure, true_params, Y, FbConstants(variant=variant))
        elapsed = time.time() - t0
        h = np.mean(np.sqrt(0.5 * np.add.reduceat((np.sqrt(post.gamma) - np.sqrt(exact)) ** 2,
                                                  structure.state_offsets[:-1], axis=0)))
        print(f"  {variant.upper():6s} H = {h:.4f}   ({elapsed * 1e3:.1f} ms, log-likelihood {post.loglik:.2f})")

    # 3. EM learning (POISE E-step by default) and Viterbi decoding
    params0 = initialize_lsim_params(structure, [Y], sigma_diag=True, seed=SEED)
    result = em_lsim(structure, params0, [Y], max_iterations=50, check_convergence=False)
    print(f"EM: log-likelihood {result.loglik_trace[0]:.2f} -> {result.loglik_trace[-1]:.2f} "
          f"after {result.n_iterations} iterations")
    print("Learned coupling matrix (columns sum to one):")
    print(np.round(result.params.coupling, 3))
    states = viterbi_lsim(structure, result.params, Y).states
    print("Viterbi states, first 10 samples of each channel:")
    for c in range(structure.C):
        print(f"  channel {c}: {states[c, :10].tolist()}")
    print(f"\nDone in {time.time() - t_start:.2f}s total.")


if __name__ == "__main__":
    main()
