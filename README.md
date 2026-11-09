# chmm-lsim-x

Inference and learning for **coupled hidden Markov models (CHMMs)** and **Latent Structure Influence Models (LSIMs)**, in **MATLAB** and **Python**.

An LSIM models C interacting channels (for example EEG electrodes or sensors). Each channel has its own hidden Markov chain, and the next state of channel ζ is drawn from a mixture of pairwise transitions from every channel c, weighted by coupling strengths θ(c, ζ):

P(x<sub>t</sub><sup>ζ</sup> | x<sub>t−1</sub>) = Σ<sub>c</sub> θ(c, ζ) · A<sup>c,ζ</sup>(x<sub>t−1</sub><sup>c</sup>, x<sub>t</sub><sup>ζ</sup>)

This keeps the model size at O((Σ M)²) instead of the O(Π M) joint state space of a full CHMM.

## What is included

| | MATLAB (`matlab/`) | Python (`chmm_lsim`) |
|---|---|---|
| Forward-backward inference: **POISE** (default), **SED**, **fast/SVI** | `forward_backward_lsim` + `lsim_fb_constants` | `forward_backward`, `FbConstants` |
| Exact inference for small models (equivalent joint HMM) | `forward_backward_chmm`, `viterbi_chmm`, `em_lsim_exact` | `forward_backward_chmm`, `viterbi_chmm`, `exact_posterior`, `viterbi_exact` |
| Maximum-likelihood EM learning | `em_lsim`, `em_lsim_poise`, `em_lsim_fast` | `em_lsim`, `em_lsim_minibatch_train` |
| Full CHMM (joint-state transitions): EM, exact inference, exact Viterbi, simulation | `em_chmm`, `forward_backward_chmm`, `viterbi_chmm`, `generate_chmm_time_series` | `em_chmm`, `forward_backward_chmm`, `viterbi_chmm`, `simulate_chmm`, `initialize_chmm_params`, `lsim_to_chmm` |
| Discriminative learning (**MMI**) of class-conditional LSIMs | `em_lsim_discriminative`, `lsim_discriminative_predict` | `em_lsim_discriminative`, `lsim_discriminative_predict` |
| LSIM simulation, approximate Viterbi decoding, model save/load | `generate_chmm_time_series`, `viterbi_lsim` | `simulate_lsim`, `viterbi_lsim`, `save_lsim_params` |
| Execution | CPU | NumPy, numba, PyTorch (CPU or CUDA, batched) |

All observations may contain missing samples (`NaN`). Emissions are Gaussian mixtures with diagonal or full covariance.

`em_lsim` and `em_chmm` share the same options, initialization, emission update, convergence rule and outputs: in MATLAB, `em_chmm` returns `em_lsim`'s outputs in the same order without the coupling matrix (`[pi_0, transitions, gmm, AIC, log_likelihood, BIC, pi_steady]`), and in Python both return the same `EmResult`. `viterbi_lsim` is a fast approximate per-channel decoder for LSIMs; `viterbi_chmm` decodes a CHMM exactly on the joint state space (in MATLAB, `viterbi_chmm` called with LSIM parameters still runs `viterbi_lsim`).

### Choosing an inference recursion

All three recursions share the same forward pass, so they give the same approximate log-likelihood at fixed parameters. They differ in the backward pass:

- **POISE** (default): weighs each channel's outgoing influence by its total outgoing coupling strength. It needs no pairwise (S×S) message, is the fastest, and gives the most accurate smoothed marginals in our simulations, especially for strongly coupled channels.
- **SED**: builds a pairwise conditional message and solves a small linear system per channel at every step. Slower, and uses more memory.
- **fast/SVI**: fixed coupling weights raised to a fixed power. As fast as POISE.

Use exact inference (Cartesian product of channel states) only for small models; its cost grows with the product of the state counts.

## Installation

**MATLAB** (tested with R2024b; the GMM initialization in `em_lsim` uses the Statistics and Machine Learning Toolbox):

```matlab
addpath(genpath('<path to chmm-lsim-x>/matlab'))
```

**Python** (3.10 or later):

```bash
pip install "chmm-lsim[all] @ git+https://github.com/sajjadkarimi91/chmm-lsim-x"
```

Optional extras: `init` (scikit-learn, for `initialize_lsim_params`), `numba` (numba CPU backend), `torch` (PyTorch CPU/CUDA backend), `all` (all three). The core needs only NumPy and SciPy. For CUDA, install a CUDA build of PyTorch first (see pytorch.org).

## Quick start: MATLAB

Observations are a C×1 cell array; `observations{c}` is a d(c)×T matrix (`NaN` = missing). For several sequences, use a C×N cell array.

```matlab
% EM learning: 4 hidden states and 1 Gaussian per channel, 100 iterations, POISE E-step
extra = struct('plot', 0, 'sigma_diag', 1);
[pi_0, coupling, transitions, gmm_para, AIC, log_lik, BIC] = ...
    em_lsim(observations, 4*ones(1,C), ones(1,C), 100, extra);

% Smoothed marginals P(x_t^c | all data); marginals{c} is M(c)xT
[log_lik, ~, ~, marginals] = forward_backward_lsim(pi_0, coupling, transitions, gmm_para, observations);

% Choose the recursion explicitly
fbc = lsim_fb_constants('sed');           % 'poise' (default), 'sed' or 'fast'
[~, ~, ~, marginals_sed] = forward_backward_lsim(pi_0, coupling, transitions, gmm_para, observations, fbc);
extra.fb_constants = fbc;                 % the same option selects the E-step of em_lsim
```

**MMI training** of one LSIM per class, starting from EM models:

```matlab
% class_obs{k}: C x N_k cell of class-k training sequences; val_obs: same format
for k = 1:numel(class_obs)                % generative start: EM per class
    [pi_0, theta, A, gmm] = em_lsim(class_obs{k}, M, K, 30, extra);
    init_models{k} = struct('pi_0_lsim', {pi_0}, 'coupling_tetha_IM', theta, ...
        'transition_matrices_IM_cell', {A}, 'gmm_para_lsim', {gmm});
end
[models, history] = em_lsim_discriminative(class_obs, M, K, [], ...
    struct('objective', 'mmi', 'validation_observations', {val_obs}), init_models);
[scores, predicted] = lsim_discriminative_predict(models, test_obs);   % test_obs: C x N cell
```

**Full CHMM** (small models; parameters have the same cell format as `em_lsim`'s, with one prod(M) x M(c) transition table per channel):

```matlab
[pi_0, transitions, gmm, AIC, log_lik, BIC] = em_chmm(observations, M, K, 100, extra);
[log_lik, ~, ~, marginals] = forward_backward_chmm(pi_0, transitions, gmm, observations);   % exact
[log_prob, states] = viterbi_chmm(pi_0, transitions, gmm, observations);                  % exact
```

Demos: `matlab/demos/demo_run_lsim.m` (POISE, SED and fast vs exact inference, then EM), `demo_run_chmm.m` (CHMM vs LSIM) and `demo_discriminative_lsim.m` (EM vs MMI classification).

## Quick start: Python

Observations of one sequence are a (D, T) array with the channels stacked along D (`NaN` = missing); a dataset is a list of such arrays.

```python
from chmm_lsim import (LSIMStructure, FbConstants, initialize_lsim_params, em_lsim,
                       forward_backward, viterbi_lsim)

structure = LSIMStructure(channel_state_num=(4, 4, 4), channel_obs_dim=(1, 2, 1),
                          num_gmm_component=(1, 1, 1))
params0 = initialize_lsim_params(structure, Y_list, sigma_diag=True, seed=0)   # needs [init]
result = em_lsim(structure, params0, Y_list, max_iterations=100)                # POISE E-step
post = forward_backward(structure, result.params, Y_list[0])                    # post.gamma: (S, T)
post_sed = forward_backward(structure, result.params, Y_list[0], FbConstants(variant="sed"))
states = viterbi_lsim(structure, result.params, Y_list[0]).states
```

Large datasets and GPUs:

```python
from chmm_lsim import em_lsim_minibatch_train, forward_backward_batch

loglik = forward_backward_batch(structure, result.params, Y_list, mode="loglik",
                                device="cuda", dtype="float32")                 # batched scoring
fit = em_lsim_minibatch_train(structure, params0, Y_list, max_iterations=20, num_minibatches=10,
                              backend="torch", backend_options={"device": "cuda", "dtype": "float64"})
```

MMI training:

```python
from chmm_lsim import em_lsim_discriminative, lsim_discriminative_predict

res = em_lsim_discriminative((Y_train, y_train), init_models,   # one LSIMParams per class
                             validation=(Y_val, y_val), objective="mmi")
scores, predicted, normalized = lsim_discriminative_predict(res.models, Y_test)
```

Full CHMM (exact, for small models; `em_chmm` returns the same `EmResult` as `em_lsim`):

```python
from chmm_lsim import initialize_chmm_params, em_chmm, forward_backward_chmm, viterbi_chmm

fit = em_chmm(structure, initialize_chmm_params(structure, Y_list, seed=0), Y_list, max_iterations=100)
post = forward_backward_chmm(structure, fit.params, Y_list[0])      # post.gamma: (S, T)
states = viterbi_chmm(structure, fit.params, Y_list[0]).states       # exact joint decoding
```

Demos: `examples/demo_run_lsim.py` and `examples/demo_run_chmm.py`.

## Benchmarks

`benchmarks/` reproduces the runtime and memory comparison of the POISE, SED and fast/SVI backward recursions on CPU and GPU, and the full E-step on 100,000 sequences. See `benchmarks/README.md`.

## Repository layout

```
matlab/
  lsim/            LSIM inference (POISE, SED, fast/SVI), EM, exact EM, constants
  chmm/            full CHMM: EM, forward-backward, Viterbi, simulation
  common/          Gaussian-mixture densities and parameter helpers
  discriminative/  MMI training, prediction, metrics (+ tests/)
  demos/
src/chmm_lsim/     Python package
tests/             Python tests (pytest), with MATLAB reference fixtures
examples/          Python demos
benchmarks/        runtime benchmark
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The Python implementation is checked against MATLAB reference outputs (`tests/fixtures/`). MATLAB tests of the MMI trainer: `runtests('matlab/discriminative/tests/TestMMI.m')`.

## License

BSD 3-Clause. See `LICENSE`.
