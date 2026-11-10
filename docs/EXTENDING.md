# Extending chmm-lsim-x

This guide is for adding models and algorithms to the toolbox without breaking existing users. The package grows along three independent axes, and each has one place in both languages:

| You are adding | Example | Python | MATLAB |
|---|---|---|---|
| a new LSIM inference recursion | a new LSIM version (new backward message) | `src/chmm_lsim/models/lsim/inference/<name>.py` | `matlab/lsim/backward/lsim_backward_<name>.m` |
| a new emission (observation) model | AR-LSIM (state-dependent autoregressive observations) | `src/chmm_lsim/emissions/<name>.py` | `matlab/emissions/` (+ the M-step, see below) |
| a new model family (transition model) | higher-order LSIM | `src/chmm_lsim/models/<name>/` | `matlab/<name>/` |

## Layout

```
src/chmm_lsim/
  __init__.py          public API (the only stable import location, with chmm_lsim.discriminative)
  core/                structure, validation, constants (FbConstants), serialization, joint-state utilities
  emissions/           EmissionModel interface + registry; gmm.py (Gaussian mixtures), gmm_init.py
  models/lsim/         LSIM: params, inference/ (POISE, SED, fast + registry + from_emissions), em, init,
                       simulate, viterbi, equivalent (exact joint-state form), serialization
  models/chmm/         full CHMM: params, inference, em, init, simulate, viterbi
  learning/            shared EM loop (EmResult), mini-batch scheduling, discriminative/ (MMI)
  backends/            execution: numpy (reference, runs every registered variant and emission model),
                       numba and torch (GMM-specialized kernels, explicit capability lists)
  <old module paths>   compatibility re-exports of the pre-1.0 layout (DeprecationWarning on import)

matlab/
  setup_chmm_lsim.m    adds every folder below to the path
  lsim/                LSIM inference and EM; lsim/backward/ one file per backward recursion
  chmm/                full CHMM
  emissions/           emission densities (Gaussian mixtures) and sampling
  common/              parameter packing helpers
  discriminative/      MMI training (+ tests/)
  demos/
```

Model code may import from `core/`, `emissions/`, `learning/` and `backends/`; `core/`, `emissions/` and `learning/` never import a specific model family.

## Compatibility rules (read before changing anything)

- **Public API** = the names in `chmm_lsim.__all__` and `chmm_lsim.discriminative.__all__`, and every MATLAB function outside `tests/`, with its documented arguments and outputs. Everything else is internal.
- `tests/test_public_api.py` compares the public names and signatures with `tests/public_api.json`. Adding names or new *optional keyword* parameters is allowed; removing or renaming a name, reordering or removing parameters, changing a default or adding a required parameter fails the test. After a deliberate, allowed change run `python tests/update_public_api.py` and commit the updated snapshot.
- To rename or retire something, keep the old form working for at least one minor release and make it warn: `chmm_lsim._deprecation.deprecated_alias(new_func, "old_name", since="1.x")` for functions, `moved_module` / `lazy_moved_package` for module paths (see the existing shims and `tests/test_deprecated_paths.py`). MATLAB: keep the old function as a thin wrapper that calls `warning('chmm_lsim:deprecated', ...)`.
- Saved models (`save_lsim_params`) must stay loadable: extend the file format only by adding fields or a new `format_version` that the loader still accepts.
- Every change goes into `CHANGELOG.md` under *Unreleased*; bump the version in `src/chmm_lsim/__init__.py` (`__version__`) when releasing: patch for fixes, minor for additions, major only for removals announced earlier.

## A. A new LSIM inference recursion

**Python.** Create `src/chmm_lsim/models/lsim/inference/<name>.py`:

```python
from chmm_lsim.models.lsim.inference.registry import register_from_emissions, register_inference

@register_from_emissions("<name>")
def <name>_from_emissions(structure, pi_0_raw, transition, coupling, B, B_comp, fb_constants, *,
                          log_scale=None, validate="params") -> Posterior:
    ...   # the recursion, on precomputed emission densities B (S,T), B_comp (S,max_K,T)

@register_inference("<name>")
def forward_backward_<name>(structure, params, Y, fb_constants=None, *, validate="params") -> Posterior:
    B, B_comp = compute_emissions(structure, params, Y)      # any registered emission model
    return <name>_from_emissions(structure, params.pi_0, params.transition, params.coupling, B, B_comp, ...)
```

Import the module in `models/lsim/inference/__init__.py`. That is enough for `FbConstants(variant="<name>")`, `forward_backward`, `forward_backward_from_emissions`, `e_step` and `em_lsim` on the NumPy backend. The forward pass is shared, so the log-likelihood needs no new code unless the recursion changes it (`register_inference(name, loglik=...)`). numba and torch kernels are optional: add the variant to that backend's `BackendCapabilities.variants` only when its kernel exists; otherwise those backends raise `UnsupportedCombination`.

**MATLAB.** Create `matlab/lsim/backward/lsim_backward_<name>.m`:

```matlab
function [beta, alpha_T] = lsim_backward_<name>(alpha, b_c_ot_nc, A, theta, layout, fb_constants)
% alpha      N x T one-step predictive marginals (N = sum(M)); alpha .* b_c_ot_nc are the filtered ones
% A, theta   packed N x N transitions and C x C coupling
% layout     .C .T .state_numbers_index .channel_num_states .mat_mult_one .vec_mat_mult .coupling_incoming
% returns beta and alpha_T (N x T, alpha_T normalized per channel block; em_lsim uses beta(:,t+1))
```

`lsim_fb_constants('<name>')` accepts it as soon as the file is on the path, and every entry point (`forward_backward_lsim`, `em_lsim`, `viterbi_lsim`, the discriminative trainer) can use it. Add new constants as name/value options of `lsim_fb_constants`.

**Tests.** Export MATLAB reference outputs (posteriors, log-likelihood, a few EM iterations) to `tests/fixtures/`, and compare the Python recursion with them (see `tests/test_inference_poise.py`). Add the variant to the README table.

## B. A new emission model (e.g. AR-LSIM)

**Python.** Create `src/chmm_lsim/emissions/<name>.py` with a per-channel parameter class and a subclass of `chmm_lsim.emissions.base.EmissionModel`, then `register_emission_model(<Model>())`. The interface (see `emissions/base.py` and the Gaussian-mixture implementation `emissions/gmm.py`):

| Method | Used for |
|---|---|
| `check_shapes`, `validate`, `copy` | parameter construction and validation |
| `densities(structure, emissions, Y)` | `B` (S,T) and `B_comp` (S,max_K,T) for a whole sequence (so AR models can use past observations) |
| `sample(rng, emission, state, *, past)` | simulation (receives the observations generated so far) |
| `new_stats`, `accumulate`, `merge_stats`, `finalize_stats`, `m_step` | EM sufficient statistics and the emission M-step (AR: weighted least squares) |
| `global_statistics`, `apply_global_statistics` | optional whole-dataset statistics for mini-batch EM |
| `n_free_parameters` | AIC/BIC |
| `check_init_options`, `initialize` | initialization from data |
| `to_arrays`, `from_arrays`, `channel_array_shapes`, `read_global_arrays` | save/load |

LSIM and CHMM parameters hold a list of per-channel emission objects; the model is found from their class (`emission_model_for`), so `em_lsim`, `em_chmm`, `forward_backward`, `forward_backward_chmm`, `simulate_*` and Viterbi work with the new model on the NumPy backend without changes. The structural (transition/coupling) half of the LSIM M-step is emission-independent (`models.lsim.em.structural_m_step`), and `forward_backward_from_emissions` runs any registered recursion on densities computed elsewhere. The numba/torch backends and the discriminative trainer are Gaussian-mixture specific and reject other emission models; extend them separately if needed. Add a public initializer (e.g. `initialize_<name>_params`) rather than changing the signature of `initialize_lsim_params`.

**MATLAB.** Put the density and sampling functions in `matlab/emissions/` (same contract: per channel, state and component `B_comp = P(k|i) p(y|i,k)`, `B = sum_k B_comp`, missing dimensions marginalized). In MATLAB the emission M-step, initialization and parameter packing are still written for Gaussian mixtures inside `em_lsim`/`em_chmm`, `lsim_pack_para` and the kernel's density call, so a new emission family needs its own EM entry point (e.g. `em_ar_lsim`) that reuses `forward_backward_lsim_kernel`'s recursions on its densities, or a refactor of that M-step into an emission-specific function.

## C. A new model family (e.g. higher-order LSIM)

**Python.** Create `src/chmm_lsim/models/<name>/` following `models/chmm/`:

- `params.py`: a parameter dataclass holding `structure`, the transition parameters and `emissions` (a list of per-channel emission objects, validated through `check_emissions_match`/`validate_emissions`);
- `inference.py`: forward-backward (reuse `core.joint_state` for exact small-model inference, or the LSIM recursions where they apply) and log-likelihood;
- `em.py`: an `iteration(params)` function (E-step: accumulate emission statistics with the emission-model helpers; M-step: transitions + `m_step_emissions`) passed to `learning.em_loop._run_em_loop`, so the new `em_<name>` returns the same `EmResult` (traces, best iterate, convergence rule) as `em_lsim` and `em_chmm`;
- `simulate.py`, `viterbi.py`, `init.py` as needed.

Export the public names from `src/chmm_lsim/__init__.py`, update the public-API snapshot, and add tests (MATLAB references when a MATLAB version exists, plus brute-force checks on tiny models).

**MATLAB.** Create `matlab/<name>/` with `em_<name>.m`, `forward_backward_<name>.m`, ... and a `Contents.m`; follow `em_lsim`'s options, `init_model` argument and output order (`[pi_0, ..., gmm, AIC, log_likelihood, BIC, pi_steady]`, as `em_chmm` does) so users can switch models without changing their code. Add the folder to `setup_chmm_lsim.m`.

## Checklist for any addition

1. Code in the place given above; no edits to unrelated model families.
2. Tests: MATLAB parity fixtures where a MATLAB version exists, independent checks (brute force, monotone EM), and the full suite on Python 3.10–3.13 (`pytest`) and MATLAB.
3. Public API: `python tests/update_public_api.py` if you added public names; never remove or change existing ones without a deprecation period.
4. `CHANGELOG.md` (*Unreleased*), README table, demo if useful.
5. Release: bump `__version__`, move *Unreleased* to the new version, tag the release.
