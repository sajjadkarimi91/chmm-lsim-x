# Changelog

All notable changes to chmm-lsim-x are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **patch** (1.0.x): bug fixes only; no change to the public API;
- **minor** (1.x.0): new models, functions, options or backends; existing code keeps working unchanged;
- **major** (x.0.0): the only releases that may remove or change public API, and only after at least one minor release in which the old form still works and issues a `DeprecationWarning` (MATLAB: a `warning`).

The public API is every name exported by `chmm_lsim` (its `__all__`) and by `chmm_lsim.discriminative`, and every MATLAB function in `matlab/` (outside `tests/`), with its documented arguments and outputs. Everything else (Python submodules such as `chmm_lsim.models.lsim.em`) is internal and may be reorganized in any release. Models saved with `save_lsim_params` stay loadable by later versions.

## [Unreleased]

## [1.0.0]

First public release, accompanying the POISE paper.

### Added
- **LSIM inference** with three backward recursions: POISE (the default), SED and fast/SVI, in MATLAB (`forward_backward_lsim`, `lsim_fb_constants`) and Python (`forward_backward`, `FbConstants`); NumPy, numba and PyTorch (CPU/CUDA, batched) backends in Python.
- **LSIM learning:** maximum-likelihood EM (`em_lsim`), mini-batch training for large datasets (`em_lsim_minibatch_train`), parameter initialization, simulation, approximate Viterbi decoding (`viterbi_lsim`), exact inference for small models, model save/load.
- **Full CHMM:** EM (`em_chmm`), exact forward-backward (`forward_backward_chmm`), exact Viterbi (`viterbi_chmm`) and simulation, in MATLAB and Python, with outputs compatible with `em_lsim` (MATLAB: the same output order without the coupling matrix; Python: the same `EmResult`).
- **Discriminative (MMI) learning** of class-conditional LSIMs: `em_lsim_discriminative`, `lsim_discriminative_predict` (MATLAB and Python).
- Runtime benchmark of the backward recursions (`benchmarks/`), demos in MATLAB and Python, tests checked against MATLAB reference outputs.
