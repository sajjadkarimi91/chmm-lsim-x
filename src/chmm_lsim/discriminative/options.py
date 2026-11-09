"""Validated options of the discriminative (MMI) trainer.

Python port of ``matlab/discriminative/lsim_disc_options.m``. The objective is maximum mutual
information (``objective='mmi'``) with the published update (``mmi_recipe='published'``,
Woodland & Povey 2002). Every unknown or invalid option raises :class:`OptionError` -- options
are never silently ignored.

Defaults follow MATLAB. ``fb_constants`` defaults to raw POISE statistics (``variant='poise'``,
no refinement), the fastest smoother; the forward scores are identical for every variant, and
``sed`` and ``fast`` remain available.

Compute options (``engine``, ``n_jobs``, ``devices``, ``chunk_size``, ``numpy_backend``),
stochastic mini-batches (``minibatches``, ``minibatch_seed``) and long-run support
(``state_path``, ``log_path``, ``save_every``) are Python-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable

import numpy as np

from chmm_lsim.compat import FbConstants

BLOCK_FAMILIES = ("pi", "A", "theta", "w", "mu", "var")


class OptionError(ValueError):
    """An unknown or invalid option."""


@dataclass(frozen=True)
class DiscOptions:
    # -- common ------------------------------------------------------------------------
    objective: str = "mmi"
    log_prior: np.ndarray | None = None
    update_blocks: tuple[str, ...] = ()
    prob_floor: float = 1e-4
    mean_margin: float = 0.5
    var_lo_frac: float = 1e-3
    var_hi_frac: float = 10.0
    # -- validation: checkpoint selection and early stopping (Python only) ------------------
    selection_metric: str = "balanced_accuracy"   # 'balanced_accuracy' (argmax BA, MATLAB) | 'auroc' |
                                                  # 'auprc' | 'max_f1' | 'ba_best_threshold' | 'objective'
    positive_class: int = 1                       # two classes: the detection (positive) class
    early_stopping_patience: int | None = None    # stop after this many iterations without an improvement
    early_stopping_min_delta: float = 0.0         # > min_delta of the selection metric (None = off)
    fb_constants: FbConstants = field(default_factory=lambda: FbConstants(variant="poise", poise_refinement=False))
    iteration_hook: Callable | None = None
    verbose: int = 1
    # -- compute (Python only) -------------------------------------------------------------
    engine: str = "numpy"              # 'numpy' (CPU, optional process pool) | 'torch' (CPU/GPU)
    n_jobs: int = 1                    # numpy engine: worker processes
    numpy_backend: str = "auto"        # numpy engine kernel: 'numba' | 'numpy' | 'auto'
    devices: tuple[str, ...] = ("cuda",)  # torch engine: one or more devices ('cuda:0', 'cuda:1', 'cpu')
    chunk_size: int | None = None      # sequences per work unit (torch: per device batch); None = auto
    # -- stochastic mini-batches (Python only) ------------------------------------------
    minibatches: int = 1               # K > 1: K class-stratified updates per epoch (iteration = epoch)
    minibatch_seed: int = 0
    # -- long runs (Python only) -------------------------------------------------------------
    state_path: str | None = None      # resumable training state (written atomically)
    log_path: str | None = None        # JSON-lines convergence log, one line per iteration
    save_every: int = 1
    # -- MMI (published recipe) ---------------------------------------------------------
    mmi_recipe: str = "published"
    score_scale: float = 1.0           # kappa in q = softmax(kappa L + log pi)
    ebw_E: float = 2.0                 # fixed EBW damping multiplier E of the Gaussian update
    ismooth_tau: float = 100.0         # I-smoothing pseudo-counts (frame units)
    ismooth_backoff: str = "ml"        # I-smoothing back-off: 'ml' | 'previous'
    ismooth_tau_discrete: float = 0.0  # tau_d * p_ML added to the discrete numerator counts
    weak_sense_iters: int = 100        # inner weak-sense iterations of the discrete updates
    weak_sense_tol: float = 1e-10


_COMMON = {
    "objective", "log_prior", "update_blocks", "prob_floor", "mean_margin", "var_lo_frac", "var_hi_frac",
    "selection_metric", "positive_class", "early_stopping_patience", "early_stopping_min_delta", "fb_constants",
    "iteration_hook", "verbose", "engine", "n_jobs", "numpy_backend", "devices", "chunk_size", "minibatches",
    "minibatch_seed", "state_path", "log_path", "save_every",
}
_MMI = {"mmi_recipe", "score_scale", "ebw_E", "ismooth_tau", "ismooth_backoff", "ismooth_tau_discrete",
        "weak_sense_iters", "weak_sense_tol"}

# MATLAB options without a Python counterpart: name -> what to do instead
_MATLAB_ONLY = {
    "kappa": "the MMI score scale is score_scale",
    "parallel": "use engine / n_jobs / devices",
    "validation_observations": "pass validation data to em_lsim_discriminative(validation=...)",
    "generative_max_iteration": "pass init_models (generative initialization is not part of the Python trainer)",
    "generative_extra": "pass init_models (generative initialization is not part of the Python trainer)",
}
_MATLAB_LEGACY = ("tol_rel", "patience", "damping_E", "damping_E_min", "damping_E_max", "max_backtracks")


def make_options(**kw: Any) -> DiscOptions:
    """Validated :class:`DiscOptions` from keyword options (MATLAB ``extra`` fields)."""
    for k in kw:
        if k in _MATLAB_ONLY:
            raise OptionError(f"option {k!r} is not available in Python: {_MATLAB_ONLY[k]}")
        if k in _MATLAB_LEGACY:
            raise OptionError(f"option {k!r} belongs to mmi_recipe 'legacy', which is available only in the "
                              "MATLAB trainer")
    objective = kw.get("objective", "mmi")
    if objective != "mmi":
        raise OptionError(f"objective must be 'mmi', got {objective!r}")
    for k in kw:
        if k not in _COMMON and k not in _MMI:
            raise OptionError(f"unknown option {k!r}")
    if "update_blocks" in kw:
        kw["update_blocks"] = tuple(kw["update_blocks"] or ())
    if "devices" in kw:
        dv = kw["devices"]
        kw["devices"] = (dv,) if isinstance(dv, str) else tuple(dv)
    if "log_prior" in kw and kw["log_prior"] is not None:
        kw["log_prior"] = np.asarray(kw["log_prior"], dtype=np.float64).ravel()
    o = DiscOptions(**kw)
    _check(o)
    return o


def _check(o: DiscOptions) -> None:
    def bad(msg: str) -> None:
        raise OptionError(msg)

    if not set(o.update_blocks) <= set(BLOCK_FAMILIES):
        bad(f"update_blocks must be a subset of {BLOCK_FAMILIES}, got {o.update_blocks}")
    if not (0 < o.prob_floor < 1):
        bad("prob_floor must be in (0, 1)")
    if o.mean_margin < 0 or o.var_lo_frac < 0 or not (o.var_hi_frac > o.var_lo_frac):
        bad("need mean_margin >= 0 and 0 <= var_lo_frac < var_hi_frac")
    from chmm_lsim.discriminative.metrics import SELECTION_METRICS

    if o.selection_metric not in SELECTION_METRICS:
        bad(f"selection_metric must be one of {SELECTION_METRICS}")
    if o.early_stopping_patience is not None and o.early_stopping_patience < 1:
        bad("early_stopping_patience must be None or >= 1")
    if o.early_stopping_min_delta < 0:
        bad("early_stopping_min_delta must be >= 0")
    if o.fb_constants.variant not in ("poise", "sed", "fast"):
        bad("fb_constants.variant must be 'poise' (default, raw POISE: fastest), 'sed' or 'fast'")
    if o.engine not in ("numpy", "torch"):
        bad("engine must be 'numpy' or 'torch'")
    if o.numpy_backend not in ("auto", "numba", "numpy"):
        bad("numpy_backend must be 'auto', 'numba' or 'numpy'")
    if o.n_jobs < 1 or (o.chunk_size is not None and o.chunk_size < 1):
        bad("n_jobs and chunk_size must be >= 1")
    if not o.devices:
        bad("devices must name at least one device")
    if o.minibatches < 1 or o.save_every < 1:
        bad("minibatches and save_every must be >= 1")
    if o.mmi_recipe == "legacy":
        bad("mmi_recipe 'legacy' is available only in the MATLAB trainer")
    if o.mmi_recipe != "published":
        bad("mmi_recipe must be 'published'")
    if not (o.score_scale > 0):
        bad("score_scale must be > 0")
    if o.ebw_E <= 0 or o.ismooth_tau < 0 or o.ismooth_tau_discrete < 0:
        bad("need ebw_E > 0, ismooth_tau >= 0, ismooth_tau_discrete >= 0")
    if o.ismooth_backoff not in ("ml", "previous"):
        bad("ismooth_backoff must be 'ml' or 'previous'")


def default_max_iteration(o: DiscOptions) -> int:
    """MATLAB default for the published MMI recipe: 8 iterations."""
    return 8


def options_to_dict(o: DiscOptions) -> dict:
    """Plain, picklable summary (no callables) for logs and saved state."""
    out = {}
    for f in fields(o):
        v = getattr(o, f.name)
        if f.name == "iteration_hook":
            continue
        if f.name == "fb_constants":
            v = {"variant": v.variant, "beta_floor": v.beta_floor, "pi0_reg": v.pi0_reg}
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


__all__ = ["BLOCK_FAMILIES", "DiscOptions", "OptionError", "default_max_iteration", "make_options",
           "options_to_dict", "replace"]
