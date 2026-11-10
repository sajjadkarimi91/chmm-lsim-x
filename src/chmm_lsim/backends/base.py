"""Execution-backend capability layer (P7, §2): resolves *where* an
inference kernel runs, a deliberately separate axis from
`inference/registry.py`'s *variant* axis (*which* algorithm -- `sed` or
`fast`). A backend declares, via `BackendCapabilities`, exactly which
variants/dtypes/modes/batching it supports; `run_backend` checks a request
against that declaration and raises `UnsupportedCombination` -- naming both
the backend and the unsupported axis -- rather than ever silently running
the request on a different backend (e.g. falling back to NumPy) or ignoring
an unsupported option.

**Lazy, optional backends.** `numpy` is always available (a hard
dependency); `numba` is optional (`pip install 'chmm-lsim[numba]'`) and is never
imported unless a caller actually requests `backend="numba"` -- `import
lsim` must stay importable without it. This module itself imports neither
`chmm_lsim.inference.*` nor `numba` at module level: each backend module
(`backends/numpy_.py`, `backends/numba_.py`) registers itself via
`register_backend` when *it* is imported, and its registered runner is free
to import whatever it actually needs lazily, inside the runner's own body,
at call time -- avoiding both an import cycle with `inference/registry.py`
(which imports `run_backend` from here) and a hard numba dependency here.
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from chmm_lsim.core.validate import check_structure_matches

if TYPE_CHECKING:
    # Only for annotations (deferred by `from __future__ import annotations`
    # above) -- a real, non-guarded import here would trigger
    # `chmm_lsim.inference`'s own package `__init__.py` (which eagerly imports
    # `inference.sed`/`inference.fast`, both of which import
    # `backends.numpy_`) before this module has finished executing,
    # producing a circular-import error the moment anything imports
    # `backends.base` before `backends.numpy_` has fully loaded.
    from chmm_lsim.models.lsim.inference.base import Posterior


class UnsupportedCombination(ValueError):
    """A requested (backend, variant, dtype, mode, batched, ...) combination
    isn't declared supported by that backend. Never caught internally to
    fall back to another backend -- always propagates to the caller."""


class MissingBackendDependency(ImportError):
    """A backend was requested whose optional dependency isn't installed
    (currently only `numba`). Carries an actionable install instruction in
    its message; never silently substitutes another backend."""


@dataclass(frozen=True)
class BackendCapabilities:
    """What one execution backend declares it supports. `variants`/`dtypes`/
    `modes` are the axes `run_backend` checks against; `sed_methods` is a
    narrower, sed-specific declaration (the `method=` kwarg of
    `forward_backward_sed`), since a backend can support `sed` overall while
    only implementing one of its two `P_cond`-construction methods (P7's
    numba backend implements `"optimized"` only, not `"reference"`).

    `emission_models` names the emission models (`chmm_lsim.emissions`) the
    backend's kernels can evaluate. A backend built only on the generic
    Python entry points (the NumPy one) sets `any_variant=True` /
    `any_emission_model=True`: it then runs every registered inference
    variant / emission model, and `get_backend` reports the currently
    registered ones in `variants` / `emission_models`."""

    name: str
    variants: tuple[str, ...]
    dtypes: tuple[str, ...] = ("float64",)
    modes: tuple[str, ...] = ("posterior",)
    batched: bool = False
    gpu: bool = False
    sed_methods: tuple[str, ...] = ("optimized", "reference")
    _: dataclasses.KW_ONLY
    emission_models: tuple[str, ...] = ("gmm",)
    any_variant: bool = False
    any_emission_model: bool = False


Runner = Callable[..., "Posterior"]  # string forward-ref: never resolved at runtime, see the TYPE_CHECKING import above

_BACKENDS: dict[str, BackendCapabilities] = {}
_RUNNERS: dict[str, Runner] = {}
LoglikRunner = Callable[..., float]
_LOGLIK_RUNNERS: dict[str, LoglikRunner] = {}

# Every backend name this module knows *how* to load lazily, without having
# imported its (possibly heavy, possibly missing) module yet. Loading is
# just "import the module" -- each backend module calls `register_backend`
# itself, at its own module level, as a side effect of being imported.
_LAZY_MODULES: dict[str, str] = {
    "numpy": "chmm_lsim.backends.numpy_",
    "numba": "chmm_lsim.backends.numba_",
    "torch": "chmm_lsim.backends.torch_batched",
}


def register_backend(caps: BackendCapabilities, runner: Runner, *, loglik_runner: "LoglikRunner | None" = None) -> None:
    """`backends/numpy_.py`/`backends/numba_.py`/`backends/torch_batched.py`
    call this once, at their own module level, to register themselves. Raises
    if `caps.name` is already registered -- re-registering a backend is a
    programming error, not a valid override (same policy as
    `inference.registry.register_inference`).

    Declaring `"loglik"` in `caps.modes` and supplying `loglik_runner` must go
    together: a backend can neither advertise forward-only log-likelihood
    without a runner nor register a runner without the capability. Both
    registries are updated only after every check passes, so a rejected
    registration leaves no partial state behind."""
    if caps.name in _BACKENDS or caps.name in _LOGLIK_RUNNERS:
        raise ValueError(f"execution backend {caps.name!r} is already registered (registered backends are {sorted(_BACKENDS)})")
    declares_loglik = "loglik" in caps.modes
    if declares_loglik and loglik_runner is None:
        raise ValueError(f"backend {caps.name!r} declares mode 'loglik' but registers no loglik_runner")
    if loglik_runner is not None and not declares_loglik:
        raise ValueError(f"backend {caps.name!r} registers a loglik_runner but does not declare mode 'loglik' in its capabilities")
    _BACKENDS[caps.name] = caps
    _RUNNERS[caps.name] = runner
    if loglik_runner is not None:
        _LOGLIK_RUNNERS[caps.name] = loglik_runner


def _ensure_loaded(name: str) -> None:
    if name in _BACKENDS:
        return
    module_path = _LAZY_MODULES.get(name)
    if module_path is None:
        known = sorted(set(_BACKENDS) | set(_LAZY_MODULES))
        raise UnsupportedCombination(f"unknown execution backend {name!r}; known backends are {known}")
    importlib.import_module(module_path)  # self-registers via register_backend; raises MissingBackendDependency if optional deps are absent
    if name not in _BACKENDS:
        raise UnsupportedCombination(  # pragma: no cover -- defensive; would indicate a backend module that forgot to self-register
            f"execution backend {name!r} was loaded but never called register_backend -- this is a bug in {module_path!r}"
        )


def get_backend(name: str) -> BackendCapabilities:
    """Looks up a backend's declared capabilities, loading it lazily (see
    module docstring) if it hasn't been imported yet. Raises
    `UnsupportedCombination` for a name that is neither registered nor
    lazily loadable, and lets `MissingBackendDependency` propagate as-is for
    a lazily-loadable backend whose optional dependency isn't installed."""
    _ensure_loaded(name)
    caps = _BACKENDS[name]
    if caps.any_variant or caps.any_emission_model:
        updates = {}
        if caps.any_variant:
            from chmm_lsim.models.lsim.inference.registry import registered_variants

            updates["variants"] = registered_variants()
        if caps.any_emission_model:
            from chmm_lsim.emissions.base import registered_emission_models

            updates["emission_models"] = registered_emission_models()
        caps = dataclasses.replace(caps, **updates)
    return caps


def emission_model_name(params) -> str:
    """Registry name of the emission model of `params`."""
    from chmm_lsim.emissions.base import emission_model_for

    return emission_model_for(params.emissions).name


def get_runner(name: str) -> Runner:
    _ensure_loaded(name)
    return _RUNNERS[name]


def check_capability(
    name: str,
    *,
    variant: str,
    dtype: str = "float64",
    mode: str = "posterior",
    batched: bool = False,
    method: str | None = None,
    emission_model: str | None = None,
) -> BackendCapabilities:
    """Raises `UnsupportedCombination` -- naming the backend and the specific
    unsupported axis -- unless every axis of the request is declared
    supported by backend `name`. This is the single source of truth for
    whether `method=` is valid for a given `(name, variant, method)`: a real
    (non-`None`) `method` value is *only* ever meaningful for `variant=
    "sed"` (`forward_backward_fast` has no `method=` parameter at all) --
    passing one for any other variant raises here, naming the backend,
    variant, and method together, regardless of whether this is being asked
    directly or via `run_backend` (a review round found `run_backend` had
    its *own*, separate copy of half of this rule, so `check_capability("
    numpy", variant="fast", method="reference")` -- and the `numba`
    equivalent -- incorrectly succeeded even though `run_backend` itself
    would have rejected the same request; there is now exactly one place
    this is decided). `method=None` (the default, meaning "no method
    requested") never raises here regardless of `variant`, since `None`
    carries no information about `sed`-specific intent -- see
    `run_backend`'s own docstring for why it separately rejects an
    *explicitly*-passed `method=None` for a non-`sed` variant, a
    distinction only visible at the `**kwargs`-inspecting layer, not here.

    `emission_model` (a registered emission-model name, or `None` for "not
    checked") must be one the backend declares in `emission_models`."""
    caps = get_backend(name)
    if variant not in caps.variants:
        raise UnsupportedCombination(f"backend={name!r} does not support inference variant={variant!r} (supported variants: {caps.variants})")
    if dtype not in caps.dtypes:
        raise UnsupportedCombination(f"backend={name!r} does not support dtype={dtype!r} (supported dtypes: {caps.dtypes})")
    if mode not in caps.modes:
        raise UnsupportedCombination(f"backend={name!r} does not support mode={mode!r} (supported modes: {caps.modes})")
    if batched and not caps.batched:
        raise UnsupportedCombination(f"backend={name!r} does not support batched execution (batched=False only)")
    if emission_model is not None and emission_model not in caps.emission_models:
        raise UnsupportedCombination(
            f"backend={name!r} does not support emission model {emission_model!r} (supported emission models: {caps.emission_models})"
        )
    if method is not None:
        if variant != "sed":
            raise UnsupportedCombination(
                f"backend={name!r} does not support method={method!r} for variant={variant!r} -- method= is only "
                f"meaningful for variant='sed' (forward_backward_fast has no such parameter at all)"
            )
        if method not in caps.sed_methods:
            raise UnsupportedCombination(f"backend={name!r} does not support sed method={method!r} (supported sed methods: {caps.sed_methods})")
    return caps


def run_backend(
    name: str,
    variant: str,
    structure,
    params,
    Y,
    fb_constants,
    *,
    validate: str = "params",
    dtype: str = "float64",
    mode: str = "posterior",
    batched: bool = False,
    **kwargs,
) -> Posterior:
    """Checks the requested combination via `check_capability`, then
    dispatches to the resolved backend's own registered runner. This is the
    single choke point both `inference.registry.forward_backward` and
    `em.e_step` go through -- neither hand-rolls its own backend check, so
    there is exactly one place a new backend or a new axis has to be taught.

    Three checks happen here, *unconditionally* -- before either backend's
    runner (and therefore before either backend's own `validate=` handling)
    ever runs, so no backend can disagree with another on any of them, and
    `validate="off"` cannot skip any of them:

    1. `structure == params.structure` (`check_structure_matches`) -- a
       mismatch here previously produced a *wrong, silent* result on the
       NumPy backend (the requested `structure`'s shapes were used for
       slicing while `params`'s actual array *values* were used underneath),
       while the numba backend already rejected it -- backend choice must
       never change whether a structure/params mismatch is caught.
    2. `variant == fb_constants.variant` -- passing a `fast`-flavored
       `fb_constants` while requesting `variant="sed"` (or vice versa) must
       raise, not silently run whichever one `variant` said, discarding the
       caller's own `fb_constants.variant` -- there is exactly one
       authoritative variant per call, and the two arguments must agree on
       what it is.
    3. `method=` is `sed`-only (`forward_backward_fast` has no such
       parameter at all) -- a real (non-`None`) value is rejected by
       `check_capability` itself now (the single source of truth for this
       rule, callable directly or via here -- a review round found this
       layer used to hold its *own*, separate copy of half the rule, so
       `check_capability(..., variant="fast", method="reference")` alone
       incorrectly succeeded even though a `run_backend` call with the same
       arguments correctly raised). What `check_capability` *cannot* decide
       on its own -- because a plain `method=None` parameter carries no
       information about whether the caller actually typed `method=None`
       or simply never mentioned `method` at all -- is handled here instead:
       an *explicitly* passed `method=None` for a non-`sed` variant is still
       rejected, since `forward_backward_fast` has no `method=` parameter
       to silently accept `None` for either; never silently discarded (the
       numba backend's runner used to pop `method` off unconditionally,
       running `fast` successfully even when `method="reference"` was
       requested -- silently ignoring a request instead of running it or
       rejecting it is exactly the "no silent fallback" failure mode this
       layer exists to prevent)."""
    check_structure_matches(structure, params)
    if fb_constants.variant != variant:
        raise ValueError(
            f"conflicting inference variant: run_backend was asked for variant={variant!r}, but "
            f"fb_constants.variant={fb_constants.variant!r} -- these must agree; there is exactly one "
            f"authoritative variant per call, not one silently overriding the other"
        )
    if variant != "sed" and "method" in kwargs and kwargs["method"] is None:
        raise UnsupportedCombination(
            f"method=None was explicitly passed for variant={variant!r}, but method= is only meaningful for "
            f"variant='sed' (forward_backward_fast has no such parameter at all) -- pass no method= at all for "
            f"'{variant}' rather than an explicit None"
        )
    # `run_backend`/`forward_backward` is *always* the single-sequence,
    # unbatched, `"posterior"`-mode contract -- unconditionally, regardless
    # of what a given backend's own `BackendCapabilities` otherwise declares
    # support for. A review round found this was previously enforced only
    # incidentally: `numpy`/`numba` happened to declare narrow capabilities
    # (at the time `modes=("posterior",)`; they now also declare "loglik",
    # served only through `run_backend_loglik`, never here), so
    # `check_capability` already rejected anything else for *them* -- but `backend="torch"`
    # genuinely supports `mode="loglik"`/`"stats"` and `batched=True`
    # elsewhere (`forward_backward_batch`), so those same requests reached
    # `check_capability` here, passed (correctly, for what `torch` can do in
    # general), and were then silently *ignored* by `_run_torch`, which
    # always hardcodes `mode="posterior"` regardless -- `run_backend("torch",
    # ..., mode="loglik")` returned an ordinary `Posterior` as if `mode=`
    # had never been passed at all. Rejecting these two axes here,
    # unconditionally, before `check_capability` ever runs, closes that gap
    # for every backend, present or future: this single-sequence API is not
    # the batch/mode-aware one -- `backends.torch_batched.forward_backward_batch`
    # is, and takes `mode=`/batching explicitly and checks them for real.
    if mode != "posterior":
        raise UnsupportedCombination(
            f"run_backend/forward_backward is always mode='posterior' (single-sequence) -- got mode={mode!r}. "
            f"Use backends.torch_batched.forward_backward_batch directly for mode='loglik'/'stats' or batched execution."
        )
    if batched:
        raise UnsupportedCombination(
            "run_backend/forward_backward is always unbatched (batched=False) -- got batched=True. "
            "Use backends.torch_batched.forward_backward_batch directly for batched execution."
        )
    check_capability(
        name, variant=variant, dtype=dtype, mode=mode, batched=batched, method=kwargs.get("method"),
        emission_model=emission_model_name(params),
    )
    runner = get_runner(name)
    # `dtype` is forwarded to the runner (P8a: the torch backend genuinely
    # needs to know it); `mode`/`batched` are not (both are fixed to
    # `"posterior"`/`False` by the checks above, so forwarding them would be
    # a no-op) -- for `numpy`/`numba`, `dtype` has already been validated as
    # `"float64"` by `check_capability` above (their only declared dtype),
    # so accepting an always-`"float64"` keyword is a no-op for them too,
    # not a new behavior.
    return runner(variant, structure, params, Y, fb_constants, validate=validate, dtype=dtype, **kwargs)


# --------------------------------------------------------------------------
# Forward-only log-likelihood (mode="loglik"), single sequence.
# --------------------------------------------------------------------------
# A separate entry point rather than a mode of `run_backend`, whose contract is
# (and stays) posterior-only. Registered atomically with the backend itself
# (`register_backend(..., loglik_runner=)`). For validate="params"/"off" the
# runner performs exactly the posterior runner's own validation, emissions,
# pi_0 regularization and forward recursion and returns the same
# `sum(log(p_ot_c))` scalar, skipping the backward recursion. validate="all"
# is never served by a forward-only runner (see run_backend_loglik).


def run_backend_loglik(name: str, variant: str, structure, params, Y, fb_constants, *, validate: str = "params") -> float:
    """Single-sequence log-likelihood on backend `name`.

    `validate="params"`/`"off"`: forward-only (no backward recursion); the
    value equals `run_backend(...).loglik` exactly for numpy/numba, and within
    the approved Torch tolerance for torch.

    `validate="all"`: NOT forward-only. "all" includes backward-side invariant
    checks (gamma/abar/V2/f<=h) that need a backward pass, so this runs the full
    posterior runner via `run_backend(..., validate="all")` and returns its
    `.loglik` -- the complete meaning of "all" is preserved; it is simply not
    the fast path.

    Same unconditional checks as `run_backend` (structure/params match,
    `variant == fb_constants.variant`), then `check_capability(mode="loglik")`."""
    if validate not in ("params", "all", "off"):
        raise ValueError(f"validate must be one of ('params','all','off'), got {validate!r}")
    check_structure_matches(structure, params)
    if fb_constants.variant != variant:
        raise ValueError(
            f"conflicting inference variant: run_backend_loglik was asked for variant={variant!r}, but "
            f"fb_constants.variant={fb_constants.variant!r} -- these must agree"
        )
    check_capability(name, variant=variant, mode="loglik", emission_model=emission_model_name(params))
    if validate == "all":
        return run_backend(name, variant, structure, params, Y, fb_constants, validate="all").loglik
    runner = _LOGLIK_RUNNERS.get(name)
    if runner is None:  # pragma: no cover -- register_backend makes this unreachable for a loglik-declaring backend
        raise UnsupportedCombination(f"backend={name!r} declares mode 'loglik' but has no registered loglik runner")
    return runner(variant, structure, params, Y, fb_constants, validate=validate)
