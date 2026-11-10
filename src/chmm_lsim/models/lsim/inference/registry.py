"""`@register_inference("name")` / `get_inference(name)`: the lookup table
mapping an `FbConstants.variant` string to its forward-backward kernel, plus
`forward_backward(...)`, the variant-dispatching counterpart of calling
`forward_backward_sed`/`forward_backward_fast` directly.

Adding an LSIM inference recursion is one module in this package that defines
a `forward_backward_<name>(structure, params, Y, fb_constants, *, validate)`
kernel returning a `Posterior` and decorates it with
`@register_inference("<name>")`, plus an import in this package's
`__init__.py`. `FbConstants(variant="<name>")` then accepts it and the NumPy
backend runs it; compiled backends (numba, torch) declare the variants they
implement in `BackendCapabilities.variants` and reject the others with
`UnsupportedCombination`. Forward-only log-likelihood scoring on NumPy uses
the shared LSIM forward recursion unless the variant registers its own
(`register_inference(name, loglik=...)`).

Deliberately separate from the execution-backend axis (§1.0, §2): this module
only resolves *which algorithm* runs (`sed`, `fast`, ...); it has no opinion on
*where* it runs. `backend=` is checked, and dispatched, entirely by
`backends.base.run_backend` (P7) -- `numpy` and `numba` are both real
execution backends there; anything else (or an unsupported
variant/dtype/mode/batching combination for a real backend) raises
`backends.base.UnsupportedCombination` (a `ValueError` subclass) naming both
axes, never silently falling back to NumPy or ignoring the option.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import run_backend
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.models.lsim.inference.base import Posterior
from chmm_lsim.models.lsim.params import LSIMParams

InferenceKernel = Callable[..., Posterior]
LoglikKernel = Callable[..., float]

_REGISTRY: dict[str, InferenceKernel] = {}
_LOGLIK_REGISTRY: dict[str, LoglikKernel] = {}


def register_inference(name: str, *, loglik: LoglikKernel | None = None) -> Callable[[InferenceKernel], InferenceKernel]:
    """Class/function decorator: `@register_inference("fast")` on a
    `forward_backward_<variant>`-shaped callable. Raises if `name` is already
    registered -- re-registering a variant (e.g. two modules both claiming
    `"sed"`) is a programming error, not a valid override.

    `loglik`, optional: a forward-only log-likelihood kernel with the
    signature of the NumPy backend's loglik runner,
    `loglik(variant, structure, params, Y, fb_constants, *, validate)`, for
    a variant whose forward recursion differs from the shared LSIM one."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"inference variant name must be a non-empty str, got {name!r}")

    def decorator(fn: InferenceKernel) -> InferenceKernel:
        if name in _REGISTRY:
            raise ValueError(f"inference variant {name!r} is already registered (registered variants are {sorted(_REGISTRY)})")
        _REGISTRY[name] = fn
        if loglik is not None:
            _LOGLIK_REGISTRY[name] = loglik
        return fn

    return decorator


_FROM_EMISSIONS_REGISTRY: dict[str, Callable[..., Posterior]] = {}


def register_from_emissions(name: str) -> Callable[[Callable[..., Posterior]], Callable[..., Posterior]]:
    """Decorator registering the emission-independent form of variant `name`:
    `kernel(structure, pi_0, transition, coupling, B, B_comp, fb_constants, *,
    log_scale=None, validate="params", **options) -> Posterior`, used by
    `forward_backward_from_emissions` (and so by any emission model)."""

    def decorator(fn):
        if name in _FROM_EMISSIONS_REGISTRY:
            raise ValueError(f"inference variant {name!r} already has a from-emissions kernel")
        _FROM_EMISSIONS_REGISTRY[name] = fn
        return fn

    return decorator


def get_from_emissions(name: str) -> Callable[..., Posterior]:
    """The from-emissions kernel of variant `name`; `ValueError` if it has none."""
    try:
        return _FROM_EMISSIONS_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"inference variant {name!r} has no registered from-emissions kernel; variants with one are "
            f"{sorted(_FROM_EMISSIONS_REGISTRY)}"
        ) from None


def registered_variants() -> tuple[str, ...]:
    """Every registered inference variant, in registration order (the
    built-in ones are registered when `chmm_lsim` is imported)."""
    import chmm_lsim.models.lsim.inference  # noqa: F401  (registers the built-in variants)

    return tuple(_REGISTRY)


def get_loglik_kernel(name: str) -> LoglikKernel | None:
    """The forward-only log-likelihood kernel registered for `name`, or `None`
    if the variant uses the shared LSIM forward recursion."""
    return _LOGLIK_REGISTRY.get(name)


def get_inference(name: str) -> InferenceKernel:
    """Looks up a registered inference kernel by variant name. Raises
    `ValueError` naming the requested and available variants -- an unknown
    `FbConstants.variant` must fail loudly here, not fall through to whatever
    variant happens to be registered first."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown inference variant {name!r}; registered variants are {sorted(_REGISTRY)}") from None


def forward_backward(
    structure: LSIMStructure,
    params: LSIMParams,
    Y: NDArray[np.float64],
    fb_constants: FbConstants | None = None,
    *,
    backend: str = "numpy",
    validate: str = "params",
    **kwargs,
) -> Posterior:
    """Dispatches to the inference kernel registered for `fb_constants.variant`
    (default: `FbConstants()`'s own default, raw `"poise"`), on the requested
    execution `backend` (default `"numpy"`), via `backends.base.run_backend`.
    `**kwargs` passes through to the resolved kernel -- e.g. `method=`,
    `sed`-only, and (for a given backend) checked against that backend's own
    declared `sed_methods`. An unknown/unsupported `backend`, or a
    variant/backend combination that backend doesn't declare support for,
    raises `backends.base.UnsupportedCombination` (a `ValueError`), naming
    both axes -- never silently substituting NumPy."""
    if fb_constants is None:
        fb_constants = FbConstants()
    return run_backend(backend, fb_constants.variant, structure, params, Y, fb_constants, validate=validate, **kwargs)
