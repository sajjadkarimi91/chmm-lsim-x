"""`@register_inference("name")` / `get_inference(name)`: the lookup table
mapping an `FbConstants.variant` string to its forward-backward kernel, plus
`forward_backward(...)`, the variant-dispatching counterpart of calling
`forward_backward_sed`/`forward_backward_fast` directly (§2.2/§2.3).

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
from chmm_lsim.compat import FbConstants
from chmm_lsim.inference.base import Posterior
from chmm_lsim.params import LSIMParams
from chmm_lsim.structure import LSIMStructure

InferenceKernel = Callable[..., Posterior]

_REGISTRY: dict[str, InferenceKernel] = {}


def register_inference(name: str) -> Callable[[InferenceKernel], InferenceKernel]:
    """Class/function decorator: `@register_inference("fast")` on a
    `forward_backward_<variant>`-shaped callable. Raises if `name` is already
    registered -- re-registering a variant (e.g. two modules both claiming
    `"sed"`) is a programming error, not a valid override."""

    def decorator(fn: InferenceKernel) -> InferenceKernel:
        if name in _REGISTRY:
            raise ValueError(f"inference variant {name!r} is already registered (registered variants are {sorted(_REGISTRY)})")
        _REGISTRY[name] = fn
        return fn

    return decorator


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
