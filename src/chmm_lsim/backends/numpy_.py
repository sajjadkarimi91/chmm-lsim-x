"""NumPy reference execution primitives (float64). These are the array
operations `chmm_lsim.inference.sed` is written in terms of -- swapping this module
for a numba/torch equivalent is the point of the axis split (§2 of the plan);
none of the LSIM algorithm itself lives here.

Also registers the `"numpy"` execution backend (P7, `backends/base.py`) --
its runner is a thin pass-through to `inference.registry.get_inference`,
imported lazily inside the runner's own body rather than at this module's
top level: `inference/sed.py`/`inference/fast.py` both import primitives
from *this* module already (at their own top level), so importing
`inference.registry` back from here at import time would work today, but
would make this module's import graph depend on load order rather than
being self-contained -- the lazy import sidesteps that fragility for a cost
that only matters once, on the first actual `backend="numpy"` call.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import BackendCapabilities, register_backend


def block_sum(x: NDArray, offsets: NDArray, axis: int = 0) -> NDArray:
    """Sum contiguous blocks of `x` along `axis`, where block `c` spans
    `offsets[c]:offsets[c+1]`. `offsets` is a structure's `state_offsets` or
    `dim_offsets` -- `(C+1,)`, starting at 0."""
    return np.add.reduceat(x, offsets[:-1], axis=axis)


def broadcast_block(v: NDArray, block_of: NDArray, axis: int = 0) -> NDArray:
    """Inverse of `block_sum`: broadcast a per-channel value back over states
    (or dims), via `block_of_state`/`block_of_dim`."""
    return np.take(v, block_of, axis=axis)


def build_coupling_gather(coupling: NDArray, block_of_row: NDArray, block_of_col: NDArray | None = None) -> NDArray:
    """`Ccup[i,j] = coupling[block_of_row[i], block_of_col[j]]` (§0.2) -- an
    `(S,S)` gather built once per call, reused by every timestep."""
    if block_of_col is None:
        block_of_col = block_of_row
    return coupling[np.ix_(block_of_row, block_of_col)]


def build_W(transition: NDArray, coupling: NDArray, block_of_state: NDArray) -> NDArray:
    """`W[i,j] = transition[i,j] * coupling[block(i), block(j)]` (§0.2): the
    single `(S,S)` matrix that drives both the forward and backward recursion."""
    return transition * build_coupling_gather(coupling, block_of_state)


def _run_numpy(variant, structure, params, Y, fb_constants, *, validate="params", dtype="float64", **kwargs):
    # `dtype` is forwarded by `backends.base.run_backend` uniformly to every
    # backend (P8a: the torch backend needs it); `check_capability` already
    # guarantees `dtype in caps.dtypes` before this runner is ever reached,
    # and `("float64",)` is this backend's only declared dtype, so accepting
    # (and discarding) it here is a no-op, not new behavior.
    from chmm_lsim.inference.registry import get_inference

    kernel = get_inference(variant)
    return kernel(structure, params, Y, fb_constants, validate=validate, **kwargs)


def _loglik_numpy(variant, structure, params, Y, fb_constants, *, validate="params"):
    """Forward-only log-likelihood: the posterior kernels' own validation,
    emissions, pi_0 regularization and `_forward_pass` (shared by `sed` and
    `fast`), stopping before the backward recursion. Bit-identical to
    `forward_backward_sed/fast(...).loglik`."""
    from chmm_lsim.emissions import compute_emissions
    from chmm_lsim.inference.sed import _forward_pass, _regularize_pi0
    from chmm_lsim.validate import check_structure_matches, validate_observations

    if fb_constants.variant != variant:
        raise ValueError(f"loglik runner requires fb_constants.variant={variant!r}, got {fb_constants.variant!r}")
    if validate not in ("params", "off"):
        # "all" is served by the full posterior runner (backends.base.run_backend_loglik)
        raise ValueError(f"forward-only loglik supports validate='params' or 'off', got {validate!r}")
    check_structure_matches(structure, params)
    if validate != "off":
        params.validate()
        validate_observations(structure, Y)
    B, _ = compute_emissions(structure, params, Y)
    W = build_W(params.transition, params.coupling, structure.block_of_state)
    pi_0 = _regularize_pi0(params.pi_0, structure, fb_constants.pi0_reg)
    _, _, loglik = _forward_pass(pi_0, W, B, structure, validate=validate)
    return loglik


register_backend(
    BackendCapabilities(
        name="numpy",
        variants=("sed", "fast", "poise"),
        dtypes=("float64",),
        modes=("posterior", "loglik"),
        batched=False,
        gpu=False,
        sed_methods=("optimized", "reference"),
    ),
    _run_numpy,
    loglik_runner=_loglik_numpy,
)
