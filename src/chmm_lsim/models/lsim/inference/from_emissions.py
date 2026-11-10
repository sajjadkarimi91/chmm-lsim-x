"""LSIM forward-backward on precomputed emission densities.

`forward_backward_from_emissions` runs a registered recursion on an externally
supplied emission bundle -- `B` (S, T), `B_comp` (S, max_K, T) and an optional
per-(channel, t) log rescaling `log_scale` (C, T) -- instead of evaluating the
model's emissions itself. The GMM entry points (`forward_backward_sed`,
`forward_backward_fast`, `forward_backward_poise`) compute their emissions and
call the same per-variant kernels, so every recursion exists once and any
emission family can reuse every registered variant. The variant is resolved
through the inference registry (`register_from_emissions`).
"""

from __future__ import annotations

import inspect

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.models.lsim.inference.base import Posterior
from chmm_lsim.models.lsim.inference.registry import get_from_emissions


def forward_backward_from_emissions(
    structure: LSIMStructure,
    pi_0: NDArray[np.float64],
    transition: NDArray[np.float64],
    coupling: NDArray[np.float64],
    B: NDArray[np.float64],
    B_comp: NDArray[np.float64],
    fb_constants: FbConstants | None = None,
    *,
    log_scale: NDArray[np.float64] | None = None,
    validate: str = "params",
    **kwargs,
) -> Posterior:
    """Smoothed posteriors of an LSIM with initial distribution `pi_0` (S,),
    transition blocks `transition` (S, S) and coupling `coupling` (C, C), given
    the emission densities `B`/`B_comp` of one sequence (optionally rescaled by
    ``exp(-log_scale)``; ``log_scale.sum()`` is added back to the
    log-likelihood). `fb_constants.variant` selects the recursion; `**kwargs`
    are variant options (e.g. ``method=`` for ``"sed"``)."""
    if fb_constants is None:
        fb_constants = FbConstants()
    v = fb_constants.variant
    kernel = get_from_emissions(v)
    if kwargs:
        params = inspect.signature(kernel).parameters
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        unknown = sorted(k for k in kwargs if k not in params)
        if unknown and not accepts_any:
            raise ValueError(f"unexpected options {unknown} for variant {v!r}")
    return kernel(structure, pi_0, transition, coupling, B, B_comp, fb_constants, log_scale=log_scale, validate=validate, **kwargs)


def forward_loglik_from_emissions(
    structure: LSIMStructure,
    pi_0: NDArray[np.float64],
    transition: NDArray[np.float64],
    coupling: NDArray[np.float64],
    B: NDArray[np.float64],
    fb_constants: FbConstants | None = None,
    *,
    log_scale: NDArray[np.float64] | None = None,
) -> float:
    """Forward-only log-likelihood from precomputed emissions (the forward
    recursion is shared by every built-in variant)."""
    from chmm_lsim.backends.numpy_ import build_W
    from chmm_lsim.models.lsim.inference.sed import _forward_pass, _regularize_pi0

    if fb_constants is None:
        fb_constants = FbConstants()
    W = build_W(transition, coupling, structure.block_of_state)
    pi0 = _regularize_pi0(pi_0, structure, fb_constants.pi0_reg)
    _a, _s, ll = _forward_pass(pi0, W, B, structure, validate="off")
    if log_scale is not None:
        ll = ll + float(np.sum(log_scale))
    return ll
