"""Inference: the E-step math (§2's axis split). Execution strategy lives in
`chmm_lsim.backends`; this package only knows the LSIM algorithm itself.

Importing this package (directly, or via any `chmm_lsim.inference.<submodule>`
import, since Python always initializes a package's `__init__.py` before its
submodules) registers every variant below with `registry.py`'s
`@register_inference` -- by the time `get_inference`/`forward_backward` can be
called at all, raw `"poise"` (the default), `"sed"`, and `"fast"` are already
all in the table."""

from chmm_lsim.inference.base import Diagnostics, Posterior
from chmm_lsim.inference.fast import forward_backward_fast
from chmm_lsim.inference.poise import forward_backward_poise
from chmm_lsim.inference.registry import forward_backward, get_inference, register_inference
from chmm_lsim.inference.sed import forward_backward_sed, reconstruct_sed_diagnostics

__all__ = [
    "Posterior",
    "Diagnostics",
    "register_inference",
    "get_inference",
    "forward_backward",
    "forward_backward_sed",
    "forward_backward_fast",
    "forward_backward_poise",
    "reconstruct_sed_diagnostics",
]
