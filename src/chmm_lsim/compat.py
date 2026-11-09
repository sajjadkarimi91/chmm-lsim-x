"""`FbConstants`: the numerical constants shared by every LSIM forward-backward
and EM entry point (`lsim_fb_constants.m`). One canonical set per variant, no
presets, no historical tuples -- §3bis.4's simplification, carried into Python.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_VARIANTS = ("sed", "fast", "poise")


@dataclass(frozen=True, slots=True)
class FbConstants:
    """`beta_floor`/`f_reg` are `sed`-only, `svi_eps` is `fast`-only and
    `poise_refinement` is `poise`-only; each variant reads only its own, exactly
    as `lsim_fb_constants.m` does (the unused ones are carried, not rejected).

    **`poise_refinement` defaults to `False`, and so does
    `lsim_fb_constants.m`** (P12, after the correction round aligned MATLAB's
    default, its standalone `forward_backward_lsim_poise.m` and `em_lsim_poise.m`
    on the same value). An explicit POISE call in the two languages therefore
    uses the same refinement mode. Python's package-wide no-argument default is
    raw POISE; MATLAB's historical no-argument constants default remains SED.
    The refinement is a second pass over the whole sequence, and its
    weight is `max(0, 0.75 - theta_out)` per channel -- identically zero for any
    channel whose total outgoing coupling strength is at least 0.75, which is
    every channel at the default initialization -- so paying for it by default
    buys nothing on a strongly-coupled model. Every P12 fixture and test still
    states the flag explicitly rather than relying on the default on either
    side."""

    variant: str = "poise"
    beta_floor: float = 1e-9
    pi0_reg: float = 1e-10
    f_reg: float = 1e-6
    svi_eps: float = 1e-8
    poise_refinement: bool = False

    def __post_init__(self) -> None:
        if self.variant not in _VALID_VARIANTS:
            raise ValueError(f"variant must be one of {_VALID_VARIANTS}, got {self.variant!r}")
        if not isinstance(self.poise_refinement, bool):
            raise ValueError(f"poise_refinement must be a bool, got {self.poise_refinement!r}")
