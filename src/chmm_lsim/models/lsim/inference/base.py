"""Posterior: the output of one forward-backward pass."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure


@dataclass
class Diagnostics:
    """Conditioning counters, not errors -- populated whenever they're cheap to
    compute, regardless of `validate` mode (§0.3.3(4))."""

    zero_alpha_next_count: int = 0
    beta_floor_hits: int = 0
    f_le_h_violations: int = 0
    nan_guard_hits: int = 0


@dataclass
class Posterior:
    loglik: float  # P_O_model -- always the true value, rescaling folded back in
    alpha: NDArray[np.float64]  # (S,T) filtered-predicted
    beta: NDArray[np.float64]  # (S,T)
    gamma: NDArray[np.float64]  # (S,T) == alpha_T, per-channel smoothed marginals
    scale: NDArray[np.float64]  # (S,T) == b_c_ot_nc -- invariant to emission rescaling
    B: NDArray[np.float64]  # (S,T) == P_observ_cond_to_state, on the stored scale
    B_comp: NDArray[np.float64]  # (S,Kmax,T) == P_observ_cond_to_state_comp, same scale
    log_scale: NDArray[np.float64] | None  # (C,T) per-(channel,t) log offset; None if not rescaled
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    structure: LSIMStructure | None = None  # kept for .channel()'s state slicing and .B_raw()'s gather

    def channel(self, c: int) -> Posterior:
        """A cheap view restricted to channel `c`'s states -- the counterpart
        of the MATLAB cell output for that channel (`forward_backward_lsim.m`'s
        `*_out{zee,1}` cells). `loglik`/`diagnostics` are model-wide quantities
        and are carried through unchanged.

        Builds a single-channel `structure`, not a slice of the original one:
        `B_raw()` gathers `log_scale` through `structure.block_of_state`, and
        that array must stay in step with the *sliced* `log_scale` row below
        (index 0, not `c`) or every state past the first raises an
        `IndexError` -- keeping the original C-channel structure here was
        exactly that bug.

        `c` must be a valid non-negative channel index. Python's negative
        indexing is deliberately not honoured: `self.structure.state_slice(-1)`
        and `channel_state_num[-1]` disagree about *which* channel "-1" means
        (the former is an empty slice at the end of `state_offsets`, the latter
        MATLAB-style "last channel") -- silently returning one while labelling
        it with the other's size is worse than rejecting it outright."""
        if not (0 <= c < self.structure.C):
            raise IndexError(f"channel index {c} out of range for C={self.structure.C} (must be 0 <= c < C)")
        sl = self.structure.state_slice(c)
        K_c = self.structure.num_gmm_component[c]
        sub_structure = LSIMStructure(
            channel_state_num=(self.structure.channel_state_num[c],),
            channel_obs_dim=(self.structure.channel_obs_dim[c],),
            num_gmm_component=(K_c,),
        )
        return Posterior(
            loglik=self.loglik,
            alpha=self.alpha[sl],
            beta=self.beta[sl],
            gamma=self.gamma[sl],
            scale=self.scale[sl],
            B=self.B[sl],
            B_comp=self.B_comp[sl][:, :K_c, :],  # global Kmax -> this channel's own K_c, cf. forward_backward_lsim.m:86
            log_scale=self.log_scale[c : c + 1] if self.log_scale is not None else None,
            diagnostics=self.diagnostics,
            structure=sub_structure,
        )

    def B_raw(self) -> NDArray[np.float64]:
        """Reconstructs the MATLAB-scale `P_observ_cond_to_state` when
        emission rescaling (§2.5) shifted `B`; a no-op when it didn't."""
        if self.log_scale is None:
            return self.B
        return self.B * np.exp(self.log_scale[self.structure.block_of_state])

    def B_comp_raw(self) -> NDArray[np.float64]:
        """`B_raw()`'s counterpart for the per-component array: reconstructs
        the MATLAB-scale `P_observ_cond_to_state_comp` (S,Kmax,T) when
        emission rescaling (§2.5) shifted `B_comp`; a no-op when it didn't.
        Same per-(state,t) offset as `B_raw()`, broadcast across the middle
        (component) axis."""
        if self.log_scale is None:
            return self.B_comp
        return self.B_comp * np.exp(self.log_scale[self.structure.block_of_state])[:, None, :]
