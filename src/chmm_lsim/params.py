"""LSIMParams: the per-channel (not max-padded) parameter representation,
plus conversions to/from the two
MATLAB-side representations: the padded flat "packed" arrays that
`lsim_pack_para.m` produces (and that every fixture stores), and the
unpadded per-channel "cells" that the MATLAB cell-array API itself uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.structure import LSIMStructure


@dataclass
class ChannelGMM:
    """One channel's GMM emission model, unpadded: exactly (M_c, K_c) etc,
    never `(max_M, max_K)`. See §2.1's "deviation from MATLAB, deliberate"."""

    weights: NDArray[np.float64]  # (M_c, K_c)
    mu: NDArray[np.float64]  # (d_c, M_c, K_c)
    sigma: NDArray[np.float64]  # (d_c, M_c, K_c) diag, or (d_c, d_c, M_c, K_c) full
    diag: bool

    def __post_init__(self) -> None:
        M_c, K_c = self.weights.shape
        if self.mu.shape[1:] != (M_c, K_c):
            raise ValueError(f"mu.shape={self.mu.shape} inconsistent with weights.shape={self.weights.shape}")
        d_c = self.mu.shape[0]
        expected_sigma_shape = (d_c, M_c, K_c) if self.diag else (d_c, d_c, M_c, K_c)
        if self.sigma.shape != expected_sigma_shape:
            raise ValueError(f"sigma.shape={self.sigma.shape} != {expected_sigma_shape} (diag={self.diag})")

    @property
    def d_c(self) -> int:
        return self.mu.shape[0]

    @property
    def M_c(self) -> int:
        return self.weights.shape[0]

    @property
    def K_c(self) -> int:
        return self.weights.shape[1]

    def copy(self) -> ChannelGMM:
        return ChannelGMM(self.weights.copy(), self.mu.copy(), self.sigma.copy(), self.diag)


def check_emissions_match(structure: LSIMStructure, emissions: list[ChannelGMM]) -> None:
    """Raises `ValueError` unless `emissions` has one GMM per channel, each of
    the channel's own `(M_c, K_c, d_c)`, all with the same covariance type."""
    C = structure.C
    if len(emissions) != C:
        raise ValueError(f"len(emissions)={len(emissions)} != C={C}")
    for c, gmm in enumerate(emissions):
        if gmm.M_c != structure.channel_state_num[c]:
            raise ValueError(f"emissions[{c}].M_c={gmm.M_c} != channel_state_num[{c}]={structure.channel_state_num[c]}")
        if gmm.d_c != structure.channel_obs_dim[c]:
            raise ValueError(f"emissions[{c}].d_c={gmm.d_c} != channel_obs_dim[{c}]={structure.channel_obs_dim[c]}")
        if gmm.K_c != structure.num_gmm_component[c]:
            raise ValueError(f"emissions[{c}].K_c={gmm.K_c} != num_gmm_component[{c}]={structure.num_gmm_component[c]}")
    if any(gmm.diag != emissions[0].diag for gmm in emissions):
        raise ValueError("all channels must agree on diagonal-vs-full covariance (packed.sigma_diag is model-wide, not per-channel)")


@dataclass
class LSIMParams:
    structure: LSIMStructure
    pi_0: NDArray[np.float64]  # (S,)
    transition: NDArray[np.float64]  # (S,S)
    coupling: NDArray[np.float64]  # (C,C), columns sum to 1
    emissions: list[ChannelGMM] = field(default_factory=list)

    def __post_init__(self) -> None:
        S, C = self.structure.S, self.structure.C
        if self.pi_0.shape != (S,):
            raise ValueError(f"pi_0.shape={self.pi_0.shape} != (S,)=({S},)")
        if self.transition.shape != (S, S):
            raise ValueError(f"transition.shape={self.transition.shape} != (S,S)=({S},{S})")
        if self.coupling.shape != (C, C):
            raise ValueError(f"coupling.shape={self.coupling.shape} != (C,C)=({C},{C})")
        check_emissions_match(self.structure, self.emissions)

    @property
    def sigma_diag(self) -> bool:
        return self.emissions[0].diag

    def validate(self, *, matlab_compat: bool = True) -> None:
        from chmm_lsim.validate import validate_params

        validate_params(self, matlab_compat=matlab_compat)

    def copy(self) -> LSIMParams:
        return LSIMParams(
            structure=self.structure,
            pi_0=self.pi_0.copy(),
            transition=self.transition.copy(),
            coupling=self.coupling.copy(),
            emissions=[gmm.copy() for gmm in self.emissions],
        )

    # -- packed <-> per-channel (lsim_pack_para.m's flat, max-padded arrays) ---

    @classmethod
    def from_packed(
        cls,
        structure: LSIMStructure,
        *,
        pi_0: NDArray[np.float64],
        transition: NDArray[np.float64],
        coupling: NDArray[np.float64],
        P_all: NDArray[np.float64],
        mu_all: NDArray[np.float64],
        sigma_all: NDArray[np.float64],
        sigma_diag: bool,
    ) -> LSIMParams:
        """Inverse of `lsim_pack_para.m`: slice the max-padded arrays down to
        each channel's own (M_c, K_c, d_c), dropping the zero padding."""
        emissions: list[ChannelGMM] = []
        for c in range(structure.C):
            M_c = structure.channel_state_num[c]
            K_c = structure.num_gmm_component[c]
            dslice = structure.dim_slice(c)
            weights = np.asarray(P_all[c, :M_c, :K_c], dtype=np.float64).copy()
            mu = np.asarray(mu_all[dslice, :M_c, :K_c], dtype=np.float64).copy()
            if sigma_diag:
                sigma = np.asarray(sigma_all[dslice, :M_c, :K_c], dtype=np.float64).copy()
            else:
                sigma = np.asarray(sigma_all[dslice, : structure.channel_obs_dim[c], :M_c, :K_c], dtype=np.float64).copy()
            emissions.append(ChannelGMM(weights, mu, sigma, diag=sigma_diag))
        return cls(
            structure=structure,
            pi_0=np.asarray(pi_0, dtype=np.float64).copy(),
            transition=np.asarray(transition, dtype=np.float64).copy(),
            coupling=np.asarray(coupling, dtype=np.float64).copy(),
            emissions=emissions,
        )

    def to_packed(self) -> dict[str, NDArray]:
        """Inverse of `from_packed`: re-pad every channel's arrays out to
        `(max_M, max_K)` / `(max_d, max_M, max_K)`, matching `lsim_pack_para.m`'s
        output exactly. `P_all`/`mu_all` pad with zeros (`lsim_pack_para.m:50-51`);
        `sigma_all` pads with **ones** (`lsim_pack_para.m:64,67`) -- not zeros,
        which would make a padded covariance entry look like a legal-but-wrong
        variance of 0 instead of the MATLAB-exact filler value."""
        s = self.structure
        P_all = np.zeros((s.C, s.max_M, s.max_K))
        mu_all = np.zeros((s.D, s.max_M, s.max_K))
        if self.sigma_diag:
            sigma_all = np.ones((s.D, s.max_M, s.max_K))
        else:
            sigma_all = np.ones((s.D, s.max_d, s.max_M, s.max_K))
        for c, gmm in enumerate(self.emissions):
            M_c, K_c = gmm.M_c, gmm.K_c
            dslice = s.dim_slice(c)
            P_all[c, :M_c, :K_c] = gmm.weights
            mu_all[dslice, :M_c, :K_c] = gmm.mu
            if self.sigma_diag:
                sigma_all[dslice, :M_c, :K_c] = gmm.sigma
            else:
                sigma_all[dslice, : s.channel_obs_dim[c], :M_c, :K_c] = gmm.sigma
        return {
            "pi_0": self.pi_0.copy(),
            "transitions_matrices": self.transition.copy(),
            "coupling_tetha_IM": self.coupling.copy(),
            "P_all": P_all,
            "mu_all": mu_all,
            "sigma_all": sigma_all,
            "sigma_diag": self.sigma_diag,
        }

    # -- cells <-> per-channel (the MATLAB cell-array API: pi_0_lsim{zee,1} etc.) --

    @classmethod
    def from_cells(
        cls,
        structure: LSIMStructure,
        *,
        pi_0_cells: list[NDArray[np.float64]],
        transition_cells: list[list[NDArray[np.float64]]],
        coupling: NDArray[np.float64],
        emissions: list[ChannelGMM],
    ) -> LSIMParams:
        """`pi_0_cells[c]` is `(M_c,)`; `transition_cells[c][zee]` is the
        `(M_c, M_zee)` block `transition_matrices_IM{c,zee}` — see
        `common function/lsim_pack_para.m`. Unlike the packed representation
        these are never padded, so this is a direct concatenation, not a slice."""
        C = structure.C
        S = structure.S
        pi_0 = np.zeros(S)
        for c in range(C):
            pi_0[structure.state_slice(c)] = pi_0_cells[c]
        transition = np.zeros((S, S))
        for c in range(C):
            for zee in range(C):
                transition[structure.state_slice(c), structure.state_slice(zee)] = transition_cells[c][zee]
        return cls(structure=structure, pi_0=pi_0, transition=transition, coupling=np.asarray(coupling, dtype=np.float64).copy(), emissions=emissions)

    def to_cells(self) -> dict[str, object]:
        s = self.structure
        pi_0_cells = [self.pi_0[s.state_slice(c)].copy() for c in range(s.C)]
        transition_cells = [
            [self.transition[s.state_slice(c), s.state_slice(zee)].copy() for zee in range(s.C)] for c in range(s.C)
        ]
        return {
            "pi_0_cells": pi_0_cells,
            "transition_cells": transition_cells,
            "coupling": self.coupling.copy(),
            "emissions": [gmm.copy() for gmm in self.emissions],
        }
