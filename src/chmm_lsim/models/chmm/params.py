"""`CHMMParams`: parameters of a full coupled HMM, and `lsim_to_chmm`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.joint_state import MAX_JOINT_STATES, EquivalentModelTooLargeError
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import _SUM_ATOL, ValidationError, _check_no_nan_inf, check_structure_matches
from chmm_lsim.emissions.base import check_emissions_match, validate_emissions
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.equivalent import _conditional_transitions
from chmm_lsim.models.lsim.params import LSIMParams


def _num_joint_states(structure: LSIMStructure) -> int:
    """`J = prod(M)`, rejected above `MAX_JOINT_STATES` (exact Python integer
    product, so it cannot overflow)."""
    J = math.prod(structure.channel_state_num)
    if J > MAX_JOINT_STATES:
        raise EquivalentModelTooLargeError(
            f"a CHMM with channel_state_num={structure.channel_state_num} has J={J} joint states, exceeding "
            f"MAX_JOINT_STATES={MAX_JOINT_STATES}; exact CHMM inference is meant for small models"
        )
    return J

def _joint_states(structure: LSIMStructure) -> NDArray[np.int64]:
    """`(C, J)` zero-based per-channel states of every joint state
    (MATLAB's `index_matrix - 1`)."""
    J = _num_joint_states(structure)
    return np.array(np.unravel_index(np.arange(J), structure.channel_state_num), dtype=np.int64).reshape(structure.C, J)

@dataclass
class CHMMParams:
    """Parameters of a full CHMM.

    Attributes
    ----------
    structure : LSIMStructure
        Per-channel numbers of states, observation dimensions and mixture
        components.
    pi_0 : ndarray, shape (S,)
        Per-channel initial distributions, stacked channel by channel; each
        block sums to one. The joint initial distribution is their product.
    transition : list of ndarray
        `transition[z]` has shape `(J, M_z)`; row `j` is
        `P(x^z_t | joint state j at t-1)` and sums to one.
    emissions : list of ChannelGMM
        Per-channel Gaussian mixtures (all diagonal or all full covariance).
    """

    structure: LSIMStructure
    pi_0: NDArray[np.float64]
    transition: list[NDArray[np.float64]]
    emissions: list[ChannelGMM] = field(default_factory=list)

    def __post_init__(self) -> None:
        structure = self.structure
        J = _num_joint_states(structure)
        self.pi_0 = np.asarray(self.pi_0, dtype=np.float64)
        if self.pi_0.shape != (structure.S,):
            raise ValueError(f"pi_0.shape={self.pi_0.shape} != (S,)=({structure.S},)")
        if len(self.transition) != structure.C:
            raise ValueError(f"len(transition)={len(self.transition)} != C={structure.C}")
        self.transition = [np.asarray(t, dtype=np.float64) for t in self.transition]
        for z, table in enumerate(self.transition):
            expected = (J, structure.channel_state_num[z])
            if table.shape != expected:
                raise ValueError(f"transition[{z}].shape={table.shape} != (J, M_z)={expected}")
        check_emissions_match(structure, self.emissions)

    @property
    def sigma_diag(self) -> bool:
        """True for diagonal covariances, False for full covariances."""
        return self.emissions[0].diag

    @property
    def n_joint_states(self) -> int:
        """`J = prod(channel_state_num)`."""
        return math.prod(self.structure.channel_state_num)

    def validate(self, *, matlab_compat: bool = True) -> None:
        """Raises `ValidationError` unless every distribution is finite,
        non-negative and sums to one (within 1e-8), and every emission model
        is valid (see the emission model's `validate`)."""
        structure = self.structure
        _check_no_nan_inf("pi_0", self.pi_0)
        if (self.pi_0 < 0).any():
            raise ValidationError("pi_0 has negative entries")
        for c in range(structure.C):
            total = float(self.pi_0[structure.state_slice(c)].sum())
            if not np.isclose(total, 1.0, rtol=0, atol=_SUM_ATOL):
                raise ValidationError(f"pi_0 block for channel {c} sums to {total}, expected 1.0")
        for z, table in enumerate(self.transition):
            _check_no_nan_inf(f"transition[{z}]", table)
            if (table < 0).any():
                raise ValidationError(f"transition[{z}] has negative entries")
            row_sums = table.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=0, atol=_SUM_ATOL):
                bad = int(np.argmax(np.abs(row_sums - 1.0)))
                raise ValidationError(f"transition[{z}] row {bad} sums to {row_sums[bad]}, expected 1.0")
        validate_emissions(self.emissions, matlab_compat=matlab_compat)

    def copy(self) -> CHMMParams:
        return CHMMParams(
            structure=self.structure,
            pi_0=self.pi_0.copy(),
            transition=[t.copy() for t in self.transition],
            emissions=[gmm.copy() for gmm in self.emissions],
        )

def lsim_to_chmm(structure: LSIMStructure, params: LSIMParams) -> CHMMParams:
    """The CHMM an LSIM is equivalent to: `transition[z][j, :] =
    sum_c coupling[c, z] * A[c, z][m_c(j), :]`, with the LSIM's own `pi_0`
    and emissions (copied)."""
    check_structure_matches(structure, params)
    states = _joint_states(structure)
    return CHMMParams(
        structure=structure,
        pi_0=params.pi_0.copy(),
        transition=_conditional_transitions(structure, params, states),
        emissions=[gmm.copy() for gmm in params.emissions],
    )
