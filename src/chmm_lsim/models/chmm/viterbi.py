"""Exact Viterbi decoding of a CHMM on its joint state space (`viterbi_chmm.m`, 4-argument form)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.viterbi import ViterbiResult
from chmm_lsim.models.chmm.inference import chmm_joint_model
from chmm_lsim.models.chmm.params import CHMMParams


def viterbi_chmm(structure: LSIMStructure, params: CHMMParams, Y: NDArray[np.float64], *, validate: bool = True) -> ViterbiResult:
    """Exact Viterbi decoding of a CHMM on its joint state space.

    Returns a `ViterbiResult` whose `states` `(C, T)` are zero-based and whose
    `log_score` is the log-probability of the decoded joint path together with
    the observations, `log P(x_1..x_T, y_1..y_T)` -- a true log-probability,
    unlike the per-channel score of the approximate `viterbi_lsim`. Ties are
    broken towards the lowest joint-state index.
    """
    model = chmm_joint_model(structure, params, Y, validate=validate)
    J, T = model.log_B.shape
    with np.errstate(divide="ignore"):
        log_A = np.log(model.transition)
        delta = np.log(model.pi) + model.log_B[:, 0]
    psi = np.zeros((J, T), dtype=np.int64)
    for t in range(1, T):
        candidate = delta[:, None] + log_A  # (previous, next)
        psi[:, t] = np.argmax(candidate, axis=0)
        delta = candidate[psi[:, t], np.arange(J)] + model.log_B[:, t]

    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(delta))
    log_prob = float(delta[path[T - 1]])
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return ViterbiResult(states=model.states[:, path].copy(), log_score=log_prob)
