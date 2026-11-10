"""Full coupled HMMs with exact joint-state inference (internal module; import from chmm_lsim)."""

from chmm_lsim.models.chmm.em import compute_chmm_aic_bic, em_chmm
from chmm_lsim.models.chmm.inference import (
    CHMMJointModel,
    CHMMPosterior,
    chmm_exact_forward_backward,
    chmm_joint_model,
    forward_backward_chmm,
)
from chmm_lsim.models.chmm.init import initialize_chmm_params
from chmm_lsim.models.chmm.params import CHMMParams, lsim_to_chmm
from chmm_lsim.models.chmm.simulate import simulate_chmm
from chmm_lsim.models.chmm.viterbi import viterbi_chmm

__all__ = [
    "CHMMParams",
    "CHMMJointModel",
    "CHMMPosterior",
    "lsim_to_chmm",
    "chmm_joint_model",
    "chmm_exact_forward_backward",
    "forward_backward_chmm",
    "viterbi_chmm",
    "compute_chmm_aic_bic",
    "em_chmm",
    "initialize_chmm_params",
    "simulate_chmm",
]
