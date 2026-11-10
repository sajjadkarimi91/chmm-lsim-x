"""Deprecated location; import from chmm_lsim (this module re-exports `chmm_lsim.models.chmm`, `chmm_lsim.models.chmm.params` and others)."""

from chmm_lsim._deprecation import moved_module
from chmm_lsim.models.chmm import __all__ as _public_all

moved_module(
    globals(),
    ['chmm_lsim.models.chmm', 'chmm_lsim.models.chmm.params', 'chmm_lsim.models.chmm.inference', 'chmm_lsim.models.chmm.em', 'chmm_lsim.models.chmm.init', 'chmm_lsim.models.chmm.simulate', 'chmm_lsim.models.chmm.viterbi', 'chmm_lsim.learning.em_loop', 'chmm_lsim.emissions.gmm'],
    replacement='import from chmm_lsim',
    aliases={'_m_step_emissions': 'chmm_lsim.emissions.base:m_step_emissions'},
    public_all=_public_all,
)
