"""Deprecated location; import from chmm_lsim (this module re-exports `chmm_lsim.models.lsim.em`, `chmm_lsim.learning.em_loop` and others)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.models.lsim.em', 'chmm_lsim.learning.em_loop', 'chmm_lsim.learning.minibatch', 'chmm_lsim.emissions.gmm'],
    replacement='import from chmm_lsim',
    aliases={'_m_step_emissions': 'chmm_lsim.emissions.base:m_step_emissions'},
)
