"""Deprecated location; import from chmm_lsim (this module re-exports `chmm_lsim.core.validate`, `chmm_lsim.models.lsim.params` and others)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.core.validate', 'chmm_lsim.models.lsim.params', 'chmm_lsim.emissions.base', 'chmm_lsim.emissions.gmm'],
    replacement='import from chmm_lsim',
)
