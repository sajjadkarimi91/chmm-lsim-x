"""Deprecated location; import from chmm_lsim (this module re-exports `chmm_lsim.core.structure`)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.core.structure'],
    replacement='import from chmm_lsim',
)
