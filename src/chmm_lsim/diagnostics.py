"""Deprecated location; import from chmm_lsim (this module re-exports `chmm_lsim.core.diagnostics`, `chmm_lsim.models.lsim.diagnostics`)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.core.diagnostics', 'chmm_lsim.models.lsim.diagnostics'],
    replacement='import from chmm_lsim',
)
