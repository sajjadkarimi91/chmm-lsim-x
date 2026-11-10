"""Deprecated location; import from chmm_lsim (this module re-exports `chmm_lsim.models.lsim.serialization`, `chmm_lsim.core.serialization`)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.models.lsim.serialization', 'chmm_lsim.core.serialization'],
    replacement='import from chmm_lsim',
)
