"""Deprecated location; import from chmm_lsim.discriminative (this module re-exports `chmm_lsim.learning.discriminative.domain`)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.learning.discriminative.domain'],
    replacement='import from chmm_lsim.discriminative',
)
