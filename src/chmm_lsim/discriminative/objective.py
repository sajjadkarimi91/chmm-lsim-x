"""Deprecated location; import from chmm_lsim.discriminative (this module re-exports `chmm_lsim.learning.discriminative.objective`)."""

from chmm_lsim._deprecation import moved_module

moved_module(
    globals(),
    ['chmm_lsim.learning.discriminative.objective'],
    replacement='import from chmm_lsim.discriminative',
)
