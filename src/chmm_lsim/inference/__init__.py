"""Deprecated location; import from chmm_lsim (this package's contents moved to
`chmm_lsim.models.lsim.inference`)."""

from chmm_lsim._deprecation import lazy_moved_package

lazy_moved_package(
    globals(),
    ["chmm_lsim.models.lsim.inference"],
    replacement="import from chmm_lsim",
    submodules=("base", "registry", "sed", "fast", "poise"),
)
