"""Discriminative (MMI) training of LSIM class models: the public
``chmm_lsim.discriminative`` namespace (implemented in
`chmm_lsim.learning.discriminative`, whose module docstring describes the method).

Entry points: `em_lsim_discriminative` (training) and
`lsim_discriminative_predict` (scores and predictions); ``train_discriminative``
and ``predict`` are aliases of these two, kept for existing code. The trainer
is specific to diagonal-covariance Gaussian-mixture emissions.
"""

from chmm_lsim.learning.discriminative import *  # noqa: F401,F403
from chmm_lsim.learning.discriminative import __all__  # noqa: F401

_SUBMODULES = ("counts", "domain", "engines", "kernels", "matlab_io", "metrics", "objective", "options", "trainer", "updates")


def __getattr__(name: str):
    # Historical submodules (`chmm_lsim.discriminative.trainer`, ...) stay reachable
    # as attributes; importing one emits a DeprecationWarning.
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
