"""Emission models (internal module; import from chmm_lsim).

`base` defines the `EmissionModel` interface and the registry
(`register_emission_model`, `get_emission_model`, `emission_model_for`);
`gmm` is the Gaussian-mixture model (``"gmm"``), registered on import, and
`gmm_init` its scikit-learn based initialization. `compute_emissions`,
`gmm_pdf_diag` and `gmm_pdf_full` are also available here, at their
historical location.
"""

from chmm_lsim.core.structure import LSIMStructure  # noqa: F401  (historical name)
from chmm_lsim.emissions.base import (
    EmissionModel,
    compute_emissions,
    emission_model_for,
    emission_model_for_stats,
    get_emission_model,
    register_emission_model,
    registered_emission_models,
    unregister_emission_model,
)
from chmm_lsim.emissions.gmm import (  # noqa: F401  (_LOG_2PI, multivariate_normal: historical names)
    _LOG_2PI,
    ChannelGMM,
    GaussianMixtureEmissions,
    gmm_pdf_diag,
    gmm_pdf_full,
    multivariate_normal,
)

__all__ = [
    "EmissionModel",
    "GaussianMixtureEmissions",
    "ChannelGMM",
    "register_emission_model",
    "unregister_emission_model",
    "get_emission_model",
    "emission_model_for",
    "emission_model_for_stats",
    "registered_emission_models",
    "compute_emissions",
    "gmm_pdf_diag",
    "gmm_pdf_full",
]


def __getattr__(name):
    # `LSIMParams` was importable from the historical `chmm_lsim.emissions` module.
    if name == "LSIMParams":
        from chmm_lsim.models.lsim.params import LSIMParams

        return LSIMParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
