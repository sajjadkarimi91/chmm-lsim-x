"""chmm_lsim: inference and learning for Latent Structure Influence Models (LSIMs).

An LSIM is a coupled hidden Markov model in which each channel's next state is
drawn from a theta-weighted mixture of pairwise transitions from every channel.
This package provides

* approximate forward-backward inference with three backward recursions
  (`FbConstants(variant=...)`): ``"poise"`` (the default), ``"sed"`` and
  ``"fast"`` (SVI), plus exact inference through the equivalent joint HMM for
  small models (`exact_posterior`, `viterbi_exact`);
* maximum-likelihood EM learning (`em_lsim`), scalable mini-batch training for
  large datasets (`em_lsim_minibatch_train`), and parameter initialization
  (`initialize_lsim_params`, needs ``chmm-lsim[init]``);
* full coupled HMMs, with every channel's transition conditioned on the joint
  previous state and exact joint-state inference: `CHMMParams`,
  `forward_backward_chmm`, `viterbi_chmm`, `em_chmm` (returning the same
  `EmResult` as `em_lsim`), `initialize_chmm_params`, `simulate_chmm`,
  `lsim_to_chmm`;
* discriminative (MMI) learning of class-conditional LSIMs
  (`chmm_lsim.discriminative`);
* simulation (`simulate_lsim`), Viterbi decoding (`viterbi_lsim`) and model
  serialization (`save_lsim_params` / `load_lsim_params`).

Execution backends: NumPy (default), numba (``backend="numba"``, needs
``chmm-lsim[numba]``) and PyTorch on CPU or CUDA (``backend="torch"`` and the
batched API `forward_backward_batch`, needs ``chmm-lsim[torch]``). Optional
dependencies are imported lazily, so ``import chmm_lsim`` only needs NumPy and
SciPy.

The names listed in ``chmm_lsim.__all__`` and the ``chmm_lsim.discriminative``
package are the public API. The subpackages (``core``, ``emissions``,
``models``, ``learning``, ``backends``) are internal and may be reorganized;
the extension points are `register_inference` (new LSIM recursions),
`register_backend` (execution backends) and
``chmm_lsim.emissions.register_emission_model`` (emission models).
"""

from chmm_lsim.backends.base import (
    BackendCapabilities,
    MissingBackendDependency,
    UnsupportedCombination,
    check_capability,
    get_backend,
    register_backend,
    run_backend,
)
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.core.diagnostics import (
    LayerBSummary,
    marginalize_joint_to_channels,
    per_channel_total_variation,
    summarize_layer_b,
    total_variation_distance,
)
from chmm_lsim.core.joint_state import (
    MAX_JOINT_MIXTURE_COMPONENTS,
    MAX_JOINT_STATES,
    EquivalentModelTooLargeError,
    joint_state_from_index,
    joint_state_index,
    scaled_forward_backward,
)
from chmm_lsim.core.sampling import select_p
from chmm_lsim.core.serialization import SerializationError
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError, validate_observations
from chmm_lsim.core.viterbi import ViterbiResult
from chmm_lsim.emissions import compute_emissions, gmm_pdf_diag, gmm_pdf_full
from chmm_lsim.emissions.gmm import ChannelExpectedStats, ChannelGMM, compute_global_nanvar, finalize_channel_stats, merge_channel_stats, repair_covariance
from chmm_lsim.emissions.gmm_init import InitialisationError
from chmm_lsim.learning.em_loop import (
    EmResult,
    ExpectedStats,
    TrialExecutionError,
    apply_left2right_pi0_smoothing,
    finalize_expected_stats,
    merge_expected_stats,
)
from chmm_lsim.learning.minibatch import (
    MinibatchEmResult,
    MinibatchSchedule,
    MinibatchTrainingResult,
    MinibatchUpdate,
    RobustConvergence,
    generate_minibatch_schedule,
    validate_minibatch_schedule,
)
from chmm_lsim.models.chmm import (
    CHMMJointModel,
    CHMMParams,
    CHMMPosterior,
    chmm_exact_forward_backward,
    chmm_joint_model,
    compute_chmm_aic_bic,
    em_chmm,
    forward_backward_chmm,
    initialize_chmm_params,
    lsim_to_chmm,
    simulate_chmm,
    viterbi_chmm,
)
from chmm_lsim.models.lsim.diagnostics import joint_log_probability
from chmm_lsim.models.lsim.em import (
    compute_aic_bic,
    e_step,
    em_lsim,
    em_lsim_minibatch,
    em_lsim_minibatch_train,
    m_step,
)
from chmm_lsim.models.lsim.equivalent import EquivalentHMM, build_equivalent_hmm, exact_posterior
from chmm_lsim.models.lsim.inference import (
    Diagnostics,
    Posterior,
    forward_backward,
    forward_backward_fast,
    forward_backward_from_emissions,
    forward_backward_poise,
    forward_backward_sed,
    forward_loglik_from_emissions,
    get_inference,
    register_inference,
)
from chmm_lsim.models.lsim.init import initialize_lsim_params
from chmm_lsim.models.lsim.params import LSIMParams, validate_params
from chmm_lsim.models.lsim.serialization import load_lsim_params, save_lsim_params
from chmm_lsim.models.lsim.simulate import simulate_lsim
from chmm_lsim.models.lsim.viterbi import viterbi_exact, viterbi_exact_from_equiv, viterbi_lsim

__version__ = "1.0.0"

__all__ = [
    "LSIMStructure",
    "LSIMParams",
    "ChannelGMM",
    "ValidationError",
    "validate_params",
    "validate_observations",
    "compute_emissions",
    "gmm_pdf_diag",
    "gmm_pdf_full",
    "FbConstants",
    "Posterior",
    "Diagnostics",
    "register_inference",
    "get_inference",
    "BackendCapabilities",
    "UnsupportedCombination",
    "MissingBackendDependency",
    "register_backend",
    "get_backend",
    "check_capability",
    "run_backend",
    "forward_backward",
    "forward_backward_sed",
    "forward_backward_fast",
    "forward_backward_poise",
    "forward_backward_from_emissions",
    "forward_loglik_from_emissions",
    "ExpectedStats",
    "ChannelExpectedStats",
    "EmResult",
    "TrialExecutionError",
    "e_step",
    "m_step",
    "merge_expected_stats",
    "merge_channel_stats",
    "finalize_expected_stats",
    "finalize_channel_stats",
    "compute_aic_bic",
    "apply_left2right_pi0_smoothing",
    "repair_covariance",
    "em_lsim",
    "MinibatchSchedule",
    "MinibatchUpdate",
    "MinibatchEmResult",
    "generate_minibatch_schedule",
    "validate_minibatch_schedule",
    "compute_global_nanvar",
    "em_lsim_minibatch",
    "MinibatchTrainingResult",
    "RobustConvergence",
    "em_lsim_minibatch_train",
    "InitialisationError",
    "initialize_lsim_params",
    "save_lsim_params",
    "SerializationError",
    "load_lsim_params",
    "select_p",
    "simulate_lsim",
    "MAX_JOINT_STATES",
    "MAX_JOINT_MIXTURE_COMPONENTS",
    "EquivalentModelTooLargeError",
    "EquivalentHMM",
    "build_equivalent_hmm",
    "scaled_forward_backward",
    "exact_posterior",
    "joint_state_index",
    "joint_state_from_index",
    "ViterbiResult",
    "viterbi_lsim",
    "viterbi_exact",
    "viterbi_exact_from_equiv",
    "joint_log_probability",
    "marginalize_joint_to_channels",
    "total_variation_distance",
    "per_channel_total_variation",
    "LayerBSummary",
    "summarize_layer_b",
    "forward_backward_batch",
    "BatchedPosteriorChunk",
    "TorchKernel",
    "estimate_chunk_size",
    "CHMMParams",
    "CHMMPosterior",
    "CHMMJointModel",
    "lsim_to_chmm",
    "chmm_joint_model",
    "chmm_exact_forward_backward",
    "forward_backward_chmm",
    "viterbi_chmm",
    "compute_chmm_aic_bic",
    "em_chmm",
    "initialize_chmm_params",
    "simulate_chmm",
    "em_lsim_discriminative",
    "lsim_discriminative_predict",
    "__version__",
]

# The batched Torch API lives in a module that imports `torch` at import time.
# It is resolved lazily (PEP 562) so that `import chmm_lsim` never needs torch;
# accessing one of these names without torch installed raises
# MissingBackendDependency.
_TORCH_BATCHED_NAMES = frozenset({"forward_backward_batch", "BatchedPosteriorChunk", "TorchKernel", "estimate_chunk_size"})


# Discriminative (MMI) entry points, named as in MATLAB; resolved lazily like the names above.
_DISCRIMINATIVE_NAMES = frozenset({"em_lsim_discriminative", "lsim_discriminative_predict"})


def __getattr__(name: str):
    if name in _DISCRIMINATIVE_NAMES:
        from chmm_lsim.learning import discriminative

        return getattr(discriminative, name)
    if name in _TORCH_BATCHED_NAMES:
        from chmm_lsim.backends import torch_batched

        return getattr(torch_batched, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _TORCH_BATCHED_NAMES | _DISCRIMINATIVE_NAMES)
