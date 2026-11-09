"""Discriminative training of LSIM class models with maximum mutual information (MMI, published
recipe of Woodland & Povey 2002; Python port of the MATLAB trainer in ``matlab/discriminative``),
with CPU process-pool and (multi-)GPU torch engines and stochastic mini-batches.

Entry points: :func:`em_lsim_discriminative` (training) and :func:`lsim_discriminative_predict`
(scores and predictions), named as in MATLAB; ``train_discriminative`` and ``predict`` are
aliases kept for existing code. See ``trainer.py`` for the method
and ``options.py`` for every option.
"""

from chmm_lsim.discriminative.counts import DiscCounts, NonfiniteStatistics, sequence_counts
from chmm_lsim.discriminative.domain import Domain, at_bounds, build_domain, project
from chmm_lsim.discriminative.engines import NumpyEngine, TorchEngine, available_cuda_devices, make_engine
from chmm_lsim.discriminative.matlab_io import load_packed_model, packed_to_params
from chmm_lsim.discriminative.metrics import (
    binary_metrics, detection_score, evaluate_models, score_metrics, threshold_metrics, tune_threshold,
)
from chmm_lsim.discriminative.objective import ObjectiveConfig, balanced_accuracy, objective
from chmm_lsim.discriminative.options import DiscOptions, OptionError, make_options
from chmm_lsim.discriminative.trainer import (
    TrainingError, TrainResult, em_lsim_discriminative, lsim_discriminative_predict, predict,
    train_discriminative,
)
from chmm_lsim.discriminative.updates import ebw_proposal, interpolate, published_update, weak_sense

__all__ = [
    "em_lsim_discriminative", "lsim_discriminative_predict", "DiscCounts", "DiscOptions", "Domain", "NonfiniteStatistics", "NumpyEngine", "ObjectiveConfig", "OptionError",
    "TorchEngine", "TrainResult", "TrainingError", "at_bounds", "available_cuda_devices", "balanced_accuracy",
    "binary_metrics", "build_domain", "detection_score", "evaluate_models", "ebw_proposal", "interpolate", "load_packed_model", "make_engine",
    "make_options", "objective", "packed_to_params", "predict", "project", "published_update", "score_metrics", "sequence_counts", "threshold_metrics", "tune_threshold",
    "train_discriminative", "weak_sense",
]
