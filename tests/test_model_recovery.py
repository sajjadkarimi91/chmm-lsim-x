"""P6 gate: model recovery -- simulate from a known small model
(`simulate.py`), initialize and fit an EM model of the same shape on the
simulated data (`init.py`/`em.py`), decode the fitted model's most likely
state per timestep, and check it recovers the *true* simulated states up to
a label permutation (EM has no reason to number its states the same way the
generator did) -- resolved via the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) on the confusion matrix.

**The threshold is measured, not guessed, and gated continuously across
seeds** (same methodology as every other calibrated tolerance in this
port): with the model below (`C=1`, `M=3`, `d=2`, well-separated but not
trivially separated means, `T=600`), seeds `0-4` gave per-seed accuracies
of `0.9600, 0.9617, 0.9567, 0.9700, 0.9800` (min `0.9567`); `0.85` below
carries comfortable margin under that measured minimum without being so
loose it would pass a genuinely broken EM/init path. The test is
parametrized over all five seeds, not just `SEED=0`, so a regression that
only shows up for some seeds (e.g. a label-matching edge case, or an
initialization sensitive to the RNG stream) cannot hide behind a single
lucky draw. A closer mean separation (tried at calibration time, not
shipped) degrades this sharply -- min accuracy `0.54` at half the
separation -- which is expected (state identifiability is genuinely harder
with more emission overlap) and confirms this isn't a threshold that would
pass regardless of what EM actually does.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.models.lsim.em import em_lsim
from chmm_lsim.models.lsim.inference.sed import forward_backward_sed
from chmm_lsim.models.lsim.init import initialize_lsim_params
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.models.lsim.simulate import simulate_lsim
from chmm_lsim.core.structure import LSIMStructure

try:
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

requires_sklearn = pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn (the 'init' extra) is not installed")

ACCURACY_THRESHOLD = 0.85
T = 600
SEPARATION = 3.0


def _true_model():
    structure = LSIMStructure(channel_state_num=(3,), channel_obs_dim=(2,), num_gmm_component=(1,))
    pi_0 = np.array([0.4, 0.3, 0.3])
    transition = np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]])
    coupling = np.array([[1.0]])
    mu = np.array([[0.0, 0.0], [SEPARATION, 0.0], [0.0, SEPARATION]]).T[:, :, None]
    sigma = np.ones((2, 3, 1))
    gmm = ChannelGMM(weights=np.ones((3, 1)), mu=mu, sigma=sigma, diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()
    return structure, params


def _hungarian_matched_accuracy(true_states: np.ndarray, fitted_states: np.ndarray, M: int) -> float:
    confusion = np.zeros((M, M))
    for t_true, t_fit in zip(true_states, fitted_states):
        confusion[t_true, t_fit] += 1
    true_idx, fit_idx = linear_sum_assignment(-confusion)
    mapping = dict(zip(fit_idx, true_idx))
    mapped = np.array([mapping[s] for s in fitted_states])
    return float(np.mean(mapped == true_states))


@requires_sklearn
@pytest.mark.parametrize("seed", range(5))
def test_model_recovery_after_simulation_and_em(seed):
    structure, true_params = _true_model()
    Y, X_true = simulate_lsim(structure, true_params, T, seed=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params0 = initialize_lsim_params(structure, [Y], sigma_diag=True, seed=seed)
        result = em_lsim(
            structure, params0, [Y], max_iterations=20, fb_constants=FbConstants(variant="sed"), check_convergence=False, validate="off"
        )
        post = forward_backward_sed(structure, result.params, Y, FbConstants(variant="sed"))

    X_fit = np.argmax(post.gamma, axis=0)
    accuracy = _hungarian_matched_accuracy(X_true[0], X_fit, structure.channel_state_num[0])

    assert accuracy > ACCURACY_THRESHOLD, accuracy
    result.params.validate()


def test_hungarian_matching_recovers_a_pure_relabeling_exactly():
    """Direct unit test of the matching helper itself, independent of
    EM/simulation: a hidden-state sequence relabeled by a fixed permutation
    must recover 100% accuracy after matching."""
    rng = np.random.default_rng(0)
    true_states = rng.integers(0, 3, size=200)
    permutation = {0: 2, 1: 0, 2: 1}
    relabeled = np.array([permutation[s] for s in true_states])

    accuracy = _hungarian_matched_accuracy(true_states, relabeled, 3)
    assert accuracy == 1.0
