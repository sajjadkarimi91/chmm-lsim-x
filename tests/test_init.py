"""P5 gate: `init.py`'s parameter initializer (`em_lsim.m`'s no-`init_model`
branch and its `fine_gmm_fitting` subfunction).

No component-by-component parity with MATLAB is claimed or tested --
`fitgmdist` and `sklearn.mixture.GaussianMixture` are different optimization
implementations (different RNG, different k-means/random-sample seeding
internals), so their fitted components will never bit-match. What *is*
tested, exactly and directly:

- The exact-uniform structural parameters (`pi_0`, `coupling`, every
  transition block) -- these are not fitted at all, so they must match
  MATLAB's formulas precisely, not statistically.
- The greedy component-to-state grouping rule, directly, on controlled fake
  GMM output -- independent of whatever `GaussianMixture` itself fits.
- The channel-local listwise NaN deletion and the undeleted-data regularizer,
  directly -- both are plain, exactly-specified numpy computations.
- The bounded, seeded `matlab_compat` retry/raise contract, deterministically
  -- via a fake fitter that always fails (or fails a fixed number of times),
  checking not just attempt counts but the actual jittered arrays and
  recomputed regularizers against an independent recomputation of MATLAB's
  own `counter*0.05*std(current data,ddof=1)` formula.
- Observation validation: `Y_trials` must be non-empty and every trial must
  pass `validate_observations` *before* any per-channel slicing -- checked
  for an empty list, a scalar, wrong `D` in both directions, non-floating
  dtype, `+/-Inf`, a zero-length trial, and a malformed trial that isn't the
  first one in the list.

**The one genuinely statistical claim (§P5 gate, `docs/PYTHON_PORT_PLAN.md`)**:
seeded initializer runs, fed through `em_lsim`, reach a final log-likelihood
comparable to a *real* MATLAB no-`init_model` run (`fine_gmm_fitting`, the
actual code path `sklearn.mixture.GaussianMixture` stands in for) on the same
data -- measured directly against `tests/fixtures/init_reference.npz`
(`tools/matlab/export_init_reference.m`), not assumed, and not the
`c3_het_diag` EM-trace fixture's `params_in(1)` (a review round found the
original version of this test compared against that by mistake -- those
`init_*`/`effective_*_iter1` fields come from `build_scenario`'s `mk_model`
ground-truth parameters passed as an explicit `init_model`, `em_lsim.m`'s
`nargin==6` branch, which never calls `fine_gmm_fitting` at all). See
`test_seeded_init_reaches_comparable_loglik_to_matlab_init` for the measured
numbers and the justified margin.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_em_trace_fixture, load_init_reference_fixture  # noqa: E402

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.models.lsim.em import em_lsim
from chmm_lsim.emissions.gmm_init import (
    InitialisationError,
    _channel_variance,
    _compute_regularizer,
    _fit_channel,
    _group_components_into_states,
)
from chmm_lsim.models.lsim.init import initialize_lsim_params
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError

import chmm_lsim.emissions.gmm_init as init_module

try:
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

requires_sklearn = pytest.mark.skipif(
    not _SKLEARN_AVAILABLE,
    reason="scikit-learn (the 'init' extra) is not installed -- see test_missing_sklearn_raises_initialisation_error for the dedicated no-sklearn contract test, which runs regardless",
)
# Most of this file's tests monkeypatch `init_module._import_gaussian_mixture`
# with `_FakeGMM` and so never touch real scikit-learn at all -- those must
# keep running (and do) with no optional dependency installed. A handful of
# tests below are genuinely statistical (seeded reproducibility against a
# real fit, or a real-`GaussianMixture` retry/failure path) and cannot be
# meaningfully faked; `requires_sklearn` skips exactly those, cleanly, in a
# core-only environment -- `test_missing_sklearn_raises_initialisation_error`
# itself is not decorated with it, since it specifically tests behavior when
# scikit-learn is absent and must keep running in that exact environment.


class _FakeGMM:
    """Stand-in for `sklearn.mixture.GaussianMixture`: records every
    constructor kwarg and every `.fit()` call's data, and either always
    raises (bounded-retry tests) or succeeds with hand-specified
    means/variances/weights (so grouping/shape tests don't depend on a real,
    randomized fit) -- or, if none are given, auto-generates arbitrary but
    correctly-shaped output from `n_components`/`X.shape[1]`, for tests that
    don't care about the fitted values at all (e.g. the structural-parameter
    test, which fits two channels with different `n_components` from the
    same monkeypatch). `fail_times` counts *across* instances (a fresh
    `GaussianMixture(...)` is constructed every retry attempt in the real
    code, so a per-instance counter would never see more than one call)."""

    calls: list[NDArray] = []
    init_kwargs: list[dict] = []
    fail_times: int = 0
    means_out = None
    variances_out = None
    weights_out = None

    def __init__(self, **kwargs):
        type(self).init_kwargs.append(kwargs)
        self._n_components = kwargs["n_components"]

    def fit(self, X):
        type(self).calls.append(X)
        attempt = len(type(self).calls)
        if attempt <= type(self).fail_times:
            raise ValueError(f"forced failure #{attempt}")
        if type(self).means_out is not None:
            self.means_ = type(self).means_out
            self.covariances_ = type(self).variances_out
            self.weights_ = type(self).weights_out
        else:
            n, d_c = self._n_components, X.shape[1]
            self.means_ = np.arange(n * d_c, dtype=float).reshape(n, d_c)
            self.covariances_ = np.ones((n, d_c))
            self.weights_ = np.full(n, 1.0 / n)
        return self


def _reset_fake(fail_times, means_out=None, variances_out=None, weights_out=None):
    _FakeGMM.calls = []
    _FakeGMM.init_kwargs = []
    _FakeGMM.fail_times = fail_times
    _FakeGMM.means_out = means_out
    _FakeGMM.variances_out = variances_out
    _FakeGMM.weights_out = weights_out


# --------------------------------------------------------------------------
# Exact-uniform structural parameters (em_lsim.m:89,248,250) -- not fitted,
# must match MATLAB's formulas exactly, on a heterogeneous M_c structure.
# --------------------------------------------------------------------------

def test_uniform_structural_parameters_heterogeneous_M(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    _reset_fake(0)  # channels need different n_components (2 vs 3); auto-shaped fake output
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(2, 50))

    params = initialize_lsim_params(structure, [Y], sigma_diag=True, seed=0)

    np.testing.assert_allclose(params.pi_0, [0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(params.coupling, np.full((2, 2), 0.5))
    # every (c, zee) block is uniformly 1/M_zee, including c==zee -- no special
    # self-transition treatment (em_lsim.m:250 sets the whole target-zee column
    # block to 1/M_zee regardless of source row)
    np.testing.assert_allclose(params.transition[:, structure.state_slice(0)], np.full((5, 2), 0.5))
    np.testing.assert_allclose(params.transition[:, structure.state_slice(1)], np.full((5, 3), 1 / 3))
    params.validate()


# --------------------------------------------------------------------------
# Heterogeneous M_c/d_c/K_c, diagonal and full-covariance output shapes.
# --------------------------------------------------------------------------

def test_heterogeneous_shapes_diag(monkeypatch):
    """Drives `_fit_channel` directly, once per channel, with per-channel-sized
    fake GMM output -- `M_c`/`d_c`/`K_c` all differ across the 3 channels."""
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    structure = LSIMStructure(channel_state_num=(2, 3, 1), channel_obs_dim=(1, 2, 3), num_gmm_component=(2, 1, 3))
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(6, 80))

    for c in range(structure.C):
        M_c, K_c, d_c = structure.channel_state_num[c], structure.num_gmm_component[c], structure.channel_obs_dim[c]
        n = M_c * K_c
        means = rng.normal(size=(n, d_c)) * (10 * np.arange(n)[:, None] + 1)
        variances = np.ones((n, d_c))
        weights = np.full(n, 1.0 / n)
        _reset_fake(0, means, variances, weights)
        gmm = _fit_channel(Y[structure.dim_slice(c)], M_c, K_c, sigma_diag=True, channel=c, seed=c, matlab_compat=True)
        assert gmm.weights.shape == (M_c, K_c)
        assert gmm.mu.shape == (d_c, M_c, K_c)
        assert gmm.sigma.shape == (d_c, M_c, K_c)
        assert gmm.diag is True


def test_heterogeneous_shapes_full_covariance(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    M_c, K_c, d_c = 2, 3, 4
    n = M_c * K_c
    rng = np.random.default_rng(2)
    means = rng.normal(size=(n, d_c))
    variances = rng.uniform(0.5, 2.0, size=(n, d_c))
    weights = np.full(n, 1.0 / n)
    _reset_fake(0, means, variances, weights)
    Y = rng.normal(size=(d_c, 100))

    gmm = _fit_channel(Y, M_c, K_c, sigma_diag=False, channel=0, seed=0, matlab_compat=True)
    assert gmm.diag is False
    assert gmm.sigma.shape == (d_c, d_c, M_c, K_c)
    # every fitted diagonal variance vector becomes a diagonal matrix -- off-diagonals
    # exactly zero, on-diagonal entries exactly the fitted (grouped) variances
    for i in range(M_c):
        for k in range(K_c):
            sigma_ik = gmm.sigma[:, :, i, k]
            np.testing.assert_array_equal(sigma_ik, np.diag(np.diag(sigma_ik)))
            assert np.all(np.diag(sigma_ik) > 0)


# --------------------------------------------------------------------------
# Same-seed reproducibility (a real GaussianMixture fit, not the fake).
# --------------------------------------------------------------------------

@requires_sklearn
def test_same_seed_is_reproducible():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(2,))
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(2, 300))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = initialize_lsim_params(structure, [Y], sigma_diag=True, seed=42)
        p2 = initialize_lsim_params(structure, [Y], sigma_diag=True, seed=42)
        p3 = initialize_lsim_params(structure, [Y], sigma_diag=True, seed=43)

    np.testing.assert_array_equal(p1.emissions[0].mu, p2.emissions[0].mu)
    np.testing.assert_array_equal(p1.emissions[0].sigma, p2.emissions[0].sigma)
    np.testing.assert_array_equal(p1.emissions[0].weights, p2.emissions[0].weights)
    assert not np.array_equal(p1.emissions[0].mu, p3.emissions[0].mu)


# --------------------------------------------------------------------------
# Channel-local listwise NaN deletion (em_lsim.m's fitgmdist behaviour,
# measured in docs/PYTHON_PORT_PLAN.md §1.3): a timestep with any NaN in
# channel c's own dimensions is dropped from c, and only from c.
# --------------------------------------------------------------------------

def test_channel_local_nan_deletion_does_not_cross_channels(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    structure = LSIMStructure(channel_state_num=(1, 1), channel_obs_dim=(2, 1), num_gmm_component=(1, 1))
    T = 20
    rng = np.random.default_rng(3)
    Y0 = rng.normal(size=(2, T))
    Y0[0, [2, 5, 9]] = np.nan  # 3 timesteps unusable for channel 0 only
    Y1 = rng.normal(size=(1, T))  # channel 1 fully observed
    Y = np.concatenate([Y0, Y1], axis=0)

    _reset_fake(0, np.zeros((1, 2)), np.ones((1, 2)), np.array([1.0]))
    _fit_channel(Y[structure.dim_slice(0)], 1, 1, sigma_diag=True, channel=0, seed=0, matlab_compat=True)
    assert _FakeGMM.calls[0].shape[0] == T - 3

    _reset_fake(0, np.zeros((1, 1)), np.ones((1, 1)), np.array([1.0]))
    _fit_channel(Y[structure.dim_slice(1)], 1, 1, sigma_diag=True, channel=1, seed=0, matlab_compat=True)
    assert _FakeGMM.calls[0].shape[0] == T  # channel 1 untouched by channel 0's NaNs


def test_channel_variance_matlab_semantics():
    """0 non-NaN -> NaN; exactly 1 -> 0.0 (not NaN); >1 -> ddof=1 sample variance."""
    Y_c = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # 5 obs, ordinary case
            [np.nan, np.nan, 7.0, np.nan, np.nan],  # 1 obs -> 0.0
            [np.nan, np.nan, np.nan, np.nan, np.nan],  # 0 obs -> NaN
        ]
    )
    variance = _channel_variance(Y_c)
    np.testing.assert_allclose(variance[0], np.var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1))
    assert variance[1] == 0.0
    assert np.isnan(variance[2])


# --------------------------------------------------------------------------
# Regularizer: 0.01*min(var(Y,omitnan)) from the *undeleted* channel data.
# --------------------------------------------------------------------------

def test_regularizer_uses_undeleted_data_not_the_listwise_deleted_subset(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    # row 0 has a NaN that would be dropped by listwise deletion; the
    # regularizer must still be computed from all 5 raw values in that row.
    # Recomputed here from plain np.var/np.nanvar, not by reusing
    # _channel_variance, so this doesn't just check the implementation
    # against itself.
    Y_c = np.array([[1.0, 2.0, np.nan, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])
    var0 = np.var(np.array([1.0, 2.0, 4.0, 5.0]), ddof=1)
    var1 = np.var(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), ddof=1)
    expected_reg = 0.01 * min(var0, var1)

    _reset_fake(0, np.zeros((1, 2)), np.ones((1, 2)), np.array([1.0]))
    _fit_channel(Y_c, 1, 1, sigma_diag=True, channel=0, seed=0, matlab_compat=True)
    np.testing.assert_allclose(_FakeGMM.init_kwargs[0]["reg_covar"], expected_reg)


def test_compute_regularizer_rejects_nan_variance():
    with pytest.raises(InitialisationError, match=r"channel 2.*NaN.*dimension"):
        _compute_regularizer(np.array([1.0, np.nan, 2.0]), channel=2)


def test_compute_regularizer_rejects_bad_value():
    """Direct unit test of the defensive finite/non-negative guard, on a
    hand-built array real `_channel_variance` output can never produce
    (variance is never negative) -- verifies the guard itself, the same way
    `repair_covariance` is tested on hand-built matrices in test_em.py rather
    than by reverse-engineering data that hits the exact boundary."""
    with pytest.raises(InitialisationError, match="unusable regularizer"):
        _compute_regularizer(np.array([-1.0, 2.0]), channel=0)


# --------------------------------------------------------------------------
# Observations are validated before anything else touches them: an empty
# trial list, and every trial through validate_observations, before any
# per-channel slicing/concatenation -- not a raw IndexError/ValueError from
# indexing a malformed array, and not a silently-wrong slice.
# --------------------------------------------------------------------------

def _structure_for_validation():
    return LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))


def test_empty_trial_list_rejected():
    with pytest.raises(ValidationError, match="empty list"):
        initialize_lsim_params(_structure_for_validation(), [], sigma_diag=True, seed=0)


def test_scalar_trial_rejected():
    with pytest.raises(ValidationError, match="2-D"):
        initialize_lsim_params(_structure_for_validation(), [np.array(5.0)], sigma_diag=True, seed=0)


def test_extra_observation_rows_rejected():
    """D=2 rows expected; 3 given -- previously silently accepted and the
    extra row simply never read (`structure.dim_slice` never reaches it)."""
    Y = np.zeros((3, 10))
    with pytest.raises(ValidationError, match=r"structure\.D"):
        initialize_lsim_params(_structure_for_validation(), [Y], sigma_diag=True, seed=0)


def test_missing_observation_rows_rejected():
    Y = np.zeros((1, 10))
    with pytest.raises(ValidationError, match=r"structure\.D"):
        initialize_lsim_params(_structure_for_validation(), [Y], sigma_diag=True, seed=0)


def test_non_floating_dtype_rejected():
    """Integer observations -- previously accepted outright."""
    Y = np.zeros((2, 10), dtype=np.int64)
    with pytest.raises(ValidationError, match="dtype"):
        initialize_lsim_params(_structure_for_validation(), [Y], sigma_diag=True, seed=0)


def test_inf_rejected():
    Y = np.zeros((2, 10))
    Y[0, 0] = np.inf
    with pytest.raises(ValidationError, match="Inf"):
        initialize_lsim_params(_structure_for_validation(), [Y], sigma_diag=True, seed=0)


def test_zero_length_trial_rejected():
    Y = np.zeros((2, 0))
    with pytest.raises(ValidationError, match="T=0"):
        initialize_lsim_params(_structure_for_validation(), [Y], sigma_diag=True, seed=0)


def test_malformed_later_trial_rejected():
    """The first trial is fine; the second (not the first) is malformed --
    every trial is validated before any concatenation, not just trials[0]."""
    good = np.random.default_rng(0).normal(size=(2, 20))
    bad = np.zeros((3, 10))  # wrong D
    with pytest.raises(ValidationError, match=r"structure\.D"):
        initialize_lsim_params(_structure_for_validation(), [good, bad], sigma_diag=True, seed=0)


# --------------------------------------------------------------------------
# Greedy component-to-state grouping (em_lsim.m:205-245), on controlled fake
# GMM output -- independent of whatever GaussianMixture itself fits.
# --------------------------------------------------------------------------

def test_greedy_grouping_picks_nearest_neighbours_and_normalizes_weights():
    # two well-separated clusters of 2 components each -> M_c=2, K_c=2
    means = np.array([[0.0], [0.1], [10.0], [10.3]])
    variances = np.array([[1.0], [2.0], [3.0], [4.0]])
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    mu, var, w = _group_components_into_states(means, variances, weights, M_c=2, K_c=2)

    np.testing.assert_allclose(np.sort(mu[0, :, 0]), [0.0, 0.1])
    np.testing.assert_allclose(np.sort(mu[1, :, 0]), [10.0, 10.3])
    np.testing.assert_allclose(w[0].sum(), 1.0)
    np.testing.assert_allclose(w[1].sum(), 1.0)
    # weights normalized *within* each state, not globally
    np.testing.assert_allclose(w[0], [0.1, 0.2] / np.sum([0.1, 0.2]))


def test_greedy_grouping_anchor_is_fixed_not_chained():
    """The distance for every one of a state's K_c-1 picks is measured to the
    *fixed* anchor, never to the most-recently-added component. 1-D distances
    correlate too strongly between "nearest to A" and "nearest to the first
    pick" to disambiguate (the first pick is itself chosen for being near A),
    so this uses 2-D means, constructed so the fixed-anchor rule and a
    chained (nearest-to-most-recent) rule provably disagree on the third
    component:

    anchor=(0,0) (index 0, picked unconditionally first); among the rest,
    (1,0) is nearest to the anchor (dist 1) and is picked next by *both*
    rules. For the third pick, remaining are (0,3) (dist-to-anchor 9,
    dist-to-(1,0) 10) and (3.5,0) (dist-to-anchor 12.25, dist-to-(1,0) 6.25)
    -- fixed-anchor prefers (0,3) (9<12.25); chained would prefer (3.5,0)
    (6.25<10) instead."""
    means = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [3.5, 0.0]])
    variances = np.ones((4, 2))
    weights = np.full(4, 0.25)

    mu, _, _ = _group_components_into_states(means, variances, weights, M_c=1, K_c=3)

    picked = {tuple(np.round(row, 3)) for row in mu[0]}
    assert picked == {(0.0, 0.0), (1.0, 0.0), (0.0, 3.0)}
    assert (3.5, 0.0) not in picked


# --------------------------------------------------------------------------
# Failure handling: insufficient complete observations, invalid/all-NaN
# variance, unusable regularizer (above), failed fit -- each names the
# channel and (matlab_compat=True) goes through the bounded retry loop.
# --------------------------------------------------------------------------

@requires_sklearn
def test_insufficient_observations_matlab_compat_false_raises_immediately():
    Y_c = np.array([[1.0, 2.0, 3.0]])  # 3 complete obs, need M_c*K_c=6
    with pytest.raises(InitialisationError, match=r"channel 0.*3 complete.*need at least 6"):
        _fit_channel(Y_c, M_c=3, K_c=2, sigma_diag=True, channel=0, seed=0, matlab_compat=False)


@requires_sklearn
def test_insufficient_observations_matlab_compat_true_retries_then_raises():
    """Real GaussianMixture (not the fake): with genuinely fewer complete
    rows than components, every one of the 10 jittered attempts fails
    identically (jitter cannot manufacture new independent samples), and the
    loop exhausts deterministically."""
    Y_c = np.array([[1.0, 2.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(InitialisationError, match=r"did not succeed in 10 attempts") as exc_info:
            _fit_channel(Y_c, M_c=3, K_c=2, sigma_diag=True, channel=0, seed=0, matlab_compat=True)
    assert exc_info.value.__cause__ is not None


@requires_sklearn
def test_all_nan_channel_raises_invalid_variance():
    Y_c = np.full((1, 30), np.nan)
    with pytest.raises(InitialisationError, match=r"channel 0.*NaN.*dimension"):
        _fit_channel(Y_c, M_c=1, K_c=1, sigma_diag=True, channel=0, seed=0, matlab_compat=True)


def test_matlab_compat_true_retries_exactly_ten_times_then_raises_chained(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _reset_fake(fail_times=999, means_out=None, variances_out=None, weights_out=None)
    Y_c = np.random.default_rng(0).normal(size=(2, 50))

    with pytest.raises(InitialisationError, match=r"did not succeed in 10 attempts") as exc_info:
        _fit_channel(Y_c, M_c=1, K_c=1, sigma_diag=True, channel=5, seed=0, matlab_compat=True)

    assert len(_FakeGMM.calls) == 10
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_matlab_compat_false_raises_immediately_on_first_fit_failure(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _reset_fake(fail_times=999, means_out=None, variances_out=None, weights_out=None)
    Y_c = np.random.default_rng(0).normal(size=(2, 50))

    with pytest.raises(InitialisationError, match="fit failed on attempt 1"):
        _fit_channel(Y_c, M_c=1, K_c=1, sigma_diag=True, channel=5, seed=0, matlab_compat=False)

    assert len(_FakeGMM.calls) == 1


def test_retry_eventually_succeeds_within_budget(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _reset_fake(fail_times=3, means_out=np.zeros((1, 2)), variances_out=np.ones((1, 2)), weights_out=np.array([1.0]))
    Y_c = np.random.default_rng(0).normal(size=(2, 50))

    gmm = _fit_channel(Y_c, M_c=1, K_c=1, sigma_diag=True, channel=0, seed=0, matlab_compat=True)

    assert len(_FakeGMM.calls) == 4  # 3 failures + 1 success
    assert gmm.mu.shape == (2, 1, 1)


def test_jitter_and_regularizer_match_matlab_formula_exactly(monkeypatch):
    """Not just retry *counts* -- the actual jittered arrays and the
    regularizer passed to `GaussianMixture` on every attempt, checked against
    an independent recomputation of `em_lsim.m:572,580,586-587`'s own
    formula: `counter*0.05*std(current data, ddof=1)*randn(...)` added
    *cumulatively* to the current (not original) data, `counter` being the
    attempt that just failed (first jitter uses `counter=1`), and the
    regularizer recomputed from that same current data every retry (only
    attempt 1 uses the fixed, undeleted-data regularizer)."""
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _reset_fake(fail_times=3, means_out=np.zeros((1, 2)), variances_out=np.ones((1, 2)), weights_out=np.array([1.0]))
    Y_c = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 12.0, 9.0, 11.0]])  # (d_c=2, T=5), no NaN
    seed = 123

    _fit_channel(Y_c, M_c=1, K_c=1, sigma_diag=True, channel=0, seed=seed, matlab_compat=True)

    assert len(_FakeGMM.calls) == 4  # 3 failures + 1 success

    # independent recomputation, consuming the *same* seeded RNG stream in the
    # same order _fit_channel itself does: one `integers()` draw per attempt
    # (for random_state), then one `standard_normal()` draw per *failed*
    # attempt (for that attempt's jitter) -- not one per success.
    rng = np.random.default_rng(seed)
    expected_data = Y_c.T  # (5, 2): no NaN here, so listwise deletion drops nothing
    expected_reg = 0.01 * float(np.min(_channel_variance(Y_c)))

    for attempt in range(1, 5):
        rng.integers(0, 2**32 - 1)  # the random_state draw _fit_channel makes every attempt
        np.testing.assert_allclose(_FakeGMM.calls[attempt - 1], expected_data)
        np.testing.assert_allclose(_FakeGMM.init_kwargs[attempt - 1]["reg_covar"], expected_reg)
        if attempt < 4:
            noise_scale = attempt * 0.05 * np.std(expected_data, axis=0, ddof=1)
            expected_data = expected_data + noise_scale[None, :] * rng.standard_normal(expected_data.shape)
            expected_reg = 0.01 * float(np.min(np.var(expected_data, axis=0, ddof=1)))


def test_singleton_retry_jitter_and_regularizer_are_zero_not_nan(monkeypatch):
    """MATLAB's singleton convention (exactly one observation -> `std`/`var`
    of `0.0`, not NaN) must hold in the *retry* path too, not just the fixed
    first-attempt regularizer (`test_channel_variance_matlab_semantics`
    already covers that one). `np.std(...,ddof=1)`/`np.var(...,ddof=1)` on a
    single row give NaN, which would jitter the data to NaN and recompute a
    NaN regularizer -- checked here against the actual arrays, not just that
    the retry eventually succeeds."""
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _reset_fake(fail_times=1, means_out=np.zeros((1, 2)), variances_out=np.ones((1, 2)), weights_out=np.array([1.0]))
    Y_c = np.array([[5.0], [7.0]])  # (d_c=2, T=1): exactly one complete observation per dim
    seed = 7

    gmm = _fit_channel(Y_c, M_c=1, K_c=1, sigma_diag=True, channel=0, seed=seed, matlab_compat=True)

    assert len(_FakeGMM.calls) == 2  # 1 failure + 1 success
    np.testing.assert_allclose(_FakeGMM.calls[0], Y_c.T)  # attempt 1: unjittered
    # attempt 2: noise scale is 1*0.05*std([single value]) == 0.0 exactly (MATLAB's
    # singleton convention) -- the "jittered" data must be bit-identical to attempt 1's,
    # not NaN (np.std(...,ddof=1) on one row gives NaN, which would poison this).
    np.testing.assert_array_equal(_FakeGMM.calls[1], Y_c.T)
    assert not np.any(np.isnan(_FakeGMM.calls[1]))
    # the recomputed regularizer for attempt 2 must be exactly 0.0 too (var of a single
    # observation is 0 under MATLAB's convention), not NaN.
    assert _FakeGMM.init_kwargs[1]["reg_covar"] == 0.0
    assert gmm.mu.shape == (2, 1, 1)


def test_missing_sklearn_raises_initialisation_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.mixture", None)
    with pytest.raises(InitialisationError, match="scikit-learn"):
        init_module._import_gaussian_mixture()


# --------------------------------------------------------------------------
# Validation of returned parameters, and using them with em_lsim.
# --------------------------------------------------------------------------

@requires_sklearn
def test_initialize_lsim_params_returns_validated_params():
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 2), num_gmm_component=(2, 1))
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(3, 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = initialize_lsim_params(structure, [Y], sigma_diag=False, seed=0)
    params.validate()  # must not raise -- already validated internally, re-checked here


@requires_sklearn
def test_initialized_params_used_successfully_with_em_lsim(fixtures_dir):
    fx = load_em_trace_fixture(fixtures_dir, "c3_het_diag")
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    Y = fx.Y_trials[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params0 = initialize_lsim_params(structure, [Y], sigma_diag=bool(fx.sigma_diag), seed=0)
        result = em_lsim(
            structure, params0, [Y], max_iterations=5, fb_constants=FbConstants(variant="sed"), check_convergence=False, validate="off"
        )

    assert len(result.loglik_trace) == 5
    assert all(np.isfinite(x) for x in result.loglik_trace)
    result.params.validate()


# --------------------------------------------------------------------------
# P5 statistical gate: seeded initializer runs vs a REAL MATLAB no-init_model
# run (fine_gmm_fitting, via tools/matlab/export_init_reference.m), compared
# by resulting EM log-likelihood -- not component parity. See SCHEMA.md's
# "Init-reference fixture" section for exactly why this fixture exists and
# is not any of the EM-trace fixtures' `params_in`/`init_*` fields.
# --------------------------------------------------------------------------

@requires_sklearn
def test_seeded_init_reaches_comparable_loglik_to_matlab_init(fixtures_dir):
    """Measured, not guessed (same methodology throughout this port): 5 seeds
    of `initialize_lsim_params`, each run through 10 `em_lsim` iterations on
    `init_reference.npz`'s own data, are compared against that fixture's own
    `loglik_params_in[-1]` -- MATLAB's `em_lsim.m` run through its *actual*
    no-`init_model` branch (`fine_gmm_fitting`, seeded), the same number of
    iterations, on the identical data. Loaded (and shape/NaN/Inf-validated)
    through `load_init_reference_fixture`, not a raw `np.load`.

    Measured worst-case relative gap vs the MATLAB reference: `5.3e-3`
    (seed 0, `-812.47` vs MATLAB's `-816.82`); measured cross-seed spread
    among the 5 sklearn-seeded runs: `7.4e-3`. Both are roughly an order of
    magnitude tighter than the very first version of this gate (`4.6e-2` /
    `3.7e-2`) -- that version was comparing against the wrong MATLAB run
    entirely (see the module docstring); a real no-`init_model` MATLAB run
    and a real sklearn-based one land much closer to each other on the same
    data than either does to an arbitrary `mk_model` ground-truth parameter
    set fed in as an `init_model`. Margins below (`0.02` each) carry
    ~2.7x/~3.8x headroom over the measured values -- deliberately tighter
    than an earlier `0.10` draft: generous enough that this comparison
    crossing two independent GMM-fitting implementations doesn't make the
    test flaky, but still tight enough to actually catch a meaningful
    regression (a `0.10` margin would have silently passed even a
    genuinely-broken initializer landing 5-9x worse than what's measured)."""
    fx = load_init_reference_fixture(fixtures_dir)
    structure = LSIMStructure(
        tuple(int(m) for m in fx.channel_num_states),
        tuple(int(x) for x in fx.dim_observation),
        tuple(int(k) for k in fx.num_gmm_component),
    )
    Y = fx.Y
    sigma_diag = fx.sigma_diag
    N_ITER = fx.num_iters
    matlab_loglik = float(fx.loglik_params_in[-1])

    VS_MATLAB_MARGIN = 0.02
    CROSS_SEED_MARGIN = 0.02

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seeded_logliks = []
        for seed in range(5):
            params0 = initialize_lsim_params(structure, [Y], sigma_diag=sigma_diag, seed=seed)
            result = em_lsim(
                structure, params0, [Y], max_iterations=N_ITER, fb_constants=FbConstants(variant="sed"), check_convergence=False, validate="off"
            )
            seeded_logliks.append(result.loglik_trace[-1])

    seeded_logliks = np.array(seeded_logliks)
    rel_gap_vs_matlab = np.abs(seeded_logliks - matlab_loglik) / abs(matlab_loglik)
    assert np.all(rel_gap_vs_matlab < VS_MATLAB_MARGIN), (rel_gap_vs_matlab, matlab_loglik, seeded_logliks)

    cross_seed_spread = (seeded_logliks.max() - seeded_logliks.min()) / abs(seeded_logliks.mean())
    assert cross_seed_spread < CROSS_SEED_MARGIN, (cross_seed_spread, seeded_logliks)



# --------------------------------------------------------------------------
# Opt-in speed controls: max_samples_per_channel (per-channel row sampling)
# and n_jobs (channel-parallel worker processes). Neither is part of MATLAB
# parity; both default to the unchanged behaviour tested above.
# --------------------------------------------------------------------------

import concurrent.futures  # noqa: E402

from chmm_lsim.emissions.gmm_init import _ChannelFitJob, _fit_channel_job_in_worker, _prepare_channel_job


def _small_hetero_problem(seed=21):
    """Heterogeneous M_c/d_c/K_c, ragged trials, missing values in a 2-D channel."""
    structure = LSIMStructure(channel_state_num=(2, 3, 2), channel_obs_dim=(1, 2, 1), num_gmm_component=(1, 2, 1))
    rng = np.random.default_rng(seed)
    Y_trials = [rng.normal(size=(4, 150)), rng.normal(size=(4, 90))]
    Y_trials[0][1, [4, 50]] = np.nan  # channel 1, dim 0
    Y_trials[1][2, [7]] = np.nan  # channel 1, dim 1
    return structure, Y_trials


def _assert_params_equal(a, b):
    np.testing.assert_array_equal(a.pi_0, b.pi_0)
    np.testing.assert_array_equal(a.transition, b.transition)
    np.testing.assert_array_equal(a.coupling, b.coupling)
    for ga, gb in zip(a.emissions, b.emissions):
        np.testing.assert_array_equal(ga.mu, gb.mu)
        np.testing.assert_array_equal(ga.sigma, gb.sigma)
        np.testing.assert_array_equal(ga.weights, gb.weights)


def test_default_arguments_pass_every_complete_row_and_never_draw_a_subsample(monkeypatch):
    """Unchanged defaults: max_samples_per_channel=None fits on every complete
    row, in time order, and the replicate seeds do not depend on the new code."""
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    structure, Y_trials = _small_hetero_problem()
    _reset_fake(0)
    initialize_lsim_params(structure, Y_trials, sigma_diag=True, seed=9)
    default_calls = [c.copy() for c in _FakeGMM.calls]
    default_seeds = [kw["random_state"] for kw in _FakeGMM.init_kwargs]

    for c in range(structure.C):
        Y_c = np.concatenate([Y[structure.dim_slice(c), :] for Y in Y_trials], axis=1)
        expected = Y_c[:, ~np.isnan(Y_c).any(axis=0)].T
        np.testing.assert_array_equal(default_calls[c], expected)

    _reset_fake(0)
    initialize_lsim_params(structure, Y_trials, sigma_diag=True, seed=9, max_samples_per_channel=10**9)
    assert [kw["random_state"] for kw in _FakeGMM.init_kwargs] == default_seeds
    for got, exp in zip(_FakeGMM.calls, default_calls):
        np.testing.assert_array_equal(got, exp)


def test_max_samples_per_channel_samples_complete_rows_deterministically(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    rng = np.random.default_rng(5)
    Y_c = rng.normal(size=(2, 50))  # multidimensional channel
    Y_c[1, [3, 8, 13, 21, 40]] = np.nan  # 45 complete rows
    complete = Y_c[:, ~np.isnan(Y_c).any(axis=0)].T

    def fitted_rows(seed):
        _reset_fake(0, np.zeros((1, 2)), np.ones((1, 2)), np.array([1.0]))
        _fit_channel(Y_c, 1, 1, sigma_diag=True, channel=0, seed=seed, matlab_compat=True, max_samples_per_channel=20)
        return _FakeGMM.calls[0]

    a, b, c = fitted_rows(0), fitted_rows(0), fitted_rows(1)
    assert a.shape == (20, 2)
    assert not np.isnan(a).any()
    np.testing.assert_array_equal(a, b)  # same seed -> same subset
    assert not np.array_equal(a, c)  # different seed -> different subset
    idx = [int(np.flatnonzero((complete == row).all(axis=1))[0]) for row in a]
    assert idx == sorted(idx) and len(set(idx)) == 20  # distinct complete rows, time order kept


@pytest.mark.parametrize("cap", [45, 46, 10**6])
def test_max_samples_per_channel_at_or_above_available_rows_is_a_no_op(monkeypatch, cap):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    Y_c = np.random.default_rng(6).normal(size=(2, 50))
    Y_c[0, [1, 2, 3, 4, 5]] = np.nan  # 45 complete rows available
    _reset_fake(0, np.zeros((1, 2)), np.ones((1, 2)), np.array([1.0]))
    _fit_channel(Y_c, 1, 1, sigma_diag=True, channel=0, seed=0, matlab_compat=True, max_samples_per_channel=cap)
    np.testing.assert_array_equal(_FakeGMM.calls[0], Y_c[:, ~np.isnan(Y_c).any(axis=0)].T)


def test_sampling_keeps_full_data_regularizer_and_first_replicate_seed(monkeypatch):
    """Regularizer from the complete undeleted data; the first attempt's
    random_state is identical with and without sampling. Retries are
    deterministic for a fixed configuration, but draw sample-sized noise, so
    their later random_state values need not match the unsampled run."""
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    Y_c = np.random.default_rng(8).normal(size=(2, 200)) * np.array([[1.0], [3.0]])
    Y_c[0, 17] = np.nan
    var0 = np.var(np.delete(Y_c[0], 17), ddof=1)
    var1 = np.var(Y_c[1], ddof=1)

    _reset_fake(fail_times=2)
    _fit_channel(Y_c, 1, 1, sigma_diag=True, channel=0, seed=11, matlab_compat=True)
    full = [dict(kw) for kw in _FakeGMM.init_kwargs]

    _reset_fake(fail_times=2)
    _fit_channel(Y_c, 1, 1, sigma_diag=True, channel=0, seed=11, matlab_compat=True, max_samples_per_channel=25)
    sampled = [dict(kw) for kw in _FakeGMM.init_kwargs]

    np.testing.assert_allclose(sampled[0]["reg_covar"], 0.01 * min(var0, var1))
    assert sampled[0]["reg_covar"] == full[0]["reg_covar"]
    assert sampled[0]["random_state"] == full[0]["random_state"]
    assert len(sampled) == 3

    sampled_calls = [c.copy() for c in _FakeGMM.calls]
    _reset_fake(fail_times=2)
    _fit_channel(Y_c, 1, 1, sigma_diag=True, channel=0, seed=11, matlab_compat=True, max_samples_per_channel=25)
    again = [dict(kw) for kw in _FakeGMM.init_kwargs]
    assert [kw["random_state"] for kw in again] == [kw["random_state"] for kw in sampled]  # deterministic
    assert len(_FakeGMM.calls) == len(sampled_calls) == 3
    for a, b in zip(_FakeGMM.calls, sampled_calls):  # identical jittered fit arrays across runs
        np.testing.assert_array_equal(a, b)


def test_sampled_retry_never_allocates_noise_for_the_full_channel(monkeypatch):
    """P10e: the jitter retry of a sampled channel must draw noise only for the
    sampled rows. Records every standard_normal request and the tracemalloc
    peak while a sampled 400k-row channel exhausts its retry budget."""
    import tracemalloc

    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    n_complete, cap, d_c = 400_000, 1_000, 2
    Y_c = np.random.default_rng(0).normal(size=(d_c, n_complete))
    job = _prepare_channel_job(Y_c, 1, 1, sigma_diag=True, channel=0, seed=5, matlab_compat=True, max_samples_per_channel=cap)
    assert job.fit_data.shape == (cap, d_c) and job.n_complete == n_complete

    requested = []
    real_default_rng = np.random.default_rng

    class _RecordingGenerator:
        def __init__(self, gen):
            self._gen = gen

        def standard_normal(self, size=None, *a, **k):
            requested.append(int(np.prod(size)) if size is not None else 1)
            return self._gen.standard_normal(size, *a, **k)

        def __getattr__(self, name):
            return getattr(self._gen, name)

    monkeypatch.setattr(init_module.np.random, "default_rng", lambda *a, **k: _RecordingGenerator(real_default_rng(*a, **k)))
    _reset_fake(fail_times=999)
    tracemalloc.start()
    try:
        with pytest.raises(InitialisationError, match="did not succeed in 10 attempts"):
            init_module._fit_channel_job(job)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(requested) == 9  # one noise draw per retry
    assert max(requested) <= cap * d_c
    assert peak < 0.05 * n_complete * d_c * 8  # far below one full-channel noise array


def test_prepared_job_carries_only_the_fit_array_not_the_full_channel():
    Y_c = np.random.default_rng(1).normal(size=(1, 5000))
    job = _prepare_channel_job(Y_c, 2, 1, sigma_diag=True, channel=4, seed=3, matlab_compat=True, max_samples_per_channel=100)
    assert isinstance(job, _ChannelFitJob)
    assert job.fit_data.shape == (100, 1) and job.fit_data.flags.c_contiguous
    assert job.channel == 4 and job.seed == 3 and (job.M_c, job.K_c) == (2, 1)
    np.testing.assert_allclose(job.reg_covar, 0.01 * np.var(Y_c, ddof=1))


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, False, "100"])
def test_max_samples_per_channel_rejects_non_positive_int(bad):
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    with pytest.raises(ValueError, match="max_samples_per_channel must be a positive int"):
        initialize_lsim_params(structure, [np.zeros((1, 10))], sigma_diag=True, max_samples_per_channel=bad)


def test_max_samples_per_channel_smaller_than_components_rejected():
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 1), num_gmm_component=(1, 2))
    with pytest.raises(ValueError, match=r"max_samples_per_channel=5 is smaller than the largest M_c\*K_c=6"):
        initialize_lsim_params(structure, [np.zeros((2, 10))], sigma_diag=True, max_samples_per_channel=5)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, None])
def test_n_jobs_rejects_non_positive_int(bad):
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    with pytest.raises(ValueError, match="n_jobs must be a positive int"):
        initialize_lsim_params(structure, [np.zeros((1, 10))], sigma_diag=True, n_jobs=bad)


@requires_sklearn
@pytest.mark.parametrize("max_samples_per_channel", [None, 120])
def test_channel_parallel_matches_serial_exactly(max_samples_per_channel):
    """Heterogeneous M_c/d_c/K_c, ragged trials, missing values, both the full
    and the sampled path. Exact in this environment; not claimed cross-platform."""
    structure, Y_trials = _small_hetero_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = initialize_lsim_params(structure, Y_trials, sigma_diag=False, seed=3, max_samples_per_channel=max_samples_per_channel)
        parallel = initialize_lsim_params(
            structure, Y_trials, sigma_diag=False, seed=3, max_samples_per_channel=max_samples_per_channel, n_jobs=3
        )
    _assert_params_equal(serial, parallel)


def test_channel_parallel_defaults_to_spawn_not_fork():
    assert init_module._PROCESS_START_METHOD == "spawn"


@requires_sklearn
def test_channel_parallel_works_under_spawn_start_method(monkeypatch):
    """Workers must not rely on fork-inherited state. A forked worker was
    measured to die intermittently once the coordinator had run native
    multi-threaded code; the initializer therefore spawns."""
    monkeypatch.setattr(init_module, "_PROCESS_START_METHOD", "spawn")
    with warnings.catch_warnings():  # warm the coordinator's native thread pools first
        warnings.simplefilter("ignore")
        from sklearn.mixture import GaussianMixture

        GaussianMixture(3, covariance_type="diag").fit(np.random.default_rng(0).normal(size=(20000, 1)))
    structure, Y_trials = _small_hetero_problem(seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = initialize_lsim_params(structure, Y_trials, sigma_diag=True, seed=1)
        spawned = initialize_lsim_params(structure, Y_trials, sigma_diag=True, seed=1, n_jobs=2)
    _assert_params_equal(serial, spawned)


def _fresh_thread_env(monkeypatch):
    """Spawned workers inherit the parent's environment; drop any externally
    imposed thread caps so a passing test cannot rely on them."""
    for name in init_module._WORKER_THREAD_ENV:
        monkeypatch.delenv(name, raising=False)


@requires_sklearn
def test_production_worker_runs_every_native_pool_single_threaded_during_the_real_fit(monkeypatch):
    """P10e item 1. Fresh spawned process, production pool factory and
    initializer, production worker entry point, real GaussianMixture fit;
    every BLAS/OpenMP pool visible during the fit must report one thread."""
    _fresh_thread_env(monkeypatch)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))  # lets spawned workers import the probe module
    import _init_worker_probe

    assert init_module._PROCESS_START_METHOD == "spawn"
    job = _prepare_channel_job(np.random.default_rng(0).normal(size=(1, 20000)), 3, 1, sigma_diag=True, channel=0, seed=0, matlab_compat=True)
    with init_module._channel_pool(1) as pool:
        gmm, record = pool.submit(_init_worker_probe.fit_and_record, job).result()

    assert record["pid"] != __import__("os").getpid()
    assert record["worker_limit_active_before_job"], "the pool initializer must apply the limit before any job runs"
    pools = record["pools_during_fit"]
    assert any(p["api"] == "openmp" for p in pools), f"expected scikit-learn's OpenMP runtime to be loaded: {pools}"
    assert all(p["num_threads"] == 1 for p in pools), pools
    assert all(v == "1" for v in record["env"].values()), record["env"]
    assert gmm.M_c == 3


@requires_sklearn
def test_parallel_initialization_leaves_the_parent_process_thread_limits_untouched(monkeypatch):
    from threadpoolctl import threadpool_info

    structure, Y_trials = _small_hetero_problem(seed=7)
    env_before = {k: __import__("os").environ.get(k) for k in init_module._WORKER_THREAD_ENV}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_lsim_params(structure, Y_trials[:1], sigma_diag=True, seed=0)  # load the native pools in the parent
        before = sorted((p["filepath"], p["num_threads"]) for p in threadpool_info())
        initialize_lsim_params(structure, Y_trials, sigma_diag=True, seed=0, n_jobs=2)
        after = sorted((p["filepath"], p["num_threads"]) for p in threadpool_info())
    assert before == after
    assert {k: __import__("os").environ.get(k) for k in init_module._WORKER_THREAD_ENV} == env_before
    assert init_module._WORKER_THREAD_LIMIT is None  # the limit only ever exists inside workers


@requires_sklearn
def test_serial_initialization_applies_no_thread_limit(monkeypatch):
    """n_jobs=1 is unchanged: the fit runs in the caller's process with the
    caller's own thread configuration."""
    from threadpoolctl import threadpool_info

    seen = []
    real_import = init_module._import_gaussian_mixture

    def recording_import():
        GaussianMixture = real_import()

        class RecordingGMM(GaussianMixture):
            def fit(self, X, y=None):
                seen.append(sorted((p["filepath"], p["num_threads"]) for p in threadpool_info()))
                return super().fit(X, y)

        return RecordingGMM

    structure, Y_trials = _small_hetero_problem(seed=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_import()
        baseline = sorted((p["filepath"], p["num_threads"]) for p in threadpool_info())
        monkeypatch.setattr(init_module, "_import_gaussian_mixture", recording_import)
        initialize_lsim_params(structure, Y_trials, sigma_diag=True, seed=0)
    assert seen and all(s == baseline for s in seen)
    assert init_module._WORKER_THREAD_LIMIT is None


class _SyncFuture(concurrent.futures.Future):
    pass


class _InstrumentedExecutor:
    """Runs each task synchronously at submit, and records how many submitted
    tasks have not yet been harvested (exception()/result() called)."""

    max_outstanding = 0
    submitted = 0
    worker_counts: list[int] = []

    initializers: list = []

    def __init__(self, max_workers=None, mp_context=None, initializer=None):
        type(self).worker_counts.append(max_workers)
        type(self).initializers.append(initializer)  # recorded, not run: it would limit THIS test process
        self._outstanding = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = _SyncFuture()
        owner = self

        def _harvest(orig):
            def inner(*a, **k):
                owner._outstanding.discard(fut)
                return orig(*a, **k)
            return inner

        fut.exception = _harvest(fut.exception)
        fut.result = _harvest(fut.result)
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            fut.set_exception(exc)
        self._outstanding.add(fut)
        type(self).submitted += 1
        type(self).max_outstanding = max(type(self).max_outstanding, len(self._outstanding))
        return fut


def _install_instrumented_executor(monkeypatch):
    """In-process stand-in for the channel pool. The production worker entry
    point calls `_init_channel_worker()`, which in a real worker sets thread
    environment variables and a persistent threadpoolctl limit; here that would
    modify THIS test process (and need threadpoolctl in a core-only install), so
    it is stubbed. The real initializer is tested in a spawned worker."""
    _InstrumentedExecutor.max_outstanding = 0
    _InstrumentedExecutor.submitted = 0
    _InstrumentedExecutor.worker_counts = []
    _InstrumentedExecutor.initializers = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InstrumentedExecutor)
    monkeypatch.setattr(init_module, "_init_channel_worker", lambda: None)
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",  # pretend ONLY sklearn/threadpoolctl are installed (core-only runs)
                        lambda name, *a, **k: object() if name in ("sklearn", "threadpoolctl") else real_find_spec(name, *a, **k))


def test_channel_jobs_are_bounded_by_n_jobs_and_collected_in_channel_order(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _install_instrumented_executor(monkeypatch)
    C = 7
    structure = LSIMStructure(channel_state_num=tuple(range(1, C + 1)), channel_obs_dim=(1,) * C, num_gmm_component=(1,) * C)
    Y = np.random.default_rng(3).normal(size=(C, 60))
    _reset_fake(0)
    params = initialize_lsim_params(structure, [Y], sigma_diag=True, n_jobs=2)
    assert _InstrumentedExecutor.worker_counts == [2]
    assert len(_InstrumentedExecutor.initializers) == 1  # the pool was given a worker initializer
    assert _InstrumentedExecutor.submitted == C
    assert _InstrumentedExecutor.max_outstanding <= 2  # never all C at once
    assert [g.M_c for g in params.emissions] == list(range(1, C + 1))  # channel order


def test_worker_failure_is_raised_for_the_lowest_failing_channel_with_channel_named(monkeypatch):
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _install_instrumented_executor(monkeypatch)
    structure = LSIMStructure(channel_state_num=(1,) * 5, channel_obs_dim=(1,) * 5, num_gmm_component=(1,) * 5)
    Y = np.random.default_rng(3).normal(size=(5, 40))
    real = init_module._fit_channel_job_in_worker

    def flaky(job):
        if job.channel in (2, 4):
            raise RuntimeError(f"boom on {job.channel}")
        return real(job)

    monkeypatch.setattr(init_module, "_fit_channel_job_in_worker", flaky)
    _reset_fake(0)
    with pytest.raises(InitialisationError, match=r"channel 2: initialization worker failed: RuntimeError: boom on 2") as info:
        initialize_lsim_params(structure, [Y], sigma_diag=True, n_jobs=2)
    assert isinstance(info.value.__cause__, RuntimeError)


@requires_sklearn
def test_real_worker_process_failure_names_the_channel():
    """Channel 1 has fewer complete rows than components; under matlab_compat
    its worker exhausts the retry budget in a real process."""
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 1), num_gmm_component=(1, 2))
    Y = np.random.default_rng(2).normal(size=(2, 40))
    Y[1, 3:] = np.nan  # channel 1: 3 complete rows, needs 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(InitialisationError, match=r"channel 1: GaussianMixture fit did not succeed in 10 attempts"):
            initialize_lsim_params(structure, [Y], sigma_diag=True, n_jobs=2)


@requires_sklearn
def test_coordinator_side_failure_is_raised_with_channel_named_in_parallel_mode():
    structure = LSIMStructure(channel_state_num=(1, 1, 1), channel_obs_dim=(1, 1, 1), num_gmm_component=(1, 1, 1))
    Y = np.random.default_rng(2).normal(size=(3, 40))
    Y[1, :] = np.nan  # channel 1 has no non-NaN observation -> NaN regularizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(InitialisationError, match=r"channel 1.*NaN"):
            initialize_lsim_params(structure, [Y], sigma_diag=True, n_jobs=2)



def test_fake_executor_tests_leave_this_process_untouched(monkeypatch):
    """Regression (P10e closure): the in-process executor tests must not apply
    worker thread settings to the test process itself."""
    import os

    env_before = {k: os.environ.get(k) for k in init_module._WORKER_THREAD_ENV}
    monkeypatch.setattr(init_module, "_import_gaussian_mixture", lambda: _FakeGMM)
    _install_instrumented_executor(monkeypatch)
    structure = LSIMStructure(channel_state_num=(1, 1, 1), channel_obs_dim=(1, 1, 1), num_gmm_component=(1, 1, 1))
    _reset_fake(0)
    initialize_lsim_params(structure, [np.random.default_rng(4).normal(size=(3, 30))], sigma_diag=True, n_jobs=2)
    assert {k: os.environ.get(k) for k in init_module._WORKER_THREAD_ENV} == env_before
    assert init_module._WORKER_THREAD_LIMIT is None


@pytest.mark.parametrize("missing,match", [("sklearn", "scikit-learn is required"), ("threadpoolctl", "requires threadpoolctl")])
def test_parallel_initialization_reports_a_missing_dependency_clearly(monkeypatch, missing, match):
    """Without the check, every spawned worker's initializer died and the user saw a
    misleading 'worker pool broke ... killed for memory' error."""
    import importlib.util
    import sys as _sys

    real_find_spec = importlib.util.find_spec
    deps = ("sklearn", "threadpoolctl")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: (None if name == missing else object()) if name in deps
                        else real_find_spec(name, *a, **k))
    if missing == "sklearn":
        monkeypatch.setitem(_sys.modules, "sklearn.mixture", None)
    created = []
    monkeypatch.setattr(init_module, "_channel_pool", lambda n: created.append(n))
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    with pytest.raises(InitialisationError, match=match):
        initialize_lsim_params(structure, [np.random.default_rng(0).normal(size=(2, 40))], sigma_diag=True, n_jobs=2)
    assert created == []  # failed before any worker process was started
