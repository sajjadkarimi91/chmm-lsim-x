"""P10f gates: chunk-vectorized global variance, chunk-vectorized observation
validation, and the private already-validated route inside the mini-batch
training drivers.

1. `compute_global_nanvar` parity with the pre-P10f per-trial implementation
   (reproduced verbatim below as `_reference_global_nanvar`) on every P10a
   scheduling fixture, missing values, fully missing channels, singleton
   observations, unequal lengths and large offsets, for several chunk sizes;
   whole-dataset vs merged chunks; bounded allocation; exactly once per run.
2. Chunked validation reports exactly the per-trial `validate_observations`
   errors and warnings, in the same order, and nothing reaches an accumulator
   before validation.
3. Training validates observations once at the boundary, never again inside
   E-steps/scoring, still validates parameters every update, gives
   bit-identical results to the fully re-validating route, and cannot be
   silently fed a trial mutated after validation.
"""

from __future__ import annotations

import inspect
import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import P10A_SCENARIO_IDS, load_p10a_scheduling_fixture  # noqa: E402

import chmm_lsim.emissions.gmm as gmm_mod  # noqa: E402
import chmm_lsim.models.lsim.em as em  # noqa: E402
import chmm_lsim.core.validate as val  # noqa: E402
from chmm_lsim.core.constants import FbConstants  # noqa: E402
from chmm_lsim.emissions.gmm import _welford_batch, _welford_merge, _nanvar_from_welford, compute_global_nanvar  # noqa: E402
from chmm_lsim.models.lsim.em import e_step  # noqa: E402
from chmm_lsim.emissions.gmm import ChannelGMM  # noqa: E402
from chmm_lsim.models.lsim.params import LSIMParams  # noqa: E402
from chmm_lsim.models.lsim.simulate import simulate_lsim  # noqa: E402
from chmm_lsim.core.structure import LSIMStructure  # noqa: E402
from chmm_lsim.core.validate import ValidationError, validate_observation_list, validate_observations  # noqa: E402

CHUNK_SIZES = (1, 3, 17, 64, 1000, None)


def _reference_global_nanvar(structure, Y_trials):
    """The pre-P10f `compute_global_nanvar` body, verbatim: per-trial
    validation, then one observed-only Welford batch per trial per channel."""
    if not Y_trials:
        raise ValueError("Y_trials must contain at least one trial")
    for i, Y in enumerate(Y_trials):
        try:
            validate_observations(structure, Y)
        except ValidationError as exc:
            raise ValidationError(f"Y_trials[{i}]: {exc}") from exc
    result = []
    for c in range(structure.C):
        dim_sl = structure.dim_slice(c)
        d_c = structure.channel_obs_dim[c]
        n_obs = np.zeros(d_c)
        welford_mean = np.zeros(d_c)
        welford_M2 = np.zeros(d_c)
        for Y in Y_trials:
            Y_c = Y[dim_sl, :]
            o = ~np.isnan(Y_c)
            o_f = o.astype(np.float64)
            Y_masked = np.where(o, Y_c, 0.0)
            obs_W, obs_mean, obs_M2 = _welford_batch(o_f, Y_masked)
            n_obs, welford_mean, welford_M2 = _welford_merge(n_obs, welford_mean, welford_M2, obs_W, obs_mean, obs_M2)
        result.append(_nanvar_from_welford(n_obs, welford_M2))
    return result


def _assert_nanvar_equal(new, ref, rtol=1e-12, atol=1e-14):
    assert len(new) == len(ref)
    for a, b in zip(new, ref):
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def _structure(M=(2, 3, 2), d=(1, 2, 3)):
    return LSIMStructure(channel_state_num=M, channel_obs_dim=d, num_gmm_component=(1,) * len(M))


def _ragged_trials(structure, n=40, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        T = int(rng.integers(1, 23))
        Y = rng.normal(loc=rng.normal(size=(structure.D, 1)) * 3, scale=2.0, size=(structure.D, T))
        if nan_frac:
            Y[rng.random(Y.shape) < nan_frac] = np.nan
        out.append(Y)
    return out


# ==========================================================================
# 1. Global variance: parity, edge cases, chunk invariance, bounded memory.
# ==========================================================================


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_global_nanvar_matches_reference_on_p10a_fixtures(fixtures_dir, scenario_id, chunk):
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()), channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    Y_trials = fx.Y_trials()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_nanvar_equal(compute_global_nanvar(structure, Y_trials, max_chunk_observations=chunk), _reference_global_nanvar(structure, Y_trials))


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_global_nanvar_missing_values_and_unequal_lengths(chunk):
    structure = _structure()
    Y_trials = _ragged_trials(structure, n=60, seed=1, nan_frac=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_nanvar_equal(compute_global_nanvar(structure, Y_trials, max_chunk_observations=chunk), _reference_global_nanvar(structure, Y_trials))


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_global_nanvar_fully_missing_channel_is_nan_and_warns(chunk):
    structure = _structure()
    Y_trials = _ragged_trials(structure, n=10, seed=2)
    for Y in Y_trials:
        Y[structure.dim_slice(1), :] = np.nan  # channel 1 never observed anywhere
    with pytest.warns(UserWarning, match="channel 1 is NaN for every timestep"):
        new = compute_global_nanvar(structure, Y_trials, max_chunk_observations=chunk)
    assert np.all(np.isnan(new[1]))
    assert np.all(np.isfinite(new[0])) and np.all(np.isfinite(new[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_nanvar_equal(new, _reference_global_nanvar(structure, Y_trials))


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_global_nanvar_singleton_observation_is_zero(chunk):
    structure = _structure(M=(2,), d=(2,))
    Y1 = np.full((2, 5), np.nan)
    Y1[0, 3] = 7.0  # dimension 0: exactly one observation in the whole dataset
    Y1[1, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
    Y2 = np.full((2, 1), np.nan)
    Y2[1, 0] = 6.0
    new = compute_global_nanvar(structure, [Y1, Y2], max_chunk_observations=chunk)
    assert new[0][0] == 0.0
    np.testing.assert_allclose(new[0][1], np.var([1, 2, 3, 4, 5, 6], ddof=1), rtol=1e-14)
    _assert_nanvar_equal(new, _reference_global_nanvar(structure, [Y1, Y2]))


@pytest.mark.parametrize("offset", [1e10, 1e12])
@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_global_nanvar_large_offsets(offset, chunk):
    """Integer-valued deviations around a huge offset: the exact sample variance
    is known in closed form, and both implementations must stay close to it
    (central moments, never a raw sum of squares)."""
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    rng = np.random.default_rng(3)
    ints = [rng.integers(-50, 51, size=int(rng.integers(1, 30))).astype(np.float64) for _ in range(200)]
    Y_trials = [(offset + x)[None, :] for x in ints]
    exact = np.var(np.concatenate(ints), ddof=1)
    new = compute_global_nanvar(structure, Y_trials, max_chunk_observations=chunk)
    ref = _reference_global_nanvar(structure, Y_trials)
    # Measured relative error vs the exact variance, over CHUNK_SIZES: offset 1e10 -> reference 2.8e-10, chunked
    # 0 (one chunk) to 2.3e-9; offset 1e12 -> reference 8.3e-8, chunked 0 to 8.3e-8. The residual is round-off
    # in `_welford_merge`'s mean update between merged batches (ulp of 1e10 is 1.9e-6, of 1e12 is 1.2e-4) --
    # the same merge the reference uses; each chunk is centered before its own sum, so a single chunk is exact.
    tol = 1e-6 if offset >= 1e12 else 1e-8
    np.testing.assert_allclose(new[0], exact, rtol=tol)
    np.testing.assert_allclose(ref[0], exact, rtol=tol)
    np.testing.assert_allclose(new[0], ref[0], rtol=2 * tol)


def test_global_nanvar_whole_dataset_equals_merged_chunks():
    structure = _structure()
    Y_trials = _ragged_trials(structure, n=300, seed=4, nan_frac=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = compute_global_nanvar(structure, Y_trials, max_chunk_observations=10**12)
        for chunk in (1, 6, 50, 999):
            _assert_nanvar_equal(compute_global_nanvar(structure, Y_trials, max_chunk_observations=chunk), whole)


def test_global_nanvar_chunks_are_bounded_not_dataset_sized(monkeypatch):
    """The concatenated chunk handed to the accumulator never exceeds the
    budget (except a single over-budget trial), and peak traced allocation is
    a small fraction of the dataset -- nothing proportional to N is built."""
    structure = _structure(M=(3, 3), d=(4, 4))
    rng = np.random.default_rng(5)
    Y_trials = [rng.normal(size=(structure.D, 20)) for _ in range(4000)]  # 4000 x 8 x 20 = 640,000 values, 4.9 MiB
    dataset_bytes = sum(Y.nbytes for Y in Y_trials)
    budget = 8 * 20 * 25  # 25 trials per chunk
    seen = []
    real = val._iter_validated_observation_chunks

    def spy(*args, **kwargs):
        for item in real(*args, **kwargs):
            seen.append(item[2].size)
            yield item

    monkeypatch.setattr(gmm_mod, "_iter_validated_observation_chunks", spy)
    tracemalloc.start()
    try:
        compute_global_nanvar(structure, Y_trials, max_chunk_observations=budget)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert max(seen) <= budget
    assert len(seen) == 160
    assert peak < dataset_bytes / 20, f"peak traced allocation {peak} bytes vs dataset {dataset_bytes} bytes"


def test_global_nanvar_single_trial_larger_than_budget_forms_its_own_chunk():
    structure = _structure(M=(2,), d=(1,))
    Y_trials = [np.arange(10.0)[None, :], np.arange(1000.0)[None, :], np.arange(5.0)[None, :]]
    _assert_nanvar_equal(compute_global_nanvar(structure, Y_trials, max_chunk_observations=50), _reference_global_nanvar(structure, Y_trials))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_global_nanvar_rejects_invalid_chunk_budget(bad):
    structure = _structure()
    with pytest.raises(ValidationError, match="max_chunk_observations"):
        compute_global_nanvar(structure, _ragged_trials(structure, n=3), max_chunk_observations=bad)


def test_global_nanvar_accumulates_float64_for_float32_trials():
    structure = _structure()
    Y64 = _ragged_trials(structure, n=30, seed=6)
    Y32 = [Y.astype(np.float32) for Y in Y64]
    new = compute_global_nanvar(structure, Y32, max_chunk_observations=40)
    assert all(v.dtype == np.float64 for v in new)
    _assert_nanvar_equal(new, _reference_global_nanvar(structure, Y32))


def test_global_nanvar_never_accumulates_unvalidated_observations(monkeypatch):
    """A later chunk holds an Inf: every value that ever reaches `_welford_batch`
    is finite-or-masked, and the call raises for the Inf trial."""
    structure = _structure()
    Y_trials = _ragged_trials(structure, n=50, seed=7)
    Y_trials[37][0, 0] = np.inf
    real = gmm_mod._welford_batch

    def spy(weight, value):
        assert np.all(np.isfinite(value)), "an unvalidated (Inf) observation reached the accumulator"
        return real(weight, value)

    monkeypatch.setattr(gmm_mod, "_welford_batch", spy)
    with pytest.raises(ValidationError, match=r"Y_trials\[37\].*Inf"):
        compute_global_nanvar(structure, Y_trials, max_chunk_observations=30)


# ==========================================================================
# 2. Chunked validation: identical errors, warnings and order.
# ==========================================================================


def _reference_validate(structure, Y_trials, label):
    for i, Y in enumerate(Y_trials):
        try:
            validate_observations(structure, Y)
        except ValidationError as exc:
            raise ValidationError(f"{label}[{i}]: {exc}") from exc


def _outcome(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn()
            err = None
        except Exception as exc:  # noqa: BLE001
            err = (type(exc), str(exc))
    return err, [str(w.message) for w in rec]


def _malformed_cases(structure):
    rng = np.random.default_rng(8)
    good = lambda T=6: rng.normal(size=(structure.D, T))  # noqa: E731
    allnan = good(4)
    allnan[structure.dim_slice(2), :] = np.nan
    allnan2 = good(1)
    allnan2[structure.dim_slice(0), :] = np.nan
    inf = good(5)
    inf[3, 2] = -np.inf
    cases = {
        "inf_middle": [good(), allnan, good(), inf, allnan2, good()],
        "structural_after_inf": [good(), inf, np.zeros((structure.D, 4), dtype=np.int64), allnan],
        "structural_before_inf": [allnan, np.zeros((structure.D + 1, 4)), inf],
        "one_d": [good(), allnan2, np.zeros(7)],
        "scalar": [np.array(3.0)],
        "T_zero": [good(2), np.zeros((structure.D, 0))],
        "int_dtype_first": [np.zeros((structure.D, 3), dtype=np.int32), good()],
        "warn_only": [allnan, good(), allnan2, allnan],
        "clean": [good(T) for T in (1, 2, 3, 9)],
    }
    return cases


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_chunked_validation_matches_per_trial_errors_and_warnings(chunk):
    structure = _structure()
    for name, data in _malformed_cases(structure).items():
        ref = _outcome(lambda: _reference_validate(structure, data, "Y_trials"))
        got = _outcome(lambda: validate_observation_list(structure, data, label="Y_trials", max_chunk_observations=chunk))
        assert got == ref, f"{name} (chunk={chunk}): {got} != {ref}"
        got_nv = _outcome(lambda: compute_global_nanvar(structure, data, max_chunk_observations=chunk))
        assert got_nv[0] == ref[0] and got_nv[1] == ref[1], f"compute_global_nanvar {name} (chunk={chunk})"


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_torch_dataset_validation_matches_per_trial_errors(chunk, monkeypatch):
    pytest.importorskip("torch")
    from chmm_lsim.backends import torch_batched as tb

    structure = _structure()
    real = tb.validate_observation_list
    monkeypatch.setattr(tb, "validate_observation_list", lambda s, d, label: real(s, d, label=label, max_chunk_observations=chunk))
    for name, data in _malformed_cases(structure).items():
        ref = _outcome(lambda: _reference_validate(structure, data, "dataset"))
        got = _outcome(lambda: tb._validate_dataset(structure, data))
        assert got == ref, f"{name} (chunk={chunk}): {got} != {ref}"


# ==========================================================================
# 3. Training: validate once at the boundary, private route afterwards.
# ==========================================================================


def _model_and_data(n_trials=24, seed=9, sigma_diag=True):
    structure = LSIMStructure(channel_state_num=(3, 2), channel_obs_dim=(1, 2), num_gmm_component=(1, 1))
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for z in range(structure.C):
        transition[:, structure.state_slice(z)] = rng.dirichlet(np.ones(structure.channel_state_num[z]) * 3, size=structure.S)
    coupling = rng.dirichlet(np.ones(2) * 3, size=2).T
    emissions = []
    for c in range(structure.C):
        M, d = structure.channel_state_num[c], structure.channel_obs_dim[c]
        mu = rng.normal(scale=3.0, size=(d, M, 1))
        sigma = np.ones((d, M, 1)) if sigma_diag else np.stack([np.eye(d)] * M, axis=-1)[..., None]
        emissions.append(ChannelGMM(weights=np.ones((M, 1)), mu=mu, sigma=sigma, diag=sigma_diag))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    Y_trials = [simulate_lsim(structure, params, int(T), seed=i)[0] for i, T in enumerate(rng.integers(1, 15, size=n_trials))]
    Y_trials[3][0, 2] = np.nan
    return structure, params, Y_trials


try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # core-only installs
    _HAS_TORCH = False

BACKENDS = ["numpy", "numba", pytest.param("torch", marks=pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed"))]


def _backend_kwargs(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    return {"backend": backend}


def _train(structure, params, Y_trials, backend, **extra):
    kw = dict(max_iterations=3, num_minibatches=4, shuffle=True, random_state=0, fb_constants=FbConstants(variant="sed"), convergence="off")
    kw.update(extra)
    return em.em_lsim_minibatch_train(structure, params.copy(), Y_trials, **_backend_kwargs(backend), **kw)


@pytest.mark.parametrize("backend", BACKENDS)
def test_training_validates_observations_once_and_params_every_update(backend, monkeypatch):
    import chmm_lsim.models.lsim.inference.fast as fast_mod
    import chmm_lsim.models.lsim.inference.sed as sed_mod

    spied_modules = [val, sed_mod, fast_mod]
    if backend == "numba":
        pytest.importorskip("numba")
        import chmm_lsim.backends.numba_ as nb_mod

        spied_modules.append(nb_mod)

    structure, params, Y_trials = _model_and_data()
    counts = {"chunk_passes": 0, "per_trial": 0, "params_validate": 0}
    real_chunks = gmm_mod._iter_validated_observation_chunks

    def chunk_spy(*a, **k):
        counts["chunk_passes"] += 1
        yield from real_chunks(*a, **k)

    def per_trial_spy(real):
        def f(*a, **k):
            counts["per_trial"] += 1
            return real(*a, **k)
        return f

    monkeypatch.setattr(gmm_mod, "_iter_validated_observation_chunks", chunk_spy)
    for mod in spied_modules:
        monkeypatch.setattr(mod, "validate_observations", per_trial_spy(mod.validate_observations))
    if backend == "torch":
        from chmm_lsim.backends import torch_batched as tb

        monkeypatch.setattr(tb, "validate_observation_list", per_trial_spy(tb.validate_observation_list))
    real_validate = LSIMParams.validate

    def params_spy(self, *a, **k):
        counts["params_validate"] += 1
        return real_validate(self, *a, **k)

    monkeypatch.setattr(LSIMParams, "validate", params_spy)
    result = _train(structure, params, Y_trials, backend, coherent_eval_interval=1)
    n_updates = len(result.minibatch.updates)
    assert counts["chunk_passes"] == 1, "observations must be validated exactly once, at the training boundary"
    assert counts["per_trial"] == 0, "no per-trial observation re-validation inside training"
    n_estep_calls = n_updates + len(result.evaluated_epoch_indices) + 1  # updates + coherent scores + pi_steady
    assert counts["params_validate"] >= n_estep_calls, "parameters must still be validated for every E-step/score"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sigma_diag", [True, False])
def test_training_prevalidated_route_is_bit_identical_to_revalidating_route(backend, sigma_diag, monkeypatch):
    structure, params, Y_trials = _model_and_data(sigma_diag=sigma_diag)
    fast = _train(structure, params, Y_trials, backend, coherent_eval_interval=1)
    monkeypatch.setattr(em, "_is_prevalidated", lambda *a, **k: False)
    if backend == "torch":
        from chmm_lsim.backends import torch_batched as tb

        monkeypatch.setattr(tb, "_is_prevalidated", lambda *a, **k: False)
    slow = _train(structure, params, Y_trials, backend, coherent_eval_interval=1)
    assert fast.loglik_trace == slow.loglik_trace
    for u, v in zip(fast.minibatch.updates, slow.minibatch.updates):
        for k, x in u.params_out.to_packed().items():
            np.testing.assert_array_equal(x, v.params_out.to_packed()[k])
    np.testing.assert_array_equal(fast.pi_steady, slow.pi_steady)


def test_training_locks_trials_read_only_and_restores_flags():
    structure, params, Y_trials = _model_and_data()
    view_base = np.random.default_rng(1).normal(size=(structure.D, 40))
    Y_trials.append(view_base[:, 5:12])  # a view: its own flag is locked and restored
    ro = Y_trials[0].copy()
    ro.flags.writeable = False
    Y_trials[0] = ro  # already read-only before training: must stay read-only afterwards
    seen_flags = []

    def hook(epoch, mb, update):
        seen_flags.append(all(not Y.flags.writeable for Y in Y_trials))

    _train(structure, params, Y_trials, "numpy", minibatch_hook=hook)
    assert seen_flags and all(seen_flags)
    assert not Y_trials[0].flags.writeable
    assert all(Y.flags.writeable for Y in Y_trials[1:])
    assert view_base.flags.writeable


def test_training_mutation_after_validation_raises_and_flags_restore_on_error():
    structure, params, Y_trials = _model_and_data()

    def mutate(epoch, mb, update):
        Y_trials[5][0, 0] = np.inf  # would silently bypass validation if allowed

    with pytest.raises(ValueError, match="read-only"):
        _train(structure, params, Y_trials, "numpy", minibatch_hook=mutate)
    assert all(Y.flags.writeable for Y in Y_trials)


def test_training_list_replacement_after_validation_is_not_used():
    structure, params, Y_trials = _model_and_data()
    baseline = _train(structure, params, list(Y_trials), "numpy")
    data = list(Y_trials)

    def replace(epoch, mb, update):
        bad = np.full_like(data[5], np.inf)
        data[5] = bad  # replacing the list element: the validated snapshot keeps the original trial

    replaced = _train(structure, params, data, "numpy", minibatch_hook=replace)
    assert replaced.loglik_trace == baseline.loglik_trace


def test_em_lsim_minibatch_uses_the_same_boundary():
    structure, params, Y_trials = _model_and_data()
    calls = {"n": 0}
    real = gmm_mod.compute_global_nanvar

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    em_mod = gmm_mod
    orig = em_mod.compute_global_nanvar
    em_mod.compute_global_nanvar = spy
    try:
        em.em_lsim_minibatch(structure, params.copy(), Y_trials, max_iterations=2, num_minibatches=3, shuffle=True, random_state=0,
                             fb_constants=FbConstants(variant="sed"))
    finally:
        em_mod.compute_global_nanvar = orig
    assert calls["n"] == 1
    assert all(Y.flags.writeable for Y in Y_trials)


def test_public_apis_still_validate_ordinary_lists():
    structure, params, Y_trials = _model_and_data()
    bad = list(Y_trials)
    bad[2] = bad[2].copy()
    bad[2][0, 0] = np.inf
    fbc = FbConstants(variant="sed")
    with pytest.raises(ValidationError, match="Inf"):
        e_step(structure, params, bad, fbc, backend="numpy")
    with pytest.raises(ValidationError, match="Inf"):
        em._forward_only_loglik(structure, params, bad, fbc, backend="numpy", validate="params", n_jobs=1, executor=None)
    torch = pytest.importorskip("torch")  # noqa: F841
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    with pytest.raises(ValidationError, match=r"dataset\[2\].*Inf"):
        forward_backward_batch(structure, params, bad, fbc, mode="loglik")
    with pytest.raises(ValidationError, match="Inf"):
        e_step(structure, params, bad, fbc, backend="torch")


def test_validated_snapshot_for_another_structure_is_revalidated():
    structure, params, Y_trials = _model_and_data()
    other = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(structure.D,), num_gmm_component=(1,))
    bad = list(Y_trials)
    bad[1] = bad[1].copy()
    bad[1][0, 0] = np.inf
    wrapped = val._ValidatedTrials(bad, other)
    with pytest.raises(ValidationError, match="Inf"):
        e_step(structure, params, wrapped, FbConstants(variant="sed"), backend="numpy")
    assert not val._is_prevalidated(structure, wrapped)
    assert val._is_prevalidated(other, wrapped)


def test_no_public_skip_validation_switch_was_added():
    functions = [e_step, em.em_lsim_minibatch_train, em.em_lsim_minibatch, em.em_lsim, compute_global_nanvar]
    if _HAS_TORCH:
        from chmm_lsim.backends.torch_batched import forward_backward_batch

        functions.append(forward_backward_batch)
    for fn in functions:
        names = set(inspect.signature(fn).parameters)
        assert not any(tag in n for n in names for tag in ("skip", "trust", "prevalid", "validated")), fn.__name__
