"""save_lsim_params / load_lsim_params: exact round trip, provenance metadata,
and 'initialize once, reuse params0' giving the identical first EM update."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from chmm_lsim import (
    ChannelGMM,
    SerializationError,
    FbConstants,
    LSIMParams,
    LSIMStructure,
    em_lsim,
    em_lsim_minibatch_train,
    load_lsim_params,
    save_lsim_params,
)


def _model(diag: bool, seed: int = 0):
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(2, 1))
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for z in range(structure.C):
        sl = structure.state_slice(z)
        transition[:, sl] = rng.dirichlet(np.ones(sl.stop - sl.start), size=structure.S)
    coupling = rng.dirichlet(np.ones(structure.C), size=structure.C).T
    emissions = []
    for c in range(structure.C):
        M, d, K = structure.channel_state_num[c], structure.channel_obs_dim[c], structure.num_gmm_component[c]
        sigma = rng.uniform(0.5, 2.0, size=(d, M, K))
        if not diag:
            full = np.zeros((d, d, M, K))
            for i in range(M):
                for k in range(K):
                    full[:, :, i, k] = np.diag(sigma[:, i, k])
            sigma = full
        emissions.append(ChannelGMM(rng.dirichlet(np.ones(K), size=M), rng.normal(size=(d, M, K)), sigma, diag=diag))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    return structure, params


def _assert_identical(a: LSIMParams, b: LSIMParams):
    assert a.structure == b.structure
    for x, y in ((a.pi_0, b.pi_0), (a.transition, b.transition), (a.coupling, b.coupling)):
        np.testing.assert_array_equal(x, y)
        assert x.dtype == y.dtype
    for ga, gb in zip(a.emissions, b.emissions):
        assert ga.diag == gb.diag
        for x, y in ((ga.weights, gb.weights), (ga.mu, gb.mu), (ga.sigma, gb.sigma)):
            np.testing.assert_array_equal(x, y)
            assert x.shape == y.shape


@pytest.mark.parametrize("diag", [True, False])
def test_round_trip_is_exact_with_metadata(tmp_path, diag):
    _, params = _model(diag)
    meta = {"seed": 3, "max_samples_per_channel": None, "n_jobs": 4, "init_seconds": 12.5, "data": "ctx1 train+dev"}
    path = tmp_path / "params0.npz"
    save_lsim_params(path, params, metadata=meta)
    loaded, loaded_meta = load_lsim_params(path)
    _assert_identical(params, loaded)
    assert loaded_meta == meta


def test_file_contains_no_pickled_objects(tmp_path):
    _, params = _model(True)
    path = tmp_path / "p.npz"
    save_lsim_params(path, params)
    with np.load(path, allow_pickle=False) as z:  # would raise on any object array
        assert all(z[k].dtype != object for k in z.files)


def _observations(structure, seed=1):
    rng = np.random.default_rng(seed)
    Y = [rng.normal(size=(structure.D, T)) for T in (60, 45, 80, 30)]
    Y[0][1, 5] = np.nan
    return Y


def test_reloaded_params_give_the_identical_first_em_update(tmp_path):
    structure, params = _model(True, seed=5)
    Y = _observations(structure)
    path = tmp_path / "params0.npz"
    save_lsim_params(path, params)
    reloaded, _ = load_lsim_params(path)

    def first_update(p0):
        seen = []
        em_lsim(structure, p0, Y, max_iterations=1, fb_constants=FbConstants(), check_convergence=False,
                iter_hook=lambda itr, p, ll, aic, bic: seen.append((p.copy(), ll)))
        return seen[0]

    (pa, lla), (pb, llb) = first_update(params), first_update(reloaded)
    assert lla == llb
    _assert_identical(pa, pb)


def test_reloaded_params_give_the_identical_first_minibatch_update(tmp_path):
    structure, params = _model(True, seed=6)
    Y = _observations(structure, seed=2)
    path = tmp_path / "params0.npz"
    save_lsim_params(path, params)
    reloaded, _ = load_lsim_params(path)

    def first(p0):
        fit = em_lsim_minibatch_train(structure, p0, Y, max_iterations=1, num_minibatches=2, shuffle=True,
                                      random_state=0, fb_constants=FbConstants(), check_convergence=False)
        return fit.minibatch.updates[0]

    ua, ub = first(params), first(reloaded)
    assert ua.trial_indices == ub.trial_indices
    assert ua.training_batch_loglik == ub.training_batch_loglik
    _assert_identical(ua.params_out, ub.params_out)


def test_saving_initializer_output_round_trips(tmp_path):
    pytest.importorskip("sklearn")
    from chmm_lsim import initialize_lsim_params

    structure, _ = _model(True)
    Y = _observations(structure, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params0 = initialize_lsim_params(structure, Y, sigma_diag=True, seed=2)
    save_lsim_params(tmp_path / "init.npz", params0, metadata={"seed": 2})
    loaded, meta = load_lsim_params(tmp_path / "init.npz")
    _assert_identical(params0, loaded)
    assert meta == {"seed": 2}



# --------------------------------------------------------------------------
# P10e item 3: strict format contract. Every case starts from a VALID file,
# plants exactly one corruption, re-writes the archive, and must be rejected
# with SerializationError before anything is constructed.
# --------------------------------------------------------------------------

import json  # noqa: E402

from chmm_lsim.models.lsim.serialization import FORMAT_VERSION  # noqa: E402


def _valid_arrays(tmp_path, diag=True):
    _, params = _model(diag)
    src = tmp_path / "valid.npz"
    save_lsim_params(src, params, metadata={"seed": 1})
    with np.load(src, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _header(**over):
    h = {"format": "lsim-params", "format_version": FORMAT_VERSION, "metadata": {"seed": 1}}
    h.update(over)
    return np.array(json.dumps(h))


def _write(tmp_path, arrays):
    p = tmp_path / "planted.npz"
    np.savez(p, **arrays)
    return p


def _mutate(arrays, **changes):
    out = dict(arrays)
    for k, v in changes.items():
        if v is _DROP:
            out.pop(k)
        else:
            out[k] = v
    return out


_DROP = object()

_CORRUPTIONS = {
    # exact keys
    "missing fixed key": (lambda a: _mutate(a, pi_0=_DROP), r"missing required keys \['pi_0'\]"),
    "missing emission key": (lambda a: _mutate(a, mu_1=_DROP), r"missing required keys \['mu_1'\]"),
    "unexpected unrelated key": (lambda a: _mutate(a, extra=np.zeros(1)), r"unexpected keys \['extra'\]"),
    "extra indexed emission channel": (lambda a: _mutate(a, weights_2=a["weights_1"]), r"inconsistently numbered emission keys \['weights_2'\]"),
    "duplicate-style emission numbering": (lambda a: _mutate(a, mu_01=a["mu_1"]), r"inconsistently numbered emission keys \['mu_01'\]"),
    "renumbered emission channel": (lambda a: _mutate(a, sigma_1=_DROP, sigma_3=a["sigma_1"]), r"missing required keys \['sigma_1'\]"),
    # header / versions / metadata
    "foreign format": (lambda a: _mutate(a, header_json=_header(format="something-else")), "not an lsim-params file"),
    "future header version": (lambda a: _mutate(a, header_json=_header(format_version=99)), "unsupported lsim-params format_version 99"),
    "boolean header version": (lambda a: _mutate(a, header_json=_header(format_version=True)), "header format_version must be an integer"),
    "numeric version disagrees with header": (lambda a: _mutate(a, format_version=np.array(2)), "numeric format_version 2 disagrees with header format_version 1"),
    "numeric version not integer": (lambda a: _mutate(a, format_version=np.array(1.0)), "format_version must be a 0-d integer array"),
    "numeric version not scalar": (lambda a: _mutate(a, format_version=np.array([1])), "format_version must be a 0-d integer array"),
    "metadata is a list": (lambda a: _mutate(a, header_json=_header(metadata=[])), "metadata must be a JSON object, got list"),
    "header extra key": (lambda a: _mutate(a, header_json=_header(extra=1)), "header_json keys must be exactly"),
    "header not an object": (lambda a: _mutate(a, header_json=np.array("[1, 2]")), "header_json must decode to a JSON object"),
    "header invalid json": (lambda a: _mutate(a, header_json=np.array("{not json")), "header_json is not valid JSON"),
    "header not a string": (lambda a: _mutate(a, header_json=np.array([1, 2])), "header_json must be a 0-d unicode string"),
    # sigma_diag
    "sigma_diag int64(2)": (lambda a: _mutate(a, sigma_diag=np.array(2, dtype=np.int64)), "sigma_diag must be a 0-d numpy.bool_"),
    "sigma_diag int 1": (lambda a: _mutate(a, sigma_diag=np.array(1)), "sigma_diag must be a 0-d numpy.bool_"),
    "sigma_diag not scalar": (lambda a: _mutate(a, sigma_diag=np.array([True])), "sigma_diag must be a 0-d numpy.bool_"),
    # structure arrays
    "channel_state_num float 1.9": (lambda a: _mutate(a, channel_state_num=np.array([1.9, 3.0])), "channel_state_num must be a 1-D integer array"),
    "channel_state_num bool": (lambda a: _mutate(a, channel_state_num=np.array([True, True])), "channel_state_num must be a 1-D integer array"),
    "channel_obs_dim 2-D": (lambda a: _mutate(a, channel_obs_dim=np.array([[1, 2]])), "channel_obs_dim must be a 1-D integer array"),
    "num_gmm_component zero": (lambda a: _mutate(a, num_gmm_component=np.array([2, 0])), r"num_gmm_component entries must all be >= 1"),
    "channel_state_num negative": (lambda a: _mutate(a, channel_state_num=np.array([2, -3])), r"channel_state_num entries must all be >= 1"),
    "structure length mismatch": (lambda a: _mutate(a, channel_obs_dim=np.array([1, 2, 1])), "structure arrays disagree on channel count"),
    # parameter shapes / dtypes
    "pi_0 wrong length": (lambda a: _mutate(a, pi_0=np.full(4, 0.25)), r"pi_0 must have shape \(5,\)"),
    "transition not square": (lambda a: _mutate(a, transition=a["transition"][:, :4]), r"transition must have shape \(5, 5\)"),
    "coupling float32": (lambda a: _mutate(a, coupling=a["coupling"].astype(np.float32)), "coupling must be float64, got float32"),
    "weights shape disagrees with structure": (lambda a: _mutate(a, weights_0=np.ones((2, 3)) / 3), r"weights_0 must have shape \(2, 2\)"),
    "mu shape disagrees with structure": (lambda a: _mutate(a, mu_1=np.zeros((2, 2, 1))), r"mu_1 must have shape \(2, 3, 1\)"),
    "diag sigma stored as full": (lambda a: _mutate(a, sigma_0=np.ones((1, 1, 2, 2))), r"sigma_0 must have shape \(1, 2, 2\)"),
    "integer parameter array": (lambda a: _mutate(a, pi_0=np.array([1, 0, 1, 0, 0])), "pi_0 must be float64"),
    # values -> normal parameter validation
    "NaN in mu": (lambda a: _mutate(a, mu_0=np.where(np.arange(a["mu_0"].size).reshape(a["mu_0"].shape) == 0, np.nan, a["mu_0"])), "stored parameters are invalid: .*mu contains NaN"),
    "Inf in transition": (lambda a: _mutate(a, transition=np.where(np.eye(5, dtype=bool), np.inf, a["transition"])), "stored parameters are invalid: .*transition contains Inf"),
    "non-stochastic pi_0": (lambda a: _mutate(a, pi_0=np.full(5, 0.9)), "stored parameters are invalid"),
}


@pytest.mark.parametrize("case", sorted(_CORRUPTIONS))
def test_planted_corruption_is_rejected(tmp_path, case):
    mutate, match = _CORRUPTIONS[case]
    path = _write(tmp_path, mutate(_valid_arrays(tmp_path)))
    with pytest.raises(SerializationError, match=match):
        load_lsim_params(path)


def test_full_covariance_shape_contract(tmp_path):
    arrays = _valid_arrays(tmp_path, diag=False)
    load_lsim_params(_write(tmp_path, arrays))  # the unmodified full-covariance file loads
    bad = _mutate(arrays, sigma_1=np.ones((2, 3, 1)))
    with pytest.raises(SerializationError, match=r"sigma_1 must have shape \(2, 2, 3, 1\)"):
        load_lsim_params(_write(tmp_path, bad))


def test_serialization_error_is_a_value_error():
    assert issubclass(SerializationError, ValueError)


@pytest.mark.parametrize("bad", ["", b"PK\x03\x04garbage", b"not a zip at all"])
def test_unreadable_archives_are_rejected(tmp_path, bad):
    p = tmp_path / "broken.npz"
    p.write_bytes(bad if isinstance(bad, bytes) else bad.encode())
    with pytest.raises(SerializationError, match="not a readable npz archive|unreadable"):
        load_lsim_params(p)


def test_pickled_object_arrays_are_rejected(tmp_path):
    arrays = _valid_arrays(tmp_path)
    arrays["header_json"] = np.array([{"format": "lsim-params"}], dtype=object)
    p = tmp_path / "objects.npz"
    np.savez(p, **arrays)
    with pytest.raises(SerializationError, match="unreadable or disallowed"):
        load_lsim_params(p)


@pytest.mark.parametrize("metadata,match", [
    ([], "metadata must be a dict"),
    ("seed=1", "metadata must be a dict"),
    ({"x": float("nan")}, "metadata is not strict JSON"),
    ({"x": float("inf")}, "metadata is not strict JSON"),
    ({"x": object()}, "metadata is not strict JSON"),
])
def test_save_rejects_non_json_object_metadata(tmp_path, metadata, match):
    _, params = _model(True)
    with pytest.raises(SerializationError, match=match):
        save_lsim_params(tmp_path / "p.npz", params, metadata=metadata)
    assert list(tmp_path.iterdir()) == []  # nothing written, no temporary left behind


def test_failed_write_leaves_existing_destination_intact(tmp_path, monkeypatch):
    _, params = _model(True)
    dest = tmp_path / "params0.npz"
    save_lsim_params(dest, params, metadata={"generation": 1})
    original = dest.read_bytes()

    import chmm_lsim.models.lsim.serialization as ser

    real_savez = ser.np.savez

    def failing_savez(f, **arrays):
        f.write(b"partial garbage that must never reach the destination")
        raise OSError("simulated disk full")

    monkeypatch.setattr(ser.np, "savez", failing_savez)
    _, other = _model(True, seed=9)
    with pytest.raises(OSError, match="simulated disk full"):
        save_lsim_params(dest, other, metadata={"generation": 2})
    monkeypatch.setattr(ser.np, "savez", real_savez)

    assert dest.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["params0.npz"]  # temporary file cleaned up
    loaded, meta = load_lsim_params(dest)
    assert meta == {"generation": 1}


def test_successful_save_replaces_atomically_and_leaves_no_temporary(tmp_path):
    _, a = _model(True, seed=1)
    _, b = _model(True, seed=2)
    dest = tmp_path / "params.npz"
    save_lsim_params(dest, a, metadata={"v": 1})
    save_lsim_params(dest, b, metadata={"v": 2})
    loaded, meta = load_lsim_params(dest)
    assert meta == {"v": 2}
    np.testing.assert_array_equal(loaded.transition, b.transition)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["params.npz"]


def test_float32_parameters_are_stored_as_float64(tmp_path):
    _, params = _model(True)
    params.coupling = np.full_like(params.coupling, 0.5, dtype=np.float32)  # exactly representable, still column-stochastic
    save_lsim_params(tmp_path / "p.npz", params)
    loaded, _ = load_lsim_params(tmp_path / "p.npz")
    assert loaded.coupling.dtype == np.float64
    np.testing.assert_array_equal(loaded.coupling, params.coupling.astype(np.float64))


def test_save_refuses_float32_values_whose_float64_form_is_invalid(tmp_path):
    """Save validates the float64 arrays it writes, so it can never produce a
    file that load_lsim_params would reject."""
    from chmm_lsim.core.validate import ValidationError

    _, params = _model(True)
    params.coupling = params.coupling.astype(np.float32)  # sums drift off 1.0 once upcast
    with pytest.raises(SerializationError, match="parameters cannot be saved: ValidationError: coupling") as info:
        save_lsim_params(tmp_path / "p.npz", params)
    assert isinstance(info.value.__cause__, ValidationError)
    assert list(tmp_path.iterdir()) == []



# --------------------------------------------------------------------------
# P10e closure: non-standard JSON constants, and one exception type on save.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("where", [
    '{{"format": "lsim-params", "format_version": 1, "metadata": {{"bad": {tok}}}}}',
    '{{"format": "lsim-params", "format_version": 1, "metadata": {{"a": {{"b": [1, 2, {{"c": {tok}}}]}}}}}}',
    '{{"format": "lsim-params", "format_version": 1, "metadata": {{"list": [{tok}]}}}}',
    '{{"format": "lsim-params", "format_version": {tok}, "metadata": {{}}}}',
])
def test_non_standard_json_constants_are_rejected_anywhere_in_the_header(tmp_path, token, where):
    arrays = _valid_arrays(tmp_path)
    arrays["header_json"] = np.array(where.format(tok=token))
    with pytest.raises(SerializationError, match=f"non-standard JSON constant {token}"):
        load_lsim_params(_write(tmp_path, arrays))


def test_invalid_parameters_on_save_raise_serialization_error_with_cause(tmp_path):
    from chmm_lsim.core.validate import ValidationError

    _, params = _model(True)
    params.pi_0 = np.full_like(params.pi_0, 0.9)  # not stochastic
    with pytest.raises(SerializationError, match="parameters cannot be saved") as info:
        save_lsim_params(tmp_path / "p.npz", params)
    assert isinstance(info.value.__cause__, ValidationError)
    assert list(tmp_path.iterdir()) == []

    params2 = _model(True)[1]
    params2.transition = np.array([["not", "numbers"]], dtype=object)
    with pytest.raises(SerializationError, match="parameters cannot be saved") as info:
        save_lsim_params(tmp_path / "p.npz", params2)
    assert info.value.__cause__ is not None


def test_filesystem_errors_on_save_are_not_disguised_as_serialization_errors(tmp_path):
    _, params = _model(True)
    missing_dir = tmp_path / "does" / "not" / "exist" / "p.npz"
    with pytest.raises(FileNotFoundError) as info:
        save_lsim_params(missing_dir, params)
    assert not isinstance(info.value, SerializationError)
