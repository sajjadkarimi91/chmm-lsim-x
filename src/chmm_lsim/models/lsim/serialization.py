"""Save and reload `LSIMParams` exactly, e.g. to initialize once and reuse the
identical `params0` across repeated EM experiments.

Format (`format_version` 1): a NumPy `.npz` archive with exactly these entries

    header_json         0-d unicode string: JSON object
                        {"format": "lsim-params", "format_version": 1, "metadata": {...}}
    format_version      0-d integer, equal to the header's format_version
    channel_state_num   (C,) integer, every entry >= 1
    channel_obs_dim     (C,) integer, every entry >= 1
    num_gmm_component   (C,) integer, every entry >= 1
    sigma_diag          0-d numpy.bool_
    pi_0                (S,)    float64
    transition          (S, S)  float64
    coupling            (C, C)  float64
    weights_<c>         (M_c, K_c)              float64, c = 0..C-1
    mu_<c>              (d_c, M_c, K_c)         float64
    sigma_<c>           (d_c, M_c, K_c)         float64 if sigma_diag
                        (d_c, d_c, M_c, K_c)    float64 otherwise

and nothing else. Parameter arrays are always written as float64 (a float32
input is upcast exactly, and the upcast values are what save validates, so a
float32 model whose float64 form is not stochastic within tolerance is refused
at save time rather than producing an unloadable file). The loader reads with `allow_pickle=False` and checks
every key, shape and dtype above BEFORE constructing `LSIMStructure`/
`LSIMParams`; non-finite or otherwise invalid parameter values are then
rejected through the normal `LSIMParams.validate()` contract. Every rejection
raises `SerializationError`.

`metadata` must be a JSON-serializable dict. NaN/Infinity are rejected on save
(`allow_nan=False`) and on load: the header is parsed with a hook that refuses
`NaN`, `Infinity` and `-Infinity` anywhere, including nested metadata. Invalid
parameters passed to `save_lsim_params` also raise `SerializationError`, with
the underlying `ValidationError` as `__cause__`; plain filesystem errors
(permissions, disk full) propagate unchanged. Writes
are atomic: the archive is written to a temporary file in the destination
directory, flushed to disk, then moved into place with `os.replace`; a failed
write removes the temporary file and leaves any existing destination intact.
"""

from __future__ import annotations

import os
import re

import numpy as np

from chmm_lsim.core.serialization import (
    SerializationError,
    _atomic_savez,
    _check_header,
    _fail,
    _make_header,
    _read_npz,
    _require_float64,
)
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError
from chmm_lsim.emissions.base import get_emission_model
from chmm_lsim.models.lsim.params import LSIMParams

FORMAT_NAME = "lsim-params"
FORMAT_VERSION = 1

# Format version 1 stores Gaussian-mixture emissions; another emission model
# needs a new format version (or a header field naming the model).
_EMISSION_MODEL = get_emission_model("gmm")

_STRUCTURE_KEYS = ("channel_state_num", "channel_obs_dim", "num_gmm_component")
_FIXED_KEYS = ("header_json", "format_version", *_STRUCTURE_KEYS, *_EMISSION_MODEL.serial_global_keys, "pi_0", "transition", "coupling")
_EMISSION_FIELDS = _EMISSION_MODEL.serial_channel_fields


def save_lsim_params(path: str | os.PathLike, params: LSIMParams, *, metadata: dict | None = None) -> None:
    """Atomically write `params` and optional JSON-object `metadata` to `path`
    (exactly that path; no `.npz` suffix is appended)."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise SerializationError(f"metadata must be a dict (JSON object), got {type(metadata).__name__}")
    model = _EMISSION_MODEL
    # Validate exactly what will be written (float64), so save never produces a
    # file that load_lsim_params would reject. Any failure to convert or validate
    # the parameters is a serialization-contract rejection (original exception
    # kept as __cause__); filesystem errors further down are NOT wrapped.
    try:
        s = params.structure
        stored = LSIMParams(
            structure=s,
            pi_0=np.asarray(params.pi_0, dtype=np.float64),
            transition=np.asarray(params.transition, dtype=np.float64),
            coupling=np.asarray(params.coupling, dtype=np.float64),
            emissions=model.as_float64(params.emissions),
        )
        stored.validate()
    except (ValidationError, ValueError, TypeError, AttributeError) as exc:
        raise SerializationError(f"parameters cannot be saved: {type(exc).__name__}: {exc}") from exc
    header = _make_header(FORMAT_NAME, FORMAT_VERSION, metadata)

    global_arrays, channel_arrays = model.to_arrays(stored.emissions)
    arrays: dict[str, np.ndarray] = {
        "header_json": np.array(header),
        "format_version": np.array(FORMAT_VERSION, dtype=np.int64),
        "channel_state_num": np.asarray(s.channel_state_num, dtype=np.int64),
        "channel_obs_dim": np.asarray(s.channel_obs_dim, dtype=np.int64),
        "num_gmm_component": np.asarray(s.num_gmm_component, dtype=np.int64),
        **global_arrays,
        "pi_0": stored.pi_0,
        "transition": stored.transition,
        "coupling": stored.coupling,
    }
    for c, fields in enumerate(channel_arrays):
        for name in _EMISSION_FIELDS:
            arrays[f"{name}_{c}"] = fields[name]
    _atomic_savez(path, arrays)


def load_lsim_params(path: str | os.PathLike) -> tuple[LSIMParams, dict]:
    """Inverse of `save_lsim_params`: returns `(params, metadata)`, with the
    params validated. Raises `SerializationError` for any contract violation
    listed in the module docstring."""
    model = _EMISSION_MODEL
    data = _read_npz(path)

    missing_fixed = [k for k in _FIXED_KEYS if k not in data]
    if missing_fixed:
        _fail(path, f"missing required keys {missing_fixed}")

    # ---- header ----
    metadata = _check_header(path, data, FORMAT_NAME, FORMAT_VERSION)

    # ---- structure ----
    dims = {}
    for name in _STRUCTURE_KEYS:
        a = data[name]
        if a.ndim != 1 or a.dtype.kind not in "iu":
            _fail(path, f"{name} must be a 1-D integer array, got shape {a.shape} dtype {a.dtype}")
        if a.size == 0 or np.any(a < 1):
            _fail(path, f"{name} entries must all be >= 1, got {a.tolist()}")
        dims[name] = tuple(int(x) for x in a)
    C = len(dims["channel_state_num"])
    if any(len(v) != C for v in dims.values()):
        _fail(path, f"structure arrays disagree on channel count: { {k: len(v) for k, v in dims.items()} }")

    options = model.read_global_arrays(path, data, _fail)

    # ---- exact key set (catches extra, duplicate-style and misnumbered emission keys) ----
    expected = set(_FIXED_KEYS) | {f"{field}_{c}" for field in _EMISSION_FIELDS for c in range(C)}
    unexpected = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if missing:
        _fail(path, f"missing required keys {missing}")
    if unexpected:
        numbered = [k for k in unexpected if re.fullmatch(r"(%s)_\d+" % "|".join(_EMISSION_FIELDS), k)]
        what = "unexpected or inconsistently numbered emission keys" if numbered else "unexpected keys"
        _fail(path, f"{what} {unexpected} (C={C})")

    # ---- shapes and dtypes ----
    S = sum(dims["channel_state_num"])
    _require_float64(path, "pi_0", data["pi_0"], (S,))
    _require_float64(path, "transition", data["transition"], (S, S))
    _require_float64(path, "coupling", data["coupling"], (C, C))
    for c in range(C):
        M, d, K = dims["channel_state_num"][c], dims["channel_obs_dim"][c], dims["num_gmm_component"][c]
        for name, shape in model.channel_array_shapes(M, d, K, options).items():
            _require_float64(path, f"{name}_{c}", data[f"{name}_{c}"], shape)

    # ---- construct and validate values ----
    try:
        structure = LSIMStructure(**dims)
        params = LSIMParams(
            structure=structure,
            pi_0=data["pi_0"].copy(),
            transition=data["transition"].copy(),
            coupling=data["coupling"].copy(),
            emissions=model.from_arrays([{name: data[f"{name}_{c}"] for name in _EMISSION_FIELDS} for c in range(C)], options),
        )
        params.validate()
    except (ValidationError, ValueError) as exc:
        raise SerializationError(f"{path}: stored parameters are invalid: {exc}") from exc
    return params, metadata
