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

import json
import os
import re
import tempfile
import zipfile

import numpy as np

from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.structure import LSIMStructure
from chmm_lsim.validate import ValidationError

FORMAT_NAME = "lsim-params"
FORMAT_VERSION = 1

_STRUCTURE_KEYS = ("channel_state_num", "channel_obs_dim", "num_gmm_component")
_FIXED_KEYS = ("header_json", "format_version", *_STRUCTURE_KEYS, "sigma_diag", "pi_0", "transition", "coupling")
_EMISSION_FIELDS = ("weights", "mu", "sigma")
_HEADER_KEYS = {"format", "format_version", "metadata"}


class SerializationError(ValueError):
    """An LSIM parameter file (or the data being saved) violates the documented
    `chmm_lsim.serialization` format contract."""


def save_lsim_params(path: str | os.PathLike, params: LSIMParams, *, metadata: dict | None = None) -> None:
    """Atomically write `params` and optional JSON-object `metadata` to `path`
    (exactly that path; no `.npz` suffix is appended)."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise SerializationError(f"metadata must be a dict (JSON object), got {type(metadata).__name__}")
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
            emissions=[
                ChannelGMM(weights=np.asarray(g.weights, dtype=np.float64), mu=np.asarray(g.mu, dtype=np.float64),
                           sigma=np.asarray(g.sigma, dtype=np.float64), diag=g.diag)
                for g in params.emissions
            ],
        )
        stored.validate()
    except (ValidationError, ValueError, TypeError, AttributeError) as exc:
        raise SerializationError(f"parameters cannot be saved: {type(exc).__name__}: {exc}") from exc
    try:
        header = json.dumps({"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "metadata": metadata},
                            sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"metadata is not strict JSON (no NaN/Infinity, only JSON types): {exc}") from exc

    arrays: dict[str, np.ndarray] = {
        "header_json": np.array(header),
        "format_version": np.array(FORMAT_VERSION, dtype=np.int64),
        "channel_state_num": np.asarray(s.channel_state_num, dtype=np.int64),
        "channel_obs_dim": np.asarray(s.channel_obs_dim, dtype=np.int64),
        "num_gmm_component": np.asarray(s.num_gmm_component, dtype=np.int64),
        "sigma_diag": np.array(bool(stored.sigma_diag), dtype=np.bool_),
        "pi_0": stored.pi_0,
        "transition": stored.transition,
        "coupling": stored.coupling,
    }
    for c, gmm in enumerate(stored.emissions):
        arrays[f"weights_{c}"] = gmm.weights
        arrays[f"mu_{c}"] = gmm.mu
        arrays[f"sigma_{c}"] = gmm.sigma

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class _NonFiniteJSON(Exception):
    pass


def _reject_json_constant(token):
    """json.loads hook for the non-standard tokens NaN, Infinity and -Infinity,
    which Python's json module accepts by default wherever they occur."""
    raise _NonFiniteJSON(token)


def _fail(path, message):
    raise SerializationError(f"{path}: {message}")


def _require_int_scalar(path, name, a):
    if a.shape != () or a.dtype.kind not in "iu":
        _fail(path, f"{name} must be a 0-d integer array, got shape {a.shape} dtype {a.dtype}")
    return int(a[()])


def _require_float64(path, name, a, shape):
    if a.dtype != np.float64:
        _fail(path, f"{name} must be float64, got {a.dtype}")
    if a.shape != shape:
        _fail(path, f"{name} must have shape {shape}, got {a.shape}")


def load_lsim_params(path: str | os.PathLike) -> tuple[LSIMParams, dict]:
    """Inverse of `save_lsim_params`: returns `(params, metadata)`, with the
    params validated. Raises `SerializationError` for any contract violation
    listed in the module docstring."""
    try:
        archive = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise SerializationError(f"{path}: not a readable npz archive: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        _fail(path, "not an npz archive")

    with archive as z:
        try:
            data = {k: z[k] for k in z.files}
        except (ValueError, OSError, zipfile.BadZipFile) as exc:  # object arrays need pickle; corrupt members
            raise SerializationError(f"{path}: unreadable or disallowed (pickled/object) array: {exc}") from exc

    missing_fixed = [k for k in _FIXED_KEYS if k not in data]
    if missing_fixed:
        _fail(path, f"missing required keys {missing_fixed}")

    # ---- header ----
    h = data["header_json"]
    if h.shape != () or h.dtype.kind != "U":
        _fail(path, f"header_json must be a 0-d unicode string, got shape {h.shape} dtype {h.dtype}")
    try:
        header = json.loads(str(h[()]), parse_constant=_reject_json_constant)
    except _NonFiniteJSON as exc:
        raise SerializationError(f"{path}: header_json contains the non-standard JSON constant {exc.args[0]} (NaN/Infinity are not allowed)") from None
    except json.JSONDecodeError as exc:
        raise SerializationError(f"{path}: header_json is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        _fail(path, "header_json must decode to a JSON object")
    if header.get("format") != FORMAT_NAME:
        _fail(path, f"not an {FORMAT_NAME} file (format={header.get('format')!r})")
    if set(header) != _HEADER_KEYS:
        _fail(path, f"header_json keys must be exactly {sorted(_HEADER_KEYS)}, got {sorted(header)}")
    hv = header["format_version"]
    if isinstance(hv, bool) or not isinstance(hv, int):
        _fail(path, f"header format_version must be an integer, got {hv!r}")
    if hv != FORMAT_VERSION:
        _fail(path, f"unsupported {FORMAT_NAME} format_version {hv!r}")
    if _require_int_scalar(path, "format_version", data["format_version"]) != hv:
        _fail(path, f"numeric format_version {int(data['format_version'][()])} disagrees with header format_version {hv}")
    metadata = header["metadata"]
    if not isinstance(metadata, dict):
        _fail(path, f"metadata must be a JSON object, got {type(metadata).__name__}")

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

    sd = data["sigma_diag"]
    if sd.shape != () or sd.dtype != np.bool_:
        _fail(path, f"sigma_diag must be a 0-d numpy.bool_, got shape {sd.shape} dtype {sd.dtype}")
    diag = bool(sd[()])

    # ---- exact key set (catches extra, duplicate-style and misnumbered emission keys) ----
    expected = set(_FIXED_KEYS) | {f"{field}_{c}" for field in _EMISSION_FIELDS for c in range(C)}
    unexpected = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if missing:
        _fail(path, f"missing required keys {missing}")
    if unexpected:
        numbered = [k for k in unexpected if re.fullmatch(r"(weights|mu|sigma)_\d+", k)]
        what = "unexpected or inconsistently numbered emission keys" if numbered else "unexpected keys"
        _fail(path, f"{what} {unexpected} (C={C})")

    # ---- shapes and dtypes ----
    S = sum(dims["channel_state_num"])
    _require_float64(path, "pi_0", data["pi_0"], (S,))
    _require_float64(path, "transition", data["transition"], (S, S))
    _require_float64(path, "coupling", data["coupling"], (C, C))
    for c in range(C):
        M, d, K = dims["channel_state_num"][c], dims["channel_obs_dim"][c], dims["num_gmm_component"][c]
        _require_float64(path, f"weights_{c}", data[f"weights_{c}"], (M, K))
        _require_float64(path, f"mu_{c}", data[f"mu_{c}"], (d, M, K))
        _require_float64(path, f"sigma_{c}", data[f"sigma_{c}"], (d, M, K) if diag else (d, d, M, K))

    # ---- construct and validate values ----
    try:
        structure = LSIMStructure(**dims)
        params = LSIMParams(
            structure=structure,
            pi_0=data["pi_0"].copy(),
            transition=data["transition"].copy(),
            coupling=data["coupling"].copy(),
            emissions=[
                ChannelGMM(weights=data[f"weights_{c}"].copy(), mu=data[f"mu_{c}"].copy(), sigma=data[f"sigma_{c}"].copy(), diag=diag)
                for c in range(C)
            ],
        )
        params.validate()
    except (ValidationError, ValueError) as exc:
        raise SerializationError(f"{path}: stored parameters are invalid: {exc}") from exc
    return params, metadata
