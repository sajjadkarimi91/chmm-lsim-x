"""Model-independent helpers for the strict `.npz` parameter formats: the
`SerializationError` type, atomic archive writes, pickle-free reading, the
JSON header check and the scalar/array contract checks."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import numpy as np


class SerializationError(ValueError):
    """An LSIM parameter file (or the data being saved) violates the documented
    parameter-file format contract (see `chmm_lsim.save_lsim_params`)."""


def _atomic_savez(path: str | os.PathLike, arrays: dict[str, np.ndarray]) -> None:
    """Write `arrays` to `path` atomically: a temporary file in the destination
    directory, flushed to disk, then moved into place with `os.replace`; a
    failed write removes the temporary file and leaves any existing
    destination intact."""
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


def _read_npz(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Every member of the archive at `path`, read with `allow_pickle=False`."""
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
            return {k: z[k] for k in z.files}
        except (ValueError, OSError, zipfile.BadZipFile) as exc:  # object arrays need pickle; corrupt members
            raise SerializationError(f"{path}: unreadable or disallowed (pickled/object) array: {exc}") from exc


_HEADER_KEYS = {"format", "format_version", "metadata"}


def _make_header(format_name: str, format_version: int, metadata) -> str:
    """The strict-JSON header string; rejects a non-dict or non-JSON `metadata`."""
    if not isinstance(metadata, dict):
        raise SerializationError(f"metadata must be a dict (JSON object), got {type(metadata).__name__}")
    try:
        return json.dumps({"format": format_name, "format_version": format_version, "metadata": metadata},
                          sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"metadata is not strict JSON (no NaN/Infinity, only JSON types): {exc}") from exc


def _check_header(path, data: dict, format_name: str, format_version: int) -> dict:
    """Parses and checks `header_json`/`format_version`; returns the metadata dict."""
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
    if header.get("format") != format_name:
        _fail(path, f"not an {format_name} file (format={header.get('format')!r})")
    if set(header) != _HEADER_KEYS:
        _fail(path, f"header_json keys must be exactly {sorted(_HEADER_KEYS)}, got {sorted(header)}")
    hv = header["format_version"]
    if isinstance(hv, bool) or not isinstance(hv, int):
        _fail(path, f"header format_version must be an integer, got {hv!r}")
    if hv != format_version:
        _fail(path, f"unsupported {format_name} format_version {hv!r}")
    if _require_int_scalar(path, "format_version", data["format_version"]) != hv:
        _fail(path, f"numeric format_version {int(data['format_version'][()])} disagrees with header format_version {hv}")
    metadata = header["metadata"]
    if not isinstance(metadata, dict):
        _fail(path, f"metadata must be a JSON object, got {type(metadata).__name__}")
    return metadata
