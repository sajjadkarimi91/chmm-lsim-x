"""Parameter and observation validation.

Raises ValidationError naming the offending field/block rather than letting a
malformed model silently produce a wrong (but finite) answer downstream.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.structure import LSIMStructure

_SUM_ATOL = 1e-8
_EIG_REL_TOL = 1e-9


class ValidationError(ValueError):
    pass


def _check_no_nan_inf(name: str, arr: NDArray) -> None:
    if np.isnan(arr).any():
        raise ValidationError(f"{name} contains NaN")
    if np.isinf(arr).any():
        raise ValidationError(f"{name} contains Inf")


def check_structure_matches(structure: LSIMStructure, params) -> None:
    """Every public function that takes `structure` and `params` as two
    separate arguments, rather than always reading `params.structure` --
    P6's own APIs (`simulate.py`, `viterbi.py`, `equivalent.py`,
    `diagnostics.py`), P3a/P3b's `forward_backward_sed`/`forward_backward_fast`,
    and P7's `backends.base.run_backend` (checked there too, redundantly, so
    a future backend that forgets its own check is still caught at the
    common dispatch layer) -- calls this. Taking `structure` separately is
    convenient for a caller that already has an `LSIMStructure` in hand, but
    it means a caller passing a `structure` that doesn't actually match
    `params.structure` would otherwise silently mix one model's shape with
    another's values (wrong slicing, wrong emission dims) instead of failing
    loudly -- confirmed as a real, not hypothetical, gap: `forward_backward_sed`/
    `forward_backward_fast` never called this at all before a P7 review round
    added it, so `backend="numpy"` silently accepted a mismatched `structure`
    and computed a wrong result. Checked unconditionally (not gated behind
    any `validate=` flag) since it's an O(1) tuple comparison --
    `LSIMStructure` is a frozen dataclass whose only compared fields are its
    three defining tuples."""
    if structure != params.structure:
        raise ValidationError(
            f"structure does not match params.structure: channel_state_num="
            f"{structure.channel_state_num} vs {params.structure.channel_state_num}, "
            f"channel_obs_dim={structure.channel_obs_dim} vs {params.structure.channel_obs_dim}, "
            f"num_gmm_component={structure.num_gmm_component} vs {params.structure.num_gmm_component}"
        )


def validate_params(params, *, matlab_compat: bool = True) -> None:
    structure: LSIMStructure = params.structure

    _check_no_nan_inf("pi_0", params.pi_0)
    _check_no_nan_inf("transition", params.transition)
    _check_no_nan_inf("coupling", params.coupling)

    if (params.pi_0 < 0).any():
        raise ValidationError("pi_0 has negative entries")
    for c in range(structure.C):
        block = params.pi_0[structure.state_slice(c)]
        total = float(block.sum())
        if not np.isclose(total, 1.0, rtol=0, atol=_SUM_ATOL):
            raise ValidationError(f"pi_0 block for channel {c} sums to {total}, expected 1.0")

    if (params.transition < 0).any():
        raise ValidationError("transition has negative entries")
    for c in range(structure.C):
        for zee in range(structure.C):
            block = params.transition[structure.state_slice(c), structure.state_slice(zee)]
            row_sums = block.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=0, atol=_SUM_ATOL):
                raise ValidationError(
                    f"transition block (c={c}, zee={zee}) rows sum to {row_sums.tolist()}, expected all 1.0"
                )

    if (params.coupling < 0).any():
        raise ValidationError("coupling has negative entries")
    col_sums = params.coupling.sum(axis=0)
    if not np.allclose(col_sums, 1.0, rtol=0, atol=_SUM_ATOL):
        raise ValidationError(f"coupling columns sum to {col_sums.tolist()}, expected all 1.0")

    validate_emissions(params.emissions, matlab_compat=matlab_compat)


def validate_emissions(emissions, *, matlab_compat: bool = True) -> None:
    """Per-channel GMM checks shared by LSIM and CHMM parameters: finite
    values, mixture weights that are non-negative and sum to one per state,
    and valid covariances (strictly positive variances, or symmetric
    positive (semi-)definite matrices -- see `_validate_full_sigma`)."""
    for c, gmm in enumerate(emissions):
        _check_no_nan_inf(f"emissions[{c}].weights", gmm.weights)
        _check_no_nan_inf(f"emissions[{c}].mu", gmm.mu)
        _check_no_nan_inf(f"emissions[{c}].sigma", gmm.sigma)

        if (gmm.weights < 0).any():
            raise ValidationError(f"emissions[{c}].weights has negative entries")
        weight_sums = gmm.weights.sum(axis=1)
        if not np.allclose(weight_sums, 1.0, rtol=0, atol=_SUM_ATOL):
            raise ValidationError(
                f"emissions[{c}].weights rows (per state) sum to {weight_sums.tolist()}, expected all 1.0"
            )

        if gmm.diag:
            if (gmm.sigma <= 0).any():
                raise ValidationError(f"emissions[{c}].sigma (diagonal) must be strictly positive")
        else:
            _validate_full_sigma(c, gmm, matlab_compat=matlab_compat)


def _validate_full_sigma(c: int, gmm, *, matlab_compat: bool) -> None:
    """A genuinely negative eigenvalue means no Gaussian density exists at all
    -- MATLAB's own `mvnpdf` errors on that matrix too (§0.4.10's `eig(...)<=0`
    threshold is the M-step's *repair* decision, not a claim that gmm_pdf can
    evaluate an indefinite covariance; a fixture with one would simply fail to
    export). So a negative
    eigenvalue is rejected unconditionally. What `matlab_compat` actually gates
    is whether a numerically-singular-but-PSD matrix (min eigenvalue within
    `_EIG_REL_TOL` of zero, e.g. the near-singular `c3_sigma_repair_*` fixtures) is
    tolerated -- `gmm_pdf_full`'s `allow_singular=True` can genuinely evaluate
    that case; `matlab_compat=False` asks for strict positive-definiteness."""
    d_c, _, M_c, K_c = gmm.sigma.shape
    for s in range(M_c):
        for k in range(K_c):
            sigma = gmm.sigma[:, :, s, k]
            if not np.allclose(sigma, sigma.T, rtol=0, atol=_SUM_ATOL):
                raise ValidationError(f"emissions[{c}].sigma[:,:,{s},{k}] is not symmetric")
            eigvals = np.linalg.eigvalsh(sigma)
            eig_tol = _EIG_REL_TOL * max(1.0, float(np.abs(eigvals).max()))
            min_eig = float(eigvals.min())
            if min_eig < -eig_tol:
                raise ValidationError(
                    f"emissions[{c}].sigma[:,:,{s},{k}] has a negative eigenvalue {min_eig} "
                    f"-- no valid Gaussian density exists (not a matlab_compat question)"
                )
            if min_eig < eig_tol:
                if not matlab_compat:
                    raise ValidationError(
                        f"emissions[{c}].sigma[:,:,{s},{k}] is numerically singular (min eigenvalue {min_eig}); "
                        f"matlab_compat=False requires strict positive-definiteness"
                    )
                # else: numerically singular but PSD -- matlab_compat=True tolerates this,
                # matching what gmm_pdf_full's allow_singular=True can actually evaluate.


def _observation_structure_error(structure: LSIMStructure, Y: NDArray) -> str | None:
    """The cheap, per-array half of `validate_observations` (shape, dtype, `D`,
    `T >= 1`), returning the exact message it raises, or `None`. Shared by
    `validate_observations` and the chunked `_iter_validated_observation_chunks`
    so both report identical errors in identical order."""
    if Y.ndim != 2:
        return f"Y.shape={Y.shape}, expected 2-D (D, T)"
    if not np.issubdtype(Y.dtype, np.floating):
        return f"Y.dtype={Y.dtype}, expected a floating dtype (§2.1.1: observations are float and may be NaN)"
    if Y.shape[0] != structure.D:
        return f"Y.shape[0]={Y.shape[0]} != structure.D={structure.D}"
    T = Y.shape[1]
    if T < 1:
        return f"Y has T={T}, expected at least 1 timestep"
    return None


_INF_MESSAGE = "Y contains +/-Inf (only NaN is a legal missing-data marker)"


def _all_nan_channel_message(c: int, T: int) -> str:
    return f"channel {c} is NaN for every timestep across all {T} samples"


def validate_observations(structure: LSIMStructure, Y: NDArray[np.float64]) -> None:
    message = _observation_structure_error(structure, Y)
    if message is not None:
        raise ValidationError(message)
    T = Y.shape[1]

    if np.isinf(Y).any():
        raise ValidationError(_INF_MESSAGE)

    for c in range(structure.C):
        channel = Y[structure.dim_slice(c), :]
        if np.isnan(channel).all():
            warnings.warn(_all_nan_channel_message(c, T), stacklevel=2)


# ==========================================================================
# P10f: chunked, vectorized dataset validation and the private "already
# validated" dataset marker used inside the mini-batch training drivers.
# ==========================================================================

DEFAULT_MAX_CHUNK_OBSERVATIONS = 1 << 22  # 4,194,304 observation values (D x T) per chunk: ~32 MiB of float64


def _resolve_max_chunk_observations(max_chunk_observations: int | None) -> int:
    if max_chunk_observations is None:
        return DEFAULT_MAX_CHUNK_OBSERVATIONS
    if not isinstance(max_chunk_observations, (int, np.integer)) or isinstance(max_chunk_observations, bool) or max_chunk_observations < 1:
        raise ValidationError(f"max_chunk_observations={max_chunk_observations!r}, expected a positive integer")
    return int(max_chunk_observations)


def _iter_validated_observation_chunks(
    structure: LSIMStructure,
    Y_trials,
    *,
    label: str,
    max_chunk_observations: int | None = None,
):
    """Validates every trial with exactly `validate_observations`'s checks and
    messages (prefixed `"{label}[i]: "`), in bounded chunks, yielding
    `(start, stop, X, nan_mask)` for each fully validated chunk: `X` is the
    chunk's trials concatenated along time as float64 `(D, sum T)` and
    `nan_mask = np.isnan(X)`. A chunk holds at most `max_chunk_observations`
    values (`D * sum T`), except that a single trial larger than that forms
    its own chunk -- memory is bounded by the chunk, never by the dataset.

    Error/warning order is identical to calling `validate_observations` on
    each trial in turn: trials are structurally checked as the chunk is
    gathered; a chunk stops before the first structurally invalid trial, is
    Inf-checked and warned for (raising for its earliest Inf trial after
    warning for the trials before it), and only then is the structural error
    raised. Nothing is yielded for a chunk until all of its values passed, so
    a caller never accumulates an observation that has not been validated."""
    budget = _resolve_max_chunk_observations(max_chunk_observations)
    D = structure.D
    n = len(Y_trials)
    i = 0
    while i < n:
        start = i
        chunk: list = []
        total = 0
        pending_error: tuple[int, str] | None = None
        while i < n:
            Y = Y_trials[i]
            message = _observation_structure_error(structure, Y)
            if message is not None:
                pending_error = (i, message)
                break
            size = D * Y.shape[1]
            if chunk and total + size > budget:
                break
            chunk.append(Y)
            total += size
            i += 1
        stop = start + len(chunk)
        if chunk:
            lengths = np.fromiter((Y.shape[1] for Y in chunk), dtype=np.int64, count=len(chunk))
            X = np.concatenate(chunk, axis=1).astype(np.float64, copy=False) if len(chunk) > 1 else np.asarray(chunk[0], dtype=np.float64)
            bounds = np.concatenate(([0], np.cumsum(lengths)))
            inf_cols = np.isinf(X).any(axis=0)
            first_bad = None
            if inf_cols.any():
                first_bad = int(np.searchsorted(bounds, int(np.argmax(inf_cols)), side="right") - 1)
            nan_mask = np.isnan(X)
            limit = first_bad if first_bad is not None else len(chunk)
            if limit > 0:
                flagged = []
                for c in range(structure.C):
                    all_nan_t = nan_mask[structure.dim_slice(c), :].all(axis=0)
                    per_trial = np.logical_and.reduceat(all_nan_t, bounds[:-1])
                    for local in np.flatnonzero(per_trial[:limit]):
                        flagged.append((int(local), c))
                for local, c in sorted(flagged):
                    warnings.warn(_all_nan_channel_message(c, int(lengths[local])), stacklevel=3)
            if first_bad is not None:
                raise ValidationError(f"{label}[{start + first_bad}]: {_INF_MESSAGE}")
            yield start, stop, X, nan_mask
        if pending_error is not None:
            index, message = pending_error
            raise ValidationError(f"{label}[{index}]: {message}")


def validate_observation_list(structure: LSIMStructure, Y_trials, *, label: str = "Y_trials", max_chunk_observations: int | None = None) -> None:
    """Every trial through `validate_observations`'s checks, chunk-vectorized
    (see `_iter_validated_observation_chunks`); raises `ValidationError`
    naming `"{label}[i]"` for the first invalid trial."""
    for _ in _iter_validated_observation_chunks(structure, Y_trials, label=label, max_chunk_observations=max_chunk_observations):
        pass


class _ValidatedTrials(tuple):
    """PRIVATE marker: an immutable snapshot of trial references whose every
    observation has already passed `validate_observation_list` for
    `structure`. Built only inside the mini-batch training drivers, after
    validation, so their repeated internal E-steps and coherent scoring skip
    re-validating the same observations. Not a public skip-validation switch:
    public functions still validate any ordinary list, and a snapshot for a
    different structure is re-validated as usual."""

    def __new__(cls, trials, structure: LSIMStructure):
        obj = super().__new__(cls, trials)
        obj.structure = structure
        return obj

    def subset(self, indices) -> "_ValidatedTrials":
        return _ValidatedTrials([self[i] for i in indices], self.structure)


def _is_prevalidated(structure: LSIMStructure, Y_trials) -> bool:
    return isinstance(Y_trials, _ValidatedTrials) and Y_trials.structure == structure
