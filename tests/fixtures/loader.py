"""Loads LSIM fixtures per SCHEMA.md. Pure Python, no pytest dependency, so P2+
can import it directly for parity tests without pulling in the test runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_MAJOR_SUPPORTED = "1"


class SchemaVersionError(RuntimeError):
    pass


class FixtureValidationError(RuntimeError):
    """Raised by load_posterior_fixture/load_em_trace_fixture when validate=True
    (the default) and the loaded fixture violates SCHEMA.md's own contract --
    shape mismatch, unexpected NaN/Inf, schema-version mismatch. Previously
    these checks lived only in test_fixture_schema.py, so importing this module
    directly (as P2+ is meant to) got no validation at all; that was a real gap
    between what SCHEMA.md promises the loader does and what it actually did.
    """


def load_manifest(fixtures_dir: Path) -> dict:
    manifest_path = Path(fixtures_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    major = str(manifest["schema_version"]).split(".")[0]
    if major != SCHEMA_MAJOR_SUPPORTED:
        raise SchemaVersionError(
            f"manifest schema_version {manifest['schema_version']!r} has major "
            f"version {major!r}, this loader supports major version "
            f"{SCHEMA_MAJOR_SUPPORTED!r} only (SCHEMA.md may have changed shape)"
        )
    return manifest


def iter_posterior_fixture_ids(fixtures_dir: Path) -> list[str]:
    manifest = load_manifest(fixtures_dir)
    ids = []
    for scenario in manifest["scenarios"]:
        for variant, entry in scenario["files"].items():
            ids.append(Path(entry["path"]).stem)
    return ids


def iter_em_trace_scenario_ids(fixtures_dir: Path) -> list[str]:
    manifest = load_manifest(fixtures_dir)
    return [t["scenario_id"] for t in manifest.get("em_traces", [])]


@dataclass
class Trial:
    Y: np.ndarray
    P_O_model: float
    alpha: np.ndarray
    beta: np.ndarray
    alpha_T: np.ndarray
    b_c_ot_nc: np.ndarray
    B: np.ndarray
    B_comp: np.ndarray


@dataclass
class FbConstants:
    variant: str
    beta_floor: float
    pi0_reg: float
    f_reg: float
    svi_eps: float
    poise_refinement: bool = False  # P12: poise only; absent (False) in the pre-P12 sed/fast fixtures


@dataclass
class SedDiagnostics:
    t_samples: np.ndarray
    f: np.ndarray
    h: np.ndarray
    delta: np.ndarray
    Gz: np.ndarray


@dataclass
class PosteriorFixture:
    fixture_id: str
    scenario_id: str
    variant: str
    schema_version: str
    matlab_release: str
    seed: int
    C: int
    S: int
    D: int
    channel_num_states: np.ndarray
    dim_observation: np.ndarray
    num_gmm_component: np.ndarray
    sigma_diag: bool
    num_trials: int
    length_observation: np.ndarray
    fb_constants: FbConstants
    state_numbers_index: np.ndarray
    dimension_numbers_index: np.ndarray
    pi_0: np.ndarray
    transitions_matrices: np.ndarray
    coupling_tetha_IM: np.ndarray
    P_all: np.ndarray
    mu_all: np.ndarray
    sigma_all: np.ndarray
    notes: str
    trials: list[Trial] = field(default_factory=list)
    sed_diagnostics: SedDiagnostics | None = None

    def trial(self, i: int) -> Trial:
        return self.trials[i]


def _check_shape(errs: list[str], name: str, arr: np.ndarray, expected_shape: tuple[int, ...]) -> bool:
    """Exact shape check that never raises: returns False (and records an
    error) instead of letting a malformed array's later use crash with
    IndexError. Every subsequent check that indexes into `arr` must be gated
    on this returning True.

    Deliberately exact, not flatten-and-count: a (C,1) column vector or a
    (1,C) row vector must fail this where SCHEMA.md declares (C,) -- the
    prior version of this check called `.reshape(-1).size`, which silently
    accepted any shape with the right element count, including transposed or
    extra-dimensioned vectors the schema does not declare.
    """
    if arr.shape != expected_shape:
        errs.append(f"{name}.shape={arr.shape} != {expected_shape}")
        return False
    return True


def _check_no_nan_inf(errs: list[str], name: str, arr: np.ndarray, *, allow_nan: bool = False) -> None:
    if np.any(np.isinf(arr)):
        errs.append(f"{name} contains Inf")
    if not allow_nan and np.any(np.isnan(arr)):
        errs.append(f"{name} contains NaN (not permitted here)")


def _check_finite_scalar(errs: list[str], name: str, value: float) -> None:
    if not np.isfinite(value):
        errs.append(f"{name} is not finite: {value}")


def _check_dtype(errs: list[str], name: str, arr: np.ndarray, kind: str) -> bool:
    """`kind` is `"i"` (integer) or `"f"` (floating), checked via
    `np.issubdtype` so e.g. `int32` and `int64` both count as integer.
    Returns whether the dtype was acceptable, so a caller can skip a
    downstream check (positivity, NaN/Inf) that would otherwise misreport a
    dtype problem as something else, or raise on a type it doesn't support."""
    expected = {"i": np.integer, "f": np.floating}[kind]
    if not np.issubdtype(arr.dtype, expected):
        label = "an integer" if kind == "i" else "a floating"
        errs.append(f"{name}.dtype={arr.dtype}, expected {label} dtype")
        return False
    return True


def _check_positive(errs: list[str], name: str, arr: np.ndarray) -> None:
    if np.any(arr <= 0):
        errs.append(f"{name} must be strictly positive, got {arr.tolist()}")


def _check_exact_dtype(errs: list[str], name: str, arr: np.ndarray, dtype: np.dtype) -> None:
    """`_check_dtype` above only checks dtype *kind* (any integer, any
    floating), so a fixture stored as float32/int32 passes it despite
    SCHEMA.md declaring float64/int64 for every P10a field. This checks the
    exact dtype the schema promises."""
    if arr.dtype != dtype:
        errs.append(f"{name}.dtype={arr.dtype}, expected exactly {np.dtype(dtype)}")


def validate_posterior_fixture(fx: "PosteriorFixture") -> None:
    """SCHEMA.md sec 'Validation the loader performs on every fixture' -- every
    schema-declared shape and NaN/Inf rule, raised as FixtureValidationError.
    Called by load_posterior_fixture(validate=True) (the default); split out
    so callers can also re-run it on a fixture they've mutated for a test.

    Never raises IndexError/ValueError itself: offset/length arrays are
    length-checked before anything indexes into them, and the whole body is
    wrapped so any exception this function did not anticipate still comes out
    as a FixtureValidationError, not a raw crash on a malformed fixture.
    """
    errs: list[str] = []
    try:
        # --- offsets and lengths: exact shape (C,)/(C+1,)/(num_trials,) first,
        # then the invariants that only make sense once the shape is right ---
        sni_ok = _check_shape(errs, "state_numbers_index", fx.state_numbers_index, (fx.C + 1,))
        dni_ok = _check_shape(errs, "dimension_numbers_index", fx.dimension_numbers_index, (fx.C + 1,))
        cns_ok = _check_shape(errs, "channel_num_states", fx.channel_num_states, (fx.C,))
        do_ok = _check_shape(errs, "dim_observation", fx.dim_observation, (fx.C,))
        _check_shape(errs, "num_gmm_component", fx.num_gmm_component, (fx.C,))
        length_obs_ok = _check_shape(errs, "length_observation", fx.length_observation, (fx.num_trials,))

        if sni_ok:
            if int(fx.state_numbers_index[0]) != 0:
                errs.append(f"state_numbers_index[0]={fx.state_numbers_index[0]} != 0")
            if int(fx.state_numbers_index[-1]) != fx.S:
                errs.append(f"state_numbers_index[-1]={fx.state_numbers_index[-1]} != S={fx.S}")
            if cns_ok:
                diffs = np.diff(fx.state_numbers_index)
                if not np.array_equal(diffs, fx.channel_num_states):
                    errs.append(f"diff(state_numbers_index)={diffs.tolist()} != "
                                f"channel_num_states={fx.channel_num_states.tolist()}")
        if dni_ok:
            if int(fx.dimension_numbers_index[0]) != 0:
                errs.append(f"dimension_numbers_index[0]={fx.dimension_numbers_index[0]} != 0")
            if int(fx.dimension_numbers_index[-1]) != fx.D:
                errs.append(f"dimension_numbers_index[-1]={fx.dimension_numbers_index[-1]} != D={fx.D}")
            if do_ok:
                diffs = np.diff(fx.dimension_numbers_index)
                if not np.array_equal(diffs, fx.dim_observation):
                    errs.append(f"diff(dimension_numbers_index)={diffs.tolist()} != "
                                f"dim_observation={fx.dim_observation.tolist()}")
        if cns_ok and int(fx.channel_num_states.sum()) != fx.S:
            errs.append(f"sum(channel_num_states)={fx.channel_num_states.sum()} != S={fx.S}")
        if do_ok and int(fx.dim_observation.sum()) != fx.D:
            errs.append(f"sum(dim_observation)={fx.dim_observation.sum()} != D={fx.D}")

        max_M = int(fx.channel_num_states.max()) if cns_ok and fx.channel_num_states.size else 0
        max_K = int(fx.num_gmm_component.max()) if fx.num_gmm_component.size else 0
        max_d = int(fx.dim_observation.max()) if do_ok and fx.dim_observation.size else 0
        sigma_shape = (fx.D, max_M, max_K) if fx.sigma_diag else (fx.D, max_d, max_M, max_K)

        # --- packed model parameters (lsim_pack_para output) ---
        _check_shape(errs, "pi_0", fx.pi_0, (fx.S,))
        _check_no_nan_inf(errs, "pi_0", fx.pi_0)
        _check_shape(errs, "transitions_matrices", fx.transitions_matrices, (fx.S, fx.S))
        _check_no_nan_inf(errs, "transitions_matrices", fx.transitions_matrices)
        _check_shape(errs, "coupling_tetha_IM", fx.coupling_tetha_IM, (fx.C, fx.C))
        _check_no_nan_inf(errs, "coupling_tetha_IM", fx.coupling_tetha_IM)
        _check_shape(errs, "P_all", fx.P_all, (fx.C, max_M, max_K))
        _check_no_nan_inf(errs, "P_all", fx.P_all)
        _check_shape(errs, "mu_all", fx.mu_all, (fx.D, max_M, max_K))
        _check_no_nan_inf(errs, "mu_all", fx.mu_all)
        _check_shape(errs, "sigma_all", fx.sigma_all, sigma_shape)
        _check_no_nan_inf(errs, "sigma_all", fx.sigma_all)

        # --- per-trial data and outputs ---
        if len(fx.trials) != fx.num_trials:
            errs.append(f"len(trials)={len(fx.trials)} != num_trials={fx.num_trials}")
        for i, trial in enumerate(fx.trials):
            if not length_obs_ok or i >= fx.length_observation.shape[0]:
                continue
            T_i = int(fx.length_observation[i])
            _check_shape(errs, f"trial{i}.Y", trial.Y, (fx.D, T_i))
            _check_no_nan_inf(errs, f"trial{i}.Y", trial.Y, allow_nan=True)  # only Y may
            for name in ("alpha", "beta", "alpha_T", "b_c_ot_nc", "B"):
                arr = getattr(trial, name)
                _check_shape(errs, f"trial{i}.{name}", arr, (fx.S, T_i))
                _check_no_nan_inf(errs, f"trial{i}.{name}", arr)
            _check_shape(errs, f"trial{i}.B_comp", trial.B_comp, (fx.S, max_K, T_i))
            _check_no_nan_inf(errs, f"trial{i}.B_comp", trial.B_comp)
            _check_finite_scalar(errs, f"trial{i}.P_O_model", trial.P_O_model)

        # --- SED diagnostics ---
        if fx.sed_diagnostics is not None:
            if fx.variant != "sed":
                errs.append(f"sed_diagnostics present on a {fx.variant}-variant fixture")
            diag = fx.sed_diagnostics
            if diag.t_samples.ndim != 1:
                errs.append(f"diag_t_samples.shape={diag.t_samples.shape}, expected 1-D (n,)")
            else:
                n = diag.t_samples.shape[0]
                _check_shape(errs, "diag_f", diag.f, (n, fx.C, fx.C))
                _check_no_nan_inf(errs, "diag_f", diag.f)
                _check_shape(errs, "diag_h", diag.h, (n, fx.C))
                _check_no_nan_inf(errs, "diag_h", diag.h)
                _check_shape(errs, "diag_delta", diag.delta, (n, fx.C, fx.C))
                _check_no_nan_inf(errs, "diag_delta", diag.delta)
                _check_shape(errs, "diag_Gz", diag.Gz, (n, fx.C, fx.S))
                _check_no_nan_inf(errs, "diag_Gz", diag.Gz)

                # Every sampled t indexes into trial 0 only (SCHEMA.md), and the
                # exporter's reconstruction reads alpha[:,t] AND alpha[:,t+1]
                # (export_fixtures.m's local_sed_diagnostics), so the valid
                # 0-based range is [0, T0-2], not [0, T0-1].
                if length_obs_ok and fx.length_observation.shape[0] >= 1:
                    T0 = int(fx.length_observation[0])
                    if np.any(diag.t_samples < 0) or np.any(diag.t_samples > T0 - 2):
                        errs.append(
                            f"diag_t_samples out of range for trial 0 (need 0..{T0 - 2}): "
                            f"{diag.t_samples.tolist()}"
                        )

    except Exception as exc:  # noqa: BLE001 -- deliberately broad: turn any
        # unanticipated crash on a malformed fixture into a validation error
        # rather than a raw traceback, per the explicit requirement that
        # malformed offsets/lengths raise FixtureValidationError, not IndexError.
        errs.append(f"validation raised {type(exc).__name__}: {exc}")

    if errs:
        raise FixtureValidationError(f"{fx.fixture_id}: " + "; ".join(errs))



def _variant_label_from_id(fixture_id: str, fallback: str) -> str:
    """An EM-trace fixture stores `fbc_variant='poise'` in both refinement modes,
    so its mode lives only in the file name (`em_trace__<scenario>__poise_refined`).
    Posterior fixtures carry the label as their own `variant` field and pass it
    directly."""
    for suffix in ("__poise_refined", "__poise"):
        if fixture_id.endswith(suffix):
            return suffix[2:]
    return fallback


def _poise_refinement_from(d, fixture_id: str, variant_label: str) -> bool:
    """SCHEMA.md 1.6.0's `fbc_poise_refinement`, read strictly.

    Three things are checked rather than assumed, because this one flag decides
    which of two different algorithms a golden fixture is the ground truth for:

    1. The stored value is an exact integer scalar (`int64`/`int32`/`uint8`,
       never a float, never an array) -- `bool(int(x))` would happily accept
       `0.5` or a `(1,1)` array and quietly produce `False`/`True`.
    2. Its value is exactly 0 or 1.
    3. It AGREES with the fixture's own variant label: `<scenario>__poise` must
       carry `False` and `<scenario>__poise_refined` must carry `True`. A file
       whose name and contents disagree is a corrupt export, not something to
       silently trust in either direction.

    Absent (schema < 1.6.0, the pre-P12 sed/fast files) means `False`: the flag
    has no meaning outside `variant='poise'`.
    """
    if "fbc_poise_refinement" not in d:
        expected_absent = {"poise": False, "poise_refined": True}.get(variant_label)
        if expected_absent:
            raise FixtureValidationError(
                f"{fixture_id}: a '{variant_label}' fixture must record fbc_poise_refinement"
            )
        return False

    raw = d["fbc_poise_refinement"]
    arr = np.asarray(raw)
    if arr.shape != ():
        raise FixtureValidationError(
            f"{fixture_id}: fbc_poise_refinement must be a scalar, got shape {arr.shape}"
        )
    if arr.dtype.kind not in ("i", "u"):
        raise FixtureValidationError(
            f"{fixture_id}: fbc_poise_refinement must be stored as an exact integer, "
            f"got dtype {arr.dtype}"
        )
    value = int(arr)
    if value not in (0, 1):
        raise FixtureValidationError(
            f"{fixture_id}: fbc_poise_refinement must be 0 or 1, got {value}"
        )
    refinement = bool(value)

    expected = {"poise": False, "poise_refined": True}.get(variant_label)
    if expected is not None and refinement != expected:
        raise FixtureValidationError(
            f"{fixture_id}: variant label {variant_label!r} implies "
            f"poise_refinement={expected}, but the fixture records {refinement}"
        )
    if expected is None and refinement:
        raise FixtureValidationError(
            f"{fixture_id}: poise_refinement is set on a {variant_label!r} fixture, where the "
            f"flag has no meaning (the unused-field policy records it as false)"
        )
    return refinement

def load_posterior_fixture(fixtures_dir: Path, fixture_id: str, validate: bool = True) -> PosteriorFixture:
    npz_path = Path(fixtures_dir) / f"{fixture_id}.npz"
    d = np.load(npz_path, allow_pickle=False)

    major = str(d["schema_version"]).split(".")[0]
    if major != SCHEMA_MAJOR_SUPPORTED:
        raise SchemaVersionError(
            f"{fixture_id}: schema_version {str(d['schema_version'])!r} has major "
            f"version {major!r}, this loader supports {SCHEMA_MAJOR_SUPPORTED!r} only"
        )

    def s(key: str) -> str:
        return str(d[key])

    num_trials = int(d["num_trials"])
    trials = []
    for i in range(num_trials):
        suf = f"_trial{i}"
        trials.append(Trial(
            Y=d[f"Y{suf}"],
            P_O_model=float(d[f"P_O_model{suf}"]),
            alpha=d[f"alpha{suf}"],
            beta=d[f"beta{suf}"],
            alpha_T=d[f"alpha_T{suf}"],
            b_c_ot_nc=d[f"b_c_ot_nc{suf}"],
            B=d[f"P_observ_cond_to_state{suf}"],
            B_comp=d[f"P_observ_cond_to_state_comp{suf}"],
        ))

    has_diag = bool(int(d["has_sed_diagnostics"]))
    sed_diag = None
    if has_diag:
        if s("variant") != "sed":
            raise SchemaVersionError(
                f"{fixture_id}: has_sed_diagnostics=1 on a {s('variant')}-variant fixture -- "
                "only the sed kernel has delta-solver intermediates to reconstruct"
            )
        sed_diag = SedDiagnostics(
            t_samples=d["diag_t_samples"], f=d["diag_f"], h=d["diag_h"],
            delta=d["diag_delta"], Gz=d["diag_Gz"],
        )
    else:
        for key in ("diag_f", "diag_h", "diag_delta", "diag_Gz", "diag_t_samples"):
            if key in d.files:
                raise SchemaVersionError(
                    f"{fixture_id}: has_sed_diagnostics=0 but {key!r} is present -- "
                    "loader/exporter disagree on whether diagnostics were written"
                )

    fx = PosteriorFixture(
        fixture_id=fixture_id,
        scenario_id=s("scenario_id"),
        variant=s("variant"),
        schema_version=s("schema_version"),
        matlab_release=s("matlab_release"),
        seed=int(d["seed"]),
        C=int(d["C"]), S=int(d["S"]), D=int(d["D"]),
        channel_num_states=d["channel_num_states"],
        dim_observation=d["dim_observation"],
        num_gmm_component=d["num_gmm_component"],
        sigma_diag=bool(int(d["sigma_diag"])),
        num_trials=num_trials,
        length_observation=d["length_observation"],
        fb_constants=FbConstants(
            variant=s("fbc_variant"),
            beta_floor=float(d["fbc_beta_floor"]),
            pi0_reg=float(d["fbc_pi0_reg"]),
            f_reg=float(d["fbc_f_reg"]),
            svi_eps=float(d["fbc_svi_eps"]),
            poise_refinement=_poise_refinement_from(d, fixture_id, s("variant")),
        ),
        state_numbers_index=d["state_numbers_index"],
        dimension_numbers_index=d["dimension_numbers_index"],
        pi_0=d["pi_0"],
        transitions_matrices=d["transitions_matrices"],
        coupling_tetha_IM=d["coupling_tetha_IM"],
        P_all=d["P_all"],
        mu_all=d["mu_all"],
        sigma_all=d["sigma_all"],
        notes=s("notes"),
        trials=trials,
        sed_diagnostics=sed_diag,
    )
    if validate:
        validate_posterior_fixture(fx)
    return fx


@dataclass
class EmIteration:
    itr: int
    params_out_pi0: np.ndarray
    params_out_transition: np.ndarray
    params_out_coupling: np.ndarray
    params_out_Pall: np.ndarray
    params_out_muall: np.ndarray
    params_out_sigmaall: np.ndarray
    loglik_params_in: float
    AIC_params_in: float
    BIC_params_in: float


@dataclass
class EmTraceFixture:
    scenario_id: str
    schema_version: str
    matlab_release: str
    seed: int
    notes: str
    C: int
    D: int
    sigma_diag: bool
    num_trials: int
    length_observation: np.ndarray
    channel_num_states: np.ndarray
    dim_observation: np.ndarray
    num_gmm_component: np.ndarray
    fb_constants: FbConstants
    state_numbers_index: np.ndarray
    dimension_numbers_index: np.ndarray
    Y: np.ndarray
    # Raw init_model, exactly as a caller would build the cell-API input --
    # NOT what iteration 1's E-step actually runs on (see effective_pi_0_iter1).
    init_pi_0: np.ndarray
    init_transitions_matrices: np.ndarray
    init_coupling_tetha_IM: np.ndarray
    init_P_all: np.ndarray
    init_mu_all: np.ndarray
    init_sigma_all: np.ndarray
    # Effective params_in for iteration 1: em_lsim.m:159-164 applies
    # pi_0 <- (1e-4+pi_0)/sum(...) per channel block once, before the first
    # E-step, whenever left2right==0. init_pi_0 above does NOT reproduce
    # em_loglik_params_in_iter1 -- only this one does (verified by the
    # exporter itself at generation time, not just asserted here).
    effective_pi_0_iter1: np.ndarray
    effective_transitions_matrices_iter1: np.ndarray
    effective_coupling_tetha_IM_iter1: np.ndarray
    effective_P_all_iter1: np.ndarray
    effective_mu_all_iter1: np.ndarray
    effective_sigma_all_iter1: np.ndarray
    iterations: list[EmIteration]

    @property
    def Y_trials(self) -> list[np.ndarray]:
        """Splits the concatenated `Y` (`em_lsim.m`'s own `all_observation`
        layout) back into one `(D, T_r)` array per trial, using
        `length_observation`'s cumulative boundaries -- `num_trials==1`
        fixtures get a one-element list, unchanged from `[self.Y]`."""
        bounds = np.concatenate([[0], np.cumsum(self.length_observation)])
        return [self.Y[:, bounds[i] : bounds[i + 1]] for i in range(self.num_trials)]

    def params_in(self, itr: int) -> dict:
        """params_in[itr] is never a stored array -- it's params_out[itr-1],
        or this fixture's EFFECTIVE (not raw init_model) parameters at
        itr==1. Recovering it here, rather than duplicating it in every
        .npz, is the point of naming the stored fields params_out /
        *_params_in instead of an unqualified 'params'."""
        if itr == 1:
            return dict(
                pi_0=self.effective_pi_0_iter1,
                transition_matrices_IM=self.effective_transitions_matrices_iter1,
                coupling_tetha_IM=self.effective_coupling_tetha_IM_iter1,
                P_all=self.effective_P_all_iter1,
                mu_all=self.effective_mu_all_iter1, sigma_all=self.effective_sigma_all_iter1,
            )
        prev = self.iterations[itr - 2]
        assert prev.itr == itr - 1
        return dict(
            pi_0=prev.params_out_pi0, transition_matrices_IM=prev.params_out_transition,
            coupling_tetha_IM=prev.params_out_coupling, P_all=prev.params_out_Pall,
            mu_all=prev.params_out_muall, sigma_all=prev.params_out_sigmaall,
        )


def validate_em_trace_fixture(fx: "EmTraceFixture") -> None:
    """Same contract as validate_posterior_fixture: every schema-declared shape
    and NaN/Inf rule for the EM-trace fixture family, and no exception escapes
    as anything other than FixtureValidationError."""
    errs: list[str] = []
    try:
        # --- offsets and lengths: exact shape first, same as
        # validate_posterior_fixture, then the offset-consistency invariants ---
        sni_ok = _check_shape(errs, "state_numbers_index", fx.state_numbers_index, (fx.C + 1,))
        dni_ok = _check_shape(errs, "dimension_numbers_index", fx.dimension_numbers_index, (fx.C + 1,))
        cns_ok = _check_shape(errs, "channel_num_states", fx.channel_num_states, (fx.C,))
        do_ok = _check_shape(errs, "dim_observation", fx.dim_observation, (fx.C,))
        _check_shape(errs, "num_gmm_component", fx.num_gmm_component, (fx.C,))
        length_obs_ok = _check_shape(errs, "length_observation", fx.length_observation, (fx.num_trials,))

        S = int(fx.state_numbers_index[-1]) if sni_ok else None
        if sni_ok:
            if int(fx.state_numbers_index[0]) != 0:
                errs.append(f"state_numbers_index[0]={fx.state_numbers_index[0]} != 0")
            if cns_ok:
                diffs = np.diff(fx.state_numbers_index)
                if not np.array_equal(diffs, fx.channel_num_states):
                    errs.append(f"diff(state_numbers_index)={diffs.tolist()} != "
                                f"channel_num_states={fx.channel_num_states.tolist()}")
        if dni_ok:
            if int(fx.dimension_numbers_index[0]) != 0:
                errs.append(f"dimension_numbers_index[0]={fx.dimension_numbers_index[0]} != 0")
            if int(fx.dimension_numbers_index[-1]) != fx.D:
                errs.append(f"dimension_numbers_index[-1]={fx.dimension_numbers_index[-1]} != D={fx.D}")
            if do_ok:
                diffs = np.diff(fx.dimension_numbers_index)
                if not np.array_equal(diffs, fx.dim_observation):
                    errs.append(f"diff(dimension_numbers_index)={diffs.tolist()} != "
                                f"dim_observation={fx.dim_observation.tolist()}")
        if S is not None and cns_ok and int(fx.channel_num_states.sum()) != S:
            errs.append(f"sum(channel_num_states)={fx.channel_num_states.sum()} != S={S}")
        if do_ok and int(fx.dim_observation.sum()) != fx.D:
            errs.append(f"sum(dim_observation)={fx.dim_observation.sum()} != D={fx.D}")

        max_M = int(fx.channel_num_states.max()) if cns_ok and fx.channel_num_states.size else 0
        max_K = int(fx.num_gmm_component.max()) if fx.num_gmm_component.size else 0
        max_d = int(fx.dim_observation.max()) if do_ok and fx.dim_observation.size else 0
        sigma_shape = (fx.D, max_M, max_K) if fx.sigma_diag else (fx.D, max_d, max_M, max_K)

        if S is not None:
            # --- raw init_model and effective iter-1 params: same shapes, two
            # distinct field-name sets (not a shared label + suffix), so an
            # error message names the exact attribute that's wrong.
            param_groups = (
                ("init_pi_0", "init_transitions_matrices", "init_coupling_tetha_IM",
                 "init_P_all", "init_mu_all", "init_sigma_all"),
                ("effective_pi_0_iter1", "effective_transitions_matrices_iter1",
                 "effective_coupling_tetha_IM_iter1", "effective_P_all_iter1",
                 "effective_mu_all_iter1", "effective_sigma_all_iter1"),
            )
            for pi0_name, trm_name, cpl_name, pall_name, mu_name, sig_name in param_groups:
                _check_shape(errs, pi0_name, getattr(fx, pi0_name), (S,))
                _check_no_nan_inf(errs, pi0_name, getattr(fx, pi0_name))
                _check_shape(errs, trm_name, getattr(fx, trm_name), (S, S))
                _check_no_nan_inf(errs, trm_name, getattr(fx, trm_name))
                _check_shape(errs, cpl_name, getattr(fx, cpl_name), (fx.C, fx.C))
                _check_no_nan_inf(errs, cpl_name, getattr(fx, cpl_name))
                _check_shape(errs, pall_name, getattr(fx, pall_name), (fx.C, max_M, max_K))
                _check_no_nan_inf(errs, pall_name, getattr(fx, pall_name))
                _check_shape(errs, mu_name, getattr(fx, mu_name), (fx.D, max_M, max_K))
                _check_no_nan_inf(errs, mu_name, getattr(fx, mu_name))
                _check_shape(errs, sig_name, getattr(fx, sig_name), sigma_shape)
                _check_no_nan_inf(errs, sig_name, getattr(fx, sig_name))

            # --- Y: requires length_observation to be trustworthy too. Y is the
            # concatenation of every trial (single-trial fixtures: T_all == T),
            # matching em_lsim.m's own internal `all_observation` layout --
            # see EmTraceFixture.Y_trials for splitting it back apart. ---
            if length_obs_ok:
                T_all = int(fx.length_observation.sum())
                _check_shape(errs, "Y", fx.Y, (fx.D, T_all))
                _check_no_nan_inf(errs, "Y", fx.Y, allow_nan=True)  # missing data is permitted

            # --- every EM output tensor, per iteration ---
            if len(fx.iterations) == 0:
                errs.append("no iterations recorded")
            for it in fx.iterations:
                _check_shape(errs, f"iter{it.itr}.params_out_pi0", it.params_out_pi0, (S,))
                _check_no_nan_inf(errs, f"iter{it.itr}.params_out_pi0", it.params_out_pi0)
                _check_shape(errs, f"iter{it.itr}.params_out_transition",
                             it.params_out_transition, (S, S))
                _check_no_nan_inf(errs, f"iter{it.itr}.params_out_transition", it.params_out_transition)
                _check_shape(errs, f"iter{it.itr}.params_out_coupling",
                             it.params_out_coupling, (fx.C, fx.C))
                _check_no_nan_inf(errs, f"iter{it.itr}.params_out_coupling", it.params_out_coupling)
                _check_shape(errs, f"iter{it.itr}.params_out_Pall", it.params_out_Pall, (fx.C, max_M, max_K))
                _check_no_nan_inf(errs, f"iter{it.itr}.params_out_Pall", it.params_out_Pall)
                _check_shape(errs, f"iter{it.itr}.params_out_muall", it.params_out_muall, (fx.D, max_M, max_K))
                _check_no_nan_inf(errs, f"iter{it.itr}.params_out_muall", it.params_out_muall)
                _check_shape(errs, f"iter{it.itr}.params_out_sigmaall", it.params_out_sigmaall, sigma_shape)
                _check_no_nan_inf(errs, f"iter{it.itr}.params_out_sigmaall", it.params_out_sigmaall)
                _check_finite_scalar(errs, f"iter{it.itr}.loglik_params_in", it.loglik_params_in)
                _check_finite_scalar(errs, f"iter{it.itr}.AIC_params_in", it.AIC_params_in)
                _check_finite_scalar(errs, f"iter{it.itr}.BIC_params_in", it.BIC_params_in)

    except Exception as exc:  # noqa: BLE001 -- see validate_posterior_fixture
        errs.append(f"validation raised {type(exc).__name__}: {exc}")

    if errs:
        raise FixtureValidationError(f"em_trace__{fx.scenario_id}: " + "; ".join(errs))


def load_em_trace_fixture(fixtures_dir: Path, scenario_id: str, validate: bool = True) -> EmTraceFixture:
    npz_path = Path(fixtures_dir) / f"em_trace__{scenario_id}.npz"
    d = np.load(npz_path, allow_pickle=False)

    def s(key: str) -> str:
        return str(d[key])

    major = str(d["schema_version"]).split(".")[0]
    if major != SCHEMA_MAJOR_SUPPORTED:
        raise SchemaVersionError(
            f"em_trace__{scenario_id}: schema_version {str(d['schema_version'])!r} has major "
            f"version {major!r}, this loader supports {SCHEMA_MAJOR_SUPPORTED!r} only"
        )

    num_iters = int(d["num_iters"])
    iterations = []
    for k in range(1, num_iters + 1):
        suf = f"_iter{k}"
        iterations.append(EmIteration(
            itr=k,
            params_out_pi0=d[f"em_params_out_pi0{suf}"],
            params_out_transition=d[f"em_params_out_transition{suf}"],
            params_out_coupling=d[f"em_params_out_coupling{suf}"],
            params_out_Pall=d[f"em_params_out_Pall{suf}"],
            params_out_muall=d[f"em_params_out_muall{suf}"],
            params_out_sigmaall=d[f"em_params_out_sigmaall{suf}"],
            loglik_params_in=float(d[f"em_loglik_params_in{suf}"]),
            AIC_params_in=float(d[f"em_AIC_params_in{suf}"]),
            BIC_params_in=float(d[f"em_BIC_params_in{suf}"]),
        ))

    fx = EmTraceFixture(
        scenario_id=s("scenario_id"),
        schema_version=s("schema_version"),
        matlab_release=s("matlab_release"),
        seed=int(d["seed"]),
        notes=s("notes"),
        C=int(d["C"]), D=int(d["D"]),
        sigma_diag=bool(int(d["sigma_diag"])),
        num_trials=int(d["num_trials"]),
        length_observation=d["length_observation"],
        channel_num_states=d["channel_num_states"],
        dim_observation=d["dim_observation"],
        num_gmm_component=d["num_gmm_component"],
        fb_constants=FbConstants(
            variant=s("fbc_variant"),
            beta_floor=float(d["fbc_beta_floor"]),
            pi0_reg=float(d["fbc_pi0_reg"]),
            f_reg=float(d["fbc_f_reg"]),
            svi_eps=float(d["fbc_svi_eps"]),
            poise_refinement=_poise_refinement_from(
                d, f"em_trace__{scenario_id}", _variant_label_from_id(scenario_id, s("fbc_variant"))
            ),
        ),
        state_numbers_index=d["state_numbers_index"],
        dimension_numbers_index=d["dimension_numbers_index"],
        Y=d["Y"],
        init_pi_0=d["init_pi_0"],
        init_transitions_matrices=d["init_transitions_matrices"],
        init_coupling_tetha_IM=d["init_coupling_tetha_IM"],
        init_P_all=d["init_P_all"],
        init_mu_all=d["init_mu_all"],
        init_sigma_all=d["init_sigma_all"],
        effective_pi_0_iter1=d["effective_pi_0_iter1"],
        effective_transitions_matrices_iter1=d["effective_transitions_matrices_iter1"],
        effective_coupling_tetha_IM_iter1=d["effective_coupling_tetha_IM_iter1"],
        effective_P_all_iter1=d["effective_P_all_iter1"],
        effective_mu_all_iter1=d["effective_mu_all_iter1"],
        effective_sigma_all_iter1=d["effective_sigma_all_iter1"],
        iterations=iterations,
    )
    if validate:
        validate_em_trace_fixture(fx)
    return fx


@dataclass
class InitReferenceFixture:
    """A single, bespoke fixture (SCHEMA.md's "Init-reference fixture"
    section) -- not manifest-driven, not part of either the posterior or
    EM-trace families above. `tools/matlab/export_init_reference.m`'s own
    run of `em_lsim.m`'s no-`init_model` branch (`fine_gmm_fitting`), the
    reference `tests/test_init.py`'s P5 statistical gate compares against."""

    schema_version: str
    matlab_release: str
    C: int
    channel_num_states: np.ndarray
    dim_observation: np.ndarray
    num_gmm_component: np.ndarray
    T: int
    sigma_diag: bool
    data_seed: int
    init_seed: int
    num_iters: int
    Y: np.ndarray
    loglik_params_in: np.ndarray

    @property
    def D(self) -> int:
        return int(self.dim_observation.sum())


def validate_init_reference_fixture(fx: "InitReferenceFixture") -> None:
    """Same contract as validate_posterior_fixture/validate_em_trace_fixture:
    every schema-declared shape and NaN/Inf rule, raised as
    FixtureValidationError, no exception escapes as anything else. Also
    checks dtype (integer per-channel fields and `loglik_params_in`/`Y`
    floating, never silently accepted the other way round), strict
    positivity of every per-channel count, and that `sigma_diag` is
    genuinely `True` (this fixture is diagonal-only by construction, not
    merely truthy) -- none of these were previously checked at all."""
    errs: list[str] = []
    try:
        cns_ok = _check_shape(errs, "channel_num_states", fx.channel_num_states, (fx.C,))
        do_ok = _check_shape(errs, "dim_observation", fx.dim_observation, (fx.C,))
        ngc_ok = _check_shape(errs, "num_gmm_component", fx.num_gmm_component, (fx.C,))

        for name, arr, shape_ok in (
            ("channel_num_states", fx.channel_num_states, cns_ok),
            ("dim_observation", fx.dim_observation, do_ok),
            ("num_gmm_component", fx.num_gmm_component, ngc_ok),
        ):
            if not shape_ok:
                continue
            _check_no_nan_inf(errs, name, arr)
            if _check_dtype(errs, name, arr, "i"):
                _check_positive(errs, name, arr)

        _check_shape(errs, "loglik_params_in", fx.loglik_params_in, (fx.num_iters,))
        _check_no_nan_inf(errs, "loglik_params_in", fx.loglik_params_in)
        _check_dtype(errs, "loglik_params_in", fx.loglik_params_in, "f")

        if do_ok:
            _check_shape(errs, "Y", fx.Y, (fx.D, fx.T))
            _check_no_nan_inf(errs, "Y", fx.Y)  # this fixture has no missingness by construction
            _check_dtype(errs, "Y", fx.Y, "f")

        if not isinstance(fx.sigma_diag, bool):
            errs.append(f"sigma_diag has type {type(fx.sigma_diag).__name__}, expected bool")
        elif fx.sigma_diag is not True:
            errs.append(f"sigma_diag={fx.sigma_diag}, expected True (this fixture is diagonal-only by construction)")

        _check_finite_scalar(errs, "T", fx.T)
        _check_finite_scalar(errs, "C", fx.C)
        _check_finite_scalar(errs, "num_iters", fx.num_iters)
        _check_finite_scalar(errs, "data_seed", fx.data_seed)
        _check_finite_scalar(errs, "init_seed", fx.init_seed)
        if fx.T < 1:
            errs.append(f"T={fx.T}, expected at least 1")
        if fx.num_iters < 1:
            errs.append(f"num_iters={fx.num_iters}, expected at least 1")

    except Exception as exc:  # noqa: BLE001 -- see validate_posterior_fixture
        errs.append(f"validation raised {type(exc).__name__}: {exc}")

    if errs:
        raise FixtureValidationError("init_reference: " + "; ".join(errs))


def load_init_reference_fixture(fixtures_dir: Path, validate: bool = True) -> InitReferenceFixture:
    npz_path = Path(fixtures_dir) / "init_reference.npz"
    d = np.load(npz_path, allow_pickle=False)

    major = str(d["schema_version"]).split(".")[0]
    if major != SCHEMA_MAJOR_SUPPORTED:
        raise SchemaVersionError(
            f"init_reference: schema_version {str(d['schema_version'])!r} has major "
            f"version {major!r}, this loader supports {SCHEMA_MAJOR_SUPPORTED!r} only"
        )

    fx = InitReferenceFixture(
        schema_version=str(d["schema_version"]),
        matlab_release=str(d["matlab_release"]),
        C=int(d["C"]),
        channel_num_states=d["channel_num_states"],
        dim_observation=d["dim_observation"],
        num_gmm_component=d["num_gmm_component"],
        T=int(d["T"]),
        sigma_diag=bool(int(d["sigma_diag"])),
        data_seed=int(d["data_seed"]),
        init_seed=int(d["init_seed"]),
        num_iters=int(d["num_iters"]),
        Y=d["Y"],
        loglik_params_in=d["loglik_params_in"],
    )
    if validate:
        validate_init_reference_fixture(fx)
    return fx


@dataclass
class P6Fixture:
    """A single, bespoke fixture (SCHEMA.md's "P6 fixture" section) -- not
    manifest-driven, produced by `tools/matlab/export_p6_fixtures.m` calling
    the actually-existing MATLAB P6 algorithms (`im_para_eqhmm.m` +
    `chmm_cartesian_product.m`, `viterbi_chmm.m`,
    `compute_joint_state_observation_prob.m`) on one hand-specified,
    heterogeneous small model. `X_star`/`X_alt` are zero-based (the exporter
    subtracts 1 from MATLAB's own 1-based hidden-state indices before
    saving), matching every other hidden-state array in this port."""

    schema_version: str
    matlab_release: str
    C: int
    channel_num_states: np.ndarray
    dim_observation: np.ndarray
    num_gmm_component: np.ndarray
    T: int
    sigma_diag: bool
    Y: np.ndarray
    joint_pi0: np.ndarray
    joint_transition: np.ndarray
    joint_P_all: np.ndarray
    joint_mu_all: np.ndarray
    joint_sigma_all: np.ndarray
    cond_trans_1: np.ndarray
    cond_trans_2: np.ndarray
    P_star_model: float
    X_star: np.ndarray
    log_joint_prob_viterbi_path: float
    X_alt: np.ndarray
    log_joint_prob_alt_path: float
    pi_0_packed: np.ndarray
    coupling_packed: np.ndarray
    transition_packed: np.ndarray
    P_all_packed: np.ndarray
    mu_all_packed: np.ndarray
    sigma_all_packed: np.ndarray

    @property
    def D(self) -> int:
        return int(self.dim_observation.sum())

    @property
    def J(self) -> int:
        return int(np.prod(self.channel_num_states))

    @property
    def K_joint(self) -> int:
        return int(np.prod(self.num_gmm_component))


def _check_row_stochastic(errs: list[str], name: str, arr: np.ndarray) -> None:
    """Every row of `arr` sums to exactly `1.0` (`rtol=0` -- a sum of
    `1.000001` must not slip through under the default `rtol=1e-5`) and
    every entry is non-negative -- a negative entry that's compensated by
    another entry to keep the sum at `1.0` would otherwise pass a sum-only
    check silently."""
    if (arr < 0).any():
        errs.append(f"{name} has negative entries")
    row_sums = arr.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-8):
        errs.append(f"{name} rows do not all sum to 1.0 (rtol=0): {row_sums.tolist()}")


def validate_p6_fixture(fx: "P6Fixture") -> None:
    """Same contract as the other fixture families: every schema-declared
    shape/dtype/NaN/Inf rule, raised as FixtureValidationError, no exception
    escapes as anything else. Also: every probability-sum check uses
    `rtol=0` (the default `rtol=1e-5` would silently accept a `joint_pi0`
    summing to `1.000001`), every probability array's non-negativity is
    checked explicitly (a negative entry compensated elsewhere would
    otherwise still sum to `1.0`), and the packed parameters are additionally
    checked by building a real `LSIMStructure`/`LSIMParams.from_packed(...)`
    and calling its own `.validate()` -- reusing the same, already-verified
    contract every other packed-parameter fixture is held to, rather than
    re-deriving an equivalent set of checks by hand here."""
    errs: list[str] = []
    try:
        cns_ok = _check_shape(errs, "channel_num_states", fx.channel_num_states, (fx.C,))
        do_ok = _check_shape(errs, "dim_observation", fx.dim_observation, (fx.C,))
        ngc_ok = _check_shape(errs, "num_gmm_component", fx.num_gmm_component, (fx.C,))
        for name, arr, shape_ok in (
            ("channel_num_states", fx.channel_num_states, cns_ok),
            ("dim_observation", fx.dim_observation, do_ok),
            ("num_gmm_component", fx.num_gmm_component, ngc_ok),
        ):
            if not shape_ok:
                continue
            _check_no_nan_inf(errs, name, arr)
            if _check_dtype(errs, name, arr, "i"):
                _check_positive(errs, name, arr)

        J, K_joint, D = fx.J, fx.K_joint, fx.D

        if not isinstance(fx.sigma_diag, bool):
            errs.append(f"sigma_diag has type {type(fx.sigma_diag).__name__}, expected bool")
        elif fx.sigma_diag is not True:
            errs.append(f"sigma_diag={fx.sigma_diag}, expected True (this fixture is diagonal-only by construction)")

        _check_finite_scalar(errs, "C", fx.C)
        if fx.C < 1:
            errs.append(f"C={fx.C}, expected at least 1")
        _check_finite_scalar(errs, "T", fx.T)
        if fx.T < 1:
            errs.append(f"T={fx.T}, expected at least 1")

        _check_shape(errs, "Y", fx.Y, (D, fx.T))
        _check_no_nan_inf(errs, "Y", fx.Y)
        _check_dtype(errs, "Y", fx.Y, "f")

        pi0_ok = _check_shape(errs, "joint_pi0", fx.joint_pi0, (J,))
        _check_no_nan_inf(errs, "joint_pi0", fx.joint_pi0)
        if _check_dtype(errs, "joint_pi0", fx.joint_pi0, "f") and pi0_ok:
            if (fx.joint_pi0 < 0).any():
                errs.append("joint_pi0 has negative entries")
            if not np.isclose(fx.joint_pi0.sum(), 1.0, rtol=0, atol=1e-8):
                errs.append(f"joint_pi0 sums to {fx.joint_pi0.sum()}, expected 1.0 (rtol=0)")

        trans_ok = _check_shape(errs, "joint_transition", fx.joint_transition, (J, J))
        _check_no_nan_inf(errs, "joint_transition", fx.joint_transition)
        if _check_dtype(errs, "joint_transition", fx.joint_transition, "f") and trans_ok:
            _check_row_stochastic(errs, "joint_transition", fx.joint_transition)

        p_all_ok = _check_shape(errs, "joint_P_all", fx.joint_P_all, (J, K_joint))
        _check_no_nan_inf(errs, "joint_P_all", fx.joint_P_all)
        if _check_dtype(errs, "joint_P_all", fx.joint_P_all, "f") and p_all_ok:
            _check_row_stochastic(errs, "joint_P_all", fx.joint_P_all)

        _check_shape(errs, "joint_mu_all", fx.joint_mu_all, (D, J, K_joint))
        _check_no_nan_inf(errs, "joint_mu_all", fx.joint_mu_all)
        _check_dtype(errs, "joint_mu_all", fx.joint_mu_all, "f")
        _check_shape(errs, "joint_sigma_all", fx.joint_sigma_all, (D, J, K_joint))
        _check_no_nan_inf(errs, "joint_sigma_all", fx.joint_sigma_all)
        _check_dtype(errs, "joint_sigma_all", fx.joint_sigma_all, "f")

        if cns_ok:
            for idx, M_c in enumerate(fx.channel_num_states, start=1):
                name = f"cond_trans_{idx}"
                arr = getattr(fx, name, None)
                if arr is None:
                    continue
                cond_ok = _check_shape(errs, name, arr, (J, int(M_c)))
                _check_no_nan_inf(errs, name, arr)
                if _check_dtype(errs, name, arr, "f") and cond_ok:
                    _check_row_stochastic(errs, name, arr)

        x_star_ok = _check_shape(errs, "X_star", fx.X_star, (fx.C, fx.T))
        _check_no_nan_inf(errs, "X_star", fx.X_star)
        _check_dtype(errs, "X_star", fx.X_star, "i")
        x_alt_ok = _check_shape(errs, "X_alt", fx.X_alt, (fx.C, fx.T))
        _check_no_nan_inf(errs, "X_alt", fx.X_alt)
        _check_dtype(errs, "X_alt", fx.X_alt, "i")
        if cns_ok:
            for name, shape_ok in (("X_star", x_star_ok), ("X_alt", x_alt_ok)):
                if not shape_ok:
                    continue
                arr = getattr(fx, name)
                for c in range(fx.C):
                    if np.any(arr[c] < 0) or np.any(arr[c] >= fx.channel_num_states[c]):
                        errs.append(f"{name}[{c}] contains a state outside [0, {fx.channel_num_states[c]})")

        _check_finite_scalar(errs, "P_star_model", fx.P_star_model)
        _check_finite_scalar(errs, "log_joint_prob_viterbi_path", fx.log_joint_prob_viterbi_path)
        _check_finite_scalar(errs, "log_joint_prob_alt_path", fx.log_joint_prob_alt_path)

        max_M = int(fx.channel_num_states.max()) if cns_ok and fx.channel_num_states.size else 0
        max_K = int(fx.num_gmm_component.max()) if ngc_ok and fx.num_gmm_component.size else 0
        S = int(fx.channel_num_states.sum()) if cns_ok else 0
        _check_shape(errs, "pi_0_packed", fx.pi_0_packed, (S,))
        _check_no_nan_inf(errs, "pi_0_packed", fx.pi_0_packed)
        _check_dtype(errs, "pi_0_packed", fx.pi_0_packed, "f")
        _check_shape(errs, "coupling_packed", fx.coupling_packed, (fx.C, fx.C))
        _check_no_nan_inf(errs, "coupling_packed", fx.coupling_packed)
        _check_dtype(errs, "coupling_packed", fx.coupling_packed, "f")
        _check_shape(errs, "transition_packed", fx.transition_packed, (S, S))
        _check_no_nan_inf(errs, "transition_packed", fx.transition_packed)
        _check_dtype(errs, "transition_packed", fx.transition_packed, "f")
        _check_shape(errs, "P_all_packed", fx.P_all_packed, (fx.C, max_M, max_K))
        _check_no_nan_inf(errs, "P_all_packed", fx.P_all_packed)
        _check_dtype(errs, "P_all_packed", fx.P_all_packed, "f")
        _check_shape(errs, "mu_all_packed", fx.mu_all_packed, (D, max_M, max_K))
        _check_no_nan_inf(errs, "mu_all_packed", fx.mu_all_packed)
        _check_dtype(errs, "mu_all_packed", fx.mu_all_packed, "f")
        _check_shape(errs, "sigma_all_packed", fx.sigma_all_packed, (D, max_M, max_K))
        _check_no_nan_inf(errs, "sigma_all_packed", fx.sigma_all_packed)
        _check_dtype(errs, "sigma_all_packed", fx.sigma_all_packed, "f")

        # Reuse LSIMParams's own, already-verified validation contract on the
        # packed parameters, rather than re-deriving an equivalent set of
        # checks by hand -- only attempted once every shape/dtype check above
        # has passed, since from_packed itself assumes well-formed inputs.
        if not errs:
            from chmm_lsim.params import LSIMParams
            from chmm_lsim.structure import LSIMStructure

            packed_structure = LSIMStructure(
                tuple(int(x) for x in fx.channel_num_states),
                tuple(int(x) for x in fx.dim_observation),
                tuple(int(x) for x in fx.num_gmm_component),
            )
            packed_params = LSIMParams.from_packed(
                packed_structure,
                pi_0=fx.pi_0_packed,
                transition=fx.transition_packed,
                coupling=fx.coupling_packed,
                P_all=fx.P_all_packed,
                mu_all=fx.mu_all_packed,
                sigma_all=fx.sigma_all_packed,
                sigma_diag=bool(fx.sigma_diag),
            )
            try:
                packed_params.validate()
            except Exception as exc:  # chmm_lsim.validate.ValidationError, not imported here on purpose
                errs.append(f"packed parameters failed LSIMParams.validate(): {exc}")

    except Exception as exc:  # noqa: BLE001 -- see validate_posterior_fixture
        errs.append(f"validation raised {type(exc).__name__}: {exc}")

    if errs:
        raise FixtureValidationError("p6_fixture: " + "; ".join(errs))


def load_p6_fixture(fixtures_dir: Path, validate: bool = True) -> P6Fixture:
    npz_path = Path(fixtures_dir) / "p6_fixture.npz"
    d = np.load(npz_path, allow_pickle=False)

    major = str(d["schema_version"]).split(".")[0]
    if major != SCHEMA_MAJOR_SUPPORTED:
        raise SchemaVersionError(
            f"p6_fixture: schema_version {str(d['schema_version'])!r} has major "
            f"version {major!r}, this loader supports {SCHEMA_MAJOR_SUPPORTED!r} only"
        )

    fx = P6Fixture(
        schema_version=str(d["schema_version"]),
        matlab_release=str(d["matlab_release"]),
        C=int(d["C"]),
        channel_num_states=d["channel_num_states"],
        dim_observation=d["dim_observation"],
        num_gmm_component=d["num_gmm_component"],
        T=int(d["T"]),
        sigma_diag=bool(int(d["sigma_diag"])),
        Y=d["Y"],
        joint_pi0=d["joint_pi0"],
        joint_transition=d["joint_transition"],
        joint_P_all=d["joint_P_all"],
        joint_mu_all=d["joint_mu_all"],
        joint_sigma_all=d["joint_sigma_all"],
        cond_trans_1=d["cond_trans_1"],
        cond_trans_2=d["cond_trans_2"],
        P_star_model=float(d["P_star_model"]),
        X_star=d["X_star"],
        log_joint_prob_viterbi_path=float(d["log_joint_prob_viterbi_path"]),
        X_alt=d["X_alt"],
        log_joint_prob_alt_path=float(d["log_joint_prob_alt_path"]),
        pi_0_packed=d["pi_0_packed"],
        coupling_packed=d["coupling_packed"],
        transition_packed=d["transition_packed"],
        P_all_packed=d["P_all_packed"],
        mu_all_packed=d["mu_all_packed"],
        sigma_all_packed=d["sigma_all_packed"],
    )
    if validate:
        validate_p6_fixture(fx)
    return fx


P10A_SCENARIO_IDS = ("diag", "full", "missing", "unequal", "uneven", "fast")

# SCHEMA.md declares every P10a array field as exactly float64 or int64 (no
# float32/int32 variant is ever valid) -- mirrors tools/convert_p10a_
# scheduling_fixtures.py's own _INT_ARRAY_FIELDS/_FLOAT_FIELDS sets, which is
# what actually produces these dtypes on disk, so the two must stay in sync.
P10A_INT64_FIELDS = (
    "channel_num_states", "dim_observation", "num_gmm_component", "length_observation",
    "batch_offsets", "epoch_permutation",
    "updates_epoch_idx", "updates_minibatch_idx", "updates_n_trials", "updates_T_all",
    "updates_n_transitions", "updates_trial_indices",
)
P10A_FLOAT64_FIELDS = (
    "Y",
    "raw_init_pi_0", "raw_init_transition", "raw_init_coupling", "raw_init_P_all", "raw_init_mu_all", "raw_init_sigma_all",
    "returned_best_pi_0", "returned_best_transition", "returned_best_coupling",
    "returned_best_P_all", "returned_best_mu_all", "returned_best_sigma_all", "returned_best_pi_steady",
    "updates_pretrial_loglik",
    "updates_pi_0_in", "updates_pi_0_out", "updates_transition_in", "updates_transition_out",
    "updates_coupling_in", "updates_coupling_out", "updates_P_all_in", "updates_P_all_out",
    "updates_mu_all_in", "updates_mu_all_out", "updates_sigma_all_in", "updates_sigma_all_out",
    "epoch_pi_0_final", "epoch_transition_final", "epoch_coupling_final", "epoch_P_all_final",
    "epoch_mu_all_final", "epoch_sigma_all_final",
    "epoch_mixed_loglik", "epoch_coherent_loglik", "epoch_legacy_mixed_pi_steady", "epoch_coherent_pi_steady",
    "log_likelyhood_trace", "AIC_trace", "BIC_trace",
)


@dataclass
class P10aSchedulingFixture:
    """A single, bespoke fixture (SCHEMA.md's "P10a scheduling fixtures"
    section) -- not manifest-driven, one file per scenario in
    `P10A_SCENARIO_IDS`, produced by `tools/matlab/
    export_p10a_scheduling_fixtures.m` running the audited, drift-settled
    `em_lsim_par.m` (see docs/PYTHON_PORT_PLAN.md's P10a section) with
    `mini_batch > 1` for a few epochs, via its own P10a instrumentation
    hooks (`extra.minibatch_hook`/`extra.epoch_hook`).

    This is P10a's own deliverable: schedule/parity *ground truth*, nothing
    more. No Python mini-batch training exists yet to consume it (P10b,
    explicitly out of scope in P10a).

    **Five distinct parameter snapshots, never to be confused with each
    other** (a correction round found the first version of this fixture
    conflated several of these):

    - `raw_init_*`: the caller-provided model, exactly as given to
      `em_lsim_par.m`'s `init_model` argument, before ANY smoothing.
    - `updates_*_in[0]`: the model actually used by the FIRST E-step --
      after `em_lsim_par.m`'s own `pi_0` smoothing (`+0.0001`, renormalized,
      unless `left2right`). Confirmed to differ measurably from
      `raw_init_pi_0` (typically `~1e-5`-`~1e-4` per-entry, scenario-
      dependent) -- see `raw_vs_effective_pi_0_max_abs_diff`.
    - `updates_*_in[u]` / `updates_*_out[u]`: every update's own pre-/post-
      update snapshot, for every `u` in `0 .. n_updates-1`
      (`n_updates = max_itration * mini_batch`).
    - `epoch_*_final[itr]`: the single coherent parameter snapshot at the
      END of epoch `itr` (== `updates_*_out[itr*mini_batch - 1]` by
      construction -- the last update of that epoch).
    - `returned_best_*`: `em_lsim_par.m`'s own actual return value -- the
      best-by-legacy-mixed-metric snapshot (tracked via its own internal
      `max_AIC`/best-iterate bookkeeping, despite the variable's name
      actually tracking log-likelihood, a pre-existing MATLAB naming quirk
      not introduced by this fixture). Equals `epoch_*_final[returned_best_
      epoch_idx - 1]` by construction -- confirmed as a cross-check both at
      export time and by this loader's own validator, not merely asserted.

    `updates_trial_indices[u, :updates_n_trials[u]]` is the exact, ORDERED
    subsequence of trial indices (1-based) that update `u` processed --
    matching `epoch_permutation[itr]` sliced by `batch_offsets[itr]`
    exactly (independently re-checked by `validate_p10a_scheduling_
    fixture`, not merely trusted from the exporter/converter). Padding past
    `updates_n_trials[u]` is `0` for `updates_trial_indices` (never a valid
    1-based index) and `NaN` for `updates_pretrial_loglik`.

    `epoch_mixed_loglik`/`epoch_legacy_mixed_pi_steady` are the §2.6
    "batch-training diagnostic" (mixed parameter snapshots across a mini-
    batch epoch) -- **legacy, not coherent**. `epoch_coherent_loglik`/
    `epoch_coherent_pi_steady` are evaluated under each epoch's own single
    `epoch_*_final` snapshot instead (a full forward-BACKWARD pass, so
    `epoch_coherent_pi_steady` is genuinely gamma-based, not merely
    loglik-based) -- these differ meaningfully early in training and
    converge as parameters stabilize (confirmed directly during the P10a
    audit)."""

    schema_version: str
    matlab_release: str
    scenario_id: str
    variant: str
    C: int
    channel_num_states: np.ndarray
    dim_observation: np.ndarray
    num_gmm_component: np.ndarray
    sigma_diag: bool
    num_trials: int
    length_observation: np.ndarray
    Y: np.ndarray
    data_seed: int
    schedule_seed: int
    max_itration: int
    mini_batch: int
    max_batch_size: int
    n_updates: int

    raw_init_pi_0: np.ndarray
    raw_init_transition: np.ndarray
    raw_init_coupling: np.ndarray
    raw_init_P_all: np.ndarray
    raw_init_mu_all: np.ndarray
    raw_init_sigma_all: np.ndarray

    updates_epoch_idx: np.ndarray
    updates_minibatch_idx: np.ndarray
    updates_n_trials: np.ndarray
    updates_T_all: np.ndarray
    updates_n_transitions: np.ndarray
    updates_trial_indices: np.ndarray
    updates_pretrial_loglik: np.ndarray
    updates_pi_0_in: np.ndarray
    updates_pi_0_out: np.ndarray
    updates_transition_in: np.ndarray
    updates_transition_out: np.ndarray
    updates_coupling_in: np.ndarray
    updates_coupling_out: np.ndarray
    updates_P_all_in: np.ndarray
    updates_P_all_out: np.ndarray
    updates_mu_all_in: np.ndarray
    updates_mu_all_out: np.ndarray
    updates_sigma_all_in: np.ndarray
    updates_sigma_all_out: np.ndarray

    epoch_permutation: np.ndarray
    batch_offsets: np.ndarray

    epoch_pi_0_final: np.ndarray
    epoch_transition_final: np.ndarray
    epoch_coupling_final: np.ndarray
    epoch_P_all_final: np.ndarray
    epoch_mu_all_final: np.ndarray
    epoch_sigma_all_final: np.ndarray
    epoch_mixed_loglik: np.ndarray
    epoch_coherent_loglik: np.ndarray
    epoch_legacy_mixed_pi_steady: np.ndarray
    epoch_coherent_pi_steady: np.ndarray

    returned_best_epoch_idx: int
    returned_best_pi_0: np.ndarray
    returned_best_transition: np.ndarray
    returned_best_coupling: np.ndarray
    returned_best_P_all: np.ndarray
    returned_best_mu_all: np.ndarray
    returned_best_sigma_all: np.ndarray
    returned_best_pi_steady: np.ndarray

    log_likelyhood_trace: np.ndarray
    AIC_trace: np.ndarray
    BIC_trace: np.ndarray

    @property
    def D(self) -> int:
        return int(self.dim_observation.sum())

    @property
    def S(self) -> int:
        return int(self.channel_num_states.sum())

    def Y_trials(self) -> list[np.ndarray]:
        """Splits the concatenated `Y` back into one `(D, T_r)` array per
        trial, in the same order `length_observation` lists them."""
        boundaries = np.concatenate([[0], np.cumsum(self.length_observation)])
        return [self.Y[:, boundaries[i] : boundaries[i + 1]] for i in range(self.num_trials)]

    def update_index(self, epoch_1based: int, minibatch_1based: int) -> int:
        """Row index into every `updates_*` array for a given (epoch,
        minibatch), both 1-based (MATLAB's own convention, matching
        `updates_epoch_idx`/`updates_minibatch_idx`'s own values)."""
        return (epoch_1based - 1) * self.mini_batch + (minibatch_1based - 1)


def _check_finite(errs: list[str], name: str, arr: np.ndarray, *, allow_nan: bool = False) -> None:
    """`Inf`/`-Inf` are never acceptable. `NaN` is acceptable only where
    `allow_nan=True` is passed explicitly (`Y` itself, and the declared
    padding regions of ragged per-update arrays) -- every other numeric
    field, including packed-array padding that is merely *unused* rather
    than meaningful, must be finite. A prior version of this validator
    missed exactly this for `Y` (accepted `Inf`), `AIC_trace` (accepted
    `NaN`), and unused `P_all` padding (accepted `NaN`)."""
    if np.isinf(arr).any():
        errs.append(f"{name}: contains Inf/-Inf")
    if not allow_nan and np.isnan(arr).any():
        errs.append(f"{name}: contains NaN")


def _check_row_permutation(errs: list[str], name: str, row: np.ndarray, num_trials: int) -> None:
    if sorted(row.tolist()) != list(range(1, num_trials + 1)):
        errs.append(f"{name}: not a permutation of 1..{num_trials}: {sorted(row.tolist())}")


def validate_p10a_scheduling_fixture(fx: "P10aSchedulingFixture") -> None:
    """Exhaustive validation, matching the standard every other fixture
    family in this file is held to: exact shape, exact dtype, scalar
    positivity/range, NaN/Inf rules (including padded regions), metadata
    consistency, packed-array padding (not only active values), schedule
    ordering/boundaries, parameter stochasticity (every packed model, not a
    sample), and cross-field diagnostic identities."""
    errs: list[str] = []
    try:
        if not isinstance(fx.sigma_diag, (bool, np.bool_)):
            # Deliberately NOT coerced (`bool(2)` would silently accept an
            # invalid value) -- a genuinely non-bool sigma_diag is a
            # corrupted fixture, reported as such, not laundered into True.
            errs.append(f"sigma_diag: expected a bool, got {fx.sigma_diag!r} (type {type(fx.sigma_diag).__name__})")
        if fx.variant not in ("sed", "fast"):
            errs.append(f"variant: expected 'sed' or 'fast', got {fx.variant!r}")
        if fx.scenario_id not in P10A_SCENARIO_IDS:
            errs.append(f"scenario_id: {fx.scenario_id!r} not in {P10A_SCENARIO_IDS}")
        for name, val in (("C", fx.C), ("num_trials", fx.num_trials), ("max_itration", fx.max_itration), ("mini_batch", fx.mini_batch), ("max_batch_size", fx.max_batch_size)):
            if val < 1:
                errs.append(f"{name}: expected >= 1, got {val}")
        if fx.n_updates != fx.max_itration * fx.mini_batch:
            errs.append(f"n_updates={fx.n_updates} != max_itration*mini_batch={fx.max_itration * fx.mini_batch}")
        if not (1 <= fx.max_batch_size <= fx.num_trials):
            errs.append(f"max_batch_size={fx.max_batch_size}: expected in [1, num_trials={fx.num_trials}]")
        if not (1 <= fx.returned_best_epoch_idx <= fx.max_itration):
            errs.append(f"returned_best_epoch_idx={fx.returned_best_epoch_idx}: expected in [1, max_itration={fx.max_itration}]")

        C = fx.C
        cns_ok = _check_shape(errs, "channel_num_states", fx.channel_num_states, (C,))
        do_ok = _check_shape(errs, "dim_observation", fx.dim_observation, (C,))
        ngc_ok = _check_shape(errs, "num_gmm_component", fx.num_gmm_component, (C,))
        _check_shape(errs, "length_observation", fx.length_observation, (fx.num_trials,))
        if cns_ok:
            cns_ok = _check_dtype(errs, "channel_num_states", fx.channel_num_states, "i")
            _check_finite(errs, "channel_num_states", fx.channel_num_states)
            _check_positive(errs, "channel_num_states", fx.channel_num_states)
        if do_ok:
            do_ok = _check_dtype(errs, "dim_observation", fx.dim_observation, "i")
            _check_finite(errs, "dim_observation", fx.dim_observation)
            _check_positive(errs, "dim_observation", fx.dim_observation)
        if ngc_ok:
            ngc_ok = _check_dtype(errs, "num_gmm_component", fx.num_gmm_component, "i")
            _check_finite(errs, "num_gmm_component", fx.num_gmm_component)
            _check_positive(errs, "num_gmm_component", fx.num_gmm_component)
        _check_dtype(errs, "length_observation", fx.length_observation, "i")
        _check_positive(errs, "length_observation", fx.length_observation)

        if cns_ok and do_ok and ngc_ok:
            _validate_p10a_body(fx, errs)
    except Exception as exc:  # noqa: BLE001 -- same discipline as every other validate_* here: never let an unrelated exception escape as something other than FixtureValidationError
        errs.append(f"unexpected error during validation: {exc}")

    if errs:
        raise FixtureValidationError("p10a_scheduling fixture: " + "; ".join(errs))


def _validate_p10a_body(fx: "P10aSchedulingFixture", errs: list[str]) -> None:
    """The bulk of validate_p10a_scheduling_fixture, factored out so the
    caller can gate it behind the per-channel shape checks (channel_num_
    states/dim_observation/num_gmm_component) without a control-flow
    exception -- S/D/max_M/max_K/max_d below are only meaningful once those
    three arrays are known to have the right shape."""
    # ---- exact dtype (not merely dtype *kind*, which _check_dtype below
    # checks): SCHEMA.md declares float64/int64 specifically, so a
    # float32/int32 fixture must be rejected even though its values could
    # otherwise pass every other check here.
    for name in P10A_INT64_FIELDS:
        _check_exact_dtype(errs, name, getattr(fx, name), np.int64)
    for name in P10A_FLOAT64_FIELDS:
        _check_exact_dtype(errs, name, getattr(fx, name), np.float64)

    C = fx.C
    S, D = fx.S, fx.D
    max_M = int(fx.channel_num_states.max())
    max_K = int(fx.num_gmm_component.max())
    max_d = int(fx.dim_observation.max())
    sigma_shape = (D, max_M, max_K) if fx.sigma_diag else (D, max_d, max_M, max_K)

    if fx.Y.shape != (D, int(fx.length_observation.sum())):
        errs.append(f"Y: shape {fx.Y.shape} != expected {(D, int(fx.length_observation.sum()))}")
    elif _check_dtype(errs, "Y", fx.Y, "f"):
        _check_finite(errs, "Y", fx.Y, allow_nan=True)  # NaN = missing observation, Inf never valid

    n_updates, max_batch_size = fx.n_updates, fx.max_batch_size
    max_itration, mini_batch, num_trials = fx.max_itration, fx.mini_batch, fx.num_trials

    # ---- raw_init_* / returned_best_* : shape, finiteness, LSIMParams --
    packed_groups: list[tuple[str, dict[str, np.ndarray]]] = [
        ("raw_init", {"pi_0": fx.raw_init_pi_0, "transition": fx.raw_init_transition, "coupling": fx.raw_init_coupling, "P_all": fx.raw_init_P_all, "mu_all": fx.raw_init_mu_all, "sigma_all": fx.raw_init_sigma_all}),
        ("returned_best", {"pi_0": fx.returned_best_pi_0, "transition": fx.returned_best_transition, "coupling": fx.returned_best_coupling, "P_all": fx.returned_best_P_all, "mu_all": fx.returned_best_mu_all, "sigma_all": fx.returned_best_sigma_all}),
    ]
    expected_shapes = {"pi_0": (S,), "transition": (S, S), "coupling": (C, C), "P_all": (C, max_M, max_K), "mu_all": (D, max_M, max_K), "sigma_all": sigma_shape}
    for group_name, group in packed_groups:
        for field, arr in group.items():
            ok = _check_shape(errs, f"{group_name}_{field}", arr, expected_shapes[field])
            if ok:
                ok = _check_dtype(errs, f"{group_name}_{field}", arr, "f")
            if ok:
                _check_finite(errs, f"{group_name}_{field}", arr)
    if _check_shape(errs, "returned_best_pi_steady", fx.returned_best_pi_steady, (S,)) and _check_dtype(errs, "returned_best_pi_steady", fx.returned_best_pi_steady, "f"):
        _check_finite(errs, "returned_best_pi_steady", fx.returned_best_pi_steady)

    from chmm_lsim.params import LSIMParams
    from chmm_lsim.structure import LSIMStructure

    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )

    def _validate_packed(label: str, pi_0, transition, coupling, P_all, mu_all, sigma_all) -> None:
        try:
            params = LSIMParams.from_packed(
                structure, pi_0=pi_0, transition=transition, coupling=coupling,
                P_all=P_all, mu_all=mu_all, sigma_all=sigma_all, sigma_diag=bool(fx.sigma_diag),
            )
            params.validate()
        except Exception as exc:  # noqa: BLE001
            errs.append(f"packed parameters ({label}) failed LSIMParams.validate(): {exc}")

    _validate_packed("raw_init", fx.raw_init_pi_0, fx.raw_init_transition, fx.raw_init_coupling, fx.raw_init_P_all, fx.raw_init_mu_all, fx.raw_init_sigma_all)
    _validate_packed("returned_best", fx.returned_best_pi_0, fx.returned_best_transition, fx.returned_best_coupling, fx.returned_best_P_all, fx.returned_best_mu_all, fx.returned_best_sigma_all)

    # ---- updates_*: shape, finiteness (padding-aware), stochasticity
    update_shapes = {
        "epoch_idx": (n_updates,), "minibatch_idx": (n_updates,), "n_trials": (n_updates,),
        "T_all": (n_updates,), "n_transitions": (n_updates,),
        "trial_indices": (n_updates, max_batch_size), "pretrial_loglik": (n_updates, max_batch_size),
        "pi_0_in": (n_updates, S), "pi_0_out": (n_updates, S),
        "transition_in": (n_updates, S, S), "transition_out": (n_updates, S, S),
        "coupling_in": (n_updates, C, C), "coupling_out": (n_updates, C, C),
        "P_all_in": (n_updates, C, max_M, max_K), "P_all_out": (n_updates, C, max_M, max_K),
        "mu_all_in": (n_updates, D, max_M, max_K), "mu_all_out": (n_updates, D, max_M, max_K),
        "sigma_all_in": (n_updates,) + sigma_shape, "sigma_all_out": (n_updates,) + sigma_shape,
    }
    _update_int_fields = {"epoch_idx", "minibatch_idx", "n_trials", "T_all", "n_transitions", "trial_indices"}
    shapes_ok = True
    for field, shape in update_shapes.items():
        ok = _check_shape(errs, f"updates_{field}", getattr(fx, f"updates_{field}"), shape)
        if ok:
            kind = "i" if field in _update_int_fields else "f"
            ok = _check_dtype(errs, f"updates_{field}", getattr(fx, f"updates_{field}"), kind)
        shapes_ok &= ok

    if shapes_ok:
        for field in ("epoch_idx", "minibatch_idx", "n_trials", "T_all", "n_transitions"):
            _check_finite(errs, f"updates_{field}", getattr(fx, f"updates_{field}"))
        if (fx.updates_epoch_idx < 1).any() or (fx.updates_epoch_idx > max_itration).any():
            errs.append("updates_epoch_idx: out of range [1, max_itration]")
        if (fx.updates_minibatch_idx < 1).any() or (fx.updates_minibatch_idx > mini_batch).any():
            errs.append("updates_minibatch_idx: out of range [1, mini_batch]")
        if (fx.updates_n_trials < 1).any() or (fx.updates_n_trials > max_batch_size).any():
            errs.append("updates_n_trials: out of range [1, max_batch_size]")

        # The flattened update axis (n_updates = max_itration*mini_batch) has
        # an exact, fully-determined layout -- update_idx = (itr-1)*mini_
        # batch + mb (P10aSchedulingFixture.update_index's own formula) --
        # not merely values in range.
        expected_epoch_idx = np.repeat(np.arange(1, max_itration + 1), mini_batch)
        if not np.array_equal(fx.updates_epoch_idx, expected_epoch_idx):
            errs.append("updates_epoch_idx != repeat(1..max_itration, mini_batch)")
        expected_minibatch_idx = np.tile(np.arange(1, mini_batch + 1), max_itration)
        if not np.array_equal(fx.updates_minibatch_idx, expected_minibatch_idx):
            errs.append("updates_minibatch_idx != tile(1..mini_batch, max_itration)")

        # An update's own n_transitions == T_all - n_trials exactly
        # (MATLAB's own L_T_1_batch identity, sum(T_i-1) over the
        # update's own trials) -- confirmed, not assumed.
        if not np.array_equal(fx.updates_n_transitions, fx.updates_T_all - fx.updates_n_trials):
            errs.append("updates_n_transitions != updates_T_all - updates_n_trials for at least one update")

        for u in range(n_updates):
            n_tr = int(fx.updates_n_trials[u])
            active_idx = fx.updates_trial_indices[u, :n_tr]
            pad_idx = fx.updates_trial_indices[u, n_tr:]
            active_ll = fx.updates_pretrial_loglik[u, :n_tr]
            pad_ll = fx.updates_pretrial_loglik[u, n_tr:]
            if (active_idx < 1).any() or (active_idx > num_trials).any() or len(set(active_idx.tolist())) != n_tr:
                errs.append(f"updates_trial_indices[{u}]: active region {active_idx.tolist()} is not {n_tr} distinct trial indices in [1,{num_trials}]")
            if np.isnan(active_idx.astype(float)).any() or np.isinf(active_idx.astype(float)).any():
                errs.append(f"updates_trial_indices[{u}]: active region contains NaN/Inf")
            if (pad_idx != 0).any():
                errs.append(f"updates_trial_indices[{u}]: padding region is not all-zero: {pad_idx.tolist()}")
            _check_finite(errs, f"updates_pretrial_loglik[{u}] (active)", active_ll)
            if not np.isnan(pad_ll).all():
                errs.append(f"updates_pretrial_loglik[{u}]: padding region is not all-NaN: {pad_ll.tolist()}")

        for field in ("pi_0_in", "pi_0_out", "transition_in", "transition_out", "coupling_in", "coupling_out", "P_all_in", "P_all_out", "mu_all_in", "mu_all_out", "sigma_all_in", "sigma_all_out"):
            _check_finite(errs, f"updates_{field}", getattr(fx, f"updates_{field}"))

        for u in range(n_updates):
            _validate_packed(f"updates[{u}].in", fx.updates_pi_0_in[u], fx.updates_transition_in[u], fx.updates_coupling_in[u], fx.updates_P_all_in[u], fx.updates_mu_all_in[u], fx.updates_sigma_all_in[u])
            _validate_packed(f"updates[{u}].out", fx.updates_pi_0_out[u], fx.updates_transition_out[u], fx.updates_coupling_out[u], fx.updates_P_all_out[u], fx.updates_mu_all_out[u], fx.updates_sigma_all_out[u])

        # ---- complete parameter-chain continuity, all six packed tensors,
        # not merely pi_0: each update's params_in must equal the PRECEDING
        # update's own params_out exactly (they are the same in-memory
        # struct on the MATLAB side, captured immediately before/after one
        # minibatch commit), and update 0's params_in must equal raw_init_*
        # exactly for every tensor except pi_0, which em_lsim_par.m smooths
        # once before the first E-step (checked independently below).
        _PARAM_FIELDS = ("pi_0", "transition", "coupling", "P_all", "mu_all", "sigma_all")
        for u in range(1, n_updates):
            for pf in _PARAM_FIELDS:
                prev_out = getattr(fx, f"updates_{pf}_out")[u - 1]
                cur_in = getattr(fx, f"updates_{pf}_in")[u]
                if not np.array_equal(prev_out, cur_in):
                    errs.append(f"updates_{pf}_in[{u}] != updates_{pf}_out[{u - 1}]")
        for pf in ("transition", "coupling", "P_all", "mu_all", "sigma_all"):
            raw = getattr(fx, f"raw_init_{pf}")
            in0 = getattr(fx, f"updates_{pf}_in")[0]
            if not np.array_equal(raw, in0):
                errs.append(f"updates_{pf}_in[0] != raw_init_{pf}")

        # pi_0 is smoothed once (em_lsim.m:159-164's 1e-4 blockwise
        # smoothing, left2right=0 for every current P10a scenario) before
        # update 0's own E-step -- recompute it independently via the same
        # helper src/lsim/em.py's own EM training uses, rather than
        # reimplementing the formula here, and require tight agreement with
        # what update 0 actually used.
        if fx.raw_init_pi_0.shape == (S,) and fx.updates_pi_0_in.shape == (n_updates, S):
            from chmm_lsim.em import apply_left2right_pi0_smoothing

            recomputed_pi0_in0 = apply_left2right_pi0_smoothing(fx.raw_init_pi_0, structure, left2right=False)
            if not np.allclose(recomputed_pi0_in0, fx.updates_pi_0_in[0], rtol=0, atol=1e-12):
                diff = float(np.max(np.abs(recomputed_pi0_in0 - fx.updates_pi_0_in[0])))
                errs.append(f"updates_pi_0_in[0] != independently recomputed effective pi_0 (1e-4 blockwise smoothing of raw_init_pi_0, max abs diff {diff:.3e})")

    # ---- epoch_permutation / batch_offsets: ordering and boundaries
    perm_ok = _check_shape(errs, "epoch_permutation", fx.epoch_permutation, (max_itration, num_trials))
    if perm_ok:
        perm_ok = _check_dtype(errs, "epoch_permutation", fx.epoch_permutation, "i")
    offsets_ok = _check_shape(errs, "batch_offsets", fx.batch_offsets, (max_itration, mini_batch + 1))
    if offsets_ok:
        offsets_ok = _check_dtype(errs, "batch_offsets", fx.batch_offsets, "i")
    if perm_ok:
        _check_finite(errs, "epoch_permutation", fx.epoch_permutation)
        for itr in range(max_itration):
            _check_row_permutation(errs, f"epoch_permutation[{itr}]", fx.epoch_permutation[itr], num_trials)
    if offsets_ok:
        _check_finite(errs, "batch_offsets", fx.batch_offsets)
        for itr in range(max_itration):
            row = fx.batch_offsets[itr]
            if row[0] != 0 or row[-1] != num_trials:
                errs.append(f"batch_offsets[{itr}]: {row.tolist()} does not start at 0 / end at num_trials={num_trials}")
            if np.any(np.diff(row) <= 0):
                errs.append(f"batch_offsets[{itr}]: not strictly increasing (an empty minibatch): {row.tolist()}")
    if shapes_ok and perm_ok and offsets_ok:
        # Cross-check: updates_trial_indices must match the
        # epoch_permutation slice batch_offsets implies -- the
        # exporter/converter already assert this; the loader
        # re-derives it independently rather than trusting them.
        for u in range(n_updates):
            itr0 = int(fx.updates_epoch_idx[u]) - 1
            mb0 = int(fx.updates_minibatch_idx[u]) - 1
            n_tr = int(fx.updates_n_trials[u])
            lo, hi = int(fx.batch_offsets[itr0, mb0]), int(fx.batch_offsets[itr0, mb0 + 1])
            expected = fx.epoch_permutation[itr0, lo:hi]
            got = fx.updates_trial_indices[u, :n_tr]
            if expected.shape != got.shape or not np.array_equal(expected, got):
                errs.append(f"updates_trial_indices[{u}] {got.tolist()} != epoch_permutation slice {expected.tolist()} (epoch {itr0+1}, minibatch {mb0+1})")

            # updates_n_trials[u] must equal the batch_offsets boundary
            # difference exactly, not merely be internally consistent with
            # the trial-index count above.
            expected_n_trials = hi - lo
            if n_tr != expected_n_trials:
                errs.append(f"updates_n_trials[{u}]={n_tr} != batch_offsets[{itr0},{mb0+1}]-batch_offsets[{itr0},{mb0}]={expected_n_trials}")

            # updates_T_all[u] must equal the sum of length_observation over
            # that update's own ORDERED trial indices (0-based into
            # length_observation), independently re-derived rather than
            # trusting the exporter's own bookkeeping.
            active_trial_idx0 = fx.updates_trial_indices[u, :n_tr] - 1
            expected_T_all = int(fx.length_observation[active_trial_idx0].sum())
            if int(fx.updates_T_all[u]) != expected_T_all:
                errs.append(f"updates_T_all[{u}]={int(fx.updates_T_all[u])} != sum(length_observation[active trial indices])={expected_T_all}")

    # ---- epoch_*_final: shape, finiteness, LSIMParams -------------
    epoch_shapes = {
        "pi_0_final": (max_itration, S), "transition_final": (max_itration, S, S), "coupling_final": (max_itration, C, C),
        "P_all_final": (max_itration, C, max_M, max_K), "mu_all_final": (max_itration, D, max_M, max_K),
        "sigma_all_final": (max_itration,) + sigma_shape,
        "mixed_loglik": (max_itration, num_trials), "coherent_loglik": (max_itration, num_trials),
        "legacy_mixed_pi_steady": (max_itration, S), "coherent_pi_steady": (max_itration, S),
    }
    epoch_shapes_ok = True
    for field, shape in epoch_shapes.items():
        ok = _check_shape(errs, f"epoch_{field}", getattr(fx, f"epoch_{field}"), shape)
        if ok:
            ok = _check_dtype(errs, f"epoch_{field}", getattr(fx, f"epoch_{field}"), "f")
        epoch_shapes_ok &= ok
    if epoch_shapes_ok:
        for field in epoch_shapes:
            _check_finite(errs, f"epoch_{field}", getattr(fx, f"epoch_{field}"))
        for itr in range(max_itration):
            _validate_packed(f"epoch[{itr}].final", fx.epoch_pi_0_final[itr], fx.epoch_transition_final[itr], fx.epoch_coupling_final[itr], fx.epoch_P_all_final[itr], fx.epoch_mu_all_final[itr], fx.epoch_sigma_all_final[itr])

    # ---- trace arrays: shape + strict finiteness -------------------
    for name in ("log_likelyhood_trace", "AIC_trace", "BIC_trace"):
        arr = getattr(fx, name)
        if _check_shape(errs, name, arr, (max_itration,)) and _check_dtype(errs, name, arr, "f"):
            _check_finite(errs, name, arr)

    # ---- cross-field diagnostic identities -------------------------
    if shapes_ok and epoch_shapes_ok:
        np.testing.assert_allclose(fx.epoch_mixed_loglik.sum(axis=1), fx.log_likelyhood_trace, rtol=1e-9, atol=1e-9, err_msg="epoch_mixed_loglik row sums != log_likelyhood_trace")

        # per-epoch sum of ACTIVE updates_pretrial_loglik entries must
        # also equal log_likelyhood_trace (a second, independent
        # identity, using the ragged per-update breakdown rather than
        # the already-assembled per-epoch row).
        per_epoch_from_updates = np.zeros(max_itration)
        for u in range(n_updates):
            itr0 = int(fx.updates_epoch_idx[u]) - 1
            n_tr = int(fx.updates_n_trials[u])
            per_epoch_from_updates[itr0] += float(np.sum(fx.updates_pretrial_loglik[u, :n_tr]))
        np.testing.assert_allclose(per_epoch_from_updates, fx.log_likelyhood_trace, rtol=1e-9, atol=1e-9, err_msg="sum of active updates_pretrial_loglik per epoch != log_likelyhood_trace")

        # epoch_*_final[itr] must equal updates_*_out[last update of that
        # epoch] exactly, for EVERY packed tensor (not only pi_0) -- the
        # epoch's final snapshot IS the last minibatch's own post-update
        # commit.
        _PARAM_FIELDS = ("pi_0", "transition", "coupling", "P_all", "mu_all", "sigma_all")
        for itr in range(max_itration):
            last_u = itr * mini_batch + (mini_batch - 1)
            for pf in _PARAM_FIELDS:
                final_arr = getattr(fx, f"epoch_{pf}_final")[itr]
                out_arr = getattr(fx, f"updates_{pf}_out")[last_u]
                if not np.array_equal(final_arr, out_arr):
                    errs.append(f"epoch_{pf}_final[{itr}] != updates_{pf}_out[{last_u}] (last update of that epoch)")

        # returned_best_* must equal epoch_*_final at the recorded best
        # epoch, exactly, for EVERY packed tensor (not only pi_0), plus
        # pi_steady.
        best_i = fx.returned_best_epoch_idx - 1
        for pf in _PARAM_FIELDS:
            best_arr = getattr(fx, f"returned_best_{pf}")
            final_arr = getattr(fx, f"epoch_{pf}_final")[best_i]
            if not np.array_equal(best_arr, final_arr):
                errs.append(f"returned_best_{pf} != epoch_{pf}_final[{best_i}] (returned_best_epoch_idx={fx.returned_best_epoch_idx})")
        if not np.allclose(fx.returned_best_pi_steady, fx.epoch_legacy_mixed_pi_steady[best_i], rtol=1e-9, atol=1e-9):
            errs.append(f"returned_best_pi_steady != epoch_legacy_mixed_pi_steady[{best_i}] (returned_best_epoch_idx={fx.returned_best_epoch_idx})")

        # Independently re-derive which epoch em_lsim_par.m selected as
        # best, replicating its own exact (and deliberately unusual)
        # selection rule: iteration 2 unconditionally overwrites iteration
        # 1's snapshot regardless of whether LL improved; only iteration 3+
        # requires strict improvement over the running max. Mirrors this
        # loader's own tools/matlab/export_p10a_scheduling_fixtures.m's
        # independent recomputation, which cross-checks the same thing on
        # the MATLAB side at export time.
        rederived_best_idx = 1
        max_ll = float(fx.log_likelyhood_trace[0])
        for itr1 in range(2, max_itration + 1):
            ll = float(fx.log_likelyhood_trace[itr1 - 1])
            if itr1 > 2:
                if max_ll < ll:
                    max_ll = ll
                    rederived_best_idx = itr1
            else:
                max_ll = ll
                rederived_best_idx = itr1
        if rederived_best_idx != fx.returned_best_epoch_idx:
            errs.append(
                f"returned_best_epoch_idx={fx.returned_best_epoch_idx} != independently re-derived best epoch "
                f"{rederived_best_idx} (em_lsim_par.m's own selection rule: itr<=2 unconditional, itr>2 strict-improvement-only)"
            )

        # epoch_legacy_mixed_pi_steady / epoch_coherent_pi_steady: finite
        # (already checked above), non-negative, and each channel's own
        # block sums to 1 -- every epoch, both variants.
        for pi_steady_name in ("epoch_legacy_mixed_pi_steady", "epoch_coherent_pi_steady"):
            arr = getattr(fx, pi_steady_name)
            if np.any(arr < 0):
                errs.append(f"{pi_steady_name}: contains negative entries")
            for itr in range(max_itration):
                for c in range(C):
                    sl = structure.state_slice(c)
                    block_sum = float(arr[itr, sl].sum())
                    if abs(block_sum - 1.0) > 1e-6:
                        errs.append(f"{pi_steady_name}[{itr}] channel {c} block does not sum to 1 (got {block_sum:.6f})")

        # AIC/BIC identities: recompute each epoch's AIC/BIC from its own
        # log-likelihood, the GLOBAL T_all (sum of length_observation across
        # every trial in the dataset, not a per-minibatch T_all), structure,
        # and sigma_diag, using the same compute_aic_bic em_lsim.m's own
        # formula and P4's own EM training share.
        from chmm_lsim.em import compute_aic_bic

        global_T_all = int(fx.length_observation.sum())
        for itr in range(max_itration):
            expected_aic, expected_bic = compute_aic_bic(structure, bool(fx.sigma_diag), float(fx.log_likelyhood_trace[itr]), global_T_all)
            if not np.isclose(expected_aic, fx.AIC_trace[itr], rtol=1e-9, atol=1e-6):
                errs.append(f"AIC_trace[{itr}]={fx.AIC_trace[itr]} != recomputed compute_aic_bic AIC {expected_aic}")
            if not np.isclose(expected_bic, fx.BIC_trace[itr], rtol=1e-9, atol=1e-6):
                errs.append(f"BIC_trace[{itr}]={fx.BIC_trace[itr]} != recomputed compute_aic_bic BIC {expected_bic}")


def _p10a_int_scalar(d, key: str) -> int:
    """Like `int(d[key])`, but rejects a non-integer-dtype npz scalar
    outright instead of silently laundering it through `int(...)` -- a
    malformed float32 scalar (e.g. `7.0` stored as float rather than the
    schema-declared int64) would otherwise cast cleanly and never surface as
    a problem."""
    arr = d[key]
    if not np.issubdtype(arr.dtype, np.integer):
        raise FixtureValidationError(f"{key}: expected an integer-dtype scalar, got dtype={arr.dtype}")
    return int(arr)


def load_p10a_scheduling_fixture(fixtures_dir: Path, scenario_id: str, validate: bool = True) -> P10aSchedulingFixture:
    if scenario_id not in P10A_SCENARIO_IDS:
        raise ValueError(f"scenario_id={scenario_id!r}, expected one of {P10A_SCENARIO_IDS}")
    npz_path = Path(fixtures_dir) / f"p10a_scheduling__{scenario_id}.npz"
    d = np.load(npz_path, allow_pickle=False)

    major = str(d["schema_version"]).split(".")[0]
    if major != SCHEMA_MAJOR_SUPPORTED:
        raise SchemaVersionError(
            f"p10a_scheduling__{scenario_id}: schema_version {str(d['schema_version'])!r} has major "
            f"version {major!r}, this loader supports {SCHEMA_MAJOR_SUPPORTED!r} only"
        )

    fx = P10aSchedulingFixture(
        schema_version=str(d["schema_version"]),
        matlab_release=str(d["matlab_release"]),
        scenario_id=str(d["scenario_id"]),
        variant=str(d["variant"]),
        C=_p10a_int_scalar(d, "C"),
        channel_num_states=d["channel_num_states"],
        dim_observation=d["dim_observation"],
        num_gmm_component=d["num_gmm_component"],
        sigma_diag=bool(d["sigma_diag"]) if d["sigma_diag"].dtype == np.bool_ else d["sigma_diag"].item(),
        num_trials=_p10a_int_scalar(d, "num_trials"),
        length_observation=d["length_observation"],
        Y=d["Y"],
        data_seed=_p10a_int_scalar(d, "data_seed"),
        schedule_seed=_p10a_int_scalar(d, "schedule_seed"),
        max_itration=_p10a_int_scalar(d, "max_itration"),
        mini_batch=_p10a_int_scalar(d, "mini_batch"),
        max_batch_size=_p10a_int_scalar(d, "max_batch_size"),
        n_updates=_p10a_int_scalar(d, "n_updates"),
        raw_init_pi_0=d["raw_init_pi_0"], raw_init_transition=d["raw_init_transition"], raw_init_coupling=d["raw_init_coupling"],
        raw_init_P_all=d["raw_init_P_all"], raw_init_mu_all=d["raw_init_mu_all"], raw_init_sigma_all=d["raw_init_sigma_all"],
        updates_epoch_idx=d["updates_epoch_idx"], updates_minibatch_idx=d["updates_minibatch_idx"],
        updates_n_trials=d["updates_n_trials"], updates_T_all=d["updates_T_all"], updates_n_transitions=d["updates_n_transitions"],
        updates_trial_indices=d["updates_trial_indices"], updates_pretrial_loglik=d["updates_pretrial_loglik"],
        updates_pi_0_in=d["updates_pi_0_in"], updates_pi_0_out=d["updates_pi_0_out"],
        updates_transition_in=d["updates_transition_in"], updates_transition_out=d["updates_transition_out"],
        updates_coupling_in=d["updates_coupling_in"], updates_coupling_out=d["updates_coupling_out"],
        updates_P_all_in=d["updates_P_all_in"], updates_P_all_out=d["updates_P_all_out"],
        updates_mu_all_in=d["updates_mu_all_in"], updates_mu_all_out=d["updates_mu_all_out"],
        updates_sigma_all_in=d["updates_sigma_all_in"], updates_sigma_all_out=d["updates_sigma_all_out"],
        epoch_permutation=d["epoch_permutation"], batch_offsets=d["batch_offsets"],
        epoch_pi_0_final=d["epoch_pi_0_final"], epoch_transition_final=d["epoch_transition_final"], epoch_coupling_final=d["epoch_coupling_final"],
        epoch_P_all_final=d["epoch_P_all_final"], epoch_mu_all_final=d["epoch_mu_all_final"], epoch_sigma_all_final=d["epoch_sigma_all_final"],
        epoch_mixed_loglik=d["epoch_mixed_loglik"], epoch_coherent_loglik=d["epoch_coherent_loglik"],
        epoch_legacy_mixed_pi_steady=d["epoch_legacy_mixed_pi_steady"], epoch_coherent_pi_steady=d["epoch_coherent_pi_steady"],
        returned_best_epoch_idx=_p10a_int_scalar(d, "returned_best_epoch_idx"),
        returned_best_pi_0=d["returned_best_pi_0"], returned_best_transition=d["returned_best_transition"], returned_best_coupling=d["returned_best_coupling"],
        returned_best_P_all=d["returned_best_P_all"], returned_best_mu_all=d["returned_best_mu_all"], returned_best_sigma_all=d["returned_best_sigma_all"],
        returned_best_pi_steady=d["returned_best_pi_steady"],
        log_likelyhood_trace=d["log_likelyhood_trace"], AIC_trace=d["AIC_trace"], BIC_trace=d["BIC_trace"],
    )
    if validate:
        # Bind the file's own internal metadata to the scenario_id the
        # caller actually requested. Without this, requesting "diag" but
        # being handed a file whose contents are internally another valid
        # scenario (e.g. a copy/rename mistake landing "fast"'s bytes at
        # p10a_scheduling__diag.npz) passes every other check silently --
        # membership in P10A_SCENARIO_IDS alone doesn't catch a mismatch
        # between filename and payload. Known mappings (current 6
        # scenarios only): variant is 'fast' iff scenario_id=='fast',
        # 'sed' otherwise; sigma_diag is False iff scenario_id=='full',
        # True otherwise.
        if fx.scenario_id != scenario_id:
            raise FixtureValidationError(
                f"p10a_scheduling__{scenario_id}.npz: internal scenario_id={fx.scenario_id!r} "
                f"does not match the requested scenario_id={scenario_id!r}"
            )
        expected_variant = "fast" if scenario_id == "fast" else "sed"
        if fx.variant != expected_variant:
            raise FixtureValidationError(
                f"p10a_scheduling__{scenario_id}.npz: variant={fx.variant!r}, expected {expected_variant!r} "
                f"for scenario_id={scenario_id!r}"
            )
        expected_sigma_diag = scenario_id != "full"
        if bool(fx.sigma_diag) != expected_sigma_diag:
            raise FixtureValidationError(
                f"p10a_scheduling__{scenario_id}.npz: sigma_diag={fx.sigma_diag!r}, expected "
                f"{expected_sigma_diag!r} for scenario_id={scenario_id!r}"
            )
        validate_p10a_scheduling_fixture(fx)
    return fx
