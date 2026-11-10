"""P7 gate: `backends/base.py`'s execution-backend capability layer --
registry duplicate/unknown handling, `UnsupportedCombination` on every axis,
`MissingBackendDependency` when `numba` genuinely can't be imported, and "no
silent NumPy fallback" (an unsupported/unknown backend must raise, never
quietly run on `numpy` instead).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from chmm_lsim.backends import base as base_module
from chmm_lsim.backends.base import (
    BackendCapabilities,
    MissingBackendDependency,
    UnsupportedCombination,
    check_capability,
    get_backend,
    get_runner,
    register_backend,
    run_backend,
)
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.models.lsim.inference.fast import forward_backward_fast
from chmm_lsim.models.lsim.inference.sed import forward_backward_sed
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure
from chmm_lsim.core.validate import ValidationError

try:
    import numba as _numba_probe  # noqa: F401

    _NUMBA_INSTALLED = True
except ImportError:
    _NUMBA_INSTALLED = False

requires_numba = pytest.mark.skipif(
    not _NUMBA_INSTALLED, reason="numba not installed in this environment (pip install 'chmm-lsim[numba]')"
)

# For `@pytest.mark.parametrize("backend", BACKEND_PARAMS)`: the "numpy" case
# always runs; the "numba" case is individually skipped (not the whole test)
# when numba genuinely isn't installed, since calling `check_capability`/
# `run_backend` with `name="numba"` lazily imports it, which would otherwise
# raise `MissingBackendDependency` instead of exercising the check under test.
BACKEND_PARAMS = ["numpy", pytest.param("numba", marks=requires_numba)]


def _tiny_model():
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    pi_0 = np.array([0.6, 0.4])
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    coupling = np.array([[1.0]])
    gmm = ChannelGMM(weights=np.ones((2, 1)), mu=np.array([0.0, 5.0]).reshape(1, 2, 1), sigma=np.ones((1, 2, 1)), diag=True)
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=[gmm])
    params.validate()
    return structure, params


@requires_numba
def test_numpy_and_numba_are_both_registered_backends():
    caps_numpy = get_backend("numpy")
    assert caps_numpy.name == "numpy"
    assert set(caps_numpy.variants) == {"sed", "fast", "poise"}
    caps_numba = get_backend("numba")
    assert caps_numba.name == "numba"
    assert set(caps_numba.variants) == {"sed", "fast", "poise"}


def test_register_backend_rejects_duplicate_name():
    caps = BackendCapabilities(name="numpy", variants=("sed",))
    with pytest.raises(ValueError, match="already registered"):
        register_backend(caps, lambda *a, **k: None)


def test_get_backend_unknown_name_raises_unsupported_combination():
    with pytest.raises(UnsupportedCombination, match="unknown execution backend"):
        get_backend("tensorflow")


def test_get_runner_unknown_name_raises_unsupported_combination():
    with pytest.raises(UnsupportedCombination):
        get_runner("tensorflow")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"variant": "torch_batched"}, "variant"),
        ({"variant": "sed", "dtype": "float32"}, "dtype"),
        ({"variant": "sed", "mode": "stats"}, "mode"),
        ({"variant": "sed", "batched": True}, "batched"),
        ({"variant": "sed", "method": "reference"}, "sed method"),
    ],
)
@requires_numba
def test_numba_check_capability_rejects_each_unsupported_axis(kwargs, match):
    """`backend="numba"` supports `sed`/`fast`, `float64`, unbatched
    `"posterior"` mode, and only the `"optimized"` sed method -- every other
    axis value must raise `UnsupportedCombination` naming that specific axis."""
    with pytest.raises(UnsupportedCombination, match=match):
        check_capability("numba", **kwargs)


@requires_numba
def test_numpy_check_capability_accepts_reference_method_numba_does_not():
    check_capability("numpy", variant="sed", method="reference")  # must not raise
    with pytest.raises(UnsupportedCombination, match="sed method"):
        check_capability("numba", variant="sed", method="reference")


def test_run_backend_unknown_backend_raises_not_silent_fallback():
    structure, params = _tiny_model()
    Y = np.zeros((1, 3))
    with pytest.raises(UnsupportedCombination, match="backend"):
        run_backend("tensorflow", "sed", structure, params, Y, FbConstants(variant="sed"))


# --------------------------------------------------------------------------
# A review round found three more real gaps, all confirmed and fixed:
# (1) check_structure_matches was never called by forward_backward_sed/fast
#     at all (only P6's own APIs called it) -- backend="numpy" silently
#     accepted a structure/params mismatch and computed a wrong result,
#     while backend="numba" already (accidentally) rejected it.
# (2) run_backend never checked variant == fb_constants.variant, so
#     run_backend("numba", "sed", ..., FbConstants(variant="fast")) silently
#     ran sed while NumPy's own forward_backward_sed rejected the mismatch
#     via its own internal fb_constants.variant check.
# (3) the numba runner unconditionally popped `method` out of kwargs before
#     ever checking it against the requested variant, so
#     run_backend("numba", "fast", ..., method="reference") silently ran
#     fast successfully instead of raising (NumPy's forward_backward_fast
#     has no method= parameter at all, so the same call raised a raw
#     TypeError there -- inconsistent with numba's silent success).
# All three are now checked once, unconditionally, in run_backend itself,
# before either backend's own runner (or validate= handling) ever runs.
# --------------------------------------------------------------------------


def _mismatched_structures():
    structure_wrong_d = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(2,), num_gmm_component=(1,))
    return structure_wrong_d, _tiny_model()[1]  # params.structure has channel_obs_dim=(1,)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("validate", ["off", "params", "all"])
def test_run_backend_rejects_structure_mismatch_regardless_of_backend_or_validate(backend, validate):
    structure_wrong, params = _mismatched_structures()
    Y = np.zeros((2, 3))
    with pytest.raises(ValidationError, match="structure"):
        run_backend(backend, "sed", structure_wrong, params, Y, FbConstants(variant="sed"), validate=validate)


@pytest.mark.parametrize("validate", ["off", "params", "all"])
def test_forward_backward_sed_direct_rejects_structure_mismatch_regardless_of_validate(validate):
    structure_wrong, params = _mismatched_structures()
    Y = np.zeros((2, 3))
    with pytest.raises(ValidationError, match="structure"):
        forward_backward_sed(structure_wrong, params, Y, FbConstants(variant="sed"), validate=validate)


@pytest.mark.parametrize("validate", ["off", "params", "all"])
def test_forward_backward_fast_direct_rejects_structure_mismatch_regardless_of_validate(validate):
    structure_wrong, params = _mismatched_structures()
    Y = np.zeros((2, 3))
    with pytest.raises(ValidationError, match="structure"):
        forward_backward_fast(structure_wrong, params, Y, FbConstants(variant="fast"), validate=validate)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("requested_variant,fbc_variant", [("sed", "fast"), ("fast", "sed")])
def test_run_backend_rejects_conflicting_variant_both_directions_both_backends(backend, requested_variant, fbc_variant):
    structure, params = _tiny_model()
    Y = np.zeros((1, 3))
    with pytest.raises(ValueError, match="conflicting inference variant"):
        run_backend(backend, requested_variant, structure, params, Y, FbConstants(variant=fbc_variant))


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_run_backend_rejects_method_for_fast_with_unsupported_combination(backend):
    """The regression this reproduces specifically: numba's runner used to
    `kwargs.pop("method", None)` unconditionally and then run `fast`
    successfully. Both backends now raise the *same* exception type with a
    message naming the backend, the variant, and the method -- not merely
    "some exception" (NumPy's pre-existing behavior for this used to be an
    incidental `TypeError` from `forward_backward_fast`'s own signature, not
    an intentional check; a review round found `check_capability` itself
    didn't share this rule with `run_backend`, so the two disagreed -- see
    `test_check_capability_rejects_method_for_non_sed_variant` below, which
    checks `check_capability` directly for the exact case the review round
    confirmed failing)."""
    structure, params = _tiny_model()
    Y = np.zeros((1, 3))
    with pytest.raises(UnsupportedCombination) as exc_info:
        run_backend(backend, "fast", structure, params, Y, FbConstants(variant="fast"), method="reference")
    message = str(exc_info.value)
    assert backend in message
    assert "fast" in message
    assert "reference" in message


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_check_capability_rejects_method_for_non_sed_variant(backend):
    """Direct regression test for the exact case a review round confirmed
    failing: `check_capability(backend, variant="fast", method="reference")`
    used to succeed silently for *both* backends, even though the very same
    request through `run_backend` correctly raised -- the exported
    capability-check API disagreeing with actual execution. `method=` is
    only ever meaningful for `variant="sed"`; a real value for any other
    variant must raise here directly, not only when routed through
    `run_backend`."""
    with pytest.raises(UnsupportedCombination) as exc_info:
        check_capability(backend, variant="fast", method="reference")
    message = str(exc_info.value)
    assert backend in message
    assert "fast" in message
    assert "reference" in message


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_check_capability_accepts_method_none_for_non_sed_variant(backend):
    """`method=None` (the default -- "no method requested") is not itself
    evidence of `sed`-specific intent, so `check_capability` alone (as
    opposed to `run_backend`, which can see whether `method` was passed at
    all) must not reject it for `variant="fast"`."""
    check_capability(backend, variant="fast", method=None)  # must not raise


@requires_numba
def test_run_backend_numba_rejects_method_for_fast_with_unsupported_combination():
    """On the numba backend specifically, the rejection is `UnsupportedCombination`
    (the "preferred" exception type per the fix), raised before
    `_run_numba`'s own body ever runs."""
    structure, params = _tiny_model()
    Y = np.zeros((1, 3))
    with pytest.raises(UnsupportedCombination, match="method"):
        run_backend("numba", "fast", structure, params, Y, FbConstants(variant="fast"), method="reference")


def test_run_backend_rejects_explicit_method_none_for_fast():
    """The one case `check_capability` alone cannot decide (see the test
    above): an *explicitly* passed `method=None` for `variant="fast"` is
    still rejected by `run_backend` itself, which can see the key was
    present in `**kwargs` at all."""
    structure, params = _tiny_model()
    Y = np.zeros((1, 3))
    with pytest.raises(UnsupportedCombination, match="method"):
        run_backend("numpy", "fast", structure, params, Y, FbConstants(variant="fast"), method=None)


@requires_numba
def test_run_backend_numba_rejects_reference_method_before_touching_numba_code():
    """The rejection must happen at the capability-check layer, before the
    numba runner (or numba itself) is ever invoked -- confirmed indirectly:
    this raises even though `Y` here is deliberately malformed (wrong D),
    which would surface as a *different* error if the request ever reached
    emission computation."""
    structure, params = _tiny_model()
    bad_Y = np.zeros((99, 3))
    with pytest.raises(UnsupportedCombination, match="sed method"):
        run_backend("numba", "sed", structure, params, bad_Y, FbConstants(variant="sed"), method="reference")


def test_missing_numba_dependency_raises_actionable_error(monkeypatch):
    """Simulates `numba` genuinely not being importable (distinct from it
    merely not being *requested*): `sys.modules["numba"] = None` makes any
    `import numba` raise `ImportError` immediately, matching what a real
    core-only install (no `numba` extra) would do. Forces a fresh import
    attempt of `chmm_lsim.backends.numba_` (it may already be cached/registered
    from an earlier test in this session) and restores everything via
    `monkeypatch`'s own teardown, so later tests still see a working numba
    backend."""
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.delitem(sys.modules, "chmm_lsim.backends.numba_", raising=False)
    monkeypatch.delitem(base_module._BACKENDS, "numba", raising=False)
    monkeypatch.delitem(base_module._RUNNERS, "numba", raising=False)

    with pytest.raises(MissingBackendDependency, match=r"pip install 'chmm-lsim\[numba\]'"):
        get_backend("numba")


def test_missing_numba_dependency_recovers_once_dependency_is_available_again(monkeypatch):
    """Confirms the previous test's failure is a property of the simulated
    environment, not permanent state corruption: with `numba` simulated as
    unavailable, `get_backend("numba")` fails; once monkeypatch tears that
    simulation down, the outcome must match whatever `numba`'s *real*
    availability in this environment is -- success if the `numba` extra is
    actually installed here, the same `MissingBackendDependency` as before
    if this is a genuine core-only install."""
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.delitem(sys.modules, "chmm_lsim.backends.numba_", raising=False)
    monkeypatch.delitem(base_module._BACKENDS, "numba", raising=False)
    monkeypatch.delitem(base_module._RUNNERS, "numba", raising=False)
    with pytest.raises(MissingBackendDependency):
        get_backend("numba")
    monkeypatch.undo()
    if _NUMBA_INSTALLED:
        assert get_backend("numba").name == "numba"
    else:
        with pytest.raises(MissingBackendDependency):
            get_backend("numba")
