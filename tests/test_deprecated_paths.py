"""Historical module paths (before the core/emissions/models/learning layout) keep
working: each imports, emits exactly one DeprecationWarning naming the new
location, and exposes the very same objects as the new modules, including the
private helpers listed in tests/legacy_module_names.json.

Each check runs in a fresh interpreter so that "warns on import" is observed even
when another test already imported the module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

LEGACY = json.loads(Path(__file__).with_name("legacy_module_names.json").read_text())
SRC = str(Path(__file__).resolve().parents[1] / "src")

# Public or still-current paths that must NOT warn.
NON_DEPRECATED = ("chmm_lsim.emissions", "chmm_lsim.discriminative")
OPTIONAL_DEPENDENCY_PATHS: dict[str, str] = {}

_PROBE = textwrap.dedent(
    """
    import importlib, json, sys, warnings
    old, names = sys.argv[1], json.loads(sys.argv[2])
    import chmm_lsim  # the package itself must import without warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(old)
        for name in names:
            getattr(module, name)
    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    sources = getattr(module, "__moved_to__", ())
    reexported = getattr(module, "__reexported_from__", ())
    aliases = getattr(module, "__aliases__", {})
    mismatched = []
    for name in names:
        obj = getattr(module, name)
        if name in aliases:
            target_module, _, attr = aliases[name].partition(":")
            ok = getattr(importlib.import_module(target_module), attr) is obj
        else:
            ok = not reexported or any(getattr(importlib.import_module(s), name, object()) is obj for s in reexported)
        if not ok:
            mismatched.append(name)
    print(json.dumps({"deprecations": deprecations, "sources": list(sources), "mismatched": mismatched}))
    """
)


def _probe(old: str, names: list[str]) -> dict:
    proc = subprocess.run([sys.executable, "-c", _PROBE, old, json.dumps(names)], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": SRC}, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("old", sorted(LEGACY))
def test_old_module_path_still_works(old):
    dep = OPTIONAL_DEPENDENCY_PATHS.get(old)
    if dep is not None:
        pytest.importorskip(dep)
    result = _probe(old, LEGACY[old])
    assert result["mismatched"] == [], f"{old}: names not identical to the new location: {result['mismatched']}"
    if old in NON_DEPRECATED:  # still a valid path (now a package): same names, no warning
        assert result["deprecations"] == []
        return
    assert len(result["deprecations"]) == 1, result["deprecations"]
    message = result["deprecations"][0]
    assert old in message and "deprecated" in message
    assert all(s in message for s in result["sources"])


@pytest.mark.parametrize("path", NON_DEPRECATED)
def test_current_paths_do_not_warn(path):
    result = _probe(path, [])
    assert result["deprecations"] == []


def test_import_chmm_lsim_never_warns():
    proc = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", "import chmm_lsim, chmm_lsim.discriminative"],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC}, timeout=300)
    assert proc.returncode == 0, proc.stderr


def test_historical_emissions_names_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from chmm_lsim.emissions import ChannelGMM, compute_emissions, gmm_pdf_diag, gmm_pdf_full
        from chmm_lsim.emissions.gmm import ChannelGMM as gmm_cls

    import chmm_lsim

    assert ChannelGMM is gmm_cls is chmm_lsim.ChannelGMM
    assert compute_emissions is chmm_lsim.compute_emissions
    assert gmm_pdf_diag is chmm_lsim.gmm_pdf_diag and gmm_pdf_full is chmm_lsim.gmm_pdf_full


def test_discriminative_aliases_are_warning_free():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from chmm_lsim.discriminative import em_lsim_discriminative, lsim_discriminative_predict, predict, train_discriminative

    assert train_discriminative is em_lsim_discriminative
    assert predict is lsim_discriminative_predict


def test_deprecated_alias_helper():
    import warnings

    from chmm_lsim import em_lsim
    from chmm_lsim._deprecation import deprecated_alias

    old = deprecated_alias(em_lsim, "chmm_lsim.old_em", since="1.1.0", removal="2.0.0")
    assert old.__wrapped__ is em_lsim and old.__name__ == "em_lsim"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            old()  # forwards to em_lsim, which needs arguments
    assert len(caught) == 1 and issubclass(caught[0].category, DeprecationWarning)
    assert "chmm_lsim.old_em" in str(caught[0].message) and "chmm_lsim.em_lsim" in str(caught[0].message)
    assert "2.0.0" in str(caught[0].message)
