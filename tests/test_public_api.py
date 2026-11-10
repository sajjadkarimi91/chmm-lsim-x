"""Public-API stability: every name in the stored snapshot (tests/public_api.json) must
still exist, and every function/class signature must stay call-compatible.

Allowed changes: new names, and new keyword-only parameters that have a default.
Everything else (a removed name, a removed/renamed/reordered parameter, a changed
default or kind, a new required parameter) fails. New names and parameters are
reported so the snapshot can be refreshed with tests/update_public_api.py.
"""

from __future__ import annotations

import json
import warnings

import pytest

from update_public_api import NAMESPACES, SNAPSHOT, collect

STORED = json.loads(SNAPSHOT.read_text())

_POSITIONAL = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")


def signature_incompatibilities(old: list[dict], new: list[dict]) -> tuple[list[str], list[str]]:
    """`(problems, additions)` between a stored and a current parameter list."""
    problems, additions = [], []
    old_pos = [p for p in old if p["kind"] in _POSITIONAL]
    new_pos = [p for p in new if p["kind"] in _POSITIONAL]
    for i, p in enumerate(old_pos):
        if i >= len(new_pos):
            problems.append(f"positional parameter {p['name']!r} removed")
            continue
        q = new_pos[i]
        if (q["name"], q["kind"]) != (p["name"], p["kind"]):
            problems.append(f"positional parameter {i} changed from {p['name']!r} ({p['kind']}) to {q['name']!r} ({q['kind']})")
        elif q["default"] != p["default"] or q["has_default"] != p["has_default"]:
            problems.append(f"default of {p['name']!r} changed from {p['default']} to {q['default']}")
    for q in new_pos[len(old_pos):]:
        problems.append(f"new positional parameter {q['name']!r} (only keyword-only parameters with defaults may be added)")

    old_kw = {p["name"]: p for p in old if p["kind"] == "KEYWORD_ONLY"}
    new_kw = {p["name"]: p for p in new if p["kind"] == "KEYWORD_ONLY"}
    for name, p in old_kw.items():
        q = new_kw.get(name)
        if q is None:
            problems.append(f"keyword-only parameter {name!r} removed")
        elif q["default"] != p["default"] or q["has_default"] != p["has_default"]:
            problems.append(f"default of {name!r} changed from {p['default']} to {q['default']}")
    for name, q in new_kw.items():
        if name not in old_kw:
            (additions if q["has_default"] else problems).append(
                f"new keyword-only parameter {name!r}" + ("" if q["has_default"] else " without a default")
            )

    for kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
        if any(p["kind"] == kind for p in old) and not any(q["kind"] == kind for q in new):
            problems.append(f"{kind} parameter removed")
    return problems, additions


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_public_api_is_backward_compatible(namespace):
    stored = STORED[namespace]
    current = collect(namespace, strict=False)
    problems, additions = [], []
    for name, old in stored.items():
        if name not in current:
            problems.append(f"{namespace}.{name}: removed from __all__")
            continue
        new = current[name]
        if new is None:  # optional dependency missing here (e.g. torch): presence checked only
            continue
        if old["kind"] != new["kind"]:
            problems.append(f"{namespace}.{name}: kind changed from {old['kind']} to {new['kind']}")
            continue
        if old.get("parameters") is not None and new.get("parameters") is not None:
            p, a = signature_incompatibilities(old["parameters"], new["parameters"])
            problems += [f"{namespace}.{name}: {m}" for m in p]
            additions += [f"{namespace}.{name}: {m}" for m in a]
    additions += [f"{namespace}.{name}: new public name" for name in current if name not in stored]
    if additions:
        message = "public API additions (refresh tests/public_api.json with tests/update_public_api.py):\n  " + "\n  ".join(additions)
        print(message)
        warnings.warn(message, UserWarning, stacklevel=1)
    assert not problems, "incompatible public API changes:\n  " + "\n  ".join(problems)


def test_compatibility_rules():
    def par(name, kind="POSITIONAL_OR_KEYWORD", default=None):
        return {"name": name, "kind": kind, "default": default, "has_default": default is not None}

    old = [par("a"), par("b", default="1"), par("c", "KEYWORD_ONLY", "True")]
    assert signature_incompatibilities(old, old) == ([], [])
    ok, added = signature_incompatibilities(old, old + [par("d", "KEYWORD_ONLY", "None")])
    assert ok == [] and added
    for new in (
        [par("a"), par("c", "KEYWORD_ONLY", "True")],                                   # removed
        [par("b", default="1"), par("a"), par("c", "KEYWORD_ONLY", "True")],            # reordered
        [par("a"), par("bb", default="1"), par("c", "KEYWORD_ONLY", "True")],           # renamed
        [par("a"), par("b", default="2"), par("c", "KEYWORD_ONLY", "True")],            # default changed
        old + [par("d", "KEYWORD_ONLY")],                                                # new required
        [par("a"), par("b", default="1"), par("x", default="0"), par("c", "KEYWORD_ONLY", "True")],  # new positional
    ):
        assert signature_incompatibilities(old, new)[0], new


def test_version_is_single_sourced():
    import chmm_lsim

    assert isinstance(chmm_lsim.__version__, str) and chmm_lsim.__version__.count(".") == 2
