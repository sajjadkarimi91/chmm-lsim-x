"""Regenerate tests/public_api.json, the public-API snapshot checked by test_public_api.py.

Run after an intentional, backward-compatible API addition:

    PYTHONPATH=src python tests/update_public_api.py

The snapshot records every name of `chmm_lsim.__all__` and
`chmm_lsim.discriminative.__all__` and, for functions and classes, the
parameters of `inspect.signature` (name, kind, default as repr). Generate it
in an environment with every optional dependency installed (torch provides
the lazily imported batched names).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("public_api.json")
NAMESPACES = ("chmm_lsim", "chmm_lsim.discriminative")


def _parameters(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [
        {
            "name": p.name,
            "kind": p.kind.name,
            "default": None if p.default is inspect.Parameter.empty else repr(p.default),
            "has_default": p.default is not inspect.Parameter.empty,
        }
        for p in sig.parameters.values()
    ]


def describe(obj) -> dict:
    if inspect.isclass(obj):
        return {"kind": "class", "parameters": _parameters(obj)}
    if callable(obj):
        return {"kind": "function", "parameters": _parameters(obj)}
    return {"kind": "constant", "type": type(obj).__name__}


def collect(namespace: str, *, strict: bool = True) -> dict[str, dict | None]:
    """`{name: description}` for every name in `namespace.__all__`; a name whose
    optional dependency is missing maps to `None` unless `strict`."""
    import importlib

    module = importlib.import_module(namespace)
    out: dict[str, dict | None] = {}
    for name in module.__all__:
        try:
            obj = getattr(module, name)
        except ImportError:
            if strict:
                raise
            out[name] = None
            continue
        out[name] = describe(obj)
    return out


def snapshot() -> dict:
    return {ns: collect(ns) for ns in NAMESPACES}


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {SNAPSHOT}")
