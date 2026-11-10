"""Deprecation helpers (internal).

* `deprecated_alias` wraps a function under an old name so that calling it
  emits a `DeprecationWarning` naming the replacement.
* `moved_module` turns a module at a historical import path into a
  re-exporting alias of its new location(s): it copies every name of the new
  modules into the old module's namespace (the very same objects, so
  ``old.f is new.f``) and emits one `DeprecationWarning` when the old path
  is imported.
* `lazy_moved_package` does the same for a historical package whose
  submodules are themselves moved modules, warning on first attribute access
  instead of on import, so importing one of its submodules warns only once.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
import warnings
from typing import Callable

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _external_stacklevel() -> int:
    """`stacklevel` that attributes a warning to the first frame outside this
    package and the import machinery (i.e. to the user's import statement)."""
    level = 1
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not (filename.startswith(_PACKAGE_DIR) or filename.startswith("<frozen importlib")):
            return level
        frame = frame.f_back
        level += 1
    return 1


def _warn(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=_external_stacklevel())


def deprecated_alias(new_func: Callable, old_name: str, *, since: str, removal: str | None = None) -> Callable:
    """`new_func` under the deprecated name `old_name`: same behaviour, plus a
    `DeprecationWarning` on every call naming `new_func` as the replacement."""
    new_name = f"{new_func.__module__}.{new_func.__qualname__}"
    if new_name.startswith("chmm_lsim.") and hasattr(sys.modules.get("chmm_lsim"), new_func.__name__):
        new_name = f"chmm_lsim.{new_func.__name__}"
    removal_note = f" and will be removed in {removal}" if removal else ""
    message = f"{old_name} is deprecated since {since}{removal_note}; use {new_name} instead"

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        _warn(message)
        return new_func(*args, **kwargs)

    wrapper.__doc__ = f"Deprecated alias of `{new_name}` (since {since}).\n\n" + (new_func.__doc__ or "")
    wrapper.__deprecated_alias_of__ = new_func
    return wrapper


def _copy_names(namespace: dict, sources: list[str], aliases: dict[str, str] | None) -> list:
    modules = [importlib.import_module(name) for name in sources]
    for module in reversed(modules):  # earlier sources win on name clashes
        for key, value in vars(module).items():
            if not (key.startswith("__") and key.endswith("__")):
                namespace[key] = value
    for old, target in (aliases or {}).items():
        module_name, _, attr = target.partition(":")
        namespace[old] = getattr(importlib.import_module(module_name), attr)
    return modules


# Modules whose names the historical modules imported and therefore also exposed.
_COMMON_SOURCES = (
    "chmm_lsim.core.structure",
    "chmm_lsim.core.validate",
    "chmm_lsim.core.constants",
    "chmm_lsim.emissions.base",
    "chmm_lsim.emissions.gmm",
    "chmm_lsim.models.lsim.params",
    "chmm_lsim.backends.base",
    "chmm_lsim.backends.numpy_",
)


def moved_module(namespace: dict, sources: list[str], *, replacement: str, aliases: dict[str, str] | None = None,
                 public_all: list[str] | None = None) -> None:
    """Make the module whose globals are `namespace` an alias of `sources`
    (new module paths, searched in order, then the commonly imported core
    modules) and warn once.

    `replacement` is the human-readable advice in the warning (e.g. "import
    from chmm_lsim"); `aliases` maps names whose new spelling differs to
    ``"module:attribute"``; `public_all` restores a historical `__all__`."""
    old = namespace["__name__"]
    reexported = list(sources) + [m for m in _COMMON_SOURCES if m not in sources]
    _copy_names(namespace, reexported, aliases)
    namespace["__all__"] = list(public_all) if public_all is not None else [
        k for k in namespace if not k.startswith("_") and not isinstance(namespace[k], type(sys))
    ]
    namespace["__moved_to__"] = tuple(sources)
    namespace["__reexported_from__"] = tuple(reexported)
    namespace["__aliases__"] = dict(aliases or {})
    _warn(f"{old} is deprecated and will be removed in a future release; {replacement} (now in {', '.join(sources)})")


def lazy_moved_package(namespace: dict, sources: list[str], *, replacement: str, submodules: tuple[str, ...]) -> None:
    """Like `moved_module`, for a historical package: nothing is copied or
    warned on import (so ``import old.package.sub`` warns only for ``sub``);
    the first attribute access that is not one of `submodules` warns and
    resolves every attribute from `sources`."""
    old = namespace["__name__"]
    state = {"warned": False}
    namespace["__moved_to__"] = tuple(sources)
    namespace["__reexported_from__"] = tuple(sources)
    namespace["__aliases__"] = {}

    def __getattr__(name: str):
        if name in submodules or (name.startswith("__") and name.endswith("__") and name != "__all__"):
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        if not state["warned"]:
            state["warned"] = True
            _warn(f"{old} is deprecated and will be removed in a future release; {replacement} (now in {', '.join(sources)})")
        for source in sources:
            module = importlib.import_module(source)
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"module {old!r} has no attribute {name!r}")

    namespace["__getattr__"] = __getattr__
