"""Compatibility shim: ``faro-embedded-search`` was renamed to
``askfaro-embedded-search``.

Importing ``faro_embedded_search`` (or any of its submodules) transparently
returns the corresponding object from ``askfaro_embedded_search`` and emits a
``DeprecationWarning``. This package will not receive further updates; please
``pip install askfaro-embedded-search`` and update your imports.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD = "faro_embedded_search"
_NEW = "askfaro_embedded_search"

warnings.warn(
    "faro-embedded-search has been renamed to askfaro-embedded-search. "
    "Update your imports to `askfaro_embedded_search` and run "
    "`pip install askfaro-embedded-search`; this shim will not be updated.",
    DeprecationWarning,
    stacklevel=2,
)


class _RenameFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect any ``faro_embedded_search[.x]`` import to ``askfaro_*``."""

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name == _OLD or name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        new_name = _NEW + spec.name[len(_OLD):]
        module = importlib.import_module(new_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):  # noqa: ARG002
        pass


sys.meta_path.insert(0, _RenameFinder())

# Re-export the top-level public API onto this module object.
from askfaro_embedded_search import *  # noqa: E402,F401,F403
