"""Clear, actionable setup errors.

These wrap the cryptic underlying failures (a bare ImportError, a Postgres
"type vector does not exist", a missing FTS5 build) with a message that tells
an adopter exactly what to do — the difference between a 10-second fix and an
hour of confusion.
"""

from __future__ import annotations


class FaroSearchError(Exception):
    """Base class for all askfaro-embedded-search setup/configuration errors."""


class MissingDependencyError(FaroSearchError, ImportError):
    """A required optional dependency (an install extra) is not installed."""

    def __init__(self, what: str, extra: str, package: str):
        super().__init__(
            f"{what} requires the '{extra}' extra, which isn't installed.\n"
            f"  Install it with:  pip install \"askfaro-embedded-search[{extra}]\"\n"
            f"  (missing package: {package})"
        )


class ConfigurationError(FaroSearchError):
    """The index is misconfigured or its backing store isn't set up."""
