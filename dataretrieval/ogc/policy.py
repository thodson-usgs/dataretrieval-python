"""Low-level OGC policy: the dialect type and control validation.

This module is the one definition of the dialect types (where a service
departs from the standard, ADR 0014) and OGC control validation.
It depends only on the stdlib, so any OGC submodule can import it without
creating cycles.

It names no endpoint: which API an OGC call targets is the *adapter's*
policy, supplied per call as ``base_url``. A default here would
direct every generic OGC caller to one API.

It must not import engine, shaping, or any collection adapter.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field


def _require_positive_int(
    value: int, name: str, *, examples: str | None = None
) -> None:
    """Validate a positive-integer OGC count control.

    Any :class:`numbers.Integral` is accepted, including NumPy and pandas
    integers, but ``bool`` is rejected despite being an ``Integral`` subtype.
    """
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
        eg = f", e.g. {examples}" if examples else ""
        raise ValueError(f"{name} must be a positive integer{eg} (got {value!r}).")


# ---------------------------------------------------------------------------
# Dialect types
# ---------------------------------------------------------------------------
# Each field is one way a service departs from the OGC standard, and its default
# is the standard (ADR 0014). The adapter that sets a field says which service
# needs it and why.


@dataclass(frozen=True)
class DateDialect:
    """How a service departs from the standard ``datetime`` grammar.

    Read only by :mod:`dataretrieval.ogc.dates`, so a service that shares the
    grammar without being an OGC Features API can be given one alone.

    Attributes
    ----------
    params : frozenset[str]
        Request parameters that carry a date or a range. The standard has one,
        ``datetime``; a service that filters on other temporal properties names
        them all here.
    date_only_collections : frozenset[str]
        Collections whose date parameters are sent as ``YYYY-MM-DD`` rather
        than a UTC timestamp.
    always_datetime : frozenset[str]
        Parameters sent as a UTC timestamp even in a date-only collection.
    open_start : str
        How the open start of a range is spelled: ``".."`` (``"../2024-01-01"``)
        or ``""`` (``"/2024-01-01"``). The standard allows both; a deployment
        may accept only one. An open end is always ``".."``.
    durations : bool
        Whether a lone ISO 8601 duration (``"P7D"``) is sent as given. The
        standard does not define durations, so by default one raises
        ``ValueError`` before any request is sent.
    """

    params: frozenset[str] = frozenset({"datetime"})
    date_only_collections: frozenset[str] = field(default_factory=frozenset)
    always_datetime: frozenset[str] = field(default_factory=frozenset)
    open_start: str = ".."
    durations: bool = False


@dataclass(frozen=True)
class OgcDialect:
    """How a service departs from OGC API Features, grouped by what reads it.

    Attributes
    ----------
    cql2_collections : frozenset[str]
        Collections that don't accept comma-separated multi-value GET
        parameters and so must be queried via POST with a CQL2 JSON body.
        Read by request construction.
    dates : DateDialect
        Date parameters and how they are spelled; see :class:`DateDialect`.
    time_cols : frozenset[str]
        Result columns to coerce to datetime when ``convert_type`` is set.
        Read by result shaping, as are the next two.
    numerical_cols : frozenset[str]
        Result columns to coerce to numeric when ``convert_type`` is set.
    sort_cols : tuple[str, ...]
        Columns to sort the combined result by, in priority order. Sorting
        is applied only when the first (primary) column is present; any
        later columns also present are added as secondary keys.
    """

    cql2_collections: frozenset[str] = field(default_factory=frozenset)
    dates: DateDialect = field(default_factory=DateDialect)
    time_cols: frozenset[str] = field(default_factory=frozenset)
    numerical_cols: frozenset[str] = field(default_factory=frozenset)
    sort_cols: tuple[str, ...] = field(default_factory=tuple)


# A service that implements the standard and nothing more.
DEFAULT_DIALECT = OgcDialect()
