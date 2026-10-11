"""Date and datetime marshalling for OGC time parameters.

Pure helpers that render a getter's date arguments -- single instants,
two-element ``[start, end]`` ranges, and open ``..`` bounds -- into the
``datetime`` grammar of OGC API Features. Which parameters carry dates, and
where a service departs from that grammar, is the caller's
:class:`~dataretrieval.ogc.policy.DateDialect` (ADR 0014). No I/O, no engine
state.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date as _date
from datetime import datetime
from typing import Any, NoReturn
from zoneinfo import ZoneInfo

import pandas as pd

from dataretrieval.ogc.policy import DateDialect

_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


# Anchored to ``[Pp]\d`` so a normal word containing ``p`` (e.g. ``"Apr"``)
# doesn't get mis-classified as an ISO 8601 duration; the optional ``T``
# admits time-only forms like ``PT36H``.
_DURATION_RE = re.compile(r"^[Pp]T?\d")

_STANDARD_DATES = DateDialect()


def _parse_datetime(value: str) -> datetime | None:
    """Parse a single datetime string against the supported formats.

    Returns a ``datetime`` (tz-aware iff the input included a UTC offset),
    or ``None`` if no format matched.
    """
    # ``datetime.strptime`` accepts a numeric offset like ``+00:00`` but not
    # the ``Z`` shorthand, so normalize trailing ``Z`` first.
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    for fmt in _DATETIME_FORMATS:
        try:
            # DTZ007: naive is the documented outcome for a naive input --
            # ``_DATETIME_FORMATS`` carries both the ``%z`` and the bare forms.
            return datetime.strptime(candidate, fmt)  # noqa: DTZ007
        except ValueError:
            continue
    return None


# The OGC API spelling of an open interval bound, as in ``"2024-01-01/.."``.
_OPEN_BOUND = ".."


# One element of a date argument: an ISO 8601 string, or a ``date``,
# ``datetime`` or ``pandas.Timestamp`` (both subclasses of ``datetime.date``),
# with ``None`` (or ``NaN`` / ``NaT``) for an open bound.
_DateLike = str | _date | None

# A getter's date argument (``time``, ``begin``, ``end``, ``datetime``,
# ``last_modified``): one value, or a range of one or two elements with
# ``None`` for an open end. Getters annotate it ``DateInput | None``.
# ``NaT`` is accepted at runtime but left out: ``pandas.api.typing.NaTType``
# needs pandas 2.1, and without pandas-stubs it is ``Any``, which would make
# the whole alias accept anything.
DateInput = str | _date | Iterable[_DateLike]


def _is_blank(dt: _DateLike) -> bool:
    """True for a None, NaN, empty-string, or ``..`` element.

    Each is a spelling of an open bound, so ``["2024-01-01", ".."]`` means the
    same range as ``["2024-01-01", None]`` and the string ``"2024-01-01/.."``.
    """
    return dt is None or bool(pd.isna(dt)) or dt in ("", _OPEN_BOUND)


def _to_datetime(dt: str | _date, *, name: str) -> datetime:
    """Read one non-blank element as a ``datetime`` (naive iff it has no zone).

    A ``date`` is midnight of that day, like the string ``"2024-01-01"``.
    Raises ``ValueError`` naming *name* for an unreadable string or a value of
    another type.
    """
    if isinstance(dt, pd.Timestamp):
        # ``Timestamp.astimezone`` is ``tz_convert`` and requires a zone, so
        # hand ``_format_one`` a plain ``datetime``.
        converted: datetime = dt.to_pydatetime(warn=False)
        return converted
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, _date):
        return datetime(dt.year, dt.month, dt.day)  # noqa: DTZ001
    parsed = _parse_datetime(dt) if isinstance(dt, str) else None
    if parsed is None:
        raise ValueError(
            f"{name} could not be read as a date or datetime: {dt!r}. "
            "Pass an ISO 8601 date or datetime such as '2024-01-01' or "
            "'2024-01-01T12:00:00Z' (a datetime.date, datetime.datetime or "
            "pandas.Timestamp also works), and None for an open end of a range "
            "(['2024-01-01', None])."
        )
    return parsed


def _format_one(dt: _DateLike, *, date: bool, name: str) -> str:
    """Format a single datetime element for inclusion in the API time arg.

    Raises ``ValueError`` naming *name* when the element is not blank and
    cannot be read as a date or datetime.
    """
    if dt is None or _is_blank(dt):
        return _OPEN_BOUND
    parsed = _to_datetime(dt, name=name)
    if date:
        return parsed.strftime("%Y-%m-%d")
    # Naive inputs are interpreted in the system local zone (for backwards
    # compatibility). Use ``.astimezone()`` rather than a fixed offset so each
    # value is resolved against the DST rules for its own date — a fixed
    # ``datetime.now()`` offset shifted off-season inputs by an hour.
    aware = parsed if parsed.tzinfo is not None else parsed.astimezone()
    return aware.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_to_list(
    datetime_input: DateInput,
    name: str = "date input",
) -> list[_DateLike]:
    """Normalize datetime input to a list, raising on invalid shapes."""
    if isinstance(datetime_input, str) or not isinstance(datetime_input, Iterable):
        # A lone value (a ``date`` is not iterable); a non-date is rejected
        # per element.
        return [datetime_input]
    if isinstance(datetime_input, Mapping):
        raise TypeError(
            f"{name} must be a date, a string, or a sequence of them, "
            f"not {type(datetime_input).__name__}."
        )
    return list(datetime_input)


def _split_interval(interval: str, *, name: str) -> tuple[str, str]:
    """Split a ``"start/end"`` string into its two sides.

    Each side is then formatted like a list element, so
    ``"2024-01-01T10:00:00/.."`` sends the same range as
    ``["2024-01-01T10:00:00", None]``.

    Raises ``ValueError`` naming *name* unless the string is two sides
    separated by one ``"/"``, neither of them a duration: the standard defines
    an interval as two instants, so ``"2024-01-01/P7D"`` is not one.
    """
    sides = interval.split("/")
    if len(sides) != 2 or any(_DURATION_RE.match(side) for side in sides):
        raise ValueError(
            f"{name} is not a valid interval: {interval!r}. Pass a start and "
            "an end separated by '/', such as '2024-01-01/2024-12-31', with "
            "'..' for an open end ('2024-01-01/..'). Neither side can be a "
            "duration."
        )
    return sides[0], sides[1]


def _all_blank(items: list[_DateLike]) -> bool:
    """True when every element is None, NaN, the empty string, or ``..``."""
    return all(_is_blank(dt) for dt in items)


def _reject_too_many(
    items: list[_DateLike], *, name: str, dialect: DateDialect
) -> NoReturn:
    """Raise for more than two values, offering only forms *dialect* accepts."""
    single = (
        "an instant or a duration ('2020-01-01', 'P7D')"
        if dialect.durations
        else "an instant ('2020-01-01')"
    )
    raise ValueError(
        f"{name} takes at most 2 values, got {len(items)}: {items!r}. "
        f"Pass one value for {single}, "
        "or two for a closed interval ('2020-01-01', '2020-12-31')."
    )


def _lone_duration(
    items: list[_DateLike], *, name: str, dialect: DateDialect
) -> str | None:
    """The duration to send as given, if *items* is one; None otherwise.

    Raises ``ValueError`` naming *name* for a duration *dialect* does not
    accept.
    """
    if len(items) != 1 or not isinstance(items[0], str):
        return None
    value = items[0]
    if not _DURATION_RE.match(value):
        return None
    if not dialect.durations:
        raise ValueError(
            f"{name} does not accept an ISO 8601 duration such as {value!r} "
            "for this service. Pass a range instead, with None for an open "
            "end, such as ['2024-01-01', None]."
        )
    return value


def _format_api_dates(
    datetime_input: DateInput | None,
    date: bool = False,
    *,
    name: str = "date input",
    dialect: DateDialect = _STANDARD_DATES,
) -> str | None:
    """
    Formats date or datetime input(s) for use with an API.

    Handles single values or ranges, converting to ISO 8601 or date-only
    formats as needed.

    Parameters
    ----------
    datetime_input : Union[str, date, List[Optional[Union[str, date]]], None]
        A single date/datetime or a list of one or two of them. Each may be a
        ``datetime.date``, ``datetime.datetime`` or ``pandas.Timestamp`` (a
        naive one is read in the local time zone, as a naive string is), or a
        string. Strings accept formats like "%Y-%m-%d %H:%M:%S", ISO 8601
        (with or without ``Z``/numeric offset), or, where *dialect* accepts
        them, a lone duration ("P7D", "PT36H"). Range endpoints may be
        ``None``/``NaN``/empty or ``".."`` to denote a half-bounded range.
    date : bool, optional
        If True, uses only the date portion ("YYYY-MM-DD"). If False (default),
        returns full datetime in UTC ISO 8601 format ("YYYY-MM-DDTHH:MM:SSZ").
    name : str, optional
        The caller's own spelling of this argument, used as the subject of
        every message raised here. Defaults to a generic "date input"; pass
        the real parameter name (``"time"``, ``"last_modified"``) so a caller
        correcting the error edits an argument their getter accepts.
    dialect : DateDialect, optional
        Where the target service departs from the standard grammar: whether
        it accepts a lone duration and how it spells an open start. Defaults
        to the standard, which accepts no duration and spells an open start
        ``".."``.

    Returns
    -------
    Union[str, None]
        - If input is a single value, returns the formatted date/datetime string.
        - If input is a list of two values, returns a date/datetime range string
          separated by "/" (e.g., "YYYY-MM-DD/YYYY-MM-DD" or
          "YYYY-MM-DDTHH:MM:SSZ/YYYY-MM-DDTHH:MM:SSZ").
        - Returns None if input is None, empty, or every element is blank.

    Raises
    ------
    ValueError
        If `datetime_input` contains more than two values, an element that
        is not blank matches no supported format or is of another type, or it
        is a lone duration that *dialect* does not accept.

    Notes
    -----
    - A single blank/NA value returns None. In a two-value range, a blank/NA
      or ``".."`` endpoint is rendered as ``".."`` to denote an open bound
      (e.g. ``"2024-01-01/.."``); the range is only None when *every* element
      is blank/NA/``".."``.
    - A lone ISO 8601 duration such as "P7D" or "PT36H" is passed through
      unchanged when ``dialect.durations`` is set, and rejected otherwise.
    - A single string containing ``"/"`` is an interval: it is split into the
      two-value form, so ``"2024-01-01/.."`` and ``["2024-01-01", None]`` are
      formatted alike. A duration is not accepted as either side.
    - Converts datetimes to UTC and formats as ISO 8601 with 'Z' suffix when
      `date` is False. Inputs with an explicit offset (``Z`` or ``+HH:MM``) are
      converted from that offset to UTC; naive inputs are interpreted in the
      local time zone for backwards compatibility.
    """
    if datetime_input is None:
        return None

    items = _coerce_to_list(datetime_input, name)
    if len(items) == 1 and isinstance(items[0], str) and "/" in items[0]:
        items = list(_split_interval(items[0], name=name))

    if _all_blank(items):
        return None

    if len(items) > 2:
        _reject_too_many(items, name=name, dialect=dialect)

    duration = _lone_duration(items, name=name, dialect=dialect)
    if duration is not None:
        return duration

    formatted = [_format_one(dt, date=date, name=name) for dt in items]
    if len(formatted) == 2 and formatted[0] == _OPEN_BOUND:
        formatted[0] = dialect.open_start
    return "/".join(formatted)


def _format_date_args(
    args: dict[str, Any], collection: str, dialect: DateDialect
) -> dict[str, Any]:
    """Render every date parameter *dialect* names in *args* to its wire form.

    Returns a new mapping; *args* is not modified. Each value is formatted
    once, here, so the chunk planner and every chunk see the same string
    (ADR 0014).
    """
    date_only = collection in dialect.date_only_collections
    out = dict(args)
    for key in dialect.params & args.keys():
        out[key] = _format_api_dates(
            args[key],
            date=date_only and key not in dialect.always_datetime,
            name=key,
            dialect=dialect,
        )
    return out
