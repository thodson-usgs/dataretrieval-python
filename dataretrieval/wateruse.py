"""Retrieve USGS water-use data from the National Water Availability
Assessment Data Companion (NWDC).

The NWDC web services provide national-scale, USGS-modeled water-use data that
underlie the `USGS National Water Availability Assessment
<https://water.usgs.gov/nwaa-data/>`_. Estimates are served on a HUC12
(12-digit hydrologic unit) spatial grid and can be queried for any county,
state, or hydrologic unit. This is the modern replacement for the defunct
legacy NWIS water-use service (``nwis.get_water_use``).

Unlike the main Water Data getters (:mod:`dataretrieval.waterdata`) and NGWMN
(:mod:`dataretrieval.ngwmn`), the NWDC is a plain CSV REST service rather than
an OGC API Features collection, so this module talks to it directly instead of
delegating to the shared OGC engine. It still follows the same conventions:
shared request headers (:func:`~dataretrieval.utils._default_headers`),
the typed :class:`~dataretrieval.exceptions.DataRetrievalError` taxonomy, and a
``(DataFrame, BaseMetadata)`` return.

See https://api.water.usgs.gov/docs/nwaa-data/ for the API reference and
https://water.usgs.gov/nwaa-data/ for the catalog of available models and
variables.

Examples
--------
.. code-block:: python

    from dataretrieval import wateruse

    # Monthly public-supply withdrawals for Rhode Island, 2020 onward.
    df, md = wateruse.get_wateruse(
        model="wu-public-supply-wd",
        variable=["pswdtot", "pswdgw", "pswdsw"],
        state="RI",
        startdate="2020-01",
        timeres="monthly",
    )

"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pandas as pd

from dataretrieval.codes.states import to_state
from dataretrieval.exceptions import DataRetrievalError
from dataretrieval.utils import (
    HTTPX_DEFAULTS,
    BaseMetadata,
    _default_headers,
    _get,
    _raise_for_status,
    to_str,
)

logger = logging.getLogger(__name__)

WATERUSE_URL = "https://api.water.usgs.gov/nwaa-data/data"

#: Water-use models (categories) served by the NWDC. The catalog at
#: https://water.usgs.gov/nwaa-data/ lists the variables available within each.
MODELS = (
    "wu-public-supply-wd",  # public-supply withdrawals
    "wu-public-supply-cu",  # public-supply consumptive use
    "wu-thermoelectric",  # thermoelectric-power water use
    "wu-irrigation-wd",  # irrigation withdrawals
    "wu-irrigation-cu",  # irrigation consumptive use
)

#: Temporal resolutions: monthly, annual calendar year, annual water year.
TIME_RESOLUTIONS = ("monthly", "annualcy", "annualwy")

#: Maximum locations fetched concurrently when a list of state/county/huc
#: selectors is fanned out (one request per location). Kept conservative
#: because this module intentionally carries no request backoff/retry; the
#: NWDC tolerates this level of concurrency without rate-limit errors (verified
#: by stress test). Set ``wateruse.MAX_CONCURRENT_REQUESTS = 1`` for serial.
MAX_CONCURRENT_REQUESTS = 4

# Page responses carry the HUC12 identifier in this column; it must stay a
# string so leading zeros (e.g. "010900020502") survive the round trip.
_HUC12_COLUMN = "huc12_id"


def get_wateruse(
    model: str,
    variable: str | Iterable[str] | None = None,
    state: str | int | Iterable[str | int] | None = None,
    county: str | Iterable[str] | None = None,
    huc: str | Iterable[str] | None = None,
    timeres: str | None = None,
    startdate: str | None = None,
    enddate: str | None = None,
    intersection: str = "overlap",
    limit: int = 600,
    ssl_check: bool = True,
) -> tuple[pd.DataFrame, BaseMetadata]:
    """Get USGS water-use data from the NWDC web service.

    Retrieves modeled water-use estimates from the USGS National Water
    Availability Assessment Data Companion. The area is given as exactly one of
    ``state``, ``county``, or ``huc``; results are always returned on a HUC12
    grid, in a long (tidy) frame with one row per HUC12 and time step. Large
    areas (e.g. a whole region or a populous state) are served across multiple
    pages, which this function follows transparently and concatenates into one
    frame.

    Each selector also accepts a list of values. The NWDC queries one area per
    request, so a list is fanned out into one request per value — up to
    :data:`MAX_CONCURRENT_REQUESTS` in parallel — and the results are
    concatenated in the order given.

    Parameters
    ----------
    model : string
        Water-use category to query. See :data:`MODELS` for the available
        options (e.g. ``"wu-public-supply-wd"``). The full catalog of models
        and their variables is at https://water.usgs.gov/nwaa-data/.
    variable : string or iterable of strings, optional
        One or more variable IDs within ``model`` (e.g. ``"pswdtot"`` for total
        public-supply withdrawals, or ``["pswdgw", "pswdsw"]`` for the
        groundwater and surface-water components). Multiple variables are
        comma-joined into a single request. If omitted, the service returns its
        default variable set for the model.
    state : string, int, or iterable, optional
        One or more US states/territories to query. Each accepts a full name
        (``"Wisconsin"``), a two-letter postal code (``"WI"``), or a two-digit
        ANSI/FIPS code (``"55"`` or ``55``), mirroring
        :func:`dataretrieval.ngwmn.get_sites`.
    county : string or iterable, optional
        One or more five-digit county FIPS codes — state FIPS + county FIPS,
        e.g. ``"55025"`` for Dane County, Wisconsin.
    huc : string or iterable, optional
        One or more hydrologic unit codes. Each code's level is taken from its
        length: a 2-digit code queries a HUC2 region, 8-digit a HUC8 subbasin,
        12-digit a single HUC12, and so on (even lengths 2-12, e.g. ``"04"``,
        ``"07070005"``, ``"010900020502"``).

        Provide exactly one of ``state``, ``county``, or ``huc`` (each may be a
        single value or a list).
    timeres : string, optional
        Temporal resolution: ``"monthly"``, ``"annualcy"`` (annual, calendar
        year), or ``"annualwy"`` (annual, water year). See
        :data:`TIME_RESOLUTIONS`.
    startdate : string, optional
        Start of the query window, formatted ``"YYYY"`` for annual data or
        ``"YYYY-MM"`` for monthly data.
    enddate : string, optional
        End of the query window, in the same format as ``startdate``.
    intersection : string, optional
        How to select HUC12s that straddle the queried-area boundary:
        ``"overlap"`` (any overlap, the default) or ``"envelop"`` (fully
        enclosed).
    limit : int, optional
        Maximum number of HUC12s returned per page. Queries spanning more than
        ``limit`` HUC12s are split across pages and reassembled. Default 600.
    ssl_check : bool, optional
        If True (default), verify SSL certificates; set False to skip
        verification (e.g. behind a TLS-intercepting proxy).

    Returns
    -------
    df : ``pandas.DataFrame``
        Water-use estimates in long form: a ``huc12_id`` column (string,
        leading zeros preserved), a time column (``year_month`` for monthly
        data or ``year`` for annual data), and one value column per requested
        variable (suffixed with its unit, e.g. ``pswdtot_mgd`` for million
        gallons per day).
    md : :class:`dataretrieval.utils.BaseMetadata`
        Metadata describing the request (URL, query time, response headers).

    Raises
    ------
    ValueError
        If not exactly one of ``state``, ``county``, or ``huc`` is given, or a
        given selector is malformed (an unrecognized state, a county code that
        is not five digits, or a HUC of invalid length).
    DataRetrievalError
        On an HTTP error response, the typed subclass for the status (see
        :func:`dataretrieval.exceptions.error_for_status`); or
        :class:`~dataretrieval.exceptions.NetworkError` on a connection-level
        failure (timeout, DNS).

    Examples
    --------
    .. doctest::
        :skipif: True  # network

        >>> from dataretrieval import wateruse
        >>> df, md = wateruse.get_wateruse(
        ...     model="wu-public-supply-wd",
        ...     variable=["pswdtot", "pswdgw", "pswdsw"],
        ...     state="RI",
        ...     startdate="2020-01",
        ...     timeres="monthly",
        ... )

    """
    base_params: dict[str, Any] = {
        "format": "csv",
        "model": model,
        "variable": to_str(variable),
        "timeres": timeres,
        "startdate": startdate,
        "enddate": enddate,
        "intersection": intersection,
        "limit": limit,
    }
    # Drop params the caller left unset; the service rejects empty values.
    base_params = {k: v for k, v in base_params.items() if v is not None}

    # The NWDC queries one location per request, so fan a multi-value selector
    # out into a request per location and concatenate the results.
    locations = _resolve_locations(state, county, huc)

    def _fetch(location: str) -> tuple[pd.DataFrame, httpx.Response]:
        return _fetch_all_pages(
            {**base_params, "location": location}, ssl_check=ssl_check
        )

    if len(locations) == 1:
        # Common case: no pool, and no extra concat copy of the whole result.
        frame, response = _fetch(locations[0])
        return frame, BaseMetadata(response)

    # Fan out concurrently (bounded), preserving input order. The locations are
    # independent single requests, so a thread pool over the synchronous fetch
    # needs no shared state or backoff; ``pool.map`` re-raises the first failure.
    workers = min(len(locations), max(1, MAX_CONCURRENT_REQUESTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_fetch, locations))
    df = pd.concat([frame for frame, _ in results], ignore_index=True)
    return df, BaseMetadata(results[0][1])


# Valid HUC code lengths (digits) → the hydrologic-unit level they query.
_HUC_LENGTHS = (2, 4, 6, 8, 10, 12)


def _resolve_locations(
    state: str | int | Iterable[str | int] | None,
    county: str | Iterable[str] | None,
    huc: str | Iterable[str] | None,
) -> list[str]:
    """Build the NWDC ``location=<type>:<id>`` value(s) from the selectors.

    Exactly one of ``state`` / ``county`` / ``huc`` must be given; each may be a
    single value or a list. ``state`` is normalized to the two-letter postal
    code ``stateCd`` requires; ``county`` is a five-digit FIPS code; and a
    ``huc`` code's length selects its level (``huc2`` … ``huc12``). Returns one
    location string per value — the caller issues one request per location.
    """
    provided = [
        name
        for name, value in (("state", state), ("county", county), ("huc", huc))
        if value is not None
    ]
    if len(provided) != 1:
        raise ValueError(
            "Specify exactly one of state, county, or huc "
            f"(got: {', '.join(provided) or 'none'})."
        )

    if state is not None:
        # to_state returns a str (scalar) or list[str] (iterable); _as_list
        # normalizes both, keeping this branch the same shape as county/huc.
        locations = [
            f"stateCd:{code}" for code in _as_list(to_state(state, to="postal"))
        ]
    elif county is not None:
        locations = [f"countyCd:{_validate_county(c)}" for c in _as_list(county)]
    else:
        locations = [f"huc{len(c)}:{c}" for c in map(_validate_huc, _as_list(huc))]

    if not locations:
        raise ValueError(
            "The chosen location selector is empty; pass at least one value."
        )
    return locations


def _as_list(value: object) -> list[Any]:
    """A scalar becomes a one-element list; any non-string iterable (list,
    tuple, Series, ndarray, generator) is materialized to a list. A string is
    treated as a scalar so it isn't exploded into characters."""
    if isinstance(value, Iterable) and not isinstance(value, str):
        return list(value)
    return [value]


def _validate_county(value: object) -> str:
    """Validate and normalize a five-digit state+county FIPS code."""
    code = str(value).strip()
    if not (code.isdigit() and len(code) == 5):
        raise ValueError(
            "county must be a five-digit state+county FIPS code "
            f"(e.g. '55025'), got {value!r}."
        )
    return code


def _validate_huc(value: object) -> str:
    """Validate a HUC code (even length 2-12 digits; level set by length)."""
    code = str(value).strip()
    if not (code.isdigit() and len(code) in _HUC_LENGTHS):
        raise ValueError(
            "huc must be a hydrologic unit code of even length 2-12 digits "
            f"(e.g. '04', '07070005', '010900020502'), got {value!r}."
        )
    return code


def _fetch_all_pages(
    params: dict[str, Any], *, ssl_check: bool
) -> tuple[pd.DataFrame, httpx.Response]:
    """Fetch every page of a water-use query and concatenate the CSV bodies.

    The NWDC paginates large areas with an RFC 8288 ``Link: <...>; rel="next"``
    header (the cursor is a ``skip`` offset). The first request carries the
    query params; each subsequent page is a fully-formed URL we request bare.
    Returns the combined frame and the first page's response (for metadata).
    """
    headers = _default_headers()
    frame, first_response = _fetch_page(WATERUSE_URL, params, headers, ssl_check)
    frames = [frame]
    # Guard against a non-advancing or cyclic ``next`` cursor (a server bug
    # would otherwise spin this loop forever, accumulating frames until OOM):
    # stop if a page points back to a URL we have already fetched.
    seen: set[str] = set()
    next_url = _next_page_url(first_response)
    while next_url is not None and next_url not in seen:
        seen.add(next_url)
        frame, response = _fetch_page(next_url, None, headers, ssl_check)
        frames.append(frame)
        next_url = _next_page_url(response)
    # Avoid re-copying the (often whole) single-page result, matching the
    # per-location concat in get_wateruse.
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return df, first_response


def _fetch_page(
    url: str,
    params: dict[str, Any] | None,
    headers: dict[str, str],
    ssl_check: bool,
) -> tuple[pd.DataFrame, httpx.Response]:
    """Fetch one water-use page and parse its CSV body into a DataFrame."""
    response = _get(
        url, params=params, headers=headers, verify=ssl_check, **HTTPX_DEFAULTS
    )
    _raise_for_status(response, detail_from=_nwdc_error_detail)
    logger.debug("Requested water-use page: %s", response.url)
    try:
        frame = pd.read_csv(io.BytesIO(response.content), dtype={_HUC12_COLUMN: str})
    except pd.errors.EmptyDataError as exc:
        # NWDC normally signals "no data" with a 400 (handled above) or rows of
        # zeros, never an empty body — but keep the typed-error contract if it
        # ever returns one rather than leaking a bare pandas exception.
        raise DataRetrievalError(
            f"NWDC returned an empty response body (URL: {response.url})."
        ) from exc
    return frame, response


def _next_page_url(response: httpx.Response) -> str | None:
    """Return the absolute URL of the next page, or None if this is the last.

    Reads the standard ``Link: <...>; rel="next"`` header (parsed by httpx into
    ``response.links``). A next link served against the bare ``water.usgs.gov``
    host is normalized to the public ``api.water.usgs.gov`` gateway so the
    follow-up request reaches the API.
    """
    url = response.links.get("next", {}).get("url")
    if not url:
        return None
    return url.replace("https://water.usgs.gov", "https://api.water.usgs.gov", 1)


def _nwdc_error_detail(response: httpx.Response) -> str | None:
    """Pull the ``detail`` message out of an NWDC JSON error envelope, if any.

    The NWDC reports errors as ``{"detail": "Invalid model name: ..."}``. Passed
    to :func:`~dataretrieval.utils._raise_for_status` as ``detail_from`` so the
    service's wording surfaces in the typed error message.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("detail") if isinstance(body, dict) else None
