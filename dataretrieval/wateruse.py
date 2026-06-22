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
        location="stateCd:RI",
        startdate="2020-01",
        timeres="monthly",
    )

"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from dataretrieval.exceptions import DataRetrievalError
from dataretrieval.utils import (
    HTTPX_DEFAULTS,
    BaseMetadata,
    _default_headers,
    _get,
    _raise_for_status,
    to_str,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

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

# Page responses carry the HUC12 identifier in this column; it must stay a
# string so leading zeros (e.g. "010900020502") survive the round trip.
_HUC12_COLUMN = "huc12_id"


def get_wateruse(
    model: str,
    variable: str | Iterable[str] | None = None,
    location: str | None = None,
    timeres: str | None = None,
    startdate: str | None = None,
    enddate: str | None = None,
    intersection: str = "overlap",
    limit: int = 600,
    ssl_check: bool = True,
) -> tuple[pd.DataFrame, BaseMetadata]:
    """Get USGS water-use data from the NWDC web service.

    Retrieves modeled water-use estimates from the USGS National Water
    Availability Assessment Data Companion. A single ``location`` is queried at
    a time; results are always returned on a HUC12 grid, in a long (tidy) frame
    with one row per HUC12 and time step. Large areas (e.g. ``"huc2:04"`` or a
    populous state) are served across multiple pages, which this function
    follows transparently and concatenates into one frame.

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
    location : string, optional
        The area to query, given as ``"<type>:<id>"``. Supported types are
        ``huc2``, ``huc4``, ``huc6``, ``huc8``, ``huc10``, ``huc12``,
        ``countyCd``, and ``stateCd`` (e.g. ``"stateCd:RI"``, ``"huc2:04"``,
        ``"countyCd:55025"``). Only one location may be queried per call.
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
        How to select HUC12s that straddle the ``location`` boundary:
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
        ...     location="stateCd:RI",
        ...     startdate="2020-01",
        ...     timeres="monthly",
        ... )

    """
    params: dict[str, Any] = {
        "format": "csv",
        "model": model,
        "variable": to_str(variable),
        "location": location,
        "timeres": timeres,
        "startdate": startdate,
        "enddate": enddate,
        "intersection": intersection,
        "limit": limit,
    }
    # Drop params the caller left unset; the service rejects empty values.
    params = {k: v for k, v in params.items() if v is not None}

    df, first_response = _fetch_all_pages(params, ssl_check=ssl_check)
    return df, BaseMetadata(first_response)


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
    return pd.concat(frames, ignore_index=True), first_response


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
