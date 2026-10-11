"""Live monitors for the date dialect fields (ADR 0014).

Each :class:`~dataretrieval.ogc.DateDialect` field that departs from the
standard default records a fact about a deployment. These tests send the
request the field prevents and assert the service still rejects it, then send
the request the field produces and assert the service accepts it. A failure
means the deployment changed and the field should be revisited.

Kept live on purpose: they are claims about the upstream services, and mocking
them would assert only that a fixture still says what we wrote.
"""

from __future__ import annotations

import httpx
import pytest

from dataretrieval.ngwmn import NGWMN_DIALECT, NGWMN_OGC_API_URL
from dataretrieval.ogc.dates import _format_api_dates
from dataretrieval.transport.http import default_headers
from dataretrieval.waterdata.endpoints import ratings_catalog_url
from dataretrieval.waterdata.ratings import _STAC_DATES
from dataretrieval.waterdata.utils import OGC_API_URL, WATERDATA_DIALECT

pytestmark = pytest.mark.live

_END = "2016-01-01T00:00:00Z"

# (service, dialect, URL, the request's other params, the date parameter)
_SERVICES = {
    "waterdata": (
        WATERDATA_DIALECT.dates,
        f"{OGC_API_URL}/collections/daily/items",
        {
            "monitoring_location_id": "USGS-01646500",
            "parameter_code": "00060",
            "skipGeometry": "true",
        },
        "time",
    ),
    "ngwmn": (
        NGWMN_DIALECT.dates,
        f"{NGWMN_OGC_API_URL}/collections/waterLevelObs/items",
        {"monitoring_location_id": "USGS-423114090161101"},
        "datetime",
    ),
    "stac": (
        _STAC_DATES,
        f"{ratings_catalog_url()}/search",
        {"bbox": "-95.0,40.0,-92.0,42.0"},
        "datetime",
    ),
}


def _status(service: str, value: str) -> int:
    _, url, params, date_param = _SERVICES[service]
    response = httpx.get(
        url,
        params={**params, date_param: value, "limit": 1, "f": "json"},
        headers=default_headers(url),
        timeout=60,
    )
    return response.status_code


@pytest.mark.parametrize("service", sorted(_SERVICES))
def test_open_start_spelling_is_still_the_one_the_service_needs(service):
    """``DateDialect.open_start``: the spelling the dialect sends is accepted,
    and the other one is still refused."""
    dialect = _SERVICES[service][0]
    sent = _format_api_dates([None, _END], dialect=dialect)
    other = "/" + _END if sent.startswith("..") else "../" + _END
    assert _status(service, sent) == 200, sent
    assert _status(service, other) >= 400, (
        f"{service} now accepts {other!r}; its open_start may be relaxed"
    )


@pytest.mark.parametrize("service", sorted(_SERVICES))
def test_duration_support_still_matches_the_dialect(service):
    """``DateDialect.durations``: a service whose dialect allows a lone
    duration still accepts one, and one whose dialect refuses it still
    rejects it."""
    dialect = _SERVICES[service][0]
    status = _status(service, "P7D")
    if dialect.durations:
        assert status == 200
    else:
        assert status >= 400, f"{service} now accepts P7D; durations may be enabled"


def test_waterdata_daily_still_filters_on_dates():
    """``date_only_collections``: ``daily`` accepts the date-only form the
    dialect sends."""
    dialect = WATERDATA_DIALECT.dates
    sent = _format_api_dates(["2024-01-01", "2024-01-10"], date=True, dialect=dialect)
    assert sent == "2024-01-01/2024-01-10"
    assert _status("waterdata", sent) == 200
