"""Offline tests for :mod:`dataretrieval.wateruse`.

All HTTP is mocked with ``pytest-httpx``; no live calls (per AGENTS.md).
"""

import re
from urllib.parse import parse_qs, urlsplit

import httpx
import pandas as pd
import pytest

import dataretrieval
from dataretrieval import wateruse
from dataretrieval.utils import BaseMetadata
from dataretrieval.wateruse import _next_page_url, get_wateruse

# Match the NWDC endpoint regardless of query string, so assertions can drill
# into the captured params without coupling registration to param order.
WU_RE = re.compile(r"^https://api\.water\.usgs\.gov/nwaa-data/data(\?.*)?$")

# A single-page monthly CSV: two HUC12s (one with a leading zero), three months.
_CSV_PAGE = """\
huc12_id,year_month,pswdgw_mgd,pswdsw_mgd,pswdtot_mgd
010900020502,2020-01,0.0,0.8313625,0.8313625
010900020502,2020-02,0.0,0.8977986,0.8977986
180600060101,2020-01,1.5,0.5,2.0
"""

# Two pages used for pagination tests; each page is its own CSV (own header).
_CSV_P1 = """\
huc12_id,year_month,pswdtot_mgd
010900020502,2020-01,0.8313625
010900020503,2020-01,0.0
"""
_CSV_P2 = """\
huc12_id,year_month,pswdtot_mgd
010900020504,2020-01,1.25
"""


def test_get_wateruse_single_page(httpx_mock):
    """Happy path: CSV parsed to a long frame; returns (df, BaseMetadata)."""
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_PAGE)

    df, md = get_wateruse(
        model="wu-public-supply-wd",
        variable=["pswdtot", "pswdgw", "pswdsw"],
        location="stateCd:RI",
        startdate="2020-01",
        timeres="monthly",
    )

    assert isinstance(df, pd.DataFrame)
    assert isinstance(md, BaseMetadata)
    assert list(df.columns) == [
        "huc12_id",
        "year_month",
        "pswdgw_mgd",
        "pswdsw_mgd",
        "pswdtot_mgd",
    ]
    assert len(df) == 3


def test_huc12_id_kept_as_string_with_leading_zero(httpx_mock):
    """The HUC12 identifier must not be coerced to int (leading zeros matter)."""
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_PAGE)

    df, _ = get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")

    # String-typed (object or the pandas StringDtype, depending on version),
    # never coerced to int — the leading zero must survive.
    assert pd.api.types.is_string_dtype(df["huc12_id"])
    assert df["huc12_id"].iloc[0] == "010900020502"


def test_variables_are_comma_joined(httpx_mock):
    """A list of variables is sent as one comma-joined query parameter."""
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_PAGE)

    get_wateruse(
        model="wu-public-supply-wd",
        variable=["pswdtot", "pswdgw", "pswdsw"],
        location="stateCd:RI",
    )

    qs = parse_qs(urlsplit(str(httpx_mock.get_requests()[0].url)).query)
    assert qs["variable"] == ["pswdtot,pswdgw,pswdsw"]
    assert qs["format"] == ["csv"]


def test_unset_params_are_dropped(httpx_mock):
    """Params left as None are omitted (the service rejects empty values)."""
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_PAGE)

    get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")

    qs = parse_qs(urlsplit(str(httpx_mock.get_requests()[0].url)).query)
    assert "enddate" not in qs
    assert "variable" not in qs
    assert "timeres" not in qs
    # Defaulted params are still present.
    assert qs["intersection"] == ["overlap"]
    assert qs["limit"] == ["600"]


def test_pagination_follows_link_header_and_concatenates(httpx_mock):
    """Pages are followed via the ``rel="next"`` Link header and concatenated."""
    httpx_mock.add_response(
        method="GET",
        url=WU_RE,
        text=_CSV_P1,
        headers={
            "link": (
                "<https://api.water.usgs.gov/nwaa-data/data"
                '?model=wu-public-supply-wd&skip=2>; rel="next"'
            )
        },
    )
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_P2)

    df, _ = get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")

    # 2 rows from page 1 + 1 row from page 2, reindexed.
    assert len(df) == 3
    assert df["huc12_id"].tolist() == [
        "010900020502",
        "010900020503",
        "010900020504",
    ]
    assert list(df.index) == [0, 1, 2]
    assert len(httpx_mock.get_requests()) == 2
    # The second request carries the Link's ``skip`` offset, not the originals.
    second_qs = parse_qs(urlsplit(str(httpx_mock.get_requests()[1].url)).query)
    assert second_qs["skip"] == ["2"]


def test_pagination_rewrites_bare_host(httpx_mock):
    """A next link on the bare ``water.usgs.gov`` host is routed to the API."""
    httpx_mock.add_response(
        method="GET",
        url=WU_RE,
        text=_CSV_P1,
        headers={
            "link": (
                "<https://water.usgs.gov/nwaa-data/data"
                '?model=wu-public-supply-wd&skip=2>; rel="next"'
            )
        },
    )
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_P2)

    get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")

    second = httpx_mock.get_requests()[1]
    assert second.url.host == "api.water.usgs.gov"


def test_http_error_raises_typed_exception_with_detail(httpx_mock):
    """A 4xx response surfaces as a typed error carrying the NWDC ``detail``."""
    httpx_mock.add_response(
        method="GET",
        url=WU_RE,
        status_code=400,
        json={"detail": "Invalid model name: bad-model"},
    )

    with pytest.raises(dataretrieval.DataRetrievalError, match="Invalid model name"):
        get_wateruse(model="bad-model", location="stateCd:RI")


def test_empty_response_body_raises_typed_error(httpx_mock):
    """An empty 200 body becomes a typed error, not a bare pandas EmptyDataError."""
    httpx_mock.add_response(method="GET", url=WU_RE, text="")

    with pytest.raises(dataretrieval.DataRetrievalError, match="empty response"):
        get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")


def test_cyclic_next_link_terminates(httpx_mock):
    """A non-advancing/cyclic ``next`` cursor must not loop forever."""
    # Page 1 points to a "next" URL; page 2 points back to that SAME URL.
    cyclic = (
        "<https://api.water.usgs.gov/nwaa-data/data"
        '?model=wu-public-supply-wd&skip=2>; rel="next"'
    )
    httpx_mock.add_response(
        method="GET", url=WU_RE, text=_CSV_P1, headers={"link": cyclic}
    )
    httpx_mock.add_response(
        method="GET", url=WU_RE, text=_CSV_P2, headers={"link": cyclic}
    )

    df, _ = get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")

    # Fetches page 1 + the cyclic page once, then breaks on the repeat — it must
    # return (not hang) with the two pages collected.
    assert len(df) == 3
    assert len(httpx_mock.get_requests()) == 2


def test_uses_shared_default_headers(httpx_mock):
    """Requests carry the shared dataretrieval User-Agent (per _default_headers)."""
    httpx_mock.add_response(method="GET", url=WU_RE, text=_CSV_PAGE)

    get_wateruse(model="wu-public-supply-wd", location="stateCd:RI")

    sent = httpx_mock.get_requests()[0]
    assert sent.headers["User-Agent"].startswith("python-dataretrieval/")


# --- _next_page_url unit tests (no HTTP) -----------------------------------


def test_next_page_url_none_when_no_link():
    resp = httpx.Response(200, text="")
    assert _next_page_url(resp) is None


def test_next_page_url_none_when_link_has_no_next():
    resp = httpx.Response(
        200,
        text="",
        headers={"link": '<https://api.water.usgs.gov/x>; rel="prev"'},
    )
    assert _next_page_url(resp) is None


def test_next_page_url_rewrites_bare_host():
    resp = httpx.Response(
        200,
        text="",
        headers={
            "link": '<https://water.usgs.gov/nwaa-data/data?skip=600>; rel="next"'
        },
    )
    assert _next_page_url(resp) == (
        "https://api.water.usgs.gov/nwaa-data/data?skip=600"
    )


def test_next_page_url_leaves_api_host_untouched():
    url = "https://api.water.usgs.gov/nwaa-data/data?skip=600"
    resp = httpx.Response(200, text="", headers={"link": f'<{url}>; rel="next"'})
    # Must not double-prefix into ``api.api.water.usgs.gov``.
    assert _next_page_url(resp) == url


def test_module_exposes_catalog_constants():
    assert "wu-public-supply-wd" in wateruse.MODELS
    assert set(wateruse.TIME_RESOLUTIONS) == {"monthly", "annualcy", "annualwy"}
