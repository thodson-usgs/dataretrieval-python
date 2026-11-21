"""An empty result's schema fetch runs in ``finalize``; its failure must resume."""

import httpx
import pytest

from dataretrieval import waterdata
from dataretrieval.interruptions import ServiceInterrupted

_IDS = ["USGS-01646500", "USGS-01646501"]
_EMPTY = {"type": "FeatureCollection", "features": [], "numberReturned": 0}
_SCHEMA = {"properties": {"id": {}, "value": {}, "monitoring_location_id": {}}}


def test_schema_503_after_empty_chunks_is_resumable(httpx_mock):
    """Two empty chunks succeed, then ``/schema`` returns 503: the getter raises a
    resumable ``ServiceInterrupted`` holding the chunks, and ``resume()`` succeeds
    once the schema is served, without re-fetching the chunks."""
    schema_up = False
    chunk_requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/schema"):
            return httpx.Response(200 if schema_up else 503, json=_SCHEMA)
        chunk_requests.append(request)
        return httpx.Response(200, json=_EMPTY)

    httpx_mock.add_callback(respond)

    with waterdata.parallel_chunks(2):
        with pytest.raises(ServiceInterrupted) as excinfo:
            waterdata.get_daily(monitoring_location_id=_IDS)

    call = excinfo.value.call
    assert call is not None
    assert call.completed_chunks == 2
    fetched = len(chunk_requests)
    assert fetched == 2

    schema_up = True
    df, _ = call.resume()
    assert df.empty
    assert "value" in df.columns
    assert len(chunk_requests) == fetched


def test_schema_404_after_empty_chunks_is_not_wrapped(httpx_mock):
    """A non-transient schema failure keeps its own type; only transients resume."""
    from dataretrieval.exceptions import HTTPError
    from dataretrieval.interruptions import FanOutInterrupted

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/schema"):
            return httpx.Response(404, json=_SCHEMA)
        return httpx.Response(200, json=_EMPTY)

    httpx_mock.add_callback(respond)
    with pytest.raises(HTTPError) as excinfo:
        waterdata.get_daily(monitoring_location_id=_IDS[0])
    assert not isinstance(excinfo.value, FanOutInterrupted)
