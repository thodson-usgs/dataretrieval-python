"""The HTTP status mapper is shared by non-OGC callers, so it lives in transport."""

from __future__ import annotations

import httpx
import pytest

from dataretrieval.exceptions import RateLimited


def test_status_mapper_is_a_public_transport_function() -> None:
    from dataretrieval.transport import status

    assert not status.raise_for_non_200.__name__.startswith("_")
    with pytest.raises(RateLimited):
        status.raise_for_non_200(httpx.Response(429))
