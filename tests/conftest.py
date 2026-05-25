"""
Test scaffolding for the dataretrieval test suite.

* Relaxes ``pytest-httpx``'s strict-mode flags so unconsumed mocks and
  unmatched real requests don't fail the suite (matches the historical
  ``requests-mock``-style permissiveness the test code was written
  against, and keeps mocked-URL setup terse).
* Pins ``API_USGS_CONCURRENT=1`` and ``API_USGS_RETRIES=0`` for every
  test by default so the historical mocked suite stays on the
  deterministic serial chunker path and a single transient surfaces
  immediately (no backoff). Async-mode and retry tests opt in by
  re-setting the env vars inside their body via ``monkeypatch.setenv``.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Apply relaxed ``pytest-httpx`` strict-mode settings to every
    test in the suite — matches the permissive defaults the historical
    tests were written against."""
    marker = pytest.mark.httpx_mock(
        assert_all_responses_were_requested=False,
        assert_all_requests_were_expected=False,
        can_send_already_matched_responses=True,
    )
    for item in items:
        item.add_marker(marker)


@pytest.fixture
def non_mocked_hosts() -> list[str]:
    """No hosts are exempted from mocking; every HTTP call must hit
    a mock registered through the ``httpx_mock`` fixture."""
    return []


@pytest.fixture(autouse=True)
def _serial_chunker(monkeypatch):
    """Default every test to the serial, no-retry chunker path.

    Production defaults ``API_USGS_CONCURRENT`` to 16 (parallel
    fan-out) and ``API_USGS_RETRIES`` to 4, but the historical tests
    assume sequential, deterministic sub-request ordering — and they
    mock the sync ``_walk_pages`` rather than the async sibling, and
    expect a single transient to surface immediately rather than be
    retried. Pinning ``API_USGS_CONCURRENT=1`` and ``API_USGS_RETRIES=0``
    keeps the test surface focused on the planner / fetch contracts;
    async-mode and retry tests opt in by overriding the env inside
    their body.
    """
    monkeypatch.setenv("API_USGS_CONCURRENT", "1")
    monkeypatch.setenv("API_USGS_RETRIES", "0")
