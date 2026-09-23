"""Live monitors for the API version each Water Data family serves.

``waterdata/endpoints.py`` puts a version in the OGC, STAC and statistics
paths. An old version keeps answering after USGS publishes a new one, so no
other test fails when that happens. (Samples and NGWMN have no version segment.)

Two conditions are checked separately:

- **A new version is available**: the family's default no longer matches
  :data:`_DEFAULT_VERSIONS`.
- **The version the package requests stopped working**: it no longer returns
  data.

The OGC and STAC roots publish a ``self`` link naming their default version,
so the version is read from it. OGC cannot be probed, because it answers 200
with an empty body for any version segment. Statistics has no root document,
so it is probed; a missing statistics version answers 404.
"""

import re

import httpx
import pytest

from dataretrieval.waterdata import endpoints

#: The version each family serves by default, read from the live service on
#: 2026-09-22. This records what the service serves, not what the package
#: requests. Update it only after the package has moved to the new version, so
#: the test keeps failing until then.
_DEFAULT_VERSIONS = {
    "ogcapi": "v1",
    "stac": "v0",
}

#: The endpoint function that builds each family's URL, so the version checked
#: is the one the package actually requests.
_FAMILIES = {
    "ogcapi": endpoints.ogc_api_url,
    "stac": endpoints.ratings_catalog_url,
    "statistics": endpoints.statistics_api_url,
}

#: A version segment anywhere in a path: ``/v0``, ``/v12/``, ``/v1?f=json``.
_VERSION_RE = re.compile(r"/(v\d+)(?=[/?#]|$)")

#: Fail a hung request before the scheduled job's own timeout does.
_TIMEOUT = 60


def _split_version(url: str) -> tuple[str, str]:
    """Split *url* into its unversioned root and its version segment."""
    match = _VERSION_RE.search(url)
    assert match is not None, f"no version segment in {url!r}"
    return url[: match.start()] + "/", match.group(1)


def _served_version(root: str) -> str:
    """The version in *root*'s ``self`` link, e.g. ``.../ogcapi/v1?f=json``."""
    response = httpx.get(root, timeout=_TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    links = response.json().get("links") or []
    self_links = [link["href"] for link in links if link.get("rel") == "self"]
    assert self_links, f"{root} published no self link: {links}"
    return _split_version(self_links[0])[1]


@pytest.mark.live
@pytest.mark.parametrize("family", sorted(_DEFAULT_VERSIONS))
def test_service_still_serves_the_recorded_default_version(family):
    """The version a family serves by default is the one recorded here.

    On failure, move the pin in ``waterdata/endpoints.py`` to the new version
    (check its release notes for dropped or renamed fields), then update
    ``_DEFAULT_VERSIONS``.
    """
    root, _ = _split_version(_FAMILIES[family]())
    served = _served_version(root)

    assert served == _DEFAULT_VERSIONS[family], (
        f"the {family} API now serves {served} by default, not "
        f"{_DEFAULT_VERSIONS[family]}. Move the package to {served} and update "
        "_DEFAULT_VERSIONS."
    )


@pytest.mark.live
def test_statistics_has_published_no_version_beyond_the_one_we_request():
    """Statistics has no root document, so the next version is probed.

    Both the unversioned root and ``/statistics/vN`` answer 404, so the
    ``/docs`` page is what shows whether a version exists.
    """
    url = _FAMILIES["statistics"]()
    root, current = _split_version(url)
    following = f"v{int(current.removeprefix('v')) + 1}"

    assert httpx.get(f"{url}/docs", timeout=_TIMEOUT).status_code == 200, (
        f"the statistics service stopped serving {current}, which this package "
        "requests; check what replaced it."
    )

    probe = httpx.get(f"{root}{following}/docs", timeout=_TIMEOUT)
    assert probe.status_code == 404, (
        f"the statistics service now answers for {following} "
        f"(HTTP {probe.status_code}); check whether the package should move to "
        "it, and whether it publishes a root document that would let this be "
        "discovered rather than probed."
    )


@pytest.mark.live
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_the_version_this_package_requests_still_returns_data(family):
    """The version the package pins still returns data.

    Checked on content, not status: OGC answers 200 with an empty body for a
    version that does not exist.
    """
    url = _FAMILIES[family]()

    if family == "statistics":
        # No collections endpoint; its docs page is what proves it is up.
        response = httpx.get(f"{url}/docs", timeout=_TIMEOUT)
        response.raise_for_status()
        assert response.text.strip(), f"{url}/docs returned an empty body"
        return

    response = httpx.get(f"{url}/collections", timeout=_TIMEOUT)
    response.raise_for_status()
    collections = response.json().get("collections")
    assert collections, (
        f"{url}/collections returned no collections, so the "
        f"{_split_version(url)[1]} {family} API this package requests has stopped "
        "serving. Move the package to the version the service now offers."
    )
