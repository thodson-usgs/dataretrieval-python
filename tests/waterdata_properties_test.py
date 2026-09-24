"""Live monitor: the columns a getter documents are the ones its collection has.

Some getters list their returned columns in the ``properties`` docstring, under
"Available options are:". The list is hand-written, so it goes stale when USGS
adds or removes a field. It is checked here rather than generated, because a
generated list would make the docs build depend on the live service and would
not reach ``help()``. When this fails, edit the docstring it names.
"""

import inspect

import httpx
import pytest

from dataretrieval import waterdata
from dataretrieval.waterdata.endpoints import ogc_api_url

#: Getters that list their returned columns, by collection. Written out rather
#: than discovered, so a getter that loses its list fails instead of dropping out.
_DOCUMENTED = {
    "daily": waterdata.get_daily,
    "continuous": waterdata.get_continuous,
    "time-series-metadata": waterdata.get_time_series_metadata,
}

_LABEL = "Available options are:"

#: ``time-series-metadata`` lists ``id`` in its schema and the others do not,
#: but all three accept it, so it is excluded from both sides.
_ALWAYS_REQUESTABLE = {"id"}

_TIMEOUT = 60


def _documented_properties(getter) -> set[str]:
    """The column names *getter* documents under :data:`_LABEL`.

    After ``inspect.getdoc`` dedents, the list runs from the label to the next
    unindented line, which is the next numpydoc parameter.
    """
    doc = inspect.getdoc(getter) or ""
    start = doc.find(_LABEL)
    assert start != -1, f"{getter.__name__} no longer documents its columns"

    lines = doc[start + len(_LABEL) :].splitlines()
    collected = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line.startswith("    "):
            break
        collected.append(line)
    return {
        name.strip().rstrip(".")
        for name in " ".join(collected).split(",")
        if name.strip()
    }


def _schema_properties(collection: str) -> set[str]:
    """The columns *collection* publishes in its OGC schema document."""
    response = httpx.get(
        f"{ogc_api_url()}/collections/{collection}/schema",
        params={"f": "json"},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    properties = response.json().get("properties")
    assert properties, f"{collection} published no schema properties"
    return set(properties)


@pytest.mark.live
@pytest.mark.parametrize("collection", sorted(_DOCUMENTED))
def test_documented_columns_match_the_collection_schema(collection):
    """A getter's documented column list matches what the collection publishes.

    On failure, edit the getter's ``properties`` docstring. For an added field,
    consider a named parameter too; a removed field is a breaking change and
    belongs in NEWS.
    """
    getter = _DOCUMENTED[collection]
    documented = _documented_properties(getter) - _ALWAYS_REQUESTABLE
    published = _schema_properties(collection) - _ALWAYS_REQUESTABLE

    assert documented == published, (
        f"{getter.__name__} documents the wrong columns for {collection}: "
        f"missing={sorted(published - documented)}, "
        f"stale={sorted(documented - published)}. Edit the 'Available options "
        "are:' list in its properties docstring."
    )
