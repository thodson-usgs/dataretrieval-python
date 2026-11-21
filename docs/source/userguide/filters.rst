.. _filters:

===========
CQL filters
===========

The Water Data getters that take ``filter`` and ``filter_lang`` pass
a server-side CQL expression to the service as the OGC ``filter`` and
``filter-lang`` query parameters. ``filter_lang`` is ``"cql-text"`` (the
default) or ``"cql-json"``.

.. code-block:: python

    from dataretrieval import waterdata

    # Two disjoint time windows in one call.
    df, md = waterdata.get_continuous(
        monitoring_location_id="USGS-02238500",
        parameter_code="00060",
        filter=(
            "(time >= '2023-06-01T12:00:00Z' AND time <= '2023-06-01T13:00:00Z') "
            "OR (time >= '2023-06-15T12:00:00Z' AND time <= '2023-06-15T13:00:00Z')"
        ),
        filter_lang="cql-text",
    )

Quote every literal
-------------------

Every queryable on the Water Data API is typed as a string, including fields
whose values look numeric (``value``, ``parameter_code`` such as ``'00060'``,
``statistic_id``). The service rejects an unquoted numeric literal with HTTP
500, so a ``cql-text`` filter that compares a field with an unquoted number
(``value >= 1000``, ``parameter_code = 60``, ``parameter_code IN (60, 61)``,
``value BETWEEN 5 AND 10``) raises ``ValueError`` before any request is sent.

Quoting the number avoids the error but gives a lexicographic comparison:
``value > '10'`` matches ``'34.52'``, and ``parameter_code = '60'`` matches
nothing because the real code is ``'00060'``. To filter on a numeric value,
fetch a wider result and reduce it in pandas.

Long filters are split automatically
------------------------------------

The service's firewall caps the size of a request. When a ``cql-text`` filter
makes the request too large, the getter splits it at each top-level ``OR`` and
sends the clauses as separate requests, then combines the results. Only
top-level ``OR`` splits without changing the answer, so a filter without one
(or one in ``cql-json``) is sent as a single request. ``OR`` inside
parentheses or inside a quoted string is not split.
