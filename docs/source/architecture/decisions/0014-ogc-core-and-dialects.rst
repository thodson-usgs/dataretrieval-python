ADR 0014: The OGC core implements the standard; each service deviation is a dialect field
=========================================================================================

Status
------

Proposed

Context
-------

``dataretrieval.ogc`` serves two services from one code path: the Water Data
OGC API and NGWMN. ``waterdata.get_ratings`` reuses its date formatting
against a third, the Water Data STAC catalog. All three implement the same
standards -- OGC API Features, and STAC's search, which shares its
``datetime`` grammar -- and all three depart from them in different ways:

.. list-table::
   :header-rows: 1

   * - Behavior (observed 2026-10-11)
     - Water Data
     - NGWMN
     - STAC ratings
   * - Date parameters
     - ``time``, ``begin``, ``end``, ``last_modified``, and ``datetime``
     - ``datetime``
     - ``datetime``
   * - A lone duration, ``P7D``
     - accepted
     - HTTP 400
     - HTTP 400
   * - An open start, ``../2016-01-01``
     - accepted
     - HTTP 403 from the firewall
     - HTTP 403 from the firewall
   * - An empty start, ``/2016-01-01``
     - HTTP 400 (500 on ``continuous``)
     - accepted
     - accepted

ADR 0003 keeps the OGC package from *importing* a service. It says nothing
about the core *naming* one, and that is how the deviations got in. The
core listed Water Data's parameter names as "the OGC date parameters",
exempted ``last_modified`` by name from date-only rendering, passed
durations through for every service because one accepts them, and assumed
every service spells an open start ``..``. Each looked like protocol code
because it sat in the protocol package. No import rule could catch it,
since a string literal is not an import. Pull request #443 then
formatted interval strings to the one spelling the core knew, which broke
the open-start workaround the other two services needed. Its review
missed this because nothing stated the rule it broke.

``OgcDialect`` already existed as the place for per-API differences, but it
was defined by example ("which collections must be POSTed as CQL2, which
render dates date-only"). With no rule for what must go there, each
contributor decided case by case.

Decision
--------

**The core implements the standard as written.** ``dataretrieval.ogc``
implements OGC API Features and the parts of OGC API Features Part 3 (CQL2)
the adapters use. Where it formats a value, it formats it the way the
standard specifies.
``DEFAULT_DIALECT`` describes a service that implements the standard and
nothing more. A behavior the standard does not specify is off by default.

**Each deviation is one dialect field, whose default is the standard.** A
service may *extend* the standard (Water Data accepts a lone duration),
*restrict* it (NGWMN rejects ``../``), or *rename* it (Water Data filters on
``time`` where the standard says ``datetime``). Each of these is a field of the
dialect that adapter passes. The field's docstring records the observed
server behavior that justifies it, because these behaviors are facts about a
deployment, and deployments change.

**The core names no service.** The core contains no service's parameter,
collection, or column names as literals; the core reads them from the
dialect. The standard's own names (``datetime``, ``bbox``, ``limit``,
``properties``, ``filter``, ``filter-lang``) are not deviations and may appear.
A comment in the core may cite a service as evidence for why the core
tolerates a variation -- a page without ``numberReturned`` -- but the core's
behavior does not depend on which service it is talking to. The adapter that
sets a field says which service needs it.

**Fields are grouped by the concern they adapt.** A dialect is organized by
which part of the core reads it, not by which service needs it:

- ``OgcDialect.dates`` is a ``DateDialect``: which parameters carry dates,
  which collections render them date-only, which parameters keep their time of
  day regardless, how an open start is spelled, and whether a lone duration is
  accepted. ``ogc.dates`` reads only this group, so a service that is not an
  OGC Features API but shares the grammar -- the STAC ratings catalog -- can be
  given a ``DateDialect`` alone.
- ``cql2_collections`` is read by request construction.
- ``time_cols``, ``numerical_cols``, and ``sort_cols`` are read by result
  shaping.

A group is introduced when one part of the core reads several fields together.
Groups are not created to make the dialect look symmetrical.

**Values are formatted once, before the query is planned.** Date arguments are
rendered to their wire form when the getter's arguments enter the engine, so the
chunk planner sees a string rather than a two-element range. The planner
therefore needs no list of date parameters, and an unreadable date fails
before any request is sized or sent.

Consequences
------------

- Adding a third OGC service means writing its dialect, not editing the core.
  The core's defaults are correct for a service that follows the standard, so a
  new adapter starts from documented behavior rather than another service's
  quirks.
- Each deviation is recorded once, in the adapter that has it, with the
  evidence beside it. Reviewing a change to how dates are sent becomes a
  question about one service's dialect.
- A deviation the core does not yet model needs a new field before an adapter
  can use it. That costs one field and its default, and that cost is
  intentional: the alternative is a special case in shared code.
- NGWMN no longer passes a lone duration to its service, which answered it with
  HTTP 400. It now raises ``ValueError`` before sending, as ``get_ratings``
  already did. That is the one change in behavior. The ``get_ratings``
  message changes wording with it, since one message now serves both.
- ``get_ratings`` drops its own duration check and its custom message hint.
  Both are now its ``DateDialect``.
- The dialect is part of the internal OGC facade (ADR 0007), not public API,
  so renaming and regrouping its fields needs no deprecation.

Alternatives rejected:

- *Keep the deviations in the core with per-service conditionals.* This is the
  state that produced #443's regression.
- *Make the dialect a flat list of fields.* The date fields are consumed
  together by one module and are meaningful without the rest, as the ratings
  catalog shows.
- *Probe the service at runtime for what it accepts.* That adds a request to
  every call to learn a fact that changes rarely, and some of these behaviors
  come from the firewall rather than from anything the API describes.

Compliance
----------

``tests/architecture_test.py`` fails if any name a shipped dialect lists --
a date parameter, a collection, a column -- appears as a string literal in
code under ``dataretrieval/ogc``. Docstrings and comments are excluded, since
the rule binds what the code does.

Each field that records a server behavior has a ``@pytest.mark.live`` test
that sends the request the dialect prevents and asserts the service still
rejects it, and sends the request the dialect produces and asserts the service
accepts it. The scheduled live workflow therefore fails when a deployment
changes, which is when a dialect field needs revisiting.

``CONTEXT.md``'s *Dialect* entry states the rule, not only examples.

Four deviations remain in the core as recorded variances. The first is a
dialect name, so the fitness function allowlists it by module and a new one
fails the suite; the other three are not dialect names, so this list is their
record:

- ``monitoring_location_id`` is validated by name against the ``AGENCY-ID``
  form both services use (``ogc.requests``). ``waterdata.ratings`` imports the
  same check, so moving it needs a validator shared with a caller that has no
  dialect.
- ``skipGeometry`` is pygeoapi's spelling, which both services run; the
  standard has no such parameter.
- ``limit`` is capped at 50000, Water Data's maximum page size, for every
  dialect.
- An unquoted numeric literal in a CQL text filter raises
  (``filters._check_numeric_filter_pitfall``), because every Water Data
  queryable is typed as a string. The check runs in the chunk planner, which
  does not receive the dialect.

Each is removed when a third service needs a different value, or when the code
that holds it is next changed for another reason.

Notes
-----

Prompted by the review of #442 and #443. The behaviors in the Context table
were observed by sending each request to each service on 2026-10-11.
