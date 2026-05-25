"""Joint URL-byte chunking for the Water Data OGC getters.

A Water Data query has several chunkable axes: every multi-value list
parameter (sites, parameter codes, …) plus the cql-text ``filter``,
which splits along its top-level OR clauses. Any of them can fan the
URL past the server's ~8 KB byte limit. ``ChunkPlan`` picks a fan-out
for each axis that minimizes total sub-requests under the URL budget;
``ChunkedCall`` iterates the joint cartesian product so every
sub-request URL fits. Requests that already fit get a trivial
single-step plan — ``ChunkedCall`` has one code path either way.

Concurrency: when ``API_USGS_CONCURRENT`` is set to an integer N > 1
(or the literal ``unbounded``), ``multi_value_chunked`` fans the plan
out across ``N`` async coroutines sharing one ``httpx.AsyncClient``
instead of issuing sub-requests serially. ``N=1`` forces the
synchronous path. The default (16) is the server-friendly sweet
spot; higher values can trip USGS burst-protection 5xx in practice.
The wrapper falls back to the serial path (with a ``UserWarning``)
when an asyncio event loop is already running (Jupyter / IPython /
async apps) or when no async fetch sibling is wired into the
decorator.

Retries: each sub-request is retried on a transient failure (429,
5xx, connect/read timeout) with exponential backoff + full jitter,
honoring a server ``Retry-After`` when present. ``API_USGS_RETRIES``
sets the cap (default 4; ``0`` disables). A ``Retry-After`` longer
than the per-call ceiling isn't slept off inline — it escalates to
the resumable interruption below so a multi-minute quota-window
reset doesn't block the call.

Interruption: any mid-stream transient failure (429, 5xx) surfaces
as a ``ChunkInterrupted`` subclass — ``QuotaExhausted`` for 429,
``ServiceInterrupted`` for 5xx. The exception carries ``.call``, a
``ChunkedCall`` handle that owns the already-completed sub-request
state (sparse-indexed on the parallel path, contiguous-prefix on
the serial path). Call ``.call.resume()`` once the underlying
condition clears; only the still-pending sub-requests are
re-issued, via the serial sync path. ``Retry-After`` (when the
server sets it) is surfaced on the exception as ``.retry_after``.

Dedup: list-axis chunks don't overlap; filter-axis chunks can, so
``_combine_chunk_frames`` dedupes by feature ``id``. ``properties``,
``bbox``, date intervals, ``limit``, ``skip_geometry``, and
``filter``/``filter_lang`` themselves are never sliced as list axes
(the filter is partitioned along its top-level OR axis instead).
"""

from __future__ import annotations

import asyncio
import copy
import functools
import itertools
import math
import os
import random
import sys
import time
import warnings
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar
from urllib.parse import quote_plus

import httpx
import pandas as pd

from dataretrieval.utils import HTTPX_DEFAULTS

from . import _progress
from .filters import (
    _check_numeric_filter_pitfall,
    _is_chunkable,
    _split_top_level_or,
)

# Empirically the API replies HTTP 414 above ~8200 bytes of full URL —
# matches nginx's default ``large_client_header_buffers`` of 8 KB. 8000
# leaves ~200 bytes for request-line framing and proxy variance.
_WATERDATA_URL_BYTE_LIMIT = 8000

# Default rule: any list-shaped kwarg with >1 element is chunked across
# sub-requests — each chunk becomes a comma-joined sub-list in the URL.
# The OGC getters expose ~90 such list-shaped params (IDs, codes,
# statuses, ...), all chunkable, so it's shorter to enumerate the
# exceptions than to maintain an allowlist that grows with the API.
# Exceptions, by reason:
#   - response shape: ``properties`` defines the columns; sharding
#                      would yield different schemas per chunk.
#   - structured:      ``bbox`` is a fixed 4-element coord tuple.
#   - intervals:       date/time ranges are not enumerable sets.
#   - handled elsewhere: ``filter`` becomes its own axis in
#                         ``_extract_axes`` (joiner ``" OR "``);
#                         comma-joining CQL clauses would emit
#                         malformed expressions.
#   - scalar by contract: ``limit``, ``skip_geometry``, ``filter_lang``
#                          — a list value would be a type-erasure smuggle.
_NEVER_CHUNK = frozenset(
    {
        "properties",
        "bbox",
        "datetime",
        "last_modified",
        "begin",
        "begin_utc",
        "end",
        "end_utc",
        "time",
        "filter",
        "filter_lang",
        "limit",
        "skip_geometry",
    }
)

# Response header USGS uses to advertise remaining hourly quota.
_QUOTA_HEADER = "x-ratelimit-remaining"

# Environment variable that controls async fan-out concurrency. Read
# at call time (not import) so test patches via ``monkeypatch.setenv``
# take effect. The default (16) is the server-friendly sweet spot:
# higher values trip the upstream into 5xx burst-protection in
# practice. Set to ``1`` to force the serial sync path, set to
# ``unbounded`` for no per-call cap (use sparingly — you own the
# upstream-burst risk).
_CONCURRENCY_ENV = "API_USGS_CONCURRENT"
_CONCURRENCY_DEFAULT = 16
_CONCURRENCY_UNBOUNDED = "unbounded"


def _read_concurrency_env() -> int | None:
    """
    Resolve the ``API_USGS_CONCURRENT`` env var to a parallelism cap.

    Returns
    -------
    int or None
        ``1`` for the serial sync path; an integer >1 for bounded
        parallelism; ``None`` to disable the per-call cap entirely
        (``unbounded`` keyword). Unset → default of
        ``_CONCURRENCY_DEFAULT``.
    """
    raw = os.environ.get(_CONCURRENCY_ENV)
    if raw is None:
        return _CONCURRENCY_DEFAULT
    raw = raw.strip()
    if raw == "":
        return _CONCURRENCY_DEFAULT
    if raw.lower() == _CONCURRENCY_UNBOUNDED:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_CONCURRENCY_ENV} must be a positive integer or "
            f"'{_CONCURRENCY_UNBOUNDED}'; got {raw!r}."
        ) from exc
    if value < 1:
        raise ValueError(
            f"{_CONCURRENCY_ENV} must be >= 1 (got {value}); use "
            f"'{_CONCURRENCY_UNBOUNDED}' to disable the cap."
        )
    return value


# Retry-with-backoff for transient sub-request failures (429 / 5xx /
# connect-read timeouts). The env var is read at call time so test
# ``monkeypatch.setenv`` takes effect; the timing constants are
# module-level so power users (and tests) can ``monkeypatch.setattr``
# them. Defaults: 4 retries, 0.5s base doubling under full jitter up to
# a 30s per-attempt ceiling, and honor a server ``Retry-After`` up to
# 60s before escalating to a resumable interruption instead.
_RETRIES_ENV = "API_USGS_RETRIES"
_RETRIES_DEFAULT = 4
_RETRY_BASE_BACKOFF = 0.5
_RETRY_MAX_BACKOFF = 30.0
_RETRY_AFTER_CAP = 60.0


def _read_retries_env() -> int:
    """
    Resolve the ``API_USGS_RETRIES`` env var to a max-retry count.

    Returns
    -------
    int
        Number of retries after the first attempt; ``0`` disables
        retrying. Unset/blank → ``_RETRIES_DEFAULT``.
    """
    raw = os.environ.get(_RETRIES_ENV)
    if raw is None or raw.strip() == "":
        return _RETRIES_DEFAULT
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{_RETRIES_ENV} must be a non-negative integer (got {raw!r})."
        ) from exc
    if value < 0:
        raise ValueError(f"{_RETRIES_ENV} must be >= 0 (got {value}).")
    return value


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry-with-backoff config for transient sub-request failures.

    An immutable value object that owns the *timing* decisions; the
    exception taxonomy ("is this worth retrying at all?") lives in
    :func:`_retryable`. Backoff is exponential with **full jitter**
    (:func:`random.uniform` over ``[0, ceiling]``) so the concurrent
    fan-out's retries don't re-burst in lockstep. A server ``Retry-After``
    hint, when present, overrides the computed backoff — unless it exceeds
    :attr:`retry_after_cap`, in which case retrying stops and the failure
    surfaces as a resumable :class:`ChunkInterrupted` (a multi-minute
    quota-window reset shouldn't block the call inline).

    Attributes
    ----------
    max_retries : int
        Retries attempted after the first try; ``0`` disables retrying.
    base_backoff : float
        Seconds; the jitter ceiling for the first retry, doubled each
        subsequent attempt.
    max_backoff : float
        Upper bound on any single attempt's backoff ceiling.
    retry_after_cap : float
        Largest ``Retry-After`` (seconds) honored inline; longer hints
        escalate to a resumable interruption.
    """

    max_retries: int = _RETRIES_DEFAULT
    base_backoff: float = _RETRY_BASE_BACKOFF
    max_backoff: float = _RETRY_MAX_BACKOFF
    retry_after_cap: float = _RETRY_AFTER_CAP

    @classmethod
    def from_env(cls) -> RetryPolicy:
        """Build a policy, resolving ``max_retries`` from ``API_USGS_RETRIES``."""
        return cls(max_retries=_read_retries_env())

    def should_retry(self, attempt: int, retry_after: float | None) -> bool:
        """Whether a just-failed ``attempt`` (1-based) warrants another try.

        A ``Retry-After`` longer than ``retry_after_cap`` is *not* slept
        off inline — it returns ``False`` so the failure escalates to a
        resumable interruption instead of blocking the call for minutes.
        """
        if attempt > self.max_retries:
            return False
        return retry_after is None or retry_after <= self.retry_after_cap

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait before retry ``attempt`` (1-based)."""
        if retry_after is not None:
            return retry_after
        ceiling = min(self.max_backoff, self.base_backoff * 2 ** (attempt - 1))
        return random.uniform(0.0, ceiling)


# Default for direct ``ChunkedCall`` / ``ChunkPlan.execute`` construction
# (and tests): no retrying. The production decorator path explicitly passes
# ``RetryPolicy.from_env()`` so retries are on by default there.
_NO_RETRY = RetryPolicy(max_retries=0)


# Client shared across all sub-requests of a single chunked call so
# paginated-loop helpers downstream (``_walk_pages``) reuse one
# connection pool across the whole fan-out. ``None`` when not inside a
# chunked call — paginated helpers fall back to their own short-lived
# client in that case.
_chunked_client: ContextVar[httpx.Client | None] = ContextVar(
    "_chunked_client", default=None
)

# Async sibling of ``_chunked_client``. Published by
# ``_publish_async_client`` during ``_fan_out_async`` so async
# paginated-loop helpers reuse one ``httpx.AsyncClient`` (and its
# connection pool) across every concurrent sub-request of a single
# chunked call.
_chunked_async_client: ContextVar[httpx.AsyncClient | None] = ContextVar(
    "_chunked_async_client", default=None
)


@contextmanager
def _publish_client(client: httpx.Client) -> Iterator[None]:
    """
    Make ``client`` visible to :func:`get_active_client` for the
    duration of the ``with`` block via the ``_chunked_client``
    ContextVar. Wraps the set/reset token dance so callers don't have to.
    """
    token = _chunked_client.set(client)
    try:
        yield
    finally:
        _chunked_client.reset(token)


@contextmanager
def _publish_async_client(client: httpx.AsyncClient) -> Iterator[None]:
    """
    Make ``client`` visible to :func:`get_active_async_client` for the
    duration of the ``with`` block. Async sibling of
    :func:`_publish_client`.
    """
    token = _chunked_async_client.set(client)
    try:
        yield
    finally:
        _chunked_async_client.reset(token)


def get_active_client() -> httpx.Client | None:
    """
    Return the chunker's currently-published sync client, or ``None``.

    Public accessor for the ``_chunked_client`` ContextVar so
    sibling modules (notably :func:`dataretrieval.waterdata.utils._client_for`)
    don't have to reach into the private ContextVar directly.

    Returns
    -------
    httpx.Client or None
        The client published by :func:`_publish_client` if currently
        inside a :class:`ChunkedCall` ``resume`` block; ``None`` otherwise.
    """
    return _chunked_client.get()


def get_active_async_client() -> httpx.AsyncClient | None:
    """
    Return the chunker's currently-published async client, or ``None``.

    Async sibling of :func:`get_active_client`. Used by async
    paginated-loop helpers to reuse the per-call AsyncClient pool.
    """
    return _chunked_async_client.get()


# Separators the two axis kinds use to join their atoms back into
# URL text. List axes comma-join values (``site=USGS-A,USGS-B``); the
# filter axis OR-joins clauses (``filter=a='1' OR a='2'``).
_LIST_SEP = ","
_OR_SEP = " OR "

_FetchOnce = Callable[[dict[str, Any]], tuple[pd.DataFrame, httpx.Response]]
_FetchOnceAsync = Callable[
    [dict[str, Any]], Awaitable[tuple[pd.DataFrame, httpx.Response]]
]


class _RetryableTransportError(RuntimeError):
    """
    Base for typed HTTP transport failures the chunker recognizes as
    transient.

    Raised by :func:`dataretrieval.waterdata.utils._raise_for_non_200`
    and walked by :func:`_classify_chunk_error`. One subclass per
    recoverable HTTP status family (429 → :class:`RateLimited`,
    5xx → :class:`ServiceUnavailable`); ``ChunkedCall`` wraps them as
    resumable :class:`ChunkInterrupted` subclasses.

    Parameters
    ----------
    message : str
        Human-readable error message.
    retry_after : float, optional
        Seconds to wait before retrying, parsed from the
        ``Retry-After`` response header.

    Attributes
    ----------
    retry_after : float or None
        Seconds to wait before retrying, parsed from the
        ``Retry-After`` response header. ``None`` when the header was
        absent or unparseable.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimited(_RetryableTransportError):
    """
    A USGS Water Data API request was rejected with HTTP 429.

    Exposed as a typed exception so callers (notably the multi-value
    chunker) can detect rate-limit failures via ``isinstance`` instead
    of string-matching error messages.
    """


class ServiceUnavailable(_RetryableTransportError):
    """
    A USGS Water Data API request was rejected with HTTP 5xx.

    Surfaced as a typed exception (parallel to :class:`RateLimited`)
    so ``ChunkedCall`` can treat transient server failures as
    resumable interruptions rather than fatal programmer errors.
    """


class RequestTooLarge(ValueError):
    """
    No chunking plan fits the URL byte limit.

    Raised when even the smallest reducible plan (every list axis at
    singleton chunks and the filter at one clause per sub-request)
    still exceeds the server's byte limit. Shrink the input lists,
    simplify the filter, or split the call manually.
    """


class ChunkInterrupted(RuntimeError):
    """
    Base class for mid-stream chunk failures whose completed work is
    preserved and resumable.

    A ``ChunkInterrupted`` subclass means: a sub-request failed, but
    ``ChunkedCall`` still owns whatever completed successfully before
    the failure. Call ``self.call.resume()`` to pick up where the
    failure stopped you — only still-pending sub-requests are
    re-issued.

    Subclasses describe *why* ``ChunkedCall`` stopped so callers can
    pick a retry policy: :class:`QuotaExhausted` for 429 (wait for the
    rate-limit window), :class:`ServiceInterrupted` for 5xx (wait for
    the upstream to recover). The ``.call`` handle is the same object
    across every interruption of a single chunked call — frames
    accumulate across retries.

    Attributes
    ----------
    call : ChunkedCall or None
        Resumable handle into the ``ChunkedCall`` that raised this
        exception. ``None`` only on hand-constructed exceptions (test
        fixtures), where ``.call``-derived accessors degrade to
        empty/``None``.
    retry_after : float or None
        Seconds the server suggested waiting (``Retry-After`` header).
        ``None`` when the server gave no hint.
    completed_chunks : int
        Number of sub-requests successfully completed before the failure.
    total_chunks : int
        Total sub-requests in the plan.
    partial_frame : pandas.DataFrame
        Combined frame of work completed by the moment this exception
        was raised. Snapshot at raise time — does NOT advance on a
        later ``call.resume()`` (use ``exc.call.partial_frame`` for
        the live view).
    partial_response : httpx.Response or None
        Aggregated response covering the completed sub-requests at
        raise time; ``None`` if nothing had completed yet. Same
        snapshot semantics as ``partial_frame``.

    Examples
    --------
    Retry on any transient interruption, honoring the server's
    ``Retry-After`` hint when present and falling back to a fixed wait
    otherwise. Each new interruption keeps the already-completed work
    intact — only the still-pending sub-requests are re-issued.

    .. code-block:: python

        import time
        from dataretrieval.waterdata import get_daily
        from dataretrieval.waterdata.chunking import ChunkInterrupted

        try:
            df, md = get_daily(monitoring_location_id=long_list_of_sites)
        except ChunkInterrupted as exc:
            while True:
                time.sleep(exc.retry_after or 5 * 60)
                try:
                    df, md = exc.call.resume()
                    break
                except ChunkInterrupted as next_exc:
                    exc = next_exc
    """

    # Subclasses override with a ``str.format`` template; the format
    # call sees ``completed_chunks`` and ``total_chunks`` as kwargs.
    _MESSAGE_TEMPLATE: ClassVar[str] = (
        "Chunked request interrupted after {completed_chunks}/"
        "{total_chunks} sub-requests; call .call.resume() to continue."
    )

    def __init__(
        self,
        *,
        completed_chunks: int,
        total_chunks: int,
        call: ChunkedCall | None = None,
        retry_after: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        message = self._MESSAGE_TEMPLATE.format(
            completed_chunks=completed_chunks, total_chunks=total_chunks
        )
        if cause is not None:
            cause_msg = str(cause) or type(cause).__name__
            message = f"{message} Cause: {type(cause).__name__}: {cause_msg}"
        super().__init__(message)
        self.completed_chunks = completed_chunks
        self.total_chunks = total_chunks
        self.call = call
        self.retry_after = retry_after
        # Snapshot partial state at raise time so the exception's view
        # stays stable across later ``call.resume()`` advances; the
        # live view lives on ``call.partial_frame``/``.partial_response``.
        # ``partial_frame`` gets a defensive ``.copy()`` because
        # ``_combine_chunk_frames`` may return a chunk frame verbatim
        # in the single-completed-chunk fast path; ``partial_response``
        # already comes via ``copy.copy`` from ``_combine_chunk_responses``.
        if call is None:
            self.partial_frame: pd.DataFrame = pd.DataFrame()
            self.partial_response: httpx.Response | None = None
        else:
            self.partial_frame = call.partial_frame.copy()
            self.partial_response = call.partial_response


class QuotaExhausted(ChunkInterrupted):
    """
    A sub-request returned HTTP 429 — the per-key rate-limit window
    is exhausted. Subclass of :class:`ChunkInterrupted`.

    The completed sub-requests are preserved on ``.call``; once the
    rate-limit window resets, ``.call.resume()`` re-issues only the
    still-pending work. ``partial_frame`` holds what completed
    before the 429.
    """

    _MESSAGE_TEMPLATE = (
        "HTTP 429 after {completed_chunks}/{total_chunks} sub-requests; "
        "catch QuotaExhausted (or ChunkInterrupted) to access "
        ".partial_frame or .call.resume() once the rate-limit "
        "window has rolled over."
    )


class ServiceInterrupted(ChunkInterrupted):
    """
    A sub-request returned HTTP 5xx — the upstream service failed
    transiently. Subclass of :class:`ChunkInterrupted`.

    The completed sub-requests are preserved on ``.call``; once the
    upstream recovers, ``.call.resume()`` resumes only the
    still-pending work.
    """

    _MESSAGE_TEMPLATE = (
        "Service error after {completed_chunks}/{total_chunks} "
        "sub-requests; catch ServiceInterrupted (or ChunkInterrupted) "
        "and call .call.resume() once the upstream service recovers."
    )


def _request_bytes(req: httpx.Request) -> int:
    """
    Return the total bytes of an httpx request: URL + body.

    GET routes have empty ``.content`` and reduce to URL length. POST
    routes (CQL2 JSON body) need body bytes — the URL stays short
    regardless of payload, so URL-only sizing would underestimate the
    request and skip chunking when it's needed.

    Parameters
    ----------
    req : httpx.Request
        The request to size.

    Returns
    -------
    int
        ``len(str(req.url)) + len(req.content)``. ``httpx.URL`` doesn't
        support ``len()`` directly, so the str-coercion is required.
    """
    return len(str(req.url)) + len(req.content)


def _safe_request_bytes(
    build_request: Callable[..., httpx.Request],
    args: dict[str, Any],
    url_limit: int,
) -> int:
    """
    Size a candidate sub-request, treating ``httpx.InvalidURL`` as
    "still too large".

    ``httpx.URL`` enforces a hard 64 KB cap per URL component
    (``MAX_URL_LENGTH``) and raises ``httpx.InvalidURL`` for anything
    bigger. We report ``url_limit + 1`` on overflow so the greedy
    halving loop in :meth:`ChunkPlan._plan` keeps shrinking the
    largest axis until ``httpx.Request`` can be constructed at all.

    Parameters
    ----------
    build_request : Callable[..., httpx.Request]
        Factory that turns a kwargs dict into a sized request.
    args : dict[str, Any]
        Per-sub-request kwargs to pass through to ``build_request``.
    url_limit : int
        The chunker's byte budget; returned + 1 on overflow.

    Returns
    -------
    int
        Real byte count when the request builds, otherwise
        ``url_limit + 1`` so the planner's "too large" branch keeps
        halving.
    """
    try:
        req = build_request(**args)
    except httpx.InvalidURL:
        return url_limit + 1
    return _request_bytes(req)


def _safe_elapsed(response: httpx.Response) -> timedelta:
    """
    Read ``response.elapsed``, falling back to ``timedelta(0)`` when
    the attribute hasn't been populated.

    httpx only writes ``.elapsed`` when a response is closed through
    its normal transport path. ``MockTransport`` (used by
    ``pytest-httpx``) and hand-constructed ``httpx.Response`` objects
    leave the attribute unset, so accessing it raises ``RuntimeError``.
    Combining responses across chunks needs a defined duration, so we
    treat the missing attribute as zero elapsed.
    """
    try:
        return response.elapsed
    except RuntimeError:
        return timedelta(0)


def _set_response_url(response: httpx.Response, url: str | httpx.URL) -> None:
    """
    Overwrite the URL surfaced by a response without back-propagating
    the change into any aliased original.

    On real ``httpx.Response`` instances ``.url`` is a read-only
    property that resolves through the bound request; rather than
    mutate the existing request's URL (which would be visible through
    any shallow copy that shares the same ``.request``), we replace
    the response's request with a fresh :class:`httpx.Request` carrying
    the new URL. On lightweight test mocks ``.url`` is a plain
    writable attribute — that path is tried first.
    """
    try:
        response.url = url  # type: ignore[misc]
    except AttributeError:
        target = httpx.URL(str(url))
        try:
            old = response.request
        except RuntimeError:
            # No request bound (some hand-built httpx.Response fixtures);
            # synthesize a minimal one to hold the URL.
            response.request = httpx.Request("GET", target)
            return
        response.request = httpx.Request(
            method=old.method, url=target, headers=old.headers
        )


@dataclass(frozen=True)
class _Axis:
    """
    A single chunkable axis of one user-level request — a list of
    atomic units and the separator that joins them in the URL.

    Both multi-value list parameters (``sites=[...]``, joiner ``","``)
    and the cql-text ``filter`` (split on top-level ``OR``, joiner
    ``" OR "``) fit this shape, so a single greedy halving loop in
    ``ChunkPlan._plan`` handles both — no need for two separate
    algorithms.

    Attributes
    ----------
    arg_key : str
        The args-dict key this axis substitutes back into when a
        sub-request is rendered.
    atoms : tuple of str
        The smallest indivisible units along this axis (one site, one
        OR-clause, …). A "chunk" is a contiguous slice of ``atoms``.
    joiner : str
        Separator placed between atoms when they are joined back into
        URL text — ``","`` for list axes, ``" OR "`` for the filter
        axis.
    """

    arg_key: str
    atoms: tuple[str, ...]
    joiner: str

    def chunk_bytes(self, chunk: list[str]) -> int:
        """
        Return the URL-encoded byte count this chunk contributes when
        substituted into the request.

        ``quote_plus`` is faithful to what the real URL builder
        produces, so values containing characters that expand under URL
        encoding (``%``, ``+``, ``/``, ``&``, …) can't be mis-ranked.

        Parameters
        ----------
        chunk : list of str
            A contiguous slice of ``self.atoms``.

        Returns
        -------
        int
            Length of ``quote_plus(self.joiner.join(chunk))``.
        """
        return len(quote_plus(self.joiner.join(map(str, chunk))))

    def render(self, chunk: list[str]) -> Any:
        """
        Convert a chunk into the form the URL builder expects.

        List axes yield a fresh list of atoms (``build_request`` will
        comma-join); the filter axis yields a pre-joined string (CQL
        doesn't take a list).

        Parameters
        ----------
        chunk : list of str
            A contiguous slice of ``self.atoms``.

        Returns
        -------
        list of str or str
            ``list(chunk)`` for list axes, ``self.joiner.join(chunk)``
            for the filter axis.
        """
        return list(chunk) if self.joiner == _LIST_SEP else self.joiner.join(chunk)


def _extract_axes(args: dict[str, Any]) -> list[_Axis]:
    """
    Build the chunkable-axis set from a request's args.

    Multi-value list params with more than one element each become an
    axis. The cql-text filter (when chunkable and split into more than
    one top-level OR-clause) becomes one too. Anything in
    ``_NEVER_CHUNK`` is excluded except ``filter`` itself, which is
    handled separately so its atoms are clauses not characters.

    Parameters
    ----------
    args : dict[str, Any]
        The user-level request kwargs (the same dict that would be
        passed to ``build_request``).

    Returns
    -------
    list[_Axis]
        Zero or more axes in insertion order: list axes first (one
        per eligible kwarg, in ``args`` order), then the filter axis
        if present.
    """
    axes: list[_Axis] = []
    for key, value in args.items():
        if key in _NEVER_CHUNK:
            continue
        if isinstance(value, (list, tuple)) and len(value) > 1:
            axes.append(_Axis(arg_key=key, atoms=tuple(value), joiner=_LIST_SEP))

    filter_expr = args.get("filter")
    if _is_chunkable(filter_expr, args.get("filter_lang")):
        _check_numeric_filter_pitfall(filter_expr)
        clauses = _split_top_level_or(filter_expr)
        if len(clauses) >= 2:
            axes.append(_Axis(arg_key="filter", atoms=tuple(clauses), joiner=_OR_SEP))
    return axes


class ChunkPlan:
    """
    Strategy for issuing one user-level request as a sequence of
    sub-requests whose URLs each fit ``url_limit``.

    Constructing a plan *is* planning:
    ``ChunkPlan(args, build_request, url_limit)`` extracts the
    chunkable axes, runs greedy halving on the biggest chunk across
    all axes, and stores the result.

    Passthrough requests (no chunkable axes, or already fitting) are
    represented as a trivial plan with empty ``axes`` / ``chunks`` and
    ``total == 1``; :meth:`iter_sub_args` yields the original args
    unchanged so the ``ChunkedCall`` loop is the same shape either
    way.

    Parameters
    ----------
    args : dict[str, Any]
        The user-level request kwargs.
    build_request : Callable[..., httpx.Request]
        Factory that turns a kwargs dict into a sized httpx request,
        e.g. ``_construct_api_requests``.
    url_limit : int
        Byte budget for the request (URL + body).

    Attributes
    ----------
    args : dict
        The original user-level args this plan was built for. Bound to
        the plan so :meth:`iter_sub_args` is self-contained.
    axes : list[_Axis]
        The chunkable axes of ``args``: each multi-value list
        parameter, plus the cql-text filter (if any) split on top-level
        OR. Empty in the passthrough case.
    chunks : dict[str, list[list[str]]]
        Per-axis partition: ``chunks[axis.arg_key]`` is the list of
        atom-sublists this axis is split into. Empty in passthrough.
    canonical_url : str or None
        URL of the user's original (un-chunked) request, used to
        overwrite a chunked response's ``.url`` so ``BaseMetadata``
        reflects the full query. ``None`` on the passthrough path
        and when no buildable URL exists.

    Raises
    ------
    RequestTooLarge
        If the request needs chunking but even the singleton plan
        doesn't fit ``url_limit``.
    """

    def __init__(
        self,
        args: dict[str, Any],
        build_request: Callable[..., httpx.Request],
        url_limit: int,
    ) -> None:
        self.args = args
        self.axes: list[_Axis] = []
        self.chunks: dict[str, list[list[str]]] = {}
        self.canonical_url: str | None = None

        axes = _extract_axes(args)
        # No chunkable axes → skip ``build_request`` entirely; the
        # common Water Data call shape shouldn't pay for an unused
        # request prep on the passthrough hot path. ``fetch_once``
        # will run with the user's args verbatim; if that produces
        # an over-budget URL, the server (or httpx itself) rejects.
        if not axes:
            return

        # Constructing the initial request can itself trip
        # ``httpx.InvalidURL`` (URL > 64 KB) — that's the canonical
        # "needs chunking" signal, so swallow it and proceed to plan.
        # When the unchunked URL does build, preserve it as
        # ``canonical_url`` so ``BaseMetadata.url`` echoes the user's
        # original query verbatim; only fall back to a worst-case
        # sub-request URL when the URL itself can't be constructed.
        try:
            initial_request = build_request(**args)
        except httpx.InvalidURL:
            initial_request = None

        if initial_request is not None:
            self.canonical_url = str(initial_request.url)
            if _request_bytes(initial_request) <= url_limit:
                return

        self.axes = axes
        self.chunks = {axis.arg_key: [list(axis.atoms)] for axis in axes}
        self._plan(build_request, url_limit)

        if self.canonical_url is None:
            # Original URL was un-constructable (httpx.InvalidURL); fall
            # back to the worst-case sub-request URL so
            # ``BaseMetadata.url`` still surfaces something
            # informative. If even that overflows, leave canonical_url
            # as None (set above) and let the response's own URL stand.
            with suppress(httpx.InvalidURL):
                self.canonical_url = str(build_request(**self._worst_case_args()).url)

    def _plan(
        self,
        build_request: Callable[..., httpx.Request],
        url_limit: int,
    ) -> None:
        """
        Greedy-halve the biggest chunk across all axes until the
        worst-case sub-request URL fits ``url_limit``. Mutates
        ``self.chunks`` in place; treats list axes and the filter axis
        uniformly — each is just a list of atoms joined by its axis's
        separator.

        Raises
        ------
        RequestTooLarge
            If even the singleton plan (every axis at one atom per
            chunk) still exceeds ``url_limit``.
        """
        while True:
            worst = self._worst_case_args()
            if _safe_request_bytes(build_request, worst, url_limit) <= url_limit:
                return

            biggest_axis: _Axis | None = None
            biggest_idx = -1
            biggest_size = -1
            for axis in self.axes:
                for idx, chunk in enumerate(self.chunks[axis.arg_key]):
                    if len(chunk) <= 1:
                        continue
                    size = axis.chunk_bytes(chunk)
                    if size > biggest_size:
                        biggest_axis, biggest_idx, biggest_size = axis, idx, size

            if biggest_axis is None:
                raise RequestTooLarge(
                    f"Request exceeds {url_limit} bytes (URL + body) at the "
                    f"smallest reducible plan (every axis at one atom per "
                    f"sub-request). Reduce input sizes, shorten or simplify "
                    f"the filter, or split the call manually."
                )
            axis_chunks = self.chunks[biggest_axis.arg_key]
            chunk = axis_chunks[biggest_idx]
            mid = len(chunk) // 2
            axis_chunks[biggest_idx : biggest_idx + 1] = [chunk[:mid], chunk[mid:]]

    def _worst_case_args(self) -> dict[str, Any]:
        """
        Args dict representing the largest sub-request the current
        ``self.chunks`` partition will issue — each axis's longest
        (by URL-encoded bytes) chunk rendered back in.
        """
        out = dict(self.args)
        for axis in self.axes:
            worst = max(self.chunks[axis.arg_key], key=axis.chunk_bytes)
            out[axis.arg_key] = axis.render(worst)
        return out

    @property
    def total(self) -> int:
        """
        Total sub-request count: product of per-axis chunk counts.

        Returns
        -------
        int
            ``1`` for the passthrough plan, otherwise the cartesian
            product of ``len(chunks[ax.arg_key])`` across all axes.
        """
        return math.prod((len(self.chunks[ax.arg_key]) for ax in self.axes), start=1)

    def iter_sub_args(self) -> Iterator[dict[str, Any]]:
        """
        Yield substituted args for each sub-request, in deterministic
        order — cartesian product over axes in extraction order.

        The same plan yields the same sub-args sequence on every
        invocation, so resume is well-defined.

        Yields
        ------
        dict[str, Any]
            A copy of ``self.args`` with each axis's current chunk
            substituted under its ``arg_key``.
        """
        if not self.axes:
            yield dict(self.args)
            return
        chunk_lists = [self.chunks[ax.arg_key] for ax in self.axes]
        for combo in itertools.product(*chunk_lists):
            sub_args = dict(self.args)
            for axis, chunk in zip(self.axes, combo):
                sub_args[axis.arg_key] = axis.render(chunk)
            yield sub_args

    def execute(
        self, fetch_once: _FetchOnce, retry_policy: RetryPolicy = _NO_RETRY
    ) -> tuple[pd.DataFrame, httpx.Response]:
        """
        Run the plan and return the combined ``(frame, response)``.

        Thin wrapper around ``ChunkedCall(self, fetch_once).resume()``;
        see :class:`ChunkedCall` for the per-sub-request semantics.

        Parameters
        ----------
        fetch_once : Callable
            Function that issues a single sub-request, given the
            substituted args dict, and returns ``(frame, response)``.
        retry_policy : RetryPolicy, optional
            Per-sub-request retry-with-backoff policy. Defaults to
            :data:`_NO_RETRY`; the decorator passes ``RetryPolicy.from_env()``.

        Returns
        -------
        df : pandas.DataFrame
            Combined data from every successful sub-request.
        response : httpx.Response
            Aggregated response (canonical URL, last page's headers,
            cumulative elapsed time).

        Raises
        ------
        ChunkInterrupted
            On a mid-stream transient failure
            (:class:`QuotaExhausted` for 429,
            :class:`ServiceInterrupted` for 5xx). The resumable handle
            is on ``exc.call``.
        """
        return ChunkedCall(self, fetch_once, retry_policy).resume()


def _classify_chunk_error(
    exc: BaseException,
) -> tuple[type[ChunkInterrupted], float | None] | None:
    """
    Classify a fetch error as a known transient (resumable) failure.

    Walks the ``__cause__`` chain of ``exc`` looking for a known typed
    transport failure. Returns the matching ``ChunkInterrupted``
    subclass and any ``Retry-After`` hint, or ``None`` if the error is
    not a recognized transient — in which case ``ChunkedCall``
    re-raises rather than wrapping (programmer errors and unknown
    failures shouldn't masquerade as resumable).

    Parameters
    ----------
    exc : BaseException
        The exception raised by a sub-request.

    Returns
    -------
    tuple[type[ChunkInterrupted], float or None] or None
        ``(interrupted_class, retry_after)`` for recognized transient
        failures; ``None`` otherwise.

    Notes
    -----
    ``_walk_pages`` re-wraps mid-pagination failures as
    ``RuntimeError`` with the typed transport exception linked as
    ``__cause__``, so this function must walk the chain rather than
    just ``isinstance`` the top-level exception.

    Bare ``httpx.HTTPError`` (``ConnectError``, ``TimeoutException``,
    etc.) and ``httpx.InvalidURL`` (server-supplied cursor URL too
    long, oversize follow-up) are also treated as transport failures
    and wrapped as :class:`ServiceInterrupted` — these don't inherit
    from ``RuntimeError`` (and ``InvalidURL`` doesn't even inherit
    from ``HTTPError``), so without explicit handling they would
    escape the chunker's catch with no resumable handle.
    """
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, RateLimited):
            return QuotaExhausted, cur.retry_after
        if isinstance(cur, ServiceUnavailable):
            return ServiceInterrupted, cur.retry_after
        if isinstance(cur, (httpx.HTTPError, httpx.InvalidURL)):
            return ServiceInterrupted, None
        cur = cur.__cause__
    return None


def _retryable(exc: BaseException) -> tuple[bool, float | None]:
    """
    Decide whether ``exc`` is a transient worth an automatic retry.

    Narrower than :func:`_classify_chunk_error`: it retries rate limits
    (429), service errors (5xx), and genuine transport transients
    (:class:`httpx.TransportError` — ``ConnectError``, ``ReadTimeout``, …)
    but NOT :class:`httpx.InvalidURL` (a too-long server cursor URL won't
    fix on retry, though it stays *resumable*). Walks the ``__cause__``
    chain because ``_walk_pages`` re-wraps mid-pagination failures as
    ``RuntimeError``.

    Returns
    -------
    tuple[bool, float or None]
        ``(retryable, retry_after)`` — the server ``Retry-After`` hint
        (seconds) when the transient carried one, else ``None``.
    """
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, (RateLimited, ServiceUnavailable)):
            return True, cur.retry_after
        if isinstance(cur, httpx.TransportError):
            return True, None
        cur = cur.__cause__
    return False, None


# Sleep hooks, indirected through module globals so tests can
# ``monkeypatch.setattr`` them to no-ops instead of waiting for real
# backoff. Production uses the stdlib calls.
_SLEEP = time.sleep
_ASLEEP = asyncio.sleep


def _note_retry(attempt: int, wait: float) -> None:
    """Surface an imminent retry on the active progress reporter, if any."""
    reporter = _progress.current()
    if reporter is not None:
        reporter.note_retry(attempt=attempt, wait=wait)


def _retry_sync(
    fn: Callable[[], tuple[pd.DataFrame, httpx.Response]],
    policy: RetryPolicy,
) -> tuple[pd.DataFrame, httpx.Response]:
    """
    Call ``fn`` with bounded retry-with-backoff on transient failures.

    On a non-retryable error, or once ``policy`` is exhausted (or the
    server's ``Retry-After`` is too long to absorb inline), the last
    exception propagates unchanged so the caller's existing handling wraps
    it as a resumable :class:`ChunkInterrupted`.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised unless retryable
            retryable, retry_after = _retryable(exc)
            attempt += 1
            if not retryable or not policy.should_retry(attempt, retry_after):
                raise
            delay = policy.backoff(attempt, retry_after)
            _note_retry(attempt, delay)
            _SLEEP(delay)


async def _retry_async(
    afn: Callable[[], Awaitable[tuple[pd.DataFrame, httpx.Response]]],
    policy: RetryPolicy,
) -> tuple[pd.DataFrame, httpx.Response]:
    """Async sibling of :func:`_retry_sync` (awaits :func:`asyncio.sleep`)."""
    attempt = 0
    while True:
        try:
            return await afn()
        except Exception as exc:  # noqa: BLE001 — re-raised unless retryable
            retryable, retry_after = _retryable(exc)
            attempt += 1
            if not retryable or not policy.should_retry(attempt, retry_after):
                raise
            delay = policy.backoff(attempt, retry_after)
            _note_retry(attempt, delay)
            await _ASLEEP(delay)


def _combine_chunk_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate per-chunk frames, dropping empties and deduping by ``id``.

    Parameters
    ----------
    frames : list[pandas.DataFrame]
        One frame per completed sub-request.

    Returns
    -------
    pandas.DataFrame
        The concatenated, deduplicated result. Empty when every input
        frame is empty.

    Notes
    -----
    ``_get_resp_data`` returns a plain ``pd.DataFrame()`` on empty
    responses; concatenating it with real ``GeoDataFrame``s downgrades
    the result to plain ``DataFrame`` and strips geometry/CRS, so
    empties are dropped first. Dedup on the pre-rename feature ``id``
    keeps overlapping user OR-clauses from producing duplicate rows
    across chunks.

    Dedup is restricted to rows whose ``id`` is non-null. ``pandas``
    treats NaN==NaN as a duplicate for ``drop_duplicates``, so a
    blanket call would collapse every id-less row into a single one —
    silent data loss if any chunk emits features without an
    ``id`` field.
    """
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        # Preserve the frame type (GeoDataFrame vs DataFrame) of the
        # input even when every chunk is empty — ``_get_resp_data``
        # returns ``gpd.GeoDataFrame()`` on empty geopd responses, and
        # returning a plain ``pd.DataFrame()`` here would downgrade
        # the type a downstream ``pd.concat([result, geo_page])`` to a
        # plain DataFrame and strip geometry/CRS.
        return frames[0] if frames else pd.DataFrame()
    if len(non_empty) == 1:
        # Single-completed-chunk fast path. Return a copy so callers
        # who treat ``ChunkedCall.partial_frame`` as a fresh result
        # (the property docstring says "live; recomputed per access")
        # don't accidentally mutate ``_chunks[0][0]`` in place.
        return non_empty[0].copy()
    combined = pd.concat(non_empty, ignore_index=True)
    if "id" in combined.columns:
        has_id = combined["id"].notna()
        if has_id.all():
            combined = combined.drop_duplicates(subset="id", ignore_index=True)
        elif has_id.any():
            # Mixed: dedupe only the id-bearing rows; preserve id-less
            # rows verbatim (their order relative to id-bearing rows
            # may shift, which is acceptable — dedup can't be id-keyed
            # for rows without an id).
            id_rows = combined[has_id].drop_duplicates(subset="id")
            no_id_rows = combined[~has_id]
            combined = pd.concat([id_rows, no_id_rows], ignore_index=True)
    return combined


def _combine_chunk_responses(
    responses: list[httpx.Response], canonical_url: str | None
) -> httpx.Response:
    """
    Fold per-sub-request responses into a single aggregated response.

    For a multi-response input, returns a shallow copy of
    ``responses[0]`` with ``.headers`` set to the last response's (so
    ``x-ratelimit-remaining`` reflects current state), ``.elapsed`` set
    to total wall-clock across every response, and ``.url`` set to the
    canonical original-query URL (when supplied) so ``BaseMetadata``
    reflects the user's full request rather than the first chunk.

    For a single-response input with no canonical-URL override,
    ``responses[0]`` is returned unchanged to skip the copy on the
    passthrough hot path.

    Parameters
    ----------
    responses : list[httpx.Response]
        One response per completed sub-request, in execution order.
    canonical_url : str or None
        URL of the unchunked original request. ``None`` skips the URL
        override — used by the passthrough path (``fetch_once``'s
        response already carries the original-query URL) and by the
        worst-case overflow path (no buildable canonical URL exists).

    Returns
    -------
    httpx.Response
        A shallow copy of the first response with aggregated
        ``headers``, ``elapsed``, and ``url``. The function is
        idempotent (the input responses' ``headers`` / ``elapsed`` /
        ``url`` are never mutated), so it's safe to call repeatedly
        via :attr:`ChunkedCall.partial_response` during error
        inspection or resume retries. ``headers`` on the returned
        object is a fresh ``httpx.Headers``, so mutations there don't
        back-propagate into any chunk's underlying response.
    """
    if len(responses) == 1 and canonical_url is None:
        return responses[0]

    # ``copy.copy`` lets repeated calls re-sum elapsed from scratch
    # rather than re-mutating ``responses[0]`` in place. The headers
    # dict is then rewrapped in a fresh ``httpx.Headers`` so the
    # aggregate's headers don't share identity with — or leak mutations
    # back into — any underlying response on ``ChunkedCall._chunks``.
    head = copy.copy(responses[0])
    if len(responses) > 1:
        head.headers = httpx.Headers(responses[-1].headers)
        head.elapsed = sum(
            (_safe_elapsed(r) for r in responses[1:]),
            start=_safe_elapsed(responses[0]),
        )
    else:
        head.headers = httpx.Headers(responses[0].headers)
    if canonical_url is not None:
        _set_response_url(head, canonical_url)
    return head


class ChunkedCall:
    """
    Stateful handle for a chunked call.

    Holds the in-flight state (per-sub-request frames and responses)
    and exposes a single :meth:`resume` entry point that drives the
    call from wherever it is to completion — used both for the first
    invocation (from :meth:`ChunkPlan.execute`) and for subsequent
    retries after a :class:`ChunkInterrupted`.

    A ``ChunkedCall`` is created internally when a :class:`ChunkPlan`
    executes; callers reach it via :attr:`ChunkInterrupted.call` on
    the exception raised by a mid-stream failure.

    :meth:`resume` is idempotent: it iterates
    :meth:`ChunkPlan.iter_sub_args` (deterministic order) and skips
    any index whose result is already in ``self._chunks``. The
    completion set is a sparse ``dict[int, (df, response)]`` so the
    parallel path can record scattered completions (e.g. indices
    [0, 2, 5] after siblings [1, 3, 4] failed) and a subsequent
    ``resume`` only re-issues the missing indices — via the serial
    sync ``fetch_once`` path.

    Parameters
    ----------
    plan : ChunkPlan
        The chunking plan to execute.
    fetch_once : Callable
        Function that issues a single sub-request, given the
        substituted args dict, and returns ``(frame, response)``.

    Attributes
    ----------
    plan : ChunkPlan
        The plan being driven (read-only after construction).
    fetch_once : Callable
        The per-sub-request fetch function.
    partial_frame : pandas.DataFrame
        Combined frame of completed sub-requests (live; recomputed per
        access).
    partial_response : httpx.Response or None
        Aggregated response with canonical URL restored, or ``None``
        when nothing has completed yet (live; recomputed per access).
    """

    def __init__(
        self,
        plan: ChunkPlan,
        fetch_once: _FetchOnce,
        retry_policy: RetryPolicy = _NO_RETRY,
    ) -> None:
        self.plan = plan
        self.fetch_once = fetch_once
        self.retry_policy = retry_policy
        # Completed (frame, response) pairs keyed by sub-args index.
        # Sparse so the parallel fan-out path can record scattered
        # completions (e.g. indices [0, 2, 5] when 1/3/4 failed) and a
        # subsequent ``resume()`` only re-issues the missing indices.
        # On the serial path this fills contiguously from 0.
        self._chunks: dict[int, tuple[pd.DataFrame, httpx.Response]] = {}

    def record(self, index: int, pair: tuple[pd.DataFrame, httpx.Response]) -> None:
        """Record a completed sub-request's ``(frame, response)`` pair
        under its sub-args index. Used by both the serial loop in
        :meth:`resume` and the parallel fan-out in
        :func:`_fan_out_async` so the completion set stays
        encapsulated."""
        self._chunks[index] = pair

    def wrap_failure(self, exc: BaseException) -> ChunkInterrupted | None:
        """Build the matching :class:`ChunkInterrupted` carrying this
        call when ``exc`` is a recognized transient transport failure;
        return ``None`` for unrecognized failures so the caller can
        re-raise. Encapsulates the
        ``classify → instantiate-with-call-state`` recipe so
        :class:`ChunkedCall`'s private fields stay private."""
        classification = _classify_chunk_error(exc)
        if classification is None:
            return None
        interrupted_class, retry_after = classification
        return interrupted_class(
            completed_chunks=len(self._chunks),
            total_chunks=self.plan.total,
            call=self,
            retry_after=retry_after,
            cause=exc,
        )

    @property
    def completed_chunks(self) -> int:
        return len(self._chunks)

    def _ordered_chunks(self) -> list[tuple[pd.DataFrame, httpx.Response]]:
        return [self._chunks[i] for i in sorted(self._chunks)]

    @property
    def partial_frame(self) -> pd.DataFrame:
        """
        Concatenated, deduplicated frame of sub-requests that have
        completed so far.

        Live — recomputed on each access so it reflects current state
        across resume attempts.

        Returns
        -------
        pandas.DataFrame
            Combined frame of completed sub-requests, or an empty
            ``DataFrame`` when nothing has completed.
        """
        if not self._chunks:
            return pd.DataFrame()
        return _combine_chunk_frames([frame for frame, _ in self._ordered_chunks()])

    @property
    def partial_response(self) -> httpx.Response | None:
        """
        Aggregated response with the canonical URL restored to the
        user's full original query.

        Live — recomputed on each access.

        Returns
        -------
        httpx.Response or None
            Aggregated response when at least one sub-request has
            completed, ``None`` otherwise.
        """
        if not self._chunks:
            return None
        return _combine_chunk_responses(
            [resp for _, resp in self._ordered_chunks()], self.plan.canonical_url
        )

    def resume(self) -> tuple[pd.DataFrame, httpx.Response]:
        """
        Drive the chunked call to completion via the sync ``fetch_once``.

        Opens one ``httpx.Client`` for the run and publishes it on
        the ``_chunked_client`` ``ContextVar`` so paginated-loop
        helpers downstream (``_walk_pages``) reuse the same connection
        pool across every sub-request instead of handshaking fresh on
        each. The client is closed when ``resume`` returns or raises;
        a follow-up ``resume`` call (after a ``ChunkInterrupted``)
        opens a new one.

        Idempotent: only sub-requests whose index isn't already in
        ``self._chunks`` are re-issued. Sub-args order matches
        :meth:`ChunkPlan.iter_sub_args` and is deterministic, so a
        parallel-mode partial completion (sparse indices) resumes
        correctly via the sync path.

        Returns
        -------
        df : pandas.DataFrame
            Combined data from every successful sub-request.
        response : httpx.Response
            Aggregated response (canonical URL, last page's headers,
            cumulative elapsed time).

        Raises
        ------
        ChunkInterrupted
            On a mid-stream transient failure
            (:class:`QuotaExhausted` for 429,
            :class:`ServiceInterrupted` for 5xx). The resumable handle
            is on ``exc.call`` — wait for the underlying condition to
            clear and call ``exc.call.resume()`` again.
        """
        with httpx.Client(**HTTPX_DEFAULTS) as client, _publish_client(client):
            reporter = _progress.current()
            if reporter is not None:
                reporter.set_chunks(self.plan.total)
            for i, sub_args in enumerate(self.plan.iter_sub_args()):
                if i in self._chunks:
                    continue
                if reporter is not None:
                    reporter.start_chunk(i + 1)
                self._issue(i, sub_args)
            ordered = self._ordered_chunks()
            frames = [frame for frame, _ in ordered]
            responses = [resp for _, resp in ordered]
            return (
                _combine_chunk_frames(frames),
                _combine_chunk_responses(responses, self.plan.canonical_url),
            )

    def _issue(self, index: int, sub_args: dict[str, Any]) -> None:
        """
        Issue one sub-request and record its ``(frame, response)`` pair
        under ``index``.

        On failure, classify the exception and either wrap it as a
        resumable :class:`ChunkInterrupted` carrying this call, or
        re-raise it unchanged to preserve its type. Catches
        ``RuntimeError`` (the layer's typed contract:
        :class:`RateLimited`, :class:`ServiceUnavailable`, or the
        mid-pagination wrapper), :class:`httpx.HTTPError`
        (transport-level failures like ``ConnectError`` /
        ``TimeoutException``), and :class:`httpx.InvalidURL` (which
        inherits directly from ``Exception``, not ``HTTPError``); all
        three feed :func:`_classify_chunk_error`.
        """
        try:
            chunk = _retry_sync(lambda: self.fetch_once(sub_args), self.retry_policy)
        except (RuntimeError, httpx.HTTPError, httpx.InvalidURL) as exc:
            interrupted = self.wrap_failure(exc)
            if interrupted is None:
                raise
            raise interrupted from exc
        self.record(index, chunk)


async def _fan_out_async(
    plan: ChunkPlan,
    fetch_once: _FetchOnce,
    fetch_async: _FetchOnceAsync,
    *,
    max_concurrent: int | None,
    retry_policy: RetryPolicy = _NO_RETRY,
) -> tuple[pd.DataFrame, httpx.Response]:
    """
    Execute ``plan`` concurrently under one shared
    :class:`httpx.AsyncClient`.

    The fan-out preserves the same resumability contract the serial
    :class:`ChunkedCall` path provides:

    * **Resumable interruptions.** Sub-requests run under
      ``asyncio.gather`` with ``return_exceptions=True`` so completed
      sub-requests survive a sibling's transient failure. On a
      recognized transient (:class:`RateLimited`,
      :class:`ServiceUnavailable`) a :class:`ChunkInterrupted`
      subclass is raised with ``.call`` set to a
      :class:`ChunkedCall` carrying the sparse completed sub-args;
      ``exc.call.resume()`` re-issues only the unfinished ones via
      the sync ``fetch_once`` path.

    In-flight sub-requests are capped by an
    :class:`asyncio.Semaphore`; ``max_concurrent=None`` ("unbounded")
    uses ``sys.maxsize`` so every call site can take the same
    ``async with semaphore`` path. The shared client is published on
    :data:`_chunked_async_client` so async paginated-loop helpers
    reuse its connection pool.

    Parameters
    ----------
    plan : ChunkPlan
        Pre-built plan whose sub-args sequence drives the fan-out.
    fetch_once : Callable
        Sync per-sub-request fetcher. Used to build the resumable
        :class:`ChunkedCall` returned via ``ChunkInterrupted.call``;
        never invoked by this function directly.
    fetch_async : Callable
        Async per-sub-request fetcher returning ``(df, response)``.
    max_concurrent : int or None
        Maximum in-flight sub-requests. ``None`` disables the cap.

    Returns
    -------
    df : pandas.DataFrame
        Combined data from every sub-request.
    response : httpx.Response
        Aggregated response (canonical URL, last sub-request's
        headers, cumulative elapsed time).

    Raises
    ------
    ChunkInterrupted
        On a transient sub-request failure. ``.call`` is a
        :class:`ChunkedCall` holding the sparse completed sub-requests;
        ``.call.resume()`` re-issues the unfinished ones serially.
    """
    sub_args_list = list(plan.iter_sub_args())

    # ``httpx.Limits()`` defaults to ``max_connections=100`` — at
    # higher concurrency the pool would silently bottleneck the
    # fan-out behind the connection cap. Match it to the semaphore,
    # or ``None`` for truly unbounded.
    limits = httpx.Limits(
        max_connections=max_concurrent, max_keepalive_connections=max_concurrent
    )
    # ``sys.maxsize`` stands in for "unbounded": ``asyncio.Semaphore``
    # only decrements a counter, never preallocates slots.
    semaphore = asyncio.Semaphore(max_concurrent or sys.maxsize)
    call = ChunkedCall(plan, fetch_once, retry_policy)

    async with httpx.AsyncClient(limits=limits, **HTTPX_DEFAULTS) as client:
        with _publish_async_client(client):
            reporter = _progress.current()
            if reporter is not None:
                reporter.set_chunks(plan.total)

            async def track(
                offset: int, args: dict[str, Any]
            ) -> tuple[pd.DataFrame, httpx.Response]:
                """One sub-request (with retry) + record + progress tick.

                The retry loop runs *inside* the semaphore, so a chunk
                backing off holds its slot — effective concurrency shrinks
                under throttling instead of re-bursting against it.
                """
                async with semaphore:
                    result = await _retry_async(lambda: fetch_async(args), retry_policy)
                call.record(offset, result)
                if reporter is not None:
                    reporter.start_chunk(call.completed_chunks)
                return result

            # Dispatch every sub-request concurrently. ``return_exceptions``
            # keeps completed pairs after a sibling fails, so partial state
            # stays recoverable via ``ChunkedCall.resume()``. Failure
            # precedence:
            #   1. Cancellation / interrupt signals (CancelledError,
            #      KeyboardInterrupt, SystemExit — non-Exception) propagate
            #      unmodified; wrapping them as a transient would swallow the
            #      user's stop signal.
            #   2. Recognized transients wrap as ChunkInterrupted so the user
            #      gets a resumable handle even when a non-transient failure
            #      landed earlier in submission order.
            #   3. Otherwise re-raise the first failure, preserving its type.
            results = await asyncio.gather(
                *(track(i, args) for i, args in enumerate(sub_args_list)),
                return_exceptions=True,
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            for exc in failures:
                if not isinstance(exc, Exception):
                    raise exc
            for exc in failures:
                if (interrupted := call.wrap_failure(exc)) is not None:
                    raise interrupted from exc
            if failures:
                raise failures[0]

    ordered = call._ordered_chunks()
    return (
        _combine_chunk_frames([df for df, _ in ordered]),
        _combine_chunk_responses([resp for _, resp in ordered], plan.canonical_url),
    )


def multi_value_chunked(
    *,
    build_request: Callable[..., httpx.Request],
    fetch_async: _FetchOnceAsync | None = None,
    url_limit: int | None = None,
) -> Callable[[_FetchOnce], _FetchOnce]:
    """
    Decorate a fetch function to transparently chunk over-budget requests.

    Splits multi-value list params and cql-text filters across
    sub-requests so each fits the URL byte limit. Builds a
    :class:`ChunkPlan` and runs it: passthrough requests are a trivial
    single-step plan, so the decorated function has one code path
    either way.

    When ``API_USGS_CONCURRENT`` resolves to a parallelism greater than
    1 (the default), the decorator routes execution through
    :func:`_fan_out_async` over the provided ``fetch_async``. The
    wrapper falls back to the synchronous :class:`ChunkedCall` path
    (with a ``UserWarning``) when ``fetch_async`` wasn't wired or
    when an asyncio event loop is already running (Jupyter / IPython /
    async apps where ``asyncio.run`` would raise ``RuntimeError``).

    Parameters
    ----------
    build_request : Callable[..., httpx.Request]
        Factory that turns a kwargs dict into a sized httpx request,
        e.g. ``_construct_api_requests``. Called during planning to
        measure each candidate plan.
    fetch_async : Callable, optional
        Async sibling of the decorated sync fetcher. Used when
        ``API_USGS_CONCURRENT`` resolves to >1; if omitted, the
        wrapper warns and stays on the serial path.
    url_limit : int, optional
        Byte budget for the request (URL + body). When ``None``
        (default), the module-level ``_WATERDATA_URL_BYTE_LIMIT`` is
        resolved at call time so test patches via
        ``monkeypatch.setattr`` take effect.

    Returns
    -------
    Callable
        A decorator that wraps a ``fetch_once(args) -> (df, response)``
        callable into one that accepts the same shape but executes the
        underlying plan transparently.

    Raises
    ------
    RequestTooLarge
        If no plan can fit ``url_limit``.
    ChunkInterrupted
        On a mid-execution 429 (:class:`QuotaExhausted`) or 5xx
        (:class:`ServiceInterrupted`). See :class:`ChunkedCall` for
        the resume semantics.

    See Also
    --------
    ChunkPlan : Planning shape (axes, partitioning, passthrough).
    ChunkedCall : Per-sub-request execution and resume semantics.
    """

    def decorator(fetch_once: _FetchOnce) -> _FetchOnce:
        @functools.wraps(fetch_once)
        def wrapper(
            args: dict[str, Any],
        ) -> tuple[pd.DataFrame, httpx.Response]:
            limit = _WATERDATA_URL_BYTE_LIMIT if url_limit is None else url_limit
            plan = ChunkPlan(args, build_request, limit)
            concurrency = _read_concurrency_env()
            retry_policy = RetryPolicy.from_env()

            # Trivial plans and explicit opt-outs stay on the sync
            # path; ``_execute_in_parallel`` owns the rest of the
            # serial/parallel decision (async wiring, running loop).
            if plan.total <= 1 or concurrency == 1:
                return plan.execute(fetch_once, retry_policy)
            return _execute_in_parallel(
                plan, fetch_once, fetch_async, concurrency, retry_policy
            )

        return wrapper

    return decorator


def _execute_in_parallel(
    plan: ChunkPlan,
    fetch_once: _FetchOnce,
    fetch_async: _FetchOnceAsync | None,
    concurrency: int | None,
    retry_policy: RetryPolicy = _NO_RETRY,
) -> tuple[pd.DataFrame, httpx.Response]:
    """
    Run ``plan`` on the parallel async path, falling back to the
    serial sync path when the runtime can't host an event loop.

    Falls back (with a one-time :class:`UserWarning`) when:

    * ``fetch_async`` wasn't wired into the decorator, or
    * an asyncio event loop is already running (Jupyter / IPython
      kernels, async apps — ``asyncio.run`` would raise).

    Otherwise opens a fresh event loop via :func:`asyncio.run` and
    drives :func:`_fan_out_async`.
    """
    if fetch_async is None:
        warnings.warn(
            f"{_CONCURRENCY_ENV} is set to {concurrency} but this "
            f"call site has no async fetch sibling wired; falling "
            f"back to the serial path. Either set "
            f"{_CONCURRENCY_ENV}=1 to silence this warning or pass "
            f"fetch_async= to @multi_value_chunked.",
            UserWarning,
            stacklevel=3,
        )
        return plan.execute(fetch_once, retry_policy)
    if _running_event_loop() is not None:
        warnings.warn(
            "Detected a running asyncio event loop; the parallel "
            f"chunker path cannot run inside one. Falling back to "
            f"the serial path. Set {_CONCURRENCY_ENV}=1 to silence "
            f"this warning.",
            UserWarning,
            stacklevel=3,
        )
        return plan.execute(fetch_once, retry_policy)
    return asyncio.run(
        _fan_out_async(
            plan,
            fetch_once,
            fetch_async,
            max_concurrent=concurrency,
            retry_policy=retry_policy,
        )
    )


def _running_event_loop() -> asyncio.AbstractEventLoop | None:
    """Return the active asyncio event loop, or ``None`` when none."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
