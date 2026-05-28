"""Layered configuration for the Water Data getters.

The Water Data module has a few runtime knobs — retry budget, concurrency
cap, API token, progress bar mode. Historically each was read directly from
its env var at the call site. This module gathers them behind one config
object with the conventional precedence rule used by ``git`` / ``npm`` /
``pip`` / ``cargo`` (closer to the action wins):

1. **Defaults** — the dataclass field defaults below.
2. **User config file** — ``$XDG_CONFIG_HOME/dataretrieval/config.cfg``
   (default ``~/.config/dataretrieval/config.cfg``) on Linux/macOS,
   ``%APPDATA%\\dataretrieval\\config.cfg`` on Windows.
3. **Local config file** — ``./dataretrieval.cfg`` in the current working
   directory.
4. **Environment variables** — ``API_USGS_RETRIES``,
   ``API_USGS_CONCURRENT``, ``API_USGS_PAT``, ``API_USGS_PROGRESS``.
5. **Python override** — :func:`override` (a ContextVar-scoped context
   manager) or :func:`set_config` (process-wide).

Config-file schema (stdlib INI via :mod:`configparser`)::

    [default]
    api_token = ...
    progress = auto  # on | off | auto

    [retry]
    max_retries = 4
    base_backoff = 0.5
    max_backoff = 30.0
    retry_after_cap = 60.0

    [concurrency]
    max_connections = 16  # int >= 1, or the string "unbounded"

Stdlib-only — no TOML / YAML deps. If/when pyproject.toml integration
becomes valuable (e.g. ``[tool.dataretrieval]``), adding ``tomli`` as a
conditional dep is a small mechanical follow-up.

Backward compatibility: every existing env var keeps working unchanged,
and the legacy ``RetryPolicy.from_env()`` / ``ConcurrencyPolicy.from_env()``
factories still build per-call from the layered loader.
"""

from __future__ import annotations

import configparser
import os
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# ---- env-var names (single source of truth) -------------------------------

ENV_RETRIES = "API_USGS_RETRIES"
ENV_CONCURRENT = "API_USGS_CONCURRENT"
ENV_PAT = "API_USGS_PAT"
ENV_PROGRESS = "API_USGS_PROGRESS"

# Sentinel string in ``API_USGS_CONCURRENT`` or ``concurrency.max_connections``
# meaning "no cap on simultaneous connections".
CONCURRENCY_UNBOUNDED = "unbounded"


# ---- defaults the dataclass fields ship with -----------------------------
# Values mirror the legacy ``_RETRY_*`` module constants so callers porting
# from the old shape see the same numbers.

_RETRIES_DEFAULT = 4
_RETRY_BASE_BACKOFF = 0.5
_RETRY_MAX_BACKOFF = 30.0
_RETRY_AFTER_CAP = 60.0
_CONCURRENCY_DEFAULT = 16


# ---- file locations -------------------------------------------------------

_CONFIG_BASENAME = "config.cfg"
_USER_CONFIG_SUBDIR = "dataretrieval"
_LOCAL_CONFIG_NAME = "dataretrieval.cfg"

# Section names in the INI files. ``DEFAULTS_SECTION`` carries the
# top-level scalars (api_token, progress); the policy sub-tables get their
# own sections.
DEFAULTS_SECTION = "default"
RETRY_SECTION = "retry"
CONCURRENCY_SECTION = "concurrency"


def _user_config_path() -> Path:
    """Cross-platform user config path. Honors ``XDG_CONFIG_HOME``."""
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows CI
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData/Roaming")
        return Path(base) / _USER_CONFIG_SUBDIR / _CONFIG_BASENAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / _USER_CONFIG_SUBDIR / _CONFIG_BASENAME


def _local_config_path() -> Path:
    """The local config file in the current working directory."""
    return Path.cwd() / _LOCAL_CONFIG_NAME


# ---- dataclasses ----------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Retry-with-backoff timing knobs for one sub-request.

    Frozen value object. Construct directly to override per-call, or use
    :meth:`from_env` to build from the layered config (defaults → user file
    → local file → env vars → Python override).

    Attributes
    ----------
    max_retries : int
        Maximum retries per sub-request (``0`` disables retry entirely).
    base_backoff : float
        First-retry delay in seconds; subsequent retries double under full
        jitter up to :attr:`max_backoff`.
    max_backoff : float
        Per-attempt ceiling on the slept delay (seconds).
    retry_after_cap : float
        If the server sends ``Retry-After`` greater than this (seconds),
        the failure escalates to a resumable :class:`ChunkInterrupted`
        instead of blocking the gather inline.
    """

    max_retries: int = _RETRIES_DEFAULT
    base_backoff: float = _RETRY_BASE_BACKOFF
    max_backoff: float = _RETRY_MAX_BACKOFF
    retry_after_cap: float = _RETRY_AFTER_CAP

    def __post_init__(self) -> None:
        # Catch invalid timing knobs here so a misconfiguration fails at
        # construction, not deep in a later ``time.sleep`` (ValueError on
        # a negative delay) or silently in ``asyncio.sleep`` (which
        # treats negative as zero).
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0 (got {self.max_retries}).")
        if self.base_backoff < 0 or self.max_backoff < 0 or self.retry_after_cap < 0:
            raise ValueError("retry backoff settings must be non-negative.")

    @classmethod
    def from_env(cls) -> RetryPolicy:
        """Build from the layered config (defaults → user file → local
        file → env vars → Python override)."""
        return current().retry

    def should_retry(self, attempt: int, retry_after: float | None) -> bool:
        """Whether ``attempt`` should be retried under this policy.

        Returns ``False`` if the policy is exhausted or if the server's
        ``Retry-After`` (seconds) exceeds :attr:`retry_after_cap`.
        """
        if attempt > self.max_retries:
            return False
        return retry_after is None or retry_after <= self.retry_after_cap

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait before the next retry of ``attempt``.

        Honor server ``Retry-After`` when present (already filtered by
        :meth:`should_retry` against :attr:`retry_after_cap`). Otherwise:
        exponential ``base_backoff * 2 ** (attempt - 1)`` capped at
        :attr:`max_backoff`, then full-jitter randomized in ``[0, capped]``.
        """
        if retry_after is not None:
            return retry_after
        ceiling = min(self.max_backoff, self.base_backoff * 2 ** (attempt - 1))
        return random.uniform(0.0, ceiling)


@dataclass(frozen=True)
class ConcurrencyPolicy:
    """Simultaneous-connection cap for chunked sub-request gather.

    ``max_connections=None`` means uncapped — the gather dispatches every
    pending sub-request and lets ``httpx`` open as many connections as
    needed.
    """

    max_connections: int | None = _CONCURRENCY_DEFAULT

    def __post_init__(self) -> None:
        if self.max_connections is not None and self.max_connections < 1:
            raise ValueError(
                f"max_connections must be >= 1 or None (got {self.max_connections})."
            )

    @classmethod
    def from_env(cls) -> ConcurrencyPolicy:
        """Build from the layered config — convenience for callers that
        only need the concurrency piece."""
        return current().concurrency


@dataclass(frozen=True)
class WaterDataConfig:
    """Top-level config composing every runtime knob the Water Data
    getters consult.

    Construct directly, or use :meth:`load` to build via the precedence
    layering. :func:`current` returns the in-effect config (the active
    override if one is set, else a freshly :meth:`load`-ed one).
    """

    retry: RetryPolicy = field(default_factory=RetryPolicy)
    concurrency: ConcurrencyPolicy = field(default_factory=ConcurrencyPolicy)
    api_token: str | None = None
    # ``None`` = auto-detect (TTY-driven); ``True`` / ``False`` are explicit
    # overrides. Mirrors the legacy ``API_USGS_PROGRESS`` semantics:
    # ``"on"`` / ``"true"`` / ``"1"`` → True, ``"off"`` / ``"false"`` /
    # ``"0"`` → False, anything else → ``None`` (auto).
    progress: bool | None = None

    @classmethod
    def load(cls) -> WaterDataConfig:
        """Build from the precedence chain — defaults → user file → local
        file → env vars (does NOT consult :func:`override`)."""
        merged: dict[str, Any] = {}
        for layer in (_load_user_file(), _load_local_file(), _load_env()):
            _deep_update(merged, layer)
        return _from_mapping(merged)


# ---- ContextVar-based runtime override -----------------------------------

_active: ContextVar[WaterDataConfig | None] = ContextVar(
    "waterdata_active_config", default=None
)


def current() -> WaterDataConfig:
    """Return the active :class:`WaterDataConfig` — the override set via
    :func:`override` / :func:`set_config` if any, else freshly loaded
    from the file + env precedence chain."""
    active = _active.get()
    if active is not None:
        return active
    return WaterDataConfig.load()


def set_config(config: WaterDataConfig | None) -> None:
    """Pin a config process-wide (or ``None`` to clear the pin and fall
    back to the layered loader). Test/notebook convenience — prefer
    :func:`override` for scoped use."""
    _active.set(config)


@contextmanager
def override(config: WaterDataConfig) -> Iterator[None]:
    """Pin ``config`` as the active config for the duration of the
    ``with`` block (thread-safe and async-safe via ``ContextVar``)::

        with override(WaterDataConfig(retry=RetryPolicy(max_retries=10))):
            ...  # every call inside sees the override
    """
    token = _active.set(config)
    try:
        yield
    finally:
        _active.reset(token)


# ---- file readers ---------------------------------------------------------


def _read_ini(path: Path) -> dict[str, dict[str, str]]:
    """Read an INI file; return ``{section: {key: value}}``. Missing file
    returns ``{}``. A malformed file raises :class:`configparser.Error`."""
    if not path.exists():
        return {}
    parser = configparser.ConfigParser(
        # Don't promote the ``[DEFAULT]`` section into every other section
        # (the configparser default behavior). Each section stands alone.
        default_section="__never_used__",
    )
    parser.read(path)
    return {section: dict(parser[section]) for section in parser.sections()}


def _ini_to_mapping(sections: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Convert raw INI string values into the shape ``_from_mapping`` expects.

    ``[default]`` keys become top-level entries (``api_token``, ``progress``);
    ``[retry]`` and ``[concurrency]`` become sub-tables. String values are
    parsed into the appropriate types (int / float / bool / None).
    """
    out: dict[str, Any] = {}
    # Top-level scalars from [default].
    defaults = sections.get(DEFAULTS_SECTION, {})
    if "api_token" in defaults:
        out["api_token"] = defaults["api_token"]
    if "progress" in defaults:
        parsed = _parse_progress(defaults["progress"])
        if parsed is not None or defaults["progress"].strip().lower() in ("auto", ""):
            out["progress"] = parsed
    # Retry sub-table.
    retry_raw = sections.get(RETRY_SECTION, {})
    if retry_raw:
        retry: dict[str, Any] = {}
        if "max_retries" in retry_raw:
            retry["max_retries"] = _coerce_int(retry_raw["max_retries"], "max_retries")
        for k in ("base_backoff", "max_backoff", "retry_after_cap"):
            if k in retry_raw:
                retry[k] = _coerce_float(retry_raw[k], k)
        if retry:
            out["retry"] = retry
    # Concurrency sub-table.
    conc_raw = sections.get(CONCURRENCY_SECTION, {})
    if "max_connections" in conc_raw:
        out["concurrency"] = {
            "max_connections": _parse_concurrency(conc_raw["max_connections"])
        }
    return out


def _load_user_file() -> dict[str, Any]:
    """Layer 2 — ``~/.config/dataretrieval/config.cfg`` (or platform
    equivalent)."""
    return _ini_to_mapping(_read_ini(_user_config_path()))


def _load_local_file() -> dict[str, Any]:
    """Layer 3 — ``./dataretrieval.cfg`` in the current working directory."""
    return _ini_to_mapping(_read_ini(_local_config_path()))


def _load_env() -> dict[str, Any]:
    """Layer 4 — the four ``API_USGS_*`` env vars, mapped onto the
    dataclass shape."""
    out: dict[str, Any] = {}
    if (raw := os.environ.get(ENV_RETRIES)) is not None:
        out["retry"] = {"max_retries": _coerce_int(raw, ENV_RETRIES)}
    if (raw := os.environ.get(ENV_CONCURRENT)) is not None:
        out["concurrency"] = {"max_connections": _parse_concurrency(raw)}
    if (raw := os.environ.get(ENV_PAT)) is not None:
        out["api_token"] = raw
    if (raw := os.environ.get(ENV_PROGRESS)) is not None:
        parsed = _parse_progress(raw)
        if parsed is not None or raw.strip().lower() in ("auto", ""):
            out["progress"] = parsed
    return out


# ---- mapping → dataclass --------------------------------------------------


def _from_mapping(mapping: dict[str, Any]) -> WaterDataConfig:
    """Construct a :class:`WaterDataConfig` from a merged mapping,
    silently dropping unknown keys so a stray entry doesn't crash
    construction. Sub-tables (``retry``, ``concurrency``) feed their
    own dataclass constructors via :func:`_filter_kwargs`."""
    retry_kw = _filter_kwargs(mapping.get("retry", {}) or {}, RetryPolicy)
    conc_kw = _filter_kwargs(mapping.get("concurrency", {}) or {}, ConcurrencyPolicy)
    return WaterDataConfig(
        retry=RetryPolicy(**retry_kw),
        concurrency=ConcurrencyPolicy(**conc_kw),
        api_token=mapping.get("api_token"),
        progress=mapping.get("progress"),
    )


def _filter_kwargs(mapping: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keep only keys that match dataclass fields of ``cls``."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in mapping.items() if k in known}


# ---- helpers --------------------------------------------------------------


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursive dict merge: nested dicts merge; scalars overwrite. Used
    to layer config sources without dropping unset sub-table keys."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _coerce_int(raw: str | int, name: str) -> int:
    if isinstance(raw, int):
        return raw
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer (got {raw!r}).") from e


def _coerce_float(raw: str | float, name: str) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number (got {raw!r}).") from e


def _parse_concurrency(raw: str | int) -> int | None:
    """Parse a concurrency value. The literal ``"unbounded"``
    (case-insensitive) → ``None``; anything else must parse as an int >= 1."""
    if isinstance(raw, str) and raw.strip().lower() == CONCURRENCY_UNBOUNDED:
        return None
    n = _coerce_int(raw, ENV_CONCURRENT)
    if n < 1:
        raise ValueError(
            f"{ENV_CONCURRENT} must be >= 1 or '{CONCURRENCY_UNBOUNDED}' (got {raw!r})."
        )
    return n


def _parse_progress(raw: str) -> bool | None:
    """Parse a progress preference. ``"on"`` / ``"true"`` / ``"1"`` →
    True; ``"off"`` / ``"false"`` / ``"0"`` → False; ``"auto"`` /
    ``""`` / anything else → ``None`` (auto)."""
    s = raw.strip().lower()
    if s in ("on", "true", "1", "yes"):
        return True
    if s in ("off", "false", "0", "no"):
        return False
    return None


# ---- public re-exports ----------------------------------------------------

__all__ = [
    "RetryPolicy",
    "ConcurrencyPolicy",
    "WaterDataConfig",
    "current",
    "override",
    "set_config",
    "ENV_RETRIES",
    "ENV_CONCURRENT",
    "ENV_PAT",
    "ENV_PROGRESS",
    "CONCURRENCY_UNBOUNDED",
]
