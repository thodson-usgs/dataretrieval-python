"""Tests for the layered config loader.

Verifies the precedence chain (defaults → user file → local file → env vars
→ Python override) and the round-trip through the INI / configparser layer.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dataretrieval.waterdata import _config
from dataretrieval.waterdata._config import (
    CONCURRENCY_UNBOUNDED,
    ConcurrencyPolicy,
    RetryPolicy,
    WaterDataConfig,
    override,
)

# ---- isolate from real config files ---------------------------------------


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Point user-config + local-config + cwd at empty tmp dirs so no real
    file or shell env leaks in. Tests that want a layer install it explicitly
    via :func:`_write` / :func:`monkeypatch.setenv`.

    Also clears the four ``API_USGS_*`` env vars (the autouse conftest
    fixture sets two, but per-layer tests need a blank slate)."""
    # Redirect user-config path.
    user_dir = tmp_path / "user_config"
    user_dir.mkdir()
    monkeypatch.setattr(_config, "_user_config_path", lambda: user_dir / "config.cfg")
    # Make Path.cwd() return the tmp dir so the local-config file lookup
    # lands here too.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    # Clear the env vars set by other autouse fixtures.
    for name in (
        _config.ENV_RETRIES,
        _config.ENV_CONCURRENT,
        _config.ENV_PAT,
        _config.ENV_PROGRESS,
    ):
        monkeypatch.delenv(name, raising=False)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())


# ---- layer 1: defaults ----------------------------------------------------


def test_defaults_when_nothing_configured():
    cfg = WaterDataConfig.load()
    assert cfg.retry == RetryPolicy()
    assert cfg.concurrency == ConcurrencyPolicy()
    assert cfg.api_token is None
    assert cfg.progress is None


# ---- layer 2: user file ---------------------------------------------------


def test_user_file_overrides_defaults():
    _write(
        _config._user_config_path(),
        """
        [default]
        api_token = user-token
        progress = on

        [retry]
        max_retries = 7
        base_backoff = 0.25

        [concurrency]
        max_connections = 32
        """,
    )
    cfg = WaterDataConfig.load()
    assert cfg.api_token == "user-token"
    assert cfg.progress is True
    assert cfg.retry.max_retries == 7
    assert cfg.retry.base_backoff == 0.25
    # Unset fields still take the dataclass defaults.
    assert cfg.retry.max_backoff == _config._RETRY_MAX_BACKOFF
    assert cfg.concurrency.max_connections == 32


# ---- layer 3: local file overrides user file ------------------------------


def test_local_file_overrides_user_file():
    _write(
        _config._user_config_path(),
        """
        [default]
        api_token = user-token

        [retry]
        max_retries = 7
        """,
    )
    _write(
        Path.cwd() / _config._LOCAL_CONFIG_NAME,
        """
        [default]
        api_token = local-token

        [retry]
        max_retries = 99
        base_backoff = 1.5
        """,
    )
    cfg = WaterDataConfig.load()
    assert cfg.api_token == "local-token"
    assert cfg.retry.max_retries == 99
    assert cfg.retry.base_backoff == 1.5


# ---- layer 4: env overrides files -----------------------------------------


def test_env_overrides_files(monkeypatch):
    _write(
        _config._user_config_path(),
        """
        [default]
        api_token = file-token

        [retry]
        max_retries = 5

        [concurrency]
        max_connections = 8
        """,
    )
    monkeypatch.setenv(_config.ENV_RETRIES, "2")
    monkeypatch.setenv(_config.ENV_CONCURRENT, CONCURRENCY_UNBOUNDED)
    monkeypatch.setenv(_config.ENV_PAT, "env-token")
    monkeypatch.setenv(_config.ENV_PROGRESS, "off")

    cfg = WaterDataConfig.load()
    assert cfg.api_token == "env-token"
    assert cfg.progress is False
    assert cfg.retry.max_retries == 2  # env wins over file's 5
    assert cfg.concurrency.max_connections is None  # "unbounded" → None


def test_env_only_sets_what_it_provides(monkeypatch):
    """An env var sets only its own field; other file-set fields are
    preserved (the deep-update keeps sibling keys)."""
    _write(
        _config._user_config_path(),
        """
        [retry]
        max_retries = 5
        base_backoff = 1.0
        max_backoff = 10.0
        """,
    )
    monkeypatch.setenv(_config.ENV_RETRIES, "2")
    cfg = WaterDataConfig.load()
    assert cfg.retry.max_retries == 2  # env overrides
    assert cfg.retry.base_backoff == 1.0  # file-set, preserved
    assert cfg.retry.max_backoff == 10.0  # file-set, preserved


# ---- layer 5: Python override wins above all ------------------------------


def test_python_override_wins(monkeypatch):
    monkeypatch.setenv(_config.ENV_RETRIES, "2")
    custom = WaterDataConfig(
        retry=RetryPolicy(max_retries=99, base_backoff=0.0, max_backoff=0.0)
    )
    with override(custom):
        # current() short-circuits to the active override (no file/env load).
        assert _config.current() is custom
        # And the legacy from_env() factories pick it up too.
        assert RetryPolicy.from_env().max_retries == 99
        assert RetryPolicy.from_env().base_backoff == 0.0
    # On exit, the layered loader resumes.
    assert RetryPolicy.from_env().max_retries == 2  # back to env


def test_override_is_contextvar_scoped():
    """Nested ``override`` blocks pop correctly; the outer override is
    restored at inner exit."""
    outer = WaterDataConfig(retry=RetryPolicy(max_retries=1))
    inner = WaterDataConfig(retry=RetryPolicy(max_retries=2))
    with override(outer):
        assert _config.current() is outer
        with override(inner):
            assert _config.current() is inner
        assert _config.current() is outer


# ---- parsing / validation -------------------------------------------------


def test_env_concurrency_unbounded_keyword(monkeypatch):
    monkeypatch.setenv(_config.ENV_CONCURRENT, "UNBOUNDED")  # case-insensitive
    assert WaterDataConfig.load().concurrency.max_connections is None


def test_env_concurrency_invalid_value(monkeypatch):
    monkeypatch.setenv(_config.ENV_CONCURRENT, "abc")
    with pytest.raises(ValueError, match=_config.ENV_CONCURRENT):
        WaterDataConfig.load()


def test_env_retries_negative_is_rejected(monkeypatch):
    monkeypatch.setenv(_config.ENV_RETRIES, "-1")
    with pytest.raises(ValueError):
        WaterDataConfig.load()


def test_progress_parser_recognizes_truthy_falsy_and_auto():
    assert _config._parse_progress("on") is True
    assert _config._parse_progress("true") is True
    assert _config._parse_progress("1") is True
    assert _config._parse_progress("off") is False
    assert _config._parse_progress("FALSE") is False  # case-insensitive
    assert _config._parse_progress("auto") is None
    assert _config._parse_progress("") is None
    assert _config._parse_progress("nonsense") is None


def test_missing_files_are_silent():
    """No user or local file → no error, just falls through to defaults."""
    assert not _config._user_config_path().exists()
    assert not (Path.cwd() / _config._LOCAL_CONFIG_NAME).exists()
    cfg = WaterDataConfig.load()
    assert cfg == WaterDataConfig()  # all defaults


def test_unknown_keys_in_file_are_ignored():
    """A stray key in a sub-table shouldn't crash construction."""
    _write(
        _config._user_config_path(),
        """
        [retry]
        max_retries = 3
        not_a_field = 42

        [concurrency]
        max_connections = 4
        also_not_a_field = "x"
        """,
    )
    cfg = WaterDataConfig.load()
    assert cfg.retry.max_retries == 3
    assert cfg.concurrency.max_connections == 4


# ---- direct dataclass validation -----------------------------------------


def test_retry_policy_rejects_negative_settings():
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(base_backoff=-0.5)


def test_concurrency_policy_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        ConcurrencyPolicy(max_connections=0)
    with pytest.raises(ValueError):
        ConcurrencyPolicy(max_connections=-1)
    # ``None`` is fine — explicit "unbounded".
    ConcurrencyPolicy(max_connections=None)
