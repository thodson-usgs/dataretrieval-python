"""The "geopandas not installed" notice is logged lazily, once per process."""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

import dataretrieval.ogc.shaping as shaping

_NOTICE = "Geopandas not installed"


def test_import_without_geopandas_is_silent():
    """``import dataretrieval`` prints nothing when geopandas is missing."""
    code = "import sys; sys.modules['geopandas'] = None; import dataretrieval"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert _NOTICE not in proc.stderr


@pytest.fixture
def no_geopandas(monkeypatch, caplog):
    monkeypatch.setattr(shaping, "GEOPANDAS", False)
    monkeypatch.setattr(shaping, "_notice_logged", False)
    caplog.set_level(logging.WARNING, logger=shaping.__name__)
    return caplog


def _notices(caplog):
    return [r for r in caplog.records if _NOTICE in r.getMessage()]


def test_notice_logged_once_at_first_geometry_request(no_geopandas):
    assert shaping._wants_geodataframe(True) is False
    assert shaping._wants_geodataframe(True) is False
    assert len(_notices(no_geopandas)) == 1


def test_no_notice_when_geometry_not_requested(no_geopandas):
    assert shaping._wants_geodataframe(False) is False
    assert _notices(no_geopandas) == []


def test_no_notice_when_geopandas_installed(monkeypatch, caplog):
    monkeypatch.setattr(shaping, "GEOPANDAS", True)
    monkeypatch.setattr(shaping, "_notice_logged", False)
    caplog.set_level(logging.WARNING, logger=shaping.__name__)
    assert shaping._wants_geodataframe(True) is True
    assert _notices(caplog) == []
