import importlib.metadata

import pytest
from typer.testing import CliRunner

from doppio.backends.loader import BackendUnavailable, available, create, supports
from doppio.cli import app
from doppio.config import BackendConfig, ConfigError
from tests.backends.fakes import AnyLanguage, FakeTranscriber

GOOD = importlib.metadata.EntryPoint(
    "fake", "tests.backends.fakes:FakeTranscriber", "doppio.backends.asr"
)
BROKEN = importlib.metadata.EntryPoint(
    "faster_whisper", "tests.backends.not_there:Nope", "doppio.backends.asr"
)


def _patch(monkeypatch, *eps):
    def fake(group=None, **_):
        return [ep for ep in eps if ep.group == group]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)


@pytest.fixture
def fake_eps(monkeypatch):
    _patch(monkeypatch, GOOD)


@pytest.fixture
def broken_ep(monkeypatch):
    _patch(monkeypatch, GOOD, BROKEN)


@pytest.mark.usefixtures("fake_eps")
def test_available_maps_names_to_classes():
    assert available("asr") == {"fake": FakeTranscriber}
    assert available("separation") == {}


@pytest.mark.usefixtures("fake_eps")
def test_create_unknown_lists_installed():
    with pytest.raises(ConfigError, match="fake"):
        create("asr", BackendConfig("nope", {}), "cpu")


@pytest.mark.usefixtures("fake_eps")
def test_create_passes_options():
    b = create("asr", BackendConfig("fake", {"model": "m"}), "cpu")
    assert isinstance(b, FakeTranscriber)
    assert b.options == {"model": "m"}
    assert b.device == "cpu"


@pytest.mark.usefixtures("broken_ep")
def test_import_failure_names_extra():
    with pytest.raises(BackendUnavailable, match=r"doppio\[whisper\]"):
        create("asr", BackendConfig("faster_whisper", {}), "cpu")
    assert available("asr") == {"fake": FakeTranscriber}


def test_supports():
    assert supports(FakeTranscriber, "ko")
    assert not supports(FakeTranscriber, "fr")
    assert supports(AnyLanguage, "fr")


@pytest.mark.usefixtures("fake_eps")
def test_backends_list_cli():
    r = CliRunner().invoke(app, ["backends", "list"])
    assert r.exit_code == 0
    assert "asr" in r.stdout
    assert "fake" in r.stdout
    assert "(none)" in r.stdout
