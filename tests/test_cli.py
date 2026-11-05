import pytest
from typer.testing import CliRunner

from doppio import __version__
from doppio.cli import app


def test_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.stdout


def test_doctor_runs():
    r = CliRunner().invoke(app, ["doctor"])
    assert "ffmpeg" in r.stdout
    assert "huggingface" in r.stdout.lower()


@pytest.mark.ffmpeg
def test_run_command_extracts(tmp_path, readonly_copy):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = CliRunner().invoke(app, ["run", str(ws), str(readonly_copy)])
    assert r.exit_code == 0, r.output
    assert len(list((ws / ".doppio" / "episodes").rglob("audio.wav"))) == 1
    again = CliRunner().invoke(app, ["run", str(ws), str(readonly_copy)])
    assert "up to date" in again.output
