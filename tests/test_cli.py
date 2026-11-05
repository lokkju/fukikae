import pytest
from typer.testing import CliRunner

from fukikae import __version__
from fukikae.cli import app


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
    r = CliRunner().invoke(app, ["run", str(ws), str(readonly_copy), "--to", "extract"])
    assert r.exit_code == 0, r.output
    assert len(list((ws / ".fukikae" / "episodes").rglob("audio.wav"))) == 1
    again = CliRunner().invoke(app, ["run", str(ws), str(readonly_copy), "--to", "extract"])
    assert "up to date" in again.output


def _user_errors():
    from fukikae.backends.loader import BackendUnavailable
    from fukikae.config import ConfigError
    from fukikae.gpu import VramTimeout

    return [
        VramTimeout("not enough free VRAM"),
        ConfigError("unknown asr backend 'nope'"),
        BackendUnavailable("asr backend 'qwen3_asr' cannot be imported"),
    ]


@pytest.mark.parametrize("exc", _user_errors(), ids=lambda e: type(e).__name__)
def test_run_reports_user_errors_without_traceback(tmp_path, monkeypatch, exc):
    import fukikae.job

    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "S01E01.mkv"
    src.write_bytes(b"x")

    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(fukikae.job, "run_stages", boom)
    r = CliRunner().invoke(app, ["run", str(ws), str(src), "--no-wait"])
    assert r.exit_code == 1
    assert isinstance(r.exception, SystemExit)
    assert f"error: {exc}" in r.output


@pytest.mark.parametrize("exc", _user_errors(), ids=lambda e: type(e).__name__)
def test_eval_asr_reports_user_errors_without_traceback(tmp_path, monkeypatch, exc):
    import fukikae.eval.asr

    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "S01E01.mkv"
    src.write_bytes(b"x")

    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(fukikae.eval.asr, "evaluate", boom)
    r = CliRunner().invoke(
        app, ["eval", "asr", str(ws), str(src), "--subtitle-track", "2", "--backend", "x"]
    )
    assert r.exit_code == 1
    assert isinstance(r.exception, SystemExit)
    assert f"error: {exc}" in r.output
