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


def test_doctor_reports_llama_server_and_its_version(tmp_path, monkeypatch):
    from fukikae import doctor

    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\necho 'version: 0.9.0-dev (build 12201, commit a2b3)' >&2\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    rows = {label: (ok, detail) for label, ok, detail in doctor.doctor()}
    ok, detail = rows["llama-server"]
    assert ok and str(fake) in detail and "build 12201" in detail
    monkeypatch.setenv("PATH", "/nonexistent")
    rows = {label: (ok, detail) for label, ok, detail in doctor.doctor()}
    assert rows["llama-server"][0] is False


def test_translate_is_the_last_stage_without_tag():
    from fukikae.cli import pipeline_stages
    from fukikae.config import Config

    assert [s.name for s in pipeline_stages(Config())][-2:] == ["characters", "translate"]


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


def test_eval_asr_appends_ledger_rows(tmp_path, monkeypatch):
    import fukikae.eval.asr
    from fukikae.eval.asr import AsrScore
    from fukikae.eval.ledger import read_rows

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "grid"
    ws.mkdir()
    (ws / "fukikae.toml").write_text(
        'source_language = "ko"\n[asr]\nbackend = "faster_whisper"\nmodel = "large-v2"\n'
    )
    src = tmp_path / "S01E02.mkv"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        fukikae.eval.asr,
        "evaluate",
        lambda *_a, **_k: [
            AsrScore("faster_whisper", 0.25, 12.0),
            AsrScore("qwen3_asr", 0.1, None),
        ],
    )
    args = ["eval", "asr", str(ws), str(src), "--subtitle-track", "2"]
    r = CliRunner().invoke(
        app, [*args, "--backend", "faster_whisper", "--backend", "qwen3_asr", "--split", "holdout"]
    )
    assert r.exit_code == 0, r.output
    rows, _ = read_rows()
    assert [(x.backend, x.split, x.language, x.test_set) for x in rows] == [
        ("faster_whisper", "holdout", "ko", "grid"),
        ("qwen3_asr", "holdout", "ko", "grid"),
    ]
    fw, qw = rows
    assert fw.stage == "asr" and fw.episodes == ["S01E02"]
    assert fw.metrics == {"cer": 0.25, "runtime_s": 12.0} and qw.metrics == {"cer": 0.1}
    assert fw.model == "large-v2" and fw.options == {"model": "large-v2"}
    assert qw.model is None and qw.options == {}
    r = CliRunner().invoke(app, [*args, "--backend", "faster_whisper", "--no-ledger"])
    assert r.exit_code == 0, r.output
    assert len(read_rows()[0]) == 2
    r = CliRunner().invoke(app, [*args, "--backend", "faster_whisper", "--split", "dev"])
    assert r.exit_code != 0
