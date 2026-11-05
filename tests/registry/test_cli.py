from datetime import UTC, datetime

from typer.testing import CliRunner

from doppio.cli import app
from doppio.job import Manifest, StageRecord, save_manifest
from doppio.registry.store import Registry
from doppio.workspace import Workspace
from tests.stages.test_characters import unit, write_diarize

runner = CliRunner()


def setup(tmp_path, done=True):
    root = tmp_path / "ws"
    root.mkdir()
    src = tmp_path / "S01E01.mkv"
    src.write_bytes(b"x")
    ws = Workspace(root)
    ep = ws.find_episode(str(src))
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    if done:
        now = datetime.now(UTC)
        names = ["extract", "separate", "language", "asr", "diarize"]
        save_manifest(
            ep,
            Manifest(stages={n: StageRecord(key="k", status="done", started=now) for n in names}),
        )
    return ws, src


def test_rematch_runs_and_reports(tmp_path):
    ws, src = setup(tmp_path)
    r = runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)])
    assert r.exit_code == 0, r.output
    assert "None -> 1" in r.output
    r = runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)])
    assert r.exit_code == 0, r.output
    # judged afresh: the actor only this episode made is replaced, never self-matched
    assert "S0: 1 -> 2" in r.output
    with Registry(ws.registry_path) as reg:
        assert [a.id for a in reg.actors()] == [2]
        assert reg.exemplar_count(2) == 1  # not doubled


def test_interrupted_rematch_keeps_assignments(tmp_path, monkeypatch):
    import doppio.job

    ws, src = setup(tmp_path)
    assert runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)]).exit_code == 0

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(doppio.job, "run_stages", boom)
    runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)])
    monkeypatch.undo()
    ep = ws.find_episode(str(src))
    assert (ep.work / "05-characters" / "assignments.json").exists()
    with Registry(ws.registry_path) as reg:
        assert [a.id for a in reg.actors()] == [1] and reg.exemplar_count(1) == 1
    r = runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)])
    assert "S0: 1 -> 2" in r.output  # never `None -> x`


def test_rematch_needs_earlier_stages(tmp_path):
    ws, src = setup(tmp_path, done=False)
    r = runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)])
    assert r.exit_code == 1
    assert "earlier stages not done" in r.output
