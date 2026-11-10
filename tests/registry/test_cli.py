from datetime import UTC, datetime

from typer.testing import CliRunner

from fukikae.cli import app
from fukikae.job import Manifest, StageRecord, save_manifest
from fukikae.registry.store import Registry
from fukikae.workspace import Workspace
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
    import fukikae.job

    ws, src = setup(tmp_path)
    assert runner.invoke(app, ["registry", "rematch", str(ws.root), str(src)]).exit_code == 0

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(fukikae.job, "run_stages", boom)
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


def _alias_ws(tmp_path):
    import numpy as np

    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace(root)
    with Registry(ws.registry_path) as reg:
        for _ in range(2):
            reg.create_actor_with_character(np.array([1.0, 0.0], dtype=np.float32))
    return ws


def test_registry_alias_sets_lists_and_removes(tmp_path):
    ws = _alias_ws(tmp_path)
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "1", "NATSU"])
    assert r.exit_code == 0, r.output
    assert "natsu" in r.output
    r = runner.invoke(app, ["registry", "alias", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert "natsu\t1\toperator" in r.output
    r = runner.invoke(app, ["registry", "show", str(ws.root), "1"])
    assert r.exit_code == 0, r.output
    assert "aliases: natsu (operator)" in r.output
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "1", "Natsu", "--remove"])
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert reg.aliases() == []


def test_registry_alias_refuses_multi_names(tmp_path):
    ws = _alias_ws(tmp_path)
    for name in ("NATSU/ZEREF", "MAKAROV/WRAITH", "ALL", ""):
        r = runner.invoke(app, ["registry", "alias", str(ws.root), "1", name])
        assert r.exit_code == 1, (name, r.output)
    with Registry(ws.registry_path) as reg:
        assert reg.aliases() == []


def test_registry_alias_takes_a_possession_name_as_its_own(tmp_path):
    ws = _alias_ws(tmp_path)  # D29 #12
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "2", "MAKAROV(WRAITH)"])
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert [(a.alias, a.character_id) for a in reg.aliases()] == [("makarov (wraith)", 2)]


def test_registry_alias_needs_replace_over_another_operator_alias(tmp_path):
    ws = _alias_ws(tmp_path)
    assert runner.invoke(app, ["registry", "alias", str(ws.root), "1", "HAPPY"]).exit_code == 0
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "2", "HAPPY"])
    assert r.exit_code == 1 and "--replace" in r.output
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "2", "HAPPY", "--remove"])
    assert r.exit_code == 1 and "character 1" in r.output  # not this character's alias
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "2", "HAPPY", "--replace"])
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert [(a.alias, a.character_id) for a in reg.aliases()] == [("happy", 2)]


def test_registry_rename_records_the_operator_and_warns_on_a_held_key(tmp_path):
    ws = _alias_ws(tmp_path)
    assert runner.invoke(app, ["registry", "alias", str(ws.root), "1", "ERZA"]).exit_code == 0
    r = runner.invoke(app, ["registry", "rename", str(ws.root), "2", "Erza"])
    assert r.exit_code == 0, r.output
    assert "character 1" in r.output
    r = runner.invoke(app, ["registry", "show", str(ws.root), "2"])
    assert "Erza (name by operator)" in r.output


def test_registry_alias_reports_an_ambiguous_key_in_one_line(tmp_path):
    ws = _alias_ws(tmp_path)
    with Registry(ws.registry_path) as reg, reg._tx() as con:  # two names, one key, no alias
        con.execute("UPDATE characters SET name = 'Lucy', name_by = 'operator'")
    r = runner.invoke(app, ["registry", "alias", str(ws.root), "2", "LUCY"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit), r.output
    (line,) = r.output.strip().splitlines()  # one line, no traceback
    assert line.startswith("error:") and "registry rename" in line and "registry merge" in line


def _names_used(ws: Workspace, episode: str, key: str) -> None:
    from fukikae.artifacts import write_json
    from fukikae.names.inputs import names_path
    from fukikae.names.models import NamedLine, NamesResult
    from fukikae.names.normalize import NameClass

    line = NamedLine(
        id=0, start=0.0, end=1.0, raw=key.upper(), key=key, display=key.title(),
        cls=NameClass.NAMED, inner=False, text="hi",
    )  # fmt: skip
    path = names_path(ws.work_dir(episode))
    path.parent.mkdir(parents=True)
    write_json(
        path,
        NamesResult(
            episode=episode,
            used=True,
            verdict="full",
            reason=None,
            source=None,
            offset_s=0.0,
            coverage={},
            lines=[line],
            hidden=[],
            pieces=[],
            labels={},
            unvoiced=[],
            excluded=[],
            dropped_row_seconds=0.0,
            suggestions=[],
        ),  # fmt: skip
    )


def test_registry_rename_stales_episodes_that_used_the_new_names_key(tmp_path):
    ws = _alias_ws(tmp_path)
    _names_used(ws, "S01E03", "erza")
    _names_used(ws, "S01E04", "lucy")
    r = runner.invoke(app, ["registry", "rename", str(ws.root), "2", "ERZA"])
    assert r.exit_code == 0, r.output
    assert "stale" in r.output and "S01E03" in r.output and "S01E04" not in r.output


def test_registry_lock_character_locks_and_unlocks(tmp_path):
    ws = _alias_ws(tmp_path)
    r = runner.invoke(app, ["registry", "lock", str(ws.root), "2", "--character"])
    assert r.exit_code == 0 and "character 2: locked" in r.output
    with Registry(ws.registry_path) as reg:
        assert [c.locked for c in reg.characters()] == [False, True]
        assert not any(a.locked for a in reg.actors())
    r = runner.invoke(app, ["registry", "lock", str(ws.root), "2", "--character", "--unlock"])
    assert r.exit_code == 0
    with Registry(ws.registry_path) as reg:
        assert not any(c.locked for c in reg.characters())
    r = runner.invoke(app, ["registry", "lock", str(ws.root), "9", "--character"])
    assert r.exit_code != 0 and "no character 9" in r.output


def test_registry_split_accepts_a_sub_cluster_label(tmp_path):
    import numpy as np

    from fukikae.artifacts import read_json, write_json
    from fukikae.models import Assignments, ClusterAssignment

    ws = _alias_ws(tmp_path)
    eid = "S01E01"
    path = ws.work_dir(eid) / "05-characters" / "assignments.json"
    path.parent.mkdir(parents=True)
    rows = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    with Registry(ws.registry_path) as reg:
        reg.add_exemplars(1, eid, "SPEAKER_08~happy", rows, 10.0)
        reg.add_exemplars(1, eid, "SPEAKER_08", rows, 10.0)
    clusters = [
        ClusterAssignment.model_validate(
            {"cluster": c, "actor_id": 1, "character_id": 1, "score": None,
             "clean_seconds": 10.0, "status": "matched", **extra}
        )
        for c, extra in (("SPEAKER_08~happy", {"name": "happy", "source": "name"}),
                         ("SPEAKER_08", {}))
    ]  # fmt: skip
    write_json(path, Assignments(episode=eid, threshold=0.8, clusters=clusters))
    r = runner.invoke(
        app, ["registry", "split", str(ws.root), "--episode", eid, "--cluster", "SPEAKER_08~happy"]
    )
    assert r.exit_code == 0, r.output
    got = {c.cluster: c for c in read_json(path, Assignments).clusters}
    assert got["SPEAKER_08"].actor_id == 1
    moved = got["SPEAKER_08~happy"]
    assert moved.actor_id not in (None, 1) and moved.name == "happy"
    with Registry(ws.registry_path) as reg:
        assert reg.cluster_actors(eid, "SPEAKER_08~happy") == [moved.actor_id]
        assert reg.cluster_actors(eid, "SPEAKER_08") == [1]
