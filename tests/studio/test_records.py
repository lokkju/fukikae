"""The decisions log as a write-ahead record: pending, completed, aborted and resumed records,
the run guard, replays, dry runs on odd registries, and accept's own-speech check."""

import json
import sqlite3
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae import job
from fukikae.artifacts import read_json, write_json
from fukikae.cards.card import parse_card
from fukikae.cards.store import CardStore
from fukikae.cli import app
from fukikae.job import Manifest, StageRecord, save_manifest
from fukikae.models import Assignments, Turn
from fukikae.registry.invalidate import episode
from fukikae.studio import apply as apply_mod
from fukikae.studio.apply import (
    DecisionRefused,
    apply_decision,
    apply_file,
    incomplete_records,
    resume_incomplete,
)
from fukikae.studio.model import parse_decision
from fukikae.workspace import Episode, Workspace
from tests.stages.test_tts import CLONE
from tests.studio.test_apply import (
    STAGES,
    _write_file,
    accept,
    char,
    decision_for,
    log_lines,
    reg_of,
    stages_left,
)


def _ids(ws: Workspace, ep: str) -> list[tuple[int | None, int | None]]:
    a = read_json(ws.work_dir(ep) / "05-characters" / "assignments.json", Assignments)
    return [(c.actor_id, c.character_id) for c in a.clusters]


def _fresh(ws: Workspace, *eps: str) -> None:
    now = datetime.now(UTC)
    for ep in eps:
        save_manifest(
            episode(ws, ep),
            Manifest(stages={s: StageRecord(key="k", status="done", started=now) for s in STAGES}),
        )


def _kinds(ws: Workspace) -> list[str]:
    out = []
    for r in log_lines(ws):
        inc = r.get("incomplete") or {}
        if inc.get("pending"):
            out.append("write-ahead")
        elif "completes" in r:
            out.append("completion" + (" (incomplete)" if inc else ""))
        elif "aborts" in r:
            out.append("aborted")
        elif "resumes" in r:
            out.append("resume" + (" (incomplete)" if inc else ""))
        else:
            out.append("?")
    return out


# --- write-ahead, completion, aborted, resumed ---------------------------------------------


def test_a_decision_writes_a_write_ahead_record_then_a_completion_record(ws):
    d = decision_for(ws, "merge", keep=1, drop=2)
    apply_decision(ws, d)
    first, done = log_lines(ws)
    assert first["decision"] == d.model_dump(mode="json")
    assert first["incomplete"]["pending"] is True
    assert [s["do"] for s in first["incomplete"]["left"]] == ["stale", "remap"]
    assert first["steps"] == first["incomplete"]["left"]
    assert done["completes"] == first["id"] and "incomplete" not in done
    assert incomplete_records(ws) == []
    with reg_of(ws) as reg:  # the marker row landed with the registry writes
        assert reg.decision_marked(first["id"])


def test_ctrl_c_mid_remap_leaves_a_resumable_record(ws, monkeypatch):
    real = apply_mod.remap_assignments

    def interrupted(w, ep, *a, **k):
        if ep == "S01E02":
            raise KeyboardInterrupt
        return real(w, ep, *a, **k)

    monkeypatch.setattr(apply_mod, "remap_assignments", interrupted)
    with pytest.raises(KeyboardInterrupt):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    assert _kinds(ws) == ["write-ahead", "completion (incomplete)"]
    (rec,) = incomplete_records(ws)
    assert [s["do"] for s in rec["incomplete"]["left"]] == ["remap"]
    assert "KeyboardInterrupt" in rec["incomplete"]["errors"][0]
    with pytest.raises(DecisionRefused, match="studio resume"):
        apply_decision(ws, decision_for(ws, "reviewed", character=1))
    monkeypatch.setattr(apply_mod, "remap_assignments", real)
    (res,) = resume_incomplete(ws)
    assert _ids(ws, "S01E02") == [(1, 1), (3, 3)]
    assert incomplete_records(ws) == []
    apply_decision(ws, decision_for(ws, "reviewed", character=1))


def test_a_crash_between_the_write_ahead_line_and_commit_is_aborted(ws, monkeypatch):
    real = apply_mod._append_log

    def crash_after_write_ahead(w, rec):
        real(w, rec)
        if (rec.get("incomplete") or {}).get("pending"):
            raise SystemExit("power cut")  # before COMMIT: the transaction never lands

    monkeypatch.setattr(apply_mod, "_append_log", crash_after_write_ahead)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    monkeypatch.setattr(apply_mod, "_append_log", real)
    with reg_of(ws) as reg:  # registry unchanged
        assert [c.id for c in reg.characters()] == [1, 2, 3]
    assert _kinds(ws) == ["write-ahead"]
    assert incomplete_records(ws) == []  # did not land: aborted, nothing to redo
    assert resume_incomplete(ws) == []
    assert _kinds(ws) == ["write-ahead", "aborted"]
    assert log_lines(ws)[1]["aborts"] == log_lines(ws)[0]["id"]
    assert _ids(ws, "S01E02") == [(2, 2), (3, 3)]
    apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))  # nothing blocks it


def test_apply_settles_an_aborted_record_itself(ws, monkeypatch):
    real = apply_mod._append_log

    def crash(w, rec):
        real(w, rec)
        if (rec.get("incomplete") or {}).get("pending"):
            raise SystemExit

    monkeypatch.setattr(apply_mod, "_append_log", crash)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.setattr(apply_mod, "_append_log", real)
    apply_decision(ws, decision_for(ws, "lock", character=3))
    assert _kinds(ws) == ["write-ahead", "aborted", "write-ahead", "completion"]


def test_a_crash_after_commit_before_any_step_is_resumable(ws, monkeypatch):
    def crash(*_a, **_k):
        raise SystemExit("killed")

    monkeypatch.setattr(apply_mod, "_after_commit", crash)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    monkeypatch.undo()
    with reg_of(ws) as reg:  # the registry part landed
        assert [c.id for c in reg.characters()] == [1, 3]
    assert _kinds(ws) == ["write-ahead"]
    (rec,) = incomplete_records(ws)
    assert [s["do"] for s in rec["incomplete"]["left"]] == ["stale", "remap"]
    assert "characters" in stages_left(ws, "S01E02") and _ids(ws, "S01E02") == [(2, 2), (3, 3)]
    with pytest.raises(DecisionRefused, match="studio resume"):
        apply_decision(ws, decision_for(ws, "reviewed", character=1))
    (res,) = resume_incomplete(ws)
    assert _ids(ws, "S01E02") == [(1, 1), (3, 3)]
    assert "characters" not in stages_left(ws, "S01E02")
    assert _kinds(ws) == ["write-ahead", "resume"]
    assert incomplete_records(ws) == []


def test_an_unwritable_log_refuses_the_decision_before_it_lands(ws, monkeypatch):
    def full(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(apply_mod, "_append_log", full)
    with pytest.raises(DecisionRefused, match="nothing changed"):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    with reg_of(ws) as reg:
        assert [c.id for c in reg.characters()] == [1, 2, 3]


# --- registry merge uses the same records --------------------------------------------------


def test_registry_merge_writes_records_and_a_failed_remap_is_resumable(ws):
    path = ws.work_dir("S01E02") / "05-characters" / "assignments.json"
    good = path.read_text()
    path.write_text("{not json")
    runner = CliRunner()
    r = runner.invoke(app, ["registry", "merge", str(ws.root), "1", "2"])
    assert r.exit_code == 1 and "fukikae studio resume" in r.output
    with reg_of(ws) as reg:
        assert [c.id for c in reg.characters()] == [1, 3]
    (rec,) = incomplete_records(ws)
    assert rec["op"] == "merge"
    # it blocks later decisions like a studio merge would
    with pytest.raises(DecisionRefused, match="studio resume"):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    r = runner.invoke(app, ["registry", "merge", str(ws.root), "1", "3"])
    assert r.exit_code == 1 and "studio resume" in r.output
    path.write_text(good)
    r = runner.invoke(app, ["studio", "resume", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert _ids(ws, "S01E02") == [(1, 1), (3, 3)]
    first = log_lines(ws)[0]
    assert first["registry"] == {"command": "merge", "keep": 1, "drop": 2}
    assert "decision" not in first  # replays skip it: no studio decision to replay
    assert apply_mod.read_decisions(ws, apply_mod.decisions_log(ws)) == []


def test_registry_merge_completes_its_record(ws):
    r = CliRunner().invoke(app, ["registry", "merge", str(ws.root), "1", "2"])
    assert r.exit_code == 0, r.output
    assert _kinds(ws) == ["write-ahead", "completion"]
    assert _ids(ws, "S01E02") == [(1, 1), (3, 3)]


# --- fukikae run refuses while a record is unfinished --------------------------------------


class _Stage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ran = 0

    def inputs(self, ep: Episode) -> list[Path]:  # noqa: ARG002
        return []

    def outputs(self, ep: Episode) -> list[Path]:  # noqa: ARG002
        return []

    def config_key(self, cfg: Any, ep: Episode) -> dict:  # noqa: ARG002
        return {}

    def run(self, ep: Episode, cfg: Any) -> None:  # noqa: ARG002
        self.ran += 1


def _incomplete_merge(ws: Workspace, monkeypatch) -> None:
    def crash(*_a, **_k):
        raise SystemExit

    monkeypatch.setattr(apply_mod, "_after_commit", crash)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    monkeypatch.undo()


def test_run_refuses_while_a_decision_is_incomplete(ws, monkeypatch):
    _incomplete_merge(ws, monkeypatch)
    ep = Episode(source=Path("x"), id="S01E02", work=ws.work_dir("S01E02"), workspace_root=ws.root)
    stage = _Stage("characters")
    with pytest.raises(ValueError, match=f"fukikae studio resume {ws.root}"):
        job.run_stages(ep, ws.config, [stage], from_stage="characters")
    with pytest.raises(ValueError, match="studio resume"):
        job.run_episodes([ep], ws.config, [stage], tail=())
    assert stage.ran == 0
    assert job.run_stages(ep, ws.config, [_Stage("diarize")]) == ["diarize"]  # reads none
    resume_incomplete(ws)
    assert job.run_stages(ep, ws.config, [stage]) == ["characters"]


def test_run_ignores_an_aborted_record(ws, monkeypatch):
    real = apply_mod._append_log

    def crash(w, rec):
        real(w, rec)
        if (rec.get("incomplete") or {}).get("pending"):
            raise SystemExit

    monkeypatch.setattr(apply_mod, "_append_log", crash)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.undo()
    ep = Episode(source=Path("x"), id="S01E01", work=ws.work_dir("S01E01"), workspace_root=ws.root)
    assert job.run_stages(ep, ws.config, [_Stage("characters")]) == ["characters"]


# --- dry runs never create tables ----------------------------------------------------------


def test_a_dry_run_refuses_an_empty_registry_file_and_creates_nothing(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    w = Workspace(root)
    w.registry_path.parent.mkdir(parents=True)
    w.registry_path.write_bytes(b"")
    with pytest.raises(DecisionRefused, match="schema"):
        apply_decision(w, parse_decision({"op": "reviewed", "character": 1}), dry_run=True,
                       force=True)  # fmt: skip
    con = sqlite3.connect(w.registry_path)
    assert con.execute("SELECT name FROM sqlite_master").fetchall() == []
    con.close()


def test_a_dry_run_never_adds_the_decision_marks_table(ws):
    d = decision_for(ws, "reviewed", character=1)
    con = sqlite3.connect(ws.registry_path)
    con.execute("DROP TABLE decision_marks")
    con.commit()
    con.close()
    with pytest.raises(DecisionRefused, match="decision_marks"):
        apply_decision(ws, d, dry_run=True)
    con = sqlite3.connect(ws.registry_path)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    con.close()
    assert "decision_marks" not in names


# --- replays skip what was applied; the workspace is logged resolved -----------------------


def test_replays_skip_decisions_already_applied(ws):
    apply_decision(ws, decision_for(ws, "alias", character=2, name="ROZA"))
    apply_decision(ws, decision_for(ws, "split", character=1, episode="S01E01", cluster="S0",
                                    by="pitch", male_below=165, female_above=200))  # fmt: skip
    apply_decision(ws, decision_for(ws, "reviewed", character=3))
    _fresh(ws, "S01E01", "S01E02")
    reviewed = char(ws, 3).reviewed
    n = len(log_lines(ws))
    results = apply_file(ws, apply_mod.decisions_log(ws))
    assert len(results) == 3 and all(r.skipped for r in results)
    assert all("already applied" in r.lines[0] for r in results)
    assert len(log_lines(ws)) == n
    assert "names" in stages_left(ws, "S01E01") and "characters" in stages_left(ws, "S01E02")
    assert char(ws, 3).reviewed == reviewed
    r = CliRunner().invoke(app, ["studio", "apply", str(ws.root), str(apply_mod.decisions_log(ws))])
    assert r.exit_code == 0, r.output
    assert "already applied" in r.output and "0 decision(s) applied, 3 already applied" in r.output
    dry = apply_file(ws, apply_mod.decisions_log(ws), dry_run=True)
    assert all(r.skipped for r in dry)


def test_a_replay_applies_an_aborted_decision(ws, monkeypatch):
    real = apply_mod._append_log

    def crash(w, rec):
        real(w, rec)
        if (rec.get("incomplete") or {}).get("pending"):
            raise SystemExit

    monkeypatch.setattr(apply_mod, "_append_log", crash)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.undo()
    copy = ws.root.parent / "copy.jsonl"
    copy.write_text(apply_mod.decisions_log(ws).read_text())
    (r,) = apply_file(ws, copy)
    assert not r.skipped and char(ws, 3).locked


def test_the_workspace_is_logged_resolved_and_compared_resolved(ws, monkeypatch):
    monkeypatch.chdir(ws.root.parent)
    apply_decision(Workspace(Path("ws")), decision_for(ws, "reviewed", character=1))
    rec = log_lines(ws)[0]
    assert rec["workspace"] == str(ws.root.resolve())
    monkeypatch.chdir(ws.root)
    results = apply_file(Workspace(Path(".")), apply_mod.decisions_log(ws))
    assert [r.skipped for r in results] == [True]


# --- a file that fails part way says what it applied first ---------------------------------


def test_a_failing_file_prints_what_it_applied_and_the_force_summary(ws, tmp_path):
    f = _write_file(ws, tmp_path / "d.json", [
        parse_decision({"op": "kind", "character": 3, "kind": "narrator"}),
        parse_decision({"op": "lock", "character": 99}),
    ])  # fmt: skip
    r = CliRunner().invoke(app, ["studio", "apply", str(ws.root), str(f), "--force"])
    assert r.exit_code == 1
    out = r.output
    assert "character 3: kind narrator" in out
    assert "--force: 1 decision(s) applied without a digest check" in out
    assert "1 of 2 decisions applied" in out
    assert out.index("character 3: kind narrator") < out.index("error:")


# --- accept: only the character's own speech -----------------------------------------------


def _turns(ws: Workspace, ep: str, turns: list[tuple[str, float, float]]) -> None:
    path = ws.work_dir(ep) / "05-characters" / "assignments.json"
    a = read_json(path, Assignments)
    a.turns = [Turn(start=s, end=e, speaker=c) for c, s, e in turns]
    write_json(path, a)


def _card(ws: Workspace, tmp_path: Path, origins: list[Any], seed: int = 1) -> str:
    store = CardStore(ws.root)
    text = CLONE.split("[[voice.references]]")[0]
    for i, o in enumerate(origins):
        wav = tmp_path / f"r{seed}-{i}.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(np.full(8000, seed * 10 + i, dtype="<i2").tobytes())
        ref = store.add_sidecar(wav, "clip")
        text += f'[[voice.references]]\nclip = "{ref}"\ntranscript = "なんだよ"\nlanguage = "ja"\n'
        if isinstance(o, tuple):
            text += f'origin = {{ episode = "{o[0]}", start = {o[1]}, end = {o[2]} }}\n'
        elif isinstance(o, dict):
            text += "origin = " + json.dumps(o).replace('": ', '" = ').replace('{"', '{ "') + "\n"
    return store.put(parse_card(text))


@pytest.fixture
def turns(ws: Workspace) -> None:
    # S0 is character 1's cluster in S01E01, S2 character 3's
    _turns(ws, "S01E01", [("S0", 0.0, 3.0), ("S0", 3.4, 6.0), ("S2", 6.0, 9.0), ("S0", 20, 22)])


@pytest.mark.usefixtures("fake_backends", "turns")
def test_accept_takes_references_inside_the_characters_own_turns(ws, tmp_path):
    cid = _card(ws, tmp_path, [("S01E01", 1.0, 4.5), ("S01E01", 20.5, 21.5)])
    r = apply_decision(ws, accept(ws, cid))
    assert any("2 of 2 references verified" in ln for ln in r.lines)
    with reg_of(ws) as reg:
        cur = reg.card_for(1, "en")
        assert cur is not None and cur.card_id == cid


@pytest.mark.usefixtures("fake_backends", "turns")
@pytest.mark.parametrize(
    ("origin", "why"),
    [
        (("S01E01", 6.5, 8.5), "S01E01 6.50-8.50"),  # character 3's turn
        (("S01E01", 5.0, 7.0), "S01E01 5.00-7.00"),  # runs into character 3's turn
        (("S01E02", 1.0, 2.0), "S01E02 1.00-2.00"),  # an episode it has no turns in
        (("S01E09", 1.0, 2.0), "S01E09"),  # an episode with no assignments
    ],
)
def test_accept_refuses_a_reference_outside_the_characters_turns(ws, tmp_path, origin, why):
    cid = _card(ws, tmp_path, [("S01E01", 1.0, 2.0), origin])
    with pytest.raises(DecisionRefused, match="D41") as e:
        apply_decision(ws, accept(ws, cid))
    assert why in str(e.value) and "character 1" in str(e.value)
    with reg_of(ws) as reg:
        assert reg.card_for(1, "en") is None


@pytest.mark.usefixtures("fake_backends", "turns")
def test_accept_refuses_an_origin_less_reference_without_this_characters_clone_path(ws, tmp_path):
    cid = _card(ws, tmp_path, [None])
    with pytest.raises(DecisionRefused, match="no origin") as e:
        apply_decision(ws, accept(ws, cid))
    assert "could not verify" in str(e.value)
    other = _card(ws, tmp_path, [{"source": "voices clone"}], seed=2)
    with reg_of(ws) as reg:  # character 3's own clone card
        reg.assign_card(3, "en", other, "assign", False)
    with pytest.raises(DecisionRefused, match="no origin"):
        apply_decision(ws, accept(ws, other))


@pytest.mark.usefixtures("fake_backends", "turns")
def test_accept_takes_an_origin_less_reference_from_this_characters_own_clone_path(ws, tmp_path):
    cid = _card(ws, tmp_path, [{"source": "voices clone"}])
    with reg_of(ws) as reg:  # `voices clone` made and assigned it to character 1
        reg.assign_card(1, "en", cid, "assign", False)
    r = apply_decision(ws, accept(ws, cid))
    assert any("not verified against its turns" in ln and "clone path" in ln for ln in r.lines)
