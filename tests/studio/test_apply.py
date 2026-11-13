"""Decisions applied through the registry: digests, refusals, invalidation and the log."""

import json
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import read_json, write_json
from fukikae.backends import loader
from fukikae.cards.card import parse_card
from fukikae.cards.store import CardStore
from fukikae.cli import app
from fukikae.job import Manifest, StageRecord, load_manifest, save_manifest
from fukikae.models import Assignments, ClusterAssignment
from fukikae.registry.invalidate import Invalidation
from fukikae.registry.store import Registry
from fukikae.studio.apply import (
    DecisionRefused,
    StaleDecision,
    apply_decision,
    apply_file,
    plan_decision,
    row_digest,
    stamp,
)
from fukikae.studio.model import Decisions, parse_decision
from fukikae.workspace import Workspace
from tests.stages.test_tts import CLASSES, CLONE

STAGES = ("diarize", "names", "characters", "translate", "tts", "mix")
# character id -> (episode, cluster) of its actor's exemplars
SPEECH = {1: [("S01E01", "S0")], 2: [("S01E02", "S1")], 3: [("S01E01", "S2"), ("S01E02", "S3")]}


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    w = Workspace(root)
    with Registry(w.registry_path) as reg:
        for cid, spans in SPEECH.items():
            v = np.zeros(4, dtype=np.float32)
            v[cid] = 1.0
            aid, got = reg.create_actor_with_character(v)
            assert (aid, got) == (cid, cid)
            for ep, cluster in spans:
                reg.add_exemplars(aid, ep, cluster, v[None, :], 20.0)
    now = datetime.now(UTC)
    for ep in ("S01E01", "S01E02"):
        work = w.work_dir(ep)
        (work / "05-characters").mkdir(parents=True)
        from fukikae.registry.invalidate import episode

        save_manifest(
            episode(w, ep),
            Manifest(stages={s: StageRecord(key="k", status="done", started=now) for s in STAGES}),
        )
        clusters = [
            ClusterAssignment(
                cluster=cl,
                actor_id=cid,
                character_id=cid,
                score=0.9,
                clean_seconds=20.0,
                status="matched",
            )  # fmt: skip
            for cid, spans in SPEECH.items()
            for e, cl in spans
            if e == ep
        ]
        write_json(
            work / "05-characters" / "assignments.json",
            Assignments(episode=ep, threshold=0.5, clusters=clusters),
        )
    return w


def reg_of(ws: Workspace) -> Registry:
    return Registry(ws.registry_path)


def decision_for(ws: Workspace, op: str, **fields: Any):
    with reg_of(ws) as reg:
        return stamp(reg, parse_decision({"op": op, **fields}))


def stages_left(ws: Workspace, ep: str) -> set[str]:
    from fukikae.registry.invalidate import episode

    return set(load_manifest(episode(ws, ep)).stages)


def char(ws: Workspace, cid: int):
    with reg_of(ws) as reg:
        return next(c for c in reg.characters() if c.id == cid)


def log_lines(ws: Workspace) -> list[dict]:
    path = ws.root / "studio" / "decisions.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines()]


# --- one decision at a time ----------------------------------------------------------------


def test_name_gender_kind_lock_each_apply_in_one_transaction(ws):
    apply_decision(ws, decision_for(ws, "name", character=3, name="Janny"))
    apply_decision(ws, decision_for(ws, "gender", character=3, gender="f"))
    apply_decision(ws, decision_for(ws, "kind", character=3, kind="narrator"))
    apply_decision(ws, decision_for(ws, "lock", character=3))
    c = char(ws, 3)
    assert (c.name, c.name_by, c.gender, c.gender_by, c.kind, c.locked) == (
        "Janny", "operator", "f", "operator", "narrator", True,
    )  # fmt: skip
    assert c.reviewed is not None  # every identity decision marks its character reviewed
    assert char(ws, 1).reviewed is None


def test_alias_and_reviewed_decisions(ws):
    apply_decision(ws, decision_for(ws, "alias", character=2, name="ROZA"))
    with reg_of(ws) as reg:
        assert [(a.alias, a.by) for a in reg.aliases(2)] == [("roza", "operator")]
    r = apply_decision(ws, decision_for(ws, "reviewed", character=1))
    assert char(ws, 1).reviewed is not None and r.invalidations == []


def test_stale_digest_refuses_without_force(ws):
    d = decision_for(ws, "name", character=3, name="Janny")
    with reg_of(ws) as reg:
        reg.rename_character(3, "Someone")  # after the page loaded the row
    with pytest.raises(StaleDecision, match="character 3") as e:
        apply_decision(ws, d)
    assert "Someone" in str(e.value)  # names what it is now
    assert char(ws, 3).name == "Someone" and log_lines(ws) == []
    apply_decision(ws, d, force=True)
    assert char(ws, 3).name == "Janny"


def test_a_decision_without_a_digest_needs_force(ws):
    d = parse_decision({"op": "gender", "character": 1, "gender": "m"})
    with pytest.raises(StaleDecision, match="no digest"):
        apply_decision(ws, d)
    apply_decision(ws, d, force=True)
    assert char(ws, 1).gender == "m"


def test_reviewed_time_is_not_part_of_the_digest(ws):
    with reg_of(ws) as reg:
        before = row_digest(reg, 1)
        reg.mark_reviewed(1)
        assert row_digest(reg, 1) == before
        reg.set_character_gender(1, "f")
        assert row_digest(reg, 1) != before


def test_registry_refusal_changes_nothing(ws):
    with reg_of(ws) as reg:
        reg.set_character_gender(1, "m")
        reg.set_character_gender(2, "f")
    d = decision_for(ws, "merge", keep=1, drop=2)
    with pytest.raises(DecisionRefused, match="gender"):
        apply_decision(ws, d)
    with reg_of(ws) as reg:
        assert [c.id for c in reg.characters()] == [1, 2, 3]
    assert char(ws, 1).reviewed is None  # the review mark rolled back with it
    assert "characters" in stages_left(ws, "S01E02") and log_lines(ws) == []


def test_unknown_character_is_refused(ws):
    d = parse_decision({"op": "lock", "character": 99})
    with pytest.raises(DecisionRefused, match="99"):
        apply_decision(ws, d, force=True)


def test_merge_remaps_assignments_and_marks_characters_stale(ws):
    r = apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    with reg_of(ws) as reg:
        assert [c.id for c in reg.characters()] == [1, 3]
        assert reg.exemplar_count(1) == 2
    a = read_json(ws.work_dir("S01E02") / "05-characters" / "assignments.json", Assignments)
    s1 = next(c for c in a.clusters if c.cluster == "S1")
    assert (s1.actor_id, s1.character_id) == (1, 1)
    assert Invalidation("S01E02", "characters") in r.invalidations
    assert "characters" not in stages_left(ws, "S01E02")
    assert char(ws, 1).reviewed is not None


def test_merge_of_one_actor_with_itself_is_refused(ws):
    with pytest.raises(DecisionRefused):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=1))


def test_gender_marks_only_tts_stale(ws):
    r = apply_decision(ws, decision_for(ws, "gender", character=3, gender="m"))
    assert sorted(r.invalidations) == [
        Invalidation("S01E01", "tts"), Invalidation("S01E02", "tts"),
    ]  # fmt: skip
    for ep in ("S01E01", "S01E02"):
        assert stages_left(ws, ep) == set(STAGES) - {"tts"}


def test_kind_marks_characters_stale_where_it_speaks(ws):
    r = apply_decision(ws, decision_for(ws, "kind", character=1, kind="ignored"))
    assert r.invalidations == [Invalidation("S01E01", "characters")]
    assert "characters" in stages_left(ws, "S01E02")


def test_cluster_split_moves_the_cluster_to_a_new_character(ws):
    d = decision_for(ws, "split", character=3, episode="S01E02", cluster="S3", by="cluster")
    r = apply_decision(ws, d)
    with reg_of(ws) as reg:
        assert reg.cluster_actors("S01E02", "S3") == [4]
    a = read_json(ws.work_dir("S01E02") / "05-characters" / "assignments.json", Assignments)
    s3 = next(c for c in a.clusters if c.cluster == "S3")
    assert (s3.actor_id, s3.character_id) == (4, 4)
    assert r.invalidations == [Invalidation("S01E02", "characters")]


def test_cluster_split_of_another_characters_cluster_is_refused(ws):
    d = decision_for(ws, "split", character=1, episode="S01E02", cluster="S3", by="cluster")
    with pytest.raises(DecisionRefused, match="S3"):
        apply_decision(ws, d)


def test_pitch_split_writes_the_split_file(ws):
    d = decision_for(ws, "split", character=3, episode="S01E01", cluster="S2", by="pitch")
    r = apply_decision(ws, d)
    split = json.loads((ws.root / "studio" / "splits" / "S01E01.json").read_text())
    assert split == {"S2": {"by": "pitch", "male_below": 165.0, "female_above": 200.0}}
    assert r.invalidations == [Invalidation("S01E01", "names")]
    assert "names" not in stages_left(ws, "S01E01")
    with reg_of(ws) as reg:  # no registry change but the review mark
        assert reg.cluster_actors("S01E01", "S2") == [3]
    # a second split in the same episode keeps the first
    apply_decision(ws, decision_for(ws, "split", character=1, episode="S01E01", cluster="S0",
                                    by="pitch"))  # fmt: skip
    split = json.loads((ws.root / "studio" / "splits" / "S01E01.json").read_text())
    assert sorted(split) == ["S0", "S2"]


# --- accept --------------------------------------------------------------------------------


@pytest.fixture
def fake_backends(monkeypatch):
    monkeypatch.setattr(loader, "backend_class", lambda _kind, name: CLASSES[name])


def clone_card(ws: Workspace, tmp_path: Path, seed: int = 1) -> str:
    store = CardStore(ws.root)
    wav = tmp_path / f"ref{seed}.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(np.full(8000, seed, dtype="<i2").tobytes())
    ref = store.add_sidecar(wav, "clip")
    text = CLONE.replace("sha256:9e1f" + "0" * 60, ref)
    return store.put(parse_card(text))


def accept(ws: Workspace, card_id: str, **kw: Any):
    fields = {
        "character": 1, "language": "en", "card_id": card_id, "draft": "1.en", "version": 4,
        "references": [{"episode": "S01E01", "start": 1.0, "end": 4.5}], "overrides": [],
        "auditions": 3,
    }  # fmt: skip
    return decision_for(ws, "accept", **{**fields, **kw})


@pytest.mark.usefixtures("fake_backends")
def test_accept_assigns_a_locked_studio_card(ws, tmp_path):
    cid = clone_card(ws, tmp_path)
    r = apply_decision(ws, accept(ws, cid))
    with reg_of(ws) as reg:
        a = reg.card_for(1, "en")
    assert a is not None and a.note is not None
    assert (a.card_id, a.chosen_by, a.locked) == (cid, "studio", True)
    assert "1.en" in a.note and "version 4" in a.note and "3 auditions" in a.note
    assert "S01E01 1.00-4.50" in a.note
    assert r.invalidations == [Invalidation("S01E01", "tts")]
    assert char(ws, 1).reviewed is None  # accept is not an identity decision


@pytest.mark.usefixtures("fake_backends")
def test_accept_over_a_newer_assignment_is_stale(ws, tmp_path):
    old = clone_card(ws, tmp_path, seed=1)
    d = accept(ws, clone_card(ws, tmp_path, seed=2))
    with reg_of(ws) as reg:  # a run re-picked meanwhile
        reg.assign_card(1, "en", old, "auto", False)
    with pytest.raises(StaleDecision, match="character 1"):
        apply_decision(ws, d)
    with reg_of(ws) as reg:
        cur = reg.card_for(1, "en")
        assert cur is not None and cur.card_id == old


@pytest.mark.usefixtures("fake_backends")
def test_accept_overrides_are_named_in_the_note(ws, tmp_path):
    cid = clone_card(ws, tmp_path)
    over = [{"episode": "S01E01", "start": 9.0, "end": 12.0}]
    apply_decision(ws, accept(ws, cid, overrides=over, references=over))
    with reg_of(ws) as reg:
        cur = reg.card_for(1, "en")
        assert cur is not None and "gated override" in (cur.note or "")


@pytest.mark.usefixtures("fake_backends")
def test_accept_of_a_stock_card_is_refused(ws):
    store = CardStore(ws.root)
    stock = store.put(parse_card(
        'target_language = "en"\n\n[voice]\nbackend = "fake"\nmodel_revision = "1.0"\n'
        'source = "stock"\nvoice = "af_heart"\n'
    ))  # fmt: skip
    with pytest.raises(DecisionRefused, match="D41"):
        apply_decision(ws, accept(ws, stock))
    with reg_of(ws) as reg:
        assert reg.card_for(1, "en") is None


def test_accept_of_a_card_not_in_the_store_is_refused(ws):
    with pytest.raises(DecisionRefused, match="not in the card store"):
        apply_decision(ws, accept(ws, "0123456789abcdef"))


@pytest.mark.usefixtures("fake_backends")
def test_accept_of_a_card_for_another_language_is_refused(ws, tmp_path):
    cid = clone_card(ws, tmp_path)
    with pytest.raises(DecisionRefused, match="'en'"):
        apply_decision(ws, accept(ws, cid, language="de"))


# --- log, dry run, preview, files ----------------------------------------------------------


def test_every_applied_decision_is_logged(ws):
    d = decision_for(ws, "gender", character=3, gender="m")
    r = apply_decision(ws, d)
    (rec,) = log_lines(ws)
    assert rec["decision"] == d.model_dump(mode="json")
    assert rec["lines"] == r.lines and rec["forced"] is False
    assert rec["invalidations"] == [["S01E01", "tts"], ["S01E02", "tts"]]
    datetime.fromisoformat(rec["time"])


def test_dry_run_writes_nothing(ws):
    r = apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2), dry_run=True)
    assert r.invalidations == [Invalidation("S01E02", "characters")]
    assert any("reruns characters on S01E02" in ln for ln in r.lines)
    with reg_of(ws) as reg:
        assert [c.id for c in reg.characters()] == [1, 2, 3]
    assert "characters" in stages_left(ws, "S01E02")
    assert log_lines(ws) == [] and not (ws.root / "studio" / "splits").exists()
    a = read_json(ws.work_dir("S01E02") / "05-characters" / "assignments.json", Assignments)
    assert {c.character_id for c in a.clusters} == {2, 3}


def test_dry_run_still_reports_a_registry_refusal(ws):
    with reg_of(ws) as reg:
        reg.set_character_gender(1, "m")
        reg.set_character_gender(2, "f")
    with pytest.raises(DecisionRefused):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2), dry_run=True)


def test_plan_decision_previews_what_reruns(ws):
    d = decision_for(ws, "name", character=3, name="Janny")
    with reg_of(ws) as reg:
        lines, inv = plan_decision(reg, ws, d)
    assert inv == [Invalidation("S01E01", "characters"), Invalidation("S01E02", "characters")]
    assert lines == ["reruns characters on S01E01, S01E02"]
    with reg_of(ws) as reg:
        lines, inv = plan_decision(reg, ws, decision_for(ws, "lock", character=1))
    assert lines == ["reruns characters on S01E01"]


def _write_file(ws: Workspace, path: Path, decisions: list) -> Path:
    doc = Decisions(workspace=str(ws.root), decisions=decisions)
    path.write_text(doc.model_dump_json(indent=2))
    return path


def test_apply_file_stops_at_the_first_refusal(ws, tmp_path):
    good = decision_for(ws, "gender", character=1, gender="m")
    stale = decision_for(ws, "name", character=2, name="Roza")
    later = decision_for(ws, "kind", character=3, kind="narrator")
    with reg_of(ws) as reg:
        reg.rename_character(2, "Someone")
    path = _write_file(ws, tmp_path / "d.json", [good, stale, later])
    with pytest.raises(StaleDecision, match="1 of 3 decisions applied"):
        apply_file(ws, path)
    assert char(ws, 1).gender == "m"  # earlier decisions stay applied
    assert char(ws, 3).kind == "character"
    assert len(log_lines(ws)) == 1


def test_apply_file_replays_the_log_format(ws, tmp_path):
    d = decision_for(ws, "gender", character=1, gender="f")
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"time": "x", "decision": d.model_dump(mode="json")}) + "\n")
    results = apply_file(ws, path)
    assert len(results) == 1 and char(ws, 1).gender == "f"


def test_apply_file_for_another_workspace_is_refused(ws, tmp_path):
    d = decision_for(ws, "gender", character=1, gender="f")
    path = tmp_path / "d.json"
    path.write_text(Decisions(workspace="/elsewhere", decisions=[d]).model_dump_json())
    with pytest.raises(DecisionRefused, match="/elsewhere"):
        apply_file(ws, path)


def test_decisions_carry_key_and_external_id(ws):
    with reg_of(ws) as reg:
        reg.rename_character(2, "Roza")
        reg.set_character_external_id(2, "anidb:77")
    d = decision_for(ws, "gender", character=2, gender="f")
    assert (d.key, d.external_id) == ("roza", "anidb:77") and d.digest


def test_unknown_op_and_fields_are_refused():
    with pytest.raises(ValueError):
        parse_decision({"op": "rename", "character": 1})
    with pytest.raises(ValueError):
        parse_decision({"op": "gender", "character": 1, "gender": "x"})
    with pytest.raises(ValueError):
        parse_decision({"op": "lock", "character": 1, "colour": "red"})


# --- the command ---------------------------------------------------------------------------


def test_studio_apply_command(ws, tmp_path):
    runner = CliRunner()
    path = _write_file(ws, tmp_path / "d.json", [decision_for(ws, "gender", character=3,
                                                             gender="m")])  # fmt: skip
    r = runner.invoke(app, ["studio", "apply", str(ws.root), str(path), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "reruns tts on S01E01, S01E02" in r.output and char(ws, 3).gender is None
    r = runner.invoke(app, ["studio", "apply", str(ws.root), str(path)])
    assert r.exit_code == 0, r.output
    assert char(ws, 3).gender == "m"
    r = runner.invoke(app, ["studio", "apply", str(ws.root), str(path)])  # now stale
    assert r.exit_code == 1 and "character 3" in r.output
    r = runner.invoke(app, ["studio", "apply", str(ws.root), str(path), "--force"])
    assert r.exit_code == 0, r.output
