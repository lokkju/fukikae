"""Replays decided by the registry, the decisions log's integrity, the run lock, and accept's
clean own-speech rule (review 14c)."""

import fcntl
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae import job
from fukikae.artifacts import read_json, write_json
from fukikae.cli import app
from fukikae.models import Assignments, Delivery, DubCue, DubCues
from fukikae.registry.invalidate import episode
from fukikae.studio import apply as apply_mod
from fukikae.studio.apply import (
    DecisionRefused,
    apply_decision,
    apply_file,
    incomplete_records,
    resume_incomplete,
    unfinished_message,
)
from fukikae.workspace import Episode, Workspace
from tests.studio.test_apply import accept, char, decision_for, log_lines, reg_of, stages_left
from tests.studio.test_records import _card, _fresh, _kinds, _Stage, _turns


def _crash_after_commit(monkeypatch) -> None:
    def crash(*_a, **_k):
        raise SystemExit("killed")

    monkeypatch.setattr(apply_mod, "_after_commit", crash)


def _ep(ws: Workspace, eid: str) -> Episode:
    return Episode(source=Path("x"), id=eid, work=ws.work_dir(eid), workspace_root=ws.root)


# --- 1. the registry decides what a replay skips -------------------------------------------


def test_a_resumed_decision_is_skipped_by_a_replay(ws, monkeypatch):
    _crash_after_commit(monkeypatch)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "split", character=1, episode="S01E01",
                                        cluster="S0", by="pitch", male_below=165,
                                        female_above=200))  # fmt: skip
    monkeypatch.undo()
    resume_incomplete(ws)
    assert _kinds(ws) == ["write-ahead", "resume"]
    (r,) = apply_file(ws, apply_mod.decisions_log(ws))
    assert r.skipped and "already applied" in r.lines[0]


def test_a_forced_replay_skips_a_resumed_lock(ws, monkeypatch):
    _crash_after_commit(monkeypatch)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.undo()
    resume_incomplete(ws)
    n = len(log_lines(ws))
    (r,) = apply_file(ws, apply_mod.decisions_log(ws), force=True)
    assert r.skipped
    assert len(log_lines(ws)) == n


def test_a_replay_into_a_restored_registry_applies_what_it_lacks_and_says_so(ws):
    backup = ws.root.parent / "reg.bak"
    with reg_of(ws) as reg:
        reg._con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy(ws.registry_path, backup)
    apply_decision(ws, decision_for(ws, "lock", character=3))
    with reg_of(ws) as reg:
        reg._con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy(backup, ws.registry_path)
    assert not char(ws, 3).locked
    copy = ws.root.parent / "log.jsonl"
    copy.write_text(apply_mod.decisions_log(ws).read_text())
    dry = apply_file(ws, copy, dry_run=True)
    assert not dry[0].skipped and "no mark of it" in dry[0].lines[0]
    (r,) = apply_file(ws, copy)
    assert not r.skipped and char(ws, 3).locked
    assert "no mark of it" in r.lines[0] and "applying it again" in r.lines[0]
    # the re-applied record names the one it replays, so the old log replays as applied
    assert log_lines(ws)[-2]["replays"] == log_lines(ws)[0]["id"]
    (again,) = apply_file(ws, copy)
    assert again.skipped


# --- 2. the log's integrity ----------------------------------------------------------------


def test_a_failed_append_is_truncated_back_and_nothing_lands(ws, monkeypatch):
    apply_decision(ws, decision_for(ws, "reviewed", character=1))
    size = apply_mod.decisions_log(ws).stat().st_size

    def fail(_fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(apply_mod, "_fsync", fail)
    with pytest.raises(DecisionRefused, match="nothing changed"):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.undo()
    assert apply_mod.decisions_log(ws).stat().st_size == size
    assert not char(ws, 3).locked
    apply_decision(ws, decision_for(ws, "lock", character=3))
    assert _kinds(ws) == ["write-ahead", "completion"] * 2


def test_an_append_starts_on_a_new_line_and_is_fsynced(ws, monkeypatch):
    path = apply_mod.decisions_log(ws)
    path.parent.mkdir(parents=True)
    path.write_text('{"id": "cut sho')
    synced: list[int] = []
    monkeypatch.setattr(apply_mod, "_fsync", synced.append)
    apply_mod._append_log(ws, {"id": "x"})
    assert path.read_text().splitlines()[1] == '{"id": "x"}'
    assert synced


def test_a_corrupt_log_line_refuses_runs_decisions_and_resume(ws, monkeypatch):
    # a write cut short (here simulated past the truncation) leaves half a record
    real = apply_mod._append_log
    calls = {"n": 0}

    def partial(w, rec):
        calls["n"] += 1
        if calls["n"] == 1:
            line = json.dumps(rec)
            apply_mod.decisions_log(w).parent.mkdir(parents=True, exist_ok=True)
            with apply_mod.decisions_log(w).open("a") as f:
                f.write(line[: len(line) // 2])
            raise OSError(28, "No space left on device")
        return real(w, rec)

    monkeypatch.setattr(apply_mod, "_append_log", partial)
    with pytest.raises(DecisionRefused, match="nothing changed"):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.undo()
    log = apply_mod.decisions_log(ws)
    with pytest.raises(DecisionRefused, match=rf"line 1 of {log}"):
        apply_decision(ws, decision_for(ws, "merge", keep=1, drop=2))
    with pytest.raises(DecisionRefused, match="line 1"):
        incomplete_records(ws)
    with pytest.raises(DecisionRefused, match="line 1"):
        resume_incomplete(ws)
    msg = unfinished_message(ws)
    assert msg is not None and "line 1" in msg and "mid-change" not in msg
    with pytest.raises(ValueError, match="line 1"):
        job.run_stages(_ep(ws, "S01E01"), ws.config, [_Stage("characters")])
    with reg_of(ws) as reg:
        assert [c.id for c in reg.characters()] == [1, 2, 3]


# --- 3. the run lock -----------------------------------------------------------------------


class _Deciding(_Stage):
    """A stage that tries a studio decision while the run holds the lock."""

    def __init__(self, ws: Workspace) -> None:
        super().__init__("characters")
        self.ws = ws
        self.refused: str | None = None

    def run(self, ep, cfg) -> None:
        super().run(ep, cfg)
        try:
            apply_decision(self.ws, decision_for(self.ws, "lock", character=3))
        except DecisionRefused as e:
            self.refused = str(e)


def test_a_decision_is_refused_while_a_run_is_in_progress(ws):
    apply_mod.studio_dir(ws).mkdir()
    stage = _Deciding(ws)
    assert job.run_stages(_ep(ws, "S01E01"), ws.config, [stage], from_stage="characters")
    assert stage.refused is not None and "a `fukikae run` is in progress" in stage.refused
    assert not char(ws, 3).locked
    apply_decision(ws, decision_for(ws, "lock", character=3))  # the run is over


def test_a_run_creates_only_the_lock_file_and_nothing_without_a_studio(ws):
    job.run_stages(_ep(ws, "S01E01"), ws.config, [_Stage("diarize")])
    assert not apply_mod.studio_dir(ws).exists()
    apply_mod.studio_dir(ws).mkdir()
    job.run_stages(_ep(ws, "S01E01"), ws.config, [_Stage("diarize")], from_stage="diarize")
    assert [p.name for p in apply_mod.studio_dir(ws).iterdir()] == [".lock"]


def test_a_run_is_refused_while_a_writer_holds_the_lock_and_names_it(ws):
    with (
        apply_mod._studio_lock(ws, "`fukikae registry merge`"),
        pytest.raises(ValueError, match=r"`fukikae registry merge` \(pid \d+\)"),
    ):
        job.run_stages(_ep(ws, "S01E01"), ws.config, [_Stage("diarize")])


def test_a_writer_is_refused_by_another_writer_by_name(ws):
    with (
        apply_mod._studio_lock(ws, "`fukikae studio resume`"),
        pytest.raises(DecisionRefused, match=r"`fukikae studio resume` \(pid \d+\)"),
    ):
        apply_decision(ws, decision_for(ws, "lock", character=3))


class _Marking(_Stage):
    """A later stage during which something marks an earlier stage stale."""

    def __init__(self, ws: Workspace) -> None:
        super().__init__("tts")
        self.ws = ws

    def run(self, ep, cfg) -> None:
        super().run(ep, cfg)
        job.mark_stale(episode(self.ws, ep.id), "characters")


def test_a_stale_mark_made_during_a_run_is_kept(ws):
    _fresh(ws, "S01E01")
    ep = _ep(ws, "S01E01")
    job.run_stages(ep, ws.config, [_Stage("characters"), _Marking(ws)], from_stage="characters")
    assert "characters" not in stages_left(ws, "S01E01")
    assert "tts" in stages_left(ws, "S01E01")


# --- 4. own speech: clean turns, and the character's own cues ------------------------------


@pytest.mark.usefixtures("fake_backends")
def test_accept_refuses_a_reference_under_another_speakers_overlap(ws, tmp_path):
    _turns(ws, "S01E01", [("S0", 0.0, 10.0), ("S2", 2.0, 8.0)])  # S2 (character 3) inside
    ok = _card(ws, tmp_path, [("S01E01", 8.5, 9.5)], seed=2)
    apply_decision(ws, accept(ws, ok))
    cid = _card(ws, tmp_path, [("S01E01", 3.0, 7.0)])
    with pytest.raises(DecisionRefused, match="overlaps another speaker"):
        apply_decision(ws, accept(ws, cid))


def _cues(ws: Workspace, ep: str, spans: list[tuple[int, float, float]]) -> None:
    cues = [
        DubCue(id=f"c{i}", start=s, end=e, character_id=c, extra=None, speaker=f"char{c}",
               source_text="なんだよ", text="what", budget=20, delivery=Delivery(reference=(s, e)))
        for i, (c, s, e) in enumerate(spans)
    ]  # fmt: skip
    path = ws.work_dir(ep) / "06-translate" / "cues.en.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, DubCues(episode=ep, source_language="ja", target_language="en",
                             path="subs", cues=cues))  # fmt: skip


@pytest.mark.usefixtures("fake_backends")
def test_accept_takes_a_cue_thin_card_of_a_character_with_no_turns(ws, tmp_path):
    _turns(ws, "S01E01", [("S2", 6.0, 9.0)])  # character 1 holds no turn
    cid = _card(ws, tmp_path, [("S01E01", 1.0, 3.2)])  # its cues' joined source span
    with reg_of(ws) as reg:
        reg.assign_card(1, "en", cid, "auto", False)  # what cards.auto assigns
    with pytest.raises(DecisionRefused, match="not inside"):
        apply_decision(ws, accept(ws, cid))
    _cues(ws, "S01E01", [(1, 1.0, 2.0), (1, 2.3, 3.2), (3, 6.0, 9.0)])
    r = apply_decision(ws, accept(ws, cid))
    assert any("1 of 1 references verified" in ln for ln in r.lines)
    other = _card(ws, tmp_path, [("S01E01", 6.5, 8.0)], seed=2)  # character 3's cue
    with pytest.raises(DecisionRefused, match="not inside"):
        apply_decision(ws, accept(ws, other))


# --- 5. registry split is recorded; the other registry writers take the lock ---------------


def test_registry_split_is_recorded_like_a_merge(ws):
    _fresh(ws, "S01E01")
    r = CliRunner().invoke(app, ["registry", "split", str(ws.root), "--episode", "S01E01",
                                 "--cluster", "S0"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert _kinds(ws) == ["write-ahead", "completion"]
    first = log_lines(ws)[0]
    assert first["registry"] == {"command": "split", "episode": "S01E01", "cluster": "S0"}
    a = read_json(ws.work_dir("S01E01") / "05-characters" / "assignments.json", Assignments)
    assert [(c.cluster, c.actor_id, c.character_id) for c in a.clusters][0] == ("S0", 4, 4)
    assert "characters" not in stages_left(ws, "S01E01")


def _run_holding(ws: Workspace):
    path = apply_mod.studio_dir(ws) / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a")
    fcntl.flock(f, fcntl.LOCK_SH)
    return f


@pytest.mark.parametrize(
    "args",
    [
        ["split", "--episode", "S01E01", "--cluster", "S0"],
        ["rename", "3", "Janny"],
        ["alias", "3", "JANNY"],
        ["lock", "--character", "3"],
        ["kind", "3", "narrator"],
        ["gender", "3", "f"],
    ],
)
def test_registry_writers_are_refused_while_a_run_is_in_progress(ws, args):
    held = _run_holding(ws)
    try:
        r = CliRunner().invoke(app, ["registry", args[0], str(ws.root), *args[1:]])
    finally:
        held.close()
    assert r.exit_code == 1 and "a `fukikae run` is in progress" in r.output, r.output
    c = char(ws, 3)
    assert (c.name, c.locked, c.kind, c.gender) == (None, False, "character", None)
    with reg_of(ws) as reg:
        assert len(reg.characters()) == 3 and reg.aliases(3) == []


# --- 6. messages ---------------------------------------------------------------------------


def test_an_unreadable_registry_is_named_as_such_by_the_run_guard(ws, monkeypatch):
    _crash_after_commit(monkeypatch)
    with pytest.raises(SystemExit):
        apply_decision(ws, decision_for(ws, "lock", character=3))
    monkeypatch.undo()
    for p in ws.registry_path.parent.glob(ws.registry_path.name + "*"):
        p.unlink()
    ws.registry_path.write_bytes(b"not a database, not at all" * 100)
    msg = unfinished_message(ws)
    assert msg is not None and "not a readable registry" in msg
    assert "mid-change" not in msg


class _SelfMarking(_Stage):
    """A stage marked stale while it runs (a decision made meanwhile, with no studio lock)."""

    def __init__(self, ws: Workspace) -> None:
        super().__init__("characters")
        self.ws = ws

    def run(self, ep, cfg) -> None:
        super().run(ep, cfg)
        job.mark_stale(episode(self.ws, ep.id), "characters")


def test_a_stage_marked_stale_while_it_runs_is_not_recorded_done(ws):
    _fresh(ws, "S01E01")
    ep = _ep(ws, "S01E01")
    assert job.run_stages(ep, ws.config, [_SelfMarking(ws)], from_stage="characters")
    assert "characters" not in stages_left(ws, "S01E01")
    assert job.run_stages(ep, ws.config, [_Stage("characters")]) == ["characters"]
    assert "characters" in stages_left(ws, "S01E01")
    assert job.load_manifest(ep).stale == {}


def _tier_card(ws: Workspace, tmp_path: Path, origin: tuple[str, float, float], tier: str,
               seed: int) -> str:  # fmt: skip
    """A clone card whose ``[meta] reference_tier`` is ``tier``, as `select.clone_card`
    writes it."""
    from dataclasses import replace

    from fukikae.cards.store import CardStore

    cid = _card(ws, tmp_path, [origin], seed=seed)
    store = CardStore(ws.root)
    card = store.get(cid)
    return store.put(replace(card, meta={**card.meta, "reference_tier": tier}))


@pytest.mark.usefixtures("fake_backends")
def test_accept_takes_a_thin_cards_overlapping_own_turn_reference_with_a_warning(ws, tmp_path):
    _turns(ws, "S01E01", [("S0", 0.0, 10.0), ("S2", 2.0, 8.0)])
    thin = _tier_card(ws, tmp_path, ("S01E01", 3.0, 7.0), "thin", seed=1)
    with reg_of(ws) as reg:
        reg.assign_card(1, "en", thin, "auto", False)
    r = apply_decision(ws, accept(ws, thin))
    warning = "overlaps another speaker's turn: S01E01 3.00-7.00"
    assert any(warning in ln for ln in r.lines)
    with reg_of(ws) as reg:
        a = reg.card_for(1, "en")
        assert a is not None
        assert a.card_id == thin and a.note is not None and warning in a.note
    outside = _tier_card(ws, tmp_path, ("S01E01", 10.5, 11.5), "thin", seed=2)
    with pytest.raises(DecisionRefused, match="not inside"):
        apply_decision(ws, accept(ws, outside))


@pytest.mark.usefixtures("fake_backends")
@pytest.mark.parametrize("tier", ["floor", "short"])
def test_accept_refuses_an_overlapping_reference_on_a_card_that_is_not_thin(ws, tmp_path, tier):
    _turns(ws, "S01E01", [("S0", 0.0, 10.0), ("S2", 2.0, 8.0)])
    cid = _tier_card(ws, tmp_path, ("S01E01", 3.0, 7.0), tier, seed=3)
    with pytest.raises(DecisionRefused, match="overlaps another speaker"):
        apply_decision(ws, accept(ws, cid))
