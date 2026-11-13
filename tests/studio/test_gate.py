"""The `[review]` gates `fukikae run` checks before translate and tts (Task 21).

The workspace is the studio tests' (`tests.studio.test_apply.ws`): characters 1, 2 and 3, each
cluster speaking ten seconds, character 3 in both episodes (S01E01: 1 and 3; S01E02: 2 and 3).
"""

# ruff: noqa: ARG002
import os
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from fukikae.artifacts import atomic_path, read_json, write_json
from fukikae.config import load_config
from fukikae.job import ReviewRequired, run_episodes, run_stages
from fukikae.models import Assignments
from fukikae.registry.invalidate import episode
from fukikae.registry.store import Registry
from fukikae.studio import gate
from fukikae.workspace import Workspace

E1, E2 = "S01E01", "S01E02"
CARD = "ab" * 8


class Fake:
    """A stage that writes ``<name>.txt`` and records that it ran."""

    def __init__(self, name: str, ran: list[str]):
        self.name = name
        self.ran = ran

    def inputs(self, ep):
        return []

    def outputs(self, ep):
        return [ep.work / f"{self.name}.txt"]

    def config_key(self, cfg, ep):
        return {}

    def run(self, ep, cfg):
        self.ran.append(f"{self.name} {ep.id}")
        with atomic_path(ep.work / f"{self.name}.txt") as tmp:
            tmp.write_text(self.name)


def stages(ran: list[str]) -> list[Fake]:
    return [Fake(n, ran) for n in ("characters", "translate", "tts", "mix")]


def cfg_of(ws: Workspace, review: str = ""):
    (ws.root / "fukikae.toml").write_text(f"[review]\n{review}\n" if review else "")
    return load_config(ws.root, user_config=None)


def reg_of(ws: Workspace) -> Registry:
    return Registry(ws.registry_path)


def review(ws: Workspace, *cids: int, when: datetime | None = None) -> None:
    with reg_of(ws) as reg:
        for cid in cids:
            reg.mark_reviewed(cid, None if when is None else when.isoformat())


def identity(ws: Workspace, ep: str, cfg) -> list[str]:
    with reg_of(ws) as reg:
        return gate.needs_identity(episode(ws, ep), cfg, reg)


def voices(ws: Workspace, ep: str, cfg) -> list[str]:
    with reg_of(ws) as reg:
        return gate.needs_voices(episode(ws, ep), cfg, reg)


@pytest.fixture(autouse=True)
def no_flags(monkeypatch):
    """No impure card, no names flag and no bimodal cluster unless a test says otherwise (the
    fixture workspace has no vocals to measure)."""
    from fukikae.cards import auto

    monkeypatch.setattr(auto, "impure_cards", lambda *_a, **_k: {})
    monkeypatch.setattr(gate, "names_flags", lambda *_a, **_k: {})
    monkeypatch.setattr(gate, "bimodal_notes", lambda *_a, **_k: [])


def test_identity_gate_stops_before_translate(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    ran: list[str] = []
    with pytest.raises(ReviewRequired) as got:
        run_stages(episode(ws, E1), cfg, stages(ran), gates=gate.gates_for(cfg))
    assert ran == [f"characters {E1}"]
    assert (got.value.episode, got.value.stage) == (E1, "translate")
    assert got.value.reasons == ["character 1 (unnamed): never reviewed",
                                 "character 3 (unnamed): never reviewed"]  # fmt: skip


def test_voices_gate_stops_before_tts(ws):
    cfg = cfg_of(ws, "voices = true\nmin_seconds = 5")
    ran: list[str] = []
    with pytest.raises(ReviewRequired) as got:
        run_stages(episode(ws, E1), cfg, stages(ran), gates=gate.gates_for(cfg))
    assert ran == [f"characters {E1}", f"translate {E1}"]
    assert got.value.stage == "tts"
    assert [r.split(":")[0] for r in got.value.reasons] == ["character 1 (unnamed)",
                                                            "character 3 (unnamed)"]  # fmt: skip
    assert all("no en card" in r for r in got.value.reasons)


def test_reviewed_character_passes_on_the_next_episode(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    review(ws, 1, 3)  # while working on S01E01
    assert identity(ws, E1, cfg) == []
    # S01E02: character 3 carries its review; only character 2 is new
    assert identity(ws, E2, cfg) == ["character 2 (unnamed): never reviewed"]


def test_accepted_card_passes_the_voices_gate_on_the_next_episode(ws):
    cfg = cfg_of(ws, "voices = true\nmin_seconds = 5")
    with reg_of(ws) as reg:
        reg.assign_card(3, "en", CARD, by="studio", locked=True)  # accepted on S01E01
        reg.assign_card(2, "en", CARD, by="auto", locked=False)  # automatic: not enough
        reg.assign_card(2, "fr", CARD, by="studio", locked=True)  # another language
    got = voices(ws, E2, cfg)
    assert [r.split(":")[0] for r in got] == ["character 2 (unnamed)"]
    assert "automatic" in got[0]


def test_imported_and_locked_cards_pass_the_voices_gate(ws):
    cfg = cfg_of(ws, "voices = true\nmin_seconds = 5")
    with reg_of(ws) as reg:
        reg.assign_card(1, "en", CARD, by="import:pack", locked=False)
        reg.assign_card(3, "en", CARD, by="auto", locked=False)
        reg.lock_card(3, "en", True)
    assert voices(ws, E1, cfg) == []


def test_short_and_ignored_characters_never_hold_an_episode(ws):
    # every character speaks 10 s per episode: under the default 20 s, none is a main one
    cfg = cfg_of(ws, "identity = true\nvoices = true")
    assert identity(ws, E1, cfg) == [] and voices(ws, E1, cfg) == []
    cfg = cfg_of(ws, "identity = true\nvoices = true\nmin_seconds = 5")
    with reg_of(ws) as reg:
        reg.set_character_kind(1, "ignored")
        reg.set_character_kind(3, "narrator")  # a narrator is a main character
    assert identity(ws, E1, cfg) == ["character 3 (unnamed): never reviewed"]
    assert [r.split(":")[0] for r in voices(ws, E1, cfg)] == ["character 3 (unnamed)"]
    # an extras cluster (status extra or review) is not its character's speech
    path = ws.work_dir(E1) / "05-characters" / "assignments.json"
    asg = read_json(path, Assignments)
    asg = asg.model_copy(update={"clusters": [
        c.model_copy(update={"status": "extra"}) if c.character_id == 3 else c
        for c in asg.clusters
    ]})  # fmt: skip
    write_json(path, asg)
    assert identity(ws, E1, cfg) == []


def test_new_flag_after_review_reopens_the_identity_gate(ws, monkeypatch):
    from fukikae.cards import auto

    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    past = datetime.now(UTC) - timedelta(hours=1)
    review(ws, 1, 3, when=past)
    path = ws.work_dir(E1) / "05-characters" / "assignments.json"
    old = (past - timedelta(hours=1)).timestamp()
    os.utime(path, (old, old))
    monkeypatch.setattr(gate, "names_flags", lambda *_a, **_k: {1: ["'lucy' acoustics disagree"]})
    assert identity(ws, E1, cfg) == []  # raised before the review: the operator saw it
    os.utime(path)  # the characters stage wrote it again since
    got = identity(ws, E1, cfg)
    assert len(got) == 1 and got[0].startswith("character 1 (unnamed): 'lucy' acoustics disagree")

    os.utime(path, (old, old))
    with reg_of(ws) as reg:
        reg.assign_card(3, "en", CARD, by="auto", locked=False)  # after the review
    impurity = auto.Impurity(CARD, ({"episode": E1, "start": 10.0, "end": 14.0,
                                     "gate": "unvoiced", "gates": ["unvoiced"]},))  # fmt: skip
    monkeypatch.setattr(auto, "impure_cards", lambda *_a, **_k: {3: impurity})
    got = identity(ws, E1, cfg)
    assert len(got) == 1 and got[0].startswith("character 3 (unnamed): its en card")
    # reviewed again: the same card is no longer news
    review(ws, 3)
    assert identity(ws, E1, cfg) == []


def test_an_operator_card_never_reopens_the_identity_gate(ws, monkeypatch):
    from fukikae.cards import auto

    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    review(ws, 1, 3, when=datetime.now(UTC) - timedelta(hours=1))
    with reg_of(ws) as reg:
        reg.assign_card(3, "en", CARD, by="studio", locked=True)
    impurity = auto.Impurity(CARD, ({"episode": E1, "start": 10.0, "end": 14.0,
                                     "gate": "unvoiced", "gates": ["unvoiced"]},))  # fmt: skip
    monkeypatch.setattr(auto, "impure_cards", lambda *_a, **_k: {3: impurity})
    assert identity(ws, E1, cfg) == []


def test_a_split_suggestion_alone_never_stops_a_run(ws, monkeypatch):
    """Controller ruling 2026-11-13: bimodal clusters are information in the stop message,
    never a reason to stop."""
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    note = f"{E1} S2 (character 3): 6.0 s low, 4.0 s high; a pitch split may separate two voices"
    monkeypatch.setattr(gate, "bimodal_notes", lambda *_a, **_k: [note])
    review(ws, 1, 3)
    assert identity(ws, E1, cfg) == []
    got = identity(ws, E2, cfg)  # character 2 never reviewed: stops, with the note
    assert got[0] == "character 2 (unnamed): never reviewed"
    assert got[1:] == [f"note: {note} (a split suggestion never stops a run)"]


def test_multi_episode_run_continues_other_episodes(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    review(ws, 2, 3)  # S01E02's characters; S01E01's character 1 is not
    ran: list[str] = []
    eps = [episode(ws, E1), episode(ws, E2)]
    results = run_episodes(eps, cfg, stages(ran), tail=("mix",), gates=gate.gates_for(cfg))
    assert isinstance(results[E1].error, ReviewRequired)
    assert results[E1].ran == ["characters"]
    assert results[E2].error is None
    assert results[E2].ran == ["characters", "translate", "tts", "mix"]


def test_gates_off_never_stop(ws):
    cfg = cfg_of(ws)
    assert (cfg.review.identity, cfg.review.voices) == (False, False)
    assert gate.gates_for(cfg) == {}
    ran: list[str] = []
    assert run_stages(episode(ws, E1), cfg, stages(ran), gates=gate.gates_for(cfg)) == [
        "characters", "translate", "tts", "mix",
    ]  # fmt: skip


def test_an_up_to_date_stage_is_not_gated(ws):
    """A gate stops work about to be done: a translate already done for these inputs runs
    nothing, so it holds nothing up."""
    ran: list[str] = []
    run_stages(episode(ws, E1), cfg_of(ws), stages(ran))
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    assert run_stages(episode(ws, E1), cfg, stages(ran), gates=gate.gates_for(cfg)) == []


def test_no_registry_holds_nothing(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    w = Workspace(root)
    cfg = cfg_of(w, "identity = true\nvoices = true")
    gates = gate.gates_for(cfg)
    assert gates["translate"](episode(w, E1), cfg) == []
    assert gates["tts"](episode(w, E1), cfg) == []
    assert not w.registry_path.exists()


def test_run_command_exits_3_naming_the_studio(ws, monkeypatch):
    from fukikae import cli

    cfg_of(ws, "identity = true\nmin_seconds = 5")
    ran: list[str] = []
    monkeypatch.setattr(cli, "pipeline_stages", lambda _cfg: stages(ran))
    monkeypatch.setattr(Workspace, "find_episode", lambda self, ref: episode(self, ref))
    r = CliRunner().invoke(cli.app, ["run", str(ws.root), E1])
    assert r.exit_code == 3, r.output
    assert f"{E1}: stopped before translate by [review] identity" in r.output
    assert "character 1 (unnamed): never reviewed" in r.output
    assert f"fukikae studio {ws.root}" in r.output
    assert "Traceback" not in r.output

    review(ws, 2, 3)
    r = CliRunner().invoke(cli.app, ["run", str(ws.root), E1, E2])
    assert r.exit_code == 3, r.output
    assert f"{E2}: ran characters, translate, tts, mix" in r.output
    assert f"{E1}: stopped before translate by [review] identity" in r.output
