"""The `[review]` gates `fukikae run` checks before translate and tts (Task 21).

The workspace is the studio tests' (`tests.studio.test_apply.ws`): characters 1, 2 and 3, each
cluster speaking ten seconds, character 3 in both episodes (S01E01: 1 and 3; S01E02: 2 and 3).
"""

# ruff: noqa: ARG002
import shutil
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from fukikae.artifacts import atomic_path, read_json, write_json
from fukikae.config import load_config
from fukikae.job import ReviewRequired, run_episodes, run_stages
from fukikae.models import Assignments
from fukikae.names.inputs import names_path, names_report_path
from fukikae.names.models import (
    DisagreeWith,
    KeyDecision,
    KeyDisagreement,
    NamesReport,
    NamesResult,
    VoiceSplit,
)
from fukikae.registry.invalidate import episode, remap_assignments
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
    """No impure card and no bimodal cluster unless a test says otherwise (the fixture
    workspace has no vocals to measure); names flags only where a test writes names."""
    from fukikae.cards import auto

    monkeypatch.setattr(auto, "impure_cards", lambda *_a, **_k: {})
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


def test_a_stock_or_empty_card_never_satisfies_the_voices_gate(ws, monkeypatch):
    """D41: a stock voice is never the character's voice, whoever chose or locked it; and an
    assignment with no card is no voice at all."""
    from dataclasses import replace

    from fukikae.cards.stock import revision, stock_card
    from fukikae.cards.store import CardStore

    cfg = cfg_of(ws, "voices = true\nmin_seconds = 5")
    stock = CardStore(ws.root).put(stock_card("kokoro", "af_heart", "en", revision("kokoro")))
    with reg_of(ws) as reg:
        reg.assign_card(1, "en", stock, by="studio", locked=True)
        reg.assign_card(3, "en", CARD, by="studio", locked=True)
    got = voices(ws, E1, cfg)
    assert who(got) == ["character 1 (unnamed)"] and "stock" in got[0]

    real = Registry.card_for

    def no_card(self, cid, lang):
        a = real(self, cid, lang)
        return replace(a, card_id=None) if a is not None and cid == 3 else a

    monkeypatch.setattr(Registry, "card_for", no_card)
    assert who(voices(ws, E1, cfg)) == ["character 1 (unnamed)", "character 3 (unnamed)"]


def test_short_and_ignored_characters_never_hold_an_episode(ws):
    # every character speaks 10 s per episode: under the default 30 s, none is a main one
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


def names_used(
    ws: Workspace,
    ep: str,
    *,
    how: str = "match",
    seconds: float = 5.0,
    name: str = "unnamed",
    splits: dict | None = None,
) -> None:
    """Names used on ``ep`` (S01E01), with two flags: key 'lucy' decided for character 1 on
    S0 and S2, stale since the assignments give S2 to character 3; and key 'erza' named onto
    character 3 while the acoustics give its S2 to character 1 (``how``). ``seconds`` and
    ``name`` change only the report's wording, as a rerun of the characters stage does."""
    work = ws.work_dir(ep)
    names_path(work).parent.mkdir(parents=True, exist_ok=True)
    write_json(names_path(work), NamesResult(
        episode=ep, used=True, verdict="full", reason=None, source=None, offset_s=0.0,
        coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[], excluded=[],
        dropped_row_seconds=0.0, suggestions=[], voice_splits=splits or {},
    ))  # fmt: skip
    keys = [
        KeyDecision(key="lucy", how="known", character_id=1, display="Lucy", seconds=20.0,
                    labels=["S0", "S2"]),
        KeyDecision(key="erza", how="known", character_id=3, display="Erza", seconds=10.0,
                    labels=["S2"]),
    ]  # fmt: skip
    line = (f"'erza' -> character 3 ({name}); acoustics disagree on {seconds:.1f} s of its "
            f"10.0 s: character 1 ({name}) by {how} {seconds:.1f} s (S2)")  # fmt: skip
    dis = [KeyDisagreement(key="erza", character_id=3, seconds=seconds,
                           against=[DisagreeWith(character_id=1, how=how, labels=["S2"],
                                                 seconds=seconds)])]  # fmt: skip
    write_json(names_report_path(work),
               NamesReport(episode=ep, keys=keys, lines=[line], disagreements=dis))  # fmt: skip


def apply_review(ws: Workspace, *cids: int) -> None:
    """The operator's `reviewed` decisions, through `studio apply` (which records the names
    flags' evidence the review saw)."""
    from fukikae.studio.apply import apply_decision, stamp
    from fukikae.studio.model import parse_decision

    for cid in cids:
        with reg_of(ws) as reg:
            d = stamp(reg, parse_decision({"op": "reviewed", "character": cid}))
        apply_decision(ws, d)


def who(reasons: list[str]) -> list[str]:
    return [r.split(":")[0] for r in reasons]


def test_names_flags_stop_an_unreviewed_flag_and_pass_a_reviewed_one(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    names_used(ws, E1)
    review(ws, 1, 3)  # reviewed, but with no evidence recorded: the flags are news
    got = identity(ws, E1, cfg)
    assert who(got) == ["character 1 (unnamed)", "character 3 (unnamed)"]
    assert "'lucy' stale: the assignments give S2 to character 3" in got[0]
    assert got[0].endswith("(raised since its review)")
    assert "acoustics disagree" in got[1]
    apply_review(ws, 1, 3)  # the studio review sees them
    assert identity(ws, E1, cfg) == []


def test_a_rename_and_a_rerun_keep_the_gate_open(ws):
    """The evidence is the flags' identity, not their text or the files' times: a name and
    other seconds in a rewritten report are the same flags."""
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    names_used(ws, E1)
    apply_review(ws, 1, 3)
    with reg_of(ws) as reg:
        reg.rename_character(1, "Lucy")
    names_used(ws, E1, seconds=6.5, name="Lucy")  # the characters stage ran again
    path = ws.work_dir(E1) / "05-characters" / "assignments.json"
    write_json(path, read_json(path, Assignments))
    assert identity(ws, E1, cfg) == []


def test_a_merge_of_another_character_keeps_the_gate_open(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    names_used(ws, E1)
    apply_review(ws, 1, 3)
    with reg_of(ws) as reg:
        chars = reg.merge_actors(3, 2)  # character 2 (S01E02 only) into 3
    for ep in (E1, E2):  # every episode's assignments rewritten by the remap
        remap_assignments(ws, ep, {2: 3}, character_map=chars)
    assert identity(ws, E1, cfg) == []


def test_a_merge_decision_records_the_evidence_its_remap_leaves(ws):
    """A merge reviews its keeper inside the decision's transaction, before the remap step
    rewrites the assignments: the evidence is read as the remap will leave them."""
    from fukikae.studio.apply import apply_decision, stamp
    from fukikae.studio.model import parse_decision

    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    names_used(ws, E1)
    path = names_report_path(ws.work_dir(E1))
    rep = read_json(path, NamesReport)
    # key 'zoe' decided for character 3 on S0 (character 1's now) and SX (gone from the
    # assignments): after 1 merges into 3, only SX is stale
    zoe = KeyDecision(key="zoe", how="known", character_id=3, display="Zoe", seconds=5.0,
                      labels=["S0", "SX"])  # fmt: skip
    write_json(path, rep.model_copy(update={"keys": [*rep.keys, zoe]}))
    with reg_of(ws) as reg:
        d = stamp(reg, parse_decision({"op": "merge", "keep": 3, "drop": 1}))
    apply_decision(ws, d)
    assert identity(ws, E1, cfg) == []


def test_a_moved_workspace_keeps_the_gate_open(ws, tmp_path):
    cfg_of(ws, "identity = true\nmin_seconds = 5")
    names_used(ws, E1)
    apply_review(ws, 1, 3)
    moved = Workspace(tmp_path / "elsewhere")
    shutil.copytree(ws.root, moved.root)
    assert identity(moved, E1, cfg_of(moved, "identity = true\nmin_seconds = 5")) == []


def test_a_changed_disagreement_stops(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    names_used(ws, E1)
    apply_review(ws, 1, 3)
    names_used(ws, E1, how="pin")  # the acoustics give S2 to character 1 another way now
    got = identity(ws, E1, cfg)
    assert who(got) == ["character 3 (unnamed)"]
    assert "by pin" in got[0] and got[0].endswith("(raised since its review)")


def test_an_episode_reviewed_before_its_names_ran_stops(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    apply_review(ws, 1, 3)  # no names on S01E01 yet: nothing to see
    assert identity(ws, E1, cfg) == []
    names_used(ws, E1)  # the names stage ran since
    assert who(identity(ws, E1, cfg)) == ["character 1 (unnamed)", "character 3 (unnamed)"]


def test_an_impure_card_picked_since_the_review_reopens_the_gate(ws, monkeypatch):
    from fukikae.cards import auto

    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    past = datetime.now(UTC) - timedelta(hours=1)
    review(ws, 1, 3, when=past)
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


def test_a_skipped_split_is_a_note_only_when_the_gate_stops(ws):
    cfg = cfg_of(ws, "identity = true\nmin_seconds = 5")
    skipped = VoiceSplit(
        male_below=165.0,
        female_above=200.0,
        seconds={},
        larger=None,
        labels=[],
        reason="the diarize cluster changed",
        check="changed",
    )
    names_used(ws, E1, splits={"S2": skipped})
    apply_review(ws, 1, 3)
    assert identity(ws, E1, cfg) == []  # never a stop by itself
    with reg_of(ws) as reg:  # a review that saw no names flags: character 1's is news
        reg.mark_reviewed(1, digests={})
    got = identity(ws, E1, cfg)
    assert who(got[:1]) == ["character 1 (unnamed)"]
    assert got[1:] == [f"note: {E1} S2 (character 3): pitch split skipped: the diarize "
                       "cluster changed (a split suggestion never stops a run)"]  # fmt: skip


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
    assert f"fukikae studio apply {ws.root} FILE" in r.output and '"reviewed"' in r.output
    assert "Traceback" not in r.output
    status = CliRunner().invoke(cli.app, ["status", str(ws.root)])
    assert "stopped at" in status.output and "failed" not in status.output, status.output

    review(ws, 2)  # S01E02 stops for character 3, S01E01 for 1 and 3
    r = CliRunner().invoke(cli.app, ["run", str(ws.root), E1, E2])
    assert r.exit_code == 3, r.output
    assert f"{E1}: stopped before translate by [review] identity" in r.output
    assert f"{E2}: stopped before translate by [review] identity" in r.output
    assert r.output.count(f"fukikae studio {ws.root}") == 1  # the hint once per run
    status = CliRunner().invoke(cli.app, ["status", str(ws.root)])
    assert "stopped at" in status.output and "failed" not in status.output, status.output

    review(ws, 1, 3)
    r = CliRunner().invoke(cli.app, ["run", str(ws.root), E1, E2])
    assert r.exit_code == 0, r.output
