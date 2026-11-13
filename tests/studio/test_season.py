"""The season index: characters across episodes, samples, gender, cards, suggestions, flags
and the review queue, built from the workspace without writing to the registry (Task 16)."""

from __future__ import annotations

import json
import sqlite3
import stat
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from fukikae.artifacts import write_json
from fukikae.cards.card import Card, Reference, Voice
from fukikae.cards.store import CardStore
from fukikae.config import load_config
from fukikae.job import Manifest, StageRecord, save_manifest
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    DubCues,
    Segment,
    Transcript,
    TtsPlan,
    TtsPlanCue,
    Turn,
)
from fukikae.registry.store import Registry
from fukikae.workspace import Workspace
from tests.fixtures.speechlike import speech

RATE = 16000
E1, E2, E3 = "S01E01", "S01E02", "S01E03"

# character -> (centroid, kind, name)
CHARS: dict[int, tuple[list[float], str, str | None]] = {
    1: ([1.0, 0.0, 0.0, 0.0], "character", "Lucy"),
    2: ([0.0, 1.0, 0.0, 0.0], "character", "Natsu"),
    3: ([0.0, 0.0, 1.0, 0.0], "character", None),
    4: ([0.95, 0.05, 0.0, 0.0], "narrator", "Narrator"),  # near Lucy, but a narrator
    5: ([0.8, 0.0, 0.6, 0.0], "character", None),  # cosine 0.8 to Lucy, 0.6 to char 3
}
# episode -> cluster -> (character, [(start, end, hz)])
SPEECH: dict[str, dict[str, tuple[int, list[tuple[float, float, float]]]]] = {
    E1: {
        "S0": (1, [(2, 5, 230), (10, 14, 230), (60, 63, 230), (70, 76, 230)]),
        "S1": (2, [(20, 24, 120), (26, 30, 120)]),
        "S2": (3, [(40, 44, 120), (45, 49, 120), (80, 84, 230), (85, 89, 230)]),
    },
    E2: {
        "S0": (1, [(2, 6, 230), (8, 11, 230), (60, 67, 230), (70, 72, 230)]),
        "S1": (2, [(20, 25, 120)]),
        "S4": (4, [(30, 36, 190)]),
        "S5": (5, [(40, 46, 220)]),
    },
}
STAGES = ("extract", "separate", "language", "asr", "asrclean", "diarize", "names", "characters")


def _wav(path: Path, x: np.ndarray, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _manifest(ws: Workspace, ep: str, stages: tuple[str, ...]) -> None:
    from fukikae.registry.invalidate import episode

    now = datetime.now(UTC)
    save_manifest(
        episode(ws, ep),
        Manifest(stages={s: StageRecord(key="k", status="done", started=now) for s in stages}),
    )


def _clone(ws: Workspace, store: CardStore, ep: str, spans: list[tuple[float, float]],
           tier: str | None = None) -> str:  # fmt: skip
    refs = []
    for i, (a, b) in enumerate(spans):
        clip = _wav(ws.root / "tmp" / f"clip-{ep}-{a}-{i}.wav", speech(230, b - a, seed=i))
        refs.append(Reference(
            clip=store.add_sidecar(clip, "clip"), transcript="x", language="ja",
            origin={"episode": ep, "start": a, "end": b},
        ))  # fmt: skip
    return store.put(Card(
        target_language="en", meta={"reference_tier": tier} if tier else {},
        voice=Voice(backend="clonefake", model_revision="rev-1", source="clone", seed=1234,
                    references=tuple(refs)),
    ))  # fmt: skip


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "fukikae.toml").write_text(
        'source_language = "ja"\ntarget_language = "en"\n', encoding="utf-8"
    )
    w = Workspace(root)
    with Registry(w.registry_path) as reg:
        for cid, (vec, kind, name) in CHARS.items():
            aid, got = reg.create_actor_with_character(np.array(vec, dtype=np.float32))
            assert got == cid
            if name:
                reg.rename_character(cid, name)
            if kind != "character":
                reg.set_character_kind(cid, kind)
    for ep, clusters in SPEECH.items():
        work = w.work_dir(ep)
        vocals = np.zeros(100 * RATE, np.float32)
        turns: list[Turn] = []
        rows: list[ClusterAssignment] = []
        segs: list[Segment] = []
        cues: list[DubCue] = []
        for cluster, (cid, spans) in clusters.items():
            rows.append(ClusterAssignment(
                cluster=cluster, actor_id=cid, character_id=cid, score=0.9,
                clean_seconds=sum(b - a for a, b, _ in spans), status="matched",
            ))  # fmt: skip
            for i, (a, b, hz) in enumerate(spans):
                x = speech(hz, b - a, seed=int(a * 10) + i, voiced_share=0.7)
                vocals[int(a * RATE) : int(a * RATE) + len(x)] = x
                turns.append(Turn(start=a, end=b, speaker=cluster))
                segs.append(Segment(start=a, end=b, text=f"{ep} line at {a:g}"))
                cues.append(DubCue(
                    id=f"c{a:g}", start=a, end=b, character_id=cid, extra=None, speaker="x",
                    source_text="src", text="line", budget=40,
                    delivery=Delivery(reference=(a, b)),
                ))  # fmt: skip
        turns.sort(key=lambda t: t.start)
        _wav(work / "01-separate" / "vocals.wav", vocals)
        for d in ("03-asr", "05-characters", "06-translate", "07-tts"):
            (work / d).mkdir(parents=True, exist_ok=True)
        write_json(work / "03-asr" / "transcript.json", Transcript(
            language="ja", backend="fake", word_timing=False, segments=segs,
        ))  # fmt: skip
        write_json(work / "05-characters" / "assignments.json", Assignments(
            episode=ep, threshold=0.5, clusters=rows, turns=turns,
        ))  # fmt: skip
        write_json(work / "06-translate" / "cues.en.json", DubCues(
            episode=ep, source_language="ja", target_language="en", path="asr", cues=cues,
        ))  # fmt: skip
        _manifest(w, ep, (*STAGES, "translate", "tts") if ep == E1 else STAGES)
    _manifest(w, E3, ("extract", "separate"))  # not through characters yet
    return w


def _cards(ws: Workspace) -> dict[int, str]:
    """Lucy a thin automatic clone, Natsu an automatic stock card, char 5 a floor clone."""
    store = CardStore(ws.root)
    thin = _clone(ws, store, E1, [(2.0, 5.0)], tier="thin")
    stock = store.put(Card(
        target_language="en",
        voice=Voice(backend="kokoro", model_revision="rev-k", source="stock", voice="am_echo"),
    ))  # fmt: skip
    floor = _clone(ws, store, E2, [(40.0, 46.0), (40.0, 46.0)], tier="floor")
    with Registry(ws.registry_path) as reg:
        reg.assign_card(1, "en", thin, by="auto", locked=False)
        reg.assign_card(2, "en", stock, by="auto", locked=False)
        reg.assign_card(5, "en", floor, by="auto", locked=False)
    return {1: thin, 2: stock, 5: floor}


def _build(ws: Workspace, **kw):
    from fukikae.studio.season import build_season

    return build_season(ws, load_config(ws.root, user_config=None), **kw)


def _entry(doc, cid: int):
    return next(c for c in doc.characters if c.id == cid)


def test_season_lists_every_character_across_episodes(ws):
    doc = _build(ws)
    assert sorted(c.id for c in doc.characters) == [1, 2, 3, 4, 5]
    lucy = _entry(doc, 1)
    assert sorted(lucy.episodes) == [E1, E2]
    assert lucy.episodes[E1].clusters == ["S0"]
    assert lucy.episodes[E1].turns == 4
    assert lucy.seconds == pytest.approx(16 + 16)
    assert lucy.turns == 8 and lucy.cues == 8
    assert (lucy.name, lucy.key, lucy.kind) == ("Lucy", "lucy", "character")
    assert lucy.gender.estimate == "f" and lucy.gender.per_episode == {E1: "f", E2: "f"}
    assert _entry(doc, 2).gender.estimate == "m"
    assert _entry(doc, 4).kind == "narrator"
    assert len(lucy.digest) == 64
    assert doc.schema_version == 1 and doc.workspace == str(ws.root)


def test_episode_without_characters_is_listed_with_its_command(ws):
    doc = _build(ws)
    eps = {e.id: e for e in doc.episodes}
    assert set(eps) == {E1, E2, E3}
    assert eps[E1].ready and eps[E1].command is None and eps[E1].last_stage == "tts"
    assert eps[E2].last_stage == "characters"
    assert not eps[E3].ready and eps[E3].last_stage == "separate"
    assert eps[E3].command == f"fukikae run {ws.root} {E3} --to characters"


def test_samples_spread_over_episodes_and_turns(ws):
    doc = _build(ws, clips=4)
    lucy = _entry(doc, 1)
    got = [(s.episode, s.start) for s in lucy.samples]
    # two per episode; in each, the longest turn of each half of the episode's turns
    assert got == [(E1, 10.0), (E1, 70.0), (E2, 2.0), (E2, 60.0)]
    assert lucy.samples[0].transcript == f"{E1} line at 10"
    audio = ws.root / "studio" / "audio"
    assert stat.S_IMODE(audio.stat().st_mode) == 0o700
    for s in lucy.samples:
        path = audio / s.clip
        assert path.is_file()
        with wave.open(str(path)) as w:
            assert (w.getframerate(), w.getnchannels()) == (24000, 1)


def test_cut_sample_is_cached_by_its_span(ws):
    from fukikae.studio.season import cut_sample

    a = cut_sample(ws, E1, 2.0, 5.0)
    b = cut_sample(ws, E1, 2.0, 5.0)
    c = cut_sample(ws, E1, 10.0, 14.0)
    assert a == b != c and a.parent == ws.root / "studio" / "audio"


def test_bimodal_cluster_gets_a_split_suggestion(ws):
    doc = _build(ws)
    three = _entry(doc, 3)
    splits = [s for s in three.suggestions if s.kind == "split"]
    assert len(splits) == 1
    s = splits[0]
    assert (s.episode, s.cluster) == (E1, "S2")
    assert s.parts is not None
    assert s.parts["m"]["seconds"] == pytest.approx(8.0)
    assert s.parts["f"]["seconds"] == pytest.approx(8.0)
    assert {x["start"] for x in s.parts["m"]["samples"]} <= {40.0, 45.0}
    assert {x["start"] for x in s.parts["f"]["samples"]} <= {80.0, 85.0}
    assert any(f.kind == "bimodal" and f.episode == E1 for f in three.flags)
    assert not [s for s in _entry(doc, 1).suggestions if s.kind == "split"]


def test_merge_suggestion_above_threshold_only_and_never_with_a_narrator(ws):
    doc = _build(ws)
    merges = {c.id: [(s.other, s.cosine) for s in c.suggestions if s.kind == "merge"]
              for c in doc.characters}  # fmt: skip
    lucy = dict(merges[1])
    assert set(lucy) == {5}  # not the narrator (0.998), not Natsu (0.0)
    assert lucy[5] == pytest.approx(0.8, abs=1e-6)
    assert dict(merges[5]) == pytest.approx({1: 0.8, 3: 0.6}, abs=1e-6)
    assert merges[4] == []
    assert all(4 not in dict(m) for m in merges.values())


def test_flags_include_thin_stock_impure_gated_and_dropped(ws, monkeypatch):
    from fukikae.cards import auto

    ids = _cards(ws)
    real = auto.impure_cards

    def impure(ws_, reg, store, language, characters=None, **kw):
        out = real(ws_, reg, store, language, characters, **kw)
        out[5] = auto.Impurity(ids[5], ({
            "episode": E2, "start": 40.0, "end": 46.0, "gate": "unvoiced",
            "gates": ["unvoiced"], "median": None, "voiced_fraction": 0.1,
        },))  # fmt: skip
        return out

    monkeypatch.setattr(auto, "impure_cards", impure)
    w1 = ws.work_dir(E1)
    write_json(w1 / "07-tts" / "plan.json", TtsPlan(episode=E1, cues=[
        TtsPlanCue(cue_id="c20", character_id=2, extra=None, card_id=ids[2], slot=4.0,
                   gender_gate=True),
        TtsPlanCue(cue_id="c26", character_id=2, extra=None, card_id=ids[2], slot=4.0),
    ]))  # fmt: skip
    dropped = ws.work_dir(E2) / "03b-asrclean" / "dropped.json"
    dropped.parent.mkdir(parents=True, exist_ok=True)
    dropped.write_text(json.dumps([
        {"start": 60.5, "end": 62.0, "text": "ご視聴ありがとうございました", "reason": "x",
         "kind": "segment"},
    ]), encoding="utf-8")  # fmt: skip
    doc = _build(ws)

    def kinds(cid: int) -> set[str]:
        return {f.kind for f in _entry(doc, cid).flags}

    assert "thin" in kinds(1) and "dropped" in kinds(1)
    assert {"stock", "gated"} <= kinds(2)
    assert "impure" in kinds(5)
    lucy_card = next(c for c in _entry(doc, 1).cards if c.language == "en")
    assert (lucy_card.card_id, lucy_card.tier, lucy_card.source) == (ids[1], "thin", "clone")
    assert "thin" in lucy_card.flags
    natsu_card = next(c for c in _entry(doc, 2).cards if c.language == "en")
    assert natsu_card.source == "stock" and "stock" in natsu_card.flags
    gated = next(f for f in _entry(doc, 2).flags if f.kind == "gated")
    assert gated.episode == E1 and "1 line" in gated.text
    dropped_flag = next(f for f in _entry(doc, 1).flags if f.kind == "dropped")
    assert dropped_flag.episode == E2
    # the index never proposes a stock voice: no suggestion names one
    assert all(s.kind in ("merge", "split") for c in doc.characters for s in c.suggestions)


def test_queue_orders_flags_before_unreviewed_before_seconds():
    from fukikae.studio.model import CharacterEntry, Flag, GenderInfo
    from fukikae.studio.season import order_queue

    def entry(cid: int, seconds: float, flags: list[str], reviewed: str | None = None,
              kind: str = "character") -> CharacterEntry:  # fmt: skip
        return CharacterEntry(
            id=cid, actor_id=cid, name=None, key=None, external_id=None, kind=kind,
            locked=False, reviewed=reviewed, digest="0" * 64, episodes={}, seconds=seconds,
            clean_seconds=seconds, turns=1, cues=0, samples=[],
            gender=GenderInfo(estimate="u", confidence=0.0, median=None, seconds={},
                              per_episode={}, stored=None, stored_by=None),
            cards=[], suggestions=[], names=[],
            flags=[Flag(kind=k, text=k, episode=None) for k in flags],
        )  # fmt: skip

    entries = [
        entry(1, 500.0, []),  # unreviewed, no flag
        entry(2, 10.0, ["floor"]),
        entry(3, 5.0, ["thin", "floor"]),  # its worst flag counts
        entry(4, 900.0, [], reviewed="2026-11-12T00:00:00+00:00"),
        entry(5, 50.0, ["stock"]),
        entry(6, 20.0, ["names"]),
        entry(7, 30.0, ["impure"]),
        entry(8, 40.0, ["bimodal"]),
        entry(9, 60.0, ["gated"]),
        entry(10, 100.0, []),
        entry(11, 1000.0, ["thin"], kind="ignored"),  # decided: last
    ]
    assert order_queue(entries) == [3, 5, 7, 8, 9, 6, 2, 1, 10, 4, 11]


def test_index_is_rebuilt_only_when_inputs_change(ws, monkeypatch):
    from fukikae.studio import season

    calls: list[int] = []
    real = season.build_season

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(season, "build_season", counting)
    cfg = load_config(ws.root, user_config=None)
    first = season.load_or_build(ws, cfg)
    assert season.season_path(ws).is_file() and len(calls) == 1
    again = season.load_or_build(ws, cfg)
    assert len(calls) == 1 and again.inputs_digest == first.inputs_digest
    with Registry(ws.registry_path) as reg:
        reg.rename_character(3, "Janny")
    renamed = season.load_or_build(ws, cfg)
    assert len(calls) == 2 and _entry(renamed, 3).name == "Janny"
    asg = ws.work_dir(E2) / "05-characters" / "assignments.json"
    asg.write_text(asg.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    season.load_or_build(ws, cfg)
    assert len(calls) == 3
    season.load_or_build(ws, cfg)
    assert len(calls) == 3


def test_building_the_index_writes_nothing_to_the_registry(ws):
    _cards(ws)

    def dump() -> list[str]:
        con = sqlite3.connect(ws.registry_path)
        try:
            return list(con.iterdump())
        finally:
            con.close()

    before = dump()
    _build(ws)
    assert dump() == before


def _names_used(monkeypatch, hints: list) -> None:
    """Every ready episode's names stage used names, and `merge_hints` answers ``hints``."""
    from fukikae.cards import auto
    from fukikae.names import flags as nflags

    class Used:
        used = True

        def visible_lines(self) -> list:
            return []

    monkeypatch.setattr(auto, "used_names", lambda _work: Used())
    monkeypatch.setattr(nflags, "disagreement_rows", lambda *_a: [])
    monkeypatch.setattr(nflags, "merge_hints", lambda *_a: list(hints))


def _hint(keep: int, drop: int):
    from fukikae.names.flags import MergeHint

    return MergeHint(keep, drop, keep, drop, "S2", 8.0, 8.0, 10.0)


def _merge_others(doc, cid: int) -> set[int]:
    return {s.other for s in _entry(doc, cid).suggestions if s.kind == "merge"}


def test_names_hints_suggest_a_merge(ws, monkeypatch):
    _names_used(monkeypatch, [_hint(keep=2, drop=3)])
    doc = _build(ws)
    assert 2 in _merge_others(doc, 3)


def test_names_hint_never_suggests_merging_an_ignored_character(ws, monkeypatch):
    with Registry(ws.registry_path) as reg:
        reg.set_character_kind(3, "ignored")
    _names_used(monkeypatch, [_hint(keep=1, drop=3)])
    doc = _build(ws)
    assert _merge_others(doc, 3) == set()


def test_names_hint_never_targets_an_ignored_character(ws, monkeypatch):
    with Registry(ws.registry_path) as reg:
        reg.set_character_kind(2, "ignored")
    _names_used(monkeypatch, [_hint(keep=2, drop=3)])
    doc = _build(ws)
    assert all(2 not in _merge_others(doc, c.id) for c in doc.characters)


def test_names_hint_never_pairs_a_narrator_with_a_dialogue_character(ws, monkeypatch):
    _names_used(monkeypatch, [_hint(keep=4, drop=3), _hint(keep=1, drop=4)])
    doc = _build(ws)
    assert 4 not in _merge_others(doc, 3)
    assert _merge_others(doc, 4) == set()


def test_missing_vocals_skip_that_episodes_samples(ws):
    (ws.work_dir(E2) / "01-separate" / "vocals.wav").unlink()
    doc = _build(ws, clips=4)
    eps = {e.id: e for e in doc.episodes}
    note = eps[E2].note
    assert eps[E2].ready and note is not None and "vocals" in note
    assert eps[E1].note is None
    lucy = _entry(doc, 1)
    assert E2 in lucy.episodes
    assert lucy.samples and {s.episode for s in lucy.samples} == {E1}
    assert len(lucy.samples) == 4
    assert all(s.episode != E2 for x in doc.extras for s in x.samples)


def test_missing_vocals_skip_an_extras_cue_sample(ws):
    from fukikae.models import TtsRun

    asg_path = ws.work_dir(E2) / "05-characters" / "assignments.json"
    asg = json.loads(asg_path.read_text(encoding="utf-8"))
    for row in asg["clusters"]:
        if row["cluster"] == "S5":
            row["status"] = "extra"
            row["character_id"] = None
    asg_path.write_text(json.dumps(asg), encoding="utf-8")
    (ws.work_dir(E2) / "01-separate" / "vocals.wav").unlink()
    write_json(ws.work_dir(E2) / "07-tts" / "run.json", TtsRun.model_validate({
        "episode": E2, "finished": "2026-11-13T00:00:00+00:00", "renders": 0,
        "rendered": [], "cached": [], "extras": {"x~c40": {"tier": "floor", "seconds": 6.0}},
    }))  # fmt: skip
    doc = _build(ws)
    got = {x.label: x for x in doc.extras if x.episode == E2}
    assert got["S5"].samples == [] and got["x~c40"].samples == []


def test_config_values_the_index_reads_change_its_digest(ws, tmp_path):
    from fukikae.studio.season import inputs_digest

    base = inputs_digest(ws, load_config(ws.root, user_config=None))
    user = tmp_path / "user.toml"
    user.write_text("[names]\nmin_new_seconds = 9.0\n", encoding="utf-8")
    assert inputs_digest(ws, load_config(ws.root, user_config=user)) != base
    assert inputs_digest(ws, load_config(ws.root, user_config=None)) == base


def test_a_rebuild_prunes_clips_no_longer_referenced(ws, tmp_path):
    first = _build(ws, clips=6)
    audio = ws.root / "studio" / "audio"
    used = {s.clip for c in first.characters for s in c.samples}
    stray = audio / ("0" * 64 + ".wav")
    stray.write_bytes(b"old clip")
    other = audio / "notes.txt"
    other.write_text("keep", encoding="utf-8")
    odd = audio / "keep.wav"
    odd.write_bytes(b"not ours")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    link = audio / ("1" * 64 + ".wav")
    link.symlink_to(outside)
    second = _build(ws, clips=2)
    kept = {s.clip for c in second.characters for s in c.samples}
    kept |= {s.clip for x in second.extras for s in x.samples}
    kept |= {smp["clip"] for c in second.characters for s in c.suggestions if s.parts
             for p in s.parts.values() for smp in p.get("samples", [])}  # fmt: skip
    assert not stray.exists()
    assert other.exists() and odd.exists() and link.is_symlink() and outside.exists()
    names = {p.name for p in audio.iterdir()}
    assert kept <= names
    assert not (used - kept) & names


def test_load_or_build_on_an_older_registry_migrates_it_and_writes_nothing_else(ws):
    import hashlib

    from fukikae.studio import season
    from tests.registry.test_store import _downgrade_to_4

    _downgrade_to_4(ws.registry_path)

    def files() -> dict[str, str]:
        return {
            str(p.relative_to(ws.root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ws.root.rglob("*") if p.is_file()
        }  # fmt: skip

    before = files()
    doc = season.load_or_build(ws, load_config(ws.root, user_config=None))
    assert sorted(c.id for c in doc.characters) == [1, 2, 3, 4, 5]
    con = sqlite3.connect(ws.registry_path)
    try:
        assert con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 5
    finally:
        con.close()
    after = files()
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    reg = str(ws.registry_path.relative_to(ws.root))
    allowed = ("studio/", ".fukikae/pitch-cache/", ".fukikae/song-cache/")
    assert {k for k in changed if not k.startswith(allowed) and not k.startswith(reg)} == set()
    assert all(before[k] == after[k] for k in before if not k.startswith((*allowed, reg)))


def test_queue_ranks_an_operator_gender_contradiction_first():
    from fukikae.studio.model import CharacterEntry, Flag, GenderInfo
    from fukikae.studio.season import order_queue

    def entry(cid: int, seconds: float, flags: list[str]) -> CharacterEntry:
        return CharacterEntry(
            id=cid, actor_id=cid, name=None, key=None, external_id=None, kind="character",
            locked=False, reviewed=None, digest="0" * 64, episodes={}, seconds=seconds,
            clean_seconds=seconds, turns=1, cues=0, samples=[],
            gender=GenderInfo(estimate="u", confidence=0.0, median=None, seconds={},
                              per_episode={}, stored=None, stored_by=None),
            cards=[], suggestions=[], names=[],
            flags=[Flag(kind=k, text=k, episode=None) for k in flags],
        )  # fmt: skip

    entries = [entry(1, 900.0, ["thin"]), entry(2, 5.0, ["gender"]), entry(3, 50.0, ["stale"])]
    assert order_queue(entries) == [2, 1, 3]
