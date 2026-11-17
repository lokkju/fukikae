"""The season index: characters across episodes, samples, gender, cards, suggestions, flags
and the review queue, built from the workspace without writing to the registry (Task 16)."""

from __future__ import annotations

import json
import os
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


def test_review_settings_never_change_the_index_digest(ws):
    """`[review]` is read by `fukikae run`'s gates only: editing it rebuilds nothing."""
    from fukikae.studio.season import inputs_digest
    from fukikae.workspace import Workspace

    toml = ws.root / "fukikae.toml"
    before = toml.read_text(encoding="utf-8") if toml.exists() else ""
    base = inputs_digest(Workspace(ws.root), load_config(ws.root, user_config=None))
    toml.write_text(before + "\n[review]\nidentity = true\nmin_seconds = 5\n", encoding="utf-8")
    cfg = load_config(ws.root, user_config=None)
    assert cfg.review.identity and cfg.review.min_seconds == 5.0
    assert inputs_digest(Workspace(ws.root), cfg) == base


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


def test_a_rebuild_prunes_hour_old_temp_clips_a_killed_cut_left(ws):
    import time

    first = _build(ws, clips=2)
    assert first.characters
    audio = ws.root / "studio" / "audio"
    old = audio / f".{'a' * 64}.{'b' * 32}.tmp.wav"
    new = audio / f".{'c' * 64}.{'d' * 32}.tmp.wav"
    odd = audio / f".{'e' * 64}.tmp.wav"
    for f in (old, new, odd):
        f.write_bytes(b"half a clip")
    hour_ago = time.time() - 3700
    os.utime(old, (hour_ago, hour_ago))
    os.utime(odd, (hour_ago, hour_ago))
    _build(ws, clips=2)
    assert not old.exists()
    assert new.exists() and odd.exists()


def test_a_symlinked_studio_audio_is_refused_and_nothing_is_cut_through_it(ws, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (ws.root / "studio").mkdir(exist_ok=True)
    (ws.root / "studio" / "audio").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        _build(ws, clips=2)
    assert list(elsewhere.iterdir()) == []


def test_season_json_is_owner_only(ws):
    from fukikae.studio.season import load_or_build, season_path

    d = ws.root / "studio"
    d.mkdir(mode=0o755, exist_ok=True)
    os.chmod(d, 0o755)  # noqa: S103 (the mode the fix must tighten)
    load_or_build(ws, load_config(ws.root, user_config=None), clips=2)
    assert stat.S_IMODE(season_path(ws).stat().st_mode) == 0o600
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


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


def test_run_gates_read_the_same_speech_and_splits_as_the_index(ws):
    """The `[review]` gates (Task 21) on this workspace's real pitch: main characters by
    their seconds, the bimodal S2 named as a note and never a reason to stop."""
    from fukikae.registry.invalidate import episode
    from fukikae.studio import gate

    (ws.root / "fukikae.toml").write_text(
        'source_language = "ja"\ntarget_language = "en"\n[review]\nidentity = true\n'
        "voices = true\nmin_seconds = 10\n",
        encoding="utf-8",
    )
    cfg = load_config(ws.root, user_config=None)
    ep = episode(ws, E1)
    assert gate.episode_seconds(ep) == pytest.approx({1: 16.0, 2: 8.0, 3: 16.0})
    with Registry(ws.registry_path) as reg:
        assert gate.main_characters(ep, reg, cfg) == [1, 3]  # Natsu's 8 s hold nothing
        got = gate.needs_identity(ep, cfg, reg)
    assert got[:2] == [
        "character 1 (Lucy): never reviewed",
        "character 3 (unnamed): never reviewed",
    ]
    assert len(got) == 3 and got[2].startswith(
        f"note: {E1} S2 (character 3): 8.0 s low, 8.0 s high"
    )
    with Registry(ws.registry_path) as reg:
        reg.mark_reviewed(1)
        reg.mark_reviewed(3)
        assert gate.needs_identity(ep, cfg, reg) == []
        assert [r.split(":")[0] for r in gate.needs_voices(ep, cfg, reg)] == [
            "character 1 (Lucy)", "character 3 (unnamed)",
        ]  # fmt: skip


def _rewrite_assignments(ws: Workspace, ep: str, **by_cluster: dict) -> None:
    from fukikae.artifacts import read_json

    path = ws.work_dir(ep) / "05-characters" / "assignments.json"
    a = read_json(path, Assignments)
    clusters = [c.model_copy(update=by_cluster.get(c.cluster, {})) for c in a.clusters]
    write_json(path, a.model_copy(update={"clusters": clusters}))


def test_a_new_character_from_a_pitch_split_is_flagged_until_reviewed(ws):
    _rewrite_assignments(ws, E1, S2={"split": "new", "status": "new"})
    doc = _build(ws)
    flags = [f for f in _entry(doc, 3).flags if f.kind == "identity"]
    assert len(flags) == 1 and flags[0].episode == E1 and "S2" in flags[0].text
    assert not [f for f in _entry(doc, 1).flags if f.kind == "identity"]
    with Registry(ws.registry_path) as reg:
        reg.mark_reviewed(3)
    assert not [f for f in _entry(_build(ws), 3).flags if f.kind == "identity"]


def test_a_split_skipped_by_its_fingerprint_flags_the_clusters_character(ws):
    from fukikae.names.models import NamesResult, VoiceSplit

    d = ws.work_dir(E1) / "04c-names"
    d.mkdir(parents=True, exist_ok=True)
    split = VoiceSplit(
        male_below=165.0, female_above=200.0, seconds={"f": 0.0, "m": 0.0, "u": 0.0},
        larger=None, labels=[], reason="the split was decided on S0 with 3 turns, 9 s",
        check="changed",
    )  # fmt: skip
    write_json(d / "names.json", NamesResult(
        episode=E1, used=False, verdict="none", reason="no named track", source=None,
        offset_s=0.0, coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[],
        excluded=[], dropped_row_seconds=0.0, suggestions=[], voice_splits={"S0": split},
    ))  # fmt: skip
    doc = _build(ws)
    flags = [f for f in _entry(doc, 1).flags if f.kind == "split"]
    assert len(flags) == 1 and flags[0].episode == E1 and "decided on S0" in flags[0].text


def test_a_voice_part_against_its_characters_stored_gender_raises_a_gender_flag(ws):
    """A stored gender set after a split never moves the keeper; the index flags it."""
    from fukikae.artifacts import read_json

    path = ws.work_dir(E1) / "05-characters" / "assignments.json"
    a = read_json(path, Assignments)
    ren = {"S0": "S0~voice-m"}
    turns = [t.model_copy(update={"speaker": ren.get(t.speaker, t.speaker)}) for t in a.turns or []]
    clusters = [
        c.model_copy(update={"cluster": ren[c.cluster], "split": "kept"}) if c.cluster in ren
        else c
        for c in a.clusters
    ]  # fmt: skip
    write_json(path, a.model_copy(update={"clusters": clusters, "turns": turns}))
    assert not [f for f in _entry(_build(ws), 1).flags if f.kind == "gender"]
    with Registry(ws.registry_path) as reg:
        reg.set_character_gender(1, "f")
    flags = [f for f in _entry(_build(ws), 1).flags if f.kind == "gender"]
    assert len(flags) == 1 and flags[0].episode == E1 and "S0~voice-m" in flags[0].text
    with Registry(ws.registry_path) as reg:
        reg.set_character_gender(1, "m")
    assert not [f for f in _entry(_build(ws), 1).flags if f.kind == "gender"]


# -- video sample clips ----------------------------------------------------------------------


def _make_source(path: Path, *, video: bool = True) -> Path:
    """100 s at 64x48, 4 fps: video 0 (unless ``video`` is False), eng stereo 880 Hz, jpn
    stereo 440 Hz (stream 2 with video)."""
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["-f", "lavfi", "-i", "testsrc=s=64x48:r=4:d=100"] if video else []
    n = 1 if video else 0
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", *inputs,
            "-f", "lavfi", "-i", "sine=frequency=880:duration=100",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=100",
            *(["-map", "0:v"] if video else []), "-map", f"{n}:a", "-map", f"{n + 1}:a",
            *(["-c:v", "mpeg4"] if video else []), "-c:a", "aac", "-ac", "2",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=jpn",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(scope="session")
def source_mkv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_source(tmp_path_factory.mktemp("source") / "made.mkv")


def _with_source(ws: Workspace, src: Path, media: Path) -> Path:
    """``src`` copied as the workspace's S01E01 under ``media`` (`source`); S01E02 has none."""
    import shutil

    media.mkdir(parents=True, exist_ok=True)
    dest = media / "Show - S01E01.mkv"
    shutil.copy(src, dest)
    toml = ws.root / "fukikae.toml"
    toml.write_text(toml.read_text(encoding="utf-8") + f'source = "{media}"\n', encoding="utf-8")
    _record_extract(ws, dest)
    return dest


def _record_extract(ws: Workspace, source: Path, episode_id: str = E1) -> None:
    """The episode's manifest records an extract run of ``source`` (its real cache key), as
    `fukikae run` would after reading it."""
    from fukikae.job import cache_key, load_manifest
    from fukikae.stages.extract import ExtractStage
    from fukikae.workspace import Episode

    ep = Episode(source=source, id=episode_id, work=ws.work_dir(episode_id),
                 workspace_root=ws.root)  # fmt: skip
    m = load_manifest(ep)
    key = cache_key(ExtractStage(), ep, load_config(ws.root, user_config=None))
    m.stages["extract"] = StageRecord(key=key, status="done", started=datetime.now(UTC))
    save_manifest(ep, m)


@pytest.fixture
def video_ws(ws: Workspace, source_mkv: Path, tmp_path: Path) -> Workspace:
    _with_source(ws, source_mkv, tmp_path / "media")
    return ws


def _videos(doc) -> list[tuple[str, str | None]]:
    """(episode, video) of every sample: characters', split parts' and extras'."""
    out = [(s.episode, s.video) for c in doc.characters for s in c.samples]
    out += [(s.episode, s.video) for x in doc.extras for s in x.samples]
    out += [(smp["episode"], smp.get("video")) for c in doc.characters for s in c.suggestions
            if s.parts for p in s.parts.values() for smp in p.get("samples", [])]  # fmt: skip
    return out


def _ffprobe(path: Path) -> dict:
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_name,codec_type,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)


@pytest.mark.ffmpeg
def test_samples_name_a_video_clip_cut_from_the_source_on_first_fetch(video_ws):
    from fukikae.studio.season import VIDEO_NAME, cut_video, video_source

    cfg = load_config(video_ws.root, user_config=None)
    doc = _build(video_ws, clips=4)
    got = _videos(doc)
    e1 = [v for e, v in got if e == E1]
    assert e1 and all(v is not None and VIDEO_NAME.fullmatch(v) for v in e1)
    assert any(smp.get("video") for c in doc.characters for s in c.suggestions if s.parts
               for p in s.parts.values() for smp in p.get("samples", []))  # fmt: skip
    assert [v for e, v in got if e == E2] and all(v is None for e, v in got if e == E2)
    eps = {e.id: e for e in doc.episodes}
    assert eps[E1].video_note is None
    note = eps[E2].video_note
    assert note is not None and "no source video" in note
    videos = video_ws.root / "studio" / "video"
    assert not list(videos.glob("*.mp4"))  # nothing is cut until it is fetched
    s = next(s for s in _entry(doc, 1).samples if s.episode == E1)
    src = video_source(video_ws, cfg, E1)
    assert (src.video, src.audio, src.interlaced) == (0, 2, False)  # the jpn track
    path = cut_video(video_ws, src, s.start, s.end)
    assert path.name == s.video and path.parent == videos
    assert stat.S_IMODE(videos.stat().st_mode) == 0o700
    info = _ffprobe(path)
    streams = {x["codec_type"]: x for x in info["streams"]}
    assert streams["video"]["codec_name"] == "h264" and streams["audio"]["codec_name"] == "aac"
    assert (streams["video"]["width"], streams["video"]["height"]) == (480, 360)
    pad = 2 * 0.3 if s.start >= 0.3 else 0.3 + s.start
    assert float(info["format"]["duration"]) == pytest.approx(s.end - s.start + pad, abs=0.35)
    assert cut_video(video_ws, src, s.start, s.end) == path  # cut once


def test_without_a_source_setting_no_sample_has_video_and_the_index_says_why(ws):
    doc = _build(ws)
    assert all(v is None for _, v in _videos(doc))
    for e in doc.episodes:
        if e.ready:
            assert e.video_note is not None and "source" in e.video_note
    assert not (ws.root / "studio" / "video").exists() or not list(
        (ws.root / "studio" / "video").iterdir()
    )


@pytest.mark.ffmpeg
def test_a_source_without_a_video_stream_gives_no_video(ws, tmp_path):
    _with_source(ws, _make_source(tmp_path / "audio-only.mkv", video=False), tmp_path / "m")
    doc = _build(ws)
    assert all(v is None for _, v in _videos(doc))
    note = next(e for e in doc.episodes if e.id == E1).video_note
    assert note is not None and "no video stream" in note


@pytest.mark.ffmpeg
def test_a_replaced_source_renames_its_clips_and_never_serves_a_stale_one(video_ws):
    from fukikae.studio.season import NoVideo, cut_video, inputs_digest, video_source

    cfg = load_config(video_ws.root, user_config=None)
    first = _build(video_ws, clips=4)
    before = inputs_digest(video_ws, cfg, clips=4)
    s = next(s for s in _entry(first, 1).samples if s.episode == E1)
    old = video_source(video_ws, cfg, E1)
    src = old.path
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert inputs_digest(video_ws, cfg, clips=4) != before
    second = _build(video_ws, clips=4)
    s2 = next(x for x in _entry(second, 1).samples if (x.episode, x.start) == (E1, s.start))
    assert s2.video is None  # not the file the extract stage read until extract runs again
    note = next(e for e in second.episodes if e.id == E1).video_note
    assert note is not None and "isn't the file the extract stage read" in note
    _record_extract(video_ws, src)
    third = _build(video_ws, clips=4)
    s3 = next(x for x in _entry(third, 1).samples if (x.episode, x.start) == (E1, s.start))
    assert s3.video is not None and s3.video != s.video
    with pytest.raises(NoVideo, match="changed"):
        cut_video(video_ws, old, s.start, s.end)  # named before the replacement: refused
    assert not (video_ws.root / "studio" / "video" / s.video).exists()


@pytest.mark.ffmpeg
def test_a_rebuild_prunes_video_clips_no_longer_referenced(video_ws, tmp_path):
    from fukikae.studio.season import cut_video, video_source

    cfg = load_config(video_ws.root, user_config=None)
    first = _build(video_ws, clips=6)
    s = next(s for s in _entry(first, 1).samples if s.episode == E1)
    kept = cut_video(video_ws, video_source(video_ws, cfg, E1), s.start, s.end)
    videos = kept.parent
    stray = videos / ("0" * 64 + ".mp4")
    stray.write_bytes(b"old clip")
    odd = videos / "keep.mp4"
    odd.write_bytes(b"not ours")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = videos / ("1" * 64 + ".mp4")
    link.symlink_to(outside)
    second = _build(video_ws, clips=6)
    assert {v for _, v in _videos(second)} >= {kept.name}
    assert kept.exists() and not stray.exists()
    assert odd.exists() and link.is_symlink() and outside.exists()


def test_the_video_filter_deinterlaces_an_interlaced_source():
    from fukikae.studio.season import video_filter

    assert video_filter(interlaced=True).startswith("yadif,")
    assert "yadif" not in video_filter(interlaced=False)
    assert "scale=480:" in video_filter(interlaced=False)


@pytest.mark.ffmpeg
def test_a_cut_gives_ffmpeg_a_timeout_of_a_minute_or_more(video_ws, monkeypatch):
    from fukikae import media
    from fukikae.studio import season

    cfg = load_config(video_ws.root, user_config=None)
    got: list[float | None] = []

    def fake(args: list[str], *, timeout: float | None = None) -> None:
        got.append(timeout)
        Path(args[-1]).write_bytes(b"clip")

    monkeypatch.setattr(media, "run_ffmpeg", fake)
    src = season.video_source(video_ws, cfg, E1)
    season.cut_video(video_ws, src, 10.0, 12.0)
    season.cut_video(video_ws, src, 10.0, 30.0)
    assert got[0] == season.CUT_TIMEOUT_MIN_S == 60.0
    assert got[1] == pytest.approx(season.CUT_TIMEOUT_PER_S * (20.0 + 2 * season.VIDEO_PAD_S))
    assert (got[1] or 0.0) > 60.0


@pytest.mark.ffmpeg
def test_a_source_the_extract_stage_did_not_read_gives_no_video(video_ws):
    from fukikae.job import load_manifest
    from fukikae.registry.invalidate import episode

    ep = episode(video_ws, E1)
    m = load_manifest(ep)
    m.stages["extract"] = m.stages["extract"].model_copy(update={"key": "another-file"})
    save_manifest(ep, m)
    doc = _build(video_ws, clips=4)
    assert all(v is None for e, v in _videos(doc) if e == E1)
    note = next(e for e in doc.episodes if e.id == E1).video_note
    assert note is not None and "isn't the file the extract stage read" in note
    del m.stages["extract"]
    save_manifest(ep, m)
    doc = _build(video_ws, clips=4)
    note = next(e for e in doc.episodes if e.id == E1).video_note
    assert note is not None and "no finished extract run" in note


def _media_tree(root: Path) -> Path:
    (root / "Season 1").mkdir(parents=True)
    (root / "Season 1" / "Show - S01E01.mkv").write_bytes(b"x")
    (root / "notes.txt").write_text("not a video")
    return root


def test_the_source_scan_is_cached_until_a_directory_changes(tmp_path, monkeypatch):
    import time

    from fukikae.studio import season

    root = _media_tree(tmp_path / "media")
    walks: list[Path] = []
    real = season._walk_sources

    def counting(r: Path):
        walks.append(r)
        return real(r)

    monkeypatch.setattr(season, "_walk_sources", counting)
    scan = season.SourceScanner(timeout=5)
    first = scan.get(root)
    assert first.note is None and set(first.files) == {"S01E01"}
    assert scan.get(root).files == first.files and len(walks) == 1  # nothing changed: no walk
    time.sleep(0.05)
    (root / "Season 1" / "Show - S01E02.mkv").write_bytes(b"x")  # the subdirectory's mtime
    assert set(scan.get(root).files) == {"S01E01", "S01E02"} and len(walks) == 2
    time.sleep(0.05)
    (root / "Season 2").mkdir()  # the root's mtime
    (root / "Season 2" / "Show - S02E01.mkv").write_bytes(b"x")
    assert set(scan.get(root).files) == {"S01E01", "S01E02", "S02E01"} and len(walks) == 3
    assert scan.get(None).files == {} and scan.get(tmp_path / "nowhere").files == {}


def test_a_stalled_source_scan_answers_within_its_timeout(tmp_path, monkeypatch):
    import threading
    import time

    from fukikae.studio import season

    root = _media_tree(tmp_path / "media")
    gate = threading.Event()
    real_walk, real_sig = season._walk_sources, season._signature

    def stuck_walk(r: Path):
        gate.wait(30)
        return real_walk(r)

    monkeypatch.setattr(season, "_walk_sources", stuck_walk)
    scan = season.SourceScanner(timeout=0.3)
    t0 = time.monotonic()
    got = scan.get(root)
    assert time.monotonic() - t0 < 2
    assert got.files == {} and got.note is not None and "didn't answer" in got.note
    t0 = time.monotonic()
    assert scan.get(root).note is not None  # the same stalled walk: no second wait
    assert time.monotonic() - t0 < 0.2
    gate.set()
    deadline = time.monotonic() + 5
    while (got := scan.get(root)).note is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert set(got.files) == {"S01E01"}
    # a stall after a good scan keeps answering with that scan
    gate.clear()

    def stuck_sig(dirs):
        gate.wait(30)
        return real_sig(dirs)

    monkeypatch.setattr(season, "_signature", stuck_sig)
    t0 = time.monotonic()
    again = scan.get(root)
    assert time.monotonic() - t0 < 2 and again.files == got.files and again.note is None
    gate.set()


def test_an_index_built_while_the_source_stalls_says_so(ws):
    from fukikae.studio.season import SourceFiles, build_season, inputs_digest

    toml = ws.root / "fukikae.toml"
    toml.write_text(toml.read_text(encoding="utf-8") + 'source = "/nowhere"\n', encoding="utf-8")
    cfg = load_config(ws.root, user_config=None)
    stalled = SourceFiles({}, note="the source directory /nowhere didn't answer within 10 s")
    doc = build_season(ws, cfg, sources=stalled)
    note = next(e for e in doc.episodes if e.id == E1).video_note
    assert note is not None and "didn't answer" in note
    assert inputs_digest(ws, cfg, sources=stalled) != inputs_digest(
        ws, cfg, sources=SourceFiles({})
    )


def test_pruning_removes_temp_files_a_killed_cut_left(ws, tmp_path):
    import time

    from fukikae.studio.season import prune_clips

    doc = _build(ws)
    videos = ws.root / "studio" / "video"
    videos.mkdir(parents=True, exist_ok=True)
    sha = "a" * 64
    old = videos / f".{sha}.{'1' * 32}.tmp.mp4"
    new = videos / f".{sha}.{'2' * 32}.tmp.mp4"
    odd = videos / f".{sha}.notauuid.tmp.wav"
    old_dir = videos / f".{sha}.{'3' * 32}.tmp.mp4"
    outside = tmp_path / "outside.mp4"
    link = videos / f".{sha}.{'4' * 32}.tmp.mp4"
    for p in (old, new, odd, outside):
        p.write_bytes(b"partial")
    old_dir.mkdir()
    link.symlink_to(outside)
    hours_ago = time.time() - 2 * 3600
    for p in (old, odd, old_dir, outside):
        os.utime(p, (hours_ago, hours_ago))
    os.utime(link, (hours_ago, hours_ago), follow_symlinks=False)
    gone = prune_clips(ws, doc)
    assert old.name in gone and not old.exists()
    assert new.exists() and odd.exists() and old_dir.is_dir() and link.is_symlink()
    assert outside.exists()


@pytest.mark.ffmpeg
def test_finding_a_video_source_bounds_every_probe(video_ws, monkeypatch):
    from fukikae import media
    from fukikae.studio import season

    cfg = load_config(video_ws.root, user_config=None)
    seen: list[tuple[str, float | None]] = []
    real = media._run

    def recording(cmd: list[str], timeout: float | None = None):
        seen.append((cmd[0], timeout))
        return real(cmd, timeout=timeout)

    monkeypatch.setattr(media, "_run", recording)
    season.video_source(video_ws, cfg, E1)
    assert seen and all(t == season.PROBE_TIMEOUT_S for _, t in seen), seen
