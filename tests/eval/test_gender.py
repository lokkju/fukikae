"""`fukikae eval gender`: cue gender agreement, reference purity, tiers, stock cues, gated lines
and wrong-gender emotion prompts on a synthetic dubbed episode (Part A evaluation)."""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.cards.card import Card, Reference, Voice
from fukikae.cards.store import CardStore
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    DubCues,
    TtsPlan,
    TtsPlanCue,
    Turn,
)
from tests.fixtures.speechlike import noise_bursts, speech

RATE = 16000
EP = "S01E01"
SRC = Path(__file__).resolve().parents[2] / "src" / "fukikae"


def _wav(path: Path, x: np.ndarray, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _voice(hz: float, seconds: float, seed: int = 0) -> np.ndarray:
    return speech(hz, seconds, seed=seed, voiced_share=0.7)


def _cue(cid: str, char: int | None, span: tuple[float, float], extra: str | None = None,
         source_text: str = "src"):  # fmt: skip
    return DubCue(
        id=cid, start=span[0], end=span[1], character_id=char, extra=extra, speaker="x",
        source_text=source_text, text="line", budget=40, delivery=Delivery(reference=span),
    )  # fmt: skip


class Episode:
    """A dubbed episode on disk: vocals with speech at given spans, assignments, cues, cards,
    renders and plan."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.work = root / ".fukikae" / "episodes" / EP
        root.mkdir(parents=True, exist_ok=True)
        (root / "fukikae.toml").write_text(
            'source_language = "ja"\ntarget_language = "en"\n', encoding="utf-8"
        )
        self.store = CardStore(root)
        self.vocals = np.zeros(80 * RATE, np.float32)
        self.turns: list[Turn] = []
        self.clusters: list[ClusterAssignment] = []
        self.cues: list[DubCue] = []
        self.plan: list[TtsPlanCue] = []

    def say(self, start: float, hz: float, seconds: float, seed: int = 0) -> None:
        a = round(start * RATE)
        seg = _voice(hz, seconds, seed)
        self.vocals[a : a + len(seg)] = seg

    def character(self, cluster: str, char: int | None, turns: list[tuple[float, float, float]],
                  status: str = "new", source: str = "voice") -> None:  # fmt: skip
        self.clusters.append(ClusterAssignment(
            cluster=cluster, actor_id=char, character_id=char, score=None, clean_seconds=10.0,
            status=status, source=source,  # type: ignore[arg-type]
        ))  # fmt: skip
        for i, (start, end, hz) in enumerate(turns):
            self.say(start, hz, end - start, seed=100 + i)
            self.turns.append(Turn(start=start, end=end, speaker=cluster))

    def clone(self, clips: list[np.ndarray], *, tier: str | None = None,
              origins: list[tuple[float, float] | tuple[float, float, str]] | None = None,
              ) -> str:  # fmt: skip
        """A clone card of ``clips``; ``origins`` are ``(start, end)`` in this episode or
        ``(start, end, episode)``."""
        refs = []
        for i, x in enumerate(clips):
            path = _wav(self.root / "tmp" / f"clip{len(self.plan)}-{i}-{id(x)}.wav", x)
            o = (origins or [(0.0, 1.0)] * len(clips))[i]
            refs.append(Reference(
                clip=self.store.add_sidecar(path, "clip"), transcript="x", language="ja",
                origin={"episode": o[2] if len(o) > 2 else EP, "start": o[0], "end": o[1]},
            ))  # fmt: skip
        meta = {"reference_tier": tier} if tier else {}
        return self.store.put(Card(
            target_language="en", meta=meta,
            voice=Voice(backend="clonefake", model_revision="rev-1", source="clone", seed=1234,
                        references=tuple(refs)),
        ))  # fmt: skip

    def stock(self, voice: str = "am_echo") -> str:
        return self.store.put(Card(
            target_language="en",
            voice=Voice(backend="kokoro", model_revision="rev-k", source="stock", voice=voice),
        ))  # fmt: skip

    def line(self, cid: str, char: int | None, span: tuple[float, float], card: str,
             render: np.ndarray | None, *, extra: str | None = None,
             passed: dict | None = None, gate: bool = False,
             source_text: str = "src") -> None:  # fmt: skip
        self.cues.append(_cue(cid, char, span, extra, source_text))
        if render is not None:
            _wav(self.work / "07-tts" / f"{cid}.wav", render)
        self.plan.append(TtsPlanCue(
            cue_id=cid, character_id=char, extra=extra, card_id=card, backend="clonefake",
            key="k" + cid if render is not None else None, slot=span[1] - span[0],
            passed=passed if passed is not None else {"emotion_audio": list(span)},
            gender_gate=gate,
        ))  # fmt: skip

    def write(self) -> Path:
        _wav(self.work / "01-separate" / "vocals.wav", self.vocals)
        for d in ("05-characters", "06-translate", "07-tts"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        write_json(self.work / "05-characters" / "assignments.json", Assignments(
            episode=EP, threshold=0.6, clusters=self.clusters, turns=self.turns,
        ))  # fmt: skip
        write_json(self.work / "06-translate" / "cues.en.json", DubCues(
            episode=EP, source_language="ja", target_language="en", path="asr", cues=self.cues,
        ))  # fmt: skip
        write_json(self.work / "07-tts" / "plan.json", TtsPlan(episode=EP, cues=self.plan))
        return self.root


def _female_char(e: Episode) -> None:
    """Character 1, cluster S1: three turns in the female band."""
    e.character("S1", 1, [(2.0, 5.0, 250), (6.0, 9.0, 250), (10.0, 13.0, 250)])


@pytest.fixture
def e(tmp_path: Path) -> Episode:
    return Episode(tmp_path / "ws")


def test_mismatch_counts_on_a_synthetic_episode(e):
    from fukikae.eval.gender import measure

    _female_char(e)
    card = e.clone([_voice(250, 4.0, 1), _voice(250, 4.0, 2)])
    e.say(20.0, 250, 2.0, seed=7)  # f source, f render
    e.say(23.0, 120, 2.0, seed=8)  # m source, f render
    # c0003's source span is silent: u
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    e.line("c0002", 1, (23.0, 25.0), card, _voice(250, 2.0, 12))
    e.line("c0003", 1, (26.0, 28.0), card, _voice(250, 2.0, 13))
    r = measure(e.write(), EP)
    assert r.cues == 3
    assert r.known == 2 and r.mismatched_strict == 1
    assert r.known_cut == 2 and r.mismatched_cut == 1
    row = {w.speaker: w for w in r.per_speaker}["character 1"]
    assert (row.cues, row.known, row.mismatched, row.m_to_f, row.f_to_m) == (3, 2, 1, 1, 0)
    assert [c.cue_id for c in r.worst] == ["c0002"]


def test_cut_counts_lines_between_the_bands(e):
    """A source line reading 172 Hz is u in the strict bands but m at the single 180 Hz cut."""
    from fukikae.eval.gender import measure

    _female_char(e)
    card = e.clone([_voice(250, 4.0, 1)])
    e.say(20.0, 160, 2.0, seed=7)  # reads about 172 Hz (its glides sit high)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    r = measure(e.write(), EP)
    assert (r.known, r.mismatched_strict) == (0, 0)
    assert (r.known_cut, r.mismatched_cut) == (1, 1)


def test_distance_purity_is_a_diagnostic_on_the_clip_files(e):
    """The 4-semitone measure reads each clip file; the clips' origins are the character's own
    250 Hz turns, so the gate rule (on the origins) passes all three."""
    from fukikae.eval.gender import measure

    _female_char(e)
    card = e.clone(
        [_voice(250, 4.0, 1), _voice(120, 4.0, 2), noise_bursts(4.0, seed=3)],
        origins=[(2.0, 5.0), (6.0, 9.0), (10.0, 13.0)],
    )
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    r = measure(e.write(), EP)
    assert r.purity_4st == {1: (1, 3)} and r.purity == {1: (3, 3)}
    ref = {x.speaker: x for x in r.references}["character 1"]
    assert ref.majority == "f" and [v.passed_4st for v in ref.clips] == [True, False, False]
    assert ref.clips[1].why == "pitch" and ref.clips[2].why == "voicing"
    assert "impure" not in ref.flags


def _gated_char(e: Episode, low: list[tuple[float, float, float]], *, source: str = "voice",
                ) -> None:  # fmt: skip
    """Character 1: female turns (2-5, 6-9, 10-13 s at 250 Hz, 14-17 s at 330 Hz) in S1 and
    the ``low`` turns in S1b (a second cluster of the same character, its ``source``)."""
    e.character("S1", 1, [(2.0, 5.0, 250), (6.0, 9.0, 250), (10.0, 13.0, 250),
                          (14.0, 17.0, 330)])  # fmt: skip
    e.character("S1b", 1, low, source=source)


def test_headline_purity_is_the_pipelines_gate_rule(e):
    """Cross-class only: a male-band origin is refused, a same-class one 4.8 st off the
    majority passes (the distance diagnostic refuses it), a clip with no origin in this
    episode isn't judged and passes."""
    from fukikae.eval.gender import measure

    _gated_char(e, [(40.0, 43.0, 120)])
    card = e.clone(
        [_voice(250, 3.0, 1), _voice(120, 3.0, 2), _voice(330, 3.0, 3), _voice(250, 3.0, 4)],
        origins=[(2.0, 5.0), (40.0, 43.0), (14.0, 17.0), (2.0, 5.0, "S01E09")],
    )
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    r = measure(e.write(), EP)
    ref = {x.speaker: x for x in r.references}["character 1"]
    assert [v.gates for v in ref.clips] == [[], ["off_register"], [], []]
    assert [v.judged for v in ref.clips] == [True, True, True, False]
    assert r.purity == {1: (3, 4)}
    assert ref.clips[2].passed and not ref.clips[2].passed_4st  # same class, 4.8 st off
    assert "impure" in ref.flags


def test_a_near_even_bimodal_register_gates_no_pitch(e):
    """Majority under 2x the minority's seconds: no register gate (controller ruling (a))."""
    from fukikae.eval.gender import measure

    _gated_char(e, [(40.0, 46.0, 120), (47.0, 53.0, 120)])  # 12 s m against 12 s f
    card = e.clone([_voice(250, 3.0, 1), _voice(120, 3.0, 2)],
                   origins=[(2.0, 5.0), (40.0, 43.0)])  # fmt: skip
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    r = measure(e.write(), EP)
    assert r.purity == {1: (2, 2)}
    assert "mixed" in {x.speaker: x for x in r.references}["character 1"].flags


def test_a_name_confirmed_clip_gates_no_pitch(e, monkeypatch):
    """A clip inside the character's name-confirmed turns of a names-used episode is exempt
    from the register gate, as selection leaves it."""
    from fukikae.cards import auto
    from fukikae.eval.gender import measure

    _gated_char(e, [(40.0, 43.0, 120)], source="name")
    card = e.clone([_voice(250, 3.0, 1), _voice(120, 3.0, 2)],
                   origins=[(2.0, 5.0), (40.0, 43.0)])  # fmt: skip
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    root = e.write()
    assert measure(root, EP).purity == {1: (1, 2)}  # no names used: gated
    monkeypatch.setattr(auto, "checked_names", lambda _w, _a: (object(), None))
    assert measure(root, EP).purity == {1: (2, 2)}


def test_card_reference_gender_against_character(e):
    from fukikae.eval.gender import measure

    _female_char(e)
    card = e.clone([_voice(120, 4.0, 1), _voice(125, 4.0, 2)])
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    ref = {x.speaker: x for x in measure(e.write(), EP).references}["character 1"]
    assert ref.card_class == "m" and ref.majority == "f"
    assert "card≠source" in ref.flags


def test_wrong_gender_prompts_reads_plan_passed(e):
    from fukikae.eval.gender import measure

    _female_char(e)
    card = e.clone([_voice(250, 4.0, 1)])
    e.say(20.0, 120, 2.0, seed=7)  # male-band line of a female speaker
    e.say(23.0, 120, 2.0, seed=8)
    e.say(26.0, 250, 2.0, seed=9)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))  # prompt passed: wrong
    e.line("c0002", 1, (23.0, 25.0), card, _voice(250, 2.0, 12),
           passed={"emotion_dropped": "gender"}, gate=True)  # fmt: skip
    e.line("c0003", 1, (26.0, 28.0), card, _voice(250, 2.0, 13))  # own register: fine
    r = measure(e.write(), EP)
    assert r.wrong_gender_prompts == 1
    assert r.wrong_gender_cues == ["c0001"]
    assert r.gated_lines == 1


def test_stock_cues_and_tiers_are_counted(e):
    from fukikae.eval.gender import measure

    _female_char(e)
    e.character("S9", None, [(30.0, 33.0, 120)], status="extra")
    thin = e.clone([_voice(250, 2.0, 1)], tier="thin")
    stock = e.stock("af_alloy")
    e.say(20.0, 250, 2.0, seed=7)
    e.say(34.0, 120, 2.0, seed=8)
    e.line("c0001", 1, (20.0, 22.0), thin, _voice(250, 2.0, 11))
    e.line("c0002", None, (34.0, 36.0), stock, _voice(250, 2.0, 12), extra="S9")
    r = measure(e.write(), EP)
    assert r.stock_cues == 1
    tier, seconds = r.tiers["character 1"]
    assert tier == "thin" and seconds == pytest.approx(2.0, abs=0.05)
    assert "extra S9" not in r.tiers  # a stock card has no reference tier
    # the extras count reads the render; the stock count reads the voice's name
    assert (r.extras_known, r.extras_disagree) == (1, 1)
    assert r.stock_voice_disagree == (1, 1)
    assert "thin" in {x.speaker: x for x in r.references}["character 1"].flags


def test_a_thin_cards_clips_are_not_judged_and_never_impure(e):
    """A2: a thin clone sets the reference gates aside, and the pipeline never calls it
    impure; its clips aren't judged (and pass), and impure_cards leaves it out."""
    from fukikae.eval.gender import measure

    _gated_char(e, [(40.0, 43.0, 120)])
    thin = e.clone([_voice(120, 2.0, 2)], tier="thin", origins=[(40.0, 42.0)])
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), thin, _voice(250, 2.0, 11))
    r = measure(e.write(), EP)
    ref = {x.speaker: x for x in r.references}["character 1"]
    assert [(v.judged, v.passed, v.gates) for v in ref.clips] == [(False, True, [])]
    assert "impure" not in ref.flags and "thin" in ref.flags
    assert r.metrics()["impure_cards"] == 0 and r.purity == {1: (1, 1)}


def test_an_extras_cluster_without_turns_reads_its_register_from_all_its_cues(e):
    """As extras_cards does: a cluster with no turn in assignments.json has its cues' spans
    as its speech, so its register is the majority over all of them (9 s male against 3 s
    female here), not the first cue's span."""
    from fukikae.eval.gender import measure

    e.character("S9", None, [], status="extra")
    card = e.clone([_voice(250, 3.0, 1)], origins=[(44.0, 47.0)])
    spans = [(44.0, 47.0, 250), (30.0, 33.0, 120), (34.0, 37.0, 120), (38.0, 41.0, 120)]
    for i, (a, b, hz) in enumerate(spans):
        e.say(a, hz, b - a, seed=20 + i)
        e.line(f"c000{i + 1}", None, (a, b), card, _voice(hz, b - a, 30 + i), extra="S9")
    r = measure(e.write(), EP)
    ref = {x.speaker: x for x in r.references}["extra S9"]
    assert [v.gates for v in ref.clips] == [["off_register"]]


def test_tier_is_inferred_for_a_card_from_before_tiers(e):
    from fukikae.eval.gender import measure

    _female_char(e)
    card = e.clone([_voice(250, 2.0, 1), _voice(250, 2.0, 2)])  # 4 s, two clips: short
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    assert measure(e.write(), EP).tiers["character 1"][0] == "short"


class FakeEmbedder:
    """Source spans of vocals embed by second; renders embed as their file name says."""

    def __init__(self, sims: dict[str, float]) -> None:
        self.sims = sims

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        rows = []
        for _ in spans:
            if audio.name.startswith("vocals"):
                rows.append(np.array([1.0, 0.0]))
            else:
                c = self.sims[audio.stem]
                rows.append(np.array([c, np.sqrt(1 - c * c)]))
        return np.array(rows)


def test_former_stock_cues_are_reported_apart_with_before(e, tmp_path):
    from fukikae.eval.gender import measure

    _female_char(e)
    e.character("S9", None, [(30.0, 33.0, 120)], status="extra")
    card = e.clone([_voice(250, 4.0, 1)])
    extra = e.clone([_voice(120, 4.0, 2)])
    old = e.stock("am_echo")
    e.say(20.0, 250, 2.0, seed=7)
    e.say(34.0, 120, 2.0, seed=8)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    e.line("c0002", None, (34.0, 36.0), extra, _voice(120, 2.0, 12), extra="S9")
    root = e.write()
    before = _before(tmp_path, [
        ("c0001", 20.0, "src", card), ("c0002", 25.0, "other", old),
        ("c0003", 34.05, "src", old),
    ])  # fmt: skip
    r = measure(root, EP, similarity=True, before=before,
                embedder=FakeEmbedder({"c0001": 0.3, "c0002": 0.2}))  # fmt: skip
    assert r.former_stock == ["c0002"]  # before's c0003: same text, start 0.05 s off
    assert r.before
    assert r.similarity is not None
    assert r.similarity["former_stock"] == {"cues": 1, "mean": pytest.approx(0.2)}
    assert r.similarity["clone"] == {"cues": 2, "mean": pytest.approx(0.25)}
    assert r.similarity["stock"] == {"cues": 0, "mean": None}
    assert measure(root, EP).similarity is None


def _before(tmp_path: Path, cues: list[tuple[str, float, str, str]]) -> Path:
    """A before episode (``<ws>/.fukikae/episodes/S01E01``): its plan and its cues, each
    ``(id, start, source text, card)``."""
    work = tmp_path / "before-ws" / ".fukikae" / "episodes" / EP
    for d in ("06-translate", "07-tts"):
        (work / d).mkdir(parents=True, exist_ok=True)
    write_json(work / "06-translate" / "cues.en.json", DubCues(
        episode=EP, source_language="ja", target_language="en", path="asr",
        cues=[_cue(cid, None, (start, start + 2.0), source_text=text)
              for cid, start, text, _ in cues],
    ))  # fmt: skip
    plan = work / "07-tts" / "plan.json"
    write_json(plan, TtsPlan(episode=EP, cues=[
        TtsPlanCue(cue_id=cid, character_id=None, extra=None, card_id=card, slot=2.0)
        for cid, _, _, card in cues
    ]))  # fmt: skip
    return plan


def test_before_matches_by_start_and_source_text_not_cue_id(e, tmp_path):
    from fukikae.eval.gender import measure, options

    _female_char(e)
    card = e.clone([_voice(250, 4.0, 1)])
    old = e.stock("am_echo")
    for i, start in enumerate((20.0, 23.0, 26.0)):
        e.say(start, 250, 2.0, seed=7 + i)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11), source_text="a")
    e.line("c0002", 1, (23.0, 25.0), card, _voice(250, 2.0, 12), source_text="b")
    e.line("c0003", 1, (26.0, 28.0), old, _voice(250, 2.0, 13), source_text="c")
    root = e.write()
    before = _before(tmp_path, [
        ("c0001", 10.0, "dropped since", old),  # renumbered: c0002 was c0003 before
        ("c0002", 20.1, "a", card),
        ("c0003", 23.0, "b", old),
        ("c0004", 26.0, "c", old),  # still stock now: the stock group only
        ("c0005", 40.0, "a", old),  # same text, far away
    ])  # fmt: skip
    r = measure(root, EP, before=before)
    assert r.former_stock == ["c0002"]
    assert options(r)["before"] is True
    nothing = _before(tmp_path / "n", [("c0001", 20.0, "a", card)])
    r = measure(root, EP, before=nothing)
    assert r.former_stock == [] and options(r)["before"] is True
    assert options(measure(root, EP))["before"] is None


def test_report_text_and_json_hold_the_headline(e):
    from fukikae.eval.gender import format_report, measure, report_json

    _female_char(e)
    card = e.clone([_voice(250, 4.0, 1)], origins=[(2.0, 5.0)])
    e.say(20.0, 120, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    r = measure(e.write(), EP)
    text = format_report(r)
    assert "strict 1/1 known" in text and "stock-voiced cues: 0" in text
    assert "passing the pipeline's gates: 1/1" in text and "(diagnostic): 1/1" in text
    doc = report_json(r)
    assert doc["metrics"]["mismatched_strict"] == 1 and doc["metrics"]["stock_cues"] == 0
    assert doc["cues"][0]["cue_id"] == "c0001"
    json.dumps(doc)  # serializable


def test_cli_writes_json_and_a_gender_ledger_row(e, tmp_path):
    from fukikae.cli import app
    from fukikae.eval.ledger import read_rows

    _female_char(e)
    card = e.clone([_voice(250, 4.0, 1)])
    e.say(20.0, 120, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    root = e.write()
    out, md, ledger = tmp_path / "g.json", tmp_path / "g.md", tmp_path / "results.jsonl"
    cache = tmp_path / "cache"
    args = ["eval", "gender", str(root), EP, "--json", str(out), "--md", str(md),
            "--ledger", str(ledger), "--pitch-cache", str(cache), "--tag", "t"]  # fmt: skip
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text())["metrics"]["mismatched_strict"] == 1
    assert "strict 1/1 known" in md.read_text()
    rows, _ = read_rows(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row.stage == "gender" and row.backend == "fukikae.pitch" and row.tag == "t"
    assert row.metrics["mismatched_strict"] == 1 and row.metrics["stock_cues"] == 0
    assert row.options["bands"] == [165, 200] and row.options["cut"] == 180
    assert row.episodes == [EP] and row.language == "en"
    # the workspace gets no pitch cache when another root is named
    assert not (root / ".fukikae" / "pitch-cache").exists() and any(cache.rglob("*.npz"))
    res = CliRunner().invoke(app, [*args[:4], "--no-ledger", "--pitch-cache", str(cache)])
    assert res.exit_code == 0, res.output
    assert len(read_rows(ledger)[0]) == 1


def test_gender_is_a_ledger_stage():
    from fukikae.eval.ledger import STAGES
    from fukikae.eval.report import HEADLINE

    assert "gender" in STAGES and HEADLINE["gender"] == "mismatch_rate"


# The command-line modules that register `fukikae eval` commands; no other module outside
# eval/ may import it.
R1_ALLOWED = {"cli.py", "registry/cli.py"}


def test_eval_never_imported_by_pipeline():
    """R1: no pipeline module (anything in src/fukikae outside eval/ but the command-line
    modules listed in ``R1_ALLOWED``) imports fukikae.eval."""
    pattern = re.compile(r"^\s*(from|import)\s+fukikae\.eval\b", re.M)
    importing = {
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if "eval" not in p.relative_to(SRC).parts[:1]
        and pattern.search(p.read_text(encoding="utf-8"))
    }
    assert importing - R1_ALLOWED == set()


def _tree(root: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("similarity", [False, True])
def test_a_named_pitch_cache_leaves_the_workspace_untouched(e, tmp_path, monkeypatch,
                                                             similarity):  # fmt: skip
    """With --pitch-cache, eval gender writes nothing under the workspace: no pitch cache, no
    16 kHz scratch for the similarity, no registry file (a lock-kept cluster makes it read
    the registry, from a copy)."""
    from fukikae.cli import app
    from fukikae.eval import gender
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    _gated_char(e, [(40.0, 43.0, 120)], source="lock")
    e.clusters[-1] = e.clusters[-1].model_copy(update={"name": "natsu"})
    card = e.clone([_voice(250, 3.0, 1), _voice(120, 3.0, 2)],
                   origins=[(2.0, 5.0), (40.0, 43.0)])  # fmt: skip
    e.say(20.0, 250, 2.0, seed=7)
    e.line("c0001", 1, (20.0, 22.0), card, _voice(250, 2.0, 11))
    root = e.write()
    with Registry(Workspace(root).registry_path):
        pass
    before = _before(tmp_path, [("c0001", 20.0, "src", card)])
    monkeypatch.setattr(gender, "wespeaker_cpu", lambda: FakeEmbedder({"c0001": 0.3}))
    from fukikae.cards import auto

    monkeypatch.setattr(auto, "checked_names", lambda _w, _a: (object(), None))
    snap = _tree(root)
    cache = tmp_path / "cache"
    args = ["eval", "gender", str(root), EP, "--pitch-cache", str(cache), "--no-ledger",
            "--before", str(before), *(["--similarity"] if similarity else [])]  # fmt: skip
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert _tree(root) == snap
    assert any(cache.rglob("*.npz"))
    assert any(cache.rglob("16k-*/*.wav")) == similarity
