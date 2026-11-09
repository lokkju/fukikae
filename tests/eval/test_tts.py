"""`fukikae eval tts`: sampling, metrics, ledger rows and the command end to end on fakes."""

import shutil
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.backends import loader
from fukikae.backends.base import Synthesizer
from fukikae.cards import stock
from fukikae.cards.card import Voice, parse_card
from fukikae.cards.store import CardStore
from fukikae.cli import app
from fukikae.config import BackendConfig, Config, SeriesInfo
from fukikae.eval import tts as ev
from fukikae.eval.ledger import ledger_path, read_rows
from fukikae.eval.report import build_matrix
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    DubCues,
    Segment,
    Tags,
    TagSpan,
    Transcript,
    Turn,
    Word,
)
from fukikae.registry.store import Registry
from fukikae.stages import tts as tts_stage
from fukikae.workspace import Workspace

# ---- helpers ------------------------------------------------------------------------------


def tone_wav(path: Path, seconds: float, rate: int, amp: float = 0.25, freq: float = 440) -> None:
    n = max(1, round(seconds * rate))
    x = amp * np.sin(2 * np.pi * freq * np.arange(n) / rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def cue(i: int, character: int | None, text: str | None, *, slot: float = 2.0, **kw: Any) -> DubCue:
    start = i * 3.0
    emotion = kw.pop("emotion", None)
    return DubCue(
        id=f"c{i:04d}",
        start=start,
        end=start + slot,
        character_id=character,
        extra=kw.pop("extra", None),
        speaker="x",
        source_text="src",
        text=text,
        budget=40,
        delivery=Delivery(reference=(start, start + slot), emotion=emotion),
        **kw,
    )


# ---- pure functions -----------------------------------------------------------------------


def _doc(cues: list[DubCue], episode: str = "S01E01") -> DubCues:
    return DubCues(
        episode=episode, source_language="ja", target_language="en", path="asr", cues=cues
    )


def test_sample_is_deterministic_and_excludes_extras():
    cues = [cue(i, 1 + i % 2, f"line {i}") for i in range(20)]
    cues += [cue(30, None, "extra", extra="S5"), cue(31, 1, None, skip="song")]
    cues += [cue(32, 1, None, error="bad json"), cue(33, 2, "   ")]
    docs = {"S01E01": _doc(cues)}
    a = ev.sample_cues(docs, per_character=4, seed=0)
    assert a == ev.sample_cues(docs, per_character=4, seed=0)
    assert len(a) == 8
    assert all(c.character_id is not None and c.text and c.text.strip() for _, c in a)
    assert {c.character_id for _, c in a} == {1, 2}
    assert ev.sample_cues(docs, per_character=4, seed=1) != a
    # Fewer eligible cues than asked: all of them.
    assert len(ev.sample_cues(docs, per_character=50)) == 20


def test_sample_spans_episodes_in_order():
    docs = {"S01E02": _doc([cue(1, 1, "b")], "S01E02"), "S01E01": _doc([cue(1, 1, "a")])}
    assert [(e, c.text) for e, c in ev.sample_cues(docs, per_character=8)] == [
        ("S01E01", "a"),
        ("S01E02", "b"),
    ]


def test_duration_fit():
    assert ev.duration_fit(1.0, 1.2) == (True, 1.0)
    assert ev.duration_fit(2.0, 1.0)[1] == 2.0
    assert ev.duration_fit(2.0, 1.0)[0] is False
    assert ev.duration_fit(1.0, 0.0) == (False, float("inf"))


def _transcript(words: list[tuple[float, float, str]]) -> Transcript:
    return Transcript(
        language="en",
        backend="faster_whisper",
        word_timing=True,
        segments=[
            Segment(
                start=words[0][0],
                end=words[-1][1],
                text=" ".join(w for _, _, w in words),
                words=[Word(start=s, end=e, text=w) for s, e, w in words],
            )
        ],
    )


def test_reference_speech_spans_the_words_in_the_slot():
    tr = _transcript([(0.5, 0.9, "before"), (10.1, 10.5, "Hello"), (10.6, 11.4, "there"),
                      (11.9, 12.2, "edge"), (13.0, 13.5, "after")])  # fmt: skip
    # Words whose midpoint lies in the slot widened by pad: first start to last end.
    assert ev.reference_speech(tr, (10.0, 11.8), pad=0.3) == pytest.approx(12.2 - 10.1)
    assert ev.reference_speech(tr, (10.0, 11.5), pad=0.0) == pytest.approx(11.4 - 10.1)
    assert ev.reference_speech(tr, (20.0, 21.0)) is None
    assert ev.reference_span(tr, (10.0, 11.5), pad=0.0) == pytest.approx((10.1, 11.4))


def test_similarity_is_cosine_to_the_mean_reference():
    refs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ev.similarity(np.array([[1.0, 1.0]]), refs) == pytest.approx(1.0)
    assert ev.similarity(np.array([[1.0, -1.0]]), refs) == pytest.approx(0.0, abs=1e-9)
    assert ev.similarity(np.array([[1.0, 0.0]]), refs) == pytest.approx(np.sqrt(0.5))


def test_word_error_rate_normalizes_case_and_punctuation():
    assert ev.word_error_rate("hello, World!", "Hello world", "en") == 0.0
    assert ev.word_error_rate("hello there", "hello world", "en") == pytest.approx(0.5)
    assert ev.word_errors("a b c", "a x", "en") == (2, 2)  # one substitution, one insertion
    assert ev.word_errors("", "", "en") == (0, 0)


def test_clean_exemplars_are_the_longest_non_overlapping_turns():
    asg = Assignments(
        episode="e",
        threshold=0.6,
        clusters=[
            ClusterAssignment(cluster="S0", actor_id=1, character_id=1, score=0.9,
                              clean_seconds=1, status="matched"),
            ClusterAssignment(cluster="S1", actor_id=2, character_id=2, score=0.9,
                              clean_seconds=1, status="matched"),
        ],
        turns=[
            Turn(start=0, end=9, speaker="S0"),  # longest, but S1 talks over it
            Turn(start=8, end=10, speaker="S1"),
            Turn(start=20, end=24, speaker="S0"),
            Turn(start=30, end=33, speaker="S0"),
            Turn(start=40, end=42, speaker="S0"),
            Turn(start=50, end=50.5, speaker="S0"),  # too short
            Turn(start=60, end=61, speaker="S0"),
        ],
    )  # fmt: skip
    assert ev.clean_exemplars(asg, 1) == [(20, 24), (30, 33), (40, 42)]
    assert ev.clean_exemplars(asg, 2) == []  # its one turn overlaps S0


def test_to_16k_writes_mono_pcm16(tmp_path):
    from scipy.io import wavfile

    src = tmp_path / "a.wav"
    wavfile.write(src, 24000, (0.1 * np.ones(24000)).astype(np.float32))
    dest = ev.to_16k(src, tmp_path / "a.16k.wav")
    with wave.open(str(dest)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() == 16000


def _score(**kw: Any) -> ev.CueScore:
    base: dict[str, Any] = {
        "episode": "S01E01", "cue_id": "c1", "character_id": 1, "card_id": "k1",
        "backend": "fake", "voice": "am_adam", "text": "0123456789", "seconds": 1.0,
        "slot": 2.0, "similarity": 0.2, "similarity_source": "original", "hyp": "",
        "errors": 0, "ref_words": 2,
    }  # fmt: skip
    return ev.CueScore(**{**base, **kw})


def test_summarize_per_character_overall_and_per_card():
    scores = [
        _score(cue_id="c1", seconds=1.0, slot=2.0, similarity=0.2, errors=0, ref_words=4),
        _score(cue_id="c2", seconds=3.0, slot=2.0, similarity=0.4, errors=2, ref_words=4),
        _score(cue_id="c3", character_id=2, card_id="k2", voice="af_heart", seconds=2.0,
               slot=1.0, similarity=0.6, errors=1, ref_words=2, text="01234"),
    ]  # fmt: skip
    s = ev.summarize(scores, max_speedup=1.25)
    assert s.cues == 3
    one = s.per_character[1]
    assert one["speaker_similarity"] == pytest.approx(0.3)
    assert one["wer"] == pytest.approx(2 / 8, abs=1e-4)
    assert one["fit_rate"] == pytest.approx(0.5)
    assert one["stretch_median"] == pytest.approx(1.25)  # 1.0 and 1.5
    assert s.overall["wer"] == pytest.approx(3 / 10, abs=1e-4)
    assert s.overall["fit_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s.overall["fit_budget_rate"] == pytest.approx(1 / 3, abs=1e-4)  # 1.5 and 2.0 exceed 1.25
    assert s.overall["characters"] == 2 and s.overall["cues"] == 3
    assert s.overall["speaker_similarity"] == pytest.approx(0.4)
    assert "emotion_agreement" not in s.overall and "ref_duration_ratio" not in s.overall
    # Speech rate per card: characters per second of the trimmed render (ruling CD8).
    assert s.per_card["k1"]["cps_median"] == pytest.approx(np.median([10.0, 10 / 3]), abs=1e-4)
    assert s.per_card["k2"]["cps_median"] == pytest.approx(2.5)
    assert s.per_card["k2"]["voice"] == "af_heart" and s.per_card["k2"]["cues"] == 1
    assert s.similarity_source == {1: "original", 2: "original"}


def test_summarize_emotion_and_reference_metrics_only_when_present():
    scores = [
        _score(cue_id="c1", source_emotion="happy", render_emotion="happy", ref_seconds=0.5,
               ref_emotion="happy", ref_similarity=0.5),
        _score(cue_id="c2", source_emotion="angry", render_emotion="neutral", ref_seconds=2.0,
               ref_emotion="angry", ref_similarity=0.3),
        _score(cue_id="c3", source_emotion=None, render_emotion="neutral"),
    ]  # fmt: skip
    s = ev.summarize(scores, max_speedup=1.25)
    assert s.overall["emotion_agreement"] == pytest.approx(0.5)
    assert s.overall["emotion_cues"] == 2
    assert s.overall["ref_duration_ratio"] == pytest.approx(np.median([2.0, 0.5]))
    assert s.overall["ref_emotion_agreement"] == pytest.approx(0.5)
    assert s.overall["ref_speaker_similarity"] == pytest.approx(0.4)
    assert s.overall["ref_cues"] == 2


def test_summarize_without_similarity_reports_it_missing():
    s = ev.summarize([_score(similarity=None, similarity_source="none")], max_speedup=1.25)
    assert "speaker_similarity" not in s.overall
    assert s.similarity_source == {1: "none"}


def test_tts_row_uses_pair_language_and_series_genre(tmp_path):
    summary = ev.summarize([_score()], max_speedup=1.25)
    cfg = Config(series=SeriesInfo("Fairy Tail", "anime", {}))
    rows = ev.tts_rows(
        summary, cfg=cfg, backend_cfg=BackendConfig("kokoro", {}), source_language="ja",
        test_set="fairytail", split="tune", episodes=["S01E11"], tag=None, workspace=tmp_path,
        options={"mode": "assigned"},
    )  # fmt: skip
    (r,) = rows
    assert (r.stage, r.language, r.genre, r.series, r.split) == (
        "tts", "ja-en", "anime", "Fairy Tail", "tune",
    )  # fmt: skip
    assert (r.backend, r.model, r.revision) == ("kokoro", "kokoro-model", "r1")
    assert "speaker_similarity" in r.metrics and "ref_duration_ratio" not in r.metrics
    for key in ("wer", "fit_rate", "stretch_median", "stretch_p90", "cues", "characters"):
        assert key in r.metrics
    assert r.options["mode"] == "assigned"
    assert set(r.detail) >= {"per_character", "per_card", "similarity_source"}
    assert r.workspace == str(tmp_path)


def test_report_shows_tts_column(tmp_path):
    summary = ev.summarize([_score()], max_speedup=1.25)
    cfg = Config(series=SeriesInfo("Fairy Tail", "anime", {}))
    (r,) = ev.tts_rows(
        summary, cfg=cfg, backend_cfg=BackendConfig("kokoro", {}), source_language="ja",
        test_set="fairytail", split="tune", episodes=["S01E11"], tag="baseline",
        workspace=tmp_path,
    )  # fmt: skip
    m = build_matrix([r])
    assert m.columns == ["ja-en/anime"]
    assert m.rows[0][1] == ["0.2"]


# ---- the command on fakes -----------------------------------------------------------------

LOG: list[str] = []


class FakeSynth(Synthesizer):
    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "tts"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    sources: ClassVar = frozenset({"stock", "clone"})
    voice_keys: ClassVar = frozenset({"rate"})
    capabilities: ClassVar = frozenset()
    sample_rate: ClassVar[int] = 24000
    speech_cps: ClassVar = {"en": 15.0}
    model_revision = "1.0"
    CALLS: ClassVar[list[str]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:  # noqa: ARG002
        self.options = options

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        pass

    def load(self) -> None:
        LOG.append(f"load {self.name}")

    def unload(self) -> None:
        LOG.append(f"unload {self.name}")

    def synthesize(self, text: str, voice: Voice, language: str, out: Path, **_kw: Any) -> float:  # noqa: ARG002
        self.CALLS.append(f"{voice.voice}:{text}")
        seconds = len(text) * 0.1 / voice.rate
        tone_wav(out, seconds, self.sample_rate, freq=200 if voice.voice == "am_adam" else 300)
        return seconds


class FakeSynth2(FakeSynth):
    name: ClassVar[str] = "fake2"


class FakeEmbed:
    name: ClassVar[str] = "wespeaker"
    kind: ClassVar[str] = "embed"
    vram_mb: ClassVar[int] = 0
    dim: ClassVar[int] = 2

    def __init__(self, options: dict[str, Any], device: str) -> None:
        pass

    def load(self) -> None:
        LOG.append("load wespeaker")

    def unload(self) -> None:
        LOG.append("unload wespeaker")

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        assert audio.name.endswith("16k.wav") or audio.name == "vocals_16k.wav"
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(spans), 1))


class FakeAsr:
    name: ClassVar[str] = "faster_whisper"
    kind: ClassVar[str] = "asr"
    vram_mb: ClassVar[int] = 0
    TEXT: ClassVar[dict[str, str]] = {}

    def __init__(self, options: dict[str, Any], device: str) -> None:  # noqa: ARG002
        self.model_name = options.get("model", "large-v3")

    def load(self) -> None:
        LOG.append("load faster_whisper")

    def unload(self) -> None:
        LOG.append("unload faster_whisper")

    def transcribe(self, audio: Path, language: str | None) -> Transcript:
        assert language == "en"
        n = len(FakeAsr.TEXT)
        FakeAsr.TEXT[audio.name] = "heard"
        text = "natsu one" if n == 0 else "something else"
        return Transcript(
            language="en", backend="faster_whisper", word_timing=False,
            segments=[Segment(start=0.0, end=1.0, text=text)],
        )  # fmt: skip


class FakeTagger:
    name: ClassVar[str] = "emotion2vec"
    kind: ClassVar[str] = "tag"
    vram_mb: ClassVar[int] = 0

    def __init__(self, options: dict[str, Any], device: str) -> None:
        pass

    def load(self) -> None:
        LOG.append("load emotion2vec")

    def unload(self) -> None:
        LOG.append("unload emotion2vec")

    def tag(self, audio: Path, turns: list[Turn]) -> Tags:  # noqa: ARG002
        return Tags(
            backend="emotion2vec", events=[],
            emotions=[TagSpan(start=t.start, end=t.end, label="happy", confidence=0.9)
                      for t in turns],
        )  # fmt: skip


CLASSES: dict[tuple[str, str], Any] = {
    ("tts", "fake"): FakeSynth,
    ("tts", "fake2"): FakeSynth2,
    ("embed", "wespeaker"): FakeEmbed,
    ("asr", "faster_whisper"): FakeAsr,
    ("tag", "emotion2vec"): FakeTagger,
}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    LOG.clear()
    FakeSynth.CALLS = []
    FakeAsr.TEXT = {}
    monkeypatch.setattr(loader, "backend_class", lambda kind, name: CLASSES[(kind, name)])
    monkeypatch.setattr(
        loader, "create", lambda kind, bc, dev: CLASSES[(kind, bc.backend)](bc.options, dev)
    )
    ident = lambda _k, bc: {"model": f"{bc.backend}-model", "revision": "r1"}  # noqa: E731
    monkeypatch.setattr(tts_stage, "identity_key", ident)
    monkeypatch.setattr(ev, "identity_key", ident)
    monkeypatch.setattr(ev, "model_identity", lambda _k, bc: (f"{bc.backend}-model", "r1"))
    monkeypatch.setattr(ev, "device_string", lambda _cfg: "cpu")
    pools = {"f": ["af_heart"], "m": ["am_adam"], "extras_f": ["xf_one"], "extras_m": ["xm_one"]}
    for name in ("fake", "fake2"):
        monkeypatch.setitem(stock.STOCK_POOLS, name, {"en": pools})


CARD = """\
target_language = "en"

[voice]
backend = "{backend}"
model_revision = "1.0"
source = "stock"
voice = "{voice}"
"""


def _workspace(tmp_path: Path, *, emotions: bool | str = True, name: str = "ws") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "fukikae.toml").write_text(
        'source_language = "ja"\n\n[series]\ntitle = "Fairy Tail"\ngenre = "anime"\n\n'
        '[tts]\nbackend = "fake"\n',
        encoding="utf-8",
    )
    work = Workspace(root).work_dir("S01E11")
    for d in ("04b-tag", "05-characters", "06-translate"):
        (work / d).mkdir(parents=True)
    tone_wav(work / "01-separate" / "vocals_16k.wav", 40.0, 16000)
    tone_wav(work / "01-separate" / "vocals.wav", 40.0, 16000)
    with Registry(Workspace(root).registry_path) as reg:
        _, natsu = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
        _, lucy = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
        reg.rename_character(natsu, "Natsu")
        reg.rename_character(lucy, "Lucy")
        store = CardStore(root)
        for cid, voice in ((natsu, "am_adam"), (lucy, "af_heart")):
            card = store.put(parse_card(CARD.format(backend="fake", voice=voice)))
            reg.assign_card(cid, "en", card, "assign", True)
    write_json(
        work / "05-characters" / "assignments.json",
        Assignments(
            episode="S01E11", threshold=0.6,
            clusters=[
                ClusterAssignment(cluster="S0", actor_id=1, character_id=1, score=0.9,
                                  clean_seconds=9, status="matched"),
                ClusterAssignment(cluster="S1", actor_id=2, character_id=2, score=0.9,
                                  clean_seconds=9, status="matched"),
            ],
            turns=[Turn(start=0, end=2, speaker="S0"), Turn(start=3, end=5, speaker="S1"),
                   Turn(start=6, end=8, speaker="S0"), Turn(start=9, end=12, speaker="S1")],
        ),
    )  # fmt: skip
    emo: dict[str, Any] = {"emotion": "happy"} if emotions else {}
    cues = [
        cue(1, 1, "natsu one", **emo),
        cue(2, 2, "lucy one", **(emo if emotions == "all" else {})),
        cue(3, 1, "natsu two is longer than its slot", **(emo if emotions == "all" else {})),
        cue(4, None, "extra line", extra="S5"),
        cue(5, 2, None, skip="song"),
    ]
    write_json(
        work / "06-translate" / "cues.en.json",
        DubCues(episode="S01E11", source_language="ja", target_language="en", path="asr",
                cues=cues),
    )  # fmt: skip
    if emotions:
        write_json(
            work / "04b-tag" / "tags.json",
            Tags(backend="sensevoice", emotion_backend="emotion2vec", events=[], emotions=[]),
        )
    return root


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(app, ["eval", "tts", *args], catch_exceptions=False)


def test_cli_writes_one_row_to_the_isolated_ledger(tmp_path):
    root = _workspace(tmp_path)
    res = _invoke(str(root), "--episodes", "S01E11", "--set", "fairytail", "--tag", "baseline")
    assert res.exit_code == 0, res.output
    rows, warnings = read_rows(ledger_path())
    assert not warnings
    (r,) = rows
    assert (r.stage, r.backend, r.language, r.genre, r.series) == (
        "tts", "fake", "ja-en", "anime", "Fairy Tail",
    )  # fmt: skip
    assert (r.model, r.revision) == ("fake-model", "r1")
    assert r.metrics["cues"] == 3 and r.metrics["characters"] == 2
    assert r.metrics["speaker_similarity"] == pytest.approx(1.0)
    # One of three source lines labelled: under half, so no headline agreement.
    assert "emotion_agreement" not in r.metrics
    assert r.metrics["emotion_low_coverage"] == 1
    assert r.metrics["emotion_source_coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert r.metrics["emotion_render_coverage"] == pytest.approx(1.0)
    assert r.detail["emotion"]["source_backend"] == "emotion2vec"
    assert r.detail["emotion"]["render_backend"] == "emotion2vec"
    assert r.metrics["fit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert r.options["mode"] == "assigned"
    assert r.detail["similarity_source"] == {"1": "original", "2": "original"}
    assert set(r.detail["per_card"]) and r.episodes == ["S01E11"] and r.tag == "baseline"
    # One model at a time, in the plan's order.
    assert LOG == [
        "load fake", "unload fake", "load wespeaker", "unload wespeaker",
        "load faster_whisper", "unload faster_whisper", "load emotion2vec", "unload emotion2vec",
    ]  # fmt: skip
    assert "Natsu" in res.output and "similarity to" in res.output
    renders = sorted((root / ".fukikae" / "eval" / "tts" / "fake-r1").glob("S01E11-c*.wav"))
    assert len([p for p in renders if not p.name.endswith(".16k.wav")]) == 3


def test_second_run_reuses_renders_and_the_tts_cache(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    assert _invoke(str(root), "--episodes", "S01E11", "--no-ledger").exit_code == 0
    assert len(FakeSynth.CALLS) == 3
    FakeSynth.CALLS.clear()
    LOG.clear()
    assert _invoke(str(root), "--episodes", "S01E11", "--no-ledger").exit_code == 0
    assert FakeSynth.CALLS == [] and "load fake" not in LOG
    assert "load emotion2vec" not in LOG  # no source labels: the tagger never loads
    assert read_rows(ledger_path())[0] == []


def test_renders_from_the_tts_stage_cache_are_linked_not_rerendered(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    cfg = Workspace(root).config
    store = CardStore(root)
    cache = root / ".fukikae" / "tts-cache"
    cache.mkdir(parents=True)
    with Registry(Workspace(root).registry_path) as reg:
        cards: dict[int | None, str] = {
            cid: str(reg.card_for(cid, "en").card_id)  # type: ignore[union-attr]
            for cid in (1, 2)
        }
    for c in [cue(1, 1, "natsu one"), cue(2, 2, "lucy one"),
              cue(3, 1, "natsu two is longer than its slot")]:  # fmt: skip
        key = tts_stage.cue_key(
            cards[c.character_id], c.text or "", c.delivery, BackendConfig("fake", {}),
            capabilities=FakeSynth.capabilities, identity={"model": "fake-model", "revision": "r1"},
            slot=c.end - c.start,
        )  # fmt: skip
        tone_wav(cache / f"{key}.wav", 0.5, 24000)
    assert store.has(cards[1]) and cfg.target_language == "en"
    assert _invoke(str(root), "--episodes", "S01E11", "--no-ledger").exit_code == 0
    assert FakeSynth.CALLS == []


def test_backend_option_renders_stock_cards_of_that_backend(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    res = _invoke(str(root), "--episodes", "S01E11", "--backend", "fake2")
    assert res.exit_code == 0, res.output
    (r,) = read_rows(ledger_path())[0]
    assert r.backend == "fake2" and r.options["mode"] == "stock"
    assert (root / ".fukikae" / "eval" / "tts" / "fake2-r1").is_dir()


def test_reference_workspace_adds_duration_and_delivery_against_the_original(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    ref = tmp_path / "en"
    work = Workspace(ref).work_dir("S01E11")
    tone_wav(work / "01-separate" / "vocals_16k.wav", 40.0, 16000)
    (work / "03-asr").mkdir(parents=True)
    write_json(
        work / "03-asr" / "transcript.json",
        _transcript([(3.1, 3.5, "hello"), (3.6, 4.1, "there"), (6.2, 7.2, "longer")]),
    )
    res = _invoke(str(root), "--episodes", "S01E11", "--reference-workspace", str(ref))
    assert res.exit_code == 0, res.output
    (r,) = read_rows(ledger_path())[0]
    assert r.metrics["ref_cues"] == 2  # c0001 (3-5 s) and c0002 (6-8 s); c0003 has no words
    assert "ref_duration_ratio" in r.metrics
    assert r.metrics["ref_emotion_agreement"] == pytest.approx(1.0)
    assert r.metrics["ref_speaker_similarity"] == pytest.approx(1.0)
    assert "load emotion2vec" in LOG


def test_missing_translation_names_the_episode(tmp_path):
    root = _workspace(tmp_path)
    res = _invoke(str(root), "--episodes", "S01E12")
    assert res.exit_code == 1
    assert "S01E12" in res.output and "translate" in res.output


def test_bad_split_is_refused(tmp_path):
    root = _workspace(tmp_path)
    res = _invoke(str(root), "--episodes", "S01E11", "--split", "test")
    assert res.exit_code == 2


def test_render_neutral_share_counts_only_labelled_renders():
    scores = [
        _score(cue_id="c1", source_emotion="happy", render_emotion="neutral"),
        _score(cue_id="c2", source_emotion="sad", render_emotion="sad"),
        _score(cue_id="c3"),  # never tagged: no source label
        _score(cue_id="c4"),
    ]
    s = ev.summarize(scores, max_speedup=1.25)
    assert s.overall["render_neutral_share"] == pytest.approx(0.5)
    assert s.overall["render_labelled"] == 2


# ---- fix round 1 --------------------------------------------------------------------------


def test_unlabelled_renders_are_not_misses():
    scores = [
        _score(cue_id="c1", source_emotion="happy", render_emotion="happy"),
        _score(cue_id="c2", source_emotion="sad", render_emotion=None),
        _score(cue_id="c3", source_emotion="sad", render_emotion="sad"),
        _score(cue_id="c4", source_emotion="angry", render_emotion="neutral"),
    ]
    s = ev.summarize(scores, max_speedup=1.25)
    assert s.overall["emotion_cues"] == 3
    assert s.overall["emotion_agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert s.overall["emotion_render_coverage"] == pytest.approx(0.75)
    assert "emotion_low_coverage" not in s.overall


def test_low_render_coverage_flags_the_row(tmp_path):
    scores = [
        _score(cue_id="c1", source_emotion="happy", render_emotion="happy"),
        _score(cue_id="c2", source_emotion="sad"),
        _score(cue_id="c3", source_emotion="sad"),
    ]
    s = ev.summarize(scores, max_speedup=1.25)
    assert "emotion_agreement" not in s.overall
    assert s.overall["emotion_low_coverage"] == 1
    assert s.emotion["agreement"] == pytest.approx(1.0)
    (r,) = ev.tts_rows(
        s, cfg=Config(), backend_cfg=BackendConfig("kokoro", {}), source_language="ja",
        test_set="t", split="tune", episodes=["S01E11"], tag=None, workspace=tmp_path,
    )  # fmt: skip
    assert r.metrics["emotion_low_coverage"] == 1 and "emotion_agreement" not in r.metrics
    assert r.detail["emotion"]["low_coverage"] is True


def test_neutral_only_backend_is_noted():
    s = ev.summarize(
        [_score(source_emotion="happy", render_emotion="neutral", neutral_only=True)],
        max_speedup=1.25,
    )
    assert "neutral share" in s.emotion["note"]


def test_similarity_is_split_by_reference_source():
    scores = [
        _score(cue_id="c1", similarity=0.8, similarity_source="card"),
        _score(cue_id="c2", character_id=2, similarity=0.1, similarity_source="original"),
    ]
    s = ev.summarize(scores, max_speedup=1.25)
    assert s.overall["speaker_similarity_card"] == pytest.approx(0.8)
    assert s.overall["speaker_similarity_original"] == pytest.approx(0.1)
    assert "speaker_similarity" not in s.overall  # never pooled across sources
    one = ev.summarize(scores[1:], max_speedup=1.25)
    assert one.overall["speaker_similarity"] == pytest.approx(0.1)
    assert one.overall["speaker_similarity_original"] == pytest.approx(0.1)


def test_sample_skips_short_slots_and_counts_them():
    cues = [cue(1, 1, "short", slot=0.6), cue(2, 1, "long enough", slot=1.5)]
    docs = {"S01E01": _doc(cues)}
    assert [c.text for _, c in ev.sample_cues(docs, min_slot=1.0)] == ["long enough"]
    assert ev.count_short(docs, min_slot=1.0) == 1
    assert len(ev.sample_cues(docs, min_slot=0.0)) == 2


def test_reference_span_default_pad_is_tight():
    tr = _transcript([(1.85, 2.05, "edge"), (2.5, 3.0, "in")])
    # "edge" has its midpoint 0.05 s before the slot: inside the 0.1 s pad.
    assert ev.reference_span(tr, (2.0, 3.2)) == pytest.approx((1.85, 3.0))
    tr2 = _transcript([(1.5, 1.8, "early"), (2.5, 3.0, "in")])
    assert ev.reference_span(tr2, (2.0, 3.2)) == pytest.approx((2.5, 3.0))


def test_sidecar_stem_names_the_clip():
    assert ev.sidecar_stem("sha256:" + "ab" * 32) == "ab" * 8


def _floor_subs(path: Path) -> Path:
    path.write_text(
        "1\n00:00:03,000 --> 00:00:04,500\nhello there my friend\n\n"
        "2\n00:00:06,100 --> 00:00:07,600\nthis is a longer line\n\n"
        "3\n00:00:20,000 --> 00:00:21,500\nnobody dubbed this one\n\n"
        "4\n00:00:30,000 --> 00:00:30,400\ntoo short ok\n",
        encoding="utf-8",
    )
    return path


def _en_workspace(tmp_path: Path) -> Path:
    ref = tmp_path / "en"
    work = Workspace(ref).work_dir("S01E11")
    tone_wav(work / "01-separate" / "vocals_16k.wav", 40.0, 16000)
    (work / "03-asr").mkdir(parents=True)
    write_json(
        work / "03-asr" / "transcript.json",
        _transcript([(3.1, 3.5, "hello"), (3.6, 4.1, "there"), (6.2, 7.2, "longer")]),
    )
    return ref


def test_wer_floor_from_the_original_track_on_the_sampled_cues(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    ref = _en_workspace(tmp_path)
    subs = _floor_subs(tmp_path / "S01E11.srt")
    res = _invoke(
        str(root), "--episodes", "S01E11", "--reference-workspace", str(ref),
        "--floor-subs", str(subs),
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    (r,) = read_rows(ledger_path())[0]
    assert "wer_floor" in r.metrics and r.metrics["wer_floor_clips"] == 2
    assert r.options["wer_floor"].startswith("reference:")
    assert r.metrics["ref_unmatched"] == 1


def test_wer_floor_from_a_fixed_clip_set_without_a_reference(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    ref = _en_workspace(tmp_path)
    subs = _floor_subs(tmp_path / "S01E11.srt")
    res = _invoke(
        str(root), "--episodes", "S01E11", "--floor-workspace", str(ref),
        "--floor-subs", str(subs), "--floor-episode", "S01E11",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    (r,) = read_rows(ledger_path())[0]
    assert r.metrics["wer_floor_clips"] == 3  # the 0.4 s cue is too short
    assert r.options["wer_floor"].startswith("fixed:")
    assert "ref_cues" not in r.metrics


def test_min_slot_is_an_option_and_exclusions_are_counted(tmp_path):
    root = _workspace(tmp_path, emotions=False)
    res = _invoke(str(root), "--episodes", "S01E11", "--min-slot", "2.5")
    # Every character cue has a 2.0 s slot: nothing left to render.
    assert res.exit_code == 1 and "no cue to render" in res.output
    res = _invoke(str(root), "--episodes", "S01E11", "--min-slot", "1.5")
    assert res.exit_code == 0, res.output
    (r,) = read_rows(ledger_path())[0]
    assert r.options["min_slot"] == 1.5 and r.metrics["excluded_short"] == 0


def test_full_coverage_reports_agreement(tmp_path):
    root = _workspace(tmp_path, emotions="all")
    assert _invoke(str(root), "--episodes", "S01E11").exit_code == 0
    (r,) = read_rows(ledger_path())[0]
    assert r.metrics["emotion_agreement"] == pytest.approx(1.0)
    assert "emotion_low_coverage" not in r.metrics


def test_mixed_taggers_across_episodes_are_refused(tmp_path):
    root = _workspace(tmp_path)
    src = Workspace(root).work_dir("S01E11")
    dst = Workspace(root).work_dir("S01E12")
    shutil.copytree(src, dst)
    write_json(dst / "04b-tag" / "tags.json", Tags(backend="sensevoice", events=[], emotions=[]))
    res = _invoke(str(root), "--episodes", "S01E11,S01E12")
    assert res.exit_code == 1 and "tagger" in res.output


def test_stage_and_eval_render_the_same_audio_under_the_same_key(tmp_path):
    from fukikae.artifacts import read_json
    from fukikae.models import TtsPlan
    from fukikae.stages.tts import TtsStage
    from fukikae.workspace import Episode

    a = _workspace(tmp_path, emotions=False, name="a")
    b = _workspace(tmp_path, emotions=False, name="b")
    assert _invoke(str(a), "--episodes", "S01E11", "--no-ledger").exit_code == 0
    cfg = Workspace(b).config
    ep = Episode(
        source=tmp_path / "s.mkv", id="S01E11", work=Workspace(b).work_dir("S01E11"),
        workspace_root=b,
    )  # fmt: skip
    TtsStage(cfg).run(ep, cfg)
    plan = {c.cue_id: c for c in read_json(ep.work / "07-tts" / "plan.json", TtsPlan).cues}
    folder = a / ".fukikae" / "eval" / "tts" / "fake-r1"
    for cue_id in ("c0001", "c0002", "c0003"):
        (render,) = [p for p in folder.glob(f"S01E11-{cue_id}.*.wav") if ".16k." not in p.name]
        key = render.name.split(".")[1]
        cached = b / ".fukikae" / "tts-cache" / f"{key}.wav"
        assert cached.read_bytes() == render.read_bytes()
        if plan[cue_id].rate_factor == 1.0:
            assert plan[cue_id].key == key
