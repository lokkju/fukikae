import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from fukikae.artifacts import read_json, write_json
from fukikae.backends import loader
from fukikae.backends.base import Synthesizer
from fukikae.cards import stock
from fukikae.cards.card import CardError, Voice, parse_card
from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    DubCues,
    TtsPlan,
    TtsRun,
    Turn,
)
from fukikae.registry.store import Registry
from fukikae.stages import tts
from fukikae.stages.tts import TtsStage, cue_key
from fukikae.workspace import Episode, Workspace

CARD = """\
target_language = "en"

[voice]
backend = "{backend}"
model_revision = "{revision}"
source = "stock"
voice = "{voice}"
"""

CLONE = """\
target_language = "en"

[voice]
backend = "fake"
model_revision = "1.0"
source = "clone"

[[voice.references]]
clip = "sha256:9e1f000000000000000000000000000000000000000000000000000000000000"
transcript = "なんだよ"
language = "ja"
"""


def tone_wav(
    path: Path, seconds: float, rate: int, amp: float = 0.25, pad: tuple[float, float] = (0, 0)
) -> float:
    n = max(1, round(seconds * rate))
    x = amp * np.sin(2 * np.pi * 440 * np.arange(n) / rate)
    x = np.concatenate([np.zeros(round(pad[0] * rate)), x, np.zeros(round(pad[1] * rate))])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return len(x) / rate


class FakeSynth(Synthesizer):
    """Writes a tone of DUR[text] (default len(text) * 0.06) seconds divided by the rate."""

    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "tts"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    sources: ClassVar = frozenset({"stock", "clone"})
    voice_keys: ClassVar = frozenset({"rate"})
    capabilities: ClassVar = frozenset()
    sample_rate: ClassVar[int] = 8000
    model_revision = "1.0"
    stock_voices: ClassVar = frozenset({"af_heart", "am_adam", "bf_emma", "xf_one", "xm_one"})
    LOG: ClassVar[list[str]] = []
    CALLS: ClassVar[list[tuple[str, str | None, float]]] = []
    KW: ClassVar[list[dict[str, Any]]] = []  # delivery keywords per call
    DUR: ClassVar[dict[str, float]] = {}
    SEED_DUR: ClassVar[dict[tuple[str, int | None], float]] = {}  # (text, seed) wins over DUR
    SEEDS: ClassVar[list[tuple[str, int | None]]] = []
    PAD: ClassVar[tuple[float, float]] = (0.0, 0.0)  # silence before and after, like Kokoro

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.options = options
        self.device = device

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        pass

    def load(self) -> None:
        self.LOG.append(f"load {self.name}")

    def unload(self) -> None:
        self.LOG.append(f"unload {self.name}")

    def synthesize(self, text: str, voice: Voice, language: str, out: Path, **kw: Any) -> float:  # noqa: ARG002
        self.CALLS.append((text, voice.voice, voice.rate))
        kw.pop("resolve", None)
        emo = kw.get("emotion_audio")
        kw["emotion_audio_exists"] = emo is not None and Path(emo).is_file()
        self.KW.append({"text": text, **kw})
        self.SEEDS.append((text, voice.seed))
        base = self.SEED_DUR.get((text, voice.seed), self.DUR.get(text, len(text) * 0.06))
        seconds = base / voice.rate
        return tone_wav(out, seconds, self.sample_rate, pad=self.PAD)


class FakeSynth2(FakeSynth):
    name: ClassVar[str] = "fake2"


CLASSES = {"fake": FakeSynth, "fake2": FakeSynth2}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeSynth.LOG = []
    FakeSynth.CALLS = []
    FakeSynth.KW = []
    FakeSynth.DUR = {}
    FakeSynth.SEED_DUR = {}
    FakeSynth.SEEDS = []
    FakeSynth.PAD = (0.0, 0.0)
    monkeypatch.setattr(loader, "backend_class", lambda _kind, name: CLASSES[name])
    monkeypatch.setattr(
        loader, "create", lambda _kind, bc, dev: CLASSES[bc.backend](bc.options, dev)
    )
    monkeypatch.setattr(
        tts, "identity_key", lambda _k, bc: {"model": f"{bc.backend}-model", "revision": "r"}
    )
    monkeypatch.setattr(tts, "device_string", lambda _cfg: "cpu")
    pools = {"f": ["af_heart"], "m": ["am_adam"], "extras_f": ["xf_one"], "extras_m": ["xm_one"]}
    for name in CLASSES:
        monkeypatch.setitem(stock.STOCK_POOLS, name, {"en": pools})


def _cue(i: int, character: int | None, text: str | None, **kw: Any) -> DubCue:
    start = i * 3.0
    return DubCue(
        id=f"c{i:04d}",
        start=start,
        end=start + kw.pop("slot", 2.0),
        character_id=character,
        extra=kw.pop("extra", None),
        speaker="x",
        source_text="src",
        text=text,
        budget=40,
        delivery=Delivery(reference=(start, start + 2.0)),
        **kw,
    )


CUES = [
    _cue(1, 1, "natsu one"),
    _cue(2, 2, "lucy one"),
    _cue(3, 1, "natsu two"),
    _cue(4, None, "extra line", extra="S5"),
    _cue(5, 2, None, skip="song"),
    _cue(6, 1, None, error="translate: response is not JSON"),
    _cue(7, None, "extra again", extra="S5"),
    _cue(8, None, "review line", extra="S9"),  # a cluster the characters stage marked review
]


def _write_cues(ep: Episode, cues: list[DubCue]) -> None:
    write_json(
        ep.work / "06-translate" / "cues.en.json",
        DubCues(episode="e1", source_language="ja", target_language="en", path="asr", cues=cues),
    )


@pytest.fixture
def ep(tmp_path: Path) -> Episode:
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "e1"
    for d in ("01-separate", "05-characters", "06-translate"):
        (work / d).mkdir(parents=True)
    tone_wav(work / "01-separate" / "vocals.wav", 30.0, 16000, amp=0.0)
    with Registry(Workspace(root).registry_path) as reg:
        _, natsu = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
        _, lucy = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
        reg.rename_character(natsu, "Natsu")
        reg.rename_character(lucy, "Lucy")
    write_json(
        work / "05-characters" / "assignments.json",
        Assignments(
            episode="e1",
            threshold=0.6,
            clusters=[
                ClusterAssignment(
                    cluster="S0", actor_id=1, character_id=1, score=0.9, clean_seconds=30,
                    status="matched",
                ),
                ClusterAssignment(
                    cluster="S1", actor_id=2, character_id=2, score=0.9, clean_seconds=30,
                    status="matched",
                ),
            ],
            turns=[Turn(start=0.0, end=2.5, speaker="S0"), Turn(start=3.0, end=5.0, speaker="S1")],
        ),
    )  # fmt: skip
    episode = Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)
    _write_cues(episode, CUES)
    return episode


CFG = Config(raw={"tts": {"backend": "fake"}})


def _assign(ep: Episode, character: int, text: str, by: str = "assign") -> str:
    cid = CardStore(ep.workspace_root).put(parse_card(text))
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        reg.assign_card(character, "en", cid, by, True)
    return cid


def _stock(voice: str, backend: str = "fake", revision: str = "1.0") -> str:
    return CARD.format(backend=backend, revision=revision, voice=voice)


def _plan(ep: Episode) -> TtsPlan:
    return read_json(ep.work / "07-tts" / "plan.json", TtsPlan)


def _run(ep: Episode) -> TtsRun:
    return read_json(ep.work / "07-tts" / "run.json", TtsRun)


def _calls_for(texts: set[str]) -> list[str]:
    return [t for t, _, _ in FakeSynth.CALLS if t in texts]


NATSU = {"natsu one", "natsu two"}
LUCY = {"lucy one"}
EXTRAS = {"extra line", "extra again", "review line"}


def _two_cards(ep: Episode) -> tuple[str, str]:
    return _assign(ep, 1, _stock("am_adam")), _assign(ep, 2, _stock("af_heart"))


def test_renders_each_spoken_cue_and_links_it_from_the_cache(ep):
    _two_cards(ep)
    TtsStage(CFG).run(ep, CFG)
    plan = _plan(ep)
    by_id = {c.cue_id: c for c in plan.cues}
    rendered = [c for c in plan.cues if c.skipped is None]
    assert [c.cue_id for c in rendered] == ["c0001", "c0002", "c0003", "c0004", "c0007", "c0008"]
    cache = ep.workspace_root / ".fukikae" / "tts-cache"
    texts = {x.id: x.text or "" for x in CUES}
    for c in rendered:
        out = ep.work / "07-tts" / f"{c.cue_id}.wav"
        assert out.is_file() and c.key is not None
        assert out.stat().st_ino == (cache / f"{c.key}.wav").stat().st_ino
        assert c.seconds == pytest.approx(len(texts[c.cue_id]) * 0.06, abs=1e-3)
        assert c.slot == pytest.approx(2.0) and c.rate_factor == 1.0
        assert c.rendered_seconds == c.seconds
        assert c.neutral_only and c.passed == {}
    assert _run(ep).cached == [] and sorted(_run(ep).rendered) == [c.cue_id for c in rendered]
    assert by_id["c0001"].character_id == 1 and by_id["c0001"].backend == "fake"
    assert FakeSynth.LOG == ["load fake", "unload fake"]


def test_extras_get_pool_voices_and_render(ep):
    """Review Focus 1: an extra, or a review cluster, gets an extras-pool voice that stays the
    same for the same label within the episode."""
    _two_cards(ep)
    TtsStage(CFG).run(ep, CFG)
    by_id = {c.cue_id: c for c in _plan(ep).cues}
    assert by_id["c0004"].extra == "S5" and by_id["c0004"].character_id is None
    assert by_id["c0004"].card_id == by_id["c0007"].card_id
    assert by_id["c0008"].card_id not in (None, by_id["c0004"].card_id)
    voices = {t: v for t, v, _ in FakeSynth.CALLS}
    assert voices["extra line"] == voices["extra again"] == "xf_one"
    assert voices["review line"] == "xm_one"


def test_second_run_renders_nothing_and_loads_nothing(ep):
    _two_cards(ep)
    TtsStage(CFG).run(ep, CFG)
    FakeSynth.CALLS.clear()
    FakeSynth.LOG.clear()
    TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.CALLS == [] and FakeSynth.LOG == []
    assert _run(ep).rendered == [] and len(_run(ep).cached) == 6


def test_swapping_one_card_rerenders_only_that_character(ep):
    _two_cards(ep)
    stage = TtsStage(CFG)
    key_before = stage.config_key(CFG, ep)
    stage.run(ep, CFG)
    FakeSynth.CALLS.clear()
    new = _assign(ep, 1, _stock("bf_emma"))
    assert stage.config_key(CFG, ep) != key_before
    assert CardStore(ep.workspace_root).card_path(new) in stage.inputs(ep)
    stage.run(ep, CFG)
    assert sorted(_calls_for(NATSU)) == sorted(NATSU)
    assert _calls_for(LUCY) == [] and _calls_for(EXTRAS) == []
    assert len(FakeSynth.CALLS) == 2
    by_id = {c.cue_id: c for c in _plan(ep).cues}
    assert by_id["c0001"].card_id == new
    assert sorted(_run(ep).rendered) == ["c0001", "c0003"] and "c0002" in _run(ep).cached


def test_auto_assigns_a_character_without_a_card(ep):
    natsu = _assign(ep, 1, _stock("am_adam"))
    TtsStage(CFG).run(ep, CFG)
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        cur = reg.card_for(2, "en")
        assert reg.card_for(1, "en").card_id == natsu  # type: ignore[union-attr]
    assert cur is not None and cur.by == "auto" and not cur.locked
    card = CardStore(ep.workspace_root).get(cur.card_id)  # type: ignore[arg-type]
    assert card.voice.backend == "fake" and card.voice.source == "stock"
    assert {c.cue_id: c.card_id for c in _plan(ep).cues}["c0002"] == cur.card_id
    assert any("Lucy" in line for line in _run(ep).report)
    log = (ep.work / "07-tts" / "assignments.log").read_text()
    assert "Lucy" in log and cur.card_id in log
    assert cur.note is not None and "gender estimate" in cur.note
    TtsStage(CFG).run(ep, CFG)  # nothing new to assign: the log only grows when something is
    assert (ep.work / "07-tts" / "assignments.log").read_text() == log


def test_missing_sidecar_fails_before_loading(ep):
    """Review Focus 2: the error names the character, the card and the sidecar, and says
    what to do; no model is loaded."""
    _assign(ep, 2, _stock("af_heart"))
    clone = _assign(ep, 1, CLONE)
    with pytest.raises(CardError) as e:
        TtsStage(CFG).run(ep, CFG)
    msg = str(e.value)
    assert "character 1 (Natsu)" in msg and f"card {clone}" in msg
    assert "sha256:9e1f" in msg and "clips" in msg
    assert "import the pack with clips" in msg and "fukikae voices assign" in msg
    assert FakeSynth.LOG == [] and FakeSynth.CALLS == []


def test_missing_card_file_fails_before_loading(ep):
    natsu, _ = _two_cards(ep)
    CardStore(ep.workspace_root).card_path(natsu).unlink()
    with pytest.raises(CardError, match=f"character 1 \\(Natsu\\).*{natsu}"):
        TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.LOG == []


def test_card_revision_other_than_the_backend_is_refused(ep):
    _assign(ep, 2, _stock("af_heart"))
    old = _assign(ep, 1, _stock("am_adam", revision="0.9"))
    with pytest.raises(CardError, match=f"{old}.*'0.9'.*'1.0'"):
        TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.LOG == []


def test_unknown_stock_voice_is_refused(ep):
    _assign(ep, 2, _stock("af_heart"))
    _assign(ep, 1, _stock("zz_nobody"))
    with pytest.raises(CardError, match="zz_nobody"):
        TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.LOG == []


def test_skipped_and_errored_cues_not_rendered(ep):
    _two_cards(ep)
    TtsStage(CFG).run(ep, CFG)
    by_id = {c.cue_id: c for c in _plan(ep).cues}
    assert by_id["c0005"].skipped == "skip: song" and by_id["c0005"].key is None
    assert by_id["c0006"].skipped == "error: translate: response is not JSON"
    assert not (ep.work / "07-tts" / "c0005.wav").exists()
    assert not (ep.work / "07-tts" / "c0006.wav").exists()


def test_stale_cue_wavs_are_removed(ep):
    _two_cards(ep)
    TtsStage(CFG).run(ep, CFG)
    _write_cues(ep, CUES[:2])
    TtsStage(CFG).run(ep, CFG)
    assert sorted(p.name for p in (ep.work / "07-tts").glob("*.wav")) == ["c0001.wav", "c0002.wav"]


def test_rate_fit_rerenders_once_within_max_speedup(ep):
    _two_cards(ep)
    FakeSynth.DUR = {"natsu one": 4.0}  # twice the 2 s slot
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert [r for t, _, r in FakeSynth.CALLS if t == "natsu one"] == [1.0, 1.25]
    assert c.rate_factor == pytest.approx(1.25) and c.seconds == pytest.approx(3.2, abs=1e-3)
    FakeSynth.CALLS.clear()
    TtsStage(CFG).run(ep, CFG)  # both renders cached
    assert FakeSynth.CALLS == []
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert "c0001" in _run(ep).cached and c.rate_factor == pytest.approx(1.25)
    assert c.rendered_seconds == pytest.approx(4.0, abs=1e-3)


def test_rate_fit_follows_the_timefit_max_speedup(ep):
    _two_cards(ep)
    FakeSynth.DUR = {"natsu one": 2.2}
    cfg = Config(raw={"tts": {"backend": "fake"}, "timefit": {"max_speedup": 1.5}})
    TtsStage(cfg).run(ep, cfg)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.rate_factor == pytest.approx(1.1)


def test_one_backend_loaded_at_a_time(ep):
    _assign(ep, 1, _stock("am_adam"))
    _assign(ep, 2, _stock("af_heart", backend="fake2"))
    TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.LOG == ["load fake", "unload fake", "load fake2", "unload fake2"]
    by_id = {c.cue_id: c for c in _plan(ep).cues}
    assert by_id["c0002"].backend == "fake2" and by_id["c0001"].backend == "fake"


def test_backend_unloaded_when_a_render_fails(ep, monkeypatch):
    _two_cards(ep)

    def boom(*_a: Any, **_kw: Any) -> float:
        raise RuntimeError("synth died")

    monkeypatch.setattr(FakeSynth, "synthesize", boom)
    with pytest.raises(RuntimeError, match="synth died"):
        TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.LOG == ["load fake", "unload fake"]


def test_gain_db_applied_before_caching(ep):
    _assign(ep, 2, _stock("af_heart"))
    _assign(ep, 1, _stock("am_adam") + "\n[voice.post]\ngain_db = 6.0\n")
    TtsStage(CFG).run(ep, CFG)
    from scipy.io import wavfile

    def peak(cue: str) -> float:
        rate, x = wavfile.read(ep.work / "07-tts" / f"{cue}.wav")
        x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
        return float(np.abs(x).max())

    assert peak("c0001") == pytest.approx(peak("c0002") * 10 ** (6 / 20), rel=0.01)


def test_config_key_has_cards_extras_and_model_identity(ep):
    natsu, lucy = _two_cards(ep)
    key = TtsStage(CFG).config_key(CFG, ep)
    assert key["cards"] == {"1": natsu, "2": lucy}
    assert set(key["extras"]) == {"S5", "S9"}
    assert key["identity"] == {"fake": {"model": "fake-model", "revision": "r"}}
    assert key["max_speedup"] == 1.25
    inputs = TtsStage(CFG).inputs(ep)
    assert ep.work / "06-translate" / "cues.en.json" in inputs
    assert ep.work / "01-separate" / "vocals.wav" in inputs
    store = CardStore(ep.workspace_root)
    assert store.card_path(natsu) in inputs and store.card_path(lucy) in inputs


def test_cue_key_hashes_what_is_passed_not_the_delivery_record():
    bc = BackendConfig("kokoro", {})
    base = cue_key("0123456789abcdef", "Hi.", {}, bc)
    assert len(base) == 16
    assert cue_key("0123456789abcdef", "Hi!", {}, bc) != base
    assert cue_key("fedcba9876543210", "Hi.", {}, bc) != base
    assert cue_key("0123456789abcdef", "Hi.", {}, BackendConfig("kokoro", {"x": 1})) != base
    angry = {"emotion_vector": {"angry": 0.9}, "emotion_strength": 0.8}
    assert cue_key("0123456789abcdef", "Hi.", angry, bc) != base
    # Kokoro's keys are what they were: a backend that is passed nothing hashes {} as before
    assert base == tts._hash(
        {"card": "0123456789abcdef", "text": "Hi.", "delivery": {}, "backend": "kokoro",
         "options": {}, "identity": None, "version": tts.RENDER_VERSION}
    )  # fmt: skip


def test_a_dropped_or_zero_strength_emotion_keys_like_nothing_passed(ep, monkeypatch):
    """With emotion strength 0 nothing is passed, so the key is the neutral render's."""
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"emotion_audio"}))
    monkeypatch.setattr(FakeSynth, "voice_keys", frozenset({"rate", "emotion_strength"}))
    _assign(ep, 1, _stock("am_adam") + "emotion_strength = 0.0\n")
    _assign(ep, 2, _stock("af_heart"))
    TtsStage(CFG).run(ep, CFG)
    zero = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset())
    TtsStage(CFG).run(ep, CFG)
    none = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert zero.passed == none.passed == {} and zero.key == none.key


def test_pipeline_runs_tts_after_translate():
    from fukikae.cli import pipeline_stages

    names = [s.name for s in pipeline_stages(Config(raw={"tts": {"backend": "kokoro"}}))]
    assert names[-4:] == ["translate", "tts", "timefit", "mix"]


def test_render_padding_is_trimmed_before_caching_and_the_rate_fit(ep):
    """Ruling CD7: Kokoro pads each render with silence; the fit must measure speech only."""
    _two_cards(ep)
    FakeSynth.PAD = (0.3, 0.45)
    FakeSynth.DUR = {"natsu one": 1.9}  # fits its 2 s slot once the padding is gone
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.rate_factor == 1.0 and len([t for t, _, _ in FakeSynth.CALLS if t == "natsu one"]) == 1
    assert c.seconds == pytest.approx(1.9 + 2 * tts.TRIM_KEEP_S, abs=0.02)


def test_trim_keeps_a_silent_render_whole(tmp_path):
    path = tmp_path / "s.wav"
    tone_wav(path, 0.5, 8000, amp=0.0)
    tts.trim_silence(path)
    assert tts.wav_seconds(path) == pytest.approx(0.5, abs=1e-3)


def test_two_runs_with_identical_audio_leave_plan_byte_identical(ep):
    TtsStage(CFG).run(ep, CFG)  # auto-assigns both characters
    first = (ep.work / "07-tts" / "plan.json").read_bytes()
    TtsStage(CFG).run(ep, CFG)
    assert (ep.work / "07-tts" / "plan.json").read_bytes() == first


def test_a_run_that_only_auto_assigns_is_up_to_date_next_time(ep):
    from fukikae.job import run_stages

    stage = TtsStage(CFG)
    assert run_stages(ep, CFG, [stage]) == ["tts"]
    assert run_stages(ep, CFG, [stage]) == []


def test_characters_with_only_skipped_cues_get_no_card(ep):
    _assign(ep, 1, _stock("am_adam"))
    _write_cues(ep, [CUES[0], CUES[4]])  # Lucy's only cue is a skipped song
    TtsStage(CFG).run(ep, CFG)
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        assert reg.card_for(2, "en") is None


def _kw(text: str) -> dict[str, Any]:
    return next(k for k in FakeSynth.KW if k["text"] == text)


def test_delivery_passed_per_capability(ep, monkeypatch):
    """A backend gets what its capabilities allow: the source line's audio as emotion_audio
    (cut to 07-tts/ref/<cue>.wav), the direction as instruction, never an emotion vector."""
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"emotion_audio", "instruction"}))
    _two_cards(ep)
    cues = [c.model_copy(deep=True) for c in CUES]
    cues[0].delivery.direction = "shouting"
    cues[0].delivery.emotion, cues[0].delivery.emotion_confidence = "angry", 0.9
    _write_cues(ep, cues)
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.passed == {
        "emotion_audio": [3.0, 5.0],
        "instruction": "shouting",
        "emotion_strength": 0.8,
    }
    assert not c.neutral_only
    kw = _kw("natsu one")
    assert kw["emotion_audio_exists"] and kw["emotion_vector"] is None
    assert kw["instruction"] == "shouting" and kw["emotion_strength"] == 0.8
    assert Path(kw["emotion_audio"]) == ep.work / "07-tts" / "ref" / "c0001.wav"
    assert tts.wav_seconds(ep.work / "07-tts" / "ref" / "c0001.wav") == pytest.approx(2.0, abs=0.01)
    # a cue without a direction still gets its emotion audio
    assert set({c.cue_id: c for c in _plan(ep).cues}["c0003"].passed) == {
        "emotion_audio",
        "emotion_strength",
    }


def test_emotion_strength_zero_passes_nothing(ep, monkeypatch):
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"emotion_audio", "emotion_vector"}))
    monkeypatch.setattr(FakeSynth, "voice_keys", frozenset({"rate", "emotion_strength"}))
    _assign(ep, 1, _stock("am_adam") + "emotion_strength = 0.0\n")
    _assign(ep, 2, _stock("af_heart"))
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.passed == {} and c.neutral_only
    kw = _kw("natsu one")
    assert kw["emotion_audio"] is None and kw["emotion_vector"] is None


def test_duration_backend_gets_target_and_skips_rate_fit(ep, monkeypatch):
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"duration"}))
    _two_cards(ep)
    FakeSynth.DUR = {"natsu one": 4.0}  # twice its 2 s slot
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.rate_factor == 1.0 and c.passed == {"target_seconds": 2.0}
    assert len(_calls_for({"natsu one"})) == 1
    assert _kw("natsu one")["target_seconds"] == 2.0


def test_neutral_only_backend_is_reported(ep, monkeypatch, caplog):
    _two_cards(ep)
    with caplog.at_level("INFO", logger="fukikae.stages.tts"):
        TtsStage(CFG).run(ep, CFG)
    rendered = [c for c in _plan(ep).cues if c.skipped is None]
    assert all(c.neutral_only for c in rendered)
    assert "6 neutral-only" in caplog.text

    caplog.clear()
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"emotion_audio"}))
    with caplog.at_level("INFO", logger="fukikae.stages.tts"):
        TtsStage(CFG).run(ep, CFG)
    assert not any(c.neutral_only for c in _plan(ep).cues if c.skipped is None)
    assert "0 neutral-only" in caplog.text


def test_emotion_audio_key_follows_the_source_audio(ep, monkeypatch):
    """The emotion reference is audio: new vocals under the same span render again."""
    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"emotion_audio"}))
    _two_cards(ep)
    TtsStage(CFG).run(ep, CFG)
    first = {c.cue_id: c.key for c in _plan(ep).cues}
    tone_wav(ep.work / "01-separate" / "vocals.wav", 30.0, 16000, amp=0.1)
    TtsStage(CFG).run(ep, CFG)
    second = {c.cue_id: c.key for c in _plan(ep).cues}
    assert first["c0001"] != second["c0001"]


def test_capabilities_for_defaults_to_capabilities():
    assert FakeSynth.capabilities_for("stock") == FakeSynth.capabilities


def test_cue_key_changes_with_model_identity():
    bc = BackendConfig("kokoro", {})

    def key(identity):
        return cue_key("0123456789abcdef", "Hi.", {}, bc, identity=identity)

    base = key({"model": "hexgrad/Kokoro-82M", "revision": "a"})
    assert key({"model": "hexgrad/Kokoro-82M", "revision": "b"}) != base
    assert key({"model": "other/model", "revision": "a"}) != base


# ---- render guards (ruling CD18) --------------------------------------------------------------


def _seeded(ep: Episode, monkeypatch) -> None:
    monkeypatch.setattr(FakeSynth, "voice_keys", frozenset({"rate", "seed"}))
    _assign(ep, 1, _stock("am_adam") + "seed = 7\n")
    _assign(ep, 2, _stock("af_heart"))


def test_a_looping_render_is_retried_once_with_the_next_seed(ep, monkeypatch):
    _seeded(ep, monkeypatch)
    FakeSynth.SEED_DUR = {("natsu one", 7): 25.0}  # a repetition loop; seed 8 is fine
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.retry_seed == 8 and c.suspect is None
    assert c.seconds == pytest.approx(9 * 0.06, abs=0.05)
    assert [sd for t, sd in FakeSynth.SEEDS if t == "natsu one"] == [7, 8]
    # the retry seed is in the cache key: the next run renders nothing and agrees
    first = (ep.work / "07-tts" / "plan.json").read_bytes()
    FakeSynth.SEEDS = []
    TtsStage(CFG).run(ep, CFG)
    assert FakeSynth.SEEDS == []
    assert (ep.work / "07-tts" / "plan.json").read_bytes() == first


def test_a_render_still_wrong_after_the_retry_is_marked_suspect(ep, monkeypatch):
    _seeded(ep, monkeypatch)
    FakeSynth.SEED_DUR = {("natsu one", 7): 0.02, ("natsu one", 8): 0.02}  # far too short
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert c.retry_seed == 8 and c.suspect == "short"
    others = [x for x in _plan(ep).cues if x.cue_id != "c0001" and x.skipped is None]
    assert all(x.suspect is None and x.retry_seed is None for x in others)


def test_a_backend_without_a_seed_is_marked_suspect_without_a_retry(ep):
    _two_cards(ep)
    FakeSynth.DUR = {"lucy one": 30.0}
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0002"]
    assert c.suspect == "long" and c.retry_seed is None
    assert len(_calls_for({"lucy one"})) == 1 + 1  # the first render and its rate fit


def test_render_guard_bounds():
    assert tts.suspect_reason(5.0, 1.0) == "long"  # over 2.5x
    assert tts.suspect_reason(1.9, 0.2) is None  # under the 2 s floor
    assert tts.suspect_reason(0.2, 1.0) == "short"  # under 0.3x
    assert tts.suspect_reason(1.0, 1.0) is None


def test_laughter_or_crying_at_the_source_drops_the_emotion_audio(ep, monkeypatch):
    """A laugh in the emotion prompt comes out as a laugh (E12 c0232): no emotion audio then."""
    from fukikae.models import Tags, TagSpan

    monkeypatch.setattr(FakeSynth, "capabilities", frozenset({"emotion_audio"}))
    _two_cards(ep)
    (ep.work / "04b-tag").mkdir()
    write_json(
        ep.work / "04b-tag" / "tags.json",
        Tags(
            backend="sensevoice",
            events=[TagSpan(start=3.2, end=4.8, label="laughter"),
                    TagSpan(start=6.0, end=6.2, label="crying")],
            emotions=[],
        ),
    )  # fmt: skip
    TtsStage(CFG).run(ep, CFG)
    plan = {c.cue_id: c for c in _plan(ep).cues}
    assert plan["c0001"].passed == {"emotion_dropped": "laughter"}
    assert plan["c0001"].neutral_only
    assert _kw("natsu one")["emotion_audio"] is None
    # a crying blip over a tenth of the span leaves the emotion audio in place
    assert "emotion_audio" in plan["c0002"].passed


# ---- device and options per backend (ruling CD14) ---------------------------------------------


class GpuSynth(FakeSynth2):
    devices: ClassVar = frozenset({"cuda", "cpu"})
    default_device: ClassVar = "cuda"


class CpuSynth(FakeSynth):
    devices: ClassVar = frozenset({"cuda", "cpu"})
    default_device: ClassVar = "cpu"


def test_device_is_per_backend_then_default_then_the_legacy_key(monkeypatch):
    monkeypatch.setattr(tts, "device_string", lambda _cfg: "cuda:0")
    legacy_cpu = Config(raw={"tts": {"backend": "fake", "device": "cpu"}})
    # the backend's own default beats the legacy stage-wide key (the FT copies' trap)
    assert tts.synth_device(legacy_cpu, GpuSynth) == "cuda:0"
    assert tts.synth_device(Config(raw={"tts": {"backend": "fake", "device": "cuda"}}),
                            CpuSynth) == "cpu"  # fmt: skip
    # [tts.<backend>] device beats both
    table = Config(raw={"tts": {"backend": "fake", "fake2": {"device": "cpu"}}})
    assert tts.synth_device(table, GpuSynth) == "cpu"

    # no default declared: the legacy key, else cuda when the backend can use it
    class Plain(FakeSynth):
        devices: ClassVar = frozenset({"cuda", "cpu"})

    assert tts.synth_device(legacy_cpu, Plain) == "cpu"
    assert tts.synth_device(Config(raw={"tts": {"backend": "fake"}}), Plain) == "cuda:0"
    with pytest.raises(ConfigError, match="device"):
        tts.synth_device(Config(raw={"tts": {"backend": "fake", "fake2": {"device": "tpu"}}}),
                         GpuSynth)  # fmt: skip


def test_a_default_device_outranking_the_legacy_key_is_logged(monkeypatch, caplog):
    """A workspace that still says `[tts] device` is told when a backend's own default wins."""
    import logging

    monkeypatch.setattr(tts, "device_string", lambda _cfg: "cuda:0")
    legacy_cuda = Config(raw={"tts": {"backend": "fake", "device": "cuda"}})
    with caplog.at_level(logging.INFO, logger="fukikae.stages.tts"):
        assert tts.synth_device(legacy_cuda, CpuSynth) == "cpu"
    assert any(
        "[tts] device" in r.getMessage() and "cpu" in r.getMessage() and r.levelno >= logging.INFO
        for r in caplog.records
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="fukikae.stages.tts"):
        # agreeing with the default, or set per backend, says nothing
        tts.synth_device(Config(raw={"tts": {"backend": "fake", "device": "cpu"}}), CpuSynth)
        tts.synth_device(
            Config(raw={"tts": {"backend": "fake", "device": "cuda", "fake": {"device": "cpu"}}}),
            CpuSynth,
        )
    assert not [r for r in caplog.records if "[tts] device" in r.getMessage()]


def test_backend_tables_hold_that_backends_options_outside_device_and_the_key():
    cfg = Config(
        raw={"tts": {"backend": "fake", "speed": 1, "fake2": {"device": "cpu", "fp16": True}}}
    )
    bc, _ = tts.stage_settings(cfg)
    assert bc.options == {"speed": 1}  # the backend table is not an option of [tts]'s backend
    assert tts.backend_config(cfg, "fake2").options == {"fp16": True}
    assert tts.backend_config(cfg, "fake").options == {"speed": 1}
    # the stage's key sees options, never the device
    moved = Config(
        raw={"tts": {"backend": "fake", "speed": 1, "fake2": {"device": "cuda", "fp16": True}}}
    )
    assert tts.backend_config(moved, "fake2") == tts.backend_config(cfg, "fake2")


def test_indextts2_refuses_the_cpu_unless_forced(monkeypatch):
    from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer
    from fukikae.backends.tts.kokoro import KokoroSynthesizer
    from fukikae.backends.tts.remote import WorkerSynthesizer

    assert KokoroSynthesizer.default_device == "cpu"
    assert WorkerSynthesizer.default_device == "cuda"
    monkeypatch.setattr(tts, "device_string", lambda _cfg: "cuda:0")
    cpu = Config(raw={"tts": {"backend": "kokoro", "indextts2": {"device": "cpu"}}})
    with pytest.raises(ConfigError, match=r"9 GB.*force_cpu"):
        tts.synth_device(cpu, IndexTts2Synthesizer)
    forced = Config(
        raw={"tts": {"backend": "kokoro", "indextts2": {"device": "cpu", "force_cpu": True}}}
    )
    assert tts.synth_device(forced, IndexTts2Synthesizer) == "cpu"
    assert tts.backend_config(forced, "indextts2").options == {}
    # no GPU at all is the same trap
    monkeypatch.setattr(tts, "device_string", lambda _cfg: "cpu")
    with pytest.raises(ConfigError, match="9 GB"):
        tts.synth_device(Config(raw={"tts": {"backend": "kokoro"}}), IndexTts2Synthesizer)
