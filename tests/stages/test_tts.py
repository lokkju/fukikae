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
from fukikae.config import BackendConfig, Config
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    DubCues,
    TtsPlan,
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


def tone_wav(path: Path, seconds: float, rate: int, amp: float = 0.25) -> float:
    n = max(1, round(seconds * rate))
    x = amp * np.sin(2 * np.pi * 440 * np.arange(n) / rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return n / rate


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
    DUR: ClassVar[dict[str, float]] = {}

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

    def synthesize(self, text: str, voice: Voice, language: str, out: Path, **_kw: Any) -> float:  # noqa: ARG002
        self.CALLS.append((text, voice.voice, voice.rate))
        seconds = self.DUR.get(text, len(text) * 0.06) / voice.rate
        return tone_wav(out, seconds, self.sample_rate)


class FakeSynth2(FakeSynth):
    name: ClassVar[str] = "fake2"


CLASSES = {"fake": FakeSynth, "fake2": FakeSynth2}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeSynth.LOG = []
    FakeSynth.CALLS = []
    FakeSynth.DUR = {}
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
        assert c.slot == pytest.approx(2.0) and c.rate_factor == 1.0 and not c.cached
        assert c.neutral_only and c.passed == {}
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
    assert all(c.cached for c in _plan(ep).cues if c.skipped is None)


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
    assert by_id["c0001"].card_id == new and not by_id["c0001"].cached
    assert by_id["c0002"].cached


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
    assert any("Lucy" in line for line in _plan(ep).report)


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
    assert c.cached and c.rate_factor == pytest.approx(1.25)


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


def test_cue_key_hashes_only_the_delivery_the_backend_receives():
    bc = BackendConfig("kokoro", {})
    plain = Delivery(reference=(1.0, 2.0))
    angry = Delivery(reference=(1.0, 2.0), emotion="angry", emotion_confidence=0.9)
    base = cue_key("0123456789abcdef", "Hi.", plain, bc, capabilities=frozenset())
    assert len(base) == 16
    assert cue_key("0123456789abcdef", "Hi.", angry, bc, capabilities=frozenset()) == base
    assert cue_key("0123456789abcdef", "Hi!", plain, bc, capabilities=frozenset()) != base
    assert cue_key("fedcba9876543210", "Hi.", plain, bc, capabilities=frozenset()) != base
    assert cue_key("0123456789abcdef", "Hi.", plain, BackendConfig("kokoro", {"x": 1})) != base
    caps = frozenset({"emotion_vector"})
    assert cue_key("0123456789abcdef", "Hi.", angry, bc, capabilities=caps) != cue_key(
        "0123456789abcdef", "Hi.", plain, bc, capabilities=caps
    )
    # without capabilities given, the whole delivery record counts
    assert cue_key("0123456789abcdef", "Hi.", angry, bc) != cue_key(
        "0123456789abcdef", "Hi.", plain, bc
    )


def test_pipeline_runs_tts_after_translate():
    from fukikae.cli import pipeline_stages

    names = [s.name for s in pipeline_stages(Config(raw={"tts": {"backend": "kokoro"}}))]
    assert names[-2:] == ["translate", "tts"]
