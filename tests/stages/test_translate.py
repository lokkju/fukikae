from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from fukikae.artifacts import read_json, write_json
from fukikae.cards.card import parse_card
from fukikae.cards.store import CardStore
from fukikae.config import Config, ConfigError
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    DubCues,
    Segment,
    Tags,
    TagSpan,
    Transcript,
    Turn,
    Word,
)
from fukikae.names.inputs import names_path
from fukikae.registry.store import Registry
from fukikae.stages import translate
from fukikae.stages.translate import TranslateStage
from fukikae.translation import (
    PromptTooLargeError,
    Rendering,
    TranslateBackendError,
    TranslateFormatError,
    TranslateRequest,
)
from fukikae.workspace import Episode, Workspace

CARD = """\
target_language = "en"

[voice]
backend = "fake"
model_revision = "1.0"
source = "stock"
voice = "{voice}"

[style]
register = "rough and loud"
catchphrases = [ {{ source = "燃えてきたぞ", target = "I'm all fired up!" }} ]
"""


class FakeTranslator:
    """Answers from SCRIPT (a list of callables taking the request) in order, then echoes."""

    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "translate"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    SCRIPT: ClassVar[list[Any]] = []
    REQUESTS: ClassVar[list[TranslateRequest]] = []
    LOG: ClassVar[list[str]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:  # noqa: ARG002
        self.options = options

    def load(self) -> None:
        self.LOG.append("load")

    def unload(self) -> None:
        self.LOG.append("unload")

    def translate(self, req: TranslateRequest) -> dict[str, Rendering]:
        self.REQUESTS.append(req)
        if self.SCRIPT:
            return self.SCRIPT.pop(0)(req)
        return {c.id: Rendering(literal="lit", text=f"ok {c.id}", direction=None) for c in req.cues}


def garbage(_req: TranslateRequest) -> dict[str, Rendering]:
    raise TranslateFormatError("response is not JSON")


def boom(_req: TranslateRequest) -> dict[str, Rendering]:
    raise RuntimeError("server died")


@pytest.fixture
def fake(monkeypatch):
    FakeTranslator.SCRIPT = []
    FakeTranslator.REQUESTS = []
    FakeTranslator.LOG = []
    monkeypatch.setattr(translate, "create", lambda _k, bc, dev: FakeTranslator(bc.options, dev))
    monkeypatch.setattr(
        translate, "identity_key", lambda _k, _bc: {"model": "fake-model", "revision": None}
    )
    monkeypatch.setattr(translate, "device_string", lambda _cfg: "cpu")


def _words(start: float, text: str) -> list[Word]:
    return [
        Word(start=start + i * 0.3, end=start + i * 0.3 + 0.2, text=t) for i, t in enumerate(text)
    ]


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "e1"
    (work / "03-asr").mkdir(parents=True)
    (work / "05-characters").mkdir()
    segs = [
        Segment(start=0.0, end=1.0, text="燃えてきたぞ", words=_words(0.0, "燃えてきたぞ")),
        Segment(start=3.0, end=4.0, text="はい", words=_words(3.0, "はい")),
    ]
    write_json(
        work / "03-asr" / "transcript.json",
        Transcript(language="ja", backend="fake", word_timing=True, segments=segs),
    )
    with Registry(Workspace(root).registry_path) as reg:
        _, natsu = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
        _, lucy = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
        reg.rename_character(natsu, "Natsu")
    write_json(
        work / "05-characters" / "assignments.json",
        Assignments(
            episode="e1",
            threshold=0.6,
            clusters=[
                ClusterAssignment(
                    cluster="S0", actor_id=natsu, character_id=natsu, score=0.9,
                    clean_seconds=30, status="matched",
                ),
                ClusterAssignment(
                    cluster="S1", actor_id=lucy, character_id=lucy, score=0.9,
                    clean_seconds=30, status="matched",
                ),
            ],
            turns=[Turn(start=0.0, end=2.5, speaker="S0"), Turn(start=2.5, end=5.0, speaker="S1")],
        ),
    )  # fmt: skip
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


def _assign(
    ep: Episode,
    character: int,
    voice: str,
    style: str | None = None,
    catchphrase: str | None = None,
) -> str:
    text = CARD.format(voice=voice)
    if style is not None:
        text = text.replace("rough and loud", style)
    if catchphrase is not None:
        text = text.replace("I'm all fired up!", catchphrase)
    cid = CardStore(ep.workspace_root).put(parse_card(text))
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        reg.assign_card(character, "en", cid, "assign", False)
    return cid


CFG = Config(
    source_language="ja",
    raw={"translate": {"backend": "fake", "max_speed": 1.25, "temperature": 0.2}},
)


def _out(ep: Episode) -> DubCues:
    return read_json(ep.work / "06-translate" / "cues.en.json", DubCues)


@pytest.mark.usefixtures("fake")
def test_translates_every_cue_with_style_and_names(ep):
    _assign(ep, 1, "am_adam")
    TranslateStage().run(ep, CFG)
    out = _out(ep)
    assert out.path == "asr" and out.source_language == "ja" and out.target_language == "en"
    assert [c.text for c in out.cues] == ["ok c0001", "ok c0002"]
    assert [c.speaker for c in out.cues] == ["Natsu", "char2"]
    (req,) = FakeTranslator.REQUESTS
    assert req.styles["Natsu"].register == "rough and loud"
    assert req.styles["Natsu"].catchphrases == [("燃えてきたぞ", "I'm all fired up!")]
    assert "char2" not in req.styles  # no card, no style
    assert FakeTranslator.LOG == ["load", "unload"]


@pytest.mark.usefixtures("fake")
def test_format_error_retries_once_then_marks_cues(ep):
    FakeTranslator.SCRIPT = [garbage, garbage]
    TranslateStage().run(ep, CFG)
    out = _out(ep)
    assert len(FakeTranslator.REQUESTS) == 2
    assert all(c.text is None and c.error == "translate: response is not JSON" for c in out.cues)

    FakeTranslator.REQUESTS.clear()
    FakeTranslator.SCRIPT = [garbage]  # the retry succeeds
    TranslateStage().run(ep, CFG)
    assert [c.error for c in _out(ep).cues] == [None, None]


@pytest.mark.usefixtures("fake")
def test_over_budget_cue_retried_with_tighter_budget(ep):
    def long_first(req):
        return {
            c.id: Rendering(literal=None, text="x" * (c.budget + 10), direction=None)
            for c in req.cues
        }

    def short_c0002(req):
        assert [c.id for c in req.cues] == ["c0001", "c0002"]
        return {
            c.id: Rendering(literal=None, text="y" * c.budget, direction=None) for c in req.cues
        }

    FakeTranslator.SCRIPT = [long_first, short_c0002]
    TranslateStage().run(ep, CFG)
    first, retry = FakeTranslator.REQUESTS
    budgets = {c.id: c.budget for c in first.cues}
    assert {c.id: c.budget for c in retry.cues} == {k: int(v * 0.85) for k, v in budgets.items()}
    out = _out(ep)
    assert all(len(c.text or "") <= c.budget for c in out.cues)
    assert all(c.budget == budgets[c.id] for c in out.cues)  # the recorded budget is the slot's


@pytest.mark.usefixtures("fake")
def test_still_over_budget_keeps_a_rendering(ep):
    def long(req):
        return {c.id: Rendering(literal=None, text="z" * 80, direction=None) for c in req.cues}

    FakeTranslator.SCRIPT = [long, long]
    TranslateStage().run(ep, CFG)
    out = _out(ep).cues
    assert all(c.text == "z" * 80 and c.error is None for c in out)
    assert all(c.over_budget and c.overrun == 80 - c.budget for c in out)


@pytest.mark.usefixtures("fake")
def test_within_budget_cues_are_not_flagged(ep):
    TranslateStage().run(ep, CFG)
    assert all(not c.over_budget and c.overrun == 0 for c in _out(ep).cues)


@pytest.mark.usefixtures("fake")
def test_server_is_unloaded_when_the_backend_raises(ep):
    FakeTranslator.SCRIPT = [boom]
    with pytest.raises(RuntimeError, match="server died"):
        TranslateStage().run(ep, CFG)
    assert FakeTranslator.LOG == ["load", "unload"]


@pytest.mark.usefixtures("fake")
def test_direction_lands_in_delivery_and_song_is_skipped(ep):
    (ep.work / "04b-tag").mkdir()
    write_json(
        ep.work / "04b-tag" / "tags.json",
        Tags(
            backend="fake",
            events=[TagSpan(start=2.9, end=4.5, label="singing")],
            emotions=[TagSpan(start=0.0, end=2.0, label="angry", confidence=0.9)],
        ),
    )

    def directed(req):
        assert [c.id for c in req.cues] == ["c0001"] and req.cues[0].marked
        return {"c0001": Rendering(literal="l", text="Fired up!", direction="shouting")}

    FakeTranslator.SCRIPT = [directed]
    TranslateStage(tags=True).run(ep, CFG)
    first, song = _out(ep).cues
    assert first.delivery.direction == "shouting" and first.delivery.emotion == "angry"
    assert song.skip == "song" and song.text is None and song.error is None


@pytest.mark.usefixtures("fake")
def test_voice_only_card_change_keeps_translate_key(ep):
    """Ruling CD6: translate reads only [style], so the key hashes the style, not the card."""
    stage = TranslateStage()
    _assign(ep, 1, "am_adam")
    a, inputs = stage.config_key(CFG, ep), stage.inputs(ep)
    assert set(a["styles"]) == {"1"} and a["identity"]["model"] == "fake-model"
    assert "cards" not in a
    _assign(ep, 1, "am_michael")  # same style, another voice
    assert stage.config_key(CFG, ep) == a and stage.inputs(ep) == inputs
    assert not any(p.parent.name == "cards" for p in inputs)


@pytest.mark.usefixtures("fake")
def test_style_change_changes_translate_key(ep):
    stage = TranslateStage()
    _assign(ep, 1, "am_adam")
    a = stage.config_key(CFG, ep)
    _assign(ep, 1, "am_adam", style="quiet and formal")
    assert stage.config_key(CFG, ep)["styles"] != a["styles"]
    _assign(ep, 1, "am_adam", catchphrase="Fired up!")
    b = stage.config_key(CFG, ep)
    assert b["styles"] != a["styles"]


@pytest.mark.usefixtures("fake")
def test_budget_follows_the_tts_backends_speech_rate(ep):
    from fukikae.backends.tts.kokoro import KokoroSynthesizer
    from fukikae.cues import budget_chars

    cps = KokoroSynthesizer.speech_cps["en"]
    cfg = Config(
        source_language="ja",
        raw={"translate": {"backend": "fake", "max_speed": 1.25}, "tts": {"backend": "kokoro"}},
    )
    assert TranslateStage().config_key(cfg, ep)["cps"]["extras"] == cps
    assert TranslateStage().config_key(cfg, ep)["cps"]["unassigned"] == cps  # no clone backend
    no_tts = TranslateStage().config_key(CFG, ep)["cps"]
    assert no_tts["extras"] == no_tts["unassigned"] == 15.0  # no [tts]: the language table
    TranslateStage().run(ep, cfg)
    for c in _out(ep).cues:
        assert c.budget == budget_chars(c.end - c.start, "en", 1.25, cps=cps)


@pytest.mark.usefixtures("fake")
def test_style_less_card_leaves_translate_key_alone(ep):
    """The tts stage auto-assigns stock cards, which carry no [style]; translate reads only
    [style], so such a card must not make translate run again."""
    stage = TranslateStage()
    before, inputs = stage.config_key(CFG, ep), stage.inputs(ep)
    plain = CardStore(ep.workspace_root).put(
        parse_card(CARD.format(voice="am_adam").split("[style]")[0])
    )
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        reg.assign_card(2, "en", plain, "auto", False)
    assert stage.config_key(CFG, ep) == before and stage.inputs(ep) == inputs


@pytest.mark.usefixtures("fake")
def test_stage_settings_stay_out_of_backend_options(ep):
    cfg = Config(
        source_language="ja",
        raw={
            "translate": {
                "backend": "fake",
                "max_speed": 1.1,
                "source": "asr",
                "subs_condense": True,
            }
        },
    )
    key = TranslateStage().config_key(cfg, ep)
    assert key["options"] == {} and key["max_speed"] == 1.1 and key["source"] == "asr"
    TranslateStage().run(ep, cfg)
    assert "max_speed" not in FakeTranslator.REQUESTS[0].model_dump()


def test_subs_source_without_subtitles_is_refused(ep):
    cfg = Config(source_language="ja", raw={"translate": {"backend": "fake", "source": "subs"}})
    with pytest.raises(ConfigError, match="no en subtitles"):
        TranslateStage().run(ep, cfg)


# ---- subtitle-first path ----

SRT = """\
1
00:00:00,000 --> 00:00:01,000
NATSU: I'm all fired up!

2
00:00:01,000 --> 00:00:02,000
This line is far too long to say in a single second.

3
00:00:03,000 --> 00:00:04,000
[door slams]

4
00:00:03,000 --> 00:00:04,000
Yes.

5
00:00:04,000 --> 00:00:05,000
And this one is also much too long for its slot.
"""


def _sidecar(ep: Episode) -> Path:
    path = ep.source.with_name(f"{ep.source.stem}.en.srt")
    path.write_text(SRT, encoding="utf-8")
    return path


@pytest.mark.usefixtures("fake")
def test_auto_prefers_subtitles_when_present(ep):
    sub = _sidecar(ep)
    stage = TranslateStage()
    assert sub in stage.inputs(ep)
    stage.run(ep, CFG)
    out = _out(ep)
    assert out.path == "subs" and out.source_language == "ja"
    assert [c.skip for c in out.cues] == [None, None, "sdh", None, None]
    assert out.cues[0].text == out.cues[0].source_text == "I'm all fired up!"
    assert out.cues[0].speaker == "Natsu" and out.cues[3].speaker == "char2"
    assert out.cues[3].text == "Yes."
    key = stage.config_key(CFG, ep)
    assert key["source"] == "auto" and key["path"] == "subs" and key["subs_track"] is None
    asr = Config(source_language="ja", raw={"translate": {"backend": "fake", "source": "asr"}})
    assert "path" not in stage.config_key(asr, ep)  # the ASR path's key is unchanged


@pytest.mark.usefixtures("fake")
def test_only_over_budget_cues_are_condensed(ep):
    _sidecar(ep)
    _assign(ep, 1, "am_adam")
    TranslateStage().run(ep, CFG)
    (req,) = FakeTranslator.REQUESTS
    assert req.mode == "condense" and req.source_language == "en" == req.target_language
    assert [c.id for c in req.cues] == ["c0002", "c0005"]
    assert [c.text for c in req.cues][0].startswith("This line is far too long")
    assert req.glossary == {}
    out = _out(ep)
    assert [c.text for c in out.cues] == [
        "I'm all fired up!", "ok c0002", None, "Yes.", "ok c0005",
    ]  # fmt: skip
    assert out.cues[1].source_text.startswith("This line")  # the subtitle stays the source
    assert FakeTranslator.LOG == ["load", "unload"]


@pytest.mark.usefixtures("fake")
def test_subs_condense_off_keeps_verbatim(ep):
    _sidecar(ep)
    cfg = Config(
        source_language="ja",
        raw={"translate": {"backend": "fake", "source": "subs", "subs_condense": False}},
    )
    TranslateStage().run(ep, cfg)
    assert FakeTranslator.REQUESTS == [] and FakeTranslator.LOG == []  # no model loaded
    out = _out(ep)
    assert out.cues[1].text == out.cues[1].source_text and out.cues[1].over_budget
    assert out.cues[1].overrun == len(out.cues[1].source_text) - out.cues[1].budget


@pytest.mark.usefixtures("fake")
def test_failed_condense_keeps_the_subtitle_line(ep):
    _sidecar(ep)
    FakeTranslator.SCRIPT = [garbage, garbage]
    TranslateStage().run(ep, CFG)
    out = _out(ep)
    assert out.cues[1].text == out.cues[1].source_text and out.cues[1].error is None
    assert out.cues[1].over_budget
    assert out.cues[1].note is not None and out.cues[1].note.startswith("condense failed: ")
    assert out.cues[0].note is None  # spoken as written, never condensed


@pytest.mark.usefixtures("fake")
def test_condensed_line_longer_than_the_subtitle_is_dropped(ep):
    _sidecar(ep)

    def longer(req):
        return {
            c.id: Rendering(literal=None, text=c.text + " and more", direction=None)
            for c in req.cues
        }

    FakeTranslator.SCRIPT = [longer, longer]
    TranslateStage().run(ep, CFG)
    out = _out(ep)
    assert out.cues[1].text == out.cues[1].source_text
    assert out.cues[1].note == "condense not shorter"


@pytest.mark.usefixtures("fake")
def test_subs_filter_is_configurable(ep):
    _sidecar(ep)
    cfg = Config(
        source_language="ja",
        raw={"translate": {"backend": "fake", "source": "subs", "subs_filter": ["song"]}},
    )
    TranslateStage().run(ep, cfg)
    assert _out(ep).cues[2].skip == "empty"  # nothing left to say once the note is dropped
    bad = Config(
        source_language="ja",
        raw={"translate": {"backend": "fake", "source": "subs", "subs_filter": ["lyrics"]}},
    )
    with pytest.raises(ConfigError, match="subs_filter"):
        TranslateStage().run(ep, bad)


@pytest.mark.usefixtures("fake")
def test_configured_subtitle_file_wins_and_joins_the_key(ep, tmp_path):
    _sidecar(ep)
    other = tmp_path / "e1-other.srt"
    other.write_text("1\n00:00:00,000 --> 00:00:01,000\nOther.\n", encoding="utf-8")
    cfg = Config(
        source_language="ja",
        raw={
            "translate": {"backend": "fake"},
            "subs": {"file": str(tmp_path / "{episode}-other.srt")},
        },
    )
    stage = TranslateStage(cfg=cfg)
    assert other in stage.inputs(ep)
    assert stage.config_key(cfg, ep)["subs_file"] == str(other)
    stage.run(ep, cfg)
    assert [c.text for c in _out(ep).cues] == ["Other."]


def test_inputs_without_assignments_do_not_crash(tmp_path: Path):
    ep = Episode(source=tmp_path / "s.mkv", id="e1", work=tmp_path / "w", workspace_root=tmp_path)
    paths = TranslateStage(target_language="en", tags=False).inputs(ep)
    assert paths[0].name == "transcript.json"


def _three_windows(ep: Episode) -> None:
    """Rewrite the transcript so the cues fall in three windows (gaps over 2 s)."""
    segs = [
        Segment(start=0.0, end=1.0, text="妖精の尻尾だ", words=_words(0.0, "妖精の尻尾だ")),
        Segment(start=3.0, end=4.0, text="はい", words=_words(3.0, "はい")),
        Segment(start=8.0, end=9.0, text="いいえ", words=_words(8.0, "いいえ")),
        Segment(start=14.0, end=15.0, text="そうか", words=_words(14.0, "そうか")),
    ]
    write_json(
        ep.work / "03-asr" / "transcript.json",
        Transcript(language="ja", backend="fake", word_timing=True, segments=segs),
    )
    asg = read_json(ep.work / "05-characters" / "assignments.json", Assignments)
    asg.turns = [Turn(start=0.0, end=2.5, speaker="S0"), Turn(start=2.5, end=20.0, speaker="S1")]
    write_json(ep.work / "05-characters" / "assignments.json", asg)


def backend_down(_req: TranslateRequest) -> dict[str, Rendering]:
    raise TranslateBackendError("http://127.0.0.1:1/v1: HTTP 500 from chat/completions")


@pytest.mark.usefixtures("fake")
def test_previous_window_and_glossary_filter(ep):
    _three_windows(ep)
    (ep.workspace_root / "glossary.toml").write_text(
        '[terms]\n"妖精の尻尾" = "Fairy Tail"\n"魔導士" = "wizard"\n', encoding="utf-8"
    )
    TranslateStage().run(ep, CFG)
    first, second, third = FakeTranslator.REQUESTS
    assert first.glossary == {"妖精の尻尾": "Fairy Tail"}  # only terms the window uses
    assert second.glossary == {} and first.previous == []
    assert [c.id for c in first.cues] == ["c0001", "c0002"]
    assert second.previous == [("妖精の尻尾だ", "ok c0001"), ("はい", "ok c0002")]
    assert third.previous == [("いいえ", "ok c0003")]


@pytest.mark.usefixtures("fake")
def test_backend_error_retries_then_marks_the_window_and_goes_on(ep):
    _three_windows(ep)
    FakeTranslator.SCRIPT = [backend_down, backend_down]  # window 1 fails twice
    TranslateStage().run(ep, CFG)
    out = {c.id: c for c in _out(ep).cues}
    assert out["c0001"].text is None
    assert (out["c0001"].error or "").startswith("translate: backend:")
    assert out["c0002"].error is not None
    assert out["c0003"].text == "ok c0003" and out["c0004"].text == "ok c0004"
    assert third_previous_is_clean(FakeTranslator.REQUESTS)

    FakeTranslator.REQUESTS.clear()
    FakeTranslator.SCRIPT = [backend_down]  # one failure, the retry works
    TranslateStage().run(ep, CFG)
    assert all(c.error is None for c in _out(ep).cues)


def third_previous_is_clean(requests: list[TranslateRequest]) -> bool:
    # after a failed window the next one gets no previous lines
    return requests[2].previous == []


@pytest.mark.usefixtures("fake")
def test_every_window_failing_on_the_backend_aborts(ep):
    _three_windows(ep)
    FakeTranslator.SCRIPT = [backend_down] * 6
    with pytest.raises(TranslateBackendError, match="3 windows in a row"):
        TranslateStage().run(ep, CFG)
    assert FakeTranslator.LOG == ["load", "unload"]


@pytest.mark.usefixtures("fake")
def test_too_large_prompt_drops_context_then_splits(ep):
    _three_windows(ep)

    def picky(req):
        if req.previous or len(req.cues) > 1:
            raise PromptTooLargeError("prompt of about 9000 tokens exceeds 4096")
        c = req.cues[0]
        return {c.id: Rendering(literal=None, text=f"ok {c.id}", direction=None)}

    FakeTranslator.SCRIPT = [picky] * 20
    TranslateStage().run(ep, CFG)
    assert all(c.text == f"ok {c.id}" for c in _out(ep).cues)
    sent = [r for r in FakeTranslator.REQUESTS if len(r.cues) == 1 and not r.previous]
    assert sorted(r.cues[0].id for r in sent) == ["c0001", "c0002", "c0003", "c0004"]


@pytest.mark.usefixtures("fake")
def test_one_cue_too_large_is_marked(ep):
    def never(_req):
        raise PromptTooLargeError("prompt too large")

    FakeTranslator.SCRIPT = [never] * 10
    TranslateStage().run(ep, CFG)
    assert all(c.error == "translate: prompt too large" for c in _out(ep).cues)


@pytest.mark.usefixtures("fake")
def test_empty_catchphrase_targets_are_not_sent(ep):
    card = CARD.replace('target = "I\'m all fired up!"', 'target = ""').format(voice="x")
    cid = CardStore(ep.workspace_root).put(parse_card(card))
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        reg.assign_card(1, "en", cid, "assign", False)
    TranslateStage().run(ep, CFG)
    assert FakeTranslator.REQUESTS[0].styles["Natsu"].catchphrases == []


@pytest.mark.usefixtures("fake")
def test_leftover_llama_server_on_the_gpu_is_named(ep, monkeypatch):
    monkeypatch.setattr(translate, "device_string", lambda _cfg: "cuda:0")
    from fukikae import gpulock

    leftover = [(4242, "llama-server -m m.gguf --api-key-file /tmp/fukikae-llama-key-x")]
    monkeypatch.setattr(gpulock, "leftover_jobs", lambda _index: leftover)
    with pytest.raises(gpulock.GpuBusy, match="4242.*llama-server"):
        TranslateStage().run(ep, CFG)
    assert FakeTranslator.LOG == []  # never loaded
    assert gpulock.held_fds() == ()


@pytest.mark.usefixtures("fake")
def test_a_managed_server_holds_the_gpu_lock(ep, monkeypatch):
    """Ruling CD15: the lock is held from before the VRAM check until the server unloads."""
    from fukikae import gpulock

    monkeypatch.setattr(translate, "device_string", lambda _cfg: "cuda:0")
    monkeypatch.setattr(translate, "free_mb", lambda _d: 99999)
    held: list[tuple[int, ...]] = []
    real_load = FakeTranslator.load
    monkeypatch.setattr(
        FakeTranslator, "load", lambda self: (held.append(gpulock.held_fds()), real_load(self))
    )
    TranslateStage().run(ep, CFG)
    assert held and held[0]
    assert gpulock.held_fds() == ()


FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fake_llama_server.py"


@pytest.mark.parametrize("failure", ["400", "500", "sleep"])
def test_real_backend_errors_mark_the_window(ep, tmp_path, monkeypatch, failure):
    import sys

    from fukikae.backends.translate.llama_server import LlamaServer
    from fukikae.backends.translate.openai_compat import OpenAICompatTranslator

    wrapper = tmp_path / "llama-server"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FIXTURE}" "$@"\n')
    wrapper.chmod(0o755)
    plan = tmp_path / "plan.txt"
    plan.write_text(f"{failure}\n{failure}\n")
    monkeypatch.setenv("FAKE_LLAMA_FAIL", str(plan))
    monkeypatch.setattr(translate, "identity_key", lambda _k, _bc: {"model": "m", "revision": None})
    monkeypatch.setattr(translate, "device_string", lambda _cfg: "cpu")
    with LlamaServer(str(wrapper), tmp_path / "m.gguf", 0) as server:
        opts = {"mode": "external", "base_url": server.base_url + "/v1", "timeout_s": 1}
        monkeypatch.setattr(
            translate, "create", lambda _k, _bc, dev: OpenAICompatTranslator(dict(opts), dev)
        )
        cfg = Config(source_language="ja", raw={"translate": {"backend": "openai_compat"}})
        TranslateStage().run(ep, cfg)
    out = _out(ep).cues
    assert all(c.text is None and (c.error or "").startswith("translate: backend:") for c in out)


@pytest.mark.usefixtures("fake")
def test_a_rename_reaches_the_translate_key(ep):
    """C3: names are baked into each cue's speaker and the prompt, so they are keyed."""
    stage = TranslateStage()
    a = stage.config_key(CFG, ep)
    assert a["names"] == {"1": "Natsu"}
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        reg.rename_character(1, "Natsu Dragneel")
    assert stage.config_key(CFG, ep)["names"] == {"1": "Natsu Dragneel"}


def _backend_card(ep: Episode, character: int, backend: str) -> None:
    text = CARD.format(voice="v").replace('backend = "fake"', f'backend = "{backend}"')
    cid = CardStore(ep.workspace_root).put(parse_card(text))
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        reg.assign_card(character, "en", cid, "assign", False)


@pytest.mark.usefixtures("fake")
def test_budget_per_cue_follows_the_speakers_voice(ep):
    """C4: a cue is budgeted for the backend that will speak it: the character's card, else
    [tts] clone_backend (a recurring character gets a clone by default), else the stage
    backend; extras always the stage backend."""
    from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer
    from fukikae.backends.tts.kokoro import KokoroSynthesizer
    from fukikae.backends.tts.qwen3_tts import Qwen3TtsSynthesizer
    from fukikae.cues import budget_chars

    cfg = Config(
        source_language="ja",
        raw={
            "translate": {"backend": "fake", "max_speed": 1.25},
            "tts": {"backend": "kokoro", "clone_backend": "indextts2"},
        },
    )
    _backend_card(ep, 1, "qwen3_tts")
    key = TranslateStage().config_key(cfg, ep)
    qwen, index = Qwen3TtsSynthesizer.speech_cps["en"], IndexTts2Synthesizer.speech_cps["en"]
    assert key["cps"] == {
        "extras": KokoroSynthesizer.speech_cps["en"],
        "unassigned": index,
        "characters": {"1": qwen},
    }
    TranslateStage().run(ep, cfg)
    by_char = {c.character_id: c for c in _out(ep).cues}
    for cid, cps in ((1, qwen), (2, index)):
        c = by_char[cid]
        assert c.budget == budget_chars(c.end - c.start, "en", 1.25, cps=cps)
    _backend_card(ep, 1, "kokoro")
    assert TranslateStage().config_key(cfg, ep)["cps"] != key["cps"]


@pytest.mark.usefixtures("fake")
def test_subtitle_mode_keeps_the_lines_when_the_backend_gives_up(ep, monkeypatch):
    """C5: in subtitle mode a dead endpoint costs only the condensing: every line is still
    spoken as written, and the episode goes on."""
    from fukikae.stages import translate

    monkeypatch.setattr(translate, "MAX_BACKEND_FAILURES", 1)
    _sidecar(ep)
    FakeTranslator.SCRIPT = [backend_down] * 8
    TranslateStage().run(ep, CFG)
    out = _out(ep)
    for c in (out.cues[1], out.cues[4]):
        assert c.text == c.source_text and c.error is None
        assert c.note is not None and c.note.startswith("condense failed: ")
    assert FakeTranslator.LOG == ["load", "unload"]


# ---- names (Task 8): a cue's own Name picks its speaker on the subtitle path ----

ASS_HEAD = """\
[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour
Style: Default,Arial,24,&H00FFFFFF

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_sidecar(ep: Episode) -> Path:
    rows = [
        ("0:00:00.00", "0:00:01.00", "NATSU", "I'm all fired up!"),
        ("0:00:03.00", "0:00:04.00", "NATSU", "Yes."),  # under Lucy's turn
        ("0:00:04.00", "0:00:05.00", "LUCY", "Fine."),
    ]
    body = "".join(f"Dialogue: 0,{a},{b},Default,{n},0,0,0,,{t}\n" for a, b, n, t in rows)
    path = ep.source.with_name(f"{ep.source.stem}.en.ass")
    path.write_text(ASS_HEAD + body, encoding="utf-8")
    return path


def _names_used(ep: Episode, how: str = "target", hidden: list[int] | None = None) -> None:
    """names.json (used) for the sidecar's three lines, and assignments in which the characters
    stage gave the sub-cluster S0~natsu to Natsu (character 1) by name."""
    from fukikae.names.models import LabelInfo, NamedLine, NamesResult
    from fukikae.names.normalize import NameClass

    lines = [
        NamedLine(id=i, start=s, end=e, raw=raw, key=raw.lower(), display=raw.title(),
                  cls=NameClass.NAMED, inner=False, text=t)
        for i, (s, e, raw, t) in enumerate(
            [(0.0, 1.0, "NATSU", "I'm all fired up!"), (3.0, 4.0, "NATSU", "Yes."),
             (4.0, 5.0, "LUCY", "Fine.")]
        )
    ]  # fmt: skip
    names_path(ep.work).parent.mkdir(parents=True, exist_ok=True)
    write_json(
        names_path(ep.work),
        NamesResult(
            episode=ep.id, used=True, verdict="full", reason=None,
            source={"how": how, "language": "en", "track": None, "codec": "ass", "title": None,
                    "file": None},
            offset_s=0.0, coverage={}, lines=lines, hidden=hidden or [], pieces=[],
            labels={"S0~natsu": LabelInfo(parent="S0", key="natsu", display="Natsu", cls="named")},
            unvoiced=[], excluded=[], dropped_row_seconds=0.0, suggestions=[],
        ),
    )  # fmt: skip
    path = ep.work / "05-characters" / "assignments.json"
    asg = read_json(path, Assignments)
    natsu = ClusterAssignment.model_validate(
        {"cluster": "S0~natsu", "actor_id": 1, "character_id": 1, "score": 0.9,
         "clean_seconds": 1.0, "status": "matched", "name": "natsu", "source": "name"}
    )  # fmt: skip
    turns = [Turn(start=0.0, end=1.0, speaker="S0~natsu"), *asg.turns[1:]]  # type: ignore[index]
    turns.insert(1, Turn(start=1.0, end=2.5, speaker="S0"))
    write_json(path, asg.model_copy(update={"clusters": [natsu, *asg.clusters], "turns": turns}))


SUBS = Config(
    source_language="ja",
    raw={"translate": {"backend": "fake", "max_speed": 1.25, "subs_condense": False}},
)


@pytest.mark.usefixtures("fake")
def test_a_cue_named_by_its_own_name_goes_to_that_character(ep):
    _ass_sidecar(ep)
    _names_used(ep)
    TranslateStage().run(ep, SUBS)
    assert [(c.text, c.speaker) for c in _out(ep).cues] == [
        ("I'm all fired up!", "Natsu"), ("Yes.", "Natsu"), ("Fine.", "char2"),
    ]  # fmt: skip


@pytest.mark.usefixtures("fake")
def test_hidden_lines_and_other_tracks_go_by_overlap(ep):
    _ass_sidecar(ep)
    _names_used(ep, hidden=[1])
    TranslateStage().run(ep, SUBS)
    assert [c.speaker for c in _out(ep).cues] == ["Natsu", "char2", "char2"]
    _names_used(ep, how="other-language")
    TranslateStage().run(ep, SUBS)
    assert [c.speaker for c in _out(ep).cues] == ["Natsu", "char2", "char2"]
    assert "cue_names" not in TranslateStage().config_key(SUBS, ep)


def test_translate_key_changes_when_names_json_changes(ep):
    _ass_sidecar(ep)
    stage = TranslateStage()
    plain = stage.config_key(SUBS, ep)
    assert names_path(ep.work) not in stage.inputs(ep)
    _names_used(ep)
    assert names_path(ep.work) in stage.inputs(ep)
    named = stage.config_key(SUBS, ep)
    assert named["cue_names"] == {"map": {"natsu": [1, None]}, "hidden": [], "offset_s": 0.0}
    assert {k: v for k, v in named.items() if k != "cue_names"} == plain
    _names_used(ep, hidden=[1])
    assert stage.config_key(SUBS, ep)["cue_names"]["hidden"] == [[3.0, 4.0]]
