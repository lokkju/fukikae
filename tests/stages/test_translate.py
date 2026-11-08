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


def _assign(ep: Episode, character: int, voice: str) -> str:
    cid = CardStore(ep.workspace_root).put(parse_card(CARD.format(voice=voice)))
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
def test_card_change_changes_translate_key(ep):
    stage = TranslateStage()
    _assign(ep, 1, "am_adam")
    a = stage.config_key(CFG, ep)
    assert a["cards"] == {"1": a["cards"]["1"]} and a["identity"]["model"] == "fake-model"
    _assign(ep, 1, "am_michael")
    b = stage.config_key(CFG, ep)
    assert a != b
    # the card files are stage inputs too
    assert any(p.parent.name == "cards" for p in stage.inputs(ep))


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
    monkeypatch.setattr(translate, "gpu_processes", lambda _dev: [(4242, "llama-server")])
    with pytest.raises(RuntimeError, match="4242.*llama-server"):
        TranslateStage().run(ep, CFG)
    assert FakeTranslator.LOG == []  # never loaded


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
