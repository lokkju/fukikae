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
from fukikae.translation import Rendering, TranslateFormatError, TranslateRequest
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
    assert all(c.text == "z" * 80 and c.error is None for c in _out(ep).cues)


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


def test_subs_source_is_refused_until_built(ep):
    cfg = Config(source_language="ja", raw={"translate": {"backend": "fake", "source": "subs"}})
    with pytest.raises(ConfigError, match="subs"):
        TranslateStage().run(ep, cfg)


def test_inputs_without_assignments_do_not_crash(tmp_path: Path):
    ep = Episode(source=tmp_path / "s.mkv", id="e1", work=tmp_path / "w", workspace_root=tmp_path)
    paths = TranslateStage(target_language="en", tags=False).inputs(ep)
    assert paths[0].name == "transcript.json"
