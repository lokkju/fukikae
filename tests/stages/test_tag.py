from pathlib import Path
from typing import Any, ClassVar

import pytest

from fukikae.artifacts import read_json, write_json
from fukikae.config import Config, load_config
from fukikae.models import Diarization, Tags, TagSpan, Turn
from fukikae.stages import tag
from fukikae.stages.tag import TagStage, tag_enabled
from fukikae.workspace import Episode

LOG: list[Any] = []


class FakeTagger:
    name: ClassVar[str] = "fake-tag"
    kind: ClassVar[str] = "tag"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0

    def __init__(self, options: dict[str, Any], device: str) -> None:
        LOG.append(("init", dict(options), device))

    def load(self) -> None:
        LOG.append("load")

    def unload(self) -> None:
        LOG.append("unload")

    def tag(self, audio: Path, turns: list[Turn]) -> Tags:
        LOG.append(("tag", audio.name, len(turns)))
        return Tags(
            backend="fake",
            events=[TagSpan(start=0.0, end=7.5, label="bgm")],
            emotions=[TagSpan(start=t.start, end=t.end, label="happy") for t in turns],
        )


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    (work / "01-separate").mkdir(parents=True)
    (work / "01-separate" / "vocals_16k.wav").write_bytes(b"x")
    (work / "04-diarize").mkdir()
    write_json(
        work / "04-diarize" / "turns.json",
        Diarization(
            backend="fake",
            turns=[Turn(start=0, end=4, speaker="S0"), Turn(start=5, end=7, speaker="S1")],
        ),
    )
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


@pytest.fixture
def fake(monkeypatch):
    LOG.clear()
    monkeypatch.setattr(tag, "create", lambda _kind, bc, dev: FakeTagger(bc.options, dev))


CFG = Config(raw={"tag": {"backend": "fake", "enabled": True, "language": "ja"}})


@pytest.mark.usefixtures("fake")
def test_tag_stage_writes_tags_json(ep):
    TagStage().run(ep, CFG)
    (out,) = TagStage().outputs(ep)
    assert out == ep.work / "04b-tag" / "tags.json"
    tags = read_json(out, Tags)
    assert tags.backend == "fake"
    assert [s.label for s in tags.emotions] == ["happy", "happy"]
    assert tags.events == [TagSpan(start=0.0, end=7.5, label="bgm")]
    # CPU by default, `enabled` is not a backend option, and the model is unloaded
    assert LOG[0] == ("init", {"language": "ja"}, "cpu")
    assert LOG[-1] == "unload"


def test_stage_reads_vocals_and_turns(ep):
    assert [p.name for p in TagStage().inputs(ep)] == ["vocals_16k.wav", "turns.json"]


def test_config_key_tracks_resolved_model_identity(monkeypatch, ep):
    from fukikae.backends import identity

    keys = []
    for v in ("v1", "v2"):
        monkeypatch.setattr(identity, "model_identity", lambda _k, _c, v=v: (f"tag-{v}", None))
        keys.append(TagStage().config_key(CFG, ep))
    assert keys[0] != keys[1]
    assert keys[0]["identity"] == {"model": "tag-v1", "revision": None}
    assert "enabled" not in keys[0]["options"]


def test_disabled_tag_stage_is_not_in_the_run_list():
    from fukikae.cli import pipeline_stages

    assert "tag" not in [s.name for s in pipeline_stages(Config())]
    on = Config(raw={"tag": {"backend": "sensevoice", "enabled": True}})
    names = [s.name for s in pipeline_stages(on)]
    assert names[-3:] == ["characters", "tag", "translate"]


def test_tag_enabled_defaults_false_and_workspace_overrides_profile():
    assert tag_enabled(Config()) is False
    assert tag_enabled(Config(profile_raw={"tag": {"enabled": True}})) is True
    both = Config(profile_raw={"tag": {"enabled": True}}, raw={"tag": {"enabled": False}})
    assert tag_enabled(both) is False


@pytest.mark.parametrize("profile", ["anime", "live-action"])
def test_shipped_profiles_enable_tag(tmp_path, profile):
    (tmp_path / "fukikae.toml").write_text(f'profile = "{profile}"\n')
    cfg = load_config(tmp_path, user_config=None)
    assert tag_enabled(cfg)
    assert cfg.resolve("tag").backend == "sensevoice"


def test_default_profile_leaves_tag_off(tmp_path):
    (tmp_path / "fukikae.toml").write_text("")
    assert not tag_enabled(load_config(tmp_path, user_config=None))


class FakeEmotion(FakeTagger):
    name: ClassVar[str] = "fake-emotion"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        LOG.append(("init-emotion", dict(options), device))

    def tag(self, audio: Path, turns: list[Turn]) -> Tags:
        LOG.append(("tag-emotion", audio.name, len(turns)))
        return Tags(
            backend="fake-emotion",
            events=[],
            emotions=[
                TagSpan(start=t.start, end=t.end, label="angry", confidence=0.9) for t in turns
            ],
        )


EMO_CFG = Config(
    raw={
        "tag": {
            "backend": "fake",
            "enabled": True,
            "emotion_backend": "fake-emotion",
            "emotion": {"revision": "abc"},
        }
    }
)


@pytest.fixture
def fake2(monkeypatch):
    LOG.clear()

    def create(_kind, bc, dev):
        return (FakeEmotion if bc.backend == "fake-emotion" else FakeTagger)(bc.options, dev)

    monkeypatch.setattr(tag, "create", create)


@pytest.mark.usefixtures("fake2")
def test_emotion_backend_supplies_emotions_and_events_backend_the_events(ep):
    TagStage().run(ep, EMO_CFG)
    (out,) = TagStage().outputs(ep)
    tags = read_json(out, Tags)
    assert tags.backend == "fake"
    assert tags.emotion_backend == "fake-emotion"
    assert tags.events == [TagSpan(start=0.0, end=7.5, label="bgm")]
    assert [(s.label, s.confidence) for s in tags.emotions] == [("angry", 0.9)] * 2
    # the events backend is handed no turns, so it decodes no emotion; options are split
    assert ("init", {}, "cpu") in LOG
    assert ("init-emotion", {"revision": "abc"}, "cpu") in LOG
    assert ("tag", "vocals_16k.wav", 0) in LOG
    assert ("tag-emotion", "vocals_16k.wav", 2) in LOG
    assert LOG.count("unload") == 2


def test_config_key_includes_both_identities(monkeypatch, ep):
    from fukikae.backends import identity

    seen = {}

    def mi(_k, bc):
        return (f"m-{bc.backend}-{seen.get('v')}", None)

    monkeypatch.setattr(identity, "model_identity", mi)
    seen["v"] = 1
    a = TagStage().config_key(EMO_CFG, ep)
    seen["v"] = 2
    b = TagStage().config_key(EMO_CFG, ep)
    assert a["identity"]["model"] == "m-fake-1"
    assert a["emotion_identity"]["model"] == "m-fake-emotion-1"
    assert a != b
    assert a["emotion_backend"] == "fake-emotion" and a["emotion_options"] == {"revision": "abc"}
    assert "emotion_backend" not in a["options"] and "emotion" not in a["options"]


def test_no_emotion_backend_keeps_the_single_backend_key(ep):
    key = TagStage().config_key(CFG, ep)
    assert key["emotion_backend"] is None and key["emotion_identity"] is None


@pytest.mark.parametrize("profile", ["anime", "live-action"])
def test_shipped_profiles_take_emotion_from_emotion2vec(tmp_path, profile):
    (tmp_path / "fukikae.toml").write_text(f'profile = "{profile}"\n')
    cfg = load_config(tmp_path, user_config=None)
    assert cfg.resolve("tag").options["emotion_backend"] == "emotion2vec"
    assert cfg.resolve("tag").backend == "sensevoice"
