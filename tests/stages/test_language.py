import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from doppio.config import Config
from doppio.stages import language
from doppio.stages.language import LanguageStage
from doppio.workspace import Episode


class FakeLid:
    name: ClassVar[str] = "fake-lid"
    kind: ClassVar[str] = "lid"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 100
    supports_lid: ClassVar[bool] = True
    events: ClassVar[list[str]] = []
    created: ClassVar[list[Any]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:
        FakeLid.created.append((options, device))

    def load(self) -> None:
        FakeLid.events.append("load")

    def unload(self) -> None:
        FakeLid.events.append("unload")

    def detect_language(self, _audio: Path) -> tuple[str, float]:
        FakeLid.events.append("detect")
        return "ko", 0.97


class NoLid(FakeLid):
    supports_lid: ClassVar[bool] = False


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    (work / "01-separate").mkdir(parents=True)
    (work / "01-separate" / "vocals_16k.wav").write_bytes(b"x")
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


@pytest.fixture
def fake(monkeypatch):
    FakeLid.events = []
    FakeLid.created = []
    monkeypatch.setattr(language, "device_string", lambda _cfg: "cpu")
    monkeypatch.setattr(language, "create", lambda _k, bc, dev: FakeLid(bc.options, dev))
    return FakeLid


def _read(ep: Episode) -> dict:
    return json.loads((ep.work / "02-language" / "language.json").read_text())


def test_pinned_language_loads_no_backend(monkeypatch, ep):
    def boom(*_a):
        raise AssertionError("backend created")

    monkeypatch.setattr(language, "create", boom)
    LanguageStage().run(ep, Config(source_language="ja"))
    assert _read(ep) == {"language": "ja", "confidence": None, "source": "config"}


def test_detects_with_default_lid_backend(monkeypatch, fake, ep):
    seen = []

    def create(kind, bc, dev):
        seen.append((kind, bc.backend, bc.options))
        return FakeLid(bc.options, dev)

    monkeypatch.setattr(language, "create", create)
    LanguageStage().run(ep, Config())
    assert _read(ep) == {"language": "ko", "confidence": 0.97, "source": "detected"}
    assert seen == [("lid", "faster_whisper", {"model": "large-v3"})]
    assert fake.events == ["load", "detect", "unload"]


@pytest.mark.usefixtures("fake")
def test_configured_lid_backend_used(monkeypatch, ep):
    seen = []
    monkeypatch.setattr(
        language,
        "create",
        lambda k, bc, d: (seen.append((k, bc.backend, bc.options)), FakeLid({}, d))[1],
    )
    cfg = Config(raw={"lid": {"backend": "x", "model": "small"}})
    LanguageStage().run(ep, cfg)
    assert seen == [("lid", "x", {"model": "small"})]


def test_backend_without_lid_is_an_error(monkeypatch, ep):
    monkeypatch.setattr(language, "device_string", lambda _cfg: "cpu")
    monkeypatch.setattr(language, "create", lambda _k, bc, d: NoLid(bc.options, d))
    with pytest.raises(ValueError, match="language detection"):
        LanguageStage().run(ep, Config())


def test_unload_after_failure(monkeypatch, fake, ep):
    def bad(_self, _a):
        FakeLid.events.append("detect")
        raise RuntimeError("boom")

    monkeypatch.setattr(FakeLid, "detect_language", bad)
    with pytest.raises(RuntimeError):
        LanguageStage().run(ep, Config())
    assert fake.events == ["load", "detect", "unload"]
    assert not (ep.work / "02-language" / "language.json").exists()


def test_paths_and_config_key(ep):
    s = LanguageStage()
    assert s.name == "language"
    assert s.inputs(ep) == [ep.work / "01-separate" / "vocals_16k.wav"]
    assert s.outputs(ep) == [ep.work / "02-language" / "language.json"]
    assert s.config_key(Config(source_language="ko"), ep) != s.config_key(Config(), ep)
    assert s.config_key(Config(), ep) != s.config_key(
        Config(raw={"lid": {"backend": "faster_whisper", "model": "small"}}), ep
    )


@pytest.mark.gpu
def test_gpu_detects_korean_on_clip(clip_ep):
    LanguageStage().run(clip_ep, Config())
    got = json.loads((clip_ep.work / "02-language" / "language.json").read_text())
    assert got["language"] == "ko" and got["source"] == "detected" and got["confidence"] > 0.5
