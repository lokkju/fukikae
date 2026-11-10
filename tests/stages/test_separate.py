import os
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from fukikae import runctx
from fukikae.config import Config, load_config
from fukikae.gpu import VramTimeout
from fukikae.models import Separation
from fukikae.stages import separate
from fukikae.stages.separate import SeparateStage
from fukikae.workspace import Episode, Workspace


def _write_wav(path: Path, hz: float, rate: int = 44100, secs: float = 1.0, ch: int = 2) -> None:
    t = np.arange(int(rate * secs)) / rate
    mono = (np.sin(2 * np.pi * hz * t) * 8000).astype("<i2")
    data = np.repeat(mono[:, None], ch, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


class FakeSeparator:
    name: ClassVar[str] = "fake-sep"
    kind: ClassVar[str] = "separation"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 6000
    events: ClassVar[list[str]] = []
    fail: ClassVar[bool] = False
    seen: ClassVar[list[tuple[dict[str, Any], str]]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:
        FakeSeparator.seen.append((options, device))

    def load(self) -> None:
        FakeSeparator.events.append("load")

    def unload(self) -> None:
        FakeSeparator.events.append("unload")

    def separate(self, _audio: Path, out_dir: Path) -> Separation:
        FakeSeparator.events.append("separate")
        if FakeSeparator.fail:
            raise RuntimeError("boom")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(out_dir / "vocals.wav", 440)
        _write_wav(out_dir / "background.wav", 220)
        return Separation(
            vocals=out_dir / "vocals.wav",
            background=out_dir / "background.wav",
            vocals_16k=out_dir / "vocals.wav",
        )


@pytest.fixture
def fake(monkeypatch):
    FakeSeparator.events = []
    FakeSeparator.fail = False
    FakeSeparator.seen = []
    monkeypatch.setattr(separate, "create", lambda _kind, _cfg, device: FakeSeparator({}, device))
    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cpu")
    return FakeSeparator


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    (work / "00-extract").mkdir(parents=True)
    _write_wav(work / "00-extract" / "audio.wav", 300)
    return Episode(source=tmp_path / "src.mkv", id="e1", work=work, workspace_root=root)


def _fmt(path: Path) -> tuple[int, int]:
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnchannels()


@pytest.mark.ffmpeg
def test_writes_three_outputs(fake, ep):
    stage = SeparateStage()
    stage.run(ep, Config())
    out = ep.work / "01-separate"
    assert _fmt(out / "vocals.wav") == (44100, 2)
    assert _fmt(out / "background.wav") == (44100, 2)
    assert _fmt(out / "vocals_16k.wav") == (16000, 1)
    assert [p.name for p in stage.outputs(ep)] == ["vocals.wav", "background.wav", "vocals_16k.wav"]
    assert fake.events == ["load", "separate", "unload"]
    assert sorted(p.name for p in out.iterdir()) == [
        "background.wav", "vocals.wav", "vocals_16k.wav"
    ]  # fmt: skip


def test_unload_runs_when_separate_raises(fake, ep):
    fake.fail = True
    with pytest.raises(RuntimeError, match="boom"):
        SeparateStage().run(ep, Config())
    assert fake.events == ["load", "separate", "unload"]
    assert not (ep.work / "01-separate" / "vocals.wav").exists()


def test_default_backend_is_demucs_and_options_pass_through(monkeypatch, ep):
    seen = []
    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cpu")

    def create(kind, cfg, _device):
        seen.append((kind, cfg.backend, cfg.options))
        raise RuntimeError("stop")

    monkeypatch.setattr(separate, "create", create)
    with pytest.raises(RuntimeError):
        SeparateStage().run(ep, Config())
    cfg = Config(raw={"separation": {"backend": "demucs", "model": "htdemucs"}})
    with pytest.raises(RuntimeError):
        SeparateStage().run(ep, cfg)
    assert seen == [
        ("separation", "demucs", {}),
        ("separation", "demucs", {"model": "htdemucs"}),
    ]


def test_config_key_tracks_backend_options(tmp_path):
    ws = Workspace(tmp_path)
    cfg = load_config(ws.root, user_config=None)
    stage = SeparateStage()
    ep = Episode(tmp_path / "s.mkv", "e", tmp_path / "w", tmp_path)
    base = stage.config_key(cfg, ep)
    other = Config(raw={"separation": {"backend": "demucs", "model": "htdemucs"}})
    assert stage.config_key(other, ep) != base
    assert stage.inputs(ep) == [tmp_path / "w" / "00-extract" / "audio.wav"]


def test_no_wait_fails_fast_without_loading(monkeypatch, fake, ep):
    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cuda:0")
    monkeypatch.setattr(separate, "free_mb", lambda _d: 100)
    monkeypatch.setattr(runctx, "vram_timeout_s", 0)
    with pytest.raises(VramTimeout):
        SeparateStage().run(ep, Config())
    assert fake.events == []


def test_waits_then_runs_when_vram_frees(monkeypatch, fake, ep):
    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cuda:0")
    frees = iter([100, 7000])
    monkeypatch.setattr(separate, "free_mb", lambda _d: next(frees))
    monkeypatch.setattr(separate, "_sleep", lambda _s: None)
    monkeypatch.setattr(runctx, "vram_timeout_s", None)
    if not os.environ.get("PATH"):
        pytest.skip("no PATH")
    SeparateStage().run(ep, Config())
    assert fake.events == ["load", "separate", "unload"]
    assert fake.seen[0][1] == "cuda:0"


@pytest.mark.parametrize("mode", ["residual", "stems"])
def test_background_option_reaches_backend(monkeypatch, ep, mode):
    got = []

    def create(_kind, bc, device):
        got.append(bc.options)
        return FakeSeparator(bc.options, device)

    FakeSeparator.events = []
    FakeSeparator.fail = False
    monkeypatch.setattr(separate, "create", create)
    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cpu")
    cfg = Config(raw={"separation": {"backend": "demucs", "background": mode}})
    SeparateStage().run(ep, cfg)
    assert got == [{"background": mode}]
    key = SeparateStage().config_key(cfg, ep)
    assert key["options"] == {"background": mode}


def test_table_without_backend_keeps_options(monkeypatch, ep):
    got = []

    def create(_kind, bc, device):
        got.append((bc.backend, bc.options))
        return FakeSeparator(bc.options, device)

    FakeSeparator.events = []
    FakeSeparator.fail = False
    monkeypatch.setattr(separate, "create", create)
    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cpu")
    cfg = Config(raw={"separation": {"background": "stems"}})
    SeparateStage().run(ep, cfg)
    assert got == [("demucs", {"background": "stems"})]
    assert SeparateStage().config_key(cfg, ep) != SeparateStage().config_key(Config(), ep)


def test_unload_runs_when_load_raises(monkeypatch, fake, ep):
    def bad_load(_self):
        FakeSeparator.events.append("load")
        raise RuntimeError("half loaded")

    monkeypatch.setattr(FakeSeparator, "load", bad_load)
    with pytest.raises(RuntimeError, match="half loaded"):
        SeparateStage().run(ep, Config())
    assert fake.events == ["load", "unload"]


@pytest.mark.usefixtures("fake")
def test_holds_the_gpu_lock_while_the_model_lives(monkeypatch, ep):
    """Ruling CD15: the lock is taken before the VRAM check and let go after unload."""
    from fukikae import gpulock

    monkeypatch.setattr(separate, "device_string", lambda _cfg: "cuda:0")
    order: list[Any] = []
    monkeypatch.setattr(
        separate, "free_mb", lambda _d: order.append(("vram", gpulock.held_fds())) or 99999
    )
    monkeypatch.setattr(
        FakeSeparator, "load", lambda _self: order.append(("load", gpulock.held_fds()))
    )
    SeparateStage().run(ep, Config())
    assert [k for k, _ in order] == ["vram", "load"]
    assert all(fds for _, fds in order)
    assert gpulock.held_fds() == ()
