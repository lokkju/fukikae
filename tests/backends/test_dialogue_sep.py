import os
import wave
from pathlib import Path

import numpy as np
import pytest

from fukikae.backends.loader import EXTRAS, names
from fukikae.backends.separation.dialogue import DialogueSeparator
from fukikae.config import Config
from fukikae.stages.separate import SeparateStage
from fukikae.workspace import Episode


def _write(path: Path, data: np.ndarray, rate: int) -> None:
    pcm = (np.clip(data, -1, 1).T * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data.reshape(-1, w.getnchannels()).astype(np.float64) / 32768


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def test_declaration():
    assert (DialogueSeparator.name, DialogueSeparator.kind) == ("dialogue", "separation")
    assert DialogueSeparator.devices == {"cuda", "cpu"}
    assert DialogueSeparator.extra == "dialogue"
    d = DialogueSeparator({}, "cpu")
    assert d.model == "JusperLee/DBS-TIGER-DnR"
    assert d.background_mode == "residual"
    assert DialogueSeparator({"model": "x/y"}, "cpu").model == "x/y"
    assert DialogueSeparator({"background": "stems"}, "cpu").background_mode == "stems"
    with pytest.raises(ValueError):
        DialogueSeparator({"background": "x"}, "cpu")
    assert EXTRAS["dialogue"] == "dialogue"
    assert DialogueSeparator.vram_mb >= 5400  # full-episode measurement, not a clip


def test_registered():
    assert "dialogue" in names("separation")


class _Fake(DialogueSeparator):
    """Stands in for the model: speech is a fixed fraction of the mono mix."""

    def load(self) -> None:
        self._model = object()

    def _run_model(self, mono: np.ndarray, rate: int) -> np.ndarray:  # noqa: ARG002
        return np.stack([mono * 0.1, mono * 0.2, mono * 0.7])  # music, sfx, speech


def _mix(tmp_path: Path, rate: int = 44100) -> tuple[Path, np.ndarray]:
    t = np.arange(rate * 2) / rate
    wav = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)])
    p = tmp_path / "audio.wav"
    _write(p, wav, rate)
    return p, wav


def test_residual_rebuilds_mix(tmp_path):
    audio, wav = _mix(tmp_path)
    d = _Fake({}, "cpu")
    d.load()
    sep = d.separate(audio, tmp_path / "out")
    v, b = _read(sep.vocals), _read(sep.background)
    assert v.shape == b.shape == (wav.shape[1], 2)
    assert np.allclose(v[:, 0], v[:, 1], atol=1e-4)  # speech is mono, copied to both channels
    mono = wav.mean(0)
    assert np.allclose(v[:, 0], 0.7 * mono, atol=1e-3)
    assert np.allclose(v + b, wav.T, atol=1e-3)
    assert sep.vocals_16k == sep.vocals


def test_stems_background(tmp_path):
    audio, wav = _mix(tmp_path)
    d = _Fake({"background": "stems"}, "cpu")
    d.load()
    sep = d.separate(audio, tmp_path / "out")
    b = _read(sep.background)
    assert np.allclose(b[:, 0], 0.3 * wav.mean(0), atol=1e-3)


def test_wrong_rate_and_unloaded(tmp_path):
    audio, _ = _mix(tmp_path, 16000)
    d = _Fake({}, "cpu")
    d.load()
    with pytest.raises(ValueError, match="44100"):
        d.separate(audio, tmp_path / "out")
    with pytest.raises(RuntimeError):
        DialogueSeparator({}, "cpu").separate(audio, tmp_path / "o2")


@pytest.mark.gpu
def test_separates_dialogue_clip(tmp_path):
    media = os.environ.get("FUKIKAE_TEST_MEDIA")
    if not media:
        pytest.skip("FUKIKAE_TEST_MEDIA not set")
    clip = Path(media).expanduser() / "S01E01-clip.mkv"
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    ep = Episode(source=clip, id="e1", work=work, workspace_root=root)
    from fukikae.stages.extract import ExtractStage

    cfg = Config(source_language="ko", raw={"separation": {"backend": "dialogue"}})
    ExtractStage().run(ep, cfg)
    SeparateStage().run(ep, cfg)
    out = work / "01-separate"
    audio = _read(work / "00-extract" / "audio.wav")
    v, b = _read(out / "vocals.wav"), _read(out / "background.wav")
    n = min(len(audio), len(v), len(b))
    resid = audio[:n] - (v[:n] + b[:n])
    assert 20 * np.log10(_rms(resid) / _rms(audio[:n])) < -30
    lo, hi = 60 * 44100, 70 * 44100
    assert _rms(v[lo:hi]) > _rms(b[lo:hi])


def test_real_glue_against_stub_dbs(tmp_path, monkeypatch):
    """load() and _run_model run their real code against a stub ``dbs`` package."""
    import sys
    import types

    torch = pytest.importorskip("torch")
    calls: dict = {}

    class _Model:
        def eval(self):
            return self

        def to(self, device):
            calls["device"] = device
            return self

    class DBSTIGER:
        @staticmethod
        def from_pretrained(name):
            calls["name"] = name
            return _Model()

    def separate(model, mixture, protocol, sample_rate=None):  # noqa: ARG001
        calls.update(shape=tuple(mixture.shape), protocol=protocol, rate=sample_rate)
        # [B, 3, T] in the order music, sfx, speech
        out = torch.stack([mixture * 0.1, mixture * 0.2, mixture * 0.7], dim=1)
        return out

    models = types.SimpleNamespace(DBSTIGER=DBSTIGER)
    engine = types.SimpleNamespace(separate=separate)
    pkg = types.SimpleNamespace(models=models, engine=engine)
    monkeypatch.setitem(sys.modules, "dbs", pkg)
    monkeypatch.setitem(sys.modules, "dbs.models", models)
    monkeypatch.setitem(sys.modules, "dbs.engine", engine)

    audio, wav = _mix(tmp_path)
    d = DialogueSeparator({"chunk_seconds": 4, "hop_seconds": 2}, "cpu")
    d.load()
    assert calls["name"] == "JusperLee/DBS-TIGER-DnR" and calls["device"] == "cpu"
    sep = d.separate(audio, tmp_path / "out")
    assert calls["shape"] == (1, 1, wav.shape[1])
    assert calls["rate"] == 44100
    assert calls["protocol"] == {"mode": "chunked", "chunk_seconds": 4.0, "hop_seconds": 2.0}
    v = _read(sep.vocals)
    assert np.allclose(v[:, 0], 0.7 * wav.mean(0), atol=1e-3)  # speech is the third stem


@pytest.mark.gpu
def test_real_model_stem_order(tmp_path):
    """Real DBS-TIGER: the third stem is speech (energy concentrates in the dialogue)."""
    media = os.environ.get("FUKIKAE_TEST_MEDIA")
    if not media:
        pytest.skip("FUKIKAE_TEST_MEDIA not set")
    from fukikae.stages.extract import ExtractStage

    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    clip = Path(media).expanduser() / "S01E01-clip.mkv"
    ep = Episode(source=clip, id="e1", work=work, workspace_root=root)
    ExtractStage().run(ep, Config(source_language="ko"))
    d = DialogueSeparator({}, "cuda:0")
    d.load()
    try:
        wav = _read(work / "00-extract" / "audio.wav")
        stems = d._run_model(wav.mean(1).astype(np.float32), 44100)
    finally:
        d.unload()
    assert stems.shape[0] == 3
    lo, hi = 60 * 44100, 70 * 44100
    music, sfx, speech = (_rms(s[lo:hi]) for s in stems)
    print(f"rms in 60-70 s: music {music:.4f} sfx {sfx:.4f} speech {speech:.4f}")
    assert speech > music and speech > sfx
