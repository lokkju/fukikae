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
