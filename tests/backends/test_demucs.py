import os
import wave
from pathlib import Path

import numpy as np
import pytest

from fukikae.backends.loader import EXTRAS, names
from fukikae.backends.separation.demucs import DemucsSeparator
from fukikae.config import Config
from fukikae.stages.separate import SeparateStage
from fukikae.workspace import Episode


def test_declaration():
    assert (DemucsSeparator.name, DemucsSeparator.kind) == ("demucs", "separation")
    assert DemucsSeparator.devices == {"cuda", "cpu"}
    assert DemucsSeparator.vram_mb == 6000
    assert DemucsSeparator({}, "cpu").model == "htdemucs_ft"
    assert DemucsSeparator({"model": "htdemucs"}, "cpu").model == "htdemucs"
    assert DemucsSeparator({}, "cpu").background_mode == "residual"
    assert DemucsSeparator({"background": "stems"}, "cpu").background_mode == "stems"
    with pytest.raises(ValueError):
        DemucsSeparator({"background": "x"}, "cpu")
    assert EXTRAS["demucs"] == "demucs"


def test_registered():
    assert "demucs" in names("separation")


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data.reshape(-1, w.getnchannels()).astype(np.float64) / 32768


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


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

    cfg = Config(source_language="ko")
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
