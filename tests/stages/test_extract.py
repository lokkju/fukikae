import wave

import pytest

from doppio.config import load_config
from doppio.job import run_stages
from doppio.stages.extract import ExtractStage
from doppio.workspace import Workspace


@pytest.mark.ffmpeg
def test_extract_on_readonly_source_with_spaces(tmp_path, readonly_copy):
    ws = Workspace(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    cfg = load_config(ws.root, user_config=None)
    ep = ws.find_episode(str(readonly_copy))
    assert run_stages(ep, cfg, [ExtractStage()]) == ["extract"]
    with wave.open(str(ep.work / "00-extract" / "audio.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (44100, 2, 2)
        assert w.getnframes() > 4 * 44100
    assert sorted(readonly_copy.parent.iterdir()) == [readonly_copy]
    assert run_stages(ep, cfg, [ExtractStage()]) == []


@pytest.mark.ffmpeg
def test_extract_picks_override_track(tmp_path, readonly_copy):
    ws = Workspace(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    cfg = load_config(ws.root, user_config=None)
    ep = ws.find_episode(str(readonly_copy))
    stage = ExtractStage()
    default_key = stage.config_key(cfg, ep)
    cfg.raw["extract"] = {"audio_track": 1}
    assert stage.config_key(cfg, ep) != default_key
