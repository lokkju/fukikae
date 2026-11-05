import pytest

from fukikae.backends import loader
from fukikae.backends.asr.faster_whisper import FasterWhisperTranscriber
from fukikae.models import Transcript


def test_declaration():
    c = FasterWhisperTranscriber
    assert (c.name, c.kind, c.languages, c.supports_lid, c.vram_mb) == (
        "faster_whisper", "asr", "*", True, 5000
    )  # fmt: skip
    assert c.devices == frozenset({"cuda", "cpu"})


def test_registered_for_asr_and_lid():
    assert "faster_whisper" in loader.names("asr")
    assert "faster_whisper" in loader.names("lid")
    assert loader.EXTRAS["faster_whisper"] == "whisper"
    assert loader.EXTRAS["wav2vec2"] == "whisper"


def test_option_defaults_follow_device():
    cuda = FasterWhisperTranscriber({}, "cuda:0")
    cpu = FasterWhisperTranscriber({}, "cpu")
    assert (cuda.model_name, cuda.compute_type, cuda.vad_filter) == ("large-v3", "float16", True)
    assert cpu.compute_type == "int8"
    assert FasterWhisperTranscriber({"compute_type": "int8_float16"}, "cuda:1").compute_type == (
        "int8_float16"
    )
    assert cuda.model == "large-v3"


def test_transcribe_requires_load(tmp_path):
    with pytest.raises(RuntimeError, match="load"):
        FasterWhisperTranscriber({}, "cpu").transcribe(tmp_path / "a.wav", "ko")


@pytest.mark.gpu
def test_detects_and_transcribes_korean(clip_ep):
    audio = clip_ep.work / "01-separate" / "vocals_16k.wav"
    b = FasterWhisperTranscriber({}, "cuda:0")
    b.load()
    try:
        lang, conf = b.detect_language(audio)
        t = b.transcribe(audio, "ko")
    finally:
        b.unload()
    assert lang == "ko" and conf > 0.5
    assert isinstance(t, Transcript)
    assert t.word_timing is False and t.language == "ko" and t.segments
    assert all(s.words is None and s.end > s.start for s in t.segments)
