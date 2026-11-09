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


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, _samples, **kw):
        from types import SimpleNamespace

        self.calls.append(kw)
        seg = SimpleNamespace(start=0.0, end=1.0, text=" hello there ")
        return iter([seg]), SimpleNamespace(language="en")


def _wav(path):
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return path


def test_decoding_options_pass_through_and_defaults_stay_whispers(tmp_path):
    """Ruling CD16: eval sets temperature, beam, conditioning and the hallucination guard;
    a stage that sets none keeps faster-whisper's defaults (nothing extra is passed)."""
    audio = _wav(tmp_path / "a.wav")
    plain = FasterWhisperTranscriber({}, "cpu")
    plain._model = _FakeModel()
    plain.transcribe(audio, "en")
    assert set(plain._model.calls[0]) == {"language", "vad_filter", "word_timestamps"}

    opts = {
        "temperature": 0.0, "beam_size": 5, "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
    }  # fmt: skip
    det = FasterWhisperTranscriber(opts, "cpu")
    det._model = _FakeModel()
    t = det.transcribe(audio, "en")
    kw = det._model.calls[0]
    assert kw["temperature"] == [0.0]  # one temperature: no fallback sampling
    assert kw["beam_size"] == 5 and kw["condition_on_previous_text"] is False
    assert kw["hallucination_silence_threshold"] == 2.0
    assert kw["word_timestamps"] is True  # the guard needs word timing
    assert t.word_timing is False and t.segments[0].text == "hello there"


def test_unknown_or_bad_decoding_options_are_refused():
    with pytest.raises(ValueError, match="temperature"):
        FasterWhisperTranscriber({"temperature": "hot"}, "cpu")
    with pytest.raises(ValueError, match="beam_size"):
        FasterWhisperTranscriber({"beam_size": 0}, "cpu")
