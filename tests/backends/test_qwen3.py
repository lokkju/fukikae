import logging
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fukikae.backends import loader
from fukikae.backends.align.qwen3 import Qwen3ForcedAligner, group_segments
from fukikae.backends.asr.qwen3 import (
    LANGUAGES,
    Qwen3AsrTranscriber,
    code_to_name,
    name_to_code,
    plan_chunks,
)
from fukikae.models import Segment, Transcript


def seg(a: float, b: float, text: str = "x") -> Segment:
    return Segment(start=a, end=b, text=text)


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def test_declarations_and_registration():
    a, g = Qwen3AsrTranscriber, Qwen3ForcedAligner
    assert (a.name, a.kind, a.supports_lid) == ("qwen3_asr", "asr", True)
    assert len(a.languages) == 30 and {"ko", "ja", "zh", "yue", "en"} <= set(a.languages)
    assert g.languages == frozenset(
        {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}
    )
    assert (g.name, g.kind) == ("qwen3_forced_aligner", "align")
    assert Qwen3AsrTranscriber({}, "cpu").model == "Qwen/Qwen3-ASR-1.7B"
    assert Qwen3ForcedAligner({}, "cpu").model == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert "qwen3_asr" in loader.names("asr") and "qwen3_asr" in loader.names("lid")
    assert "qwen3_forced_aligner" in loader.names("align")
    assert loader.EXTRAS["qwen3_asr"] == loader.EXTRAS["qwen3_forced_aligner"] == "qwen"


def test_default_chunk_plus_padding_fits_the_models_30_s_window():
    import inspect

    from fukikae.backends.asr import qwen3

    t = Qwen3AsrTranscriber({}, "cpu")
    assert t.max_chunk_s == 28.0
    assert t.max_chunk_s + 2 * qwen3._PAD_S <= 30.0
    assert inspect.signature(plan_chunks).parameters["max_s"].default == t.max_chunk_s


def test_language_names_map_to_iso_codes():
    assert name_to_code("Korean") == "ko" and name_to_code("cantonese") == "yue"
    assert name_to_code("Chinese") == "zh" and name_to_code("Japanese") == "ja"
    assert code_to_name("ko") == "Korean" and code_to_name("yue") == "Cantonese"
    assert name_to_code("Klingon") is None
    with pytest.raises(ValueError, match="xx"):
        code_to_name("xx")
    assert len(LANGUAGES) == 30


def test_group_segments_cuts_at_segment_boundaries():
    groups = group_segments([seg(0, 200), seg(200, 350), seg(350, 400)])
    assert [[(s.start, s.end) for s in g] for g in groups] == [[(0, 200)], [(200, 350), (350, 400)]]


def test_group_segments_oversize_segment_gets_its_own_chunk(caplog):
    with caplog.at_level(logging.WARNING):
        groups = group_segments([seg(0, 10), seg(10, 410), seg(410, 420)])
    assert [[(s.start, s.end) for s in g] for g in groups] == [[(0, 10)], [(10, 410)], [(410, 420)]]
    assert "400" in caplog.text


def test_plan_chunks_cuts_long_audio_at_a_silence():
    sr = 16000
    rng = np.random.default_rng(0)
    speech = lambda s: (rng.standard_normal(int(s * sr)) * 0.2).astype(np.float32)  # noqa: E731
    gap = np.zeros(int(0.6 * sr), dtype=np.float32)
    audio = np.concatenate([speech(20), gap, speech(20), gap, speech(10)])
    chunks = plan_chunks(audio, sr, max_s=30.0)
    assert len(chunks) == 3 or len(chunks) == 2
    assert all(b - a <= 30.0 for a, b in chunks)
    assert all(a < b for a, b in chunks) and chunks == sorted(chunks)
    # the first cut falls inside the first silence (20.0-20.6 s), not mid-speech
    assert 19.9 <= chunks[0][1] <= 20.7


def test_plan_chunks_drops_silence_and_falls_back_to_fixed_windows():
    sr = 16000
    assert plan_chunks(np.zeros(5 * sr, dtype=np.float32), sr) == []
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(70 * sr) * 0.2).astype(np.float32)  # no pauses anywhere
    chunks = plan_chunks(noise, sr, max_s=30.0)
    assert all(b - a <= 30.0 + 1e-6 for a, b in chunks)
    assert chunks[0][0] == pytest.approx(0.0, abs=0.05) and chunks[-1][1] == pytest.approx(
        70.0, abs=0.05
    )
    assert all(abs(b - c) < 1.0 for (_, b), (c, _) in zip(chunks, chunks[1:], strict=False))


class FakeAsr:
    def __init__(self, language="Korean"):
        self.calls = []
        self.language = language

    def transcribe(self, audio, language=None):
        self.calls.append((len(audio), language))
        return [
            SimpleNamespace(language=self.language, text=f"텍스트 {i}") for i in range(len(audio))
        ]


def test_transcribe_emits_one_segment_per_chunk(tmp_path):
    sr = 16000
    rng = np.random.default_rng(2)
    sp = lambda s: (rng.standard_normal(int(s * sr)) * 0.2).astype(np.float32)  # noqa: E731
    gap = np.zeros(int(0.6 * sr), dtype=np.float32)
    wav = tmp_path / "a.wav"
    write_wav(wav, np.concatenate([sp(20), gap, sp(20)]))
    b = Qwen3AsrTranscriber({}, "cpu")
    fake = FakeAsr()
    b._model = fake
    t = b.transcribe(wav, "ko")
    assert t.language == "ko" and t.backend == "qwen3_asr" and t.word_timing is False
    assert len(t.segments) == 2 and t.segments[0].end <= t.segments[1].start
    assert t.segments[0].text == "텍스트 0" and all(s.words is None for s in t.segments)
    assert fake.calls[0][1] == ["Korean", "Korean"]


def test_detect_language_maps_name_to_code(tmp_path):
    rng = np.random.default_rng(3)
    wav = tmp_path / "a.wav"
    write_wav(wav, (rng.standard_normal(16000 * 12) * 0.2).astype(np.float32))
    b = Qwen3AsrTranscriber({}, "cpu")
    b._model = FakeAsr(language="Cantonese")
    lang, conf = b.detect_language(wav)
    assert lang == "yue" and conf == 1.0


def test_requires_load(tmp_path):
    with pytest.raises(RuntimeError, match="load"):
        Qwen3AsrTranscriber({}, "cpu").transcribe(tmp_path / "a.wav", "ko")
    t = Transcript(language="ko", backend="x", word_timing=False, segments=[])
    with pytest.raises(RuntimeError, match="load"):
        Qwen3ForcedAligner({}, "cpu").align(tmp_path / "a.wav", t)


def test_aligner_unsupported_language(tmp_path):
    a = Qwen3ForcedAligner({}, "cpu")
    a._net = object()
    t = Transcript(language="th", backend="x", word_timing=False, segments=[seg(0, 1, "a")])
    with pytest.raises(ValueError, match="th"):
        a.align(tmp_path / "a.wav", t)


class FakeNet:
    """Stand-in for qwen_asr's Qwen3ForcedAligner: one item per whitespace word, 0.5 s each."""

    def __init__(self):
        self.calls = []
        self.aligner_processor = SimpleNamespace(
            encode_timestamp=lambda text, _lang: (text.split(), "")
        )

    def align(self, audio, text, language):
        self.calls.append((len(audio[0][0]), text, language))
        words = text[0].split()
        return [
            [
                SimpleNamespace(text=w, start_time=0.5 * i, end_time=0.5 * i + 0.4)
                for i, w in enumerate(words)
            ]
        ]


def test_align_groups_segments_and_maps_words_back(tmp_path):
    wav = tmp_path / "a.wav"
    write_wav(wav, np.zeros(16000 * 10, dtype=np.float32))
    a = Qwen3ForcedAligner({}, "cpu")
    net = FakeNet()
    a._net = net
    t = Transcript(
        language="ko",
        backend="qwen3_asr",
        word_timing=False,
        segments=[seg(1.0, 3.0, "안녕 하세요"), seg(4.0, 6.0, "반갑 습니다 여러분")],
    )
    out = a.align(wav, t)
    assert out.word_timing is True and out.backend == "qwen3_asr" and out.language == "ko"
    assert len(net.calls) == 1 and net.calls[0][2] == ["Korean"]
    s0, s1 = out.segments
    assert [w.text for w in s0.words or []] == ["안녕", "하세요"]
    assert [w.text for w in s1.words or []] == ["반갑", "습니다", "여러분"]
    # crop starts at the first segment, so the first word begins there
    assert (s0.words or [])[0].start == pytest.approx(1.0)
    flat = [w for s in out.segments for w in (s.words or [])]
    assert all(w.start <= w.end for w in flat)
    assert all(a.start <= b.start for a, b in zip(flat, flat[1:], strict=False))
    assert all(s.start <= w.start and w.end <= s.end for s in out.segments for w in (s.words or []))


@pytest.mark.gpu
def test_qwen_transcribes_and_aligns_korean_clip(clip_ep):
    import re

    audio = clip_ep.work / "01-separate" / "vocals_16k.wav"
    asr = Qwen3AsrTranscriber({}, "cuda:0")
    asr.load()
    try:
        lang, conf = asr.detect_language(audio)
        t = asr.transcribe(audio, "ko")
    finally:
        asr.unload()
    assert lang == "ko" and conf > 0.5
    assert t.segments and re.search("[가-힣]", " ".join(s.text for s in t.segments))
    al = Qwen3ForcedAligner({}, "cuda:0")
    al.load()
    try:
        out = al.align(audio, t)
    finally:
        al.unload()
    assert out.word_timing
    with wave.open(str(audio)) as w:
        dur = w.getnframes() / w.getframerate()
    flat = [w for s in out.segments for w in (s.words or [])]
    assert flat
    assert all(0 <= a.start <= a.end <= dur + 0.01 for a in flat)
    assert all(a.start <= b.start for a, b in zip(flat, flat[1:], strict=False))
