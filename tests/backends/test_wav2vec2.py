import numpy as np
import pytest
import torch

from doppio.backends import loader
from doppio.backends.align.wav2vec2 import MODELS, Wav2Vec2Aligner, ctc_align, split_words
from doppio.models import Transcript


def test_declaration_and_registration():
    c = Wav2Vec2Aligner
    assert (c.name, c.kind) == ("wav2vec2", "align")
    assert c.languages == frozenset({"ko", "ja", "zh", "en"})
    assert MODELS["ko"] == "kresnik/wav2vec2-large-xlsr-korean"
    assert MODELS["ja"] == "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    assert MODELS["zh"] == "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    assert "wav2vec2" in loader.names("align")


def test_split_words():
    assert split_words("안녕 하세요, 여러분", "ko") == ["안녕", "하세요,", "여러분"]
    assert split_words("こんにちは、世界 1", "ja") == list("こんにちは世界1")
    assert split_words("你好，世界", "zh") == list("你好世界")
    assert split_words("Hello  there", "en") == ["Hello", "there"]


def test_ctc_align_is_monotonic_on_synthetic_emission():
    # vocabulary: 0 blank, 1..3; target 1 2 3 emitted at frames 2, 5, 8 of 10.
    em = torch.full((1, 10, 4), -10.0)
    em[:, :, 0] = 0.0
    for f, tok in ((2, 1), (5, 2), (8, 3)):
        em[0, f, tok] = 5.0
    em = em.log_softmax(-1)
    spans = ctc_align(em, [1, 2, 3], blank=0)
    assert spans is not None and len(spans) == 3
    assert [s[0] for s in spans] == [2, 5, 8]
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_ctc_align_returns_none_when_target_too_long():
    em = torch.zeros(1, 2, 4).log_softmax(-1)
    assert ctc_align(em, [1, 2, 3, 1], blank=0) is None


def test_align_requires_load(tmp_path):
    t = Transcript(language="ko", backend="x", word_timing=False, segments=[])
    with pytest.raises(RuntimeError, match="load"):
        Wav2Vec2Aligner({}, "cpu").align(tmp_path / "a.wav", t)


def test_unsupported_language_raises():
    a = Wav2Vec2Aligner({}, "cpu")
    with pytest.raises(ValueError, match="fr"):
        a.load_for("fr")


@pytest.mark.gpu
def test_aligns_korean_clip(clip_ep):
    from doppio.backends.asr.faster_whisper import FasterWhisperTranscriber

    audio = clip_ep.work / "01-separate" / "vocals_16k.wav"
    w = FasterWhisperTranscriber({}, "cuda:0")
    w.load()
    try:
        t = w.transcribe(audio, "ko")
    finally:
        w.unload()
    a = Wav2Vec2Aligner({}, "cuda:0")
    a.load()
    try:
        out = a.align(audio, t)
    finally:
        a.unload()
    assert out.word_timing and len(out.segments) == len(t.segments)
    scored = 0
    for seg in out.segments:
        assert seg.words
        prev = seg.start - 1e-6
        for w_ in seg.words:
            assert prev <= w_.start <= w_.end
            assert seg.start - 0.3 <= w_.start and w_.end <= seg.end + 0.3
            prev = w_.start
            scored += w_.score is not None
    assert scored > 0
    assert np.isfinite([w_.start for s in out.segments for w_ in s.words or []]).all()
