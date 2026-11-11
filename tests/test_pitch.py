"""The pitch module: F0 track, summaries, median-based gender and the cache (A0)."""

from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from fukikae import pitch
from fukikae.pitch import (
    F0Summary,
    band_seconds,
    bimodal,
    cached_track,
    classify,
    f0_track,
    majority_register,
    summarize,
    turn_classes,
)

RATE = 16000


def tone(f0: float, seconds: float, partials: tuple[float, ...] = (1, 1 / 2, 1 / 3, 1 / 4, 1 / 5)):
    t = np.arange(int(seconds * RATE)) / RATE
    x = np.zeros_like(t)
    for k, a in enumerate(partials, start=1):
        x += a * np.sin(2 * np.pi * f0 * k * t)
    return (0.2 * x / max(1.0, float(np.abs(x).max()) / 2)).astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def write_harmonic(tmp_path: Path, hz: float, seconds: float) -> Path:
    return write_wav(tmp_path / f"h{hz:g}.wav", tone(hz, seconds))


def write_noise(tmp_path: Path, seconds: float) -> Path:
    rng = np.random.default_rng(0)
    return write_wav(tmp_path / "noise.wav", 0.2 * rng.standard_normal(int(seconds * RATE)))


def summary(median: float | None, voiced_s: float) -> F0Summary:
    return F0Summary(
        median=median, p10=median, p20=median, p90=median,
        voiced_s=voiced_s, seconds=voiced_s, voiced_fraction=1.0,
    )  # fmt: skip


def test_constants_match_the_spec():
    assert (pitch.F0_MIN, pitch.F0_MAX) == (60, 800)
    assert (pitch.MALE_BELOW, pitch.FEMALE_ABOVE) == (165, 200)
    assert (pitch.SPEAKER_MIN_VOICED_S, pitch.LINE_MIN_VOICED_S) == (1.0, 0.4)
    assert pitch.CONFIDENCE_ST == 4.0
    assert (pitch.BIMODAL_MIN_S, pitch.BIMODAL_MIN_SHARE) == (5.0, 0.15)
    assert pitch.VERSION == 2  # bumped with every estimator change (the cache key)


def test_harmonic_tone_reads_its_fundamental(tmp_path: Path):
    for hz in (90, 220, 420, 640):  # 420 and 640 were folded under the old 400 Hz cap
        wav = write_harmonic(tmp_path, hz, seconds=2.0)
        s = summarize(f0_track(wav, [(0.0, 2.0)]))
        assert s.median is not None
        assert abs(12 * math.log2(s.median / hz)) < 0.5, (hz, s.median)


def test_strong_second_harmonic_does_not_halve_f0(tmp_path: Path):
    wav = write_wav(tmp_path / "h2.wav", tone(200, 2.0, partials=(0.5, 1.0, 0.2)))
    s = summarize(f0_track(wav, [(0.0, 2.0)]))
    assert s.median is not None and abs(12 * math.log2(s.median / 200)) < 0.5


def test_rich_low_voice_reads_its_fundamental(tmp_path: Path):
    # a weak fundamental under strong partials 2 to 8 (even ones strongest): the window's taper
    # shrank the long-lag peak at the period under 90% of the half-period's, and 90 to 110 Hz
    # read an octave high before the window correction
    rich = (0.3, 1.0, 0.4, 1.0, 0.4, 0.8, 0.3, 0.6)
    for hz in (90, 100, 110, 120):
        wav = write_wav(tmp_path / f"rich{hz}.wav", tone(hz, 2.0, partials=rich))
        s = summarize(f0_track(wav, [(0.0, 2.0)]))
        assert s.median is not None and abs(12 * math.log2(s.median / hz)) < 0.5, (hz, s.median)


def _write_any(path: Path, x: np.ndarray, rate: int = RATE) -> Path:
    from scipy.io import wavfile

    wavfile.write(path, rate, x)
    return path


def test_int16_stereo_float_and_8bit_wavs_read_alike(tmp_path: Path):
    x = tone(220, 2.0)
    x44 = np.interp(np.arange(88200) / 44100, np.arange(len(x)) / RATE, x)
    files = {
        "float32": _write_any(tmp_path / "f.wav", x),
        "int16 stereo": _write_any(
            tmp_path / "s.wav", (np.stack([x, 0.5 * x], axis=1) * 32767).astype("<i2")
        ),
        "int32": _write_any(tmp_path / "i32.wav", (x * (2**31 - 1)).astype("<i4")),
        "uint8": _write_any(tmp_path / "u8.wav", (x * 127 + 128).astype(np.uint8)),
        "44.1 kHz": _write_any(
            tmp_path / "44k.wav",
            (x44 * 32767).astype("<i2"),
            44100,
        ),
    }
    for name, wav in files.items():
        s = summarize(f0_track(wav, [(0.0, 2.0)]))
        assert s.median is not None and abs(12 * math.log2(s.median / 220)) < 0.5, (name, s.median)
        assert s.seconds == pytest.approx(2.0, abs=0.01), name


def test_reversed_and_zero_length_spans_keep_their_place(tmp_path: Path):
    wav = write_harmonic(tmp_path, 260, 2.0)
    track = f0_track(wav, [(1.0, 1.0), (0.0, 2.0), (1.5, 0.5)])
    assert track.spans == 3
    np.testing.assert_allclose(track.span_seconds, [0.0, 2.0, 0.0], atol=0.01)
    assert set(np.unique(track.span_index)) == {1}
    assert summarize(track, 0).median is None and summarize(track, 0).seconds == 0.0
    assert summarize(track, 2).median is None
    assert [c.gender for c in turn_classes(track, (165, 200))] == ["u", "f", "u"]


def test_batched_track_equals_one_batch(tmp_path: Path, monkeypatch):
    wav, spans = _turns(tmp_path, [(130, 1.3), (0, 0.5), (260, 2.1), (185, 0.7), (420, 1.0)])
    spans = [*spans, (0.2, 4.9), (3.0, 3.0)]
    monkeypatch.setattr(pitch, "_BATCH", 1 << 30)
    whole = f0_track(wav, spans)
    monkeypatch.setattr(pitch, "_BATCH", 7)
    batched = f0_track(wav, spans)
    np.testing.assert_array_equal(whole.f0, batched.f0)
    np.testing.assert_array_equal(whole.span_index, batched.span_index)
    np.testing.assert_array_equal(whole.span_seconds, batched.span_seconds)
    assert whole.seconds == batched.seconds


def test_long_track_memory_stays_bounded(tmp_path: Path):
    import tracemalloc

    minutes = 10
    x = np.tile(tone(220, 3.0), 20 * minutes)
    wav = write_wav(tmp_path / "long.wav", x)
    del x
    tracemalloc.start()
    try:
        track = f0_track(wav, [(0.0, 60.0 * minutes)])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # every frame materialised at once peaked at 438 MB here; batches keep it flat in length
    assert peak < 80e6, peak / 1e6
    s = summarize(track)
    assert s.median is not None and abs(s.median - 220) < 5


def test_noise_is_unvoiced(tmp_path: Path):
    s = summarize(f0_track(write_noise(tmp_path, 2.0), [(0, 2)]))
    assert s.median is None and s.voiced_fraction < 0.1


def test_track_records_each_frames_span_and_the_seconds_read(tmp_path: Path):
    wav = write_wav(tmp_path / "two.wav", np.concatenate([tone(130, 1.0), tone(260, 1.0)]))
    track = f0_track(wav, [(1.0, 2.0), (0.0, 1.0)])  # input order, not time order
    assert set(np.unique(track.span_index)) == {0, 1}
    assert len(track.f0) == len(track.span_index)
    assert track.seconds == pytest.approx(2.0, abs=0.01)
    first, second = summarize(track, 0), summarize(track, 1)
    assert first.median is not None and abs(first.median - 260) < 8
    assert second.median is not None and abs(second.median - 130) < 5
    assert first.seconds == pytest.approx(1.0, abs=0.01)
    assert 0.8 < first.voiced_s <= 1.0 and first.voiced_fraction > 0.9


def test_empty_and_out_of_range_spans_read_nothing(tmp_path: Path):
    wav = write_harmonic(tmp_path, 220, 1.0)
    assert summarize(f0_track(wav, [])).median is None
    track = f0_track(wav, [(5.0, 6.0)])
    s = summarize(track)
    assert s.median is None and s.seconds == 0.0 and s.voiced_fraction == 0.0


def test_classify_bands_on_the_median():
    assert classify(summary(median=140, voiced_s=3)).gender == "m"
    assert classify(summary(median=240, voiced_s=3)).gender == "f"
    assert classify(summary(median=185, voiced_s=3)).gender == "u"
    assert classify(summary(median=240, voiced_s=0.5)).gender == "u"  # under the speaker minimum
    assert classify(summary(median=None, voiced_s=0)).gender == "u"
    assert classify(summary(median=185, voiced_s=3), female_above=180).gender == "f"


def test_confidence_grows_with_distance_from_the_edge():
    near = classify(summary(median=210, voiced_s=3))
    far = classify(summary(median=320, voiced_s=3))
    assert 0 < near.confidence < far.confidence == 1.0
    assert near.confidence == pytest.approx(abs(12 * math.log2(210 / 200)) / 4.0)
    assert classify(summary(median=185, voiced_s=3)).confidence == 0.0
    male = classify(summary(median=140, voiced_s=3))
    assert male.confidence == pytest.approx(min(1.0, abs(12 * math.log2(140 / 165)) / 4.0))
    assert male.median == 140 and male.voiced_s == 3


def _turns(tmp_path: Path, plan: Sequence[tuple[float, float]]) -> tuple[Path, list]:
    """One wav of back-to-back turns, each (hz, seconds); hz 0 is silence."""
    parts, spans, t = [], [], 0.0
    for hz, sec in plan:
        parts.append(tone(hz, sec) if hz else np.zeros(int(sec * RATE), np.float32))
        spans.append((t, t + sec))
        t += sec
    return write_wav(tmp_path / "turns.wav", np.concatenate(parts)), spans


def test_turn_classes_one_per_span(tmp_path: Path):
    wav, spans = _turns(tmp_path, [(130, 1.0), (260, 1.0), (185, 1.0), (0, 1.0), (260, 0.3)])
    classes = turn_classes(f0_track(wav, spans), (165, 200))
    assert [c.gender for c in classes] == ["m", "f", "u", "u", "u"]  # the last under 0.4 s voiced


def test_majority_register_ignores_the_minor_mode(tmp_path: Path):
    # 6 turns at 130 Hz, 14 turns at 260 Hz
    wav, spans = _turns(tmp_path, [(130, 1.0)] * 6 + [(260, 1.0)] * 14)
    track = f0_track(wav, spans)
    g, s = majority_register(track, (165, 200))
    assert g == "f" and s.median is not None and abs(s.median - 260) < 8
    seconds = band_seconds(track, (165, 200))
    assert seconds["m"] == pytest.approx(6.0, abs=0.05)
    assert seconds["f"] == pytest.approx(14.0, abs=0.05)
    assert seconds["u"] == pytest.approx(0.0, abs=0.05)


def test_majority_register_of_nothing_classified_is_unknown(tmp_path: Path):
    wav, spans = _turns(tmp_path, [(0, 1.0), (185, 1.0)])
    g, _ = majority_register(f0_track(wav, spans), (165, 200))
    assert g == "u"


def test_bimodal_needs_seconds_and_share():
    assert bimodal({"m": 19, "u": 34, "f": 70}) and not bimodal({"m": 6, "u": 47, "f": 177})
    assert not bimodal({"m": 4, "u": 0, "f": 10})  # share but under 5 s
    assert not bimodal({"m": 0, "u": 30, "f": 0})


def test_cached_track_is_reused_and_keyed_by_spans(tmp_path: Path, monkeypatch):
    wav = write_harmonic(tmp_path, 220, 2.0)
    root = tmp_path / "ws"
    calls: list[int] = []
    real = pitch.f0_track

    def counting(vocals, spans):
        calls.append(1)
        return real(vocals, spans)

    monkeypatch.setattr(pitch, "f0_track", counting)
    a = cached_track(root, wav, [(0.0, 1.0)])
    b = cached_track(root, wav, [(0.0, 1.0)])
    assert len(calls) == 1
    np.testing.assert_array_equal(a.f0, b.f0)
    np.testing.assert_array_equal(a.span_index, b.span_index)
    assert a.seconds == b.seconds
    assert summarize(a, 0) == summarize(b, 0)
    assert len(list((root / ".fukikae" / "pitch-cache").glob("*.npz"))) == 1
    cached_track(root, wav, [(0.0, 2.0)])
    assert len(calls) == 2
    # rewritten vocals miss
    write_wav(wav, tone(130, 2.0))
    c = cached_track(root, wav, [(0.0, 1.0)])
    assert len(calls) == 3
    s = summarize(c)
    assert s.median is not None and abs(s.median - 130) < 5


def test_cached_track_recomputes_a_truncated_cache_file(tmp_path: Path, monkeypatch):
    wav = write_harmonic(tmp_path, 220, 2.0)
    root = tmp_path / "ws"
    first = cached_track(root, wav, [(0.0, 2.0)])
    (path,) = (root / ".fukikae" / "pitch-cache").glob("*.npz")
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    calls: list[int] = []
    real = pitch.f0_track

    def counting(vocals, spans):
        calls.append(1)
        return real(vocals, spans)

    monkeypatch.setattr(pitch, "f0_track", counting)
    again = cached_track(root, wav, [(0.0, 2.0)])
    assert calls == [1]
    np.testing.assert_array_equal(first.f0, again.f0)
    path.write_bytes(b"")  # an empty file is EOFError for np.load
    cached_track(root, wav, [(0.0, 2.0)])
    assert calls == [1, 1]
    with np.load(path) as z:  # rewritten whole
        np.testing.assert_array_equal(z["f0"], first.f0)


def test_cache_key_names_the_estimator_version(tmp_path: Path, monkeypatch):
    wav = write_harmonic(tmp_path, 220, 1.0)
    key = pitch.track_key(wav, [(0.0, 1.0)])
    monkeypatch.setattr(pitch, "VERSION", pitch.VERSION + 1)
    assert pitch.track_key(wav, [(0.0, 1.0)]) != key
