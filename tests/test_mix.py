import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from fukikae.mix import (
    MixSettings,
    build_dub,
    measure_loudness,
    mux,
    normalize,
    removal_windows,
    subtract,
    union,
    write_ass,
)
from fukikae.models import Delivery, DubCue, Tags, TagSpan

RATE = 8000


def _tags(*events: tuple[float, float, str]) -> Tags:
    return Tags(
        backend="sensevoice",
        events=[TagSpan(start=a, end=b, label=lab) for a, b, lab in events],
        emotions=[],
    )


def test_union_and_subtract():
    assert union([(5, 6), (1, 3), (2, 4)]) == [(1, 4), (5, 6)]
    assert subtract([(0, 10)], [(2, 3), (5, 7)]) == [(0, 2), (3, 5), (7, 10)]
    assert subtract([(1, 2)], [(0, 5)]) == []


def test_removal_windows_are_the_union_of_placed_spans():
    assert removal_windows([(1, 3), (2, 4), (6, 7)], None) == [(1, 4), (6, 7)]


def test_laughter_never_inside_removal_window():
    w = removal_windows([(1, 3), (5, 8)], _tags((2, 2.5, "laughter")))
    assert w == [(1, 2), (2.5, 3), (5, 8)]


def test_singing_span_keeps_original_vocals():
    w = removal_windows([(4, 8)], _tags((5, 6, "singing")))
    assert w == [(4, 5), (6, 8)]


def test_bgm_spans_protected_only_when_asked():
    tags = _tags((5, 6, "bgm"))
    assert removal_windows([(4, 8)], tags) == [(4, 8)]
    assert removal_windows([(4, 8)], tags, protect_bgm=True) == [(4, 5), (6, 8)]


def test_song_spans_from_any_source_keep_original_vocals():
    assert removal_windows([(0, 10)], None, songs=[(2, 3)]) == [(0, 2), (3, 10)]


def _mono(path: Path, x: np.ndarray, rate: int = RATE) -> Path:
    wavfile.write(path, rate, x.astype(np.float32))
    return path


def _stereo(path: Path, x: np.ndarray, rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        st = np.stack([x, x], axis=1)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _tone(freq: float, seconds: float, amp: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


@pytest.fixture
def stems(tmp_path):
    bg = _stereo(tmp_path / "bg.wav", np.zeros(6 * RATE))
    vo = _stereo(tmp_path / "vo.wav", _tone(1000, 6, 0.2))
    return bg, vo


NO_MATCH = MixSettings(fade_ms=0, match_dialogue=False)


def test_vocals_removed_inside_windows_and_kept_outside(tmp_path, stems):
    bg, vo = stems
    out = tmp_path / "dub.wav"
    build_dub(bg, vo, [], [(2.0, 4.0)], NO_MATCH, out)
    rate, y = wavfile.read(out)
    assert rate == RATE and y.ndim == 2
    assert _rms(y[int(2.1 * RATE) : int(3.9 * RATE), 0]) < 1e-4
    assert _rms(y[int(0.5 * RATE) : int(1.5 * RATE), 0]) == pytest.approx(
        0.2 / np.sqrt(2), rel=0.02
    )


def test_dub_cue_placed_at_its_start_and_resampled(tmp_path, stems):
    bg, vo = stems
    cue = _mono(tmp_path / "c.wav", _tone(300, 1.0, 0.3, rate=24000), rate=24000)
    out = tmp_path / "dub.wav"
    build_dub(bg, vo, [(cue, 2.5)], [(2.5, 3.5)], NO_MATCH, out)
    _, y = wavfile.read(out)
    inside = y[int(2.6 * RATE) : int(3.4 * RATE), 0]
    assert _rms(inside) == pytest.approx(0.3 / np.sqrt(2), rel=0.03)
    assert _rms(y[int(1.0 * RATE) : int(2.0 * RATE), 1]) == pytest.approx(
        0.2 / np.sqrt(2), rel=0.02
    )


def test_overlapping_cues_both_audible(tmp_path, stems):  # Review Focus 5
    bg, vo = stems
    a = _mono(tmp_path / "a.wav", _tone(300, 2.0, 0.2))
    b = _mono(tmp_path / "b.wav", _tone(500, 2.0, 0.2))
    out = tmp_path / "dub.wav"
    build_dub(bg, vo, [(a, 1.0), (b, 2.0)], [(1.0, 4.0)], NO_MATCH, out)
    _, y = wavfile.read(out)
    seg = y[int(2.1 * RATE) : int(2.9 * RATE), 0].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1 / RATE)
    peaks = {round(f) for f in freqs[np.argsort(spectrum)[-2:]]}
    assert peaks == {300, 500}


def test_duck_keeps_quiet_bleed(tmp_path, stems):
    bg, vo = stems
    out = tmp_path / "dub.wav"
    build_dub(
        bg, vo, [], [(2.0, 4.0)], MixSettings(duck_db=-18, fade_ms=0, match_dialogue=False), out
    )
    _, y = wavfile.read(out)
    inside = _rms(y[int(2.1 * RATE) : int(3.9 * RATE), 0])
    assert inside / (0.2 / np.sqrt(2)) == pytest.approx(10 ** (-18 / 20), rel=0.03)  # ~0.126


def test_fade_ramps_the_vocals_at_window_edges(tmp_path, stems):
    bg, vo = stems
    out = tmp_path / "dub.wav"
    build_dub(bg, vo, [], [(2.0, 4.0)], MixSettings(fade_ms=200, match_dialogue=False), out)
    _, y = wavfile.read(out)
    # Gone by the window start, halfway down 100 ms before it.
    assert _rms(y[int(2.0 * RATE) : int(2.1 * RATE), 0]) < 1e-4
    mid = _rms(y[int(1.88 * RATE) : int(1.92 * RATE), 0]) / (0.2 / np.sqrt(2))
    assert 0.3 < mid < 0.7


def test_without_original_vocals_only_bed_and_dub_remain(tmp_path, stems):
    bg, vo = stems
    out = tmp_path / "dub.wav"
    s = MixSettings(keep_original_vocals=False, fade_ms=0, match_dialogue=False)
    build_dub(bg, vo, [], [(0.0, 6.0)], s, out)
    _, y = wavfile.read(out)
    assert _rms(y[:, 0]) < 1e-4


def test_dialogue_level_matched_to_the_original_vocals(tmp_path, stems):
    bg, vo = stems
    cue = _mono(tmp_path / "c.wav", _tone(300, 1.0, 0.8))  # four times the vocals' level
    out = tmp_path / "dub.wav"
    stats = build_dub(bg, vo, [(cue, 2.0)], [(2.0, 3.0)], MixSettings(fade_ms=0), out)
    assert stats.dub_gain_db == pytest.approx(-12.04, abs=0.2)
    _, y = wavfile.read(out)
    assert _rms(y[int(2.1 * RATE) : int(2.9 * RATE), 0]) == pytest.approx(
        0.2 / np.sqrt(2), rel=0.03
    )


def test_a_cue_past_the_end_is_not_cut(tmp_path, stems):
    bg, vo = stems
    cue = _mono(tmp_path / "c.wav", _tone(300, 2.0, 0.3))
    out = tmp_path / "dub.wav"
    build_dub(bg, vo, [(cue, 5.0)], [(5.0, 6.0)], NO_MATCH, out)
    rate, y = wavfile.read(out)
    assert len(y) / rate == pytest.approx(7.0, abs=0.01)


def _dub_cue(i: int, speaker: str, text: str | None, start: float, end: float) -> DubCue:
    return DubCue(
        id=f"c{i:04d}",
        start=start,
        end=end,
        character_id=None,
        extra=None,
        speaker=speaker,
        source_text="src",
        text=text,
        budget=20,
        delivery=Delivery(reference=(start, end)),
    )


def test_write_ass_names_speakers(tmp_path):
    out = tmp_path / "dub.ass"
    write_ass(
        [
            _dub_cue(1, "Natsu", "Let's go!\nNow.", 1.5, 3.25),
            _dub_cue(2, "extra:SPEAKER_03", "A {brace} line", 3661.0, 3662.5),
        ],
        out,
    )
    text = out.read_text(encoding="utf-8")
    assert "[Script Info]" in text and "[V4+ Styles]" in text and "[Events]" in text
    events = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    assert (
        events[0]
        == "Dialogue: 0,0:00:01.50,0:00:03.25,Default,Natsu,0,0,0,,Natsu: Let's go!\\NNow."
    )
    assert events[1] == (
        "Dialogue: 0,1:01:01.00,1:01:02.50,Default,extra:SPEAKER_03,0,0,0,,Extra: A (brace) line"
    )


def _noise(seconds: float, amp: float, rate: int = 44100) -> np.ndarray:
    rng = np.random.default_rng(1)
    return amp * rng.standard_normal(round(seconds * rate)).clip(-3, 3) / 3


@pytest.mark.ffmpeg
def test_normalize_matches_a_target_loudness(tmp_path):
    src = _stereo(tmp_path / "a.wav", _noise(6, 0.05), 44100)
    before = measure_loudness(src)
    out = tmp_path / "b.wav"
    result = normalize(src, -20.0, out)
    after = measure_loudness(out)
    assert before < -25
    assert after == pytest.approx(-20.0, abs=0.5)
    assert result["normalization_type"] in ("linear", "dynamic")
    rate, _ = wavfile.read(out)
    assert rate == 44100


def _probe(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)["streams"]


@pytest.fixture
def source_mkv(tmp_path):
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:01,500\nHello\n", encoding="utf-8")
    made = tmp_path / "src.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-i", str(srt),
            "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "srt",
            "-metadata:s:a:0", "language=jpn", "-metadata:s:s:0", "language=eng",
            "-disposition:a:0", "default", "-disposition:s:0", "default",
            str(made),
        ],
        check=True,
    )  # fmt: skip
    return made


@pytest.mark.ffmpeg
def test_mux_keeps_original_tracks(tmp_path, source_mkv):
    dub = _stereo(tmp_path / "dub.wav", _tone(300, 3.0, 0.2, rate=44100), 44100)
    ass = tmp_path / "dub.ass"
    write_ass([_dub_cue(1, "Natsu", "Hi", 0.5, 1.5)], ass)
    out = tmp_path / "out" / "e.fukikae.mkv"
    mux(source_mkv, dub, ass, "en", out)
    streams = _probe(out)
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    subs = [s for s in streams if s["codec_type"] == "subtitle"]
    assert len(video) == 1 and video[0]["codec_name"] == "mpeg4"
    assert [a["tags"]["language"] for a in audio] == ["eng", "jpn"]
    assert audio[0]["codec_name"] == "aac" and audio[0]["disposition"]["default"] == 1
    assert audio[1]["disposition"]["default"] == 0
    assert [s["codec_name"] for s in subs] == ["ass", "subrip"]
    assert [s["tags"]["language"] for s in subs] == ["eng", "eng"]
    assert all(s["disposition"]["default"] == 0 for s in subs)
