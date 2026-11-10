import wave
from pathlib import Path

import numpy as np
import pytest

from fukikae.songs import SongSpan, frame_features, theme_songs

DUR = 1430


def _episode(songs: list[tuple[int, int]], *, good_share: float = 1.0, dur: int = DUR):
    """Per-second features: dialogue (quiet bed, speech half the time) with songs (loud bed,
    vocals nearly always on) at ``songs``."""
    bg = np.full(dur, -45.0)
    vo = np.full(dur, -28.0)
    act = np.full(dur, 0.5)
    rng = np.random.default_rng(0)
    for a, b in songs:
        bg[a:b] = -25.0
        vo[a:b] = -27.0
        on = rng.random(b - a) < good_share
        act[a:b] = np.where(on, 0.9, 0.3)
    return bg, vo, act


def test_finds_the_opening_and_the_ending():
    spans = theme_songs(*_episode([(70, 160), (1310, 1400)]))
    assert [(round(s.start), round(s.end)) for s in spans] == [(69, 160), (1309, 1400)]
    assert all(s.source == "detector" and s.score is not None and s.score > 0.9 for s in spans)


def test_opening_after_a_long_cold_open_is_found():
    (span,) = theme_songs(*_episode([(210, 300)]))
    assert span.start == pytest.approx(209, abs=1) and span.end == pytest.approx(300, abs=1)


def test_a_song_mid_episode_is_not_a_theme():
    assert theme_songs(*_episode([(600, 690)])) == []


def test_dialogue_over_loud_music_is_not_a_song():
    bg, vo, act = _episode([])
    bg[:] = -24.0  # music as loud as the voices, but speech only half the time
    assert theme_songs(bg, vo, act) == []


def test_quiet_bed_with_constant_vocals_is_not_a_song():
    bg, vo, act = _episode([])
    act[:300] = 0.95  # a long monologue over a quiet bed
    assert theme_songs(bg, vo, act) == []


def test_patchy_evidence_stays_below_the_score_bar():
    assert theme_songs(*_episode([(70, 160)], good_share=0.35)) == []


def test_episode_shorter_than_the_window_has_no_theme():
    assert theme_songs(*_episode([], dur=60)) == []


def test_spans_never_exceed_the_cap():
    (span,) = theme_songs(*_episode([(20, 340)]))
    assert span.end - span.start <= 100 + 2  # MAX_SONG_S plus the padding


def _wav(path: Path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        st = np.stack([x, x], axis=1)
        w.writeframes((st * 32767).astype("<i2").tobytes())


def test_frame_features_from_stems(tmp_path):
    rate = 8000
    t = np.arange(4 * rate) / rate
    bg = 0.1 * np.sin(2 * np.pi * 200 * t)  # about -23 dBFS
    vo = np.where(t < 2, 0.1 * np.sin(2 * np.pi * 300 * t), 0.0)  # on for the first 2 s
    _wav(tmp_path / "bg.wav", bg, rate)
    _wav(tmp_path / "vo.wav", vo, rate)
    f_bg, f_vo, f_act = frame_features(tmp_path / "bg.wav", tmp_path / "vo.wav")
    assert len(f_bg) == len(f_vo) == len(f_act) == 4
    assert f_bg == pytest.approx([-23.0] * 4, abs=0.5)
    assert list(f_act) == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert f_vo[0] == pytest.approx(-23.0, abs=0.5) and f_vo[3] < -100


def test_song_span_is_a_value():
    assert SongSpan(1.0, 2.0, "subs") == SongSpan(1.0, 2.0, "subs", None)


# -- candidates, voicing and settings (Task 7 fix round 1) ---------------------------------

from fukikae.songs import SongSettings, find_candidates, voiced_share  # noqa: E402


def test_candidates_report_every_window_score_accepted_or_not():
    cands = find_candidates(*_episode([(70, 160)]), SongSettings())
    assert [c.region for c in cands] == ["head", "tail"]
    head, tail = cands
    assert head.accepted and head.score > 0.9 and head.reason == "accepted"
    assert not tail.accepted and tail.score < 0.1 and tail.reason == "score"


def test_shouting_over_music_is_refused_by_the_voicing_check():
    """A battle (loud music, vocals on all the time, but rough and unpitched) scores like a
    song on energy alone (Fairy Tail E13 at 611 s scored 0.86); singing is periodic."""
    feats = _episode([(70, 160)])
    cands = find_candidates(*feats, SongSettings(), voiced=lambda *_: 0.53)
    assert not cands[0].accepted and cands[0].reason == "voiced"
    assert cands[0].voiced == pytest.approx(0.53)
    sung = find_candidates(*feats, SongSettings(), voiced=lambda *_: 0.87)
    assert sung[0].accepted
    assert theme_songs(*feats, voiced=lambda *_: 0.53) == []


def test_settings_change_the_detector():
    feats = _episode([(500, 590)])
    assert theme_songs(*feats) == []
    (span,) = theme_songs(*feats, settings=SongSettings(head_s=700))
    assert span.start == pytest.approx(499, abs=1)
    with pytest.raises(ValueError, match="min_score"):
        SongSettings(min_score=1.5)


def _mono16(path: Path, x: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def test_voiced_share_tells_pitched_from_noisy(tmp_path):
    t = np.arange(4 * 16000) / 16000
    sung = 0.2 * sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 5))
    noisy = 0.2 * np.random.default_rng(0).standard_normal(len(t)).clip(-3, 3) / 3
    x = np.concatenate([sung, noisy, np.zeros(len(t))])
    p = _mono16(tmp_path / "v16.wav", x)
    assert voiced_share(p, 0.0, 4.0) > 0.9
    assert voiced_share(p, 4.0, 8.0) < 0.1
    assert voiced_share(p, 8.0, 12.0) == 0.0  # silence: nothing active
