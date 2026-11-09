"""Theme-song detection from the separation stems, independent of the tag stage.

SenseVoice has no singing tag, and the ASR path transcribes and translates an opening theme's
lyrics as dialogue, so the mix needs its own evidence of where a song plays. An opening or
ending theme is a long stretch where the music bed is at least as loud as the voice
(``background`` within ``MUSIC_MARGIN_DB`` of ``vocals``) and the voice is on nearly all the
time (``ACTIVE_SHARE`` of 100 ms frames above ``ACTIVE_DB``); dialogue over music has gaps
between lines, and dialogue over a quiet bed has a bed far below the voice.

A second counts as song-like when both hold. In the first ``HEAD_S`` and the last ``TAIL_S``
seconds of the episode (where openings, after any cold open, and endings sit), the best
``WINDOW_S`` window is taken; it is a theme song when at least ``MIN_SCORE`` of its seconds are
song-like. The span then grows while the next ``GROW_S`` seconds on either side are at least
``GROW_SHARE`` song-like (an intro or outro with fewer vocals), up to ``MAX_SONG_S``, and is
padded by ``PAD_S``.

Measured on the separation stems (``docs/superpowers/ledgers/2026-11-08-character-cards.md``,
Task 7): Fairy Tail E11-E14 openings and endings found within a few seconds of the lyrics
(scores 0.69 to 0.94, including E13's opening after a 3.5 minute cold open), Kaze, Kaoru's
opening found on E01-E03, and no span on The Big Bang Theory (en, de; best 0.22) or Grid
(0.04). A mid-episode insert song is not a theme and is left to the tag stage and subtitles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FRAME_S = 0.1
ACTIVE_DB = -40.0
ACTIVE_SHARE = 0.6
MUSIC_MARGIN_DB = 3.0
WINDOW_S = 80
MIN_SCORE = 0.5
GROW_S = 5
GROW_SHARE = 0.4
MAX_SONG_S = 100
HEAD_S = 360
TAIL_S = 240
PAD_S = 1.0
VERSION = 1


@dataclass(frozen=True)
class SongSpan:
    start: float
    end: float
    source: str
    score: float | None = None


def detector_constants() -> dict[str, Any]:
    """Every constant the detector uses, for the mix stage's key."""
    return {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, int | float)
    }


def _frame_db(path: Path) -> np.ndarray:
    """dBFS of each FRAME_S frame of the wav (channels averaged), read in blocks."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path, mmap=True)
    n = max(1, int(round(rate * FRAME_S)))
    frames = len(data) // n
    out = np.empty(frames, dtype=np.float64)
    scale = float(np.iinfo(data.dtype).max) + 1.0 if np.issubdtype(data.dtype, np.integer) else 1.0
    block = 600  # frames per read
    for i in range(0, frames, block):
        k = min(block, frames - i)
        seg = np.asarray(data[i * n : (i + k) * n], dtype=np.float64) / scale
        if seg.ndim > 1:
            seg = seg.mean(axis=1)
        power = (seg.reshape(k, n) ** 2).mean(axis=1)
        out[i : i + k] = 10 * np.log10(power + 1e-12)
    return out


def frame_features(background: Path, vocals: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per second: the bed's dBFS, the voice's dBFS and the share of 100 ms frames with the
    voice on."""
    bg = _frame_db(background)
    vo = _frame_db(vocals)
    per = int(round(1 / FRAME_S))
    seconds = min(len(bg), len(vo)) // per
    bg, vo = bg[: seconds * per].reshape(seconds, per), vo[: seconds * per].reshape(seconds, per)

    def power_mean(x: np.ndarray) -> np.ndarray:
        return 10 * np.log10((10 ** (x / 10)).mean(axis=1))

    return power_mean(bg), power_mean(vo), (vo > ACTIVE_DB).mean(axis=1)


def theme_songs(bg_db: np.ndarray, vo_db: np.ndarray, active: np.ndarray) -> list[SongSpan]:
    """Opening and ending theme spans from per-second features (see the module docstring)."""
    good = ((active >= ACTIVE_SHARE) & (bg_db >= vo_db - MUSIC_MARGIN_DB)).astype(np.float64)
    dur = len(good)
    if dur < WINDOW_S:
        return []
    score = np.convolve(good, np.ones(WINDOW_S), "valid") / WINDOW_S  # score[i]: window at i
    spans: list[SongSpan] = []
    for lo, hi in ((0, min(HEAD_S, dur)), (max(0, dur - TAIL_S), dur)):
        if hi - lo < WINDOW_S:
            continue
        i = lo + int(np.argmax(score[lo : hi - WINDOW_S + 1]))
        best = float(score[i])
        if best < MIN_SCORE:
            continue
        s, e = i, i + WINDOW_S
        while s > 0 and e - s < MAX_SONG_S and good[max(0, s - GROW_S) : s].mean() >= GROW_SHARE:
            s -= 1
        while e < dur and e - s < MAX_SONG_S and good[e : e + GROW_S].mean() >= GROW_SHARE:
            e += 1
        spans.append(
            SongSpan(max(0.0, s - PAD_S), min(float(dur), e + PAD_S), "detector", round(best, 3))
        )
    merged: list[SongSpan] = []
    for sp in sorted(spans, key=lambda x: x.start):
        if merged and sp.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = SongSpan(
                last.start, max(last.end, sp.end), "detector", max(last.score or 0, sp.score or 0)
            )
        else:
            merged.append(sp)
    return merged


def detect_theme_songs(background: Path, vocals: Path) -> list[SongSpan]:
    return theme_songs(*frame_features(background, vocals))
