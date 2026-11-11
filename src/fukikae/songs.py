"""Theme-song detection from the separation stems, independent of the tag stage.

SenseVoice has no singing tag, and the ASR path transcribes and translates an opening theme's
lyrics as dialogue, so the mix needs its own evidence of where a song plays. An opening or
ending theme is a long stretch where the music bed is at least as loud as the voice
(``background`` within ``MUSIC_MARGIN_DB`` of ``vocals``) and the voice is on nearly all the
time (``ACTIVE_SHARE`` of 100 ms frames above ``ACTIVE_DB``); dialogue over music has gaps
between lines, and dialogue over a quiet bed has a bed far below the voice.

A second counts as song-like when both hold. In the first ``head_s`` and the last ``tail_s``
seconds of the episode (where openings, after any cold open, and endings sit), the best
``window_s`` window is taken; it is a candidate when at least ``min_score`` of its seconds are
song-like. Energy alone also passes a battle (shouting nearly all the time over loud music:
Fairy Tail E13 at 611 s scores 0.86), so a candidate must also be sung: at least
``voiced_share`` of the active 40 ms frames of ``vocals_16k.wav`` in the window are strongly
periodic (normalized autocorrelation peak at least ``voiced_strength`` between 80 and 1000 Hz).
Songs measured 0.77 to 0.94, that battle 0.53, dialogue 0.32 to 0.54. An accepted span grows
while the next ``grow_s`` seconds on either side are at least ``grow_share`` song-like (an intro
or outro with fewer vocals), up to ``max_song_s``, and is padded by ``pad_s``. Every candidate,
accepted or not, is reported with its scores. The settings are ``[mix.songs]``.

Measured on the separation stems (``docs/superpowers/ledgers/2026-11-08-character-cards.md``,
Task 7; ``scripts/song_scores.py`` reproduces it): Fairy Tail E11-E14 openings and endings
found within a few seconds of the lyrics, including E13's opening after a 3.5 minute cold open;
Kaze, Kaoru's opening on E01-E03; no span on The Big Bang Theory (en, de) or Grid. A
mid-episode insert song is not a theme and is left to the tag stage and subtitles.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

FRAME_S = 0.1
VOICED_FRAME_S = 0.04
VOICED_HOP_S = 0.02
VOICED_ACTIVE_DB = -45.0
VERSION = 2  # 2: the voicing check


@dataclass(frozen=True)
class SongSettings:
    """``[mix.songs]``; see the module docstring for what each does."""

    active_db: float = -40.0
    active_share: float = 0.6
    music_margin_db: float = 3.0
    window_s: int = 80
    min_score: float = 0.5
    voiced_strength: float = 0.7
    voiced_share: float = 0.7
    grow_s: int = 5
    grow_share: float = 0.4
    max_song_s: int = 100
    head_s: int = 360
    tail_s: int = 240
    pad_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("active_share", "min_score", "voiced_strength", "voiced_share", "grow_share"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, not {v!r}")
        for name in ("window_s", "grow_s", "max_song_s", "head_s", "tail_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a whole number of seconds >= 1, not {v!r}")
        if self.max_song_s < self.window_s:
            raise ValueError("max_song_s must be at least window_s")
        for name in ("active_db", "music_margin_db", "pad_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        if self.pad_s < 0:
            raise ValueError("pad_s must be >= 0")


SONG_KEYS = tuple(f.name for f in fields(SongSettings))


@dataclass(frozen=True)
class SongSpan:
    start: float
    end: float
    source: str
    score: float | None = None


@dataclass(frozen=True)
class Candidate:
    """The best window of one region: its energy ``score``, its ``voiced`` share (None when
    not measured), whether it was ``accepted`` and why not (``score``, ``voiced``)."""

    region: str
    start: float
    end: float
    score: float
    voiced: float | None
    accepted: bool
    reason: str


def detector_constants(s: SongSettings | None = None) -> dict[str, Any]:
    """Everything the detector's output depends on, for the mix stage's key."""
    return {
        **asdict(s or SongSettings()),
        "frame_s": FRAME_S,
        "voiced_frame_s": VOICED_FRAME_S,
        "voiced_hop_s": VOICED_HOP_S,
        "voiced_active_db": VOICED_ACTIVE_DB,
        "version": VERSION,
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


def frame_features(
    background: Path, vocals: Path, active_db: float = SongSettings.active_db
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per second: the bed's dBFS, the voice's dBFS and the share of 100 ms frames with the
    voice on (above ``active_db``)."""
    bg = _frame_db(background)
    vo = _frame_db(vocals)
    per = int(round(1 / FRAME_S))
    seconds = min(len(bg), len(vo)) // per
    bg, vo = bg[: seconds * per].reshape(seconds, per), vo[: seconds * per].reshape(seconds, per)

    def power_mean(x: np.ndarray) -> np.ndarray:
        return 10 * np.log10((10 ** (x / 10)).mean(axis=1))

    return power_mean(bg), power_mean(vo), (vo > active_db).mean(axis=1)


def voiced_share(
    vocals_16k: Path, start: float, end: float, strength: float = SongSettings.voiced_strength
) -> float:
    """Share of the active 40 ms frames (above ``VOICED_ACTIVE_DB``) of ``vocals_16k`` in
    [start, end) whose normalized autocorrelation peaks at ``strength`` or more between 80 and
    1000 Hz: sung or spoken vowels are periodic, shouting and noise much less. 0.0 when no
    frame is active."""
    from scipy.io import wavfile

    rate, data = wavfile.read(vocals_16k, mmap=True)
    scale = float(np.iinfo(data.dtype).max) + 1.0 if np.issubdtype(data.dtype, np.integer) else 1.0
    seg = np.asarray(data[int(start * rate) : int(end * rate)], dtype=np.float64) / scale
    if seg.ndim > 1:
        seg = seg.mean(axis=1)
    n, hop = int(VOICED_FRAME_S * rate), int(VOICED_HOP_S * rate)
    if len(seg) < n:
        return 0.0
    idx = np.arange(0, len(seg) - n + 1, hop)[:, None] + np.arange(n)[None, :]
    frames = seg[idx] * np.hanning(n)
    rms = np.sqrt((frames**2).mean(axis=1))
    frames = frames[rms >= 10 ** (VOICED_ACTIVE_DB / 20)]
    if len(frames) == 0:
        return 0.0
    spec = np.fft.rfft(frames, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(rate / 1000), int(rate / 80)
    return float((ac[:, lo:hi].max(axis=1) >= strength).mean())


def song_like_mask(
    bg_db: np.ndarray, vo_db: np.ndarray, active: np.ndarray, s: SongSettings
) -> np.ndarray:
    """Per second, whether it is song-like: the voice on (``active_share`` of its 100 ms
    frames) and the bed within ``music_margin_db`` of the voice. The windows of
    `find_candidates` are scored on this; the A1 reference gate reads it per turn."""
    return (active >= s.active_share) & (bg_db >= vo_db - s.music_margin_db)


def song_like_seconds(
    background: Path, vocals: Path, settings: SongSettings | None = None
) -> np.ndarray:
    """`song_like_mask` over the whole episode, from the separation stems."""
    s = settings or SongSettings()
    return song_like_mask(*frame_features(background, vocals, s.active_db), s)


def find_candidates(
    bg_db: np.ndarray,
    vo_db: np.ndarray,
    active: np.ndarray,
    s: SongSettings,
    voiced: Callable[[float, float], float] | None = None,
) -> list[Candidate]:
    """The best window of the head and of the tail region, scored; ``voiced(start, end)``
    measures the voicing share of a window that passes on energy (no voicing check when
    None)."""
    good = song_like_mask(bg_db, vo_db, active, s).astype(np.float64)
    dur, w = len(good), s.window_s
    if dur < w:
        return []
    score = np.convolve(good, np.ones(w), "valid") / w  # score[i]: the window starting at i
    out: list[Candidate] = []
    for region, (lo, hi) in (
        ("head", (0, min(s.head_s, dur))),
        ("tail", (max(0, dur - s.tail_s), dur)),
    ):
        if hi - lo < w:
            continue
        i = lo + int(np.argmax(score[lo : hi - w + 1]))
        best = round(float(score[i]), 3)
        if best < s.min_score:
            out.append(Candidate(region, float(i), float(i + w), best, None, False, "score"))
            continue
        v = None if voiced is None else round(voiced(float(i), float(i + w)), 3)
        if v is not None and v < s.voiced_share:
            out.append(Candidate(region, float(i), float(i + w), best, v, False, "voiced"))
            continue
        a, b = i, i + w
        while (
            a > 0 and b - a < s.max_song_s and good[max(0, a - s.grow_s) : a].mean() >= s.grow_share
        ):
            a -= 1
        while b < dur and b - a < s.max_song_s and good[b : b + s.grow_s].mean() >= s.grow_share:
            b += 1
        out.append(
            Candidate(region, max(0.0, a - s.pad_s), min(float(dur), b + s.pad_s), best, v, True,
                      "accepted")
        )  # fmt: skip
    return out


def _spans(cands: list[Candidate]) -> list[SongSpan]:
    merged: list[SongSpan] = []
    for c in sorted((c for c in cands if c.accepted), key=lambda c: c.start):
        if merged and c.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = SongSpan(
                last.start, max(last.end, c.end), "detector", max(last.score or 0, c.score)
            )
        else:
            merged.append(SongSpan(c.start, c.end, "detector", c.score))
    return merged


def theme_songs(
    bg_db: np.ndarray,
    vo_db: np.ndarray,
    active: np.ndarray,
    *,
    settings: SongSettings | None = None,
    voiced: Callable[[float, float], float] | None = None,
) -> list[SongSpan]:
    """Opening and ending theme spans from per-second features (see the module docstring)."""
    return _spans(find_candidates(bg_db, vo_db, active, settings or SongSettings(), voiced))


def detect_theme_songs(
    background: Path, vocals: Path, vocals_16k: Path, settings: SongSettings | None = None
) -> tuple[list[SongSpan], list[Candidate]]:
    """(theme spans, every candidate) from the separation stems."""
    s = settings or SongSettings()
    feats = frame_features(background, vocals, s.active_db)
    cands = find_candidates(
        *feats, s, voiced=lambda a, b: voiced_share(vocals_16k, a, b, s.voiced_strength)
    )
    return _spans(cands), cands
