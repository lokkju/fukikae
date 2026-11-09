"""Choosing a character's voice from the original audio. Task 6 needs only a gender estimate,
for the stock pools; reference selection and clone cards come with the cloning backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

FRAME_S = 0.040
HOP_S = 0.020
F0_MIN = 70.0
F0_MAX = 400.0
# Defaults; a profile can widen the unknown band between them ([tts] gender, the anime one does).
MALE_BELOW = 165.0
FEMALE_ABOVE = 180.0
# The F0 percentile read: shouting lifts the median, the low end of a voice moves less.
F0_PERCENTILE = 20
# A frame is voiced when its normalized autocorrelation peak reaches this.
VOICED_PEAK = 0.5
# Frames quieter than this fraction of the loudest decile of the speaker's frames are skipped.
QUIET = 0.1
MIN_VOICED_FRAMES = 10
# Seconds of speech read per character; more doesn't move a median.
MAX_SECONDS = 120.0


def _read(vocals: Path, spans: list[tuple[float, float]]) -> tuple[list[np.ndarray], int]:
    """The spans' samples as mono float32, read from a wav of any rate, width or channels."""
    from scipy.io import wavfile

    rate, data = wavfile.read(vocals, mmap=True)
    pieces: list[np.ndarray] = []
    total = 0.0
    for start, end in sorted(spans):
        if total >= MAX_SECONDS:
            break
        end = min(end, start + MAX_SECONDS - total)
        seg = np.asarray(data[max(0, int(start * rate)) : max(0, int(end * rate))])
        if seg.size == 0:
            continue
        if seg.dtype.kind in "iu":  # scale integer PCM before any mixdown
            seg = seg.astype(np.float32) / float(np.iinfo(seg.dtype.str).max)
        if seg.ndim > 1:
            seg = seg.mean(axis=1)
        pieces.append(seg.astype(np.float32))
        total += len(seg) / rate
    return pieces, int(rate)


def _frames(x: np.ndarray, rate: int) -> np.ndarray:
    n, hop = int(FRAME_S * rate), int(HOP_S * rate)
    if len(x) < n:
        return np.empty((0, n), np.float32)
    count = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def speaker_f0(vocals: Path, spans: list[tuple[float, float]]) -> tuple[float | None, float]:
    """(20th-percentile F0 in Hz over the voiced frames, or None when too few are voiced;
    seconds read).

    Autocorrelation per 40 ms frame (20 ms hop) between 70 and 400 Hz. The shortest lag whose
    peak is within 90% of the best is taken, which keeps a strong subharmonic from halving F0.
    """
    pieces, rate = _read(vocals, spans)
    seconds = sum(len(p) for p in pieces) / rate if pieces else 0.0
    frames = [_frames(p, rate) for p in pieces]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None, seconds
    fr = np.concatenate(frames).astype(np.float64)
    fr -= fr.mean(axis=1, keepdims=True)
    rms = np.sqrt((fr**2).mean(axis=1))
    loud = np.quantile(rms, 0.9)
    keep = (rms > max(1e-4, QUIET * loud)) if loud > 0 else np.zeros(len(fr), bool)
    fr = fr[keep]
    if len(fr) < MIN_VOICED_FRAMES:
        return None, seconds
    n = fr.shape[1]
    spec = np.fft.rfft(fr * np.hanning(n), 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac /= np.maximum(ac[:, :1], 1e-12)
    lo, hi = int(rate / F0_MAX), min(n - 2, int(rate / F0_MIN))
    window = ac[:, lo : hi + 1]
    # only local maxima count, so a slope still falling from lag 0 is never a period
    peak = (window >= ac[:, lo - 1 : hi]) & (window >= ac[:, lo + 1 : hi + 2])
    window = np.where(peak, window, -np.inf)
    best = window.max(axis=1)
    voiced = best >= VOICED_PEAK
    if voiced.sum() < MIN_VOICED_FRAMES:
        return None, seconds
    window, best = window[voiced], best[voiced]
    lag = lo + np.argmax(window >= 0.9 * best[:, None], axis=1)
    return float(np.percentile(rate / lag, F0_PERCENTILE)), seconds


def gender_of(
    f0: float | None, male_below: float = MALE_BELOW, female_above: float = FEMALE_ABOVE
) -> Literal["f", "m", "u"]:
    if f0 is None:
        return "u"
    if f0 < male_below:
        return "m"
    if f0 > female_above:
        return "f"
    return "u"


def estimate_gender(
    vocals: Path,
    spans: list[tuple[float, float]],
    male_below: float = MALE_BELOW,
    female_above: float = FEMALE_ABOVE,
) -> Literal["f", "m", "u"]:
    """``m`` under ``male_below`` (165 Hz) 20th-percentile F0, ``f`` over ``female_above``
    (180 Hz), else (or with too little voiced speech) ``u``."""
    return gender_of(speaker_f0(vocals, spans)[0], male_below, female_above)
