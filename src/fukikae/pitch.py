"""One pitch estimator for the pipeline (A0): a per-frame F0 track, its summaries, and a voice
register ("gender": the register as heard, never the character's gender in the story) read on
the median.

``f0_track`` reads the given spans of a vocals wav (any rate, width or channel count) and runs
an autocorrelation per 40 ms Hann-windowed frame (20 ms hop) between ``F0_MIN`` and ``F0_MAX``
(60 to 800 Hz). The autocorrelation is divided by the window's own, so a low voice's long period
isn't penalized for the window's taper (without it, rich voices under about 120 Hz read an
octave high). Only local maxima count, and the shortest lag whose peak is within ``PEAK_SHARE``
(80%) of the best is taken, which keeps a subharmonic from halving F0. A frame is voiced when
its best peak reaches ``VOICED_PEAK``; frames quieter than ``QUIET`` of the loudest decile of
the whole track are never voiced. Unvoiced frames are NaN, and each frame records the input
span it came from, so one track answers both a speaker's summary and each turn's. Frames are
read and processed ``_BATCH`` at a time in two passes (loudness, then F0), so memory stays
flat on a whole-episode span.

``classify`` bands a summary's median: ``m`` under ``male_below`` (``MALE_BELOW``, 165 Hz),
``f`` over ``female_above`` (``FEMALE_ABOVE``, 200 Hz), else ``u``, and ``u`` too under a
minimum of voiced speech (``SPEAKER_MIN_VOICED_S`` 1.0 s for a speaker,
``LINE_MIN_VOICED_S`` 0.4 s for a line). Its confidence is the distance in semitones from the
nearest band edge over ``CONFIDENCE_ST`` (4), capped at 1, and 0 for ``u``.

``cached_track`` keeps tracks in ``<root>/.fukikae/pitch-cache/<key>.npz``; the key is the
sha256 of the vocals file's bytes, the spans and ``VERSION``, which is bumped with every change
to the estimator. An unreadable cache file (truncated, empty, another layout) is recomputed.

CPU only, numpy and scipy only; this is pipeline code and never imports ``fukikae.eval``.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Gender = Literal["f", "m", "u"]

# In the cache key: bump it with any change to what f0_track returns (frame sizes, range,
# peak rule, voicing or loudness thresholds), or old tracks are served from the cache.
VERSION = 2

F0_MIN = 60
F0_MAX = 800
MALE_BELOW = 165
FEMALE_ABOVE = 200
SPEAKER_MIN_VOICED_S = 1.0
LINE_MIN_VOICED_S = 0.4
CONFIDENCE_ST = 4.0
REF_PITCH_ST = 4.0
REF_VOICED_MIN = 0.5
REF_SONG_SHARE = 0.5
BIMODAL_MIN_S = 5.0
BIMODAL_MIN_SHARE = 0.15

FRAME_S = 0.040
HOP_S = 0.020
# A frame is voiced when its normalized autocorrelation peak reaches this.
VOICED_PEAK = 0.5
# Frames quieter than this fraction of the loudest decile of the track's frames are skipped.
QUIET = 0.1
# Fewer voiced frames than this (0.2 s) give no median.
MIN_VOICED_FRAMES = 10
# A peak within this share of the best (window-corrected) peak at a shorter lag wins, which
# keeps a subharmonic from halving F0.
PEAK_SHARE = 0.8
# Frames per batch (about 5 MB of float64 frames at 16 kHz), which bounds memory on a long track.
_BATCH = 1024


@dataclass(frozen=True)
class F0Track:
    f0: np.ndarray  # Hz per 20 ms hop, NaN when unvoiced
    span_index: np.ndarray  # which input span each frame came from
    seconds: float  # seconds read over all spans
    span_seconds: np.ndarray  # seconds read per input span

    @property
    def spans(self) -> int:
        return len(self.span_seconds)


@dataclass(frozen=True)
class F0Summary:
    median: float | None
    p10: float | None
    p20: float | None
    p90: float | None
    voiced_s: float
    seconds: float
    voiced_fraction: float


@dataclass(frozen=True)
class GenderEstimate:
    gender: Gender
    confidence: float
    median: float | None
    voiced_s: float


def pcm_float(seg: np.ndarray) -> np.ndarray:
    """Integer PCM as float32 in [-1, 1]: signed by its maximum, unsigned (8-bit wav) centred
    on its midpoint first."""
    if seg.dtype.kind == "u":
        mid = (int(np.iinfo(seg.dtype).max) + 1) / 2
        return ((seg.astype(np.float32) - mid) / mid).astype(np.float32)
    return seg.astype(np.float32) / float(np.iinfo(seg.dtype).max)


def _mono(seg: np.ndarray) -> np.ndarray:
    """Samples as mono float32: integer PCM scaled before any mixdown, channels averaged."""
    if seg.dtype.kind in "iu":
        seg = pcm_float(seg)
    if seg.ndim > 1:
        seg = seg.mean(axis=1)
    return seg.astype(np.float32)


def _open(vocals: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    rate, data = wavfile.read(vocals, mmap=True)
    return int(rate), data


def _bounds(data: np.ndarray, rate: int, span: tuple[float, float]) -> tuple[int, int]:
    """Sample offsets of a span, clipped to the file; empty when reversed or outside it."""
    a = max(0, int(span[0] * rate))
    b = min(len(data), max(0, int(span[1] * rate)))
    return a, max(a, b)


def read_spans(vocals: Path, spans: Sequence[tuple[float, float]]) -> tuple[list[np.ndarray], int]:
    """Each span's samples as mono float32, in input order (an empty array for a span outside
    the file), read from a wav of any rate, width or channels."""
    rate, data = _open(vocals)
    pieces = []
    for span in spans:
        a, b = _bounds(data, rate, span)
        pieces.append(_mono(np.asarray(data[a:b])))
    return pieces, rate


def frames(x: np.ndarray, rate: int) -> np.ndarray:
    n, hop = int(FRAME_S * rate), int(HOP_S * rate)
    if len(x) < n:
        return np.empty((0, n), np.float32)
    count = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def _frame_count(length: int, rate: int) -> int:
    n, hop = int(FRAME_S * rate), int(HOP_S * rate)
    return 0 if length < n else 1 + (length - n) // hop


def _batches(data: np.ndarray, rate: int, a: int, b: int):
    """The frames of samples ``a`` to ``b`` as mean-removed float64 rows, ``_BATCH`` frames at
    a time, read from the (memory-mapped) file one batch at a time."""
    n, hop = int(FRAME_S * rate), int(HOP_S * rate)
    count = _frame_count(b - a, rate)
    for i in range(0, count, _BATCH):
        j = min(count, i + _BATCH)
        seg = _mono(np.asarray(data[a + i * hop : a + (j - 1) * hop + n]))
        fr = frames(seg, rate).astype(np.float64)
        fr -= fr.mean(axis=1, keepdims=True)
        yield fr


WINDOW_FLOOR = 1e-3


def _window_ac(n: int) -> np.ndarray:
    """The Hann window's own normalized autocorrelation, which a windowed frame's ACF is
    divided by so a long lag (a low voice's period) is not penalized for the window's taper."""
    w = np.hanning(n)
    wa = np.fft.irfft(np.abs(np.fft.rfft(w, 2 * n)) ** 2)[:n]
    return np.maximum(wa / wa[0], WINDOW_FLOOR)


def _frame_f0(fr: np.ndarray, rate: int) -> np.ndarray:
    """F0 in Hz per (mean-removed, float64) frame, NaN where unvoiced."""
    n = fr.shape[1]
    spec = np.fft.rfft(fr * np.hanning(n), 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac /= np.maximum(ac[:, :1], 1e-12)
    ac /= _window_ac(n)[None, :]
    lo, hi = max(1, int(rate / F0_MAX)), min(n - 2, int(rate / F0_MIN))
    window = ac[:, lo : hi + 1]
    # only local maxima count, so a slope still falling from lag 0 is never a period
    peak = (window >= ac[:, lo - 1 : hi]) & (window >= ac[:, lo + 1 : hi + 2])
    window = np.where(peak, window, -np.inf)
    best = window.max(axis=1)
    lag = lo + np.argmax(window >= PEAK_SHARE * best[:, None], axis=1)
    return np.where(best >= VOICED_PEAK, rate / lag, np.nan)


def f0_track(vocals: Path, spans: Sequence[tuple[float, float]]) -> F0Track:
    """The F0 track of ``spans`` (in input order) of ``vocals``. Two passes over the frames,
    ``_BATCH`` at a time: the first measures each frame's RMS for the loudness threshold, the
    second runs the autocorrelation on the frames above it, so a whole-episode span never
    holds more than one batch of frames."""
    if not spans:
        return F0Track(np.empty(0), np.empty(0, np.int64), 0.0, np.empty(0))
    rate, data = _open(vocals)
    bounds = [_bounds(data, rate, span) for span in spans]
    span_seconds = np.array([(b - a) / rate for a, b in bounds], np.float64)
    counts = [_frame_count(b - a, rate) for a, b in bounds]
    index = np.concatenate([np.full(c, i, np.int64) for i, c in enumerate(counts)])
    f0 = np.full(len(index), np.nan)
    if len(index):
        rms = np.concatenate(
            [np.sqrt((fr**2).mean(axis=1)) for a, b in bounds for fr in _batches(data, rate, a, b)]
        )
        loud = float(np.quantile(rms, 0.9))
        floor = max(1e-4, QUIET * loud) if loud > 0 else math.inf
        at = 0
        for a, b in bounds:
            for fr in _batches(data, rate, a, b):
                rows = np.flatnonzero(rms[at : at + len(fr)] > floor)
                if len(rows):
                    f0[at + rows] = _frame_f0(fr[rows], rate)
                at += len(fr)
    return F0Track(f0, index, float(span_seconds.sum()), span_seconds)


def _summary(f0: np.ndarray, seconds: float) -> F0Summary:
    voiced = f0[~np.isnan(f0)]
    fraction = len(voiced) / len(f0) if len(f0) else 0.0
    voiced_s = len(voiced) * HOP_S
    if len(voiced) < MIN_VOICED_FRAMES:
        return F0Summary(None, None, None, None, voiced_s, seconds, fraction)
    p10, p20, med, p90 = (float(v) for v in np.percentile(voiced, [10, 20, 50, 90]))
    return F0Summary(med, p10, p20, p90, voiced_s, seconds, fraction)


def summarize(track: F0Track, span: int | None = None) -> F0Summary:
    """The summary of the whole track, or of one input span's frames."""
    if span is None:
        return _summary(track.f0, track.seconds)
    return _summary(track.f0[track.span_index == span], float(track.span_seconds[span]))


def classify(
    summary: F0Summary,
    male_below: float = MALE_BELOW,
    female_above: float = FEMALE_ABOVE,
    min_voiced_s: float = SPEAKER_MIN_VOICED_S,
) -> GenderEstimate:
    med = summary.median
    if med is None or summary.voiced_s < min_voiced_s:
        return GenderEstimate("u", 0.0, med, summary.voiced_s)
    if med < male_below:
        gender, edge = "m", male_below
    elif med > female_above:
        gender, edge = "f", female_above
    else:
        return GenderEstimate("u", 0.0, med, summary.voiced_s)
    confidence = min(1.0, abs(12 * math.log2(med / edge)) / CONFIDENCE_ST)
    return GenderEstimate(gender, confidence, med, summary.voiced_s)


def turn_classes(
    track: F0Track, bands: tuple[float, float], min_voiced_s: float = LINE_MIN_VOICED_S
) -> list[GenderEstimate]:
    """One estimate per input span, each on its own median."""
    lo, hi = bands
    return [classify(summarize(track, i), lo, hi, min_voiced_s) for i in range(track.spans)]


def band_seconds(track: F0Track, bands: tuple[float, float]) -> dict[str, float]:
    """Seconds of turns (input spans) in each class, ``m``, ``u`` and ``f``."""
    out = {"m": 0.0, "u": 0.0, "f": 0.0}
    for i, est in enumerate(turn_classes(track, bands)):
        out[est.gender] += float(track.span_seconds[i])
    return out


def majority_register(track: F0Track, bands: tuple[float, float]) -> tuple[Gender, F0Summary]:
    """The majority class of the track's turns by seconds, between ``m`` and ``f``, and the
    summary of that class's turns only; ``u`` with the whole track's summary when no turn is
    classified or the two tie."""
    classes = turn_classes(track, bands)
    secs = {"m": 0.0, "f": 0.0}
    for i, est in enumerate(classes):
        if est.gender != "u":
            secs[est.gender] += float(track.span_seconds[i])
    if secs["m"] == secs["f"]:
        return "u", summarize(track)
    gender: Gender = "m" if secs["m"] > secs["f"] else "f"
    chosen = np.array([c.gender == gender for c in classes])
    mask = chosen[track.span_index] if len(track.span_index) else np.zeros(0, bool)
    seconds = float(track.span_seconds[chosen].sum())
    return gender, _summary(track.f0[mask], seconds)


def bimodal(seconds: Mapping[str, float]) -> bool:
    """Whether the minor of ``m`` and ``f`` holds at least ``BIMODAL_MIN_S`` seconds and
    ``BIMODAL_MIN_SHARE`` of the classified (``m`` plus ``f``) seconds."""
    m, f = float(seconds.get("m", 0.0)), float(seconds.get("f", 0.0))
    minor, classified = min(m, f), m + f
    return classified > 0 and minor >= BIMODAL_MIN_S and minor / classified >= BIMODAL_MIN_SHARE


# -- cache ------------------------------------------------------------------------------------

_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    """sha256 of the file's bytes, memoized per process by (path, size, mtime)."""
    st = path.stat()
    memo = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    hit = _DIGESTS.get(memo)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _DIGESTS[memo] = h.hexdigest()
    return _DIGESTS[memo]


def cache_dir(root: Path) -> Path:
    return Path(root) / ".fukikae" / "pitch-cache"


def track_key(vocals: Path, spans: Sequence[tuple[float, float]]) -> str:
    payload = {
        "vocals": _file_digest(Path(vocals)),
        "spans": [[float(a), float(b)] for a, b in spans],
        "version": VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_track(root: Path, vocals: Path, spans: Sequence[tuple[float, float]]) -> F0Track:
    """``f0_track`` through ``<root>/.fukikae/pitch-cache``."""
    from fukikae.artifacts import atomic_path

    path = cache_dir(root) / f"{track_key(vocals, spans)}.npz"
    try:
        with np.load(path) as z:
            return F0Track(z["f0"], z["span_index"], float(z["seconds"]), z["span_seconds"])
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile):
        pass
    track = f0_track(vocals, spans)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp, tmp.open("wb") as fh:
        np.savez(
            fh, f0=track.f0, span_index=track.span_index,
            seconds=np.float64(track.seconds), span_seconds=track.span_seconds,
        )  # fmt: skip
    return track
