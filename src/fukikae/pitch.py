"""One pitch estimator for the pipeline (A0): a per-frame F0 track, its summaries, and a voice
register ("gender": the register as heard, never the character's gender in the story) read on
the median.

``f0_track`` reads the given spans of a vocals wav (any rate, width or channel count) and runs
an autocorrelation per 40 ms Hann-windowed frame (20 ms hop) between ``F0_MIN`` and ``F0_MAX``
(60 to 800 Hz). The autocorrelation is divided by the window's own, so a low voice's long period
isn't penalized for the window's taper (without it, rich voices under about 120 Hz read an
octave high). Only local maxima count, and the shortest lag whose peak is within ``PEAK_SHARE``
(80%) of the best is taken, which keeps a subharmonic from halving F0. A frame is voiced when
its best peak reaches ``VOICED_PEAK`` and the autocorrelation dips to ``VALLEY_MAX`` (0) or
below before that lag, which keeps low rumble from reading as a voice; frames quieter than
``QUIET`` of the loudest decile of their group are never voiced (``loud`` records which
frames passed). The group is the whole track by default; ``groups`` gives each input span one,
so a reference candidate (the turns it is made of) is read against its own loudness and its
verdict never depends on which other spans share the track. Unvoiced frames are NaN, and each
frame records the input span it came from, so one track answers both a speaker's summary and
each turn's (``summarize``) or each group's (``summarize_group``). A summary's
``loud_voiced_fraction`` is the share of loud frames that are voiced: pauses inside speech
are quiet and don't count against it, while loud noise, whispers and effects do. Frames are
read and processed ``_BATCH`` at a time in two passes (loudness, then F0), so memory stays
flat on a whole-episode span. ``stability`` and ``is_sung`` tell held notes from speech's
gliding syllables (A1 gate 3).

``classify`` bands a summary's median: ``m`` under ``male_below`` (``MALE_BELOW``, 165 Hz),
``f`` over ``female_above`` (``FEMALE_ABOVE``, 200 Hz), else ``u``, and ``u`` too under a
minimum of voiced speech (``SPEAKER_MIN_VOICED_S`` 1.0 s for a speaker,
``LINE_MIN_VOICED_S`` 0.4 s for a line). Its confidence is the distance in semitones from the
nearest band edge over ``CONFIDENCE_ST`` (4), capped at 1, and 0 for ``u``.

``cached_track`` keeps tracks in ``<root>/.fukikae/pitch-cache/<key>.npz``; the key is the
sha256 of the vocals file's bytes, the spans, the groups and ``VERSION``, which is bumped
with every change to the estimator. An unreadable cache file (truncated, empty, another
layout) is recomputed.

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
VERSION = 4

F0_MIN = 60
F0_MAX = 800
MALE_BELOW = 165
FEMALE_ABOVE = 200
SPEAKER_MIN_VOICED_S = 1.0
LINE_MIN_VOICED_S = 0.4
CONFIDENCE_ST = 4.0
REF_PITCH_ST = 4.0
# A reference candidate under this share of voiced loud frames is refused (A1 gate 2). The
# spec's 0.5 was measured over every frame, pauses included; over loud frames, ground-truth
# dialogue lines of 3 s or more on the tuning sets (Fairy Tail E11-E13, The Big Bang Theory
# S01E01-E02, Kaze S01E01-E03) fall under 0.35 in 0.6% at most (BBT; none on the other two),
# while whispered dialogue (BBT S01E02's night scene) reads 0.13 to 0.30.
REF_VOICED_MIN = 0.35
REF_SONG_SHARE = 0.5
# Sung-ness (A1 gate 3): a candidate is sung when the median change between consecutive voiced
# 20 ms frames is at most SUNG_JITTER_ST semitones and its median voiced run lasts at least
# SUNG_RUN_S seconds (held notes, not gliding syllables). On the tuning sets' theme-song turns
# against ground-truth dialogue over a music bed (song mask share at least 0.5): 91% of song
# turns sung, no dialogue line (0 of 18; 6 of 81 at a mask share of 0.2 or more).
SUNG_JITTER_ST = 0.4
SUNG_RUN_S = 0.2
BIMODAL_MIN_S = 5.0
BIMODAL_MIN_SHARE = 0.15

FRAME_S = 0.040
HOP_S = 0.020
# A frame is voiced when its normalized autocorrelation peak reaches this.
VOICED_PEAK = 0.5
# Frames quieter than this fraction of the loudest decile of their group's frames are skipped.
QUIET = 0.1
# Fewer voiced frames than this (0.2 s) give no median.
MIN_VOICED_FRAMES = 10
# A peak within this share of the best (window-corrected) peak at a shorter lag wins, which
# keeps a subharmonic from halving F0.
PEAK_SHARE = 0.8
# A period's peak counts only when the (window-corrected) autocorrelation dips to this or lower
# somewhere between lag 0 and that peak: a voice anti-correlates with itself within one period
# (real voiced frames on Fairy Tail and Sol Bianca dip to -0.32 or lower in 99% of frames), while
# low rumble (brown noise) stays positive at every short lag and, without the check, read as
# voiced at about 270 Hz in 17% of its frames.
VALLEY_MAX = 0.0
# Frames per batch (about 5 MB of float64 frames at 16 kHz), which bounds memory on a long track.
_BATCH = 1024


@dataclass(frozen=True)
class F0Track:
    f0: np.ndarray  # Hz per 20 ms hop, NaN when unvoiced
    span_index: np.ndarray  # which input span each frame came from
    seconds: float  # seconds read over all spans
    span_seconds: np.ndarray  # seconds read per input span
    loud: np.ndarray | None = None  # per frame: above its group's loudness floor
    groups: np.ndarray | None = None  # per input span: its loudness group (None: one group)

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
    loud_voiced_fraction: float = 0.0  # voiced frames over loud frames (0 with none loud)


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


# A guard only: over the lags searched (to rate / F0_MIN, 266 at 16 kHz in a 640-sample frame)
# the Hann window's autocorrelation stays above 0.3, so the correction's gain is at most about
# 3.3 and this floor never binds. It keeps a longer search range from dividing by ~0.
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
    # the lowest the autocorrelation goes between lag 1 and the chosen lag (the valley)
    valley = np.minimum.accumulate(ac[:, 1 : hi + 1], axis=1)[np.arange(len(lag)), lag - 1]
    voiced = (best >= VOICED_PEAK) & (valley <= VALLEY_MAX)
    return np.where(voiced, rate / lag, np.nan)


def f0_track(
    vocals: Path, spans: Sequence[tuple[float, float]], groups: Sequence[int] | None = None
) -> F0Track:
    """The F0 track of ``spans`` (in input order) of ``vocals``. Two passes over the frames,
    ``_BATCH`` at a time: the first measures each frame's RMS for the loudness threshold (per
    group of ``groups``, one per span; the whole track when None), the second runs the
    autocorrelation on the frames above it, so a whole-episode span never holds more than one
    batch of frames."""
    group_arr = None if groups is None else np.asarray(list(groups), np.int64)
    if group_arr is not None and len(group_arr) != len(spans):
        raise ValueError(f"{len(group_arr)} groups for {len(spans)} spans")
    if not spans:
        return F0Track(
            np.empty(0), np.empty(0, np.int64), 0.0, np.empty(0), np.empty(0, bool), group_arr
        )
    rate, data = _open(vocals)
    bounds = [_bounds(data, rate, span) for span in spans]
    span_seconds = np.array([(b - a) / rate for a, b in bounds], np.float64)
    counts = [_frame_count(b - a, rate) for a, b in bounds]
    index = np.concatenate([np.full(c, i, np.int64) for i, c in enumerate(counts)])
    f0 = np.full(len(index), np.nan)
    loud = np.zeros(len(index), bool)
    if len(index):
        rms = np.concatenate(
            [np.sqrt((fr**2).mean(axis=1)) for a, b in bounds for fr in _batches(data, rate, a, b)]
        )
        frame_group = np.zeros(len(index), np.int64) if group_arr is None else group_arr[index]
        for g in np.unique(frame_group):
            mine = frame_group == g
            top = float(np.quantile(rms[mine], 0.9))
            floor = max(1e-4, QUIET * top) if top > 0 else math.inf
            loud[mine] = rms[mine] > floor
        at = 0
        for a, b in bounds:
            for fr in _batches(data, rate, a, b):
                rows = np.flatnonzero(loud[at : at + len(fr)])
                if len(rows):
                    f0[at + rows] = _frame_f0(fr[rows], rate)
                at += len(fr)
    return F0Track(f0, index, float(span_seconds.sum()), span_seconds, loud, group_arr)


def _summary(f0: np.ndarray, seconds: float, loud: np.ndarray | None = None) -> F0Summary:
    voiced = f0[~np.isnan(f0)]
    fraction = len(voiced) / len(f0) if len(f0) else 0.0
    n_loud = int(loud.sum()) if loud is not None else len(f0)
    loud_fraction = len(voiced) / n_loud if n_loud else 0.0
    voiced_s = len(voiced) * HOP_S
    if len(voiced) < MIN_VOICED_FRAMES:
        return F0Summary(None, None, None, None, voiced_s, seconds, fraction, loud_fraction)
    p10, p20, med, p90 = (float(v) for v in np.percentile(voiced, [10, 20, 50, 90]))
    return F0Summary(med, p10, p20, p90, voiced_s, seconds, fraction, loud_fraction)


def _masked(track: F0Track, mask: np.ndarray, seconds: float) -> F0Summary:
    loud = track.loud[mask] if track.loud is not None else None
    return _summary(track.f0[mask], seconds, loud)


def summarize(track: F0Track, span: int | None = None) -> F0Summary:
    """The summary of the whole track, or of one input span's frames."""
    if span is None:
        return _summary(track.f0, track.seconds, track.loud)
    return _masked(track, track.span_index == span, float(track.span_seconds[span]))


def summarize_group(track: F0Track, group: int) -> F0Summary:
    """The summary of the frames of every input span in ``group`` (``F0Track.groups``)."""
    if track.groups is None:
        raise ValueError("the track has no groups")
    members = np.flatnonzero(track.groups == group)
    mask = np.isin(track.span_index, members)
    return _masked(track, mask, float(track.span_seconds[members].sum()))


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


@dataclass(frozen=True)
class Stability:
    """How steady a track's voiced frames are: the median absolute change in semitones
    between consecutive voiced frames (None with no such pair), and the median length in
    seconds of runs of consecutive voiced frames."""

    jitter_st: float | None
    run_s: float


def stability(track: F0Track, group: int | None = None) -> Stability:
    """`Stability` of a group's frames (``F0Track.groups``), or the whole track's; runs and
    frame changes never cross from one input span to the next."""
    if group is None:
        spans = np.arange(track.spans)
    else:
        if track.groups is None:
            raise ValueError("the track has no groups")
        spans = np.flatnonzero(track.groups == group)
    changes: list[np.ndarray] = []
    runs: list[int] = []
    for i in spans:
        f0 = track.f0[track.span_index == i]
        if not len(f0):
            continue
        st = 12 * np.log2(f0 / 100.0)
        d = np.abs(np.diff(st))
        changes.append(d[np.isfinite(d)])
        voiced = np.isfinite(st).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced, [0]])))
        runs += list(edges[1::2] - edges[0::2])
    pooled = np.concatenate(changes) if changes else np.empty(0)
    jitter = float(np.median(pooled)) if len(pooled) else None
    run_s = float(np.median(runs)) * HOP_S if runs else 0.0
    return Stability(jitter, run_s)


def is_sung(s: Stability) -> bool:
    """Whether the frames hold notes: jitter at most ``SUNG_JITTER_ST`` and a median voiced
    run of at least ``SUNG_RUN_S``."""
    return s.jitter_st is not None and s.jitter_st <= SUNG_JITTER_ST and s.run_s >= SUNG_RUN_S


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
    return gender, _masked(track, mask, seconds)


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


def track_key(
    vocals: Path, spans: Sequence[tuple[float, float]], groups: Sequence[int] | None = None
) -> str:
    payload: dict[str, object] = {
        "vocals": _file_digest(Path(vocals)),
        "spans": [[float(a), float(b)] for a, b in spans],
        "version": VERSION,
    }
    if groups is not None:
        payload["groups"] = [int(g) for g in groups]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_track(
    root: Path,
    vocals: Path,
    spans: Sequence[tuple[float, float]],
    groups: Sequence[int] | None = None,
) -> F0Track:
    """``f0_track`` through ``<root>/.fukikae/pitch-cache``."""
    from fukikae.artifacts import atomic_path

    path = cache_dir(root) / f"{track_key(vocals, spans, groups)}.npz"
    try:
        with np.load(path) as z:
            return F0Track(
                z["f0"], z["span_index"], float(z["seconds"]), z["span_seconds"], z["loud"],
                None if groups is None else z["groups"],
            )  # fmt: skip
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile):
        pass
    track = f0_track(vocals, spans, groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp, tmp.open("wb") as fh:
        np.savez(
            fh, f0=track.f0, span_index=track.span_index,
            seconds=np.float64(track.seconds), span_seconds=track.span_seconds,
            loud=track.loud if track.loud is not None else np.zeros(len(track.f0), bool),
            groups=track.groups if track.groups is not None else np.zeros(0, np.int64),
        )  # fmt: skip
    return track
