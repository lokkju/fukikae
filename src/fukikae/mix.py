"""The dub mix and the final mux.

``build_dub`` lays the dub cues over the separation's background bed and the original vocals,
with the original vocals removed (or ducked) inside removal windows and kept everywhere else, so
laughs, screams, songs and lines nobody dubbed still sound like the original. The stage decides
the windows (``removal_windows``): where a dubbed cue plays, minus laughter spans (both audible)
and song spans (the original vocals untouched). ``normalize`` then matches the mix's EBU R128
integrated loudness to the original's (one linear gain, then a true-peak limiter at -1 dBTP),
``write_ass`` writes the dub's lines with their speakers, and ``mux`` puts it all in one mkv.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from fukikae import media
from fukikae.artifacts import atomic_path
from fukikae.models import DubCue, Tags
from fukikae.songs import SongSettings

Span = tuple[float, float]

# Dialogue level matching: the dub's gain is clamped to this range (dB).
DUB_GAIN_RANGE = (-30.0, 12.0)
TRUE_PEAK_DB = -1.0
DUB_BITRATE = "192k"


@dataclass(frozen=True)
class MixSettings:
    """``keep_original_vocals``: keep the original vocals outside dub windows (false: only
    song and laughter spans keep them). ``duck_db``: inside a window, keep the original voice
    this many dB down instead of removing it. ``fade_ms``: ramp the original voice down before
    a window and up after it. ``match_dialogue``: set the dub's level to the original vocals'
    (one gain for the episode). ``protect_bgm``: treat the tag stage's ``bgm`` spans as songs.
    ``detect_songs``: run the theme-song detector (``fukikae.songs``), with ``songs`` (the
    ``[mix.songs]`` table) as its settings."""

    keep_original_vocals: bool = True
    duck_db: float | None = None
    fade_ms: int = 40
    match_dialogue: bool = True
    protect_bgm: bool = False
    detect_songs: bool = True
    songs: SongSettings = field(default_factory=SongSettings)

    def __post_init__(self) -> None:
        if self.duck_db is not None and not (math.isfinite(self.duck_db) and self.duck_db < 0):
            raise ValueError(f"duck_db must be a negative number of dB, not {self.duck_db!r}")
        if self.fade_ms < 0:
            raise ValueError(f"fade_ms must be >= 0, not {self.fade_ms!r}")


@dataclass(frozen=True)
class MixStats:
    seconds: float
    dub_gain_db: float | None
    peak: float


# -- spans ---------------------------------------------------------------------------------


def union(spans: Iterable[Span]) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(spans: Iterable[Span], cut: Iterable[Span]) -> list[Span]:
    cuts = union(cut)
    out: list[Span] = []
    for a, b in union(spans):
        cur = a
        for c, d in cuts:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def removal_windows(
    placed: Sequence[Span],
    tags: Tags | None,
    *,
    songs: Sequence[Span] = (),
    protect_bgm: bool = False,
) -> list[Span]:
    """Where the original vocals go: the union of ``placed`` (the dubbed spans), minus the
    tag stage's ``laughter`` and ``singing`` spans (and ``bgm`` with ``protect_bgm``) and minus
    ``songs`` from any other source."""
    keep: list[Span] = list(songs)
    for e in tags.events if tags else []:
        if e.label in ("laughter", "singing") or (protect_bgm and e.label == "bgm"):
            keep.append((e.start, e.end))
    return subtract(placed, keep)


# -- audio ---------------------------------------------------------------------------------


def _as_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / (float(np.iinfo(data.dtype).max) + 1.0)
    return data.astype(np.float32, copy=False)


def _read(path: Path) -> tuple[int, np.ndarray]:
    """(rate, float32 samples as (n, channels))."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    x = _as_float(data)
    return rate, x[:, None] if x.ndim == 1 else x


def _cue_audio(path: Path, rate: int) -> np.ndarray:
    """The cue as mono float32 at ``rate``."""
    from scipy.signal import resample_poly

    cue_rate, x = _read(path)
    mono = x.mean(axis=1)
    if cue_rate != rate:
        g = gcd(rate, cue_rate)
        mono = resample_poly(mono, rate // g, cue_rate // g).astype(np.float32)
    return mono


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), x.shape[1]), dtype=x.dtype)])


def build_dub(
    background: Path,
    vocals: Path,
    placed: Sequence[tuple[Path, float]],
    windows: Sequence[Span],
    s: MixSettings,
    out: Path,
) -> MixStats:
    """Write the dub mix to ``out`` (float32 wav at the bed's rate and channels): the bed, the
    original vocals outside ``windows`` (faded over ``fade_ms`` at each edge, or ducked by
    ``duck_db`` inside), and each ``(wav, start)`` cue resampled and summed in (overlapping
    cues are both heard). A cue running past the end lengthens the mix; nothing is cut."""
    from scipy.io import wavfile

    rate, bed = _read(background)
    vrate, vo = _read(vocals)
    if vrate != rate:
        raise ValueError(f"{vocals} is {vrate} Hz and {background} is {rate} Hz")
    channels = bed.shape[1]
    if vo.shape[1] != channels:
        vo = np.repeat(vo.mean(axis=1, keepdims=True), channels, axis=1)
    cues = [(_cue_audio(p, rate), max(0, round(start * rate))) for p, start in placed]
    n = max([len(bed), len(vo), *(i + len(x) for x, i in cues)])
    bed, vo = _fit(bed, n), _fit(vo, n)

    gain_db: float | None = None
    if s.match_dialogue and cues:
        ratios = []
        for x, i in cues:
            ref, dub = _rms(vo[i : i + len(x)].mean(axis=1)), _rms(x)
            if ref > 1e-5 and dub > 1e-5:
                ratios.append(20 * math.log10(ref / dub))
        if ratios:
            lo, hi = DUB_GAIN_RANGE
            gain_db = float(min(hi, max(lo, float(np.median(ratios)))))
    dub_gain = 10 ** ((gain_db or 0.0) / 20)

    removed = np.zeros(n, dtype=np.float32)
    fade = round(s.fade_ms / 1000 * rate)
    for a, b in union(windows):
        ia, ib = max(0, round(a * rate)), min(n, round(b * rate))
        if ib <= ia:
            continue
        removed[ia:ib] = 1.0
        if fade:
            lo = max(0, ia - fade)
            ramp = np.linspace(0.0, 1.0, ia - lo + 1, dtype=np.float32)[:-1] if ia > lo else None
            if ramp is not None:
                removed[lo:ia] = np.maximum(removed[lo:ia], ramp)
            hi = min(n, ib + fade)
            if hi > ib:
                down = np.linspace(1.0, 0.0, hi - ib + 1, dtype=np.float32)[1:]
                removed[ib:hi] = np.maximum(removed[ib:hi], down)
    floor = 0.0 if s.duck_db is None else 10 ** (s.duck_db / 20)
    vo *= (1.0 - removed * (1.0 - floor))[:, None]
    bed += vo
    del vo
    for x, i in cues:
        bed[i : i + len(x)] += (x * dub_gain)[:, None]
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        wavfile.write(tmp, rate, bed)
    return MixStats(n / rate, gain_db, float(np.abs(bed).max()) if n else 0.0)


# -- loudness ---------------------------------------------------------------------------

# The limiter runs at this multiple of the sample rate, so it catches inter-sample (true)
# peaks, not only sample peaks.
LIMITER_OVERSAMPLE = 4

_EBU = {
    "i": re.compile(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+|-inf)\s*LUFS"),
    "lra": re.compile(r"Loudness range:\s*\n\s*LRA:\s*(-?[\d.]+)\s*LU"),
    "tp": re.compile(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+|-inf)\s*dBFS"),
}


def loudness_stats(path: Path) -> dict[str, float]:
    """EBU R128 measurements of ``path`` (ffmpeg ``ebur128``): integrated loudness ``i``
    (LUFS, ``-inf`` for silence), loudness range ``lra`` (LU) and true peak ``tp`` (dBTP)."""
    stderr = media.run_ffmpeg_log(
        ["-i", str(path), "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"]
    )
    summary = stderr[stderr.rfind("Summary:") :]
    out: dict[str, float] = {}
    for key, pattern in _EBU.items():
        m = pattern.search(summary)
        if m is None:
            raise media.MediaError(f"ebur128 printed no {key} for {path}:\n{stderr[-2000:]}")
        out[key] = float(m.group(1))
    return out


def measure_loudness(path: Path) -> float:
    """EBU R128 integrated loudness of ``path`` in LUFS (``-inf`` for silence)."""
    return loudness_stats(path)["i"]


def normalize(
    src: Path, target_lufs: float, out: Path, ceiling_db: float = TRUE_PEAK_DB
) -> dict[str, Any]:
    """Write ``src`` to ``out`` (16-bit wav at the source's rate) at ``target_lufs`` integrated
    loudness with one linear gain, so the loudness range is unchanged, then a true-peak limiter
    at ``ceiling_db`` dBTP (ffmpeg ``alimiter`` at ``LIMITER_OVERSAMPLE`` times the rate, no
    auto level), which touches only the peaks the gain pushed over (ruling CD10). Returns the
    gain and the source's measurements."""
    from scipy.io import wavfile

    rate, _ = wavfile.read(src, mmap=True)
    before = loudness_stats(src)
    gain = target_lufs - before["i"] if math.isfinite(before["i"]) else 0.0
    limit = 10 ** (ceiling_db / 20)
    chain = (
        f"volume={gain:.4f}dB,aresample={rate * LIMITER_OVERSAMPLE},"
        f"alimiter=limit={limit:.6f}:level=0:attack=1:release=50:asc=0,aresample={rate}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        media.run_ffmpeg(["-i", str(src), "-af", chain, "-c:a", "pcm_s16le", str(tmp)])
    return {"gain_db": round(gain, 4), "ceiling_db": ceiling_db, "source": before}


# -- subtitles -----------------------------------------------------------------------------

ASS_HEADER = """\
[Script Info]
; Written by fukikae: the dub's lines with their speakers
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,\
2,60,60,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    cs = max(0, round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    sec, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def display_speaker(speaker: str) -> str:
    """The speaker as a viewer reads it: ``Extra`` for an unnamed cluster."""
    return "Extra" if speaker.startswith("extra:") else speaker


def _no_controls(text: str) -> str:
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" and ch not in "\u2028\u2029"
    )


def _no_overrides(text: str) -> str:
    """Braces start ASS override tags; full-width ones read the same and do nothing."""
    return text.replace("{", "\uff5b").replace("}", "\uff5d")


def _ass_name(speaker: str) -> str:
    """A speaker for the ``Name`` field and the ``Speaker:`` prefix: no control characters
    (a newline would forge an event), no braces, no commas (the field separator)."""
    return _no_overrides(_no_controls(speaker)).replace(",", " ").strip()


def _ass_text(text: str) -> str:
    """Dialogue text: line breaks become ``\\N``, other control characters go, and braces
    can't open override tags (names, ASR, translation and subtitles all reach here)."""
    lines = re.split(r"\r\n|\r|\n|\u2028|\u2029", text.strip())
    return r"\N".join(_no_overrides(_no_controls(line)) for line in lines)


def write_ass(cues: Sequence[DubCue], out: Path) -> None:
    """One ``Dialogue`` event per cue with text, ``Speaker: text``, the speaker also in the
    event's ``Name`` field. A cue's ``start`` and ``end`` are the event's times (the stage
    passes the span its audio plays). Nothing a name or a line holds can add an event or an
    override tag."""
    lines = [ASS_HEADER.rstrip("\n")]
    for c in sorted(cues, key=lambda c: (c.start, c.end, c.id)):
        if not c.text:
            continue
        name = _ass_name(c.speaker)
        lines.append(
            f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,{name},0,0,0,,"
            f"{display_speaker(name)}: {_ass_text(c.text)}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- mux -----------------------------------------------------------------------------------


def mux(source: Path, dub: Path, ass: Path, language: str, out: Path) -> None:
    """``out``: the source's video copied, the dub (AAC) as the first and default audio track
    tagged ``language``, every original audio track after it (copied, not default), the dub's
    ASS as the first and default subtitle track, then every original subtitle track (copied;
    mov_text becomes SRT; neither default nor forced, ruling CD10), and the source's
    attachments (fonts), chapters and metadata."""
    probe = media.probe(source)
    lang = media._norm_lang(language) or language  # noqa: SLF001
    args = [
        "-i", str(Path(source).resolve()), "-i", str(dub), "-i", str(ass),
        "-map", "0:v?", "-map", "1:a:0", "-map", "0:a?", "-map", "2:s:0", "-map", "0:s?",
        "-map", "0:t?",
        "-c", "copy", "-c:a:0", "aac", "-b:a:0", DUB_BITRATE,
        "-metadata:s:a:0", f"language={lang}",
        "-metadata:s:a:0", f"title=fukikae dub ({language})",
        "-disposition:a:0", "default",
        "-metadata:s:s:0", f"language={lang}",
        "-metadata:s:s:0", f"title=fukikae dub ({language}, speakers)",
        "-disposition:s:0", "default",
    ]  # fmt: skip
    for i, _ in enumerate(probe.audio, start=1):
        args += [f"-disposition:a:{i}", "0"]
    for i, t in enumerate(probe.subtitles, start=1):
        args += [f"-disposition:s:{i}", "0"]
        if t.codec == "mov_text":
            args += [f"-c:s:{i}", "srt"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        media.run_ffmpeg([*args, str(tmp)])
