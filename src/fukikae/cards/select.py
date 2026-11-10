"""Choosing a character's voice from the original audio: a gender estimate for the stock pools,
and reference clips for clone cards.

Reference selection (``select_references``, ruling CD5): a character's turns that overlap no
other speaker's turn, last 3 to 12 s and have an ASR transcript are ranked by signal-to-noise
ratio on the separated vocals (``snr_db``: the RMS of the loudest half of the turn's 20 ms
frames over the RMS of its quietest tenth) and taken greedily, up to 3 clips and 20 s. Before
the length check, the character's clean turns at most ``JOIN_GAP_S`` (0.6 s) apart with no
other turn in the gap are joined, up to 12 s (D32 #1), so speech cut into fragments can still
make a reference; a joined turn that fails a check falls back to its pieces. A turn that a
scored emotion source labels non-neutral (``tags.emotion_backend`` set, confidence at least
0.5) ranks after every other eligible turn: it is taken only when the rest fall short of the
clone floor (``FLOOR_CLIPS`` clips and ``FLOOR_S`` seconds: 2 and 10 s, D32 #2), and then
only until that is reached. SenseVoice's labels carry no score and demote nothing; unlabelled
and low-confidence turns rank by SNR alone. Whenever a chosen reference is scored
non-neutral, ``reference_report`` says "references may carry emotion: <labels>", and
``clone_card`` records the labels in ``[meta] reference_emotions`` (outside the card id),
which ``fukikae voices show`` prints. Under ``min_total`` (6 s) there is no clone; between it
and the floor the best picks are still a clone, which ``cards.auto`` flags for review.
A turn whose transcript is far sparser than the speaker's usual text rate and sparse in
absolute terms (``SPARSE_*``: one word over five seconds) is mostly someone else, effort
sounds or silence, and is no reference either. Turns overlapping ``exclude``
(``reference_exclusions``: the theme songs the mix stage's detector finds on the stems, and
the tag stage's singing spans) are never references:
separated song vocals are clean and loud, so they would otherwise rank first.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer
    from fukikae.cards.card import Card
    from fukikae.cards.store import CardStore
    from fukikae.models import Tags, Transcript, Turn

log = logging.getLogger(__name__)

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


def pcm_float(seg: np.ndarray) -> np.ndarray:
    """Integer PCM as float32 in [-1, 1]: signed by its maximum, unsigned (8-bit wav) centred
    on its midpoint first."""
    if seg.dtype.kind == "u":
        mid = (int(np.iinfo(seg.dtype).max) + 1) / 2
        return ((seg.astype(np.float32) - mid) / mid).astype(np.float32)
    return seg.astype(np.float32) / float(np.iinfo(seg.dtype).max)


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
            seg = pcm_float(seg)
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


# -- reference selection (ruling CD5) -------------------------------------------------------

SNR_FRAME_S = 0.020
SNR_FLOOR = 1e-5  # digital silence on separated vocals; keeps the ratio finite
SCORED_MIN = 0.5
CLIP_RATE = 24000
CLONE_SEED = 1234
NO_SPACE_LANGUAGES = frozenset({"ja", "zh", "yue", "ko"})


@dataclass(frozen=True)
class RefPick:
    start: float
    end: float
    transcript: str
    snr_db: float
    emotion: str | None = None  # the label when a scored source calls the turn non-neutral

    @property
    def seconds(self) -> float:
        return self.end - self.start


def _samples(data: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    seg = np.asarray(data[max(0, int(start * rate)) : max(0, int(end * rate))])
    if seg.dtype.kind in "iu":
        seg = pcm_float(seg)
    if seg.ndim > 1:
        seg = seg.mean(axis=1)
    return seg.astype(np.float32)


def snr_db(x: np.ndarray, rate: int) -> float:
    """RMS of the loudest half of 20 ms frames over the RMS of the quietest tenth, in dB."""
    n = max(1, int(SNR_FRAME_S * rate))
    frames = len(x) // n
    if frames < 2:
        return 0.0
    rms = np.sort(np.sqrt((x[: frames * n].astype(np.float64).reshape(frames, n) ** 2).mean(1)))
    loud = math.sqrt(float((rms[frames // 2 :] ** 2).mean()))
    quiet = math.sqrt(float((rms[: max(1, frames // 10)] ** 2).mean()))
    return 20 * math.log10(max(loud, SNR_FLOOR) / max(quiet, SNR_FLOOR))


def turn_text(transcript: Transcript, start: float, end: float) -> str:
    """The ASR text inside ``[start, end]``: the words whose midpoint falls in it (segments,
    when the transcript has no word timing)."""
    sep = "" if transcript.language in NO_SPACE_LANGUAGES else " "
    parts: list[str] = []
    for seg in transcript.segments:
        if transcript.word_timing and seg.words:
            parts += [w.text for w in seg.words if start <= (w.start + w.end) / 2 <= end]
        elif start <= (seg.start + seg.end) / 2 <= end:
            parts.append(seg.text)
    return sep.join(p.strip() for p in parts if p.strip()).strip()


def _emotion(tags: Tags | None, start: float, end: float) -> str | None:
    """The label of the emotion span overlapping the turn most, when a scored source calls it
    non-neutral at ``SCORED_MIN`` or more; else None."""
    if tags is None or not tags.emotion_backend:
        return None
    best, overlap = None, 0.0
    for e in tags.emotions:
        o = min(end, e.end) - max(start, e.start)
        if o > overlap:
            best, overlap = e, o
    if best is None or best.confidence is None or best.confidence < SCORED_MIN:
        return None
    return None if best.label == "neutral" else best.label


# A turn whose ASR text is this sparse is mostly someone else's speech, effort sounds or
# nothing (FT E12: a 5.2 s turn transcribed as one word, "くそ", Task 12): it is no reference
# when its text rate is under SPARSE_SHARE of the speaker's median rate (over every turn of a
# second or more with text) and under the absolute floor, characters per second for languages
# written without spaces, words per second otherwise.
SPARSE_SHARE = 0.3
SPARSE_FLOOR_CHARS = 1.0
SPARSE_FLOOR_WORDS = 0.4
SPARSE_MIN_TURN_S = 1.0


def text_units(text: str, language: str | None) -> int:
    """Characters (letters and digits) for a language written without spaces, else words."""
    if language in NO_SPACE_LANGUAGES:
        return sum(1 for ch in text if ch.isalnum())
    return len(text.split())


def _sparse(rate: float, median: float, language: str | None) -> bool:
    floor = SPARSE_FLOOR_CHARS if language in NO_SPACE_LANGUAGES else SPARSE_FLOOR_WORDS
    return rate < floor and rate < SPARSE_SHARE * median


def rank_key(p: RefPick) -> tuple[bool, float, float]:
    """The CD5 order: unscored-or-neutral turns first, then by SNR, loudest first."""
    return (p.emotion is not None, -p.snr_db, p.start)


# D32 #1: a character's adjacent turns this close together, with no other speaker's turn
# between them, are joined into one candidate before the length check, so speech the
# diarizer (or the subtitle names) cut into fragments can still make a reference.
JOIN_GAP_S = 0.6
# D32 #2: a clone wants at least this many clips and ``total[0]`` (10 s) of speech.
FLOOR_CLIPS = 2
FLOOR_S = 10.0
# A turn with this share of it under speech the subtitle names give someone else is no
# reference (``foreign``; auto.STALE_OVERLAP is the same rule for cards).
FOREIGN_MAX = 0.1


def share_inside(start: float, end: float, spans: Sequence[tuple[float, float]]) -> float:
    """The share of ``start``-``end`` inside the union of ``spans``."""
    covered, edge = 0.0, start
    for a, b in sorted(spans):
        a, b = max(a, edge), min(b, end)
        if b > a:
            covered += b - a
            edge = b
    return covered / (end - start) if end > start else 1.0


def _chains(
    clean: Sequence[tuple[float, float]],
    blockers: Sequence[tuple[float, float]],
    gap: float,
    longest: float,
) -> list[list[tuple[float, float]]]:
    """``clean`` (sorted) cut into runs of turns at most ``gap`` apart, with no ``blockers``
    span inside a gap, each run no longer than ``longest`` from its first start to its end."""
    out: list[list[tuple[float, float]]] = []
    for s, e in clean:
        if out:
            run = out[-1]
            head, tail = run[0][0], max(b for _, b in run)
            if (
                s - tail <= gap
                and max(e, tail) - head <= longest
                and not any(a < s and b > tail for a, b in blockers)
            ):
                run.append((s, e))
                continue
        out.append([(s, e)])
    return out


def eligible_references(
    turns: list[Turn],
    speaker: str | Collection[str],
    transcript: Transcript,
    tags: Tags | None,
    vocals: Path,
    *,
    each: tuple[float, float] = (3.0, 12.0),
    exclude: Sequence[tuple[float, float]] = (),
    foreign: Sequence[tuple[float, float]] = (),
    join_gap: float = JOIN_GAP_S,
) -> list[RefPick]:
    """Every turn of the speaker that could be a reference (no overlap with another speaker,
    within ``each``, outside ``exclude``, under `FOREIGN_MAX` of it in ``foreign``, with an
    ASR transcript), in ``rank_key`` order. ``select_references`` picks from this list; eval
    tts draws held-out exemplars from it (ruling CD17).

    Before the length check, the speaker's clean turns (overlapping no other speaker, outside
    ``exclude`` and ``foreign``) at most ``join_gap`` apart (`JOIN_GAP_S`) are joined, when no
    other turn (another speaker's, or one of the speaker's own that isn't clean) falls in the
    gap and the result stays within ``each[1]`` (D32 #1). A joined turn is checked like any
    other (``exclude`` and ``foreign`` over its gaps, its transcript, its text rate); one that
    fails falls back to its pieces, each checked alone."""
    from scipy.io import wavfile

    mine = {speaker} if isinstance(speaker, str) else set(speaker)
    others = [t for t in turns if t.speaker not in mine]
    own = sorted((t for t in turns if t.speaker in mine), key=lambda t: (t.start, t.end))
    dropped = [t for t in own if foreign and share_inside(t.start, t.end, foreign) >= FOREIGN_MAX]
    own = [t for t in own if t not in dropped]
    lang = transcript.language
    rates = [
        text_units(turn_text(transcript, t.start, t.end), lang) / (t.end - t.start)
        for t in own
        if t.end - t.start >= SPARSE_MIN_TURN_S
    ]
    rates = [r for r in rates if r > 0]
    median = float(np.median(rates)) if rates else 0.0
    rate, data = wavfile.read(vocals, mmap=True)
    rate = int(rate)

    def pick(start: float, end: float, joined: bool = False) -> RefPick | None:
        if not each[0] <= end - start <= each[1]:
            return None
        if any(o.start < end and start < o.end for o in others):
            return None
        if any(a < end and start < b for a, b in exclude):
            return None
        if joined and foreign and share_inside(start, end, foreign) >= FOREIGN_MAX:
            return None  # a single turn was checked (unrounded) when ``dropped`` was made
        text = turn_text(transcript, start, end)
        if not text:
            return None
        if _sparse(text_units(text, lang) / (end - start), median, lang):
            log.debug("%.2f-%.2f: text too sparse for a reference (%r)", start, end, text)
            return None
        snr = snr_db(_samples(data, rate, start, end), rate)
        return RefPick(start, end, text, round(snr, 2), _emotion(tags, start, end))

    def clean(s: float, e: float) -> bool:
        return not any(o.start < e and s < o.end for o in others) and not any(
            a < e and s < b for a, b in exclude
        )

    spans = [(round(t.start, 3), round(t.end, 3)) for t in own]
    good = [sp for sp in spans if clean(*sp)]
    blockers = [(o.start, o.end) for o in others]
    blockers += [sp for sp in spans if not clean(*sp)]
    blockers += [(t.start, t.end) for t in dropped]
    eligible: list[RefPick] = []
    for run in _chains(good, blockers, join_gap, each[1]):
        joined = pick(run[0][0], max(b for _, b in run), True) if len(run) > 1 else None
        if joined is not None:
            eligible.append(joined)
            continue
        eligible += [p for p in (pick(s, e) for s, e in run) if p is not None]
    return sorted(eligible, key=rank_key)


def meets_floor(picks: Sequence[RefPick]) -> bool:
    """Whether the picks reach the clone floor: `FLOOR_CLIPS` clips and `FLOOR_S` seconds."""
    return len(picks) >= FLOOR_CLIPS and sum(p.seconds for p in picks) >= FLOOR_S - 1e-9


def floor_shortfall(clips: Sequence[float]) -> str | None:
    """How clips of these lengths (seconds) fall short of the clone floor, as a phrase; None
    when they reach it."""
    seconds = sum(clips)
    if len(clips) >= FLOOR_CLIPS and seconds >= FLOOR_S - 1e-9:
        return None
    n = len(clips)
    return (
        f"{n} clip{'' if n == 1 else 's'} and {seconds:.1f} s, under the floor of "
        f"{FLOOR_CLIPS} clips and {FLOOR_S:g} s"
    )


def pick_references(
    ranked: Sequence[RefPick],
    *,
    count: int = 3,
    total: tuple[float, float] = (FLOOR_S, 20.0),
    min_total: float = 6.0,
    floor_clips: int = FLOOR_CLIPS,
) -> list[RefPick] | None:
    """The greedy CD5 fill over ``ranked`` (``eligible_references``' order): up to ``count``
    clips and ``total[1]`` seconds, scored non-neutral turns only to make up the floor
    (``floor_clips`` clips and ``total[0]`` seconds, D32 #2); None under ``min_total``. Under
    the floor the best picks there are still returned (`meets_floor` tells)."""
    picks: list[RefPick] = []
    used = 0.0
    for p in ranked:
        if len(picks) >= count:
            break
        if p.emotion is not None and used >= total[0] and len(picks) >= floor_clips:
            break  # scored non-neutral turns only make up a shortfall
        if used + p.seconds <= total[1]:
            picks.append(p)
            used += p.seconds
    return picks if used >= min_total else None


def select_references(
    turns: list[Turn],
    speaker: str | Collection[str],
    transcript: Transcript,
    tags: Tags | None,
    vocals: Path,
    *,
    count: int = 3,
    total: tuple[float, float] = (FLOOR_S, 20.0),
    each: tuple[float, float] = (3.0, 12.0),
    min_total: float = 6.0,
    exclude: Sequence[tuple[float, float]] = (),
) -> list[RefPick] | None:
    """Reference clips for the speaker (one diarize label, or several of one character), in
    the order taken; None when they add up to less than ``min_total`` seconds. ``total[0]``
    is the preferred minimum (the spec's 10 to 20 s), the floor's seconds with 2 clips (D32);
    the greedy fill stops at ``total[1]``."""
    ranked = eligible_references(
        turns, speaker, transcript, tags, vocals, each=each, exclude=exclude
    )
    return pick_references(ranked, count=count, total=total, min_total=min_total)


def reference_exclusions(
    work: Path, tags: Tags | None, settings: Any = None
) -> list[tuple[float, float]]:
    """Spans no reference may overlap, for an episode work dir: theme songs from the
    separation stems (``fukikae.songs.detect_theme_songs`` with ``settings``, a
    ``SongSettings``; skipped when the stems are missing) and the tag stage's singing spans."""
    from fukikae import songs

    out: list[tuple[float, float]] = []
    sep = work / "01-separate"
    stems = [sep / "background.wav", sep / "vocals.wav", sep / "vocals_16k.wav"]
    missing = [p.name for p in stems if not p.exists()]
    if not missing:
        spans, _ = songs.detect_theme_songs(stems[0], stems[1], stems[2], settings)
        out += [(float(sp.start), float(sp.end)) for sp in spans]
    else:
        log.warning(
            "%s: no %s, so theme songs are not excluded from reference selection (only tagged "
            "singing is)", work, ", ".join(missing),
        )  # fmt: skip
    for e in tags.events if tags else []:
        if e.label == "singing":
            out.append((float(e.start), float(e.end)))
    return out


def reference_report(picks: list[RefPick]) -> list[str]:
    """The CD5 warning for chosen references a scored source calls non-neutral, if any."""
    labels = sorted({p.emotion for p in picks if p.emotion})
    return [f"references may carry emotion: {', '.join(labels)}"] if labels else []


def cut_clip(vocals: Path, start: float, end: float, dest: Path, rate: int = CLIP_RATE) -> Path:
    """``[start, end]`` of the vocals as a mono 16-bit wav at ``rate``, reproducibly."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    src_rate, data = wavfile.read(vocals, mmap=True)
    x = _samples(data, int(src_rate), start, end).astype(np.float64)
    if int(src_rate) != rate:
        g = math.gcd(int(src_rate), rate)
        x = resample_poly(x, rate // g, int(src_rate) // g)
    wavfile.write(dest, rate, (np.clip(x, -1.0, 1.0) * 32767).round().astype("<i2"))
    return dest


def clone_card(
    backend: type[Synthesizer],
    picks: list[RefPick],
    store: CardStore,
    vocals: Path,
    source_language: str,
    target_language: str,
    episode: str,
) -> Card:
    """A clone card of ``backend`` from the picks: each cut to a 24 kHz mono wav sidecar,
    with its transcript and ``origin = {episode, start, end}``; ``seed = 1234``, the default
    ``emotion_strength`` (0.8), the backend's ``model_revision``. Not put in the store."""
    from fukikae.cards.card import Card, Reference, Voice

    refs: list[Reference] = []
    with tempfile.TemporaryDirectory(prefix="fukikae-clip-") as tmp:
        for i, p in enumerate(picks):
            clip = cut_clip(vocals, p.start, p.end, Path(tmp) / f"{i}.wav")
            refs.append(
                Reference(
                    clip=store.add_sidecar(clip, "clip"),
                    transcript=p.transcript,
                    language=source_language,
                    origin={"episode": episode, "start": p.start, "end": p.end},
                )
            )
    labels = sorted({p.emotion for p in picks if p.emotion})
    return Card(
        target_language=target_language,
        voice=Voice(
            backend=backend.name,
            model_revision=backend.model_revision,
            source="clone",
            seed=CLONE_SEED,
            references=tuple(refs),
        ),
        meta={"reference_emotions": labels} if labels else {},
    )
