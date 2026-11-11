"""Choosing a character's voice from the original audio: a gender estimate for the stock pools
(the median F0 from ``fukikae.pitch``), and reference clips for clone cards.

Reference selection (``select_references``, ruling CD5): a character's turns that overlap no
other speaker's turn, last 3 to 12 s and have an ASR transcript are ranked by signal-to-noise
ratio on the separated vocals (``snr_db``: the RMS of the loudest half of the turn's 20 ms
frames over the RMS of its quietest tenth) and taken greedily, up to 3 clips and 20 s. Before
the length check, the character's clean turns at most ``JOIN_GAP_S`` (0.6 s) apart with no
other turn in the gap are joined, up to 12 s (D32 #1), so speech cut into fragments can still
make a reference; a run of fragments is split where a check fails, keeping the longest valid
joins (`_split`). A turn that a
scored emotion source labels non-neutral (``tags.emotion_backend`` set, confidence at least
0.5) ranks after every other eligible turn: it is taken only when the rest fall short of the
clone floor (``FLOOR_CLIPS`` clips and ``FLOOR_S`` seconds: 2 and 10 s, D32 #2), and then
only until that is reached. When the 3-clip cap stops that fill short of a floor the
episode's turns can reach, the best combination of up to 3 clips that reaches it replaces the
greedy picks (`pick_references`). SenseVoice's labels carry no score and demote nothing; unlabelled
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

import itertools
import logging
import math
import tempfile
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW, pcm_float

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer
    from fukikae.cards.card import Card
    from fukikae.cards.store import CardStore
    from fukikae.models import Tags, Transcript, Turn

log = logging.getLogger(__name__)

# The bands are fukikae.pitch's (MALE_BELOW, FEMALE_ABOVE: 165 and 200 Hz on the median,
# calibrated in docs/results/2026-11-11-gender-bands.md); [tts] gender overrides them.

# Seconds of speech read per character by ``speaker_f0``; more doesn't move a median.
MAX_SECONDS = 120.0


def _capped(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The spans in time order, cut to ``MAX_SECONDS`` of speech in all."""
    out: list[tuple[float, float]] = []
    total = 0.0
    for start, end in sorted(spans):
        if total >= MAX_SECONDS:
            break
        end = min(end, start + MAX_SECONDS - total)
        if end > start:
            out.append((start, end))
            total += end - start
    return out


def speaker_f0(vocals: Path, spans: list[tuple[float, float]]) -> tuple[float | None, float]:
    """(median F0 in Hz over the voiced frames, or None when too few are voiced; seconds read),
    from ``fukikae.pitch`` over at most ``MAX_SECONDS`` of the spans."""
    from fukikae.pitch import f0_track, summarize

    s = summarize(f0_track(vocals, _capped(spans)))
    return s.median, s.seconds


def gender_of(
    f0: float | None, male_below: float = MALE_BELOW, female_above: float = FEMALE_ABOVE
) -> Literal["f", "m", "u"]:
    """The band of a median F0: ``m`` under ``male_below``, ``f`` over ``female_above``, else
    (or with no F0) ``u``; ``fukikae.pitch.classify`` without the voiced-speech minimum."""
    from fukikae.pitch import F0Summary, classify

    summary = F0Summary(f0, f0, f0, f0, voiced_s=0.0, seconds=0.0, voiced_fraction=0.0)
    return classify(summary, male_below, female_above, min_voiced_s=0.0).gender


def estimate_gender(
    vocals: Path,
    spans: list[tuple[float, float]],
    male_below: float = MALE_BELOW,
    female_above: float = FEMALE_ABOVE,
) -> Literal["f", "m", "u"]:
    """``m`` under ``male_below`` (165 Hz) median F0, ``f`` over ``female_above`` (200 Hz),
    else (or with too little voiced speech) ``u``."""
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
    clean: Sequence[tuple[float, float]], blockers: Sequence[tuple[float, float]], gap: float
) -> list[list[tuple[float, float]]]:
    """``clean`` (sorted) cut into runs of turns at most ``gap`` apart (overlapping turns are
    0 apart), with no ``blockers`` span inside a gap. `_split` decides where a run joins."""
    out: list[list[tuple[float, float]]] = []
    for s, e in clean:
        if out:
            run = out[-1]
            tail = max(b for _, b in run)
            if s - tail <= gap and not any(a < s and b > tail for a, b in blockers):
                run.append((s, e))
                continue
        out.append([(s, e)])
    return out


def _split(
    run: Sequence[tuple[float, float]],
    pick: Callable[[float, float, bool], RefPick | None],
    longest: float,
) -> list[RefPick]:
    """The references a run of fragments (`_chains`) makes: the run cut into consecutive
    pieces, each one joined turn (``pick(start, end, joined)``, no longer than ``longest``),
    a turn alone, or dropped, cut where the most seconds of speech end up in references, then
    fewest references (longer joins). So a run whose whole join fails a check is split at the
    failing gap with its valid sub-joins kept, and a short last piece joins its neighbour when
    another split keeps it. A turn that overlaps the end of the piece before it is dropped
    rather than cut into a reference that repeats that audio. Ties go to the longest join that
    starts earliest, so the result is deterministic."""
    n = len(run)
    memo: dict[tuple[int, int], RefPick | None] = {}
    # best[i]: ((seconds, -references), picks) over run[i:]
    best: list[tuple[tuple[float, int], list[RefPick]]] = [((0.0, 0), [])] * (n + 1)
    for i in range(n - 1, -1, -1):
        ends: list[float] = []
        end = -math.inf
        for j in range(i, n):
            end = max(end, run[j][1])
            if end - run[i][0] > longest:
                break
            ends.append(end)
        chosen = None
        for j in range(len(ends) - 1, -1, -1):
            j_abs, end = i + j, ends[j]
            if (i, j_abs) not in memo:
                memo[(i, j_abs)] = pick(run[i][0], end, j_abs > i)
            p = memo[(i, j_abs)]
            if p is None:
                continue
            k = j_abs + 1
            while k < n and run[k][0] < end:
                k += 1  # a turn overlapping this piece's end is dropped
            (secs, neg), rest = best[k]
            score = (round(secs + p.seconds, 3), neg - 1)
            if chosen is None or score > chosen[0]:
                chosen = (score, [p, *rest])
        if chosen is None or best[i + 1][0] > chosen[0]:
            chosen = best[i + 1]
        best[i] = chosen
    return best[0][1]


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
    other (``exclude`` and ``foreign`` over its gaps, its transcript, its text rate); where
    one fails, `_split` cuts the run at the failing gap and keeps the valid sub-joins."""
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
    for run in _chains(good, blockers, join_gap):
        eligible += _split(run, pick, each[1])
    return sorted(eligible, key=rank_key)


def reaches_floor(
    clips: Sequence[float], floor_s: float = FLOOR_S, floor_clips: int = FLOOR_CLIPS
) -> bool:
    """Whether clips of these lengths (seconds) reach ``floor_clips`` clips and ``floor_s``
    seconds, the seconds judged at the tenth every report prints (9.96 s is 10.0 s)."""
    return len(clips) >= floor_clips and round(sum(clips), 1) >= floor_s


def meets_floor(picks: Sequence[RefPick]) -> bool:
    """Whether the picks reach the clone floor: `FLOOR_CLIPS` clips and `FLOOR_S` seconds."""
    return floor_shortfall([p.seconds for p in picks]) is None


def floor_shortfall(clips: Sequence[float]) -> str | None:
    """How clips of these lengths (seconds) fall short of the clone floor, as a phrase; None
    when they reach it. Seconds are judged at the tenth every report prints (9.97 s is
    10.0 s), so a card the log calls 10.0 s is never said to be under 10 s."""
    seconds = sum(clips)
    if reaches_floor(clips):
        return None
    n = len(clips)
    return (
        f"{n} clip{'' if n == 1 else 's'} and {seconds:.1f} s, under the floor of "
        f"{FLOOR_CLIPS} clips and {FLOOR_S:g} s"
    )


# `pick_references`' search for a floor the clip cap stopped the greedy fill short of reads
# this many of the best-ranked turns (C(64, 3) is 41,664 combinations).
SEARCH_POOL = 64


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
    (``floor_clips`` clips and ``total[0]`` seconds, D32 #2, judged at a tenth as
    `reaches_floor` does); None under ``min_total``.

    When the clip cap stops the fill short of the floor, the combinations of ``floor_clips``
    to ``count`` of the `SEARCH_POOL` best-ranked turns within ``total[1]`` that reach it are
    searched (`_floor_combination`). Under the floor the best picks there are are still
    returned (`meets_floor` tells)."""
    picks: list[RefPick] = []
    used = 0.0
    capped = False
    for p in ranked:
        if len(picks) >= count:
            capped = True
            break
        if p.emotion is not None and reaches_floor(
            [q.seconds for q in picks], total[0], floor_clips
        ):
            break  # scored non-neutral turns only make up a shortfall
        if used + p.seconds <= total[1]:
            picks.append(p)
            used += p.seconds
    if capped and not reaches_floor([q.seconds for q in picks], total[0], floor_clips):
        found = _floor_combination(ranked, picks, count, total, floor_clips)
        if found is not None:
            picks = found
            used = sum(p.seconds for p in picks)
    return picks if used >= min_total else None


def _floor_combination(
    ranked: Sequence[RefPick],
    greedy: Sequence[RefPick],
    count: int,
    total: tuple[float, float],
    floor_clips: int,
) -> list[RefPick] | None:
    """The best combination of ``floor_clips`` to ``count`` of the `SEARCH_POOL` best-ranked
    turns within ``total[1]`` seconds that reaches the floor, in rank order; None when none
    does. Best is dropping the fewest of the greedy fill's neutral picks (none, when any
    combination keeps them all: the floor wins only when it must), then the highest-ranked
    clips (rank positions compared in order)."""
    pool = list(ranked[:SEARCH_POOL])
    secs = [p.seconds for p in pool]
    longest = sorted(secs, reverse=True)[:count]
    if not reaches_floor(longest, total[0], floor_clips):
        return None  # no combination can reach it
    neutral = {i for i, p in enumerate(pool) if p.emotion is None and p in greedy}
    best: tuple[int, tuple[int, ...]] | None = None
    for k in range(floor_clips, count + 1):
        for combo in itertools.combinations(range(len(pool)), k):
            seconds = sum(secs[i] for i in combo)
            if seconds > total[1] or round(seconds, 1) < total[0]:
                continue
            key = (len(neutral.difference(combo)), combo)
            if best is None or key < best:
                best = key
    return None if best is None else [pool[i] for i in best[1]]


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
