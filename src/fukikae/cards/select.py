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

Reference gates (A1, identification workflow). After those checks, ``eligible_references``
can drop candidates by three gates; they filter only, and the survivors keep the CD5 order.
Each candidate is measured on its own speech: the speaker's turns inside it (a join's gaps are
left out), read as one loudness group of the F0 track (``fukikae.pitch``), so its verdict
never depends on which other candidates share the track (`measure_candidates`,
`verdict`). With ``register`` (the speaker's majority register in the episode,
`speaker_register`), a candidate is ``off_register`` when its own class (``pitch.classify``
of its median at the line minimum) crosses the register's: the other band, or between the
bands under a register that sits a full ``CONFIDENCE_ST`` (4 st) inside its band. A candidate
in the register's own band is never refused, however far from the register's median, and a
``u`` register gates no pitch. A bimodal register (``pitch.bimodal``: a cluster with two
voices) is flagged as needing a split; it still gates pitch when its majority class has at
least ``pitch.BIMODAL_GATE_RATIO`` (2x) the minority's seconds, and gates none when the two
are near even (``pitch.even_bimodal``). A candidate under ``REF_VOICED_MIN`` (0.35) of its loud
frames voiced is ``unvoiced`` (whispers, effects, processed voices). With ``song_like`` (the
song detector's per-second mask, ``songs.song_like_seconds``), a candidate with
``REF_SONG_SHARE`` (half) or more of its seconds song-like whose frames are sung
(``pitch.is_sung``: held notes, not gliding syllables) is ``song_like``: an insert song in no
theme span. With ``song_spans`` (the spans the song detector accepted, `song_spans`), a
candidate with ``REF_SONG_SHARE`` or more of its seconds inside them is ``song_like`` too, sung
or not: the detector has already judged the span sung (controller ruling (b); the theme
exclusions keep new candidates out of those spans, so this reads mostly on an old card's
references). Each candidate's verdict (`RefVerdict`) can be collected, and `Gated` counts
the refusals for ``reference_report``. The fourth gate is about the set: ``pick_references``
takes at most one clip per scene (starts within ``SCENE_GAP_S``, 60 s, of a taken pick)
until the floor needs more, so one long scene (an opening narration) can't fill every
slot; ``scene_gap=None`` turns it off (narrators).

Reference tiers (A2, D41: fukikae never renders a stock voice). `tiered_references` is the one
entry point for any speaker, a character or an extras cluster: ``floor`` when the gated picks
reach the D32 floor (2 clips, 10 s); ``short`` when they reach ``CLONE_MIN_S`` (3 s, after
D32's fragment join) but not the floor; else ``thin``: the speaker's longest turns, whole,
up to ``THIN_MAX_S`` (15 s, IndexTTS2's prompt limit; a single longer turn is cut to it),
with the A1 gates and the length, SNR and sparse-text checks set aside. Two exclusions still
hold in the thin tier, because breaking either clones a voice that is not the speaker's: a
turn overlapping a theme song span and a turn with `FOREIGN_MAX` or more of it under speech
the subtitle names give someone else. When they leave nothing, the speaker's own turns are
used regardless, so a speaker with any turn always gets picks.
"""

from __future__ import annotations

import itertools
import logging
import math
import tempfile
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from fukikae.pitch import (
    FEMALE_ABOVE,
    LINE_MIN_VOICED_S,
    MALE_BELOW,
    REF_SONG_SHARE,
    REF_VOICED_MIN,
    Gender,
    pcm_float,
)

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
    # measured by the A1 gates (None when they didn't run): median F0 in Hz, the share of
    # loud frames voiced, the share of the turn's seconds the song detector calls song-like,
    # and whether its frames hold notes (pitch.is_sung)
    median: float | None = None
    voiced_fraction: float | None = None
    song_share: float | None = None
    sung: bool | None = None

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


# -- A1: reference gates ----------------------------------------------------------------------

# Gate 4: picks whose starts are this close count as one scene (`pick_references`).
SCENE_GAP_S = 60.0

Gate = Literal["off_register", "unvoiced", "song_like"]
GATES: tuple[Gate, ...] = ("off_register", "unvoiced", "song_like")
type Bands = tuple[float, float]


type RegisterLike = Register | tuple[Gender, float | None]


class Register(NamedTuple):
    """A speaker's register in an episode: the majority class, the median of that class's
    turns, whether the turns are bimodal (two voices: flagged for a split) and whether the
    two are near even (``pitch.even_bimodal``: no pitch gate). A plain ``(gender, median)``
    pair is accepted wherever a register is."""

    gender: Gender
    median: float | None
    bimodal: bool = False
    even: bool = False


@dataclass(frozen=True)
class RefVerdict:
    """One candidate's verdict: the candidate (with what the gates measured), the gates that
    refuse it (empty when it passes) and its position in the ungated CD5 ranking."""

    pick: RefPick
    gates: tuple[Gate, ...]
    rank: int

    @property
    def passed(self) -> bool:
        return not self.gates


@dataclass(frozen=True)
class Gated:
    """How many candidates each gate refused (a candidate refused by several counts under
    each), ``top``: how many refused candidates would have ranked in the first three,
    ``split``: the speaker's turns are bimodal, so it needs a split, and ``even``: the two
    registers are near even, so no pitch gate ran."""

    off_register: int = 0
    unvoiced: int = 0
    song_like: int = 0
    top: int = 0
    split: bool = False
    even: bool = False

    def phrase(self) -> str:
        parts = [
            f"{n} {label}"
            for n, label in (
                (self.off_register, "off-register"),
                (self.unvoiced, "unvoiced"),
                (self.song_like, "song-like"),
            )
            if n
        ]
        return ", ".join(parts)


def as_register(r: RegisterLike) -> Register:
    """A `Register` from a register or a plain ``(gender, median)`` pair."""
    return r if isinstance(r, Register) else Register(r[0], r[1])


SPLIT_NOTE = "needs a split: the speaker's turns are in both registers, so no register gate ran"
SPLIT_GATED_NOTE = (
    "needs a split: the speaker's turns are in both registers; the register gate read the "
    "larger part's"
)


def speaker_register(
    vocals: Path,
    spans: Sequence[tuple[float, float]],
    bands: Bands = (MALE_BELOW, FEMALE_ABOVE),
    root: Path | None = None,
) -> Register:
    """The speaker's register in these turns (`fukikae.pitch.majority_register`: the
    majority class by seconds, each turn classed on its own median), the median of that
    class's turns only, so a mixed cluster gets its larger part's register, and whether the
    turns are bimodal (``pitch.bimodal`` on ``pitch.band_seconds``) and near even
    (``pitch.even_bimodal``). ``u`` (no class, or a tie) and near-even bimodal registers
    gate no pitch. ``root`` reads through the workspace's pitch cache."""
    from fukikae.pitch import (
        band_seconds,
        bimodal,
        cached_track,
        even_bimodal,
        f0_track,
        majority_register,
    )

    spans = list(spans)
    track = cached_track(root, vocals, spans) if root is not None else f0_track(vocals, spans)
    gender, summary = majority_register(track, bands)
    seconds = band_seconds(track, bands)
    return Register(gender, summary.median, bimodal(seconds), even_bimodal(seconds))


def song_share(start: float, end: float, song_like: np.ndarray) -> float:
    """The share of ``start``-``end`` inside the seconds ``song_like`` marks (second ``i``
    covers ``i`` to ``i + 1``; past the mask is not song-like)."""
    if end <= start:
        return 0.0
    covered = 0.0
    for i in range(max(0, int(start)), min(len(song_like), math.ceil(end))):
        if song_like[i]:
            covered += max(0.0, min(end, i + 1) - max(start, i))
    return covered / (end - start)


def own_pieces(
    start: float, end: float, turn_spans: Sequence[tuple[float, float]] | None
) -> list[tuple[float, float]]:
    """The parts of ``turn_spans`` (the speaker's turns) inside ``start``-``end``, merged where
    they overlap; the whole span when no turn falls in it (or none are given)."""
    pieces = sorted(
        (max(a, start), min(b, end)) for a, b in (turn_spans or ()) if a < end and b > start
    )
    out: list[tuple[float, float]] = []
    for a, b in pieces:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out or [(start, end)]


@dataclass(frozen=True)
class Measured:
    """What the A1 gates read of one candidate's own speech."""

    median: float | None
    voiced_fraction: float  # voiced share of its loud frames
    voiced_s: float
    sung: bool


def measure_candidates(
    vocals: Path,
    spans: Sequence[tuple[float, float]],
    turn_spans: Sequence[tuple[float, float]] | None = None,
    pitch_root: Path | None = None,
) -> list[Measured]:
    """Each candidate's `Measured`, from one F0 track where each candidate (the speaker's
    turns inside it, `own_pieces`) is its own loudness group, so no candidate's reading
    depends on another's. ``pitch_root`` reads through the workspace's pitch cache."""
    from fukikae.pitch import (
        cached_track,
        f0_track,
        is_sung,
        stability,
        summarize_group,
    )

    if not spans:
        return []
    pieces: list[tuple[float, float]] = []
    groups: list[int] = []
    for i, (start, end) in enumerate(spans):
        own = own_pieces(start, end, turn_spans)
        pieces += own
        groups += [i] * len(own)
    track = (
        cached_track(pitch_root, vocals, pieces, groups)
        if pitch_root is not None
        else f0_track(vocals, pieces, groups)
    )
    out = []
    for i in range(len(spans)):
        s = summarize_group(track, i)
        out.append(
            Measured(s.median, round(s.loud_voiced_fraction, 3), s.voiced_s,
                     is_sung(stability(track, i)))
        )  # fmt: skip
    return out


def _cross_class(m: Measured, register: Register, bands: Bands) -> bool:
    """Whether the candidate's own class crosses the register's (see the module notes)."""
    from fukikae.pitch import F0Summary, classify

    if register.gender == "u" or register.even or m.median is None:
        return False
    s = F0Summary(m.median, m.median, m.median, m.median, m.voiced_s, m.voiced_s, 1.0, 1.0)
    own = classify(s, bands[0], bands[1], LINE_MIN_VOICED_S)
    if own.gender not in ("u", register.gender):
        return True
    if own.gender == "u" and own.voiced_s >= LINE_MIN_VOICED_S and register.median is not None:
        centre = register.median
        reg = F0Summary(centre, centre, centre, centre, 9e9, 9e9, 1.0, 1.0)
        return classify(reg, bands[0], bands[1]).confidence >= 1.0
    return False


def verdict(
    m: Measured,
    *,
    register: RegisterLike | None = None,
    song: float | None = None,
    bands: Bands = (MALE_BELOW, FEMALE_ABOVE),
    in_song: float | None = None,
) -> tuple[Gate, ...]:
    """The gates refusing a measured candidate: 1 and 2 when ``register`` is given (gate 1
    only for a classed register that is not near-even bimodal), 3 when ``song`` (its
    song-like share) or ``in_song`` (its share inside the detector's accepted song spans)
    is."""
    gates: list[Gate] = []
    if register is not None:
        reg = as_register(register)
        if _cross_class(m, reg, bands):
            gates.append("off_register")
        if m.voiced_fraction < REF_VOICED_MIN:
            gates.append("unvoiced")
    if (song is not None and song >= REF_SONG_SHARE and m.sung) or (
        in_song is not None and in_song >= REF_SONG_SHARE
    ):
        gates.append("song_like")
    return tuple(gates)


def gate_candidates(
    ranked: Sequence[RefPick],
    vocals: Path,
    *,
    register: RegisterLike | None = None,
    song_like: np.ndarray | None = None,
    turn_spans: Sequence[tuple[float, float]] | None = None,
    bands: Bands = (MALE_BELOW, FEMALE_ABOVE),
    pitch_root: Path | None = None,
    song_spans: Sequence[tuple[float, float]] | None = None,
) -> list[RefVerdict]:
    """Each of ``ranked``'s verdicts, in order (`measure_candidates`, `verdict`): gates 1
    and 2 when ``register`` is given, gate 3 when ``song_like`` or ``song_spans`` is.
    ``turn_spans`` are the speaker's turns, so a joined candidate is read without its gaps."""
    if register is None and song_like is None and song_spans is None:
        return [RefVerdict(p, (), i) for i, p in enumerate(ranked)]
    spans = [(p.start, p.end) for p in ranked]
    measured = measure_candidates(vocals, spans, turn_spans, pitch_root)
    out: list[RefVerdict] = []
    for i, (p, m) in enumerate(zip(ranked, measured, strict=True)):
        share = None if song_like is None else round(song_share(p.start, p.end, song_like), 3)
        inside = None if song_spans is None else share_inside(p.start, p.end, song_spans)
        gates = verdict(m, register=register, song=share, bands=bands, in_song=inside)
        pick = RefPick(
            p.start, p.end, p.transcript, p.snr_db, p.emotion,
            None if m.median is None else round(m.median, 1), m.voiced_fraction, share, m.sung,
        )  # fmt: skip
        for gate in gates:
            log.debug("%.2f-%.2f: gated as a reference (%s)", p.start, p.end, gate)
        out.append(RefVerdict(pick, gates, i))
    return out


def count_gated(verdicts: Sequence[RefVerdict], split: bool = False, even: bool = False) -> Gated:
    """`Gated` of these verdicts; ``top`` counts refused ones ranked in the first three."""
    refused = [v for v in verdicts if not v.passed]
    return Gated(
        off_register=sum("off_register" in v.gates for v in refused),
        unvoiced=sum("unvoiced" in v.gates for v in refused),
        song_like=sum("song_like" in v.gates for v in refused),
        top=sum(v.rank < 3 for v in refused),
        split=split,
        even=even,
    )


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
    register: RegisterLike | None = None,
    song_like: np.ndarray | None = None,
    bands: Bands = (MALE_BELOW, FEMALE_ABOVE),
    pitch_root: Path | None = None,
    gated: list[Gated] | None = None,
    verdicts: list[RefVerdict] | None = None,
    song_spans: Sequence[tuple[float, float]] | None = None,
) -> list[RefPick]:
    """Every turn of the speaker that could be a reference (no overlap with another speaker,
    within ``each``, outside ``exclude``, under `FOREIGN_MAX` of it in ``foreign``, with an
    ASR transcript), in ``rank_key`` order. ``select_references`` picks from this list; eval
    tts draws held-out exemplars from it (ruling CD17).

    Then the A1 gates (`gate_candidates`) when ``register``, ``song_like`` or ``song_spans``
    is given: the
    refused candidates are dropped, the survivors keep their order and carry what the gates
    measured, each measured on the speaker's own turns inside it. ``gated`` receives one
    `Gated` and ``verdicts`` every candidate's `RefVerdict` (out-parameters, so existing
    callers are unchanged). A joined candidate a gate refuses is dropped whole, not split
    again.

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
    eligible.sort(key=rank_key)
    if register is None and song_like is None and song_spans is None:
        if verdicts is not None:
            verdicts += [RefVerdict(p, (), i) for i, p in enumerate(eligible)]
        return eligible
    judged = gate_candidates(
        eligible, vocals, register=register, song_like=song_like, turn_spans=spans,
        bands=bands, pitch_root=pitch_root, song_spans=song_spans,
    )  # fmt: skip
    if gated is not None:
        reg = None if register is None else as_register(register)
        gated.append(count_gated(judged, reg is not None and reg.bimodal,
                                 reg is not None and reg.bimodal and reg.even))  # fmt: skip
    if verdicts is not None:
        verdicts += judged
    return [v.pick for v in judged if v.passed]


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
    scene_gap: float | None = SCENE_GAP_S,
) -> list[RefPick] | None:
    """The greedy CD5 fill over ``ranked`` (``eligible_references``' order): up to ``count``
    clips and ``total[1]`` seconds, scored non-neutral turns only to make up the floor
    (``floor_clips`` clips and ``total[0]`` seconds, D32 #2, judged at a tenth as
    `reaches_floor` does); None under ``min_total``.

    One scene (A1 gate 4): a turn starting within ``scene_gap`` seconds of a taken pick is
    passed over while the fill can still reach the floor without it; only when it falls short
    are those turns taken, in rank order, until the floor is reached. ``None`` turns the gate
    off (narrators). The picks come back in rank order.

    When the clip cap stops the fill short of the floor, the combinations of ``floor_clips``
    to ``count`` of the `SEARCH_POOL` best-ranked turns within ``total[1]`` that reach it are
    searched (`_floor_combination`), fewest scored non-neutral clips first, then one clip per
    scene. Under the floor the best
    picks there are are still returned (`meets_floor` tells)."""
    order = {id(p): i for i, p in enumerate(ranked)}
    picks: list[RefPick] = []
    used = 0.0

    def floor() -> bool:
        return reaches_floor([q.seconds for q in picks], total[0], floor_clips)

    def same_scene(p: RefPick) -> bool:
        return scene_gap is not None and any(abs(p.start - q.start) < scene_gap for q in picks)

    # passes: neutral turns of new scenes (all of them, as CD5 fills), then, each only while
    # the floor isn't reached, neutral turns of a used scene, scored non-neutral turns of new
    # scenes, and scored non-neutral turns of a used scene
    passes = [(False, False), (False, True), (True, False), (True, True)]
    for emotional, repeat in passes:
        if (emotional or repeat) and floor():
            break
        for p in ranked:
            if (p.emotion is not None) != emotional or any(p is q for q in picks):
                continue
            if len(picks) >= count:
                break
            if (emotional or repeat) and floor():
                break
            if not repeat and same_scene(p):
                continue
            if used + p.seconds <= total[1]:
                picks.append(p)
                used += p.seconds
    picks.sort(key=lambda p: order[id(p)])
    capped = len(picks) >= count and len(ranked) > len(picks)
    if capped and not floor():
        found = _floor_combination(ranked, picks, count, total, floor_clips, scene_gap)
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
    scene_gap: float | None = None,
) -> list[RefPick] | None:
    """The best combination of ``floor_clips`` to ``count`` of the `SEARCH_POOL` best-ranked
    turns within ``total[1]`` seconds that reaches the floor, in rank order; None when none
    does. Best is the fewest scored non-neutral clips (CD5 takes them only to make up the
    floor, so a neutral second clip of a scene beats an emotional new scene, as the greedy
    passes order it), then the fewest pairs in one scene (starts within ``scene_gap``; A1
    gate 4), then dropping the fewest of the greedy fill's neutral picks (none, when any
    combination keeps them all: the floor wins only when it must), then the highest-ranked
    clips (rank positions compared in order)."""
    pool = list(ranked[:SEARCH_POOL])
    secs = [p.seconds for p in pool]
    longest = sorted(secs, reverse=True)[:count]
    if not reaches_floor(longest, total[0], floor_clips):
        return None  # no combination can reach it
    neutral = {i for i, p in enumerate(pool) if p.emotion is None and p in greedy}
    starts = [p.start for p in pool]

    def repeats(combo: tuple[int, ...]) -> int:
        if scene_gap is None:
            return 0
        return sum(
            abs(starts[i] - starts[j]) < scene_gap for i, j in itertools.combinations(combo, 2)
        )

    emotional = [p.emotion is not None for p in pool]
    best: tuple[int, int, int, tuple[int, ...]] | None = None
    for k in range(floor_clips, count + 1):
        for combo in itertools.combinations(range(len(pool)), k):
            seconds = sum(secs[i] for i in combo)
            if seconds > total[1] or round(seconds, 1) < total[0]:
                continue
            scored = sum(emotional[i] for i in combo)
            key = (scored, repeats(combo), len(neutral.difference(combo)), combo)
            if best is None or key < best:
                best = key
    return None if best is None else [pool[i] for i in best[3]]


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
    scene_gap: float | None = SCENE_GAP_S,
) -> list[RefPick] | None:
    """Reference clips for the speaker (one diarize label, or several of one character), in
    the order taken; None when they add up to less than ``min_total`` seconds. ``total[0]``
    is the preferred minimum (the spec's 10 to 20 s), the floor's seconds with 2 clips (D32);
    the greedy fill stops at ``total[1]``."""
    ranked = eligible_references(
        turns, speaker, transcript, tags, vocals, each=each, exclude=exclude
    )
    return pick_references(
        ranked, count=count, total=total, min_total=min_total, scene_gap=scene_gap
    )


# -- A2: reference tiers ----------------------------------------------------------------------

Tier = Literal["floor", "short", "thin"]
# Under this many seconds of gated picks a clone is made from the speaker's longest turns
# (the thin tier) and flagged harder (open decision 3: the top of D41's 2 to 3 s range).
CLONE_MIN_S = 3.0
# The most reference speech the thin tier takes: IndexTTS2's prompt limit.
THIN_MAX_S = 15.0


def span_reference(
    vocals: Path, transcript: Transcript, start: float, end: float, text: str = ""
) -> RefPick:
    """One span as a reference pick, no checks: its ASR text (``text`` when it has none) and
    its SNR. A cluster-less line's own clone (A2) is made of one."""
    from scipy.io import wavfile

    rate, data = wavfile.read(vocals, mmap=True)
    snr = round(snr_db(_samples(data, int(rate), start, end), int(rate)), 2)
    return RefPick(start, end, turn_text(transcript, start, end) or text, snr)


def thin_references(
    turns: list[Turn],
    speaker: str | Collection[str],
    transcript: Transcript,
    vocals: Path,
    *,
    theme_spans: Sequence[tuple[float, float]] = (),
    foreign: Sequence[tuple[float, float]] = (),
    fallback: bool = True,
) -> list[RefPick]:
    """The thin tier's picks: the speaker's turns, longest first, whole, while they fit in
    ``THIN_MAX_S`` (a turn that doesn't fit is passed over for a shorter one; a first turn
    longer than the limit is cut to it), leaving out turns overlapping ``theme_spans`` and
    turns with `FOREIGN_MAX` or more under ``foreign``; when that leaves nothing, from all of
    the speaker's turns (with ``fallback``; else nothing). Empty only for a speaker with no
    turn, or with ``fallback`` off. Gates, lengths, SNR order and
    text checks are set aside; the picks carry their ASR text (possibly empty) and SNR."""
    from scipy.io import wavfile

    mine = {speaker} if isinstance(speaker, str) else set(speaker)
    own = sorted(
        {
            (round(t.start, 3), round(t.end, 3))
            for t in turns
            if t.speaker in mine and t.end > t.start
        }
    )

    def allowed(a: float, b: float) -> bool:
        if any(x < b and a < y for x, y in theme_spans):
            return False
        return not (foreign and share_inside(a, b, foreign) >= FOREIGN_MAX)

    pool = [sp for sp in own if allowed(*sp)] or (own if fallback else [])
    pool.sort(key=lambda sp: (-(sp[1] - sp[0]), sp[0]))
    chosen: list[tuple[float, float]] = []
    used = 0.0
    for a, b in pool:
        if not chosen and b - a > THIN_MAX_S:
            chosen.append((a, round(a + THIN_MAX_S, 3)))
            break
        if used + (b - a) <= THIN_MAX_S + 1e-9:
            chosen.append((a, b))
            used += b - a
    if not chosen:
        return []
    rate, data = wavfile.read(vocals, mmap=True)
    rate = int(rate)
    return [
        RefPick(
            a, b, turn_text(transcript, a, b), round(snr_db(_samples(data, rate, a, b), rate), 2)
        )
        for a, b in chosen
    ]


def tiered_references(
    turns: list[Turn],
    speaker: str | Collection[str],
    transcript: Transcript,
    tags: Tags | None,
    vocals: Path,
    *,
    register: RegisterLike | None = None,
    song_like: np.ndarray | None = None,
    song_spans: Sequence[tuple[float, float]] | None = None,
    theme_spans: Sequence[tuple[float, float]] = (),
    exclude: Sequence[tuple[float, float]] | None = None,
    foreign: Sequence[tuple[float, float]] = (),
    scene_gap: float | None = SCENE_GAP_S,
    bands: Bands = (MALE_BELOW, FEMALE_ABOVE),
    pitch_root: Path | None = None,
    gated: list[Gated] | None = None,
) -> tuple[Tier, list[RefPick]]:
    """The speaker's tier and reference picks (see the module notes): `eligible_references`
    with the A1 gates (``exclude``, by default ``theme_spans``; ``foreign``; ``register``,
    ``song_like``, ``song_spans``) and `pick_references` down to `CLONE_MIN_S` give the
    ``floor`` or ``short`` tier; under it, `thin_references` (``theme_spans`` and ``foreign``
    still excluded). Never an empty pick list for a speaker with any turn. ``gated``
    receives the gated eligible turns' `Gated`, as `eligible_references` gives it."""
    ranked = eligible_references(
        turns, speaker, transcript, tags, vocals,
        exclude=theme_spans if exclude is None else exclude, foreign=foreign,
        register=register, song_like=song_like, song_spans=song_spans, bands=bands,
        pitch_root=pitch_root, gated=gated,
    )  # fmt: skip
    picks = pick_references(ranked, min_total=CLONE_MIN_S, scene_gap=scene_gap)
    if picks:
        return ("floor" if meets_floor(picks) else "short"), picks
    thin = thin_references(
        turns, speaker, transcript, vocals, theme_spans=theme_spans, foreign=foreign
    )
    return "thin", thin


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


def reference_gating(work: Path, settings: Any = None) -> np.ndarray | None:
    """Gate 3's bed evidence for an episode work dir: the song detector's per-second
    song-like mask over the whole episode (``songs.song_like_seconds`` with ``settings``, a
    ``SongSettings``); None without the background or vocals stem. Whether a candidate is
    sung is read from its own F0 track (``pitch.is_sung``)."""
    from fukikae import songs

    s = settings or songs.SongSettings()
    sep = work / "01-separate"
    background, vocals = sep / "background.wav", sep / "vocals.wav"
    if not (background.exists() and vocals.exists()):
        return None
    return songs.song_like_seconds(background, vocals, s)


def song_spans(work: Path, settings: Any = None) -> list[tuple[float, float]]:
    """The song spans the song detector accepts for an episode work dir (``songs.
    detect_theme_songs`` with ``settings``, a ``SongSettings``: the spans the mix stage keeps
    the original audio over as ``detector``), for gate 3; empty without the stems."""
    from fukikae import songs

    sep = work / "01-separate"
    stems = [sep / "background.wav", sep / "vocals.wav", sep / "vocals_16k.wav"]
    if not all(p.exists() for p in stems):
        return []
    spans, _ = songs.detect_theme_songs(stems[0], stems[1], stems[2], settings)
    return [(float(sp.start), float(sp.end)) for sp in spans]


def reference_report(picks: list[RefPick], gated: Gated | None = None) -> list[str]:
    """The CD5 warning for chosen references a scored source calls non-neutral, if any, the
    A1 line when a gate refused a candidate that would have ranked in the first three, and
    when the speaker's turns are bimodal `SPLIT_NOTE` (near even: no register gate ran) or
    `SPLIT_GATED_NOTE` (the gate read the larger part's register)."""
    labels = sorted({p.emotion for p in picks if p.emotion})
    out = [f"references may carry emotion: {', '.join(labels)}"] if labels else []
    if gated is not None and gated.top > 0:
        out.append(f"gated: {gated.phrase()}")
    if gated is not None and gated.split:
        out.append(SPLIT_NOTE if gated.even else SPLIT_GATED_NOTE)
    return out


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
    tier: Tier | None = None,
) -> Card:
    """A clone card of ``backend`` from the picks: each cut to a 24 kHz mono wav sidecar,
    with its transcript and ``origin = {episode, start, end}``; ``seed = 1234``, the default
    ``emotion_strength`` (0.8), the backend's ``model_revision``. The tier is always recorded
    in ``[meta] reference_tier`` (outside the card id; the thin flags read it), so a re-put of
    the same id replaces the stored tier (``CardStore.put`` keeps stored ``[meta]`` only for a
    card with none). With no ``tier``, the picks' own: ``floor`` when they reach the D32 floor,
    ``short`` from `CLONE_MIN_S`, else ``thin``. Not put in the store."""
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
    meta: dict[str, Any] = {"reference_emotions": labels} if labels else {}
    if tier is None:
        seconds = sum(p.seconds for p in picks)
        tier = "floor" if meets_floor(picks) else "short" if seconds >= CLONE_MIN_S else "thin"
    meta["reference_tier"] = tier
    return Card(
        target_language=target_language,
        voice=Voice(
            backend=backend.name,
            model_revision=backend.model_revision,
            source="clone",
            seed=CLONE_SEED,
            references=tuple(refs),
        ),
        meta=meta,
    )
