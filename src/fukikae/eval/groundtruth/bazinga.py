"""Bazinga word-level speaker annotations as named-ASS ground truth.

Each annotation line is `<episode> <speaker_id> <start> <end> <token> <confidence> [...]`.
Speaker ids are gold; timings come from forced alignment, and words at the lowest confidence
(0.1) are evenly spaced fillers, not aligned. Consecutive words of one speaker make a turn, a gap
over `MAX_GAP_S` starts a new one, and a turn's span comes only from its confident words."""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fukikae.eval.groundtruth.transcript import NamedLine

MAX_GAP_S = 1.0
FILM_SCALE = 25 / 23.976  # PAL-fast audio slowed back to film speed
PSEUDO_SPEAKERS = {"all", "multiple_persons"}

_EPISODE = re.compile(r"Season(\d+)\.Episode(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Word:
    episode: str
    speaker: str
    start: float
    end: float
    token: str
    confidence: float


@dataclass
class TurnStats:
    kept: int = 0  # words inside a kept turn's confident span
    dropped: Counter[str] = field(default_factory=Counter)  # words, by reason
    turns: int = 0  # turns written
    turns_dropped: int = 0  # turns of a real speaker with no confident word
    speakers: dict[str, int] = field(default_factory=dict)  # turns per speaker


def read_words(path: Path) -> list[Word]:
    words: list[Word] = []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        p = raw.split(" ")
        try:
            words.append(Word(p[0], p[1], float(p[2]), float(p[3]), p[4], float(p[5])))
        except (IndexError, ValueError) as e:
            raise ValueError(f"{path}:{n}: not a Bazinga annotation line: {raw[:60]!r}") from e
    return words


def episode_code(name: str) -> str:
    """`S01E01` from a Bazinga episode name such as `TheBigBangTheory.Season01.Episode01`."""
    m = _EPISODE.search(name)
    if not m:
        raise ValueError(f"{name}: no SeasonNN.EpisodeNN in the episode name")
    return f"S{int(m[1]):02d}E{int(m[2]):02d}"


def is_pseudo(speaker: str) -> bool:
    return speaker in PSEUDO_SPEAKERS or "unknown" in speaker


def words_to_turns(
    words: list[Word],
    min_confidence: float = 0.9,
    time_scale: float = 1.0,
    max_gap: float = MAX_GAP_S,
) -> tuple[list[NamedLine], TurnStats]:
    """Named turns, sorted by start, with every time multiplied by `time_scale`."""
    groups: list[list[Word]] = []
    for w in words:
        last = groups[-1][-1] if groups else None
        if last is not None and last.speaker == w.speaker and w.start - last.end <= max_gap:
            groups[-1].append(w)
        else:
            groups.append([w])

    stats = TurnStats()
    lines: list[NamedLine] = []
    speakers: Counter[str] = Counter()
    for g in groups:
        speaker = g[0].speaker
        if is_pseudo(speaker):
            stats.dropped["pseudo-speaker"] += len(g)
            continue
        good = [w for w in g if w.confidence >= min_confidence]
        stats.dropped["low-confidence"] += len(g) - len(good)
        if not good:
            stats.turns_dropped += 1
            continue
        stats.kept += len(good)
        start = min(w.start for w in good) * time_scale
        end = max(w.end for w in good) * time_scale
        lines.append(NamedLine(start, end, speaker, " ".join(w.token for w in good)))
        speakers[speaker] += 1
    lines.sort(key=lambda ln: (ln.start, ln.end))
    stats.turns = len(lines)
    stats.speakers = dict(speakers)
    return lines, stats
