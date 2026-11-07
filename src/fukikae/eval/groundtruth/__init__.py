"""Ground-truth speaker labels read from subtitles: ASS speaker names, SDH/CC speaker tags and
colour-coded subtitles, all as timed `GroundTruthLine`s, plus the alignment check against the
detected speech."""

from collections import Counter
from pathlib import Path
from typing import Literal

from fukikae.eval.groundtruth.ass_names import read_ass
from fukikae.eval.groundtruth.base import (
    AlignmentError,
    AlignmentReport,
    GroundTruthLine,
    Namer,
    alignment,
    check_alignment,
    episode_label,
    read_cues,
    read_text,
)
from fukikae.eval.groundtruth.colour import read_colour
from fukikae.eval.groundtruth.sdh_prefix import DEFAULT_SOUNDS, read_sdh

Kind = Literal["ass", "sdh", "colour"]
KINDS: tuple[str, ...] = ("ass", "sdh", "colour")

__all__ = [
    "KINDS",
    "AlignmentError",
    "AlignmentReport",
    "GroundTruthLine",
    "Kind",
    "alignment",
    "check_alignment",
    "read_ground_truth",
]


def read_ground_truth(
    path: Path,
    kind: Kind,
    aliases: dict[str, str] | None,
    *,
    episode: str | None = None,
    sounds: frozenset[str] = DEFAULT_SOUNDS,
    seen: dict[str, Counter[str]] | None = None,
    hits: Counter[tuple[str, str]] | None = None,
) -> list[GroundTruthLine]:
    """Timed speaker lines from the subtitle file at `path`. Names are canonicalized and run
    through `aliases` for every kind. `episode` labels colours that no name tag identifies
    (default: the SxxEyy in the file name). `seen`/`hits` collect spellings and alias uses."""
    if kind == "ass":
        return read_ass(read_text(path), aliases, seen, hits)
    namer = Namer(aliases, seen, hits)
    cues = read_cues(path)
    if kind == "sdh":
        return read_sdh(cues, namer, sounds)
    if kind == "colour":
        return read_colour(cues, episode or episode_label(path), namer, sounds)
    raise ValueError(f"unknown ground-truth source {kind!r}; use one of {', '.join(KINDS)}")
