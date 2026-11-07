"""Colour-coded subtitles: one speaker per text colour (WebVTT classes, EBU-TT styles, SRT
`<font color>`, ASS colour overrides). A colour is named by the speaker tag that co-occurs with
it in at least three lines of the episode; otherwise it is `<colour>@<episode>`."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from fukikae.eval.groundtruth.base import Cue, GroundTruthLine, Namer, split_times, strip_markup
from fukikae.eval.groundtruth.sdh_prefix import DEFAULT_SOUNDS, cue_segments

MIN_TAGGED_LINES = 3
DEFAULT_COLOUR = "white"
_MARK = re.compile(r"<c(?:\.([a-z]+))?>|</c>|\n|[^<\n]+|<")
_SOUND_ONLY = re.compile(r"^\s*(?:[(\[（【].*[)\]）】]|[♪♫*].*)\s*$", re.DOTALL)


def line_colours(text: str) -> list[tuple[str | None, str]]:
    """(colour, line markup) per line; the colour is the one covering most of the line's
    letters, None for default or white."""
    stack: list[str | None] = []
    lines: list[list[tuple[str | None, str]]] = [[]]
    for m in _MARK.finditer(text):
        tok = m.group(0)
        if tok == "\n":
            lines.append([])
        elif tok.startswith("<c"):
            stack.append(m.group(1))
        elif tok == "</c>":
            if stack:
                stack.pop()
        else:
            lines[-1].append((stack[-1] if stack else None, tok))
    out: list[tuple[str | None, str]] = []
    for runs in lines:
        weight: Counter[str | None] = Counter()
        for col, tok in runs:
            weight[col] += len(re.sub(r"\s", "", strip_markup(tok)))
        col = max(weight, key=lambda c: weight[c], default=None) if sum(weight.values()) else None
        out.append((None if col == DEFAULT_COLOUR else col, "".join(t for _, t in runs)))
    return out


@dataclass(frozen=True)
class _Piece:
    start: float
    end: float
    colour: str | None
    name: str | None  # raw tag name


def has_colour(cues: list[Cue]) -> int:
    return sum(1 for c in cues if any(col for col, _ in line_colours(c.text)))


def read_colour(
    cues: list[Cue],
    episode: str,
    namer: Namer,
    sounds: frozenset[str] = DEFAULT_SOUNDS,
) -> list[GroundTruthLine]:
    pieces: list[_Piece] = []
    for cue in cues:
        groups: list[tuple[str | None, list[str]]] = []
        for col, markup in line_colours(cue.text):
            if not strip_markup(markup).strip():
                continue
            if groups and groups[-1][0] == col:
                groups[-1][1].append(markup)
            else:
                groups.append((col, [markup]))
        segs: list[tuple[str | None, str | None, str]] = []
        for col, markups in groups:
            for seg in cue_segments("\n".join(markups), sounds):
                if seg.name is None and (not seg.text or _SOUND_ONLY.match(seg.text)):
                    continue  # a sound description, not a speaker
                segs.append((col, seg.name, seg.text))
        for (col, name, _), (a, b) in zip(
            segs, split_times(cue.start, cue.end, [max(1, len(s[2])) for s in segs]), strict=True
        ):
            pieces.append(_Piece(a, b, col, name))
    named: dict[str, Counter[str]] = defaultdict(Counter)
    for p in pieces:
        if p.colour and p.name and (n := namer.peek(p.name)):
            named[p.colour][n] += 1
    colour_name = {
        col: max(sorted(c), key=lambda n: c[n])
        for col, c in named.items()
        if c and max(c.values()) >= MIN_TAGGED_LINES
    }
    out: list[GroundTruthLine] = []
    for p in pieces:
        if p.name:
            speaker = namer(p.name)
        elif p.colour:
            speaker = namer(colour_name.get(p.colour) or f"{p.colour}@{episode}")
        else:
            speaker = None
        if speaker:
            out.append(GroundTruthLine(p.start, p.end, speaker))
    return out
