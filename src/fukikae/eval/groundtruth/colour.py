"""Colour-coded subtitles: one speaker per text colour (WebVTT classes, EBU-TT styles, SRT
`<font color>`, ASS colour overrides). A colour is named by the speaker tag that co-occurs with
it in at least three lines of the episode; otherwise it is `<colour>@<episode>`. A series colour
map (`labels-colours.yaml`, built from the named lines of every episode by `propose_colour_map`)
replaces that per-episode guess for dramas that keep a character's colour across the run."""

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from fukikae.eval.groundtruth.base import (
    COLOURS,
    Cue,
    GroundTruthLine,
    Namer,
    split_times,
    strip_markup,
)
from fukikae.eval.groundtruth.sdh_prefix import DEFAULT_SOUNDS, cue_segments
from fukikae.eval.speakers import label_key

MIN_TAGGED_LINES = 3
COLOUR_MAP_FILE = "labels-colours.yaml"
RUBY_STYLE = re.compile(r"rub[iy]|ルビ", re.IGNORECASE)
Anonymous = Literal["episode", "series", "drop"]
ANONYMOUS_MODES: tuple[str, ...] = ("drop", "series", "episode")
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


def _pieces(cues: list[Cue], sounds: frozenset[str]) -> list[_Piece]:
    pieces: list[_Piece] = []
    for cue in cues:
        if cue.ruby or RUBY_STYLE.search(cue.style):
            continue  # ARIB converters emit furigana as events of their own
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
    return pieces


def _tagged_counts(pieces: list[_Piece], namer: Namer) -> dict[str, Counter[str]]:
    named: dict[str, Counter[str]] = defaultdict(Counter)
    for p in pieces:
        if p.colour and p.name and (n := namer.peek(p.name)):
            named[p.colour][n] += 1
    return named


def colour_counts(
    cues_by_episode: Mapping[str, list[Cue]],
    namer: Namer,
    sounds: frozenset[str] = DEFAULT_SOUNDS,
) -> dict[str, dict[str, Counter[str]]]:
    """Episode, then colour, then the tagged names that co-occur with it, with line counts."""
    return {
        ep: _tagged_counts(_pieces(cues, sounds), namer) for ep, cues in cues_by_episode.items()
    }


def _majority(c: Counter[str]) -> str:
    return max(sorted(c), key=lambda n: c[n])


def propose_colour_map(
    counts: Mapping[str, Mapping[str, Counter[str]]],
    min_tagged: int = MIN_TAGGED_LINES,
    min_agreement: float = 0.8,
) -> tuple[dict[str, str], list[str]]:
    """A series-wide colour -> name map from `colour_counts`. A colour maps to the name holding
    at least `min_tagged` of its pooled tagged lines and `min_agreement` of them. A colour whose
    per-episode majority names differ, or whose pooled names split too evenly, is left out and
    described in the second result."""
    pooled: dict[str, Counter[str]] = defaultdict(Counter)
    majors: dict[str, dict[str, Counter[str]]] = defaultdict(dict)
    for ep in sorted(counts):
        for col, c in counts[ep].items():
            if c:
                pooled[col].update(c)
                majors[col][ep] = c
    mapping: dict[str, str] = {}
    conflicts: list[str] = []
    for col in sorted(pooled):
        c = pooled[col]
        per_ep = {ep: _majority(cc) for ep, cc in majors[col].items()}
        if len(set(per_ep.values())) > 1:
            who = ", ".join(f"{ep} {n} ({majors[col][ep][n]})" for ep, n in per_ep.items())
            conflicts.append(f"{col}: episodes disagree: {who}")
            continue
        top = _majority(c)
        if c[top] < min_tagged:
            continue
        if c[top] / sum(c.values()) < min_agreement:
            split = ", ".join(f"{n} {k}" for n, k in c.most_common())
            conflicts.append(
                f"{col}: no name holds {min_agreement:.0%} of its tagged lines: {split}"
            )
            continue
        mapping[col] = top
    return mapping, conflicts


def load_colour_map(path: Path) -> dict[str, str]:
    """`labels-colours.yaml` (colour: name); empty when the file is absent."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of colour: name")
    out = {str(k): str(v) for k, v in data.items()}
    _check_colours(out, path)
    return out


def _check_colours(mapping: Mapping[str, str], where: object) -> None:
    bad = sorted(set(mapping) - set(COLOURS))
    if bad:
        raise ValueError(f"{where}: unknown colour {', '.join(bad)}; use {', '.join(COLOURS)}")


def write_colour_map(path: Path, mapping: Mapping[str, str]) -> None:
    _check_colours(mapping, path)
    path.write_text(
        yaml.safe_dump(dict(sorted(mapping.items())), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def workspace_colour_map(root: Path, kind: str, anonymous: str | None) -> dict[str, str] | None:
    """The workspace's `labels-colours.yaml` for a colour run, None when there is none (then
    `anonymous` is an error: it only says what to do with colours the map lacks)."""
    if kind != "colour":
        if anonymous is not None:
            raise ValueError("--colour-anonymous applies to the colour ground truth only")
        return None
    mapping = load_colour_map(root / COLOUR_MAP_FILE)
    if not mapping:
        if anonymous is not None:
            raise ValueError(
                f"--colour-anonymous needs a colour map ({COLOUR_MAP_FILE}); "
                "run `fukikae eval colour-map --write` first"
            )
        return None
    return mapping


def read_colour(
    cues: list[Cue],
    episode: str,
    namer: Namer,
    sounds: frozenset[str] = DEFAULT_SOUNDS,
    *,
    colour_names: Mapping[str, str] | None = None,
    anonymous: Anonymous | None = None,
) -> list[GroundTruthLine]:
    """Speaker lines from colours. With no `colour_names` each colour is named by the tags that
    co-occur with it in this episode (else `<colour>@<episode>`). With a map, a mapped colour
    takes the map's name and an unmapped one is dropped (`anonymous` None or "drop"), kept as
    `colour:<name>` ("series") or `<colour>@<episode>` ("episode"). A tag keeps its line, except
    on a mapped colour whose name it contradicts: that line is dropped (GT4)."""
    if anonymous is not None and anonymous not in ANONYMOUS_MODES:
        raise ValueError(
            f"unknown colour-anonymous mode {anonymous!r}; use {', '.join(ANONYMOUS_MODES)}"
        )
    if colour_names is None and anonymous is not None:
        raise ValueError(
            "--colour-anonymous needs a colour map (labels-colours.yaml); "
            "run `fukikae eval colour-map --write` first"
        )
    pieces = _pieces(cues, sounds)
    local: dict[str, str] = {}
    if colour_names is None:
        local = {
            col: _majority(c)
            for col, c in _tagged_counts(pieces, namer).items()
            if max(c.values()) >= MIN_TAGGED_LINES
        }
    out: list[GroundTruthLine] = []
    for p in pieces:
        if p.name:
            if colour_names is not None and p.colour is not None and p.colour in colour_names:
                tagged, mapped = namer.peek(p.name), namer.peek(colour_names[p.colour])
                if tagged and mapped and label_key(tagged) != label_key(mapped):
                    continue  # the tag and the colour map disagree: drop the line (GT4)
            speaker = namer(p.name)
        elif not p.colour:
            speaker = None
        elif colour_names is None:
            speaker = namer(local.get(p.colour) or f"{p.colour}@{episode}")
        elif p.colour in colour_names:
            speaker = namer(colour_names[p.colour])
        elif anonymous == "series":
            speaker = namer(f"colour:{p.colour}")
        elif anonymous == "episode":
            speaker = namer(f"{p.colour}@{episode}")
        else:
            speaker = None
        if speaker:
            out.append(GroundTruthLine(p.start, p.end, speaker))
    return out
