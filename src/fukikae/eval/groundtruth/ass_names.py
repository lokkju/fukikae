"""Speaker names in the Name field of an ASS track's Dialogue lines."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from fukikae.eval.groundtruth.base import (
    IGNORED_NAMES,
    GroundTruthLine,
    Namer,
    canonical_name,
    load_alias_file,
    normalize_aliases,
    usable_name,
)
from fukikae.eval.speakers import label_key

__all__ = [
    "IGNORED_NAMES",
    "UNKNOWN",
    "Line",
    "canonical_name",
    "label_spans",
    "load_alias_file",
    "normalize_aliases",
    "parse_ass",
    "read_ass",
    "usable_name",
]

UNKNOWN = "?"
_DEFAULT_FORMAT = "Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"


@dataclass(frozen=True)
class Line:
    start: float
    end: float
    name: str
    text: str


def _seconds(stamp: str) -> float:
    h, m, s = stamp.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_ass(
    text: str,
    aliases: dict[str, str] | None = None,
    seen: dict[str, Counter[str]] | None = None,
    hits: Counter[tuple[str, str]] | None = None,
) -> list[Line]:
    """Dialogue lines that name exactly one speaker, in file order. Names are canonicalized,
    then mapped through `aliases`. `seen` collects raw spellings per canonical key and `hits`
    counts alias uses as (canonical FROM, TO), for the report."""
    namer = Namer(aliases, seen, hits)
    fields = [f.strip().casefold() for f in _DEFAULT_FORMAT.split(",")]
    lines: list[Line] = []
    in_events = False
    for raw in text.splitlines():
        row = raw.strip()
        if row.startswith("["):
            in_events = row.casefold() == "[events]"
            continue
        if not in_events:
            continue
        if row.casefold().startswith("format:"):
            fields = [f.strip().casefold() for f in row.split(":", 1)[1].split(",")]
        elif row.casefold().startswith("dialogue:"):
            parts = row.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
            if len(parts) < len(fields):
                continue
            d = dict(zip(fields, parts, strict=True))
            try:
                start, end = _seconds(d["start"]), _seconds(d["end"])
            except (KeyError, ValueError):
                continue
            name = namer(d.get("name", ""))
            if name is not None:
                lines.append(Line(start, end, name, d.get("text", "")))
    return lines


def read_ass(
    text: str,
    aliases: dict[str, str] | None = None,
    seen: dict[str, Counter[str]] | None = None,
    hits: Counter[tuple[str, str]] | None = None,
) -> list[GroundTruthLine]:
    return [
        GroundTruthLine(ln.start, ln.end, ln.name) for ln in parse_ass(text, aliases, seen, hits)
    ]


def label_spans(
    spans: list[tuple[float, float]], lines: list[Line], min_overlap: float
) -> list[str]:
    """One label per span: the name covering at least `min_overlap` of it, else `?`."""
    spelling: dict[str, str] = {}
    for ln in sorted(lines, key=lambda x: x.start):
        spelling.setdefault(label_key(ln.name), ln.name)
    out: list[str] = []
    for a, b in spans:
        dur = b - a
        cover: dict[str, float] = defaultdict(float)
        for ln in lines:
            ov = min(b, ln.end) - max(a, ln.start)
            if ov > 0:
                cover[label_key(ln.name)] += ov
        best = max(cover, key=lambda k: cover[k], default=None)
        if best is not None and dur > 0 and min(cover[best], dur) / dur >= min_overlap:
            out.append(spelling[best])
        else:
            out.append(UNKNOWN)
    return out
