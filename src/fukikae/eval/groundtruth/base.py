"""Shared pieces of the ground-truth readers: the line type, name canonicalization and
aliases, subtitle cue parsing (SRT, WebVTT, ASS, TTML) and the subtitle/speech alignment check."""

import html
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from fukikae.eval.speakers import label_key
from fukikae.models import Turn

IGNORED_NAMES = {"sign", "signs", "title", "caption", "note", "song", "op", "ed", "lyrics"}
_MULTI = re.compile(r"[&/,]|\band\b", re.IGNORECASE)
_THOUGHT = re.compile(r"\s*(?:/\s*(?:I|OS|OC|VO)|\((?:V\.O\.|O\.S\.)\))\s*$", re.IGNORECASE)

MIN_SHARE = 0.6
MIN_GAIN = 0.1
MAX_OFFSET_S = 10.0
OFFSET_STEP_S = 0.1


@dataclass(frozen=True)
class GroundTruthLine:
    """A stretch of time a named speaker is talking; `speaker` is canonicalized and aliased."""

    start: float
    end: float
    speaker: str


# ---- names and aliases ----


def canonical_name(name: str) -> str:
    """The name without a trailing thought/off-screen marker (`/I`, `/OS`, `(V.O.)`), with
    whitespace collapsed and spacing around parentheses normalized."""
    name = _THOUGHT.sub("", name.strip())
    name = re.sub(r"\s*\(\s*", " (", name)
    name = re.sub(r"\s*\)", ")", name)
    return re.sub(r"\s+", " ", name).strip()


def usable_name(name: str) -> bool:
    if not name or label_key(name) in IGNORED_NAMES:
        return False
    return not _MULTI.search(name)


def normalize_aliases(aliases: dict[str, str] | None) -> dict[str, str]:
    """FROM (canonical, case-insensitive) -> TO (canonical spelling)."""
    return {label_key(canonical_name(a)): canonical_name(b) for a, b in (aliases or {}).items()}


def load_alias_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of FROM: TO")
    return {str(a): str(b) for a, b in data.items()}


class Namer:
    """Canonicalizes a raw speaker name, drops unusable ones and applies the aliases.
    `seen` collects raw spellings per canonical key and `hits` counts alias uses as
    (canonical FROM, TO), for the auto-label report."""

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        seen: dict[str, Counter[str]] | None = None,
        hits: Counter[tuple[str, str]] | None = None,
    ) -> None:
        self.amap = normalize_aliases(aliases)
        self.seen = seen
        self.hits = hits

    def peek(self, raw: str) -> str | None:
        """The name `__call__` would return, without recording it."""
        name = canonical_name(raw.strip())
        if not usable_name(name):
            return None
        return self.amap.get(label_key(name)) or name

    def __call__(self, raw: str) -> str | None:
        name = canonical_name(raw.strip())
        if not usable_name(name):
            return None
        if self.seen is not None:
            self.seen.setdefault(label_key(name), Counter())[raw.strip()] += 1
        target = self.amap.get(label_key(name))
        if target is not None and label_key(target) != label_key(name):
            if self.hits is not None:
                self.hits[(name, target)] += 1
            name = target
        return name


# ---- cues ----


@dataclass(frozen=True)
class Cue:
    """A timed block of subtitle text. Line breaks are `\\n`; colour is carried as `<c.NAME>`
    markers (NAME from COLOURS), other markup is left in place for `strip_markup`."""

    start: float
    end: float
    text: str


COLOURS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}


def colour_from_rgb(r: int, g: int, b: int) -> str:
    return min(
        COLOURS, key=lambda n: sum((x - y) ** 2 for x, y in zip(COLOURS[n], (r, g, b), strict=True))
    )


def colour_from_hex(value: str) -> str | None:
    v = value.strip().lstrip("#")
    if len(v) in (8,):
        v = v[:6]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", v):
        return COLOURS_BY_NAME.get(v.casefold())
    return colour_from_rgb(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


COLOURS_BY_NAME = {n: n for n in COLOURS} | {"lime": "green", "aqua": "cyan", "fuchsia": "magenta"}


def colour_from_class(cls: str) -> str | None:
    """`vtt_yellow`, `yellow`, `colorFFFF00`, `textYellow` -> a COLOURS name."""
    c = cls.strip().casefold()
    for prefix in ("vtt_", "text", "color", "colour"):
        if c.startswith(prefix):
            c = c[len(prefix) :]
            break
    return colour_from_hex(c) if c else None


_TAG = re.compile(r"<[^>]*>|\{[^}]*\}")
_STAMP = re.compile(r"(?:(\d+):)?(\d+):(\d+)(?:[.,](\d+))?")


def strip_markup(text: str) -> str:
    return html.unescape(_TAG.sub("", text)).replace(" ", " ").replace("‎", "")


def _stamp(s: str) -> float:
    m = _STAMP.fullmatch(s.strip())
    if not m:
        raise ValueError(f"bad timestamp {s!r}")
    h, mi, se, frac = m.groups()
    return (
        int(h or 0) * 3600 + int(mi) * 60 + int(se) + (int(frac) / 10 ** len(frac) if frac else 0)
    )


def _font_to_class(text: str) -> str:
    def font(m: re.Match[str]) -> str:
        c = colour_from_hex(m.group(1))
        return f"<c.{c}>" if c else ""

    text = re.sub(r"<font\b[^>]*?\bcolor\s*=\s*[\"']?([#\w]+)[\"']?[^>]*>", font, text, flags=re.I)
    return re.sub(r"</font\s*>", "</c>", text, flags=re.I)


def _vtt_classes(text: str) -> str:
    def cls(m: re.Match[str]) -> str:
        for part in m.group(1).split("."):
            c = colour_from_class(part)
            if c:
                return f"<c.{c}>"
        return "<c>"

    return re.sub(r"<c((?:\.[^\s>.]+)+)(?:\s[^>]*)?>", cls, text)


def parse_text_cues(text: str) -> list[Cue]:
    """SRT and WebVTT."""
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")):
        lines = block.split("\n")
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->", 1)
                try:
                    start, end = _stamp(a), _stamp(b.split()[0])
                except (ValueError, IndexError):
                    break
                body = _vtt_classes(_font_to_class("\n".join(lines[i + 1 :]).strip()))
                if body:
                    cues.append(Cue(start, end, body))
                break
    return cues


_ASS_FORMAT = "Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"


def _ass_text(raw: str) -> str:
    raw = raw.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")

    def override(m: re.Match[str]) -> str:
        out = ""
        for c in re.finditer(r"\\1?c&H([0-9A-Fa-f]{1,6})&?", m.group(0)):
            v = c.group(1).rjust(6, "0")
            b, g, r = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
            out = f"<c.{colour_from_rgb(r, g, b)}>"
        if "\\r" in m.group(0) and not out:
            out = "<c.white>"
        return out

    return re.sub(r"\{[^}]*\}", override, raw)


def parse_ass_cues(text: str) -> list[Cue]:
    fields = [f.strip().casefold() for f in _ASS_FORMAT.split(",")]
    cues: list[Cue] = []
    in_events = False
    for raw in text.splitlines():
        row = raw.strip()
        if row.startswith("["):
            in_events = row.casefold() == "[events]"
        elif not in_events:
            continue
        elif row.casefold().startswith("format:"):
            fields = [f.strip().casefold() for f in row.split(":", 1)[1].split(",")]
        elif row.casefold().startswith("dialogue:"):
            parts = row.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
            if len(parts) < len(fields):
                continue
            d = dict(zip(fields, parts, strict=True))
            try:
                start, end = _stamp(d["start"]), _stamp(d["end"])
            except (KeyError, ValueError):
                continue
            cues.append(Cue(start, end, _ass_text(d.get("text", ""))))
    return cues


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_ttml_cues(text: str) -> list[Cue]:
    """TTML / EBU-TT-D: span colour from `tts:color` or a style (by its colour, else its id)."""
    root = ET.fromstring(text.lstrip("﻿").encode("utf-8"))  # noqa: S314 - a local subtitle file, no network entities
    styles: dict[str, str | None] = {}
    for el in root.iter():
        if _local(el.tag) == "style":
            sid = next((v for k, v in el.attrib.items() if k.endswith("}id")), None)
            col = next((v for k, v in el.attrib.items() if _local(k) == "color"), None)
            if sid:
                styles[sid] = (colour_from_hex(col) if col else None) or colour_from_class(sid)

    def colour(el: ET.Element) -> str | None:
        direct = next((v for k, v in el.attrib.items() if _local(k) == "color"), None)
        if direct:
            return colour_from_hex(direct)
        return styles.get(el.attrib.get("style", ""))

    def render(el: ET.Element) -> str:
        out = el.text or ""
        for ch in el:
            if _local(ch.tag) == "br":
                out += "\n"
            else:
                c = colour(ch)
                inner = render(ch)
                out += f"<c.{c}>{inner}</c>" if c else inner
            out += ch.tail or ""
        return out

    cues: list[Cue] = []
    for el in root.iter():
        if _local(el.tag) != "p" or "begin" not in el.attrib or "end" not in el.attrib:
            continue
        try:
            start, end = _stamp(el.attrib["begin"]), _stamp(el.attrib["end"])
        except ValueError:
            continue
        c = colour(el)
        body = re.sub(r"[ \t]*\n[ \t]*", "\n", render(el)).strip()
        if c:
            body = f"<c.{c}>{body}</c>"
        if body:
            cues.append(Cue(start, end, body))
    return cues


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def read_cues(path: Path) -> list[Cue]:
    text = read_text(path)
    head = text.lstrip()[:200]
    suffix = Path(path).suffix.casefold()
    if (
        suffix in {".ass", ".ssa"}
        or "[script info]" in head.casefold()
        or head.startswith("[Events]")
    ):
        return parse_ass_cues(text)
    if suffix in {".xml", ".ttml", ".dfxp"} or head.startswith("<"):
        return parse_ttml_cues(text)
    return parse_text_cues(text)


def episode_label(path: Path) -> str:
    m = re.search(r"S\d+E\d+", Path(path).name, re.IGNORECASE)
    return m.group(0).upper() if m else Path(path).stem


def split_times(start: float, end: float, weights: list[int]) -> list[tuple[float, float]]:
    """The cue's time cut into consecutive pieces proportional to `weights`."""
    total = sum(weights) or 1
    out, t = [], 0
    for w in weights:
        a = start + (end - start) * t / total
        t += w
        out.append((a, start + (end - start) * t / total))
    return out


# ---- alignment ----


@dataclass(frozen=True)
class AlignmentReport:
    overlap_share: float  # share of line time overlapping detected speech, as given
    best_offset_s: float  # how late the lines run against the audio (lines = audio + offset)
    share_at_offset: float  # the same share once the lines are moved back by best_offset_s


class AlignmentError(ValueError):
    """Ground truth that does not line up with the detected speech."""


def _union(turns: list[Turn]) -> tuple[np.ndarray, np.ndarray]:
    spans = sorted((t.start, t.end) for t in turns if t.end > t.start)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    x, y, acc = [], [], 0.0
    for a, b in merged:
        x += [a, b]
        y += [acc, acc + (b - a)]
        acc += b - a
    return np.array(x), np.array(y)


def alignment(lines: list[GroundTruthLine], speech_turns: list[Turn]) -> AlignmentReport:
    """Share of the lines' time that overlaps detected speech, and the global offset within
    +-10 s that maximizes it (ties go to the smaller offset). A positive offset means the
    lines run late against the audio; correct them by subtracting it."""
    ls = [ln for ln in lines if ln.end > ln.start]
    x, y = _union(speech_turns)
    if not ls or len(x) == 0:
        return AlignmentReport(0.0, 0.0, 0.0)
    s = np.array([ln.start for ln in ls])
    e = np.array([ln.end for ln in ls])
    total = float((e - s).sum())

    def share(offset: float) -> float:
        cum = np.interp(e - offset, x, y) - np.interp(s - offset, x, y)
        return float(cum.sum() / total)

    n = int(round(MAX_OFFSET_S / OFFSET_STEP_S))
    grid = [round(k * OFFSET_STEP_S, 3) for k in range(-n, n + 1)]
    scores = [share(o) for o in grid]
    best = max(range(len(grid)), key=lambda i: (round(scores[i], 9), -abs(grid[i])))
    return AlignmentReport(share(0.0), grid[best], scores[best])


def check_alignment(
    lines: list[GroundTruthLine],
    speech_turns: list[Turn],
    force: bool = False,
    min_share: float = MIN_SHARE,
    min_gain: float = MIN_GAIN,
) -> tuple[list[GroundTruthLine], AlignmentReport, bool]:
    """(lines, report, offset applied). Refuses below `min_share` at the best offset unless
    `force`; moves the lines back by the best offset when that gains `min_gain` of overlap
    and reaches `min_share` (a junk offset is never applied)."""
    rep = alignment(lines, speech_turns)
    if not lines:
        return lines, rep, False
    if rep.share_at_offset < min_share and not force:
        raise AlignmentError(
            f"subtitles do not line up with the speech: {rep.overlap_share:.0%} overlap as is, "
            f"{rep.share_at_offset:.0%} at the best offset ({rep.best_offset_s:+.1f} s, "
            f"limit +-{MAX_OFFSET_S:g} s); a different release or an intro offset? "
            "Pass --force to score anyway"
        )
    gain = rep.share_at_offset - rep.overlap_share
    if gain >= min_gain - 1e-9 and rep.share_at_offset >= min_share and rep.best_offset_s:
        off = rep.best_offset_s
        return [replace(ln, start=ln.start - off, end=ln.end - off) for ln in lines], rep, True
    return lines, rep, False
