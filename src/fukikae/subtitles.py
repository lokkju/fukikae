"""Subtitle cues for the pipeline and the evaluation: parsing (SRT, WebVTT, ASS, TTML), the
filter that keeps signs, songs and SDH notes out of synthesis, and the spoken text of a cue.

The parsers moved here from ``fukikae.eval.groundtruth.base`` (R1: pipeline code never imports
``fukikae.eval``), which imports them back.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cue:
    """A timed block of subtitle text. Line breaks are `\\n`; colour is carried as `<c.NAME>`
    markers (NAME from COLOURS), other markup is left in place for `strip_markup`."""

    start: float
    end: float
    text: str
    style: str = ""  # the ASS event's style name; empty for other formats
    ruby: bool = False  # an ASS furigana event (a `\\fscy` under 70 in its overrides)
    name: str = ""  # the ASS event's Name field (a speaker, or SIGN); empty for other formats
    # ASS placement from the overrides: "pos" for `\\pos`/`\\move`, "top" for `\\an7`-`\\an9`
    placement: str = ""


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


# Invisible directional marks (Netflix starts every cue with U+202A) hide a tag at the line start.
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def strip_markup(text: str) -> str:
    return _INVISIBLE.sub("", html.unescape(_TAG.sub("", text)).replace("\xa0", " "))


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
        for c in re.finditer(r"\\1?c&H([0-9A-Fa-f]{1,8})&?", m.group(0)):
            v = c.group(1)
            v = v[-6:] if len(v) > 6 else v.rjust(6, "0")  # AABBGGRR: drop the alpha byte
            b, g, r = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
            out = f"<c.{colour_from_rgb(r, g, b)}>"
        if "\\r" in m.group(0) and not out:
            out = "<c.white>"
        return out

    return re.sub(r"\{[^}]*\}", override, raw)


_FSCY = re.compile(r"\\fscy(\d+(?:\.\d+)?)")


def _is_ruby(raw: str) -> bool:
    """Furigana events are small (`\\fscy50`); tagged parentheses shrink with `\\fscx` only."""
    return any(
        float(m.group(1)) < 70 for b in re.findall(r"\{[^}]*\}", raw) for m in _FSCY.finditer(b)
    )


_POS = re.compile(r"\\(?:pos|move)\s*\(")
_TOP = re.compile(r"\\an[789]")


def _placement(raw: str) -> str:
    blocks = re.findall(r"\{[^}]*\}", raw)
    if any(_POS.search(b) for b in blocks):
        return "pos"
    if any(_TOP.search(b) for b in blocks):
        return "top"
    return ""


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
            raw_text = d.get("text", "")
            cues.append(
                Cue(
                    start,
                    end,
                    _ass_text(raw_text),
                    d.get("style", ""),
                    _is_ruby(raw_text),
                    d.get("name", "").strip(),
                    _placement(raw_text),
                )
            )
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
    """UTF-16 by its BOM, else UTF-8 (a UTF-8 BOM dropped); newlines translated to `\\n` as
    `Path.read_text` does."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


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


# ---- the synthesis filter ----

RULES: tuple[str, ...] = ("sdh", "song", "sign")
SIGN_NAMES = frozenset({"sign", "signs", "title", "caption", "note", "text"})
SONG_NAMES = frozenset({"song", "op", "ed", "lyrics", "karaoke", "insert"})
_SIGN_STYLE = re.compile(r"(?:^|[^a-z])(?:signs?|titles?|captions?)(?:[^a-z]|$)")
_SONG_STYLE = re.compile(
    r"(?:^|[^a-z])(?:songs?|op|ed|karaoke|lyrics|opening|ending|insert)(?:\d|[^a-z]|$)"
)
_NOTE = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|【[^】]*】")
_MUSIC = re.compile("[♪♫♬♩]")
_DASH = re.compile(r"^\s*[-‐–—]\s*")
# An SDH speaker prefix: an upper-case name and a colon (`LUCY:`, `- YOUNG MAKAROV:`).
_PREFIX = re.compile(r"^[A-Z][A-Z0-9 .'’\-]{0,30}(?:\([^)]*\))?\s*:\s+")


def _is_speaker(name: str) -> bool:
    folded = name.strip().casefold()
    return bool(folded) and folded not in SIGN_NAMES | SONG_NAMES


def filter_cue(cue: Cue, rules: Sequence[str] = RULES) -> str | None:
    """The rule that keeps ``cue`` out of synthesis, or None to speak it.

    ``song``: a music mark (``♪``), or an ASS style or Name of song, OP, ED, karaoke or lyrics.
    ``sign``: an ASS line placed with ``\\pos``/``\\move``, a style of sign or title, a Name of
    SIGN (Crunchyroll's), or a line raised with ``\\an8`` that names no speaker (a named one is
    dialogue moved off a sign at the bottom). ``sdh``: the whole text is bracketed notes,
    ``[door slams]`` or ``(sighs)``.
    """
    unknown = sorted(set(rules) - set(RULES))
    if unknown:
        raise ValueError(f"unknown subtitle filter rule(s) {', '.join(unknown)}; use {RULES}")
    style = cue.style.casefold()
    name = cue.name.strip().casefold()
    text = strip_markup(cue.text)
    if "song" in rules and (_MUSIC.search(text) or _SONG_STYLE.search(style) or name in SONG_NAMES):
        return "song"
    if "sign" in rules and (
        cue.placement == "pos"
        or _SIGN_STYLE.search(style)
        or name in SIGN_NAMES
        or (cue.placement == "top" and not _is_speaker(cue.name))
    ):
        return "sign"
    if "sdh" in rules:
        lines = [_DASH.sub("", ln).strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        if lines and all(not _NOTE.sub("", ln).strip(" .,!?…") for ln in lines):
            return "sdh"
    return None


def spoken_text(cue: Cue) -> str:
    """What a voice says: markup, a leading ``NAME:`` speaker prefix, line-leading dialogue
    dashes, bracketed notes and music marks dropped, lines joined with a space."""
    out: list[str] = []
    for ln in strip_markup(cue.text).split("\n"):
        ln = _PREFIX.sub("", _DASH.sub("", ln.strip()))
        ln = _MUSIC.sub("", _NOTE.sub("", ln))
        ln = re.sub(r"\s+([,.!?…])", r"\1", re.sub(r"\s+", " ", ln)).strip()
        if ln:
            out.append(ln)
    return " ".join(out)
