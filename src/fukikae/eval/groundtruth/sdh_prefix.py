"""Speaker tags in SDH/closed captions: `NAME:`, `(NAME)`, `[NAME]`, `- NAME:`."""

import re
from dataclasses import dataclass

from fukikae.eval.groundtruth.base import (
    Cue,
    GroundTruthLine,
    Namer,
    split_times,
    strip_markup,
)

DEFAULT_SOUNDS = frozenset(
    {
        "music", "музыка", "음악", "音乐", "音樂", "laughs", "laughter", "laughing", "смеётся",
        "смеется", "웃음", "笑", "applause", "sighs", "gasps", "screams", "sfx", "silence",
        "тишина", "аплодисменты", "박수", "掌声", "掌聲",
        "grunts", "groans", "chuckles", "whispers", "cries", "sobs", "yells", "shouts",
        "ríe", "susurra", "susurrando", "asiente", "gime", "recita", "silabeando", "suspira",
        "llora", "grita", "todos", "ambos", "música", "risas", "aplausos",
        "sigh", "gasp", "scoffs", "mutters", "whimpers", "snores", "coughs", "yelling",
        "radio", "tv", "vídeo", "video", "смех", "вздох", "вздыхает", "кричит",
        "웃", "한숨", "음성", "소리", "신음", "경적", "자막", "함께", "音声", "声",
    }
)  # fmt: skip

# A tag starting with one of these words is a fragment or a language note, not a name.
LEADING_WORDS = frozenset(
    {"in", "on", "over", "through", "with", "from", "at", "of", "for", "to", "en", "sur", "por",
     "по", "в", "acento", "voz", "sound", "sounds"}
)  # fmt: skip
_CJK = re.compile(r"[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u9fff\uac00-\ud7af]")
# `A과 B`, `A와 B`, `A랑 B` (Korean) and `AとB`, `A和B` (Japanese, Chinese): several speakers.
_PAIR = re.compile(
    r"[\uac00-\ud7af]+[과와랑]\s+[\uac00-\ud7af]+"
    r"|[\u3040-\u30ff\u3400-\u9fff]+[と和与跟][\u3040-\u30ff\u3400-\u9fff]+"
)
_HANGUL_DESCRIPTION_END = re.compile(
    r"[며면서다음]$"
)  # `웃으며`, `울린다`, `조작음`: an action or sound

MAX_NAME_CHARS = 24
MAX_NAME_WORDS = 4
_NAME_CHARS = re.compile(r"[\w .'’\-]+")
_DASH = re.compile(r"^\s*[-–—]\s*")
_BRACKET = re.compile(r"^[(\[（【]\s*([^)\]）】]{1,40}?)\s*[)\]）】]\s*(.*)$", re.DOTALL)
_COLON = re.compile(r"^([^:：(\[]{1,40}?)\s*[:：]\s*(.*)$", re.DOTALL)


def _is_description(name: str, sounds: frozenset[str]) -> bool:
    folded = name.casefold()
    entries = {x.casefold() for x in sounds}
    words = folded.split()
    if folded in entries or any(w in entries for w in words):
        return True
    if any(_CJK.search(e) and e in folded for e in entries):
        return True  # no spaces in Hangul/CJK: a listed word anywhere in the tag
    if words[0] in LEADING_WORDS or _PAIR.search(name):
        return True
    return bool(_HANGUL_DESCRIPTION_END.search(words[-1]) and _CJK.search(words[-1]))


def valid_name(name: str, sounds: frozenset[str], upper_only: bool = False) -> bool:
    """1-4 words, at most 24 characters, letters of any script (digits, dots, hyphens and
    apostrophes allowed), not a sound description, not all lowercase. `upper_only` demands
    capitals for cased scripts (the `NAME:` form, which plain sentences also resemble)."""
    name = name.strip()
    if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > MAX_NAME_WORDS:
        return False
    if not _NAME_CHARS.fullmatch(name) or not any(c.isalpha() for c in name):
        return False
    if sounds and _is_description(name, sounds):
        return False
    cased = name.upper() != name.lower()
    if cased and name == name.lower():
        return False
    return not (cased and upper_only and name != name.upper())


def extract_tag(line: str, sounds: frozenset[str] = DEFAULT_SOUNDS) -> tuple[str, str] | None:
    """(name, rest of the line) when `line` starts with a speaker tag, after markup and a
    leading dash are removed."""
    text = _DASH.sub("", strip_markup(line)).strip()
    m = _BRACKET.match(text)
    if m and valid_name(m.group(1), sounds):
        return m.group(1).strip(), m.group(2).strip()
    m = _COLON.match(text)
    if m and valid_name(m.group(1), sounds, upper_only=True):
        return m.group(1).strip(), m.group(2).strip()
    return None


@dataclass(frozen=True)
class Segment:
    """One speaker's share of a cue: the lines from a dash or tag to the next, as plain text
    with the tag and dash removed."""

    name: str | None
    text: str


def cue_segments(text: str, sounds: frozenset[str] = DEFAULT_SOUNDS) -> list[Segment]:
    """Split a cue where a dash starts a new speaker or a tag appears; a tag carries over to
    the lines that follow it. A tag with no text after it in its segment is not a tag (it is
    a sound description such as `(Relinchos)`)."""
    segs: list[tuple[str | None, list[str]]] = []
    for line in text.split("\n"):
        plain = strip_markup(line).strip()
        if not plain:
            continue
        tag = extract_tag(line, sounds)
        dash = bool(_DASH.match(plain))
        if dash or tag is not None or not segs:
            segs.append((tag[0] if tag else None, []))
        segs[-1][1].append(tag[1] if tag else _DASH.sub("", plain, count=1) if dash else plain)
    out = []
    for name, lines in segs:
        body = " ".join(x for x in lines if x).strip()
        out.append(Segment(name if body else None, body))
    return out


def read_sdh(
    cues: list[Cue], namer: Namer, sounds: frozenset[str] = DEFAULT_SOUNDS
) -> list[GroundTruthLine]:
    out: list[GroundTruthLine] = []
    for cue in cues:
        segs = cue_segments(cue.text, sounds)
        spans = split_times(
            cue.start, cue.end, [max(1, len(re.sub(r"\s", "", s.text))) for s in segs]
        )
        for seg, (a, b) in zip(segs, spans, strict=True):
            name = namer(seg.name) if seg.name else None
            if name:
                out.append(GroundTruthLine(a, b, name))
    return out
