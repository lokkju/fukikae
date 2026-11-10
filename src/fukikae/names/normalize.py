"""Subtitle Name values as speaker keys. A raw Name loses its inner-voice or off-screen marker,
is canonicalised (whitespace, spacing around parentheses) and folded (NFKC, casefold) into a key;
`parse_name` also classifies it as named, role, multi or ignored."""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from fukikae.subtitles import SIGN_NAMES, SONG_NAMES

# A trailing inner-voice or off-screen marker: `/I`, `/OS`, `/OC`, `/VO`, `(V.O.)`, `(O.S.)`.
_THOUGHT = re.compile(r"\s*(?:/\s*(?:I|OS|OC|VO)|\((?:V\.O\.|O\.S\.)\))\s*$", re.IGNORECASE)
# Two speakers or possession: a slash, ampersand, comma or the word `and`.
_MULTI = re.compile(r"[&/,]|\band\b", re.IGNORECASE)
# A slot in one scene: the name ends in one letter or up to two digits after a space, with an
# optional trailing period (`PRINCE A.`).
_ROLE = re.compile(r"\s(?:[^\W\d_]|\d{1,2})\.?$")
# ASS override blocks (`{\\i1}`) go before keying; any other braces lose the brace characters
# only (`{NATSU}` is NATSU).
_OVERRIDE = re.compile(r"\{\\[^}]*\}")
GROUP_WORDS = frozenset({"all", "everyone", "both", "crowd", "together", "all together"})

# suggest_aliases: both keys this long, the rare one under RARE_MAX_LINES lines, the common one
# at least COMMON_RATIO times as frequent.
SUGGEST_MIN_LENGTH = 5
RARE_MAX_LINES = 5
COMMON_RATIO = 5


class NameClass(StrEnum):
    NAMED = "named"  # a person across the season
    ROLE = "role"  # a slot within one episode (`TOWNSPEOPLE A`, `GUARD 2`)
    MULTI = "multi"  # two speakers, possession or a group: names nobody
    IGNORED = "ignored"  # empty, or a sign or song value


@dataclass(frozen=True)
class ParsedName:
    raw: str
    key: str | None  # the folded canonical name; None for multi and ignored
    display: str | None  # the name as shown; None for multi and ignored
    cls: NameClass
    inner: bool  # an inner-voice or off-screen marker was stripped


def fold(text: str) -> str:
    """Case- and width-insensitive form of a name (NFKC, then casefold)."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


def canonical_name(name: str) -> str:
    """The name without a trailing thought/off-screen marker (`/I`, `/OS`, `(V.O.)`), with
    whitespace collapsed and spacing around parentheses normalized."""
    name = _THOUGHT.sub("", name.strip())
    name = re.sub(r"\s*\(\s*", " (", name)
    name = re.sub(r"\s*\)", ")", name)
    return re.sub(r"\s+", " ", name).strip()


def display_name(canonical: str) -> str:
    """Title case per word when the name has no lower-case letter (`YOUNG MAKAROV` becomes
    `Young Makarov`); otherwise the name as written."""
    if any(ch.islower() for ch in canonical):
        return canonical
    words = []
    for word in canonical.split(" "):
        low = word.lower()
        i = next((k for k, ch in enumerate(low) if ch.isalpha()), None)
        words.append(low if i is None else low[:i] + low[i].upper() + low[i + 1 :])
    return " ".join(words)


def strip_tags(name: str) -> str:
    """The Name without ASS override blocks or stray braces."""
    return re.sub(r"[{}]", "", _OVERRIDE.sub("", name))


def parse_name(raw: str) -> ParsedName:
    stripped = strip_tags(raw).strip()
    inner = bool(_THOUGHT.search(stripped))
    canonical = canonical_name(stripped)
    key = fold(canonical)
    if not key or key in SIGN_NAMES | SONG_NAMES:
        return ParsedName(raw, None, None, NameClass.IGNORED, inner)
    if _MULTI.search(canonical) or key in GROUP_WORDS:
        return ParsedName(raw, None, None, NameClass.MULTI, inner)
    cls = NameClass.ROLE if _ROLE.search(canonical) else NameClass.NAMED
    return ParsedName(raw, key, display_name(canonical), cls, inner)


def _one_edit(a: str, b: str) -> bool:
    """Levenshtein distance exactly 1."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        return a[i + 1 :] == b[i + 1 :]
    return a[i:] == b[i + 1 :]


def suggest_aliases(counts: Mapping[str, int]) -> list[tuple[str, str]]:
    """(rare, common) key pairs one edit apart, both at least SUGGEST_MIN_LENGTH characters,
    the rare key under RARE_MAX_LINES lines and the common one at least COMMON_RATIO times as
    many. Suggestions only: names are never matched fuzzily."""
    keys = sorted(k for k in counts if len(k) >= SUGGEST_MIN_LENGTH)
    out = []
    for rare in keys:
        n = counts[rare]
        if n <= 0 or n >= RARE_MAX_LINES:
            continue
        for common in keys:
            if counts[common] >= COMMON_RATIO * n and _one_edit(rare, common):
                out.append((rare, common))
    return out
