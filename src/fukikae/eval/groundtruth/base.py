"""Shared pieces of the ground-truth readers: the line type, usable names and aliases. Name
canonicalization and the subtitle/speech alignment check moved to
`fukikae.names` (R1) and are re-exported here. Subtitle cue parsing (SRT, WebVTT, ASS, TTML)
lives in `fukikae.subtitles`, which the pipeline uses too; it is re-exported here."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from fukikae.eval.speakers import label_key
from fukikae.names.align import (  # noqa: F401 - re-exported: the check moved to the pipeline (R1)
    MAX_OFFSET_S,
    MIN_GAIN,
    MIN_SHARE,
    OFFSET_STEP_S,
    AlignmentError,
    AlignmentReport,
    _union,
    alignment,
    check_alignment,
)
from fukikae.names.normalize import canonical_name  # noqa: F401 - re-exported (R1)
from fukikae.subtitles import (  # noqa: F401 - re-exported: the parsers moved to the pipeline (R1)
    COLOURS,
    COLOURS_BY_NAME,
    Cue,
    colour_from_class,
    colour_from_hex,
    colour_from_rgb,
    parse_ass_cues,
    parse_text_cues,
    parse_ttml_cues,
    read_cues,
    read_text,
    strip_markup,
)

IGNORED_NAMES = {"sign", "signs", "title", "caption", "note", "song", "op", "ed", "lyrics"}
_MULTI = re.compile(r"[&/,]|\band\b", re.IGNORECASE)


@dataclass(frozen=True)
class GroundTruthLine:
    """A stretch of time a named speaker is talking; `speaker` is canonicalized and aliased."""

    start: float
    end: float
    speaker: str


# ---- names and aliases ----


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
