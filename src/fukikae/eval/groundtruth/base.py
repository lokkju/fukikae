"""Shared pieces of the ground-truth readers: the line type, name canonicalization and
aliases, and the subtitle/speech alignment check. Subtitle cue parsing (SRT, WebVTT, ASS, TTML)
lives in `fukikae.subtitles`, which the pipeline uses too; it is re-exported here."""

import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from fukikae.eval.speakers import label_key
from fukikae.models import Turn
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
    label: str = "",
) -> tuple[list[GroundTruthLine], AlignmentReport, bool]:
    """(lines, report, offset applied). Refuses below `min_share` at the best offset unless
    `force`; zero lines refuse the same way; moves the lines back by the best offset when
    that gains `min_gain` of overlap and reaches `min_share` (a junk offset is never applied)."""
    rep = alignment(lines, speech_turns)
    if not lines:
        if force:
            return lines, rep, False
        raise AlignmentError(
            f"no {label or 'speaker lines'}: the subtitles name no speaker, so every span "
            "would be '?'. Pass --force to continue anyway"
        )
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
