"""The subtitle/speech alignment check: how much of the lines' time overlaps detected speech,
and the global offset that best lines them up. Moved from `eval.groundtruth.base` (R1); works
on any frozen dataclass with `start` and `end` (eval's `GroundTruthLine`, the names stage's
lines)."""

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

import numpy as np

from fukikae.models import Turn

MIN_SHARE = 0.6
MIN_GAIN = 0.1
MAX_OFFSET_S = 10.0
OFFSET_STEP_S = 0.1


class TimedLine(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


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


def alignment[L: TimedLine](lines: list[L], speech_turns: list[Turn]) -> AlignmentReport:
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


def check_alignment[L: TimedLine](
    lines: list[L],
    speech_turns: list[Turn],
    force: bool = False,
    min_share: float = MIN_SHARE,
    min_gain: float = MIN_GAIN,
    label: str = "",
) -> tuple[list[L], AlignmentReport, bool]:
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
