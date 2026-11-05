"""Segmentation variants: the stretches of an episode that get one embedding and one identity.

`diarization`: the clean spans of each ``04-diarize`` cluster (the speaker's turns minus any
overlap with another speaker, padded in by 0.1 s, at least 1.5 s; `stages.diarize.clean_spans`),
all of them, clusters in order of first appearance and each cluster's spans in time order.
A span is labeled with the ground-truth name covering at least 60% of it (the rule
`eval auto-label` uses), else ``None``.

`subtitles`: the ground-truth lines themselves. Each speaker's lines are merged where they
touch, then any time another speaker's line covers is removed (overlapped speech); pieces
shorter than 1.0 s are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fukikae.artifacts import read_json
from fukikae.eval.groundtruth import GroundTruthLine
from fukikae.eval.groundtruth.ass_names import UNKNOWN, Line, label_spans
from fukikae.eval.speakers import label_key
from fukikae.models import Diarization, Turn
from fukikae.stages.diarize import clean_spans
from fukikae.workspace import Workspace

Kind = Literal["diarization", "subtitles"]
KINDS: tuple[str, ...] = ("diarization", "subtitles")
MIN_OVERLAP = 0.6
MIN_SUBTITLE_SECONDS = 1.0


@dataclass(frozen=True)
class Segment:
    episode: str
    start: float
    end: float
    label: str | None  # ground-truth speaker; None when unlabeled
    cluster: str | None = None  # the diarization cluster; None for subtitle segments

    @property
    def duration(self) -> float:
        return self.end - self.start


def diarization_segments(
    episode: str,
    turns: list[Turn],
    gt: list[GroundTruthLine],
    min_overlap: float = MIN_OVERLAP,
) -> list[Segment]:
    lines = [Line(g.start, g.end, g.speaker, "") for g in gt]
    out: list[Segment] = []
    for cluster in dict.fromkeys(t.speaker for t in turns):
        spans = sorted(clean_spans(turns, cluster))
        for (a, b), lab in zip(spans, label_spans(spans, lines, min_overlap), strict=True):
            out.append(Segment(episode, a, b, None if lab == UNKNOWN else lab, cluster))
    return out


def _merge(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract(
    iv: list[tuple[float, float]], cut: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in iv:
        pieces = [(a, b)]
        for c, d in cut:
            nxt: list[tuple[float, float]] = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if c > x:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            pieces = nxt
        out.extend(pieces)
    return out


def subtitle_segments(
    episode: str, gt: list[GroundTruthLine], min_len: float = MIN_SUBTITLE_SECONDS
) -> list[Segment]:
    by: dict[str, list[tuple[float, float]]] = {}
    shown: dict[str, str] = {}
    for g in gt:
        if g.end > g.start:
            key = label_key(g.speaker)
            shown.setdefault(key, g.speaker)
            by.setdefault(key, []).append((g.start, g.end))
    merged = {spk: _merge(iv) for spk, iv in by.items()}
    out: list[Segment] = []
    for spk, iv in merged.items():
        others = _merge([x for o, ivs in merged.items() if o != spk for x in ivs])
        for a, b in _subtract(iv, others):
            if b - a >= min_len - 1e-9:
                out.append(Segment(episode, a, b, shown[spk]))
    return sorted(out, key=lambda s: (s.start, s.end))


def segments(
    ws: Workspace | Path, episode: str, kind: Kind, gt: list[GroundTruthLine]
) -> list[Segment]:
    """The episode's segments for `kind`; `diarization` reads ``04-diarize/turns.json``."""
    if kind == "subtitles":
        return subtitle_segments(episode, gt)
    if kind != "diarization":
        raise ValueError(f"unknown segmentation {kind!r}; use one of {', '.join(KINDS)}")
    w = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
    path = w.work_dir(episode) / "04-diarize" / "turns.json"
    if not path.is_file():
        raise FileNotFoundError(f"{episode}: no diarization at {path}")
    return diarization_segments(episode, read_json(path, Diarization).turns, gt)
