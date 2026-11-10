"""Segmentation variants: the stretches of an episode that get one embedding and one identity.

`diarization`: the clean spans of each identity cluster (``names.inputs.identity_inputs``: the
names stage's ``<cluster>~<slug>`` sub-clusters when names were used, else ``04-diarize``'s
clusters; or the turns ``assignments.json`` recorded, with ``--from-assignments``), that is the
speaker's turns minus any
overlap with another speaker, padded in by 0.1 s, at least 1.5 s; `stages.diarize.clean_spans`),
all of them, clusters in order of first appearance and each cluster's spans in time order.
A span is labeled with the ground-truth name covering at least 60% of it (the rule
`eval auto-label` uses), else ``None``.

`subtitles`: the ground-truth lines themselves. Each speaker's lines are merged where they
touch, then any time another speaker's line covers is removed (overlapped speech); pieces
shorter than 1.0 s are dropped.

`asr`: sentences from ``03-asr/transcript.json`` (word timings; a sentence ends at terminal
punctuation, in a word or in the segment text right after it, or at a gap of more than
`max_gap` between words), each cut into word windows of about `window` seconds that a speaker
embedder embeds; the sentence is split between consecutive windows whose cosine distance
exceeds `split`. Overlapped speech goes: any time two ground-truth speakers' lines share, or
two ``04-diarize`` speakers' turns share. Pieces shorter than 1.0 s are dropped and the rest
labeled by the 60% rule of `diarization`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fukikae.artifacts import read_json
from fukikae.eval.groundtruth import GroundTruthLine
from fukikae.eval.groundtruth.ass_names import UNKNOWN, Line, label_spans
from fukikae.eval.speakers import label_key
from fukikae.models import Diarization, Transcript, Turn, Word
from fukikae.names.inputs import identity_inputs
from fukikae.stages.diarize import clean_spans
from fukikae.workspace import Workspace

Kind = Literal["diarization", "subtitles", "asr"]
KINDS: tuple[str, ...] = ("diarization", "subtitles", "asr")
ASR_WINDOW_S = 1.5
# Adjacent 1.5 s ECAPA windows of one speaker: median distance 0.62, 95th pct 0.78
# (fairytail E12, a tuning episode).
ASR_SPLIT = 0.8
ASR_MAX_GAP_S = 1.0
# Sentence-final punctuation in the scripts the test sets use (Latin, CJK, Korean uses Latin).
_TERMINAL = re.compile(r"[.!?。！？．…]")

WindowEmbed = Callable[[list[tuple[float, float]]], np.ndarray]
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


def _ends_sentence(word: Word, after: str) -> bool:
    if _TERMINAL.search(word.text.strip()[-1:] or ""):
        return True
    return bool(after) and bool(_TERMINAL.match(after.lstrip()[:1]))


def asr_sentences(transcript: Transcript, max_gap: float = ASR_MAX_GAP_S) -> list[list[Word]]:
    """The transcript's words grouped into sentences, in time order. Segments without word
    timings are skipped."""
    out: list[list[Word]] = []
    for seg in transcript.segments:
        words = [w for w in seg.words or [] if w.end > w.start]
        if not words:
            continue
        text, pos = seg.text, 0
        cur: list[Word] = []
        for k, w in enumerate(words):
            if cur and w.start - cur[-1].end > max_gap:
                out.append(cur)
                cur = []
            cur.append(w)
            token = w.text.strip()
            at = text.find(token, pos) if token else -1
            after = ""
            if at >= 0:
                pos = at + len(token)
                nxt = words[k + 1].text.strip() if k + 1 < len(words) else ""
                stop = text.find(nxt, pos) if nxt else len(text)
                after = text[pos : stop if stop >= 0 else len(text)]
            if _ends_sentence(w, after):
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
    return out


def _windows(words: list[Word], window: float) -> list[list[Word]]:
    """Consecutive runs of words spanning at least `window` seconds; a tail shorter than
    half a window joins the run before it."""
    out: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        if cur[-1].end - cur[0].start >= window - 1e-9:
            out.append(cur)
            cur = []
    if cur:
        if out and cur[-1].end - cur[0].start < window / 2:
            out[-1] = out[-1] + cur
        else:
            out.append(cur)
    return out


def _overlaps(iv_by_speaker: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Time that two speakers' intervals share."""
    merged = {k: _merge(v) for k, v in iv_by_speaker.items()}
    keys = sorted(merged)
    out: list[tuple[float, float]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            for x, y in merged[a]:
                for c, d in merged[b]:
                    lo, hi = max(x, c), min(y, d)
                    if hi > lo:
                        out.append((lo, hi))
    return _merge(out)


def asr_segments(
    episode: str,
    transcript: Transcript,
    gt: list[GroundTruthLine],
    window_embed: WindowEmbed,
    *,
    window: float = ASR_WINDOW_S,
    split: float = ASR_SPLIT,
    max_gap: float = ASR_MAX_GAP_S,
    min_len: float = MIN_SUBTITLE_SECONDS,
    turns: Sequence[Turn] | None = None,
    min_overlap: float = MIN_OVERLAP,
) -> list[Segment]:
    """See the module docstring. `window_embed` gets every window of every sentence with
    more than one window in a single call and returns one row per window."""
    sentences = [_windows(s, window) for s in asr_sentences(transcript, max_gap)]
    spans = [(ws[0].start, ws[-1].end) for sent in sentences if len(sent) > 1 for ws in sent]
    rows = np.asarray(window_embed(spans), dtype=np.float64) if spans else np.zeros((0, 1))
    if len(rows) != len(spans):
        raise ValueError(f"{len(spans)} word windows but {len(rows)} embeddings")
    if len(rows):
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        rows = rows / np.where(norms == 0, 1.0, norms)
    pieces: list[tuple[float, float]] = []
    at = 0
    for sent in sentences:
        if len(sent) == 1:
            pieces.append((sent[0][0].start, sent[0][-1].end))
            continue
        r = rows[at : at + len(sent)]
        at += len(sent)
        first = 0
        for k in range(1, len(sent)):
            if 1.0 - float(r[k - 1] @ r[k]) > split:
                pieces.append((sent[first][0].start, sent[k - 1][-1].end))
                first = k
        pieces.append((sent[first][0].start, sent[-1][-1].end))
    by_gt: dict[str, list[tuple[float, float]]] = {}
    for g in gt:
        if g.end > g.start:
            by_gt.setdefault(label_key(g.speaker), []).append((g.start, g.end))
    by_turn: dict[str, list[tuple[float, float]]] = {}
    for t in turns or []:
        by_turn.setdefault(t.speaker, []).append((t.start, t.end))
    cut = _merge(_overlaps(by_gt) + _overlaps(by_turn))
    kept = [
        (round(a, 6), round(b, 6))
        for a, b in _subtract(sorted(pieces), cut)
        if b - a >= min_len - 1e-9
    ]
    lines = [Line(g.start, g.end, g.speaker, "") for g in gt]
    return [
        Segment(episode, a, b, None if lab == UNKNOWN else lab)
        for (a, b), lab in zip(kept, label_spans(kept, lines, min_overlap), strict=True)
    ]


def segments(
    ws: Workspace | Path,
    episode: str,
    kind: Kind,
    gt: list[GroundTruthLine],
    window_embed: WindowEmbed | None = None,
    options: dict[str, Any] | None = None,
    turns: list[Turn] | None = None,
) -> list[Segment]:
    """The episode's segments for `kind`; `diarization` reads the identity turns
    (`identity_inputs`: ``04c-names/turns.json`` when names were used, else
    ``04-diarize/turns.json``) unless `turns` gives them (``assignments.turns``, so labels match
    the assignments' clusters), `asr` reads ``03-asr/transcript.json`` and the turns (for
    overlap) and needs `window_embed`; `options` are `asr_segments`' keyword options (window,
    split, max_gap)."""
    if kind == "subtitles":
        return subtitle_segments(episode, gt)
    if kind not in ("diarization", "asr"):
        raise ValueError(f"unknown segmentation {kind!r}; use one of {', '.join(KINDS)}")
    w = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
    if turns is None:
        path = identity_inputs(w.work_dir(episode))[0]
        if not path.is_file():
            raise FileNotFoundError(f"{episode}: no diarization at {path}")
        turns = read_json(path, Diarization).turns
    if kind == "diarization":
        return diarization_segments(episode, turns, gt)
    if window_embed is None:
        raise ValueError("asr segmentation needs a word-window embedder")
    opts = dict(options or {})
    unknown = set(opts) - {"window", "split", "max_gap"}
    if unknown:
        raise ValueError(f"unknown asr segmentation option(s): {', '.join(sorted(unknown))}")
    tpath = w.work_dir(episode) / "03-asr" / "transcript.json"
    if not tpath.is_file():
        raise FileNotFoundError(f"{episode}: no transcript at {tpath}")
    transcript = read_json(tpath, Transcript)
    if not transcript.word_timing:
        raise ValueError(f"{episode}: the transcript has no word timings")
    return asr_segments(
        episode,
        transcript,
        gt,
        window_embed,
        turns=turns,
        **{k: float(v) for k, v in opts.items()},
    )
