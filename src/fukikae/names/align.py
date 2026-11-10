"""Subtitle lines against diarize turns.

The alignment check (how much of the lines' time overlaps detected speech, and the global
offset that best lines them up) moved from `eval.groundtruth.base` (R1) and works on any frozen
dataclass with `start` and `end`. `attribute` gives each diarize turn the name key that covers
it, cutting the few turns that hold two names; `refine_clusters` moves the diarize stage's
embedding rows onto the refined `<cluster>~<slug>` labels without re-embedding, dropping rows
that cross a cut (D29), so the names stage stays CPU-only."""

import hashlib
import itertools
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from fukikae.artifacts import atomic_path
from fukikae.models import Turn
from fukikae.names.models import LabelInfo, NamedLine, Piece
from fukikae.names.normalize import NameClass

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


# ---- attribution: named lines onto diarize turns ----

TURN_SHARE = 0.6  # one key covering this share of a turn names it
CUT_SIDE_S = 0.5  # each side of a cut covers at least this much of the turn
MIN_PIECE_S = 0.3  # a shorter piece joins its longer neighbour
UNVOICED_S = 0.2  # a named line with less turn under it is unvoiced
CONTAIN = 0.9  # an embedding row this much inside one piece moves to it


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _slug(key: str) -> str | None:
    """`[a-z0-9-]` from the key once accents are stripped (NFKD, combining marks dropped, so
    `chloé` is `chloe`), or None when what is left isn't ASCII or has no letter or digit."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", key) if not unicodedata.combining(c))
    if not plain.isascii():
        return None
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or None


def label_for(cluster: str, key: str, *, hashed: bool = False) -> str:
    """`<cluster>~<slug>`: the key's `[a-z0-9-]` slug when it has one, else `n` plus the first
    8 hex digits of sha256(key). `hashed` forces the second form, for keys whose slugs collide
    within one cluster."""
    slug = None if hashed else _slug(key)
    if slug is None:
        slug = "n" + hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{cluster}~{slug}"


def _assign_lines(turns: list[Turn], lines: list[NamedLine]) -> dict[int, list[NamedLine]]:
    """Turn index -> the lines attributed to it. A line counts for every turn it overlaps,
    except a turn that overlaps another turn the line covers more (ties go to the earlier
    turn): when two turns overlap, a line covering both names only one."""
    by_turn: dict[int, list[NamedLine]] = defaultdict(list)
    for ln in lines:
        hits = [
            (i, _overlap(ln.start, ln.end, t.start, t.end))
            for i, t in enumerate(turns)
            if t.start < ln.end and ln.start < t.end
        ]
        hits = [(i, ov) for i, ov in hits if ov > 0]
        for i, ov in hits:
            ti = turns[i]
            beaten = any(
                (ov2 > ov or (ov2 == ov and j < i))
                and turns[j].start < ti.end
                and ti.start < turns[j].end
                for j, ov2 in hits
                if j != i
            )
            if not beaten:
                by_turn[i].append(ln)
    return by_turn


def _union_length(spans: list[tuple[float, float]]) -> float:
    total, reach = 0.0, float("-inf")
    for a, b in sorted(spans):
        a = max(a, reach)
        if b > a:
            total += b - a
            reach = b
    return total


_GAP = "\0gap"  # an elementary stretch no line covers


def _cut(
    t: Turn, lines: list[NamedLine], keys: set[str], min_piece_s: float
) -> list[tuple[float, float, str | None]]:
    """(start, end, key) pieces of a turn cut between lines of different keys. The turn is split
    at every line boundary: a stretch one key's lines alone cover takes that key; a stretch a
    multi line covers, or lines of two keys cover at once (a line nested in another key's, or
    the overlap of two), is unnamed. A stretch no line covers goes to its neighbours: to the
    one piece when both sides have the same key, else split at its midpoint (the midpoint of
    the furthest end reached so far and the next line's start); at the turn's ends it joins
    the first or last piece. So cuts are monotonic and every keyed piece overlaps a line of its
    key. A piece under `min_piece_s` then joins its longer neighbour."""
    segs: list[tuple[float, float, str | None]] = []
    for ln in lines:
        if ln.cls is NameClass.MULTI:
            key = None
        elif ln.key in keys:
            key = ln.key
        else:
            continue
        a, b = max(ln.start, t.start), min(ln.end, t.end)
        if b > a:
            segs.append((a, b, key))
    bounds = sorted({t.start, t.end, *(x for a, b, _ in segs for x in (a, b))})
    runs: list[list] = []
    for a, b in itertools.pairwise(bounds):
        active = {k for s0, s1, k in segs if s0 < b and a < s1}
        lab: str | None = _GAP if not active else (next(iter(active)) if len(active) == 1 else None)
        if runs and runs[-1][2] == lab:
            runs[-1][1] = b
        else:
            runs.append([a, b, lab])
    pieces: list[list] = []
    for k, (a, b, lab) in enumerate(runs):
        if lab != _GAP:
            pieces.append([a, b, lab])
            continue
        left = pieces[-1] if pieces else None
        right = runs[k + 1] if k + 1 < len(runs) else None  # never a gap: runs alternate
        if left is None and right is None:
            pieces.append([a, b, None])
        elif left is None:
            assert right is not None
            right[0] = a
        elif right is None or right[2] == left[2]:
            left[1] = b
        else:
            mid = round((a + b) / 2, 6)
            left[1] = mid
            right[0] = mid
    pieces = _merge_same(pieces)
    while len(pieces) > 1:
        short = [k for k, p in enumerate(pieces) if p[1] - p[0] < min_piece_s - 1e-9]
        if not short:
            break
        k = min(short, key=lambda k: pieces[k][1] - pieces[k][0])
        left = pieces[k - 1] if k > 0 else None
        right = pieces[k + 1] if k + 1 < len(pieces) else None
        into = (
            k - 1
            if right is None or (left is not None and left[1] - left[0] >= right[1] - right[0])
            else k + 1
        )
        lo, hi = sorted((k, into))
        keep = pieces[into][2]
        pieces[lo : hi + 1] = [[pieces[lo][0], pieces[hi][1], keep]]
        pieces = _merge_same(pieces)
    return [(p[0], p[1], p[2]) for p in pieces]


def _merge_same(pieces: list[list]) -> list[list]:
    """Neighbours with the same key are one piece."""
    merged: list[list] = []
    for p in pieces:
        if merged and merged[-1][2] == p[2]:
            merged[-1][1] = p[1]
        else:
            merged.append(p)
    return merged


def attribute(
    turns: list[Turn],
    lines: list[NamedLine],
    *,
    turn_share: float = TURN_SHARE,
    cut_side_s: float = CUT_SIDE_S,
    min_piece_s: float = MIN_PIECE_S,
    hidden: Collection[int] = (),
) -> tuple[list[Piece], list[int]]:
    """Pieces covering every turn, in the turns' order, and the ids of unvoiced lines (named
    or role lines with under UNVOICED_S of turn under them).

    A turn where two or more keys each cover at least `cut_side_s` and together reach
    `turn_share` of it is cut between them; otherwise a key covering `turn_share` of it names
    the whole turn; otherwise it is the cluster's unnamed remainder. A key's cover is the union
    of its lines' spans within the turn. Named and role lines count for their key (inner-voice
    lines too); multi lines name nothing and are never a side of a cut. `hidden` ids are left
    out altogether: they name nothing, cut nothing, and are in no piece's `line_ids`. Keys whose
    slugs collide within one cluster get hashed labels."""
    hidden = set(hidden)
    visible = [ln for ln in lines if ln.id not in hidden]
    keyed = [ln for ln in visible if ln.key is not None]
    by_turn = _assign_lines(turns, visible)
    staged: list[tuple[int, Turn, float, float, str | None, list[int]]] = []
    for i, t in enumerate(turns):
        mine = by_turn.get(i, [])
        dur = t.end - t.start
        spans_by_key: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for ln in mine:
            if ln.key is not None and ln.cls is not NameClass.MULTI:
                a, b = max(ln.start, t.start), min(ln.end, t.end)
                if b > a:
                    spans_by_key[ln.key].append((a, b))
        cover = {k: _union_length(v) for k, v in spans_by_key.items()}
        sides = {k for k, s in cover.items() if s >= cut_side_s - 1e-9}
        spans: list[tuple[float, float, str | None]]
        if len(sides) >= 2 and sum(cover[k] for k in sides) >= turn_share * dur - 1e-9:
            spans = _cut(t, mine, sides, min_piece_s)
        else:
            best = max(cover, key=lambda k: (cover[k], k), default=None)
            named = best is not None and dur > 0 and cover[best] >= turn_share * dur - 1e-9
            spans = [(t.start, t.end, best if named else None)]
        for a, b, key in spans:
            ids = sorted(ln.id for ln in mine if _overlap(ln.start, ln.end, a, b) > 0)
            staged.append((i, t, a, b, key, ids))
    clash = _slug_clashes((t.speaker, key) for _, t, _, _, key, _ in staged if key is not None)
    pieces = [
        Piece(
            start=a,
            end=b,
            cluster=t.speaker,
            label=t.speaker
            if key is None
            else label_for(t.speaker, key, hashed=(t.speaker, key) in clash),
            key=key,
            line_ids=ids,
            turn=i,
        )
        for i, t, a, b, key, ids in staged
    ]
    unvoiced = [ln.id for ln in keyed if _voiced(ln, turns) < UNVOICED_S - 1e-9]
    return pieces, sorted(unvoiced)


def _slug_clashes(pairs: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """The (cluster, key) pairs whose key shares its slug with another key of the same cluster:
    judged per cluster, so a label never changes because an unrelated key appears elsewhere."""
    slugs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cluster, key in set(pairs):
        slug = _slug(key)
        if slug is not None:
            slugs[(cluster, slug)].add(key)
    return {(c, k) for (c, _), ks in slugs.items() if len(ks) > 1 for k in ks}


def _voiced(ln: NamedLine, turns: list[Turn]) -> float:
    """Seconds of the line under the union of the turns."""
    spans = sorted(
        (max(ln.start, t.start), min(ln.end, t.end))
        for t in turns
        if t.start < ln.end and ln.start < t.end
    )
    total, reach = 0.0, ln.start
    for a, b in spans:
        a = max(a, reach)
        if b > a:
            total += b - a
            reach = b
    return total


def refined_turns(pieces: list[Piece]) -> list[Turn]:
    """The refined diarization: one turn per piece, labelled with the piece's label."""
    return [Turn(start=p.start, end=p.end, speaker=p.label) for p in pieces]


def label_infos(pieces: list[Piece], lines: list[NamedLine]) -> dict[str, LabelInfo]:
    """Each refined label's parent cluster, key, display form and class, in first-appearance
    order. The display form is the key's most common one among the lines. Pass only the
    visible lines (never hidden ones), so a hidden line can't lend a key its display form or
    class."""
    displays: dict[str, Counter[str]] = defaultdict(Counter)
    classes: dict[str, NameClass] = {}
    for ln in lines:
        if ln.key is not None:
            if ln.display:
                displays[ln.key][ln.display] += 1
            if ln.cls is NameClass.NAMED or ln.key not in classes:
                classes[ln.key] = ln.cls
    out: dict[str, LabelInfo] = {}
    for p in pieces:
        if p.label in out:
            continue
        if p.key is None:
            out[p.label] = LabelInfo(parent=p.cluster, key=None, display=None, cls="remainder")
            continue
        common = displays[p.key].most_common(1)
        cls = "role" if classes.get(p.key) is NameClass.ROLE else "named"
        out[p.label] = LabelInfo(
            parent=p.cluster,
            key=p.key,
            display=common[0][0] if common else p.key,
            cls=cls,
            slug_clash=p.label != label_for(p.cluster, p.key),
        )
    return out


def excluded_spans(lines: list[NamedLine]) -> list[tuple[float, float]]:
    """Inner-voice and multi lines' spans, in start order: never clone references."""
    return sorted((ln.start, ln.end) for ln in lines if ln.inner or ln.cls is NameClass.MULTI)


def refine_clusters(
    clusters_npz: Path,
    pieces: list[Piece],
    turns_out: list[Turn],
    out: Path,
    *,
    contain: float = CONTAIN,
) -> float:
    """Write the refined `clusters.npz` in the diarize stage's format and return the seconds of
    embedding rows dropped. A row (a clean span of one of its cluster's turns) at least
    `contain` inside one of that cluster's pieces moves to the piece's label; a row across a cut
    is dropped (nothing is re-embedded, D29). Labels are in first-appearance order of
    `turns_out`; `clean_seconds` per label is recomputed over `turns_out`, and rows stay
    longest first."""
    from fukikae.stages.diarize import clean_spans

    by_cluster: dict[str, list[Piece]] = defaultdict(list)
    for p in pieces:
        by_cluster[p.cluster].append(p)
    labels: list[str] = []
    for t in turns_out:
        if t.speaker not in labels:
            labels.append(t.speaker)
    rows: dict[str, list[tuple[float, float, np.ndarray]]] = defaultdict(list)
    dropped = 0.0
    with np.load(clusters_npz) as z:
        src_labels = [str(x) for x in z["labels"]]
        dim = next((int(z[f"{lb}/embeddings"].shape[1]) for lb in src_labels), 0)
        for lb in src_labels:
            emb = z[f"{lb}/embeddings"]
            spans = z[f"{lb}/spans"]
            for vec, (s, e) in zip(emb, spans, strict=True):
                s, e = float(s), float(e)
                best = max(
                    by_cluster.get(lb, []),
                    key=lambda p: _overlap(s, e, p.start, p.end),
                    default=None,
                )
                if (
                    best is not None
                    and _overlap(s, e, best.start, best.end) >= contain * (e - s) - 1e-9
                ):
                    if best.label not in labels:
                        raise ValueError(
                            f"refine_clusters: a row of {lb} moves to {best.label}, which "
                            "turns_out does not hold; pass refined_turns(pieces)"
                        )
                    rows[best.label].append((s, e, vec))
                else:
                    dropped += e - s
    arrays: dict[str, Any] = {"labels": np.array(labels, dtype=str)}
    for lb in labels:
        kept = sorted(rows.get(lb, []), key=lambda r: r[1] - r[0], reverse=True)
        n = len(kept)
        arrays[f"{lb}/embeddings"] = (
            np.stack([r[2] for r in kept]).astype(np.float32)
            if kept
            else np.zeros((0, dim), dtype=np.float32)
        )
        arrays[f"{lb}/spans"] = np.array([(r[0], r[1]) for r in kept], dtype=np.float64).reshape(
            n, 2
        )
        clean = clean_spans(turns_out, lb)
        arrays[f"{lb}/clean_seconds"] = np.array(
            float(sum(e - s for s, e in clean)), dtype=np.float64
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        _savez_stable(tmp, arrays)
    return round(dropped, 6)


_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _savez_stable(path: Path, arrays: dict[str, Any]) -> None:
    """`np.savez`'s format (uncompressed zip of `<name>.npy` members) with every member stamped
    1980-01-01, so the same arrays always give the same bytes and an unchanged rerun leaves the
    next stage's input hash as it was."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)
