"""Cross-episode identity metrics.

The bar (global constraints, floor per Ruling R6): a *recurring* character has at least
`min_speech` (default 5) seconds of
labeled speech in every episode compared. A recurring character *passes* when at least `keep`
of its labeled speech sits on its modal character id. The run passes when at least `share` of
recurring characters pass and no two recurring characters share a modal id; with no recurring
character the verdict is `insufficient` (Review Focus 2), never a pass.

Alongside the bar, as in the 2026-11-05 Fairy Tail results doc and `eval speakers`, over every
name labeled in every episode compared (any amount):

- `strict_fraction`: share of those names whose every labeled segment is on one character id
  (a segment with no id makes the name inconsistent);
- `weighted_share`: their speech on their modal id over all their speech.

Speech on no character id (an `extra` or `review` cluster) counts in a name's total and never
as its modal id. Unlabeled segments (no ground-truth name, or `?`) count nowhere except in
coverage, which is the share of ground-truth speech covered by segments that are labeled and
have a character id (with `gt`), else the share of labeled segment time that has an id.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from fukikae.eval.groundtruth import GroundTruthLine
from fukikae.eval.identity.segments import Segment
from fukikae.eval.speakers import IGNORED_LABELS, label_key

# Ruling R6: 20 s per episode let a rework-grade system pass on Fairy Tail (two characters
# reached it); labeled speech per character per episode is small, so the floor is 5 s.
MIN_SPEECH_S = 5.0
OVERALL = "(overall)"
KEEP = 0.8
SHARE = 0.9

Assignment = Mapping[tuple[str, int], int | None]


@dataclass(frozen=True)
class CharacterScore:
    name: str
    speech: float  # labeled seconds over all episodes compared
    by_episode: dict[str, float]
    modal_id: int | None
    share: float  # of `speech` on `modal_id`
    share_by_episode: dict[str, float]  # of each episode's speech on `modal_id`
    ids: dict[int | None, float]  # seconds per character id (None: no id)
    recurring: bool  # meets the bar's speech minimum in every episode
    passing: bool  # recurring and share >= keep
    labeled_everywhere: bool  # labeled in every episode compared (strict/weighted set)
    strict: bool  # every segment on one character id


@dataclass(frozen=True)
class IdPurity:
    speech: float
    dominant: str
    purity: float
    others: dict[str, float]  # name -> share of the id's speech


@dataclass(frozen=True)
class IdentityScore:
    recurring: int
    passing: int
    fraction: float
    merges: list[tuple[str, str, int]]  # recurring characters sharing a modal id
    strict_fraction: float
    purity_by_id: dict[int, IdPurity]
    coverage: float
    verdict: str  # pass | fail | insufficient
    labeled_recurring: int = 0
    weighted_share: float = 0.0
    episodes: list[str] = field(default_factory=list)
    characters: list[CharacterScore] = field(default_factory=list)


def _union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(x for x in iv if x[1] > x[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _length(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)


def _intersection(x: list[tuple[float, float]], y: list[tuple[float, float]]) -> float:
    i = j = 0
    total = 0.0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            total += hi - lo
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return total


def score(
    assign: Assignment,
    segs: Mapping[str, Sequence[Segment]],
    min_speech: float = MIN_SPEECH_S,
    keep: float = KEEP,
    share: float = SHARE,
    *,
    gt: Mapping[str, Sequence[GroundTruthLine]] | None = None,
) -> IdentityScore:
    """Score `assign` ((episode, index into segs[episode]) -> character id or None) over the
    episodes in `segs`. A segment missing from `assign` has no id."""
    episodes = list(segs)
    speech: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    on_id: dict[str, dict[str, dict[int | None, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    by_id: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    display: dict[str, str] = {}
    labeled_time = covered_time = 0.0
    covered: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ep, lst in segs.items():
        for i, s in enumerate(lst):
            if s.label is None or label_key(s.label) in IGNORED_LABELS:
                continue
            key = label_key(s.label)
            display.setdefault(key, s.label.strip())
            cid = assign.get((ep, i))
            d = s.duration
            speech[key][ep] += d
            on_id[key][ep][cid] += d
            labeled_time += d
            if cid is not None:
                by_id[cid][key] += d
                covered_time += d
                covered[ep].append((s.start, s.end))

    chars: list[CharacterScore] = []
    for key in sorted(speech, key=lambda k: (-sum(speech[k].values()), display[k])):
        ids: dict[int | None, float] = defaultdict(float)
        for per in on_id[key].values():
            for cid, d in per.items():
                ids[cid] += d
        total = sum(ids.values())
        real = {cid: d for cid, d in ids.items() if cid is not None}
        modal = max(sorted(real), key=lambda c: real[c]) if real else None
        frac = real[modal] / total if modal is not None and total > 0 else 0.0
        per_ep = {
            ep: on_id[key][ep].get(modal, 0.0) / speech[key][ep]
            if modal is not None and speech[key][ep]
            else 0.0
            for ep in episodes
            if ep in speech[key]
        }
        everywhere = len(episodes) >= 2 and all(speech[key].get(ep, 0.0) > 0 for ep in episodes)
        recurring = len(episodes) >= 2 and all(
            speech[key].get(ep, 0.0) >= min_speech for ep in episodes
        )
        chars.append(
            CharacterScore(
                name=display[key],
                speech=total,
                by_episode={ep: speech[key][ep] for ep in episodes if ep in speech[key]},
                modal_id=modal,
                share=frac,
                share_by_episode=per_ep,
                ids=dict(ids),
                recurring=recurring,
                passing=recurring and frac >= keep - 1e-9,
                labeled_everywhere=everywhere,
                strict=len(ids) == 1 and None not in ids,
            )
        )

    rec = [c for c in chars if c.recurring]
    passing = sum(c.passing for c in rec)
    fraction = passing / len(rec) if rec else 0.0
    by_modal: dict[int, list[str]] = defaultdict(list)
    for c in rec:
        if c.modal_id is not None:
            by_modal[c.modal_id].append(c.name)
    merges = [
        (a, b, cid)
        for cid in sorted(by_modal)
        for i, a in enumerate(sorted(by_modal[cid]))
        for b in sorted(by_modal[cid])[i + 1 :]
    ]
    every = [c for c in chars if c.labeled_everywhere]
    strict_fraction = sum(c.strict for c in every) / len(every) if every else 0.0
    every_total = sum(c.speech for c in every)
    weighted = (
        sum(c.ids.get(c.modal_id, 0.0) for c in every if c.modal_id is not None) / every_total
        if every_total > 0
        else 0.0
    )

    purity: dict[int, IdPurity] = {}
    for cid, names in sorted(by_id.items()):
        tot = sum(names.values())
        top = max(sorted(names), key=lambda k: names[k])
        purity[cid] = IdPurity(
            speech=tot,
            dominant=display[top],
            purity=names[top] / tot,
            others={
                display[k]: d / tot
                for k, d in sorted(names.items(), key=lambda kv: -kv[1])
                if k != top
            },
        )

    if gt is not None:
        gt_total = cov = 0.0
        for ep in episodes:
            lines = _union([(g.start, g.end) for g in gt.get(ep, [])])
            gt_total += _length(lines)
            cov += _intersection(_union(covered.get(ep, [])), lines)
        coverage = cov / gt_total if gt_total > 0 else 0.0
    else:
        coverage = covered_time / labeled_time if labeled_time > 0 else 0.0

    if not rec:
        verdict = "insufficient"
    elif fraction >= share - 1e-9 and not merges:
        verdict = "pass"
    else:
        verdict = "fail"
    return IdentityScore(
        recurring=len(rec),
        passing=passing,
        fraction=fraction,
        merges=merges,
        strict_fraction=strict_fraction,
        purity_by_id=purity,
        coverage=coverage,
        verdict=verdict,
        labeled_recurring=len(every),
        weighted_share=weighted,
        episodes=episodes,
        characters=chars,
    )


def nearest_centroid_accuracy(
    train_segs: Sequence[Segment],
    train_embs: np.ndarray,
    test_segs: Sequence[Segment],
    test_embs: np.ndarray,
) -> dict[str, float]:
    """Embedding separability without clustering or matching (Ruling R8): one centroid per
    character from the labeled training segments (normalized mean), each labeled test segment
    classified to the most cosine-similar centroid. Returns the speech-weighted share classified
    correctly per character, plus `OVERALL` over all scored speech. Test characters with no
    training speech are not scored; empty when nothing can be scored."""

    def labeled(segs: Sequence[Segment]) -> list[tuple[int, str]]:
        return [
            (i, label_key(s.label))
            for i, s in enumerate(segs)
            if s.label is not None and label_key(s.label) not in IGNORED_LABELS
        ]

    display: dict[str, str] = {}
    rows: dict[str, list[int]] = defaultdict(list)
    for i, key in labeled(train_segs):
        rows[key].append(i)
        display.setdefault(key, str(train_segs[i].label).strip())
    names = sorted(rows)
    if not names:
        return {}
    tr = np.asarray(train_embs, dtype=np.float64)
    cents = np.stack([tr[rows[k]].mean(axis=0) for k in names])
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    te = np.asarray(test_embs, dtype=np.float64)
    right: dict[str, float] = defaultdict(float)
    total: dict[str, float] = defaultdict(float)
    for i, key in labeled(test_segs):
        if key not in rows:
            continue
        v = te[i] / max(float(np.linalg.norm(te[i])), 1e-12)
        guess = names[int(np.argmax(cents @ v))]
        d = test_segs[i].duration
        total[key] += d
        if guess == key:
            right[key] += d
    if not total:
        return {}
    out = {display[k]: right[k] / total[k] for k in sorted(total, key=lambda k: -total[k])}
    out[OVERALL] = sum(right.values()) / sum(total.values())
    return out
