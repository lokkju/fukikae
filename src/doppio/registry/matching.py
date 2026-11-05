"""Match diarized clusters against the registry's actors, and pin them across re-runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from doppio.models import Assignments, ClusterAssignment, Diarization, Turn
from doppio.registry.store import Actor, Character

PIN_OVERLAP = 0.5


@dataclass(frozen=True)
class ClusterEvidence:
    cluster: str
    embedding: np.ndarray | None  # normalized mean of the cluster's rows; None with no rows
    clean_seconds: float


def mean_embedding(rows: np.ndarray) -> np.ndarray | None:
    """Normalized mean of the rows, or None with zero rows (or a zero mean)."""
    if len(rows) == 0:
        return None
    mean = np.asarray(rows, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return None if norm == 0.0 else mean / norm


def load_clusters(path: Path) -> tuple[list[ClusterEvidence], dict[str, np.ndarray]]:
    """Read a diarize stage's clusters.npz: evidence per cluster and its embedding rows."""
    rows: dict[str, np.ndarray] = {}
    evidence: list[ClusterEvidence] = []
    with np.load(path) as z:
        for label in [str(x) for x in z["labels"]]:
            emb = np.asarray(z[f"{label}/embeddings"], dtype=np.float32)
            rows[label] = emb
            evidence.append(
                ClusterEvidence(label, mean_embedding(emb), float(z[f"{label}/clean_seconds"]))
            )
    return evidence, rows


def match(
    evidence: list[ClusterEvidence],
    actors: list[Actor],
    threshold: float,
    min_new_seconds: float,
    characters: list[Character] | None = None,
    claimed: set[int] | None = None,
) -> list[ClusterAssignment]:
    """Assign each cluster to an actor (`matched`), a new actor (`new`) or nobody (`extra`).

    When several clusters best-match the same actor, only the highest-scoring is `matched`;
    the others are `review` (kept apart, with their score and actor). A cluster whose best
    match is in `claimed` (actors another cluster of this episode already holds, such as a
    pinned one) is `review` too: one character never takes two clusters of an episode
    silently. `new` carries no
    actor or character yet: the caller creates them. `matched` takes the actor's
    lowest-id character from `characters`.
    """
    first_char: dict[int, int] = {}
    for c in sorted(characters or [], key=lambda c: c.id):
        first_char.setdefault(c.actor_id, c.id)
    scored = [a for a in actors if a.centroid is not None]

    out: dict[str, ClusterAssignment] = {}
    best: dict[str, tuple[int, float]] = {}
    for ev in evidence:
        top: tuple[int, float] | None = None
        if ev.embedding is not None:
            for a in scored:
                assert a.centroid is not None
                denom = float(np.linalg.norm(a.centroid) * np.linalg.norm(ev.embedding))
                s = float(np.dot(a.centroid, ev.embedding)) / denom if denom else 0.0
                if top is None or s > top[1]:
                    top = (a.id, s)
        score = None if top is None else top[1]
        if top is not None and top[1] >= threshold:
            best[ev.cluster] = top
            continue
        status = (
            "new" if ev.clean_seconds >= min_new_seconds and ev.embedding is not None else "extra"
        )
        out[ev.cluster] = ClusterAssignment(
            cluster=ev.cluster,
            actor_id=None,
            character_id=None,
            score=score,
            clean_seconds=ev.clean_seconds,
            status=status,
        )

    secs = {e.cluster: e.clean_seconds for e in evidence}
    taken = claimed or set()
    winners: dict[int, str] = {}
    for cluster, (aid, s) in best.items():
        if aid in taken:
            continue
        if aid not in winners or s > best[winners[aid]][1]:
            winners[aid] = cluster
    for cluster, (aid, s) in best.items():
        won = winners.get(aid) == cluster
        out[cluster] = ClusterAssignment(
            cluster=cluster,
            actor_id=aid,
            character_id=first_char.get(aid) if won else None,
            score=s,
            clean_seconds=secs[cluster],
            status="matched" if won else "review",
        )
    return [out[e.cluster] for e in evidence]


def _intervals(turns: list[Turn]) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        out.setdefault(t.speaker, []).append((t.start, t.end))
    return out


def pin_sources(
    previous: Assignments, previous_turns: Diarization, current_turns: Diarization
) -> dict[str, str]:
    """Map each current cluster that pins to the previous cluster it inherits from.

    A current cluster inherits from the previous cluster it overlaps most in time when that
    overlap exceeds half of the current cluster's total turn time. Only previous assignments
    with a character (matched, new, pinned) are inherited; `extra` and `review` clusters are
    matched afresh on each run.
    """
    assigned = {c.cluster for c in previous.clusters if c.character_id is not None}
    prev_iv = _intervals(previous_turns.turns)
    sources: dict[str, str] = {}
    for label, cur in _intervals(current_turns.turns).items():
        total = sum(e - s for s, e in cur)
        if total <= 0:
            continue
        best_label, best_overlap = None, 0.0
        for plabel, piv in prev_iv.items():
            if plabel not in assigned:
                continue
            ov = sum(max(0.0, min(e, pe) - max(s, ps)) for s, e in cur for ps, pe in piv)
            if ov > best_overlap:
                best_label, best_overlap = plabel, ov
        if best_label is not None and best_overlap > PIN_OVERLAP * total:
            sources[label] = best_label
    return sources


def pin(
    previous: Assignments, previous_turns: Diarization, current_turns: Diarization
) -> dict[str, ClusterAssignment]:
    """Carry earlier assignments onto the current clusters (see `pin_sources`), as `pinned`."""
    prev_by_label = {c.cluster: c for c in previous.clusters}
    return {
        cur: prev_by_label[src].model_copy(update={"cluster": cur, "status": "pinned"})
        for cur, src in pin_sources(previous, previous_turns, current_turns).items()
    }
