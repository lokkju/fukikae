"""Matchers: resolve one episode's segments to registry characters, against an in-memory
registry state built from the episodes resolved before it.

`Matcher.fit_episode(state, episode, embs, segs)` returns one character id per segment
(None for a segment no character takes: an extra, or a cluster sent to review) and updates
`state` the way the characters stage updates the registry. The identity harness replays
whole series through it offline; the characters stage can select a matcher by name later
(Task 7). Segments are any objects with ``start``, ``end``, ``duration`` and ``cluster``
attributes (`fukikae.eval.identity.segments.Segment`); this module imports nothing from eval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from fukikae.models import ClusterAssignment
from fukikae.registry.matching import ClusterEvidence, match, mean_embedding
from fukikae.registry.store import Actor, Character

# The diarize stage embeds at most this many spans per cluster (the longest), so a cluster's
# exemplars are those rows; `stages.diarize.MAX_SPANS_PER_CLUSTER`.
MAX_ROWS_PER_CLUSTER = 50


class SegmentLike(Protocol):
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def duration(self) -> float: ...
    @property
    def cluster(self) -> str | None: ...


@dataclass
class RegistryState:
    """Actors with their exemplar rows, one character per actor, ids from 1 as a fresh
    registry hands them out. Centroids are the normalized mean of all of an actor's rows
    (`Registry.recompute_centroid`)."""

    exemplars: dict[int, list[np.ndarray]] = field(default_factory=dict)
    character_of: dict[int, int] = field(default_factory=dict)
    _next_actor: int = 1
    _next_character: int = 1

    @property
    def actors(self) -> list[int]:
        return sorted(self.exemplars)

    def centroid(self, actor_id: int) -> np.ndarray | None:
        rows = self.exemplars[actor_id]
        return mean_embedding(np.vstack(rows)) if rows else None

    def actor_list(self) -> list[Actor]:
        return [Actor(a, None, None, self.centroid(a), False) for a in self.actors]

    def character_list(self) -> list[Character]:
        return [Character(c, a, None, None, False) for a, c in sorted(self.character_of.items())]

    def create(self, rows: np.ndarray) -> tuple[int, int]:
        aid, cid = self._next_actor, self._next_character
        self._next_actor += 1
        self._next_character += 1
        self.exemplars[aid] = [np.asarray(rows, dtype=np.float32)]
        self.character_of[aid] = cid
        return aid, cid

    def add(self, actor_id: int, rows: np.ndarray) -> None:
        if len(rows):
            self.exemplars[actor_id].append(np.asarray(rows, dtype=np.float32))


class Matcher(Protocol):
    name: ClassVar[str]

    def fit_episode(
        self,
        state: RegistryState,
        episode: str,
        embs: np.ndarray,
        segs: Sequence[SegmentLike],
    ) -> list[int | None]: ...


def group_segments(
    embs: np.ndarray, segs: Sequence[SegmentLike], distance: float
) -> list[tuple[str, list[int]]]:
    """(cluster label, segment indices) in order of first appearance. Diarization segments
    keep their cluster; segments without one are grouped by average-linkage cosine distance
    at `distance` (`registry.cluster.seed_clusters` uses the same linkage)."""
    with_cluster = [s.cluster is not None for s in segs]
    if any(with_cluster) and not all(with_cluster):
        raise ValueError("segments mix diarization clusters and unclustered segments")
    if not segs:
        return []
    if all(with_cluster):
        labels = [str(s.cluster) for s in segs]
    elif len(segs) == 1:
        labels = ["G1"]
    else:
        x = np.asarray(embs, dtype=np.float64)
        ids = fcluster(
            linkage(x, method="average", metric="cosine"), t=distance, criterion="distance"
        )
        labels = [f"G{int(g)}" for g in ids]
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    return list(groups.items())


class CentroidMatcher:
    """Today's pipeline logic (`stages.characters` on a fresh run, `registry.matching.match`):
    each cluster's evidence is the normalized mean of its longest `max_rows` segment
    embeddings and the total length of all its segments; a cluster whose best actor scores
    at least `threshold` is `matched` (only the best-scoring cluster per actor; the others are
    `review`, no character), else `new` with at least `min_new_seconds`, else `extra`. New and
    matched clusters add their rows to the actor, whose centroid is then recomputed. Subtitle
    segments are first grouped per episode by agglomerative cosine clustering at distance
    1 - threshold."""

    name: ClassVar[str] = "centroid"

    def __init__(
        self,
        threshold: float = 0.6,
        min_new_seconds: float = 20.0,
        max_rows: int = MAX_ROWS_PER_CLUSTER,
    ) -> None:
        self.threshold = float(threshold)
        self.min_new_seconds = float(min_new_seconds)
        self.max_rows = int(max_rows)
        # The episode's ClusterAssignments from the last fit_episode, `new` ones with their
        # created actor and character.
        self.last: list[ClusterAssignment] = []
        self.last_groups: list[tuple[str, list[int]]] = []

    def fit_episode(
        self,
        state: RegistryState,
        episode: str,  # noqa: ARG002 - part of the protocol
        embs: np.ndarray,
        segs: Sequence[SegmentLike],
    ) -> list[int | None]:
        embs = np.asarray(embs, dtype=np.float32)
        if len(embs) != len(segs):
            raise ValueError(f"{len(segs)} segments but {len(embs)} embeddings")
        groups = group_segments(embs, segs, 1.0 - self.threshold)
        rows: dict[str, np.ndarray] = {}
        evidence: list[ClusterEvidence] = []
        for label, idx in groups:
            longest = sorted(idx, key=lambda i: segs[i].duration, reverse=True)[: self.max_rows]
            rows[label] = embs[longest]
            evidence.append(
                ClusterEvidence(
                    label, mean_embedding(rows[label]), float(sum(segs[i].duration for i in idx))
                )
            )
        result = match(
            evidence,
            state.actor_list(),
            self.threshold,
            self.min_new_seconds,
            characters=state.character_list(),
        )
        touched: list[ClusterAssignment] = []
        char: dict[str, int | None] = {}
        for a in result:
            if a.status == "new":
                aid, cid = state.create(rows[a.cluster])
                a = a.model_copy(update={"actor_id": aid, "character_id": cid})
            elif a.status == "matched" and a.actor_id is not None:
                state.add(a.actor_id, rows[a.cluster])
            touched.append(a)
            char[a.cluster] = a.character_id
        self.last = touched
        self.last_groups = groups
        out: list[int | None] = [None] * len(segs)
        for label, idx in groups:
            for i in idx:
                out[i] = char[label]
        return out


MATCHERS: dict[str, type[Any]] = {"centroid": CentroidMatcher}


def make_matcher(name: str, **options: Any) -> Matcher:
    cls = MATCHERS.get(name)
    if cls is None:
        raise ValueError(f"unknown matcher {name!r}; use one of {', '.join(sorted(MATCHERS))}")
    return cls(**options)
