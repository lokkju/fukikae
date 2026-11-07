"""Matchers: resolve one episode's segments to registry characters, against an in-memory
registry state built from the episodes resolved before it.

`Matcher.fit_episode(state, episode, embs, segs)` returns one character id per segment
(None for a segment no character takes: an extra, or a cluster sent to review) and updates
`state` the way the characters stage updates the registry. The identity harness
(`fukikae eval identity`) replays whole series through these classes offline. The pipeline
does not select from this module: the characters stage has its own ``centroid`` and ``lda``
matching (`registry.matching`, `registry.projection`, the projection stored by `registry
adapt`), and `fukikae registry resolve` runs `JointMatcher` (`registry.resolve`). Segments
are any objects with ``start``, ``end``, ``duration`` and ``cluster`` attributes
(`fukikae.eval.identity.segments.Segment`); this module imports nothing from eval.

Matchers by name (`make_matcher`), each taking `threshold` and `min_new_seconds`:

- `centroid`: today's pipeline (one normalized mean per actor).
- `knn` (`k`, default 5): each cluster row votes for the actors of its `k` nearest exemplar
  rows, weighted by similarity; the winner's score is the mean similarity of its votes.
- `multi-centroid` (`max_centroids`, default 3; `min_rows`, default 5): up to that many
  spherical k-means centroids per actor, one per `min_rows` exemplar rows; a cluster scores
  its best centroid.
- `lda` (`dims`, default all): centroid matching in a linear discriminant space fit on the
  labeled segments of the episodes resolved before this one (never this one's).
- `joint` (`seed`, default clusters): one constrained agglomerative clustering of every
  episode at once (`AllEpisodesMatcher.fit_all`); no registry state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np

from fukikae.models import ClusterAssignment
from fukikae.registry.matching import ClusterEvidence, decide, mean_embedding
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


class AllEpisodesMatcher(Protocol):
    """A matcher that resolves every episode in one pass, with no registry state."""

    name: ClassVar[str]

    def fit_all(
        self,
        order: list[str],
        embs: Mapping[str, np.ndarray],
        segs: Mapping[str, Sequence[SegmentLike]],
    ) -> dict[str, list[int | None]]: ...


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def group_segments(
    embs: np.ndarray, segs: Sequence[SegmentLike], distance: float
) -> list[tuple[str, list[int]]]:
    """(cluster label, segment indices) in order of first appearance. Diarization segments
    keep their cluster; segments without one are grouped by centroid linkage (Ruling R11):
    the most similar pair of groups merges while the cosine of their mean rows is at least
    1 - `distance`, never joining segments that overlap in time (`JointMatcher` with
    `seed="segments"` on the one episode). Average linkage, the mean pairwise cosine of
    single noisy rows, ran far below that scale and left almost every segment alone."""
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
        joint = JointMatcher(
            threshold=1.0 - distance,
            min_new_seconds=float("-inf"),
            seed="segments",
            episode_cannot_link=False,
        )
        ids = joint.fit_all(["_"], {"_": np.asarray(embs)}, {"_": segs})["_"]
        labels = [f"G{g}" for g in ids]
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    return list(groups.items())


class _ScoringMatcher:
    """The one per-episode flow (grouping, the longest `max_rows` rows per cluster,
    `registry.matching.decide`, rows added to the actor) with the cluster-to-actor score
    left to `_best`, in the space `_project` maps embeddings into (the identity here).
    The default `_prepare`/`_best` are the centroid scorer, computed as
    `registry.matching.match` computes it. The state keeps raw rows; projection happens at
    matching time."""

    name: ClassVar[str] = "scoring"

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
        self._centroids: list[tuple[int, np.ndarray]] = []

    def _project(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def _prepare(self, state: RegistryState) -> None:
        """Called once per episode before scoring, with the state as it stands; here, each
        actor's centroid (`RegistryState.centroid`) in actor order."""
        self._centroids = [(a, c) for a in state.actors if (c := state.centroid(a)) is not None]

    def _best(self, rows: np.ndarray, mean: np.ndarray) -> tuple[int, float] | None:  # noqa: ARG002
        """The best (actor id, score) for a cluster's rows and normalized mean, against the
        state `_prepare` saw; here the cosine with each centroid, first best on ties."""
        top: tuple[int, float] | None = None
        for aid, c in self._centroids:
            denom = float(np.linalg.norm(c) * np.linalg.norm(mean))
            sc = float(np.dot(c, mean)) / denom if denom else 0.0
            if top is None or sc > top[1]:
                top = (aid, sc)
        return top

    def _before(self, episode: str) -> None:
        """Called first in each fit_episode."""

    def _after(self, episode: str, embs: np.ndarray, segs: Sequence[SegmentLike]) -> None:
        """Called last in each fit_episode, once its ids are decided."""

    def fit_episode(
        self,
        state: RegistryState,
        episode: str,
        embs: np.ndarray,
        segs: Sequence[SegmentLike],
    ) -> list[int | None]:
        self._before(episode)
        raw = np.asarray(embs, dtype=np.float32)
        if len(raw) != len(segs):
            raise ValueError(f"{len(segs)} segments but {len(raw)} embeddings")
        space = self._project(raw) if len(raw) else raw
        groups = group_segments(space, segs, 1.0 - self.threshold)
        self._prepare(state)
        raw_rows: dict[str, np.ndarray] = {}
        evidence: list[ClusterEvidence] = []
        tops: dict[str, tuple[int, float] | None] = {}
        for label, idx in groups:
            longest = sorted(idx, key=lambda i: segs[i].duration, reverse=True)[: self.max_rows]
            raw_rows[label] = raw[longest]
            mean = mean_embedding(space[longest])
            evidence.append(ClusterEvidence(label, mean, float(sum(segs[i].duration for i in idx))))
            tops[label] = (
                None if mean is None or not state.actors else self._best(space[longest], mean)
            )
        result = decide(
            evidence, tops, self.threshold, self.min_new_seconds, state.character_list()
        )
        touched: list[ClusterAssignment] = []
        char: dict[str, int | None] = {}
        for a in result:
            if a.status == "new":
                aid, cid = state.create(raw_rows[a.cluster])
                a = a.model_copy(update={"actor_id": aid, "character_id": cid})
            elif a.status == "matched" and a.actor_id is not None:
                state.add(a.actor_id, raw_rows[a.cluster])
            touched.append(a)
            char[a.cluster] = a.character_id
        self.last = touched
        self.last_groups = groups
        out: list[int | None] = [None] * len(segs)
        for label, idx in groups:
            for i in idx:
                out[i] = char[label]
        self._after(episode, raw, segs)
        return out


class CentroidMatcher(_ScoringMatcher):
    """Today's pipeline logic (`stages.characters` on a fresh run, `registry.matching.match`):
    each cluster's evidence is the normalized mean of its longest `max_rows` segment
    embeddings and the total length of all its segments; a cluster whose best actor scores
    at least `threshold` is `matched` (only the best-scoring cluster per actor; the others are
    `review`, no character), else `new` with at least `min_new_seconds`, else `extra`. New and
    matched clusters add their rows to the actor, whose centroid is then recomputed. Subtitle
    segments are first grouped per episode by agglomerative cosine clustering at distance
    1 - threshold."""

    name: ClassVar[str] = "centroid"


class KnnMatcher(_ScoringMatcher):
    """Each of a cluster's rows (its longest `max_rows`) takes its `k` nearest exemplar rows
    among all actors' and votes for their actors with their cosine similarity; the actor
    with the largest vote total is the cluster's best. Its score is the cosine of the
    cluster's mean with the local centroid (normalized mean) of the exemplar rows that voted
    for it: the centroid matcher's mean-vs-mean scale, so one threshold reads the same for
    both (single rows are noisy, and row-to-row similarity runs far lower). A speaker heard
    in several registers keeps the exemplars of each, so a cluster in one register finds its
    own exemplars where the actor's mean lies between registers. The local centroid is
    built from the exemplars nearest the cluster, so the score runs somewhat above the
    centroid matcher's for the same cluster (an upward bias, more so for small k): the same
    threshold is a little more permissive here. Options: `k` (default 5)."""

    name: ClassVar[str] = "knn"

    def __init__(
        self,
        threshold: float = 0.6,
        min_new_seconds: float = 20.0,
        max_rows: int = MAX_ROWS_PER_CLUSTER,
        k: int = 5,
    ) -> None:
        super().__init__(threshold, min_new_seconds, max_rows)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"knn: k must be a whole number of at least 1, not {k!r}")
        self.k = k
        self._x = np.zeros((0, 0))
        self._owner = np.zeros(0, dtype=int)

    def _prepare(self, state: RegistryState) -> None:
        mats, owners = [], []
        for aid in state.actors:
            rows = np.vstack(state.exemplars[aid])
            mats.append(rows)
            owners.append(np.full(len(rows), aid))
        self._x = _unit_rows(np.vstack(mats)) if mats else np.zeros((0, 0))
        self._owner = np.concatenate(owners) if owners else np.zeros(0, dtype=int)

    def _best(self, rows: np.ndarray, mean: np.ndarray) -> tuple[int, float] | None:  # noqa: ARG002
        if not len(self._x):
            return None
        sims = _unit_rows(rows) @ self._x.T  # (rows, exemplars)
        k = min(self.k, sims.shape[1])
        nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        votes: dict[int, float] = {}
        voted: dict[int, set[int]] = {}
        for r in range(len(sims)):
            for j in nn[r]:
                a = int(self._owner[j])
                votes[a] = votes.get(a, 0.0) + float(sims[r, j])
                voted.setdefault(a, set()).add(int(j))
        aid = max(votes, key=lambda a: (votes[a], -a))
        local = mean_embedding(self._x[sorted(voted[aid])])
        if local is None:
            return None
        m = np.asarray(mean, dtype=np.float64)
        return aid, float(local @ (m / (np.linalg.norm(m) or 1.0)))


def _spherical_kmeans(x: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    """`k` unit centroids of unit rows `x`: farthest-point initialisation from the row
    nearest the mean (deterministic), then cosine k-means."""
    mean = mean_embedding(x)
    first = int(np.argmax(x @ mean)) if mean is not None else 0
    cents = [x[first]]
    for _ in range(1, k):
        sim = np.max(x @ np.vstack(cents).T, axis=1)
        cents.append(x[int(np.argmin(sim))])
    c = np.vstack(cents)
    for _ in range(iters):
        assign = np.argmax(x @ c.T, axis=1)
        new = np.vstack(
            [
                (m if (m := mean_embedding(x[assign == j])) is not None else c[j])
                for j in range(len(c))
            ]
        )
        if np.allclose(new, c):
            break
        c = new
    return c


class MultiCentroidMatcher(_ScoringMatcher):
    """Up to `max_centroids` spherical k-means centroids per actor, one per `min_rows` of its
    exemplar rows (an actor with fewer than 2 × `min_rows` rows keeps one centroid, its
    normalized mean); a cluster's score for an actor is the cosine of its mean with the
    nearest of them. One centroid is the centroid matcher. Options: `max_centroids`
    (default 3), `min_rows` (default 5)."""

    name: ClassVar[str] = "multi-centroid"

    def __init__(
        self,
        threshold: float = 0.6,
        min_new_seconds: float = 20.0,
        max_rows: int = MAX_ROWS_PER_CLUSTER,
        max_centroids: int = 3,
        min_rows: int = 5,
    ) -> None:
        super().__init__(threshold, min_new_seconds, max_rows)
        for key, v in (("max_centroids", max_centroids), ("min_rows", min_rows)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"multi-centroid: {key} must be a whole number >= 1, not {v!r}")
        self.max_centroids = max_centroids
        self.min_rows = min_rows
        self._cents: dict[int, np.ndarray] = {}

    def _prepare(self, state: RegistryState) -> None:
        self._cents = {}
        for aid in state.actors:
            rows = np.vstack(state.exemplars[aid])
            k = max(1, min(self.max_centroids, len(rows) // self.min_rows))
            if k == 1:
                c = state.centroid(aid)
                if c is not None:
                    self._cents[aid] = np.asarray(c, dtype=np.float64)[None]
            else:
                self._cents[aid] = _spherical_kmeans(_unit_rows(rows), k)

    def _best(self, rows: np.ndarray, mean: np.ndarray) -> tuple[int, float] | None:  # noqa: ARG002
        top: tuple[int, float] | None = None
        m = np.asarray(mean, dtype=np.float64)
        m = m / (np.linalg.norm(m) or 1.0)
        for aid, c in self._cents.items():
            s = float(np.max(_unit_rows(c) @ m))
            if top is None or s > top[1]:
                top = (aid, s)
        return top


class LdaMatcher(_ScoringMatcher):
    """Centroid matching in a linear discriminant space (scikit-learn's
    LinearDiscriminantAnalysis, eigen solver with automatic shrinkage), fit on the labeled
    segments (``label`` not None) of the episodes this matcher resolved before the current
    one, and never on the current episode's: an episode's labels join the training set only
    after its ids are decided, and resolving an episode twice is an error. For E12 then E13
    the space is fit on E12's labels and E13 is resolved in it. With fewer than two labels
    of two or more segments each to fit on (the first episode), there is no space and the
    episode resolves exactly as `centroid`. Embeddings, exemplars and cluster means are
    projected, then L2-normalized; grouping of unclustered segments happens in the space.
    Options: `dims` (components, at most labels - 1; default or "all": that maximum)."""

    name: ClassVar[str] = "lda"

    def __init__(
        self,
        threshold: float = 0.6,
        min_new_seconds: float = 20.0,
        max_rows: int = MAX_ROWS_PER_CLUSTER,
        dims: int | str | None = None,
    ) -> None:
        super().__init__(threshold, min_new_seconds, max_rows)
        if dims == "all":
            dims = None
        if dims is not None and (isinstance(dims, bool) or not isinstance(dims, int) or dims < 1):
            raise ValueError(f'lda: dims must be a whole number >= 1 or "all", not {dims!r}')
        self.dims: int | None = dims
        self._seen: list[str] = []
        self._train_x: list[np.ndarray] = []
        self._train_y: list[str] = []
        self._lda: Any = None

    def _before(self, episode: str) -> None:
        if episode in self._seen:
            raise ValueError(f"lda: episode {episode} was already resolved")
        self._lda = self._fit()

    def _fit(self) -> Any:
        from fukikae.registry.projection import fit_lda

        if not self._train_x:
            return None
        return fit_lda(np.vstack(self._train_x), self._train_y, self.dims)

    def _project(self, x: np.ndarray) -> np.ndarray:
        if self._lda is None:
            return np.asarray(x, dtype=np.float32)
        return self._lda.apply(x)

    def _prepare(self, state: RegistryState) -> None:
        if self._lda is None:
            super()._prepare(state)
            return
        self._centroids = [
            (a, c)
            for a in state.actors
            if (c := mean_embedding(self._project(np.vstack(state.exemplars[a])))) is not None
        ]

    def _after(self, episode: str, embs: np.ndarray, segs: Sequence[SegmentLike]) -> None:
        self._seen.append(episode)
        for row, seg in zip(embs, segs, strict=True):
            label = getattr(seg, "label", None)
            if label is not None:
                self._train_x.append(np.asarray(row, dtype=np.float64))
                self._train_y.append(str(label))


class JointMatcher:
    """One agglomerative clustering of every episode's segments at once; no registry state.

    Seeding: `seed="clusters"` (default) starts each episode's diarization cluster as one
    group, and segments without a cluster start alone; `seed="segments"` starts every segment
    alone. The most similar pair of groups then merges while its similarity reaches
    `threshold`: with `linkage="centroid"` (default) the cosine of the two groups' mean rows,
    the scale the centroid matcher's threshold is on; with `"average"` the mean pairwise
    cosine of their rows, which noisy single rows pull far lower.

    Constraints, carried through every merge: two groups holding segments of one episode that
    overlap in time never merge. With `episode_cannot_link` (default on), segments of
    different diarization clusters of one episode never share a group either, since the
    diarizer has already called them different speakers (every segmentation drops overlapped
    speech, so the time-overlap rule alone almost never fires); this also stops it rejoining
    a speaker the diarizer split in two. Segments without a cluster (subtitle and asr
    segmentations) are untouched by it. `fit_all`'s `always_cannot_link` applies the same
    rule to named episodes whatever `episode_cannot_link` says.

    Groups with at least `min_new_seconds` of segment time become characters, numbered from 1
    in order of first appearance (episode order, then segment order); the rest get no id.
    Callers hand it every episode through `fit_all`: the identity harness, and `fukikae
    registry resolve` (`registry.resolve`), which adds the registry's existing actors as one
    pseudo-episode in `always_cannot_link` so they never merge with each other."""

    name: ClassVar[str] = "joint"

    def __init__(
        self,
        threshold: float = 0.6,
        min_new_seconds: float = 20.0,
        seed: str = "clusters",
        linkage: str = "centroid",
        episode_cannot_link: bool = True,
    ) -> None:
        if not isinstance(episode_cannot_link, bool):
            raise ValueError("joint: episode_cannot_link must be true or false")
        self.episode_cannot_link = episode_cannot_link
        if seed not in ("clusters", "segments"):
            raise ValueError(f'joint: seed must be "clusters" or "segments", not {seed!r}')
        if linkage not in ("centroid", "average"):
            raise ValueError(f'joint: linkage must be "centroid" or "average", not {linkage!r}')
        self.linkage = linkage
        self.threshold = float(threshold)
        self.min_new_seconds = float(min_new_seconds)
        self.seed = seed

    def fit_all(
        self,
        order: list[str],
        embs: Mapping[str, np.ndarray],
        segs: Mapping[str, Sequence[SegmentLike]],
        always_cannot_link: Collection[str] = (),
    ) -> dict[str, list[int | None]]:
        """`always_cannot_link` names episodes whose distinct clusters never share a group
        even with `episode_cannot_link` off (`registry resolve` keeps existing actors apart
        this way)."""
        items: list[tuple[str, int]] = []
        rows: list[np.ndarray] = []
        for ep in order:
            e = np.asarray(embs[ep])
            if len(e) != len(segs[ep]):
                raise ValueError(f"{ep}: {len(segs[ep])} segments but {len(e)} embeddings")
            for i in range(len(segs[ep])):
                items.append((ep, i))
                rows.append(e[i])
        out: dict[str, list[int | None]] = {ep: [None] * len(segs[ep]) for ep in order}
        if not items:
            return out
        x = _unit_rows(np.vstack(rows))
        # Seed groups.
        group_of: list[int] = []
        seeds: dict[tuple[str, str], int] = {}
        n_groups = 0
        for ep, i in items:
            c = segs[ep][i].cluster
            if self.seed == "clusters" and c is not None:
                key = (ep, str(c))
                if key not in seeds:
                    seeds[key] = n_groups
                    n_groups += 1
                group_of.append(seeds[key])
            else:
                group_of.append(n_groups)
                n_groups += 1
        g = np.asarray(group_of)
        sums = np.zeros((n_groups, x.shape[1]))
        np.add.at(sums, g, x)
        counts = np.bincount(g, minlength=n_groups).astype(np.float64)
        cannot = np.zeros((n_groups, n_groups), dtype=bool)
        for ep in order:
            idx = [k for k, (e, _) in enumerate(items) if e == ep]
            spans = sorted((segs[ep][items[k][1]].start, segs[ep][items[k][1]].end, k) for k in idx)
            for a in range(len(spans)):
                s0, e0, ka = spans[a]
                for b in range(a + 1, len(spans)):
                    s1, _e1, kb = spans[b]
                    if s1 >= e0:
                        break
                    ga, gb = g[ka], g[kb]
                    if ga != gb:
                        cannot[ga, gb] = cannot[gb, ga] = True
        always = set(always_cannot_link)
        if self.episode_cannot_link or always:
            keys: dict[int, set[tuple[str, str]]] = {}
            for k, (ep, i) in enumerate(items):
                c = segs[ep][i].cluster
                if c is not None and (self.episode_cannot_link or ep in always):
                    keys.setdefault(int(g[k]), set()).add((ep, str(c)))
            by_ep: dict[str, list[int]] = {}
            for gid, ks in keys.items():
                for ep in {e for e, _ in ks}:
                    by_ep.setdefault(ep, []).append(gid)
            for ep, gids in by_ep.items():
                for x in range(len(gids)):
                    kx = {c for e, c in keys[gids[x]] if e == ep}
                    for y in range(x + 1, len(gids)):
                        ky = {c for e, c in keys[gids[y]] if e == ep}
                        if kx != ky or len(kx) > 1:
                            cannot[gids[x], gids[y]] = cannot[gids[y], gids[x]] = True
        alive = np.ones(n_groups, dtype=bool)

        def scale(i: np.ndarray | int) -> np.ndarray:
            if self.linkage == "average":
                return np.asarray(counts[i], dtype=np.float64)
            return np.asarray(np.linalg.norm(sums[i], axis=-1), dtype=np.float64)

        sim = (sums @ sums.T) / np.outer(scale(np.arange(n_groups)), scale(np.arange(n_groups)))
        sim[cannot] = -np.inf
        np.fill_diagonal(sim, -np.inf)
        parent = np.arange(n_groups)
        while True:
            flat = int(np.argmax(sim))
            a, b = divmod(flat, n_groups)
            if not np.isfinite(sim[a, b]) or sim[a, b] < self.threshold:
                break
            # Merge b into a.
            sums[a] += sums[b]
            counts[a] += counts[b]
            alive[b] = False
            parent[parent == b] = a
            cannot[a] |= cannot[b]
            cannot[:, a] = cannot[a]
            with np.errstate(divide="ignore", invalid="ignore"):
                row = (sums @ sums[a]) / (scale(np.arange(n_groups)) * scale(a))
            row[~alive] = -np.inf
            row[cannot[a]] = -np.inf
            row[a] = -np.inf
            sim[a, :] = row
            sim[:, a] = row
            sim[b, :] = -np.inf
            sim[:, b] = -np.inf
        final = parent[g]
        seconds: dict[int, float] = {}
        for k, (ep, i) in enumerate(items):
            seconds[int(final[k])] = seconds.get(int(final[k]), 0.0) + segs[ep][i].duration
        ids: dict[int, int] = {}
        for k, (ep, i) in enumerate(items):
            f = int(final[k])
            if seconds[f] < self.min_new_seconds:
                continue
            if f not in ids:
                ids[f] = len(ids) + 1
            out[ep][i] = ids[f]
        return out


MATCHERS: dict[str, type[Any]] = {
    "centroid": CentroidMatcher,
    "knn": KnnMatcher,
    "multi-centroid": MultiCentroidMatcher,
    "lda": LdaMatcher,
    "joint": JointMatcher,
}


def make_matcher(name: str, **options: Any) -> Any:
    """A matcher by name: a `Matcher`, or for `joint` an `AllEpisodesMatcher`."""
    cls = MATCHERS.get(name)
    if cls is None:
        raise ValueError(f"unknown matcher {name!r}; use one of {', '.join(sorted(MATCHERS))}")
    return cls(**options)
