"""Batch re-resolve (Ruling R12): the joint matcher over several episodes' diarization clusters
at once, written back to the registry and to each episode's ``assignments.json``.

The joint matcher (`registry.matchers.JointMatcher`) clusters every listed episode's cluster
rows (the ``clusters.npz`` embeddings, up to 50 per cluster) in one centroid-linkage pass,
seeded one group per diarization cluster. Against an existing registry:

- The listed episodes' exemplars are removed first. Actors left with no exemplars (every one
  came from these episodes) are orphans: they take no part, so the season is never judged
  against its own earlier result, and are deleted at the end unless a cluster claimed them.
- Every other actor is an *anchor*: one more group, seeded with its remaining exemplar rows
  (from episodes outside the list), or with its stored centroid when it has none left or is
  locked. A curated orphan (locked, named, or with a named or voiced character) is kept and
  anchored by its centroid. Anchors never share a group, whatever `episode_cannot_link` says:
  merging two existing actors is `registry merge`'s decision, not this command's.
- A final group holding an anchor maps to that actor (status ``matched``, the actor's
  lowest-id character); a group without one becomes a new actor with a new character when
  its clusters total at least `min_new_seconds` of clean speech (``new``), else its clusters
  are ``extra``. With `episode_cannot_link` on (the default) a group holds at most one
  cluster per episode; with it off, one character may take several clusters of an episode.

One registry transaction removes the listed episodes' exemplars, creates the new actors, adds
the rows of every matched and new cluster, recomputes centroids (a locked actor keeps its own)
and prunes the unclaimed orphans: an interrupted resolve leaves the registry as it was. Ids
come from the registry's high-water mark and are never reused. Each episode's ``characters``
manifest record is deleted before the transaction and written back, done and current, after
its ``assignments.json`` (with the turns, for pinning) is written, so an interruption after
the commit leaves at most a stale characters stage, which the next `fukikae run` or another
resolve redoes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from fukikae.artifacts import read_json, write_json
from fukikae.job import StageRecord, cache_key, load_manifest, mark_stale, save_manifest
from fukikae.models import Assignments, ClusterAssignment, Diarization
from fukikae.registry.matchers import JointMatcher
from fukikae.registry.matching import load_clusters, mean_embedding
from fukikae.registry.store import Registry
from fukikae.workspace import Episode, Workspace

ANCHORS = "@registry"  # the pseudo-episode holding existing actors' rows


@dataclass(frozen=True)
class _Row:
    """A pseudo-segment for the joint matcher: one embedding row of a cluster."""

    start: float
    end: float
    cluster: str | None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ResolveSummary:
    episodes: list[str]
    matched: int  # clusters given an existing actor
    new_actors: list[int]
    extras: int
    pruned: list[int]


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / d if d else None


def resolve_episodes(
    ws: Workspace,
    episodes: list[Episode],
    threshold: float,
    min_new_seconds: float | None = None,
    options: dict[str, Any] | None = None,
) -> ResolveSummary:
    """Joint-resolve `episodes` (each with a done diarize stage; the caller checks) and write
    the registry and their assignments. `options` are `JointMatcher` options; ``seed`` must
    stay ``clusters`` (assignments are per cluster)."""
    from fukikae.stages.characters import CharactersStage

    cfg = ws.config
    min_new = cfg.registry.min_new_seconds if min_new_seconds is None else float(min_new_seconds)
    opts = dict(options or {})
    for reserved in ("threshold", "min_new_seconds"):
        if reserved in opts:
            raise ValueError(f"joint option {reserved}: pass it as --{reserved.replace('_', '-')}")
    if opts.get("seed", "clusters") != "clusters":
        raise ValueError('joint option seed: resolve assigns whole clusters; keep seed "clusters"')
    try:
        joint = JointMatcher(threshold=threshold, min_new_seconds=float("-inf"), **opts)
    except TypeError as e:
        raise ValueError(f"joint: {e}") from e

    stage = CharactersStage()
    order = [ep.id for ep in episodes]
    turns = {ep.id: read_json(stage.inputs(ep)[0], Diarization) for ep in episodes}
    loaded = {ep.id: load_clusters(stage.inputs(ep)[1]) for ep in episodes}
    seconds = {(e, ev.cluster): ev.clean_seconds for e, (evs, _) in loaded.items() for ev in evs}

    dims = {
        e: next((r.shape[1] for r in rows.values() if len(r)), 0) for e, (_, rows) in loaded.items()
    }
    dim = next((d for d in dims.values() if d), 0)
    ref = next((e for e in order if dims[e]), None)
    for e in order:
        if dims[e] and dims[e] != dim:
            raise ValueError(
                f"{e}'s clusters are {dims[e]}-dimensional but {ref}'s are {dim}-dimensional: "
                "the embedder changed between them. Re-run the diarize stage of the older "
                "episodes with one embedder, then resolve"
            )

    for ep in episodes:  # stale until its assignments are rewritten
        mark_stale(ep, "characters")

    result: dict[str, dict[str, ClusterAssignment]] = {e: {} for e in order}
    new_actors: list[int] = []
    with Registry(ws.registry_path) as reg, reg.transaction():
        orphans: set[int] = set()
        for e in order:
            orphans |= set(reg.delete_episode(e))
        protected = reg.protected_actors()
        remaining = reg.exemplar_rows()
        anchors: list[tuple[int, np.ndarray]] = []
        for a in reg.actors():
            if a.id in orphans and a.id not in protected:
                continue
            if a.locked or a.id not in remaining:
                if a.centroid is None:
                    continue
                anchors.append((a.id, a.centroid[None, :].astype(np.float32)))
            else:
                anchors.append((a.id, remaining[a.id]))

        embs: dict[str, np.ndarray] = {}
        segs: dict[str, list[_Row]] = {}
        bad = [(aid, rows.shape[1]) for aid, rows in anchors if dim and rows.shape[1] != dim]
        if bad:
            aid, d = bad[0]
            raise ValueError(
                f"actor {aid} holds {d}-dimensional embeddings but the listed episodes' "
                f"clusters are {dim}-dimensional: the embedder changed since the registry was "
                "built. Re-embed the registry's episodes with one embedder (re-run their "
                "diarize stage, then `registry rematch` or resolve them together)"
            )
        if anchors:
            embs[ANCHORS] = np.vstack([rows for _, rows in anchors])
            segs[ANCHORS] = [
                _Row(float(k), float(k), f"actor{aid}")
                for k, (aid, rows) in enumerate((aid, r) for aid, rs in anchors for r in rs)
            ]
        for e in order:
            evs, rows = loaded[e]
            parts = [(ev.cluster, rows[ev.cluster]) for ev in evs if len(rows[ev.cluster])]
            spans = _spans(stage.inputs(next(ep for ep in episodes if ep.id == e))[1])
            embs[e] = np.vstack([r for _, r in parts]) if parts else np.zeros((0, dim), np.float32)
            segs[e] = [
                _Row(*spans[c][k], c) if c in spans and k < len(spans[c]) else _Row(0.0, 0.0, c)
                for c, r in parts
                for k in range(len(r))
            ]
        full = ([ANCHORS] if anchors else []) + order
        ids = joint.fit_all(full, embs, segs, always_cannot_link={ANCHORS})

        group_of: dict[tuple[str, str], int] = {}
        for e in full:
            for row, gid in zip(segs[e], ids[e], strict=True):
                assert gid is not None and row.cluster is not None
                group_of[(e, row.cluster)] = gid
        anchor_of_group = {group_of[(ANCHORS, f"actor{aid}")]: aid for aid, _ in anchors}
        anchor_mean = {aid: mean_embedding(rows) for aid, rows in anchors}
        first_char: dict[int, int] = {}
        for c in sorted(reg.characters(), key=lambda c: c.id):
            first_char.setdefault(c.actor_id, c.id)

        group_secs: dict[int, float] = {}
        for (e, c), gid in group_of.items():
            if e != ANCHORS:
                group_secs[gid] = group_secs.get(gid, 0.0) + seconds[(e, c)]
        created: dict[int, tuple[int, int]] = {}
        touched: set[int] = set()
        for e in order:
            evs, rows = loaded[e]
            for ev in evs:
                gid = group_of.get((e, ev.cluster))
                status, aid, cid, score = "extra", None, None, None
                if gid is not None and gid in anchor_of_group:
                    aid = anchor_of_group[gid]
                    status, cid = "matched", first_char.get(aid)
                    score = _cosine(anchor_mean[aid], ev.embedding)
                elif gid is not None and group_secs[gid] >= min_new:
                    if gid not in created:
                        members = [
                            loaded[e2][1][c2]
                            for (e2, c2), g2 in group_of.items()
                            if g2 == gid and e2 != ANCHORS
                        ]
                        centroid = mean_embedding(np.vstack(members))
                        assert centroid is not None
                        created[gid] = reg.create_actor_with_character(centroid)
                        new_actors.append(created[gid][0])
                    status, (aid, cid) = "new", created[gid]
                if aid is not None and len(rows[ev.cluster]):
                    reg.add_exemplars(aid, e, ev.cluster, rows[ev.cluster], ev.clean_seconds)
                    touched.add(aid)
                result[e][ev.cluster] = ClusterAssignment(
                    cluster=ev.cluster,
                    actor_id=aid,
                    character_id=cid,
                    score=score,
                    clean_seconds=ev.clean_seconds,
                    status=status,  # type: ignore[arg-type]
                )
        for aid in sorted(touched):
            reg.recompute_centroid(aid)
        pruned = reg.prune_actors(sorted(orphans - touched))

    for ep in episodes:
        evs, _ = loaded[ep.id]
        (path,) = stage.outputs(ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            Assignments(
                episode=ep.id,
                threshold=threshold,
                clusters=[result[ep.id][ev.cluster] for ev in evs],
                turns=turns[ep.id].turns,
            ),
        )
        manifest = load_manifest(ep)
        now = datetime.now(UTC)
        manifest.stages[stage.name] = StageRecord(
            key=cache_key(stage, ep, cfg), status="done", started=now, finished=now
        )
        save_manifest(ep, manifest)

    statuses = [a.status for r in result.values() for a in r.values()]
    return ResolveSummary(
        order, statuses.count("matched"), new_actors, statuses.count("extra"), pruned
    )


def _spans(path: Any) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    with np.load(path) as z:
        for label in [str(x) for x in z["labels"]]:
            key = f"{label}/spans"
            if key in z.files:
                out[label] = [(float(a), float(b)) for a, b in np.asarray(z[key]).reshape(-1, 2)]
    return out
