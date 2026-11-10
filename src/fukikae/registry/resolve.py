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
  locked. A curated orphan (locked, named, or with a curated character) is kept and
  anchored by its centroid. Anchors never share a group, whatever `episode_cannot_link` says:
  merging two existing actors is `registry merge`'s decision, not this command's.
- Subtitle names (D27): each episode reads its identity inputs (`identity_inputs`: the names
  stage's sub-clusters when names were used), and a label its current ``assignments.json``
  decided by name or lock (``source`` ``name`` or ``lock``) is not grouped. It keeps its
  character, ``name``, ``source`` and status (a named extra stays an extra), its rows are
  added to that character's actor, and they join that actor's anchor (a locked actor's
  anchor stays its stored centroid alone); a character names created that has no other
  exemplars is anchored by those rows alone. Anchors never share a group, so two keys never
  merge, and only unnamed clusters (remainders, episodes without names, keys several
  characters answered to) are grouped against them. A name decision whose character no
  longer exists (merged or deleted by hand) follows its key (`Registry.character_for_key`)
  and keeps its ``name``; one whose key answers nobody, or a lock decision whose character is
  gone, is grouped like an unnamed cluster. A remainder left with no embedding rows of its
  own goes by its parent's named sub-clusters, as the characters stage does
  (`fukikae.registry.siblings`): the character holding at least 60% of the parent's named
  seconds after this resolve (``matched``, source ``voice``), else ``extra``; each such
  decision is a line in the summary's ``notes``. Only when the parent has no named
  sub-cluster does the remainder take its parent diarize cluster's match: the anchor nearest
  the mean of all the parent's sub-cluster rows, when at least the threshold, else ``extra``.
- A names-used episode's ``assignments.json`` must be the characters stage's answer to its
  current refined clusters (the same turns, every label covered); otherwise (names reran
  after characters, or characters never ran) resolve refuses: run the characters stage
  first.
- A final group holding an anchor maps to that actor (status ``matched``, the actor's
  lowest-id character); a group without one becomes a new actor with a new character when
  its clusters total at least `min_new_seconds` of clean speech (``new``), else its clusters
  are ``extra``. With `episode_cannot_link` on (the default) a group holds at most one
  cluster per episode; with it off, one character may take several clusters of an episode.

One registry transaction removes the listed episodes' exemplars, creates the new actors, adds
the rows of every matched and new cluster, recomputes centroids (a locked actor keeps its own)
and prunes the unclaimed orphans: an interrupted resolve leaves the registry as it was. Ids
come from the registry's high-water mark and are never reused. Each episode's ``characters``
manifest record is deleted inside the transaction, once every check that can refuse has
passed and before any registry write lands, and written back, done and current, after its
``assignments.json`` (with the turns, for pinning) is written: a refused resolve changes
nothing, and an interruption after the commit leaves at most a stale characters stage, which
the next `fukikae run` or another resolve redoes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from fukikae.artifacts import read_json, write_json
from fukikae.job import StageRecord, cache_key, load_manifest, mark_stale, save_manifest
from fukikae.models import Assignments, ClusterAssignment, Diarization
from fukikae.names.inputs import identity_inputs
from fukikae.registry.matchers import JointMatcher
from fukikae.registry.matching import load_clusters, mean_embedding
from fukikae.registry.siblings import has_named_siblings, sibling_verdict
from fukikae.registry.store import Registry
from fukikae.registry.usage import actors_in_use
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
    notes: list[str] = field(default_factory=list)  # one line per rowless remainder decided by
    # its named siblings (Task 11 ruling)


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
    if "threshold" in opts:
        raise ValueError("joint option threshold: pass it as --threshold or [resolve] threshold")
    if "min_new_seconds" in opts:
        raise ValueError(
            "joint option min_new_seconds: resolve uses the workspace's [registry] "
            "min_new_seconds; set it there"
        )
    if opts.get("seed", "clusters") != "clusters":
        raise ValueError('joint option seed: resolve assigns whole clusters; keep seed "clusters"')
    try:
        joint = JointMatcher(threshold=threshold, min_new_seconds=float("-inf"), **opts)
    except TypeError as e:
        raise ValueError(f"joint: {e}") from e

    stage = CharactersStage()
    order = [ep.id for ep in episodes]
    inputs = {ep.id: identity_inputs(ep.work) for ep in episodes}
    turns = {e: read_json(i[0], Diarization) for e, i in inputs.items()}
    loaded = {e: load_clusters(i[1]) for e, i in inputs.items()}
    # each names-used label's parent diarize cluster, for a rowless remainder's match
    parents: dict[str, dict[str, str]] = {
        e: {lb: info.parent for lb, info in i[2].labels.items()}
        for e, i in inputs.items()
        if i[2] is not None and i[2].used
    }
    # Name decisions (D27): the labels the characters stage gave a character by name or lock
    # keep it; they are not grouped, and their rows seed their character's anchor.
    decided = {
        ep.id: _name_decisions(
            ep, turns[ep.id], {ev.cluster for ev in loaded[ep.id][0]}, ep.id in parents
        )
        for ep in episodes
    }
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
                "the embedder changed between them. If the registry holds exemplars of the "
                "older episodes, run `fukikae registry reembed <workspace>` first; then re-run "
                "the older episodes' diarize stage (`fukikae run`) and resolve"
            )

    result: dict[str, dict[str, ClusterAssignment]] = {e: {} for e in order}
    new_actors: list[int] = []
    notes: list[str] = []
    with Registry(ws.registry_path) as reg, reg.transaction():
        orphans: set[int] = set()
        for e in order:
            orphans |= set(reg.delete_episode(e))
        protected = reg.protected_actors()
        remaining = reg.exemplar_rows()
        known = {a.id for a in reg.actors()}
        chars = {c.id: c for c in reg.characters()}
        # a name decision whose character is gone (merged or deleted by hand) follows its key;
        # one whose key answers nobody (and a gone lock decision) is grouped afresh
        by_key: dict[str, Any] = {}

        def follow(key: str) -> Any:
            if key not in by_key:
                try:
                    by_key[key] = reg.character_for_key(key)
                except ValueError:
                    by_key[key] = None
            return by_key[key]

        for e in order:
            current: dict[str, ClusterAssignment] = {}
            for lb, a in decided[e].items():
                if a.character_id is None:
                    current[lb] = a
                    continue
                c = chars.get(a.character_id)
                if (c is None or c.actor_id not in known) and a.source == "name" and a.name:
                    c = follow(a.name)
                if c is None or c.actor_id not in known:
                    continue
                current[lb] = a.model_copy(update={"actor_id": c.actor_id, "character_id": c.id})
            decided[e] = current
        named_rows: dict[int, list[np.ndarray]] = {}
        for e in order:
            rows_e = loaded[e][1]
            for lb, a in decided[e].items():
                if a.actor_id is not None and len(rows_e[lb]):
                    named_rows.setdefault(a.actor_id, []).append(rows_e[lb])
        anchors: list[tuple[int, np.ndarray]] = []
        # the actors as stored before this run's named rows: a rowless remainder's parent
        # match scores against these alone, as the characters stage's snapshot does, so a
        # character this run's names created (or re-seeded) is no anchor for it
        stored_mean: dict[int, np.ndarray | None] = {}
        for a in reg.actors():
            named = named_rows.get(a.id, [])
            if a.locked and a.centroid is not None:
                stored_mean[a.id] = a.centroid.astype(np.float32)
            elif a.id in remaining:
                stored_mean[a.id] = mean_embedding(remaining[a.id])
            elif a.centroid is not None and (a.id not in orphans or a.id in protected):
                stored_mean[a.id] = a.centroid.astype(np.float32)
            if a.id in orphans and a.id not in protected and not named:
                continue
            if a.locked and a.centroid is not None:  # its own centroid alone, as it matches
                seed = [a.centroid[None, :].astype(np.float32)]
            elif a.id in remaining:
                seed = [remaining[a.id], *named]
            elif a.centroid is not None and not named:
                seed = [a.centroid[None, :].astype(np.float32)]
            else:  # a character names created: its named rows are its only seed
                seed = list(named)
            if seed:
                anchors.append((a.id, np.vstack(seed).astype(np.float32)))

        embs: dict[str, np.ndarray] = {}
        segs: dict[str, list[_Row]] = {}
        bad = [(aid, rows.shape[1]) for aid, rows in anchors if dim and rows.shape[1] != dim]
        if bad:
            aid, d = bad[0]
            raise ValueError(
                f"actor {aid} holds {d}-dimensional embeddings but the listed episodes' "
                f"clusters are {dim}-dimensional: the embedder changed since the registry was "
                "built. Run `fukikae registry reembed <workspace>` to re-embed the registry "
                "with the configured embedder, then resolve"
            )
        # Every refusal is above: a refused resolve leaves the characters records alone.
        for ep in episodes:  # stale until its assignments are rewritten
            mark_stale(ep, "characters")
        if anchors:
            embs[ANCHORS] = np.vstack([rows for _, rows in anchors])
            segs[ANCHORS] = [
                _Row(float(k), float(k), f"actor{aid}")
                for k, (aid, rows) in enumerate((aid, r) for aid, rs in anchors for r in rs)
            ]
        for e in order:
            evs, rows = loaded[e]
            parts = [
                (ev.cluster, rows[ev.cluster])
                for ev in evs
                if len(rows[ev.cluster]) and ev.cluster not in decided[e]
            ]
            spans = _spans(identity_inputs(next(ep for ep in episodes if ep.id == e).work)[1])
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
        claimed: set[int] = set()
        for e in order:
            evs, rows = loaded[e]
            names_e = inputs[e][2]
            infos = names_e.labels if names_e is not None and e in parents else {}
            followers: list[str] = []
            for ev in evs:
                kept = decided[e].get(ev.cluster)
                if kept is not None:  # a name or lock decision, as the characters stage made it
                    aid = kept.actor_id
                    if aid is not None and len(rows[ev.cluster]):
                        reg.add_exemplars(aid, e, ev.cluster, rows[ev.cluster], ev.clean_seconds)
                        touched.add(aid)
                    result[e][ev.cluster] = kept.model_copy(
                        update={
                            "clean_seconds": ev.clean_seconds,
                            "score": None
                            if aid is None
                            else _cosine(anchor_mean.get(aid), ev.embedding),
                        }
                    )
                    continue
                gid = group_of.get((e, ev.cluster))
                status, aid, cid, score = "extra", None, None, None
                rowless = gid is None and not len(rows[ev.cluster])
                if (
                    rowless
                    and ev.cluster in infos
                    and infos[ev.cluster].cls == "remainder"
                    and has_named_siblings(ev.cluster, infos)
                ):
                    followers.append(ev.cluster)  # once every sibling is decided
                    continue
                if rowless and ev.cluster in parents.get(e, {}):
                    top = _parent_match(ev.cluster, parents[e], rows, stored_mean)
                    if top is not None and top[1] >= threshold:
                        aid, score = top
                        status, cid = "matched", first_char.get(aid)
                        claimed.add(aid)
                elif gid is not None and gid in anchor_of_group:
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
            if followers:  # Task 11 ruling: a rowless remainder follows its named siblings
                secs: dict[str, float] = {}
                for t in turns[e].turns:
                    secs[t.speaker] = secs.get(t.speaker, 0.0) + t.end - t.start
                live = {c.id: c for c in reg.characters()}
                by_label = {ev.cluster: ev for ev in evs}
                for lb in followers:
                    verdict = sibling_verdict(lb, infos, result[e], secs)
                    assert verdict is not None  # has_named_siblings
                    c = live.get(verdict.character_id) if verdict.character_id else None
                    who = (
                        "no character" if c is None else f"character {c.id} ({c.name or 'unnamed'})"
                    )
                    notes.append(f"{e} {verdict.line(lb, infos[lb].parent, who)}")
                    takes = verdict.how == "follow" and c is not None
                    if takes and c is not None:
                        claimed.add(c.actor_id)
                    result[e][lb] = ClusterAssignment(
                        cluster=lb,
                        actor_id=c.actor_id if takes and c is not None else None,
                        character_id=c.id if takes and c is not None else None,
                        score=None,
                        clean_seconds=by_label[lb].clean_seconds,
                        status="matched" if takes else "extra",
                    )
        for aid in sorted(touched):
            reg.recompute_centroid(aid)
        in_use = actors_in_use(ws.root, skip=set(order))  # another episode still names them
        pruned = reg.prune_actors(sorted(orphans - touched - claimed - in_use))

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
        order, statuses.count("matched"), new_actors, statuses.count("extra"), pruned, notes
    )


def _name_decisions(
    ep: Episode, turns: Diarization, labels: set[str], names_used: bool
) -> dict[str, ClusterAssignment]:
    """The episode's current assignments that names decided (``source`` name or lock), for
    labels still in its identity clusters: named sub-clusters keep their character (or stay
    named extras), whatever resolve's grouping would say. ValueError for a names-used episode
    whose ``assignments.json`` is missing or answers other clusters than the identity pair
    (other turns, or labels it doesn't cover): its decisions would be stale."""
    from fukikae.stages.characters import CharactersStage

    (path,) = CharactersStage().outputs(ep)
    a = read_json(path, Assignments) if path.exists() else None
    if names_used and (
        a is None or a.turns != turns.turns or not labels <= {c.cluster for c in a.clusters}
    ):
        why = "has no assignments.json" if a is None else "has assignments older than its names"
        raise ValueError(
            f"{ep.id} {why}: its names stage refined the clusters after the characters stage "
            "decided them; run the characters stage first (`fukikae run`), then resolve"
        )
    if a is None:
        return {}
    return {
        c.cluster: c for c in a.clusters if c.source in ("name", "lock") and c.cluster in labels
    }


def _parent_match(
    label: str,
    parents: dict[str, str],
    rows: dict[str, np.ndarray],
    anchor_mean: dict[int, np.ndarray | None],
) -> tuple[int, float] | None:
    """The best (stored actor, cosine) for the mean of every row of ``label``'s parent diarize
    cluster (all its sub-clusters', decided or not, as the characters stage pools them), first
    best on ties; None without rows or stored actors. ``anchor_mean`` is the actors as stored
    before this run's named rows (Task 8 review), as the characters stage's snapshot is."""
    parent = parents[label]
    pooled = [rows[lb] for lb, p in parents.items() if p == parent and lb in rows and len(rows[lb])]
    mean = mean_embedding(np.concatenate(pooled)) if pooled else None
    best: tuple[int, float] | None = None
    for aid, m in anchor_mean.items():
        s = _cosine(m, mean)
        if s is not None and (best is None or s > best[1]):
            best = (aid, s)
    return best


def _spans(path: Any) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    with np.load(path) as z:
        for label in [str(x) for x in z["labels"]]:
            key = f"{label}/spans"
            if key in z.files:
                out[label] = [(float(a), float(b)) for a, b in np.asarray(z[key]).reshape(-1, 2)]
    return out
