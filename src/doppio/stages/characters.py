"""Characters stage: give each diarized cluster a registry character.

Reads ``04-diarize/turns.json`` and ``clusters.npz``; writes under ``05-characters/``:

``assignments.json``
    An ``Assignments``: one ``ClusterAssignment`` per cluster.
    Also holds ``turns``, the diarization turns this run used, so the labels and the turns
    they refer to are one artifact. The next run pins its clusters to these assignments by
    overlap in time, so a re-run never silently reassigns a character.

Each run rebuilds the episode's exemplars from scratch in one registry transaction: it
deletes them all, then adds the current rows of every ``matched``, ``new`` and ``pinned``
cluster (``review`` and ``extra`` clusters never add any). An interrupted run therefore
leaves the registry as it was, and a rerun never counts an exemplar twice.

When there is no earlier run to pin to (the first run, a deleted ``assignments.json``, or
``rematch``), actors whose exemplars all came from this episode are deleted before
matching, so the episode is judged against the rest of the series and never against its
own earlier result. An actor someone has curated (locked, named, or with a named or voiced
character) is kept. On a pinned rerun such actors stay matchable, and are deleted only if
no cluster claims them again. The registry database lives in the workspace and is shared
state outside the manifest's fingerprinting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from doppio.artifacts import read_json, write_json
from doppio.config import Config
from doppio.models import Assignments, ClusterAssignment, Diarization
from doppio.registry.matching import load_clusters, match, mean_embedding, pin
from doppio.registry.store import Registry
from doppio.workspace import Episode, Workspace


class CharactersStage:
    name = "characters"

    def __init__(self, rematch: bool = False) -> None:
        # rematch: ignore the previous assignments.json and match every cluster afresh
        self.rematch = rematch

    def inputs(self, ep: Episode) -> list[Path]:
        d = ep.work / "04-diarize"
        return [d / "turns.json", d / "clusters.npz"]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "05-characters" / "assignments.json"]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        return {
            "threshold": cfg.registry.match_threshold,
            "min_new_seconds": cfg.registry.min_new_seconds,
        }

    def run(self, ep: Episode, cfg: Config) -> None:
        turns_path, clusters_path = self.inputs(ep)
        (assignments_path,) = self.outputs(ep)
        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        turns = read_json(turns_path, Diarization)
        threshold = cfg.registry.match_threshold

        evidence, rows = load_clusters(clusters_path)

        previous: Assignments | None = None
        if not self.rematch and assignments_path.exists():
            previous = read_json(assignments_path, Assignments)
        fresh = previous is None or previous.turns is None
        pinned: dict[str, ClusterAssignment] = {}
        if previous is not None and previous.turns is not None:
            pinned = pin(previous, Diarization(backend=turns.backend, turns=previous.turns), turns)

        result: dict[str, ClusterAssignment] = {}
        with Registry(Workspace(ep.workspace_root).registry_path) as reg, reg.transaction():
            orphans = reg.delete_episode(ep.id)
            known = {a.id for a in reg.actors()}
            # A pin to an actor that no longer exists (deleted by hand) is matched afresh.
            pinned = {k: v for k, v in pinned.items() if v.actor_id in known}
            claimed = {v.actor_id for v in pinned.values() if v.actor_id is not None}
            if fresh:
                reg.prune_actors(orphans)
            touched: set[int] = set()
            for ev in evidence:
                p = pinned.get(ev.cluster)
                if p is None or p.actor_id is None:
                    continue
                if len(rows[ev.cluster]):
                    reg.add_exemplars(
                        p.actor_id, ep.id, ev.cluster, rows[ev.cluster], ev.clean_seconds
                    )
                    touched.add(p.actor_id)
                result[ev.cluster] = p.model_copy(update={"clean_seconds": ev.clean_seconds})
            matched = match(
                [ev for ev in evidence if ev.cluster not in result],
                reg.actors(),
                threshold,
                cfg.registry.min_new_seconds,
                characters=reg.characters(),
                claimed=claimed,
            )
            for a in matched:
                emb = rows[a.cluster]
                if a.status == "new":
                    centroid = mean_embedding(emb)
                    assert centroid is not None  # `new` needs an embedding
                    aid, cid = reg.create_actor_with_character(centroid)
                    a = a.model_copy(update={"actor_id": aid, "character_id": cid})
                if a.status in ("new", "matched") and a.actor_id is not None and len(emb):
                    reg.add_exemplars(a.actor_id, ep.id, a.cluster, emb, a.clean_seconds)
                    touched.add(a.actor_id)
                result[a.cluster] = a
            for aid in sorted(touched):
                reg.recompute_centroid(aid)
            reg.prune_actors([a for a in orphans if a not in claimed])

        write_json(
            assignments_path,
            Assignments(
                episode=ep.id,
                threshold=threshold,
                clusters=[result[ev.cluster] for ev in evidence],
                turns=turns.turns,
            ),
        )
