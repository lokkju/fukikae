"""Characters stage: give each diarized cluster a registry character.

Reads ``04-diarize/turns.json`` and ``clusters.npz``; writes under ``05-characters/``:

``assignments.json``
    An ``Assignments``: one ``ClusterAssignment`` per cluster.
``turns.prev.json``
    A copy of the turns this run used. The next run pins its clusters to this run's
    assignments by overlap in time, so a re-run never silently reassigns a character.

Only ``matched`` and ``new`` clusters add exemplars; ``pinned`` ones did so on their
earlier run, ``review`` and ``extra`` ones never do. The registry database lives in the
workspace and is shared state outside the manifest's fingerprinting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from doppio.artifacts import read_json, write_json
from doppio.config import Config
from doppio.models import Assignments, ClusterAssignment, Diarization
from doppio.registry.matching import ClusterEvidence, match, mean_embedding, pin
from doppio.registry.store import Registry
from doppio.workspace import Episode, Workspace


class CharactersStage:
    name = "characters"

    def inputs(self, ep: Episode) -> list[Path]:
        d = ep.work / "04-diarize"
        return [d / "turns.json", d / "clusters.npz"]

    def outputs(self, ep: Episode) -> list[Path]:
        d = ep.work / "05-characters"
        return [d / "assignments.json", d / "turns.prev.json"]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        return {
            "threshold": cfg.registry.match_threshold,
            "min_new_seconds": cfg.registry.min_new_seconds,
        }

    def run(self, ep: Episode, cfg: Config) -> None:
        turns_path, clusters_path = self.inputs(ep)
        assignments_path, prev_turns_path = self.outputs(ep)
        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        turns = read_json(turns_path, Diarization)
        threshold = cfg.registry.match_threshold

        rows: dict[str, np.ndarray] = {}
        evidence: list[ClusterEvidence] = []
        with np.load(clusters_path) as z:
            for label in [str(x) for x in z["labels"]]:
                emb = np.asarray(z[f"{label}/embeddings"], dtype=np.float32)
                rows[label] = emb
                evidence.append(
                    ClusterEvidence(label, mean_embedding(emb), float(z[f"{label}/clean_seconds"]))
                )

        pinned: dict[str, ClusterAssignment] = {}
        if assignments_path.exists() and prev_turns_path.exists():
            pinned = pin(
                read_json(assignments_path, Assignments),
                read_json(prev_turns_path, Diarization),
                turns,
            )

        result: dict[str, ClusterAssignment] = {
            ev.cluster: pinned[ev.cluster] for ev in evidence if ev.cluster in pinned
        }
        todo = [ev for ev in evidence if ev.cluster not in pinned]
        with Registry(Workspace(ep.workspace_root).registry_path) as reg:
            matched = match(
                todo,
                reg.actors(),
                threshold,
                cfg.registry.min_new_seconds,
                characters=reg.characters(),
            )
            touched: set[int] = set()
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

        write_json(
            assignments_path,
            Assignments(
                episode=ep.id,
                threshold=threshold,
                clusters=[result[ev.cluster] for ev in evidence],
            ),
        )
        write_json(prev_turns_path, turns)
