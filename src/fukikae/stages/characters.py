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

Matching is ``[characters] matcher`` (resolved like a backend: workspace over profile, with
``[characters.lang.<code>]`` overrides): ``centroid`` (default) scores each cluster's
normalized mean against each actor's centroid; ``lda`` scores the same means in the series'
stored LDA space (`fukikae registry adapt`), option ``dims`` keeping only the first
components. Without a stored projection ``lda`` falls back to ``centroid`` with a warning.
The batch ``joint`` matcher needs a season at once and is `fukikae registry resolve`.
Fit the projection only on episodes it will not later resolve for evaluation: leave the
held-out episode(s) out of `registry adapt`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fukikae.artifacts import read_json, write_json
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.models import Assignments, ClusterAssignment, Diarization
from fukikae.registry.matching import decide, load_clusters, match, mean_embedding, pin
from fukikae.registry.projection import (
    KIND_LDA,
    Projection,
    load_series_projection,
    projected_tops,
)
from fukikae.registry.store import Registry
from fukikae.stages.language import LanguageResult
from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)
STAGE_MATCHERS = ("centroid", "lda")


def matcher_config(cfg: Config, language: str | None) -> BackendConfig:
    """The ``[characters]`` matcher and its options; ``centroid`` when none is set."""
    try:
        mc = cfg.resolve("characters", language, key="matcher")
    except ConfigError:
        return BackendConfig("centroid", {})
    if mc.backend not in STAGE_MATCHERS:
        if mc.backend == "joint":
            raise ConfigError(
                "[characters] matcher joint resolves a whole season at once; "
                "run `fukikae registry resolve` instead"
            )
        raise ConfigError(
            f"[characters] matcher {mc.backend!r}: the characters stage takes "
            f"{' or '.join(STAGE_MATCHERS)}"
        )
    allowed = {"centroid": set(), "lda": {"dims"}}[mc.backend]
    unknown = set(mc.options) - allowed
    if unknown:
        raise ConfigError(
            f"[characters] matcher {mc.backend}: unknown option(s) {', '.join(sorted(unknown))}"
        )
    dims = mc.options.get("dims")
    if dims is not None and (isinstance(dims, bool) or not isinstance(dims, int) or dims < 1):
        raise ConfigError(f"[characters] dims must be a whole number >= 1, not {dims!r}")
    return mc


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

    @staticmethod
    def _language(ep: Episode, cfg: Config) -> str | None:
        path = ep.work / "02-language" / "language.json"
        if path.exists():
            return read_json(path, LanguageResult).language
        return cfg.source_language

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        key: dict[str, Any] = {
            "threshold": cfg.registry.match_threshold,
            "min_new_seconds": cfg.registry.min_new_seconds,
        }
        mc = matcher_config(cfg, self._language(ep, cfg))
        if mc.backend != "centroid":  # centroid keeps the key manifests already hold
            created = None
            with Registry(Workspace(ep.workspace_root).registry_path) as reg:
                stored, _ = load_series_projection(reg, cfg, KIND_LDA, warn=False)
                created = None if stored is None else stored.created
            key["matcher"] = {"name": mc.backend, "options": mc.options, "projection": created}
        return key

    @staticmethod
    def _projection(reg: Registry, cfg: Config, mc: BackendConfig, dim: int) -> Projection | None:
        """The series' LDA projection for ``matcher = "lda"``, else None (centroid)."""
        if mc.backend != "lda":
            return None
        stored, tried = load_series_projection(reg, cfg, KIND_LDA)
        if stored is None:
            log.warning(
                "[characters] matcher lda: no lda projection stored for series %s; matching "
                "by centroid. Fit one with `fukikae registry adapt`.",
                " or ".join(repr(s) for s in tried),
            )
            return None
        series = stored.series
        p = Projection(stored.matrix, stored.mean)
        if dim and p.dim != dim:
            raise ConfigError(
                f"the lda projection of series {series!r} takes {p.dim}-dimensional "
                f"embeddings, but the clusters have {dim}; the embedder changed since it was "
                "fit: run `fukikae registry adapt` again"
            )
        return p.truncated(mc.options.get("dims"))

    def run(self, ep: Episode, cfg: Config) -> None:
        turns_path, clusters_path = self.inputs(ep)
        (assignments_path,) = self.outputs(ep)
        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        turns = read_json(turns_path, Diarization)
        threshold = cfg.registry.match_threshold

        evidence, rows = load_clusters(clusters_path)
        mc = matcher_config(cfg, self._language(ep, cfg))
        dim = next((r.shape[1] for r in rows.values() if r.ndim == 2 and len(r)), 0)

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
            rest = [ev for ev in evidence if ev.cluster not in result]
            projection = self._projection(reg, cfg, mc, dim)
            if projection is None:
                matched = match(
                    rest,
                    reg.actors(),
                    threshold,
                    cfg.registry.min_new_seconds,
                    characters=reg.characters(),
                    claimed=claimed,
                )
            else:
                actor_rows = reg.exemplar_rows()
                for a in reg.actors():  # a locked actor matches by its own centroid
                    if a.centroid is not None and (a.locked or a.id not in actor_rows):
                        actor_rows[a.id] = a.centroid[None, :]
                for aid, r in actor_rows.items():
                    if r.shape[1] != projection.dim:
                        raise ConfigError(
                            f"actor {aid} holds {r.shape[1]}-dimensional embeddings but the "
                            f"lda projection takes {projection.dim}: the embedder changed. "
                            "Run `fukikae registry reembed <workspace>`, then "
                            "`fukikae registry adapt` again"
                        )
                matched = decide(
                    rest,
                    projected_tops(rest, rows, actor_rows, projection),
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
