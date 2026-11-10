"""Characters stage: give each diarized cluster a registry character.

Reads ``04c-names/names.json`` and the identity inputs it selects
(``names.inputs.identity_inputs``): the names stage's refined ``turns.json`` and
``clusters.npz`` when names were used, else ``04-diarize``'s. Writes under ``05-characters/``:

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
own earlier result. An actor someone has curated (locked, named, or with a curated
character: named by the operator, with an operator alias or an external id, or with a
hand-assigned or locked card) is kept. On
a pinned rerun such actors stay matchable, and are deleted only if no cluster claims them
again. The registry database lives in the workspace and is shared
state outside the manifest's fingerprinting.

Matching is ``[characters] matcher`` (resolved like a backend: workspace over profile, with
``[characters.lang.<code>]`` overrides): ``centroid`` (default) scores each cluster's
normalized mean against each actor's centroid; ``lda`` scores the same means in the series'
stored LDA space (`fukikae registry adapt`), option ``dims`` keeping only the first
components. Without a stored projection ``lda`` falls back to ``centroid`` with a warning.
The batch ``joint`` matcher needs a season at once and is `fukikae registry resolve`.
Fit the projection only on episodes it will not later resolve for evaluation: leave the
held-out episode(s) out of `registry adapt`.

Subtitle names (D27, D29). When names were used, each named sub-cluster (``<cluster>~<slug>``)
gets its character from its name key (`resolve_names`), never from the matcher, in this
order: a sub-cluster whose pin or parent cluster's match is a locked actor (or locked
character) stays with it (``source = "lock"``, reported); a key a character answers to
(`Registry.character_for_key`: operator and subtitle aliases, then names) maps every
sub-cluster of it to that character, several clusters of the episode included; a role label,
or a key with under ``[names] min_new_seconds`` of named speech and no character, is
``extra`` with its ``name``; an unnamed, uncurated character whose named speech here is at
least ``RENAME_SHARE`` one key, where that key has at least ``RENAME_SHARE`` of its speech on
the character, is named by subtitles (`Registry.name_character`) and takes it; any other key
creates a character, named by subtitles. Locked actors, operator names and operator aliases
beat subtitle names; a sub-cluster whose acoustics point at another character is reported,
never acted on. A named sub-cluster's acoustics are its parent cluster's (the remainder's pin,
else its own pin from an acoustic assignment, else the parent's match), never its own earlier
name or lock decision. A key two characters answer to at the same rank takes the remainder
path. The unnamed remainder of each cluster goes through the pin and match path above,
judged, like the parents, against one snapshot of the actors taken before this episode's
named exemplars (plus the characters names created here); a character named sub-clusters
took never sends a remainder to ``review``. A remainder left with no embedding rows of its
own and no pin goes by its parent's named sub-clusters once they are decided
(`fukikae.registry.siblings`, the Task 11 ruling amending R1): the character holding at least
60% of the parent's named seconds (``matched``, source ``voice``, reported as following its
named siblings), else ``extra`` (reported); only a parent with no named sub-cluster gives it
the parent's match over all its sub-clusters' rows, at the threshold. On a fresh run (or
``rematch``) an actor whose character answers to one of this episode's keys (or is one an
ambiguous key could mean) is not pruned before matching.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fukikae.artifacts import read_json, write_json
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.models import Assignments, ClusterAssignment, Diarization
from fukikae.names.inputs import identity_inputs, names_path, names_report_path
from fukikae.names.models import KeyDecision, KeyHow, NamesReport, NamesResult
from fukikae.registry.matching import (
    ClusterEvidence,
    centroid_tops,
    decide,
    load_clusters,
    mean_embedding,
    pin,
)
from fukikae.registry.projection import (
    KIND_LDA,
    Projection,
    load_series_projection,
    projected_tops,
)
from fukikae.registry.siblings import has_named_siblings, sibling_verdict
from fukikae.registry.store import AliasConflictError, Registry
from fukikae.registry.usage import actors_in_use
from fukikae.stages.language import LanguageResult
from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)
STAGE_MATCHERS = ("centroid", "lda")
RENAME_SHARE = 0.6  # D29 #3, both directions
NAMES_MIN_NEW_S = 5.0  # D29 #5; `[names] min_new_seconds` overrides it

Tops = dict[str, tuple[int, float] | None]


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom else None


def _describe(c: Any) -> str:
    return f"character {c.id} ({c.name or 'unnamed'})"


def resolve_names(
    reg: Registry,
    cfg: Config,
    result: NamesResult,
    turns: Diarization,
    evidence: list[ClusterEvidence],
    rows: dict[str, np.ndarray],
    pinned: dict[str, ClusterAssignment],
    parent_tops: Tops,
    own_pins: dict[str, ClusterAssignment] | None = None,
    decisions: list[KeyDecision] | None = None,
) -> tuple[dict[str, ClusterAssignment], list[str]]:
    """Characters for every named sub-cluster of a names-used episode, in the order the module
    docstring gives, and the report lines (locks kept, disagreements with acoustics, ambiguous
    keys). Creates and names characters in ``reg`` (the caller's transaction); adds no
    exemplars. A key several characters answer to (`Registry.character_for_key` raises) is
    reported and its sub-clusters left out of the result, for the caller's acoustic path.

    A named sub-cluster's acoustic evidence is its parent cluster's: the pin of the parent's
    remainder label in ``pinned`` (the remainder path's pins), else the sub-cluster's own pin
    in ``own_pins`` (inherited only from a voice-sourced assignment, `pin(voice_only=True)`,
    so never from its own earlier name or lock decision), else ``parent_tops``, each parent
    cluster's best (actor, score) over all its rows.

    ``decisions`` (when given) gets one `KeyDecision` per key, most named speech first: how
    the key went, for `fukikae names show`."""
    threshold = cfg.registry.match_threshold
    min_new = cfg.names.min_new_seconds
    language = (result.source or {}).get("language")
    ev = {e.cluster: e for e in evidence}
    named = {lb: info for lb, info in result.labels.items() if info.key is not None and lb in ev}
    if not named:
        return {}, []
    seconds: dict[str, float] = defaultdict(float)
    for t in turns.turns:
        seconds[t.speaker] += t.end - t.start
    actors = {a.id: a for a in reg.actors()}
    chars = {c.id: c for c in reg.characters()}
    first_char: dict[int, int] = {}
    for c in sorted(chars.values(), key=lambda c: c.id):
        first_char.setdefault(c.actor_id, c.id)
    curated = reg.curated_characters()

    def acoustic(label: str) -> tuple[int, int | None, str] | None:
        """The (actor, character, how) the label's acoustics give: its parent's pin, else its
        own voice-sourced pin, else its parent cluster's match at the threshold."""
        parent = named[label].parent
        p = pinned.get(parent) if parent not in named else None
        if p is None:
            p = (own_pins or {}).get(label)
        if p is not None and p.actor_id in actors:
            return p.actor_id, p.character_id or first_char.get(p.actor_id), "pin"
        top = parent_tops.get(named[label].parent)
        if top is not None and top[1] >= threshold and top[0] in actors:
            return top[0], first_char.get(top[0]), "voice"
        return None

    def describe(cid: int | None) -> str:
        c = chars.get(cid) if cid is not None else None
        return f"character {cid} ({c.name if c and c.name else 'unnamed'})"

    heard = {lb: acoustic(lb) for lb in named}
    by_key: dict[str, list[str]] = defaultdict(list)
    for lb, info in named.items():
        assert info.key is not None
        by_key[info.key].append(lb)
    key_seconds = {k: sum(seconds[lb] for lb in lbs) for k, lbs in by_key.items()}
    order = sorted(by_key, key=lambda k: (-key_seconds[k], k))
    out: dict[str, ClusterAssignment] = {}
    report: list[str] = []
    went_by: dict[str, tuple[KeyHow, int | None]] = {}  # key -> (how, character)

    def assign(label: str, status: str, actor_id: int | None, cid: int | None, source: str):
        info = named[label]
        a = actors.get(actor_id) if actor_id is not None else None
        out[label] = ClusterAssignment.model_validate(
            {
                "cluster": label,
                "actor_id": actor_id,
                "character_id": cid,
                "score": None if a is None else _cosine(ev[label].embedding, a.centroid),
                "clean_seconds": ev[label].clean_seconds,
                "status": status,
                "name": info.key,
                "source": source,
            }
        )

    # 1. locked actors (and locked characters) keep their speech, whatever the name says
    for lb in named:
        h = heard[lb]
        if h is None:
            continue
        aid, cid, how = h
        c = chars.get(cid) if cid is not None else None
        if actors[aid].locked or (c is not None and c.locked):
            assign(lb, "pinned" if how == "pin" else "matched", aid, cid, "lock")
            release = (
                f"`fukikae registry lock <ws> {aid} --unlock`"
                if actors[aid].locked
                else f"`fukikae registry lock <ws> {cid} --character --unlock`"
            )
            report.append(
                f"{lb}: {seconds[lb]:.1f} s named {named[lb].key!r} kept by lock on "
                f"{describe(cid)}; release with {release}, then rerun from names"
            )
    # 2. keys a character answers to
    taken: set[int] = set()
    unknown: list[str] = []
    for k in order:
        free = [lb for lb in by_key[k] if lb not in out]
        try:
            c = reg.character_for_key(k)
        except ValueError as e:
            report.append(f"{k!r} ({key_seconds[k]:.1f} s) left to acoustics: {e}")
            went_by[k] = ("ambiguous", None)
            continue
        if c is None:
            unknown.append(k)
            continue
        taken.add(c.id)
        went_by[k] = ("known", c.id)
        for lb in free:
            assign(lb, "matched", c.actor_id, c.id, "name")
    # 3. role labels and short keys are extras
    rest: list[str] = []
    for k in unknown:
        role = any(named[lb].cls == "role" for lb in by_key[k])
        if role or key_seconds[k] < min_new:
            went_by[k] = ("extra", None)
            for lb in by_key[k]:
                if lb not in out:
                    assign(lb, "extra", None, None, "name")
        else:
            rest.append(k)
    # 4. 60/60 rename of an unnamed, uncurated character, else 5. create
    on_char: dict[int, float] = defaultdict(float)
    key_on: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for lb, h in heard.items():
        if h is not None and h[1] is not None:
            on_char[h[1]] += seconds[lb]
            key_on[named[lb].key or ""][h[1]] += seconds[lb]

    def renameable(cid: int) -> bool:
        c = chars.get(cid)
        return (
            c is not None
            and c.name is None
            and not c.locked
            and not actors[c.actor_id].locked
            and cid not in curated
            and cid not in taken
        )

    for k in rest:
        labels = [lb for lb in by_key[k] if lb not in out]
        if not labels:
            continue
        display: str = next((d for lb in labels if (d := named[lb].display)), k)
        best = max(key_on[k].items(), key=lambda kv: (kv[1], -kv[0]), default=None)
        if (
            best is not None
            and best[1] >= RENAME_SHARE * key_seconds[k]
            and best[1] >= RENAME_SHARE * on_char[best[0]]
            and renameable(best[0])
        ):
            cid = best[0]
            try:
                reg.name_character(cid, display, k, language)
            except (ValueError, AliasConflictError) as e:
                report.append(f"{k!r}: not naming {describe(cid)}: {e}")
            else:
                taken.add(cid)
                report.append(
                    f"{describe(cid)} named {display!r} by subtitles: {best[1]:.1f} s of its "
                    f"{on_char[cid]:.1f} s named speech here, of {key_seconds[k]:.1f} s for the key"
                )
                chars[cid] = next(c for c in reg.characters() if c.id == cid)
                went_by[k] = ("renamed", cid)
                for lb in labels:
                    assign(lb, "matched", chars[cid].actor_id, cid, "name")
                continue
        key_rows = [rows[lb] for lb in labels if len(rows[lb])]
        centroid = mean_embedding(np.concatenate(key_rows)) if key_rows else None
        aid, cid = reg.create_actor_with_character(centroid)
        reg.name_character(cid, display, k, language)
        actors[aid] = next(a for a in reg.actors() if a.id == aid)
        chars[cid] = next(c for c in reg.characters() if c.id == cid)
        taken.add(cid)
        went_by[k] = ("created", cid)
        for lb in labels:
            assign(lb, "new", aid, cid, "name")
    # disagreements, one line per key: reported, never acted on. Extras have no character to
    # disagree with.
    against: dict[str, dict[tuple[int, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    for lb, a in out.items():
        h = heard[lb]
        if a.source != "name" or a.character_id is None or h is None or h[1] is None:
            continue
        if h[1] != a.character_id:
            against[a.name or ""][(h[1], h[2])].append(lb)
    for k in order:
        if k not in against:
            continue
        cid = next(
            a.character_id
            for lb in by_key[k]
            if (a := out.get(lb)) and a.source == "name" and a.character_id
        )
        parts = sorted(
            against[k].items(), key=lambda kv: (-sum(seconds[lb] for lb in kv[1]), kv[0])
        )
        total = sum(seconds[lb] for lbs in against[k].values() for lb in lbs)
        listed = "; ".join(
            f"{describe(other)} by {how} {sum(seconds[lb] for lb in lbs):.1f} s ({', '.join(lbs)})"
            for (other, how), lbs in parts
        )
        report.append(
            f"{k!r} -> {describe(cid)}; acoustics disagree on {total:.1f} s of its "
            f"{key_seconds[k]:.1f} s: {listed}"
        )
    if decisions is not None:
        for k in order:
            locked = {
                lb: a.character_id
                for lb in by_key[k]
                if (a := out.get(lb)) is not None and a.source == "lock"
            }
            unlocked = [lb for lb in by_key[k] if lb not in locked]
            went, cid = went_by.get(k, ("lock", None))
            if not unlocked and went != "ambiguous":  # every sub-cluster stayed with a lock
                keepers = set(locked.values())
                went, cid = "lock", keepers.pop() if len(keepers) == 1 else None
            shown = next((d for lb in by_key[k] if (d := named[lb].display)), None)
            decisions.append(
                KeyDecision(key=k, how=went, character_id=cid, display=shown,
                            seconds=round(key_seconds[k], 3), labels=sorted(by_key[k]),
                            locked=locked)
            )  # fmt: skip
    return out, report


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
        turns, clusters, _ = identity_inputs(ep.work)
        return [names_path(ep.work), turns, clusters]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "05-characters" / "assignments.json"]

    @staticmethod
    def report_path(ep: Episode) -> Path:
        """``names-report.json`` beside the assignments: how each name key went (not an
        output, so it never enters a key)."""
        return names_report_path(ep.work)

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
        names = identity_inputs(ep.work)[2]
        if names is not None and names.used:  # the no-names key stays as manifests hold it
            key["names_min_new_seconds"] = cfg.names.min_new_seconds
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
        turns_path, clusters_path, names = identity_inputs(ep.work)
        if names is not None and not names.used:
            names = None
        (assignments_path,) = self.outputs(ep)
        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = self.report_path(ep)
        report_path.unlink(missing_ok=True)  # only a run that used names writes one
        decisions: list[KeyDecision] = []
        report: list[str] = []
        turns = read_json(turns_path, Diarization)
        threshold = cfg.registry.match_threshold

        evidence, rows = load_clusters(clusters_path)
        mc = matcher_config(cfg, self._language(ep, cfg))
        dim = next((r.shape[1] for r in rows.values() if r.ndim == 2 and len(r)), 0)
        labels = {} if names is None else names.labels
        keyed = {lb: info.key for lb, info in labels.items() if info.key is not None}

        previous: Assignments | None = None
        if not self.rematch and assignments_path.exists():
            previous = read_json(assignments_path, Assignments)
        fresh = previous is None or previous.turns is None
        pinned: dict[str, ClusterAssignment] = {}
        voice_pins: dict[str, ClusterAssignment] = {}
        if previous is not None and previous.turns is not None:
            prev_turns = Diarization(backend=turns.backend, turns=previous.turns)
            pinned = pin(previous, prev_turns, turns)
            if keyed:
                voice_pins = pin(previous, prev_turns, turns, voice_only=True)

        result: dict[str, ClusterAssignment] = {}
        with Registry(Workspace(ep.workspace_root).registry_path) as reg, reg.transaction():
            # an actor another episode's assignments name stays, exemplars or not
            in_use = actors_in_use(ep.workspace_root, skip={ep.id})
            orphans = [a for a in reg.delete_episode(ep.id) if a not in in_use]
            known = {a.id for a in reg.actors()}
            # A pin to an actor that no longer exists (deleted by hand) is matched afresh.
            pinned = {k: v for k, v in pinned.items() if v.actor_id in known}
            voice_pins = {k: v for k, v in voice_pins.items() if v.actor_id in known}
            # The characters this episode's keys answer to; a key several answer to is ambiguous
            # and its sub-clusters take the remainder path (pin, else acoustic match), so every
            # character it might mean is spared too.
            answering: set[int] = set()
            ambiguous: set[str] = set()
            for k in sorted(set(keyed.values())):
                try:
                    c = reg.character_for_key(k)
                except ValueError:
                    ambiguous.add(k)
                    answering |= {c.actor_id for c in reg.characters_answering(k)}
                    continue
                if c is not None:
                    answering.add(c.actor_id)
            named = {lb for lb, k in keyed.items() if k not in ambiguous}
            # Named sub-clusters take their character from their names, never from a pin; the
            # remainders' pins (and an ambiguous key's, from acoustic decisions only) claim
            # actors against acoustic matches.
            path_pins = {k: v for k, v in pinned.items() if k not in keyed}
            path_pins.update({k: v for k, v in voice_pins.items() if k in keyed and k not in named})
            claimed = {v.actor_id for v in path_pins.values() if v.actor_id is not None}
            if fresh:  # never before this run's keys had their say on the characters they name
                reg.prune_actors([a for a in orphans if a not in answering])
            touched: set[int] = set()
            for ev in evidence:
                p = path_pins.get(ev.cluster)
                if p is None or p.actor_id is None:
                    continue
                if len(rows[ev.cluster]):
                    reg.add_exemplars(
                        p.actor_id, ep.id, ev.cluster, rows[ev.cluster], ev.clean_seconds
                    )
                    touched.add(p.actor_id)
                result[ev.cluster] = p.model_copy(update={"clean_seconds": ev.clean_seconds})
            projection = self._projection(reg, cfg, mc, dim)
            # One snapshot of the actors (centroids, and rows for lda) from before this
            # episode's named exemplars and named characters, so the parents' matches and the
            # remainders' see the same registry.
            snapshot = reg.actors()
            actor_rows: dict[int, np.ndarray] = {}
            if projection is not None:
                actor_rows.update(reg.exemplar_rows())
                for a in snapshot:  # a locked actor matches by its own centroid
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

            def tops_of(evs: list[ClusterEvidence], evrows: dict[str, np.ndarray]) -> Tops:
                if projection is None:
                    return centroid_tops(evs, snapshot)
                return projected_tops(evs, evrows, actor_rows, projection)

            # each parent diarize cluster's match over all its sub-clusters' rows: a named
            # label's acoustic evidence, and a rowless remainder's (Task 7 review R1)
            parent_tops: Tops = {}
            if names is not None:
                parent_rows: dict[str, list[np.ndarray]] = defaultdict(list)
                for lb, info in names.labels.items():
                    if lb in rows and len(rows[lb]):
                        parent_rows[info.parent].append(rows[lb])
                prows = {p: np.concatenate(r) for p, r in parent_rows.items()}
                parents = [ClusterEvidence(p, mean_embedding(r), 0.0) for p, r in prows.items()]
                parent_tops = tops_of(parents, prows)
            if names is not None and keyed:  # resolve_names reports ambiguous keys
                by_name, report = resolve_names(
                    reg,
                    cfg,
                    names,
                    turns,
                    evidence,
                    rows,
                    path_pins,
                    parent_tops,
                    own_pins={k: v for k, v in voice_pins.items() if k in named},
                    decisions=decisions,
                )
                for line in report:
                    log.info("characters %s: %s", ep.id, line)
                for lb, a in by_name.items():
                    adds = a.status in ("new", "matched", "pinned") and a.actor_id is not None
                    if adds and a.actor_id is not None and len(rows[lb]):
                        reg.add_exemplars(a.actor_id, ep.id, lb, rows[lb], a.clean_seconds)
                        touched.add(a.actor_id)
                    result[lb] = a
                # characters names created here join both views with their named rows (their
                # creation centroid), so a remainder may still match one
                created: dict[int, list[np.ndarray]] = defaultdict(list)
                for lb, a in by_name.items():
                    if a.status == "new" and a.actor_id is not None:
                        created[a.actor_id].append(rows[lb])
                snapshot += [a for a in reg.actors() if a.id in created]
                for aid, rs in created.items():
                    r = np.concatenate(rs)
                    if projection is not None and len(r):
                        actor_rows[aid] = r
            # a rowless, unpinned remainder whose parent has named sub-clusters follows them
            # once every sibling is decided (Task 11 ruling): ambiguous keys go through decide
            followers = [
                ev.cluster
                for ev in evidence
                if ev.cluster not in result
                and ev.embedding is None
                and ev.cluster in labels
                and labels[ev.cluster].cls == "remainder"
                and has_named_siblings(ev.cluster, labels)
            ]
            rest = [
                ev for ev in evidence if ev.cluster not in result and ev.cluster not in followers
            ]
            tops = tops_of(rest, rows)
            for ev in rest:  # a remainder with no rows of its own goes by its parent's match
                info = labels.get(ev.cluster)
                if ev.embedding is None and info is not None and info.cls == "remainder":
                    top = parent_tops.get(info.parent)
                    if top is not None and top[1] >= threshold:
                        tops[ev.cluster] = top
            matched = decide(
                rest,
                tops,
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
            if followers:
                seconds: dict[str, float] = defaultdict(float)
                for t in turns.turns:
                    seconds[t.speaker] += t.end - t.start
                chars = {c.id: c for c in reg.characters()}
                ev_of = {ev.cluster: ev for ev in evidence}
                for lb in followers:
                    verdict = sibling_verdict(lb, labels, result, seconds)
                    assert verdict is not None  # has_named_siblings
                    c = chars.get(verdict.character_id) if verdict.character_id else None
                    who = "no character" if c is None else _describe(c)
                    line = verdict.line(lb, labels[lb].parent, who)
                    log.info("characters %s: %s", ep.id, line)
                    report.append(line)
                    follow = verdict.how == "follow" and c is not None
                    result[lb] = ClusterAssignment(
                        cluster=lb,
                        actor_id=c.actor_id if follow and c is not None else None,
                        character_id=c.id if follow and c is not None else None,
                        score=None,
                        clean_seconds=ev_of[lb].clean_seconds,
                        status="matched" if follow else "extra",
                    )
            for aid in sorted(touched):
                reg.recompute_centroid(aid)
            kept = claimed | {a.actor_id for a in result.values() if a.actor_id is not None}
            reg.prune_actors([a for a in orphans if a not in kept])

        write_json(
            assignments_path,
            Assignments(
                episode=ep.id,
                threshold=threshold,
                clusters=[result[ev.cluster] for ev in evidence],
                turns=turns.turns,
            ),
        )
        if names is not None:
            write_json(report_path, NamesReport(episode=ep.id, keys=decisions, lines=report))
