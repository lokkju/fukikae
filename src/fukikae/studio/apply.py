"""The studio's one writer: decisions applied to the registry, with digests and invalidation.

`apply_decision` is the only way the studio (its page, `fukikae studio apply`, a script)
changes the registry or the card store. Each decision runs in one `Registry.transaction()`:
its characters must exist, the digest of their rows must match the one the decision was made
against (`StaleDecision` otherwise, unless forced), and a registry refusal rolls everything
back (`DecisionRefused`). Then come the file changes a registry change needs (assignments
remapped after a merge or split, a pitch split file), `job.mark_stale` on every stage the
decision invalidates, and a line in ``<ws>/studio/decisions.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fukikae.registry.invalidate import (
    Invalidation,
    character_episodes,
    episodes_with_key,
    mark_stale,
    remap_assignments,
)
from fukikae.registry.store import Character, Registry, alias_key
from fukikae.studio.model import (
    IDENTITY_OPS,
    AcceptDecision,
    AliasDecision,
    Decision,
    Decisions,
    GenderDecision,
    KindDecision,
    LockDecision,
    MergeDecision,
    NameDecision,
    ReviewedDecision,
    SplitDecision,
    parse_decision,
)
from fukikae.workspace import Workspace


class DecisionError(Exception):
    """A decision that was not applied; nothing it would have changed has changed."""


class StaleDecision(DecisionError):  # noqa: N818 (the plan's name)
    """The registry rows the decision was made against have changed since (or it carries no
    digest to check). Reload the character, or force it."""


class DecisionRefused(DecisionError):  # noqa: N818
    """The registry (or the card checks) refused the decision; the message is theirs."""


@dataclass(frozen=True)
class ApplyResult:
    lines: list[str]
    invalidations: list[Invalidation] = field(default_factory=list)


class _DryRun(Exception):  # noqa: N818
    """Raised inside the transaction to roll a dry run back."""


def studio_dir(ws: Workspace) -> Path:
    return ws.root / "studio"


def decisions_log(ws: Workspace) -> Path:
    return studio_dir(ws) / "decisions.jsonl"


def split_path(ws: Workspace, episode: str) -> Path:
    """``<ws>/studio/splits/<episode>.json``, the operator pitch splits the names stage reads:
    ``{cluster: {"by": "pitch", "male_below": Hz, "female_above": Hz}}``."""
    return studio_dir(ws) / "splits" / f"{episode}.json"


def _workspace(ws: Workspace | Path) -> Workspace:
    return ws if isinstance(ws, Workspace) else Workspace(Path(ws))


# --- digests -------------------------------------------------------------------------------


def _find(reg: Registry, character_id: int) -> Character | None:
    return next((c for c in reg.characters() if c.id == character_id), None)


def _snapshot(reg: Registry, character_id: int, language: str | None) -> dict[str, Any] | None:
    """The rows a decision about this character is made against: the character (without its
    review time, which identity decisions themselves set), its actor (without the centroid,
    which every run's exemplars move), and with a language its current card assignment."""
    c = _find(reg, character_id)
    if c is None:
        return None
    a = next(x for x in reg.actors() if x.id == c.actor_id)
    snap: dict[str, Any] = {
        "character": {
            "id": c.id, "actor_id": c.actor_id, "name": c.name, "name_by": c.name_by,
            "external_id": c.external_id, "locked": c.locked, "gender": c.gender,
            "gender_by": c.gender_by, "kind": c.kind,
        },
        "actor": {"id": a.id, "name": a.name, "external_id": a.external_id, "locked": a.locked},
    }  # fmt: skip
    if language is not None:
        cur = reg.card_for(character_id, language)
        snap["card"] = None
        if cur is not None:
            snap["card"] = {
                "language": cur.target_language, "n": cur.n, "card_id": cur.card_id,
                "chosen_by": cur.chosen_by, "locked": cur.locked,
            }  # fmt: skip
    return snap


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def row_digest(reg: Registry, character_id: int, language: str | None = None) -> str:
    """sha256 of the character's registry rows (`_snapshot`): the character, its actor, and
    with ``language`` its card assignment for that language."""
    return _hash(_snapshot(reg, character_id, language))


def _targets(d: Decision) -> tuple[list[int], str | None]:
    if isinstance(d, MergeDecision):
        return [d.keep, d.drop], None
    if isinstance(d, AcceptDecision):
        return [d.character], d.language
    return [d.character], None


def decision_digest(reg: Registry, d: Decision) -> str:
    """The digest a decision carries: `row_digest` of its character, or for a merge of both
    characters (keep, then drop)."""
    ids, language = _targets(d)
    if len(ids) == 1:
        return row_digest(reg, ids[0], language)
    return _hash([_snapshot(reg, cid, language) for cid in ids])


def stamp(reg: Registry, d: Decision) -> Decision:
    """The decision with its digest, key and external id taken from the registry now: what
    the page does when it builds a decision from the rows it shows."""
    ids, _ = _targets(d)
    c = _find(reg, ids[0])
    key = alias_key(c.name) if c is not None and c.name else None
    ext = c.external_id if c is not None else None
    return d.model_copy(update={"digest": decision_digest(reg, d), "key": key, "external_id": ext})


def _describe(reg: Registry, character_id: int, language: str | None) -> str:
    snap = _snapshot(reg, character_id, language)
    if snap is None:
        return f"character {character_id} no longer exists"
    c, a = snap["character"], snap["actor"]
    parts = [
        f"name {c['name']!r}" + (f" ({c['name_by']})" if c["name"] and c["name_by"] else ""),
        f"gender {c['gender'] or '-'}" + (f" ({c['gender_by']})" if c["gender"] else ""),
        f"kind {c['kind']}",
        "locked" if c["locked"] else "unlocked",
        f"actor {a['id']}" + (" (locked)" if a["locked"] else ""),
    ]
    if c["external_id"]:
        parts.append(f"external id {c['external_id']}")
    if language is not None:
        card = snap.get("card")
        parts.append(
            f"{language} card {card['card_id']} ({card['chosen_by']}, "
            f"{'locked' if card['locked'] else 'unlocked'}, assignment {card['n']})"
            if card
            else f"no {language} card"
        )
    return f"character {character_id}: " + ", ".join(parts)


def _check_digest(reg: Registry, d: Decision) -> None:
    ids, language = _targets(d)
    who = " and ".join(f"character {i}" for i in ids)
    if d.digest is None:
        raise StaleDecision(
            f"{d.op} of {who} carries no digest of the rows it was made against; "
            "pass --force to apply it as it stands"
        )
    if d.digest != decision_digest(reg, d):
        now = "; ".join(_describe(reg, i, language) for i in ids)
        raise StaleDecision(
            f"{who} changed since this {d.op} decision was made (now {now}); reload "
            f"{'them' if len(ids) > 1 else 'it'}, or pass --force"
        )


# --- planning ------------------------------------------------------------------------------


def _require(reg: Registry, character_id: int) -> Character:
    c = _find(reg, character_id)
    if c is None:
        raise DecisionRefused(f"no character {character_id}")
    return c


def _inv(episodes: Iterable[str], stage: str) -> list[Invalidation]:
    return [Invalidation(e, stage) for e in sorted(set(episodes))]


def _cluster_holder(ws: Workspace, reg: Registry, d: SplitDecision, c: Character) -> None:
    """Refuse a split of a cluster that isn't the character's (by its exemplars, or for a
    pitch split by the episode's assignments)."""
    holders = reg.cluster_actors(d.episode, d.cluster)
    if holders == [c.actor_id]:
        return
    if d.by == "pitch" and not holders:
        from fukikae.artifacts import read_json
        from fukikae.models import Assignments

        path = ws.work_dir(d.episode) / "05-characters" / "assignments.json"
        if path.is_file():
            a = read_json(path, Assignments)
            if any(x.cluster == d.cluster and x.character_id == c.id for x in a.clusters):
                return
    found = ", ".join(f"actor {h}" for h in holders) or "nobody"
    raise DecisionRefused(
        f"{d.episode} {d.cluster} is not character {c.id}'s cluster (exemplars: {found})"
    )


def _invalidations(reg: Registry, ws: Workspace, d: Decision) -> list[Invalidation]:
    """Which stages of which episodes the decision makes stale, worked out before it is
    applied (the spec's identity table, as `fukikae registry` does it)."""
    from fukikae.registry.usage import episodes_naming

    if isinstance(d, MergeDecision):
        keep, drop = _require(reg, d.keep), _require(reg, d.drop)
        if keep.actor_id == drop.actor_id:
            raise DecisionRefused(
                f"characters {d.keep} and {d.drop} are one actor's ({keep.actor_id}); "
                "nothing to merge"
            )
        return _inv(reg.episode_counts(drop.actor_id), "characters")
    c = _require(reg, d.character)
    if isinstance(d, NameDecision):
        key = alias_key(d.name)
        users = episodes_with_key(ws, key) if key is not None else []
        return _inv([*reg.episode_counts(c.actor_id), *users], "characters")
    if isinstance(d, AliasDecision):
        key = alias_key(d.name)
        if key is None:
            raise DecisionRefused(
                f"{d.name!r} names nobody (two speakers, a group word, a sign or a song, or "
                "blank); only a single speaker's name can be an alias"
            )
        try:
            before = reg.character_for_key(key)
        except ValueError as e:
            raise DecisionRefused(str(e)) from e
        actors = {c.actor_id} | ({before.actor_id} if before is not None else set())
        eps = {e for aid in actors for e in reg.episode_counts(aid)}
        return _inv([*eps, *episodes_with_key(ws, key)], "characters")
    if isinstance(d, GenderDecision | AcceptDecision):
        return _inv(character_episodes(ws, reg, c.id, c.actor_id), "tts")
    if isinstance(d, KindDecision):
        return _inv(character_episodes(ws, reg, c.id, c.actor_id), "characters")
    if isinstance(d, SplitDecision):
        _cluster_holder(ws, reg, d, c)
        return _inv([d.episode], "names" if d.by == "pitch" else "characters")
    if isinstance(d, LockDecision):
        users = episodes_naming(ws.root, (), (c.id,))
        return _inv([*reg.episode_counts(c.actor_id), *users], "characters")
    assert isinstance(d, ReviewedDecision)
    return []


def preview_lines(invalidations: list[Invalidation]) -> list[str]:
    """``reruns <stage> on E01, E02`` per stage, for the page's preview."""
    by_stage: dict[str, list[str]] = {}
    for inv in invalidations:
        by_stage.setdefault(inv.stage, []).append(inv.episode)
    if not by_stage:
        return ["nothing reruns"]
    return [f"reruns {stage} on {', '.join(eps)}" for stage, eps in by_stage.items()]


def plan_decision(
    reg: Registry, ws: Workspace | Path, d: Decision
) -> tuple[list[str], list[Invalidation]]:
    """What applying ``d`` would invalidate, as preview lines and (episode, stage) pairs;
    `DecisionRefused` for what can be refused without writing (no such character, a merge of
    one actor, a split of another character's cluster, an alias that keys nobody)."""
    invs = _invalidations(reg, _workspace(ws), d)
    return preview_lines(invs), invs


# --- applying ------------------------------------------------------------------------------


def _accept(ws: Workspace, reg: Registry, d: AcceptDecision) -> str:
    """Check the card as `voices assign` does, then assign it: studio, locked."""
    from fukikae.cards.card import CardError
    from fukikae.cards.store import CardStore
    from fukikae.config import normalize_language
    from fukikae.registry.store import _CARD_ID
    from fukikae.stages.tts import check_loaded_card

    language = normalize_language(d.language)
    if not _CARD_ID.fullmatch(d.card_id):
        raise DecisionRefused(f"not a card id (16 lowercase hex digits): {d.card_id!r}")
    store = CardStore(ws.root)
    if not store.has(d.card_id):
        raise DecisionRefused(f"card {d.card_id} is not in the card store ({store.root})")
    who = f"character {d.character}"
    try:
        card = store.get(d.card_id)
        if card.voice.source == "stock":
            raise DecisionRefused(
                f"card {d.card_id} is a stock voice; stock voices are retired (D41), and the "
                "studio accepts only clones"
            )
        check_loaded_card(store, card, d.card_id, who, language)
        missing = store.missing_sidecars(card)
    except CardError as e:
        raise DecisionRefused(str(e)) from e
    if missing:
        raise DecisionRefused(
            f"{who}: card {d.card_id} names sidecar(s) missing from {store.root}: "
            f"{', '.join(missing)}"
        )
    note = f"studio: draft {d.draft} version {d.version}, {d.auditions} auditions; references " + (
        ", ".join(r.label() for r in d.references) or "none"
    )
    if d.overrides:
        note += "; gated override: " + ", ".join(r.label() for r in d.overrides)
    reg.assign_card(d.character, language, d.card_id, "studio", True, note=note)
    return f"character {d.character}: {language} card {d.card_id} accepted (studio, locked)"


def _write(ws: Workspace, reg: Registry, d: Decision, lines: list[str]) -> list:
    """The registry writes of ``d`` (inside the caller's transaction); returns the file
    changes to make after it commits, as callables."""
    after: list = []
    reviewed: list[int] = []
    if isinstance(d, NameDecision):
        warning = reg.rename_character(d.character, d.name)
        if warning:
            lines.append(f"warning: {warning}")
        lines.append(f"character {d.character}: named {d.name!r} (operator)")
        reviewed.append(d.character)
    elif isinstance(d, AliasDecision):
        key = alias_key(d.name)
        assert key is not None  # planned
        got = reg.add_alias(d.character, key, by="operator")
        lines.append(f"character {d.character}: answers to {got.alias!r} (operator)")
        reviewed.append(d.character)
    elif isinstance(d, GenderDecision):
        reg.set_character_gender(d.character, d.gender, by="operator")
        lines.append(
            f"character {d.character}: "
            + (f"gender {d.gender} (operator)" if d.gender else "gender cleared")
        )
        reviewed.append(d.character)
    elif isinstance(d, MergeDecision):
        keep, drop = _require(reg, d.keep), _require(reg, d.drop)
        affected = list(reg.episode_counts(drop.actor_id))
        notes: list[str] = []
        chars = reg.merge_actors(keep.actor_id, drop.actor_id, report=notes)
        lines.extend(notes)
        lines.append(
            f"character {d.drop} merged into character {d.keep} "
            f"(actor {drop.actor_id} into {keep.actor_id})"
        )
        target = next(iter(chars.values()), d.keep)
        reviewed.append(target)
        for eid in affected:
            after.append(
                lambda eid=eid: remap_assignments(
                    ws, eid, {drop.actor_id: keep.actor_id}, character_map=chars
                )
            )
    elif isinstance(d, SplitDecision):
        reviewed.append(d.character)
        if d.by == "cluster":
            old = _require(reg, d.character).actor_id
            new_aid, new_cid = reg.split_cluster(old, d.episode, d.cluster)
            lines.append(f"{d.episode} {d.cluster}: actor {old} -> {new_aid}, character {new_cid}")
            after.append(
                lambda: remap_assignments(
                    ws,
                    d.episode,
                    {old: new_aid},
                    only_cluster=d.cluster,
                    character=(new_aid, new_cid),
                )  # fmt: skip
            )
        else:
            lines.append(
                f"{d.episode} {d.cluster}: split by pitch (male below {d.male_below:g} Hz, "
                f"female above {d.female_above:g} Hz)"
            )
            after.append(lambda: _write_split(ws, d))
    elif isinstance(d, KindDecision):
        reg.set_character_kind(d.character, d.kind)
        lines.append(f"character {d.character}: kind {d.kind}")
        reviewed.append(d.character)
    elif isinstance(d, LockDecision):
        reg.lock_character(d.character, True)
        lines.append(f"character {d.character}: locked")
        reviewed.append(d.character)
    elif isinstance(d, ReviewedDecision):
        lines.append(f"character {d.character}: reviewed")
        reviewed.append(d.character)
    else:
        assert isinstance(d, AcceptDecision)
        lines.append(_accept(ws, reg, d))
    assert (d.op in IDENTITY_OPS) == bool(reviewed)
    for cid in reviewed:
        reg.mark_reviewed(cid)
    return after


def _write_split(ws: Workspace, d: SplitDecision) -> None:
    from fukikae.artifacts import atomic_path

    path = split_path(ws, d.episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    splits[d.cluster] = {"by": "pitch", "male_below": d.male_below, "female_above": d.female_above}
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log(ws: Workspace, d: Decision, result: ApplyResult, force: bool) -> None:
    path = decisions_log(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": d.model_dump(mode="json"),
        "forced": force,
        "lines": result.lines,
        "invalidations": [list(i) for i in result.invalidations],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def apply_decision(
    ws: Workspace | Path, d: Decision, *, dry_run: bool = False, force: bool = False
) -> ApplyResult:
    """Apply one decision: digest check (`StaleDecision` unless ``force``), the registry
    writes and the review mark in one transaction (`DecisionRefused` on a registry refusal,
    nothing changed), then the file changes, ``job.mark_stale`` for each invalidation and
    the log line. ``dry_run`` runs the same checks and writes, rolls them back, and returns
    the preview lines."""
    w = _workspace(ws)
    lines: list[str] = []
    with Registry(w.registry_path) as reg:
        try:
            with reg.transaction():
                ids, _ = _targets(d)
                for cid in ids:
                    _require(reg, cid)
                if not force:
                    _check_digest(reg, d)
                invs = _invalidations(reg, w, d)
                after = _write(w, reg, d, lines)
                if dry_run:
                    raise _DryRun
        except _DryRun:
            return ApplyResult(lines + preview_lines(invs), invs)
        except DecisionError:
            raise
        except (ValueError, KeyError) as e:
            what = f"not found: {e.args[0]!r}" if isinstance(e, KeyError) else str(e)
            raise DecisionRefused(f"{d.op} refused: {what}") from e
        except Exception as e:  # CardError and friends from the card checks
            from fukikae.cards.card import CardError

            if isinstance(e, CardError):
                raise DecisionRefused(str(e)) from e
            raise
    for change in after:
        change()
    by_stage: dict[str, list[str]] = {}
    for inv in invs:
        by_stage.setdefault(inv.stage, []).append(inv.episode)
    for stage, eps in by_stage.items():
        lines.append(mark_stale(w, eps, stage))
    result = ApplyResult(lines, invs)
    _log(w, d, result, force)
    return result


def read_decisions(ws: Workspace | Path, path: Path) -> list[Decision]:
    """The decisions in a file: a `Decisions` document, one decision, a list of them, or
    JSON lines of decisions or of ``decisions.jsonl`` records (whose ``decision`` is
    replayed). A document for another workspace is refused."""
    w = _workspace(ws)
    text = Path(path).read_text(encoding="utf-8")
    try:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "decisions" in obj:
            doc = Decisions.model_validate(obj)
            if doc.workspace and Path(doc.workspace).resolve() != w.root.resolve():
                raise DecisionRefused(
                    f"{path} is for workspace {doc.workspace}, not {w.root}; registry ids are "
                    "workspace-local"
                )
            return list(doc.decisions)
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            obj = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        return [
            parse_decision(x["decision"] if isinstance(x, dict) and "decision" in x else x)
            for x in obj
        ]
    except (ValidationError, json.JSONDecodeError) as e:
        raise DecisionRefused(f"{path}: not a decisions file: {e}") from e


def apply_file(
    ws: Workspace | Path, path: Path, *, dry_run: bool = False, force: bool = False
) -> list[ApplyResult]:
    """Apply a file's decisions in order, each in its own transaction. The first refusal
    stops the file (decisions before it stay applied) and its error says how many were."""
    decisions = read_decisions(ws, path)
    out: list[ApplyResult] = []
    for i, d in enumerate(decisions):
        try:
            out.append(apply_decision(ws, d, dry_run=dry_run, force=force))
        except DecisionError as e:
            done = "checked" if dry_run else "applied"
            raise type(e)(
                f"decision {i + 1} ({d.op}): {e}; {i} of {len(decisions)} decisions {done}"
            ) from e
    return out
