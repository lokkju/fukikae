"""The studio's one writer: decisions applied to the registry, with digests and invalidation.

`apply_decision` is the only way the studio (its page, `fukikae studio apply`, a script)
changes the registry or the card store. Each decision runs in one `Registry.transaction()`:
its characters must exist, the digest of their rows must match the one the decision was made
against (`StaleDecision` otherwise; `UncheckedDecision` when it carries none), and a registry
refusal rolls everything back (`DecisionRefused`). ``force`` skips the digest check, for
every decision it is given; the results say which were unchecked and which carried no
digest, and `fukikae studio apply --force` counts them.

After the commit, `job.mark_stale` runs on every stage the decision invalidates, then the
file changes the registry change needs (assignments remapped after a merge or split, a pitch
split file), then a line goes into ``<ws>/studio/decisions.jsonl``. Those steps are data in
that line. When one fails, the line is still written, with ``incomplete`` naming the errors
and the steps left, and `IncompleteDecision` says the registry part landed; every step is
idempotent, so `resume_incomplete` (`fukikae studio resume`) redoes only what is left, and
until it has, `apply_decision` refuses new decisions.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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
    merge_episodes,
    remap_assignments,
)
from fukikae.registry.store import SCHEMA_VERSION, Character, Registry, alias_key
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
    """The registry rows the decision was made against have changed since. Reload the
    character, or force it."""


class UncheckedDecision(DecisionError):  # noqa: N818
    """The decision carries no digest, so nothing says what it was made against; only
    ``force`` applies it."""


class DecisionRefused(DecisionError):  # noqa: N818
    """The registry (or the card checks) refused the decision; the message is theirs."""


class IncompleteDecision(DecisionError):  # noqa: N818
    """Unlike the others, the decision's registry part landed: a step after the commit (a
    stale mark, a file change, the log line) failed. The message names what is left and how
    to finish it (`resume_incomplete`)."""


@dataclass(frozen=True)
class ApplyResult:
    lines: list[str]
    invalidations: list[Invalidation] = field(default_factory=list)
    unchecked: bool = False  # applied with force: the digest was not checked
    undigested: bool = False  # it carried no digest


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


def _snapshot(reg: Registry, character_id: int) -> dict[str, Any] | None:
    """The rows an identity decision about this character is made against: the character
    (without its review time, which identity decisions themselves set) and its actor (without
    the centroid, which every run's exemplars move)."""
    c = _find(reg, character_id)
    if c is None:
        return None
    a = next(x for x in reg.actors() if x.id == c.actor_id)
    return {
        "character": {
            "id": c.id, "actor_id": c.actor_id, "name": c.name, "name_by": c.name_by,
            "external_id": c.external_id, "locked": c.locked, "gender": c.gender,
            "gender_by": c.gender_by, "kind": c.kind,
        },
        "actor": {"id": a.id, "name": a.name, "external_id": a.external_id, "locked": a.locked},
    }  # fmt: skip


def _card_snapshot(reg: Registry, character_id: int, language: str) -> dict[str, Any] | None:
    """The row an accept is made against: the character's card assignment for the language
    (its number, card, chooser and lock). An identity change meanwhile doesn't make an accept
    stale; a run that re-picked the card does."""
    if _find(reg, character_id) is None:
        return None
    cur = reg.card_for(character_id, language)
    card = None
    if cur is not None:
        card = {"n": cur.n, "card_id": cur.card_id, "chosen_by": cur.chosen_by,
                "locked": cur.locked}  # fmt: skip
    return {"character": character_id, "language": language, "card": card}


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def row_digest(reg: Registry, character_id: int, language: str | None = None) -> str:
    """sha256 of the rows a decision about the character is made against: without
    ``language``, the character and its actor (`_snapshot`, identity decisions); with it,
    only its card assignment for that language (`_card_snapshot`, an accept)."""
    if language is not None:
        return _hash(_card_snapshot(reg, character_id, language))
    return _hash(_snapshot(reg, character_id))


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
    return _hash([_snapshot(reg, cid) for cid in ids])


def stamp(reg: Registry, d: Decision) -> Decision:
    """The decision with its digest, key and external id taken from the registry now: what
    the page does when it builds a decision from the rows it shows."""
    ids, _ = _targets(d)
    c = _find(reg, ids[0])
    key = alias_key(c.name) if c is not None and c.name else None
    ext = c.external_id if c is not None else None
    return d.model_copy(update={"digest": decision_digest(reg, d), "key": key, "external_id": ext})


def _describe(reg: Registry, character_id: int, language: str | None) -> str:
    if language is not None:
        cs = _card_snapshot(reg, character_id, language)
        if cs is None:
            return f"character {character_id} no longer exists"
        card = cs["card"]
        return f"character {character_id}: " + (
            f"{language} card {card['card_id']} ({card['chosen_by']}, "
            f"{'locked' if card['locked'] else 'unlocked'}, assignment {card['n']})"
            if card
            else f"no {language} card"
        )
    snap = _snapshot(reg, character_id)
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
    return f"character {character_id}: " + ", ".join(parts)


def _check_digest(reg: Registry, d: Decision) -> None:
    ids, language = _targets(d)
    who = " and ".join(f"character {i}" for i in ids)
    if d.digest is None:
        raise UncheckedDecision(
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
        return _inv(merge_episodes(ws, reg, drop.actor_id), "characters")
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
        before = _call(d.op, reg.character_for_key, key)
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


def _call(op: str, fn: Any, *args: Any, **kw: Any) -> Any:
    """A registry method, its refusals turned into `DecisionRefused`: ValueError (with its
    ProtectedAssignmentError and AliasConflictError) and KeyError for an unknown id. Called
    only on registry methods, so a ValueError or KeyError of this module's own propagates."""
    try:
        return fn(*args, **kw)
    except KeyError as e:
        raise DecisionRefused(f"{op} refused: not found: {e.args[0]!r}") from e
    except ValueError as e:
        raise DecisionRefused(f"{op} refused: {e}") from e


def _ref_label(r: Any) -> str:
    """A card reference as its origin span (``S01E01 1.00-4.50``), or its clip's short hash."""
    o = r.origin or {}
    try:
        return f"{o['episode']} {float(o['start']):.2f}-{float(o['end']):.2f}"
    except (KeyError, TypeError, ValueError):
        return r.clip[:19]


def _accept(ws: Workspace, reg: Registry, d: AcceptDecision) -> str:
    """Check the card as `voices assign` does, then assign it: studio, locked. Only a clone
    card is accepted (D41: never a stock voice; every voice is a clone of the speaker's own
    speech)."""
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
        if card.voice.source != "clone":
            raise DecisionRefused(
                f"card {d.card_id} is a {card.voice.source} voice; the studio accepts only a "
                "clone of the speaker's own speech (D41: never a stock voice, and no blend, "
                "design or latent voice in its place)"
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
        ", ".join(_ref_label(r) for r in card.voice.references) or "none"
    )
    if d.overrides:
        note += "; gated override: " + ", ".join(r.label() for r in d.overrides)
    _call(d.op, reg.assign_card, d.character, language, d.card_id, "studio", True, note=note)
    return f"character {d.character}: {language} card {d.card_id} accepted (studio, locked)"


def _write(ws: Workspace, reg: Registry, d: Decision, lines: list[str]) -> list[dict[str, Any]]:
    """The registry writes of ``d`` (inside the caller's transaction); returns the file
    changes to make after it commits, as steps (`_run_step`)."""
    after: list[dict[str, Any]] = []
    reviewed: list[int] = []
    op = d.op
    if isinstance(d, NameDecision):
        warning = _call(op, reg.rename_character, d.character, d.name)
        if warning:
            lines.append(f"warning: {warning}")
        lines.append(f"character {d.character}: named {d.name!r} (operator)")
        reviewed.append(d.character)
    elif isinstance(d, AliasDecision):
        key = alias_key(d.name)
        assert key is not None  # planned
        got = _call(op, reg.add_alias, d.character, key, by="operator")
        lines.append(f"character {d.character}: answers to {got.alias!r} (operator)")
        reviewed.append(d.character)
    elif isinstance(d, GenderDecision):
        _call(op, reg.set_character_gender, d.character, d.gender, by="operator")
        lines.append(
            f"character {d.character}: "
            + (f"gender {d.gender} (operator)" if d.gender else "gender cleared")
        )
        reviewed.append(d.character)
    elif isinstance(d, MergeDecision):
        keep, drop = _require(reg, d.keep), _require(reg, d.drop)
        affected = merge_episodes(ws, reg, drop.actor_id)
        notes: list[str] = []
        chars = _call(op, reg.merge_actors, keep.actor_id, drop.actor_id, report=notes)
        lines.extend(notes)
        lines.append(
            f"character {d.drop} merged into character {d.keep} "
            f"(actor {drop.actor_id} into {keep.actor_id})"
        )
        reviewed.append(next(iter(chars.values()), d.keep))
        after.extend(
            {
                "do": "remap",
                "episode": eid,
                "actor_map": [[drop.actor_id, keep.actor_id]],
                "character_map": [[k, v] for k, v in chars.items()],
            }  # fmt: skip
            for eid in affected
        )
    elif isinstance(d, SplitDecision):
        reviewed.append(d.character)
        if d.by == "cluster":
            old = _require(reg, d.character).actor_id
            new_aid, new_cid = _call(op, reg.split_cluster, old, d.episode, d.cluster)
            lines.append(f"{d.episode} {d.cluster}: actor {old} -> {new_aid}, character {new_cid}")
            after.append({
                "do": "remap", "episode": d.episode, "actor_map": [[old, new_aid]],
                "only_cluster": d.cluster, "character": [new_aid, new_cid],
            })  # fmt: skip
        else:
            lines.append(
                f"{d.episode} {d.cluster}: split by pitch (male below {d.male_below:g} Hz, "
                f"female above {d.female_above:g} Hz)"
            )
            after.append({
                "do": "split", "episode": d.episode, "cluster": d.cluster,
                "male_below": d.male_below, "female_above": d.female_above,
            })  # fmt: skip
    elif isinstance(d, KindDecision):
        _call(op, reg.set_character_kind, d.character, d.kind)
        lines.append(f"character {d.character}: kind {d.kind}")
        reviewed.append(d.character)
    elif isinstance(d, LockDecision):
        _call(op, reg.lock_character, d.character, True)
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
        _call(op, reg.mark_reviewed, cid)
    return after


def _write_split(
    ws: Workspace, episode: str, cluster: str, male_below: float, female_above: float
) -> None:
    """Add one cluster's pitch split to the episode's split file, the read-modify-write under
    an exclusive lock on ``studio/splits/.lock`` so two writers can't lose each other's."""
    from fukikae.artifacts import atomic_path

    path = split_path(ws, episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        splits: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
        splits[cluster] = {"by": "pitch", "male_below": male_below, "female_above": female_above}
        with atomic_path(path) as tmp:
            tmp.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- steps after the commit, the log, and finishing an incomplete decision ------------------


def _step_label(s: dict[str, Any]) -> str:
    if s["do"] == "stale":
        return f"mark {s['stage']} stale on {', '.join(s['episodes'])}"
    if s["do"] == "remap":
        return f"remap {s['episode']} assignments.json"
    return f"write {s['episode']}'s pitch split for {s['cluster']}"


def _run_step(ws: Workspace, s: dict[str, Any]) -> str | None:
    """One step after the commit, from its data (so a logged step can be redone). Each is
    idempotent: a stale mark, a remap that maps ids already mapped to themselves, a split
    entry written again."""
    if s["do"] == "stale":
        return mark_stale(ws, s["episodes"], s["stage"])
    if s["do"] == "remap":
        character = s.get("character")
        remap_assignments(
            ws,
            s["episode"],
            {int(a): int(b) for a, b in s["actor_map"]},
            only_cluster=s.get("only_cluster"),
            character=(int(character[0]), int(character[1])) if character else None,
            character_map={int(a): int(b) for a, b in s.get("character_map") or []} or None,
        )
        return None
    assert s["do"] == "split", s
    _write_split(ws, s["episode"], s["cluster"], s["male_below"], s["female_above"])
    return None


def _run_steps(
    ws: Workspace, steps: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Run every step, carrying on past a failure (the steps are independent); returns the
    lines the done ones printed, the steps left and their errors."""
    lines: list[str] = []
    left: list[dict[str, Any]] = []
    errors: list[str] = []
    for s in steps:
        try:
            line = _run_step(ws, s)
        except Exception as e:  # noqa: BLE001 (the record must be written; re-raised below)
            left.append(s)
            errors.append(f"{_step_label(s)}: {type(e).__name__}: {e}")
            continue
        lines.append(line or f"done: {_step_label(s)}")
    return lines, left, errors


def _log_records(ws: Workspace) -> list[dict[str, Any]]:
    path = decisions_log(ws)
    if not path.is_file():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def incomplete_records(ws: Workspace | Path) -> list[dict[str, Any]]:
    """The log's incomplete records that no later resume finished, oldest first."""
    pending: dict[str, dict[str, Any]] = {}
    for rec in _log_records(_workspace(ws)):
        if rec.get("resumes"):
            pending.pop(rec["resumes"], None)
        if rec.get("incomplete") and rec.get("id"):
            pending[rec["id"]] = rec
    return list(pending.values())


def _record_op(rec: dict[str, Any]) -> str:
    return (rec.get("decision") or {}).get("op") or rec.get("op") or "decision"


def _refuse_if_incomplete(ws: Workspace) -> None:
    pending = incomplete_records(ws)
    if pending:
        rec = pending[0]
        raise DecisionRefused(
            f"the {_record_op(rec)} logged at {rec.get('time')} is incomplete "
            f"({'; '.join(rec['incomplete']['errors'])}); fix the cause and run "
            f"`fukikae studio resume {ws.root}` before applying more decisions"
        )


def _append_log(ws: Workspace, rec: dict[str, Any]) -> None:
    path = decisions_log(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _new_record(ws: Workspace) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "workspace": str(ws.root),
    }


def _finish(
    ws: Workspace,
    d: Decision,
    lines: list[str],
    invs: list[Invalidation],
    steps: list[dict[str, Any]],
    force: bool,
) -> ApplyResult:
    """After the commit: the stale marks first (so a file change that fails can't leave an
    episode looking current), then the file changes, then the log line, which is written
    whatever failed before it."""
    by_stage: dict[str, list[str]] = {}
    for inv in invs:
        by_stage.setdefault(inv.stage, []).append(inv.episode)
    stale = [{"do": "stale", "stage": st, "episodes": eps} for st, eps in by_stage.items()]
    done, left, errors = _run_steps(ws, stale + steps)
    lines = lines + [ln for ln in done if not ln.startswith("done: ")]
    result = ApplyResult(lines, invs, unchecked=force, undigested=d.digest is None)
    rec = _new_record(ws) | {
        "decision": d.model_dump(mode="json"),
        "forced": force,
        "lines": result.lines,
        "invalidations": [list(i) for i in invs],
    }
    if left:
        rec["incomplete"] = {"errors": errors, "left": left}
    try:
        _append_log(ws, rec)
    except OSError as e:
        todo = ", ".join(_step_label(s) for s in left) or "nothing"
        raise IncompleteDecision(
            f"{d.op} landed in the registry, but the decisions log {decisions_log(ws)} could "
            f"not be written ({e}); `studio resume` can't finish what it doesn't know about, "
            f"so finish by hand: {todo}" + (f" (errors: {'; '.join(errors)})" if errors else "")
        ) from e
    if left:
        stale_ok = not any(s["do"] == "stale" for s in left)
        raise IncompleteDecision(
            f"{d.op} landed in the registry, but {len(left)} step(s) after it failed: "
            f"{'; '.join(errors)}. "
            + ("Its invalidations are marked stale. " if stale_ok else "")
            + f"Fix the cause and run `fukikae studio resume {ws.root}`, which redoes only "
            "what is left; until then `studio apply` refuses new decisions"
        )
    return result


def resume_incomplete(ws: Workspace | Path) -> list[ApplyResult]:
    """Finish every incomplete decision in the log: run the steps it left (all idempotent)
    and append a ``resumes`` record. `IncompleteDecision` when a step still fails (its new
    record carries what is left, so a later resume picks it up)."""
    w = _workspace(ws)
    out: list[ApplyResult] = []
    for rec in incomplete_records(w):
        done, left, errors = _run_steps(w, rec["incomplete"]["left"])
        new = _new_record(w) | {"resumes": rec["id"], "op": _record_op(rec), "lines": done}
        if left:
            new["incomplete"] = {"errors": errors, "left": left}
        _append_log(w, new)
        if left:
            raise IncompleteDecision(
                f"the {_record_op(rec)} logged at {rec.get('time')} is still incomplete: "
                f"{'; '.join(errors)}; fix the cause and run `fukikae studio resume {w.root}` "
                "again"
            )
        invs = [Invalidation(*i) for i in rec.get("invalidations", [])]
        out.append(ApplyResult(done, invs))
    return out


# --- the registry, transactions, decisions and files ---------------------------------------


def _schema_version(path: Path) -> int | None:
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return None if row is None else row[0]
    except sqlite3.DatabaseError:
        return None
    finally:
        con.close()


def _busy(e: sqlite3.OperationalError) -> bool:
    msg = str(e).lower()
    return "locked" in msg or "busy" in msg


@contextmanager
def _session(w: Workspace, dry_run: bool) -> Iterator[Registry]:
    """The workspace's registry, open. Refused when there is none (a decision names registry
    rows; nothing creates one here), and in a dry run when opening it would migrate it. A
    locked registry (another run or studio writing it) is a refusal, not a traceback."""
    path = w.registry_path
    if not path.is_file():
        raise DecisionRefused(
            f"{w.root} has no registry ({path}); decisions name registry rows, so there is "
            "nothing to apply them to (`fukikae run` creates it)"
        )
    if dry_run:
        version = _schema_version(path)
        if version is not None and version < SCHEMA_VERSION:
            raise DecisionRefused(
                f"registry {path} is schema version {version}, and a dry run never migrates "
                f"it to {SCHEMA_VERSION}; apply for real, or run `fukikae registry list "
                f"{w.root}`, which migrates it first"
            )
    try:
        with Registry(path) as reg:
            yield reg
    except sqlite3.OperationalError as e:
        if not _busy(e):
            raise
        raise DecisionRefused(
            f"the registry {path} is busy ({e}): another `fukikae run` or studio is writing "
            "it; nothing changed, try again when it is done"
        ) from e


def _stage(
    reg: Registry, w: Workspace, d: Decision, force: bool
) -> tuple[list[str], list[Invalidation], list[dict[str, Any]]]:
    """The checks, invalidations and registry writes of one decision in a transaction (the
    caller's, when one is open: a file dry run nests every decision in one)."""
    with reg.transaction():
        ids, _ = _targets(d)
        for cid in ids:
            _require(reg, cid)
        if not force:
            _check_digest(reg, d)
        invs = _invalidations(reg, w, d)
        lines: list[str] = []
        steps = _write(w, reg, d, lines)
    return lines, invs, steps


def _preview(d: Decision, lines: list[str], invs: list[Invalidation], force: bool) -> ApplyResult:
    return ApplyResult(
        lines + preview_lines(invs), invs, unchecked=force, undigested=d.digest is None
    )


def apply_decision(
    ws: Workspace | Path, d: Decision, *, dry_run: bool = False, force: bool = False
) -> ApplyResult:
    """Apply one decision: digest check (`StaleDecision`, or `UncheckedDecision` with no
    digest, unless ``force``), the registry writes and the review mark in one transaction
    (`DecisionRefused` on a registry refusal, nothing changed), then ``job.mark_stale`` for
    each invalidation, the file changes and the log line (`IncompleteDecision` when one of
    those fails). ``dry_run`` runs the same checks and writes, rolls them back, and returns
    the preview lines."""
    w = _workspace(ws)
    _refuse_if_incomplete(w)
    with _session(w, dry_run) as reg:
        if dry_run:
            try:
                with reg.transaction():
                    lines, invs, _ = _stage(reg, w, d, force)
                    raise _DryRun
            except _DryRun:
                return _preview(d, lines, invs, force)
        lines, invs, steps = _stage(reg, w, d, force)
    return _finish(w, d, lines, invs, steps, force)


def read_decisions(ws: Workspace | Path, path: Path) -> list[Decision]:
    """The decisions in a file: a `Decisions` document, one decision, a list of them, or
    JSON lines of decisions or of ``decisions.jsonl`` records (whose ``decision`` is
    replayed; resume records carry none and are skipped). A document or record for another
    workspace is refused."""
    w = _workspace(ws)
    text = Path(path).read_text(encoding="utf-8")

    def ours(where: str | None) -> None:
        if where and Path(where).resolve() != w.root.resolve():
            raise DecisionRefused(
                f"{path} is for workspace {where}, not {w.root}; registry ids are workspace-local"
            )

    try:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "decisions" in obj:
            doc = Decisions.model_validate(obj)
            ours(doc.workspace)
            return list(doc.decisions)
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            obj = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        out: list[Decision] = []
        for x in obj:
            if isinstance(x, dict) and "resumes" in x:
                continue
            if isinstance(x, dict) and "decision" in x:
                ours(x.get("workspace"))
                x = x["decision"]
            out.append(parse_decision(x))
        return out
    except (ValidationError, json.JSONDecodeError) as e:
        raise DecisionRefused(f"{path}: not a decisions file: {e}") from e


def apply_file(
    ws: Workspace | Path, path: Path, *, dry_run: bool = False, force: bool = False
) -> list[ApplyResult]:
    """Apply a file's decisions in order, each in its own transaction. The first refusal
    stops the file (decisions before it stay applied) and its error says how many were.
    A dry run checks them all in one transaction, rolled back at the end, so a decision that
    depends on an earlier one is checked as it will apply."""
    w = _workspace(ws)
    decisions = read_decisions(w, path)
    n = len(decisions)
    out: list[ApplyResult] = []
    if dry_run:
        _refuse_if_incomplete(w)
        with _session(w, dry_run=True) as reg:
            try:
                with reg.transaction():
                    for i, d in enumerate(decisions):
                        try:
                            lines, invs, _ = _stage(reg, w, d, force)
                        except DecisionError as e:
                            raise type(e)(
                                f"decision {i + 1} ({d.op}): {e}; {i} of {n} decisions checked"
                            ) from e
                        out.append(_preview(d, lines, invs, force))
                    raise _DryRun
            except _DryRun:
                return out
    for i, d in enumerate(decisions):
        try:
            out.append(apply_decision(w, d, force=force))
        except DecisionError as e:
            done = i + 1 if isinstance(e, IncompleteDecision) else i
            raise type(e)(f"decision {i + 1} ({d.op}): {e}; {done} of {n} decisions applied") from e
    return out
