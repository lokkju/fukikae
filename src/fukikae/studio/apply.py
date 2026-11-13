"""The studio's one writer: decisions applied to the registry, with digests and invalidation.

`apply_decision` is the only way the studio (its page, `fukikae studio apply`, a script)
changes the registry or the card store. Each decision runs in one `Registry.transaction()`:
its characters must exist, the digest of their rows must match the one the decision was made
against (`StaleDecision` otherwise; `UncheckedDecision` when it carries none), and a registry
refusal rolls everything back (`DecisionRefused`). ``force`` skips the digest check, for
every decision it is given; the results say which were unchecked and which carried no
digest, and `fukikae studio apply --force` counts them.

Before the transaction commits, a write-ahead record goes into
``<ws>/studio/decisions.jsonl`` (the decision, its steps, ``incomplete: {left: <every step>,
pending: true}``) and a `decision_marks` row with the record's id goes into the registry in
the same transaction, so the row says afterwards whether the commit landed. After the
commit, `job.mark_stale` runs on every stage the decision invalidates, then the file changes
the registry change needs (assignments remapped after a merge or split, a pitch split file),
then a completion record closes the write-ahead one, naming any step that failed or was
interrupted (`IncompleteDecision` says the registry part landed). Every step is idempotent,
so `resume_incomplete` (`fukikae studio resume`) redoes only what is left; a pending record
whose commit never landed is logged as aborted, with nothing to redo. Until every record is
finished, `apply_decision`, `fukikae registry merge` (`apply_registry_change`, the same
records) and `fukikae run` (`unfinished_message`) refuse to start. Writers hold
``studio/.lock`` exclusively and `job.run_stages` holds it shared (`run_lock`), so neither
runs beside the other. A replay skips what the registry's `decision_marks` say landed
(`_replay_state`), never what the log alone says.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
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
    """A decision that was not applied; nothing it would have changed has changed. From
    `apply_file`, ``results`` holds the results of the decisions before it in the file."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.results: list[ApplyResult] = []


class StaleDecision(DecisionError):  # noqa: N818 (the plan's name)
    """The registry rows the decision was made against have changed since. Reload the
    character, or force it."""


class UncheckedDecision(DecisionError):  # noqa: N818
    """The decision carries no digest, so nothing says what it was made against; only
    ``force`` applies it."""


class DecisionRefused(DecisionError):  # noqa: N818
    """The registry (or the card checks) refused the decision; the message is theirs."""


class StudioBusy(DecisionRefused):  # noqa: N818
    """Another process holds ``studio/.lock``: a writer (named when it said who it is) or a
    `fukikae run`. Nothing changed."""


class LogUnreadable(DecisionRefused):  # noqa: N818
    """A line of ``studio/decisions.jsonl`` is not a record; nothing proceeds until the
    operator looks at it."""


class RegistryUnreadable(DecisionRefused):  # noqa: N818
    """The registry can't be read to tell which logged decisions landed."""


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
    skipped: bool = False  # a replayed log record whose decision had already landed


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
    own, overlapping = _own_speech(ws, reg, d.character, d.card_id, card)
    note = f"studio: draft {d.draft} version {d.version}, {d.auditions} auditions; references " + (
        ", ".join(_ref_label(r) for r in card.voice.references) or "none"
    )
    if overlapping:
        note += "; " + _overlap_phrase(overlapping)
    if d.overrides:
        note += "; gated override: " + ", ".join(r.label() for r in d.overrides)
    _call(d.op, reg.assign_card, d.character, language, d.card_id, "studio", True, note=note)
    return f"character {d.character}: {language} card {d.card_id} accepted (studio, locked); {own}"


# How far a reference may run past the character's turns (the origin's float rounding).
_SPAN_SLACK_S = 0.01
_Spans = list[tuple[float, float]]


def _own_turns(
    asg: Any, character: int
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The character's turns in an episode and every other turn. Its turns are the diarize
    turns of its clusters (extras and review clusters left out, as `cards.auto.character_spans`
    reads them), adjacent ones joined across a gap of at most `select.JOIN_GAP_S` that no
    other turn falls in, as reference selection joins them; the others are every other
    cluster's turns (its own extra or review clusters' included), which a reference may not
    overlap (`select.clean_span`)."""
    from fukikae.cards.select import JOIN_GAP_S

    own = {
        a.cluster
        for a in asg.clusters
        if a.character_id == character and a.status not in ("extra", "review")
    }
    turns = [(float(t.start), float(t.end), t.speaker in own) for t in asg.turns or []]
    mine = sorted((s, e) for s, e, ours in turns if ours)
    others = [(s, e) for s, e, ours in turns if not ours]
    out: list[tuple[float, float]] = []
    for a, b in mine:
        if out:
            pa, pb = out[-1]
            gap_free = not any(s < a and e > pb for s, e in others)
            if a - pb <= JOIN_GAP_S and gap_free:
                out[-1] = (pa, max(pb, b))
                continue
        out.append((a, b))
    return out, others


def _cue_spans(work: Path, character: int) -> list[tuple[float, float]]:
    """The source spans (``delivery.reference``) of the character's own spoken cues in the
    episode, from every ``06-translate/cues.<target>.json``, joined where at most
    `select.JOIN_GAP_S` apart: what the cue-thin tier (`cards.auto.cue_references`) clones a
    character with no turns from. The subtitle names put the character on these lines."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import _spoken
    from fukikae.cards.select import JOIN_GAP_S
    from fukikae.models import DubCues

    spans: set[tuple[float, float]] = set()
    for path in sorted((work / "06-translate").glob("cues.*.json")):
        try:
            cues = read_json(path, DubCues).cues
        except (OSError, ValueError):
            continue
        spans |= {
            (float(c.delivery.reference[0]), float(c.delivery.reference[1]))
            for c in cues
            if c.character_id == character and _spoken(c)
        }
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= JOIN_GAP_S:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _overlap_phrase(labels: list[str]) -> str:
    return "overlaps another speaker's turn: " + ", ".join(labels)


def _own_speech(
    ws: Workspace, reg: Registry, character: int, card_id: str, card: Any
) -> tuple[str, list[str]]:
    """Refuse a clone card whose references are not verifiably the character's own speech
    (D41), and say what was verified. A reference with an origin span (episode, start, end)
    must lie inside one of the character's turns in that episode, read from the episode's
    assignments (the registry keeps no exemplar spans), and overlap no other cluster's turn
    (`select.clean_span`, the rule reference selection uses); or lie inside the source spans
    of the character's own spoken cues there (`_cue_spans`, what the cue-thin tier clones a
    character with no turns from). A reference with no origin (a sidecar only, such as a
    `voices clone` clip) can't be checked against either; it is taken only when the card came
    from this character's own clone path: the card's ``[meta] character`` names it, or the
    character's card history already names the card.

    A thin-tier card (``[meta] reference_tier = "thin"``), which the pipeline builds from the
    character's turns whether or not another speaker overlaps them, is not held to the clean
    rule: a reference inside the character's turns that overlaps another speaker is taken,
    and returned among the overlapping labels for the accept line and note to list. Returns
    the line saying what was verified and those labels."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import is_thin
    from fukikae.cards.select import clean_span
    from fukikae.models import Assignments

    turns_of: dict[str, tuple[_Spans, _Spans] | None] = {}
    cues_of: dict[str, _Spans] = {}

    def turns(ep: str) -> tuple[_Spans, _Spans] | None:
        if ep not in turns_of:
            path = ws.work_dir(ep) / "05-characters" / "assignments.json"
            got = None
            if path.is_file():
                try:
                    asg = read_json(path, Assignments)
                except (OSError, ValueError):
                    asg = None
                if asg is not None and asg.turns is not None:
                    got = _own_turns(asg, character)
            turns_of[ep] = got
        return turns_of[ep]

    def cue_spans(ep: str) -> _Spans:
        if ep not in cues_of:
            cues_of[ep] = _cue_spans(ws.work_dir(ep), character)
        return cues_of[ep]

    def within(start: float, end: float, spans: _Spans) -> bool:
        return any(a - _SPAN_SLACK_S <= start and end <= b + _SPAN_SLACK_S for a, b in spans)

    thin = is_thin(card)
    refs = card.voice.references
    refused: list[str] = []
    overlapping: list[str] = []
    verified = bare = 0
    for r in refs:
        o = r.origin or {}
        try:
            ep, start, end = str(o["episode"]), float(o["start"]), float(o["end"])
        except (KeyError, TypeError, ValueError):
            bare += 1
            continue
        label = f"{ep} {start:.2f}-{end:.2f}"
        if within(start, end, cue_spans(ep)):
            verified += 1
            continue
        got = turns(ep)
        if got is None:
            refused.append(
                f"{label} (no assignments with turns for {ep} to check it against, and not "
                f"inside character {character}'s spoken cues there)"
            )
            continue
        mine, others = got
        if not within(start, end, mine):
            refused.append(f"{label} (not inside character {character}'s turns or cues in {ep})")
        elif not clean_span(start, end, others, slack=_SPAN_SLACK_S) and thin:
            verified += 1
            overlapping.append(label)
        elif not clean_span(start, end, others, slack=_SPAN_SLACK_S):
            refused.append(
                f"{label} (inside character {character}'s turns, but it overlaps another "
                f"speaker's turn in {ep}, so it is not the character's voice alone)"
            )
        else:
            verified += 1
    if bare:
        tagged = card.meta.get("character") if isinstance(card.meta, dict) else None
        if tagged is not None:
            own_path = str(tagged) == str(character)
            how = f"its [meta] character is {tagged}"
        else:
            own_path = card_id in reg.card_ids_for(character)
            names = "names" if own_path else "never names"
            how = f"character {character}'s card history {names} it"
        if not own_path:
            refused.append(
                f"{bare} reference(s) with no origin span (a sidecar only), and the card is not "
                f"from character {character}'s own clone path ({how})"
            )
    if refused:
        raise DecisionRefused(
            f"character {character}: card {card_id} is not verifiably the character's own "
            f"speech (D41: only a clone of the speaker's own speech): {verified} of "
            f"{len(refs)} reference(s) verified as its own (inside its clean turns or its "
            "spoken cues); could not verify " + "; ".join(refused)
        )
    line = (
        f"{verified} of {len(refs)} references verified as its own speech (inside its clean "
        "turns or its spoken cues)"
    )
    if bare:
        line += (
            f"; {bare} with no origin span not verified against its turns, taken because the "
            f"card came from character {character}'s own clone path ({how})"
        )
    if overlapping:
        line += f"; thin-tier card, taken though it {_overlap_phrase(overlapping)}"
    return line, overlapping


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


# --- steps after the commit ----------------------------------------------------------------


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


@dataclass
class _Progress:
    """What `_run_steps` got through, filled in as it goes so an interrupt can record it."""

    steps: list[dict[str, Any]]
    lines: list[str] = field(default_factory=list)
    done: set[int] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    @property
    def left(self) -> list[dict[str, Any]]:
        return [s for i, s in enumerate(self.steps) if i not in self.done]


def _run_steps(ws: Workspace, p: _Progress) -> None:
    """Run every step, carrying on past a failure (the steps are independent). An interrupt
    (Ctrl-C, SystemExit) stops at once and propagates, with ``p`` saying what was done."""
    for i, s in enumerate(p.steps):
        try:
            line = _run_step(ws, s)
        except Exception as e:  # noqa: BLE001 (recorded; the caller raises IncompleteDecision)
            p.errors.append(f"{_step_label(s)}: {type(e).__name__}: {e}")
            continue
        except BaseException as e:
            p.errors.append(f"{_step_label(s)}: interrupted ({type(e).__name__})")
            raise
        p.done.add(i)
        p.lines.append(line or f"done: {_step_label(s)}")


def _stale_steps(invs: list[Invalidation]) -> list[dict[str, Any]]:
    by_stage: dict[str, list[str]] = {}
    for inv in invs:
        by_stage.setdefault(inv.stage, []).append(inv.episode)
    return [{"do": "stale", "stage": st, "episodes": eps} for st, eps in by_stage.items()]


# --- the decisions log: write-ahead, completion, aborted and resume records -----------------
#
# A decision (or `fukikae registry merge`) appends a write-ahead record inside its registry
# transaction, before COMMIT: the decision (or the registry command), its lines, its
# invalidations, its steps, and ``incomplete: {left: <every step>, pending: true}``. The same
# transaction inserts a `decision_marks` row under the record's id, so the row exists exactly
# when the registry writes do. After the commit and the steps, a completion record
# (``completes: <id>``) closes it, carrying ``incomplete: {left, errors}`` when a step failed
# or was interrupted. A pending record with no completion is settled by the next writer:
# without its mark the commit never landed, and an ``aborts: <id>`` record closes it with
# nothing to redo; with it, its steps are left and `resume_incomplete` reruns them, closing it
# with ``resumes: <id>``. Writers hold ``studio/.lock`` from before the write-ahead record to
# after the completion record, so a pending record seen by a lock holder is never live.


def _log_records(ws: Workspace) -> list[dict[str, Any]]:
    """Every record of the log, oldest first. A line that is not a JSON object (half a record
    from a write cut short by a crash) is `LogUnreadable`, never skipped: it might be the
    record that says a decision is unfinished."""
    path = decisions_log(ws)
    if not path.is_file():
        return []
    out = []
    for n, ln in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            why = None if isinstance(rec, dict) else f"a JSON {type(rec).__name__}, not a record"
        except json.JSONDecodeError as e:
            why = f"not JSON: {e}"
        if why is not None:
            raise LogUnreadable(
                f"line {n} of {path} is not a decisions record ({why}; it begins "
                f"{ln[:60]!r}), so nothing can tell whether a decision is unfinished; nothing "
                "proceeds until it is dealt with. Half a record is what a crash mid-write "
                "leaves, and it is safe to delete: a write-ahead record is complete on disk "
                "before its commit, and whatever a later record would have closed is redone "
                f"by `fukikae studio resume {ws.root}`. Delete or repair the line, then try "
                "again"
            )
        out.append(rec)
    return out


_CLOSERS = ("completes", "resumes", "aborts")


def _open_records(ws: Workspace) -> list[dict[str, Any]]:
    """Records with ``incomplete`` that no later record closed, oldest first: pending
    write-ahead records and completion or resume records that left steps."""
    pending: dict[str, dict[str, Any]] = {}
    for rec in _log_records(ws):
        for k in _CLOSERS:
            if rec.get(k):
                pending.pop(rec[k], None)
        if rec.get("incomplete") and rec.get("id"):
            pending[rec["id"]] = rec
    return list(pending.values())


def _is_pending(rec: dict[str, Any]) -> bool:
    return bool((rec.get("incomplete") or {}).get("pending"))


def _marks(ws: Workspace, what: str) -> set[str]:
    """The ids in the registry's `decision_marks` (read-only; none without a registry or the
    table). `RegistryUnreadable` when it can't be read; ``what`` says what was being asked."""
    path = ws.registry_path
    if not path.is_file():
        return set()
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("SELECT id FROM decision_marks")}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return set()
        raise RegistryUnreadable(
            f"can't tell {what}: the registry {path} can't be read ({e}); try again"
        ) from e
    except sqlite3.DatabaseError as e:
        raise RegistryUnreadable(
            f"can't tell {what}: {path} is not a readable registry ({e}); restore it, or check "
            "the file"
        ) from e
    finally:
        con.close()


def _landed(ws: Workspace, record_id: str) -> bool:
    """Whether the write-ahead record's transaction committed: its `decision_marks` row."""
    return record_id in _marks(ws, f"whether the decision logged as {record_id} landed")


def _classify(ws: Workspace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The open records split into those with steps left (pending records whose commit
    landed, and incomplete completions) and pending records whose commit never landed."""
    todo: list[dict[str, Any]] = []
    aborted: list[dict[str, Any]] = []
    for rec in _open_records(ws):
        if _is_pending(rec) and not _landed(ws, rec["id"]):
            aborted.append(rec)
        else:
            todo.append(rec)
    return todo, aborted


def _lock_path(ws: Workspace | Path) -> Path:
    root = ws.root if isinstance(ws, Workspace) else Path(ws)
    return root / "studio" / ".lock"


def _holder(path: Path) -> str | None:
    """Who holds ``studio/.lock`` exclusively, as the writer wrote it there."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _lock_refusal(ws: Workspace, path: Path, f: Any) -> StudioBusy:
    """The refusal when ``studio/.lock`` can't be had exclusively: shared means `fukikae run`
    (only runs take it shared), exclusive another writer, named when it said who it is."""
    try:
        fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        who = _holder(path) or "another `fukikae studio` or `fukikae registry` command"
        return StudioBusy(
            f"{who} is applying decisions to {ws.root} ({path} is held); nothing changed, try "
            "again when it is done"
        )
    fcntl.flock(f, fcntl.LOCK_UN)
    return StudioBusy(
        f"a `fukikae run` is in progress on this workspace ({ws.root}; it holds {path}), and "
        "its stages read the assignments and manifests a decision changes; nothing changed, "
        "try again when it is done"
    )


@contextmanager
def _studio_lock(ws: Workspace, holder: str) -> Iterator[None]:
    """``studio/.lock``, held exclusively by a writer from before its write-ahead record to
    after its completion record, with ``holder`` (and the pid) written in it so a refusal can
    name it. Taken without waiting: another writer, or a `fukikae run` (which holds it
    shared), is `StudioBusy`."""
    path = _lock_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise _lock_refusal(ws, path, f) from e
        try:
            f.truncate(0)
            f.write(f"{holder} (pid {os.getpid()})\n")
            f.flush()
            yield
        finally:
            with suppress(OSError):
                f.truncate(0)
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def studio_lock(ws: Workspace | Path, holder: str) -> Iterator[None]:
    """`_studio_lock` for the `fukikae registry` commands that change identity outside a
    recorded decision (rename, alias, lock, kind, gender): their registry write and stale
    marks never run beside a decision or a `fukikae run`."""
    with _studio_lock(_workspace(ws), holder):
        yield


@contextmanager
def _no_writer(ws: Workspace) -> Iterator[None]:
    """``studio/.lock`` held shared, for reading the log: no writer is live while it is held,
    so a pending record is never one being written. A `fukikae run` holds it shared too."""
    path = _lock_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as e:
            who = _holder(path) or "another `fukikae studio` or `fukikae registry` command"
            raise StudioBusy(f"{who} is applying decisions to {ws.root} ({path} is held)") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def run_lock(root: Path) -> Iterator[bool]:
    """``studio/.lock`` held shared for one `job.run_stages` call, so no studio or registry
    writer changes assignments, manifests or the registry under a run (they take it
    exclusively, without waiting, and are refused). Only when ``<root>/studio`` exists: a run
    creates the lock file at most, never the studio. ValueError, naming the writer, while one
    holds it. Yields whether it is held."""
    path = _lock_path(root)
    if not path.parent.is_dir():
        yield False
        return
    with path.open("a", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as e:
            who = _holder(path) or "a `fukikae studio` or `fukikae registry` command"
            raise ValueError(
                f"{who} is applying a decision to {root} ({path} is held); its registry change "
                "and the stale marks and file changes after it must finish before a run "
                "starts; try again when it is done"
            ) from e
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def incomplete_records(ws: Workspace | Path) -> list[dict[str, Any]]:
    """The log's records whose steps are still to do, oldest first: pending write-ahead
    records whose commit landed (every step left) and completion or resume records that left
    steps. A pending record whose commit never landed is aborted, so it is not among them
    (the next writer logs that). Read-only. `StudioBusy` while a writer holds the studio
    lock and a pending record might be its own; `LogUnreadable` for a line that is no record;
    `RegistryUnreadable` when the registry can't say whether a pending record landed."""
    w = _workspace(ws)
    recs = _open_records(w)
    if not any(_is_pending(r) for r in recs):
        return recs
    with _no_writer(w):
        return _classify(w)[0]


def unfinished_message(ws: Workspace | Path) -> str | None:
    """Why stages that read assignments must not start now (a decision whose steps are left,
    one being applied, a log line that is no record, a registry that can't say whether a
    decision landed), or None. What `fukikae run` checks."""
    w = _workspace(ws)
    if not decisions_log(w).is_file():
        return None
    try:
        recs = incomplete_records(w)
    except StudioBusy as e:
        return f"{e}; assignments may be mid-change"
    except DecisionRefused as e:
        return str(e)
    if not recs:
        return None
    return (
        f"the {_record_op(recs[0])} logged at {recs[0].get('time')} is incomplete "
        f"({_why(recs[0])}), so assignments may not match the registry; run "
        f"`fukikae studio resume {w.root}` first"
    )


def _record_op(rec: dict[str, Any]) -> str:
    return (rec.get("decision") or {}).get("op") or rec.get("op") or "decision"


def _why(rec: dict[str, Any]) -> str:
    errors = (rec.get("incomplete") or {}).get("errors")
    if errors:
        return "; ".join(errors)
    return "interrupted after its registry part committed, before its steps finished"


_fsync = os.fsync  # a seam for the tests' failing disk


def _append_log(ws: Workspace, rec: dict[str, Any]) -> None:
    """Append one record as one line, on disk (fsync) before returning. A file that doesn't end
    in a newline (half a line from a crash) gets one first, so the record is a line of its
    own. On OSError the file is cut back to its size before the append, so a failed write
    leaves no half record behind."""
    path = decisions_log(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    created = not path.exists()
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        try:
            if size and os.pread(fd, 1, size - 1) != b"\n":
                data = b"\n" + data
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            _fsync(fd)
            if created:
                dfd = os.open(path.parent, os.O_RDONLY)
                try:
                    _fsync(dfd)
                finally:
                    os.close(dfd)
        except OSError:
            with suppress(OSError):
                os.ftruncate(fd, size)
            raise
    finally:
        os.close(fd)


def _new_record(ws: Workspace) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "workspace": str(ws.root.resolve()),
    }


def _settle(ws: Workspace) -> list[dict[str, Any]]:
    """With the studio lock held: log an ``aborts`` record for each pending record whose
    commit never landed, and return the records with steps left."""
    todo, aborted = _classify(ws)
    for rec in aborted:
        _append_log(ws, _new_record(ws) | {
            "aborts": rec["id"], "op": _record_op(rec),
            "reason": "its registry transaction never committed (no decision_marks row); "
            "nothing changed, nothing to redo",
        })  # fmt: skip
    return todo


def _refusal(ws: Workspace, rec: dict[str, Any]) -> DecisionRefused:
    return DecisionRefused(
        f"the {_record_op(rec)} logged at {rec.get('time')} is incomplete ({_why(rec)}); fix "
        f"the cause and run `fukikae studio resume {ws.root}` before applying more decisions"
    )


def _refuse_if_incomplete(ws: Workspace) -> None:
    """With the studio lock held (it may log aborted records)."""
    todo = _settle(ws)
    if todo:
        raise _refusal(ws, todo[0])


def _refuse_if_incomplete_readonly(ws: Workspace) -> None:
    """For dry runs: the same refusal, writing nothing."""
    todo = incomplete_records(ws)
    if todo:
        raise _refusal(ws, todo[0])


def _commit_recorded(
    w: Workspace,
    reg: Registry,
    op: str,
    base: dict[str, Any],
    stage: Callable[[], tuple[list[str], list[Invalidation], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[str], list[Invalidation], list[dict[str, Any]]]:
    """``stage`` (the registry writes, returning lines, invalidations and file steps) in one
    transaction with the decision's mark row and its write-ahead record, appended before
    COMMIT. A log that can't be written rolls everything back."""
    rec = _new_record(w) | base
    with reg.transaction():
        lines, invs, file_steps = stage()
        steps = _stale_steps(invs) + file_steps
        reg.mark_decision(rec["id"], op)
        rec |= {
            "lines": lines, "invalidations": [list(i) for i in invs], "steps": steps,
            "incomplete": {"left": steps, "pending": True},
        }  # fmt: skip
        try:
            _append_log(w, rec)
        except OSError as e:
            raise DecisionRefused(
                f"{op}: the decisions log {decisions_log(w)} could not be written ({e}); "
                "nothing changed"
            ) from e
    return rec, lines, invs, steps


def _after_commit(
    w: Workspace, rec: dict[str, Any], op: str, steps: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Run the steps (stale marks first, so a file change that fails can't leave an episode
    looking current) and log the completion record. An interrupt logs it too (with what is
    left) and propagates. Returns the lines printed, the steps left and their errors."""
    p = _Progress(steps)
    try:
        _run_steps(w, p)
    except BaseException:
        with suppress(OSError):  # the pending record still says every step is left
            _append_log(w, _completion(w, rec, op, p))
        raise
    try:
        _append_log(w, _completion(w, rec, op, p))
    except OSError as e:
        raise IncompleteDecision(
            f"{op} landed in the registry, but its completion record could not be written to "
            f"{decisions_log(w)} ({e}); its write-ahead record stays pending, so fix the cause "
            f"and run `fukikae studio resume {w.root}`, which reruns its steps (each is "
            "idempotent)"
        ) from e
    return [ln for ln in p.lines if not ln.startswith("done: ")], p.left, p.errors


def _completion(w: Workspace, rec: dict[str, Any], op: str, p: _Progress) -> dict[str, Any]:
    out = _new_record(w) | {"completes": rec["id"], "op": op, "lines": p.lines}
    if p.left:
        out["incomplete"] = {"errors": p.errors, "left": p.left}
    return out


def _incomplete(w: Workspace, op: str, left: list[dict[str, Any]], errors: list[str]) -> str:
    stale_ok = not any(s["do"] == "stale" for s in left)
    return (
        f"{op} landed in the registry, but {len(left)} step(s) after it failed: "
        f"{'; '.join(errors)}. "
        + ("Its invalidations are marked stale. " if stale_ok else "")
        + f"Fix the cause and run `fukikae studio resume {w.root}`, which redoes only what is "
        "left; until then `studio apply`, `registry merge` and `fukikae run` refuse to start"
    )


def resume_incomplete(ws: Workspace | Path) -> list[ApplyResult]:
    """Settle the log: an ``aborts`` record for each pending record whose commit never
    landed, and for every record with steps left, run them (all idempotent) and append a
    ``resumes`` record. `IncompleteDecision` when a step still fails (its new record carries
    what is left, so a later resume picks it up)."""
    w = _workspace(ws)
    out: list[ApplyResult] = []
    with _studio_lock(w, "`fukikae studio resume`"):
        for rec in _settle(w):
            p = _Progress(rec["incomplete"]["left"])
            new = _new_record(w) | {"resumes": rec["id"], "op": _record_op(rec)}
            try:
                _run_steps(w, p)
            finally:
                new["lines"] = p.lines
                if p.left:
                    new["incomplete"] = {"errors": p.errors, "left": p.left}
                _append_log(w, new)
            if p.left:
                raise IncompleteDecision(
                    f"the {_record_op(rec)} logged at {rec.get('time')} is still incomplete: "
                    f"{'; '.join(p.errors)}; fix the cause and run `fukikae studio resume "
                    f"{w.root}` again"
                )
            invs = [Invalidation(*i) for i in rec.get("invalidations", [])]
            out.append(ApplyResult(p.lines, invs))
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


def _tables(path: Path) -> set[str]:
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.DatabaseError:
        return set()
    finally:
        con.close()


def _busy(e: sqlite3.OperationalError) -> bool:
    msg = str(e).lower()
    return "locked" in msg or "busy" in msg


def _require_registry(w: Workspace) -> None:
    if not w.registry_path.is_file():
        raise DecisionRefused(
            f"{w.root} has no registry ({w.registry_path}); decisions name registry rows, so "
            "there is nothing to apply them to (`fukikae run` creates it)"
        )


@contextmanager
def _session(w: Workspace, dry_run: bool) -> Iterator[Registry]:
    """The workspace's registry, open. Refused when there is none (a decision names registry
    rows; nothing creates one here), and in a dry run when opening it would write it: no
    schema version (an empty or foreign file), an older version (a migration), or a table
    this version adds missing. A locked registry (another run or studio writing it) is a
    refusal, not a traceback."""
    path = w.registry_path
    _require_registry(w)
    if dry_run:
        version = _schema_version(path)
        if version is None:
            raise DecisionRefused(
                f"registry {path} has no schema version (an empty file, or not a fukikae "
                "registry), and a dry run never creates its tables; check the file, or apply "
                "for real"
            )
        if version < SCHEMA_VERSION:
            raise DecisionRefused(
                f"registry {path} is schema version {version}, and a dry run never migrates "
                f"it to {SCHEMA_VERSION}; apply for real, or run `fukikae registry list "
                f"{w.root}`, which migrates it first"
            )
        if "decision_marks" not in _tables(path):
            raise DecisionRefused(
                f"registry {path} has no decision_marks table yet, and a dry run never "
                f"creates tables; run `fukikae registry list {w.root}`, which adds it, or "
                "apply for real"
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
    ws: Workspace | Path,
    d: Decision,
    *,
    dry_run: bool = False,
    force: bool = False,
    replays: str | None = None,
) -> ApplyResult:
    """Apply one decision: digest check (`StaleDecision`, or `UncheckedDecision` with no
    digest, unless ``force``), the registry writes, the review mark, the decision's mark row
    and its write-ahead log record in one transaction (`DecisionRefused` on a registry
    refusal, nothing changed), then ``job.mark_stale`` for each invalidation, the file changes
    and the completion record (`IncompleteDecision` when one of those fails). ``dry_run`` runs
    the same checks and writes, rolls them back, and returns the preview lines. ``replays``
    is the id of the log record a replay applies again; the new write-ahead record names it,
    so that record counts as applied from then on (`_replay_state`)."""
    w = _workspace(ws)
    if dry_run:
        _refuse_if_incomplete_readonly(w)
        with _session(w, dry_run=True) as reg:
            try:
                with reg.transaction():
                    lines, invs, _ = _stage(reg, w, d, force)
                    raise _DryRun
            except _DryRun:
                return _preview(d, lines, invs, force)
    _require_registry(w)  # before the lock, which would create studio/
    base: dict[str, Any] = {"decision": d.model_dump(mode="json"), "forced": force}
    if replays is not None:
        base["replays"] = replays
    with _studio_lock(w, f"a studio {d.op} decision"):
        _refuse_if_incomplete(w)
        with _session(w, dry_run=False) as reg:
            rec, lines, invs, steps = _commit_recorded(
                w, reg, d.op, base,
                lambda: _stage(reg, w, d, force),
            )  # fmt: skip
        done, left, errors = _after_commit(w, rec, d.op, steps)
    if left:
        raise IncompleteDecision(_incomplete(w, d.op, left, errors))
    return ApplyResult(lines + done, invs, unchecked=force, undigested=d.digest is None)


def apply_registry_change(
    ws: Workspace | Path,
    op: str,
    command: dict[str, Any],
    stage: Callable[[Registry], tuple[list[str], list[Invalidation], list[dict[str, Any]]]],
) -> ApplyResult:
    """A `fukikae registry` change with file steps after it (a merge's remaps), recorded as a
    decision is: write-ahead record and mark row in its transaction, then the steps and a
    completion record, so a failed step is finished by `fukikae studio resume` and blocks
    later decisions until it is. ``stage(reg)`` makes the registry writes (raising
    `DecisionRefused` for a refusal) and returns lines, invalidations and steps."""
    w = _workspace(ws)
    _require_registry(w)  # before the lock, which would create studio/
    with _studio_lock(w, f"`fukikae registry {op}`"):
        _refuse_if_incomplete(w)
        with _session(w, dry_run=False) as reg:
            rec, lines, invs, steps = _commit_recorded(
                w, reg, op, {"op": op, "registry": command}, lambda: stage(reg)
            )
        done, left, errors = _after_commit(w, rec, op, steps)
    if left:
        raise IncompleteDecision(_incomplete(w, op, left, errors))
    return ApplyResult(lines + done, invs)


def _replay_state(ws: Workspace) -> tuple[set[str], set[str]]:
    """What a replay skips, decided by the registry, never by the log alone: ``landed``, the
    ids of logged records whose registry part is in the registry (their own `decision_marks`
    row, or the row of a later record that replays them, however each was closed: a
    completion, a resume, or nothing yet); and ``unmarked``, records the log closes (completes
    or resumes) whose mark the registry lacks, as after a registry restored from a backup.
    Those are applied again, and the result says so."""
    recs = _log_records(ws)
    marks = _marks(ws, "which logged decisions are in the registry")
    landed = {r["id"] for r in recs if r.get("id") in marks}
    replayed = [(r["id"], r["replays"]) for r in recs if r.get("id") and r.get("replays")]
    grew = True
    while grew:
        grew = False
        for rid, orig in replayed:
            if rid in landed and orig not in landed:
                landed.add(orig)
                grew = True
    closed = {r[k] for r in recs for k in ("completes", "resumes") if r.get(k)}
    return landed, closed - landed


def _read_entries(w: Workspace, path: Path) -> list[tuple[Decision, dict[str, Any] | None]]:
    """The decisions in a file, each with the log record it came from (None for a bare
    decision or a `Decisions` document)."""
    text = Path(path).read_text(encoding="utf-8")

    def ours(where: str | None) -> None:
        if where and Path(where).resolve() != w.root.resolve():
            raise DecisionRefused(
                f"{path} is for workspace {where}, not {w.root.resolve()}; registry ids are "
                "workspace-local"
            )

    try:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "decisions" in obj:
            doc = Decisions.model_validate(obj)
            ours(doc.workspace)
            return [(d, None) for d in doc.decisions]
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            obj = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        out: list[tuple[Decision, dict[str, Any] | None]] = []
        for x in obj:
            if isinstance(x, dict) and "decision" in x:
                ours(x.get("workspace"))
                out.append((parse_decision(x["decision"]), x))
            elif isinstance(x, dict) and any(k in x for k in (*_CLOSERS, "registry")):
                continue  # a completion, resume or aborted record, or a registry command's
            else:
                out.append((parse_decision(x), None))
        return out
    except (ValidationError, json.JSONDecodeError) as e:
        raise DecisionRefused(f"{path}: not a decisions file: {e}") from e


def read_decisions(ws: Workspace | Path, path: Path) -> list[Decision]:
    """The decisions in a file: a `Decisions` document, one decision, a list of them, or
    JSON lines of decisions or of ``decisions.jsonl`` records (whose ``decision`` is
    replayed; completion, resume and aborted records and `registry merge` records carry none
    and are skipped). A document or record for another workspace is refused."""
    return [d for d, _ in _read_entries(_workspace(ws), path)]


def _unmarked(i: int, d: Decision, rec: dict[str, Any], r: ApplyResult) -> ApplyResult:
    note = (
        f"decision {i + 1} ({d.op}): logged {rec['id'][:8]} at {rec.get('time')} as applied, but "
        "the registry has no mark of it (a registry restored from a backup?); applying it again"
    )
    return replace(r, lines=[note, *r.lines])


def _already(i: int, d: Decision, rec: dict[str, Any]) -> ApplyResult:
    return ApplyResult(
        [f"decision {i + 1} ({d.op}): already applied (logged {rec['id'][:8]} at "
         f"{rec.get('time')}); skipped"],
        skipped=True,
    )  # fmt: skip


def apply_file(
    ws: Workspace | Path, path: Path, *, dry_run: bool = False, force: bool = False
) -> list[ApplyResult]:
    """Apply a file's decisions in order, each in its own transaction. A log record whose
    decision is in the registry (its `decision_marks` row, `_replay_state`) is skipped and
    says so; one the log calls applied but the registry lacks is applied again, and says so.
    The first refusal stops the file (decisions before it stay applied); its error says how
    many were, and carries their results (``results``). A dry run checks them all in one
    transaction, rolled back at the end, so a decision that depends on an earlier one is
    checked as it will apply."""
    w = _workspace(ws)
    entries = _read_entries(w, path)
    applied, unmarked = _replay_state(w)
    n = len(entries)
    out: list[ApplyResult] = []
    if dry_run:
        _refuse_if_incomplete_readonly(w)
        with _session(w, dry_run=True) as reg:
            try:
                with reg.transaction():
                    for i, (d, rec) in enumerate(entries):
                        if rec is not None and rec.get("id") in applied:
                            out.append(_already(i, d, rec))
                            continue
                        try:
                            lines, invs, _ = _stage(reg, w, d, force)
                        except DecisionError as e:
                            err = type(e)(
                                f"decision {i + 1} ({d.op}): {e}; {i} of {n} decisions checked"
                            )
                            err.results = out
                            raise err from e
                        r = _preview(d, lines, invs, force)
                        if rec is not None and rec.get("id") in unmarked:
                            r = _unmarked(i, d, rec, r)
                        out.append(r)
                    raise _DryRun
            except _DryRun:
                return out
    for i, (d, rec) in enumerate(entries):
        if rec is not None and rec.get("id") in applied:
            out.append(_already(i, d, rec))
            continue
        rid = rec.get("id") if rec is not None else None
        try:
            r = apply_decision(w, d, force=force, replays=rid)
        except DecisionError as e:
            done = sum(not r.skipped for r in out) + isinstance(e, IncompleteDecision)
            err = type(e)(f"decision {i + 1} ({d.op}): {e}; {done} of {n} decisions applied")
            err.results = out
            raise err from e
        out.append(_unmarked(i, d, rec, r) if rec is not None and rid in unmarked else r)
    return out
