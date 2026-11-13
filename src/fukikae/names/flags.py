"""What an episode's subtitle names say is wrong, shared by `fukikae names show` and the season
index (`fukikae.studio.season`, Task 16): stale key decisions, voice disagreements, merge hints
for a nameless leftover, and the flagged clone cards. Each check has one implementation here;
`names show` formats its lines, the index reads the rows behind them.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fukikae.models import Assignments
    from fukikae.names.models import KeyDecision, NamesResult
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

MERGE_SHARE = 0.5  # the majority of a parent's named speech one character holds for a merge hint


def char_label(reg: Registry | None, cid: int | None) -> str:
    if cid is None:
        return "nobody"
    name = None
    if reg is not None:
        name = next((c.name for c in reg.characters() if c.id == cid), None)
    return f"character {cid} ({name})" if name else f"character {cid} (unnamed)"


def label_seconds(asg: Assignments) -> dict[str, float]:
    """Seconds of (refined) turns per label."""
    out: dict[str, float] = defaultdict(float)
    for t in asg.turns or []:
        out[t.speaker] += t.end - t.start
    return out


@dataclass(frozen=True)
class StaleMoves:
    """Why a `names-report.json` entry no longer describes the episode, as data: the
    characters it names that are gone from the registry, and each of its sub-clusters the
    current assignments give to another character (``(label, character now)``, by label;
    None: nobody)."""

    gone: tuple[int, ...]
    moved: tuple[tuple[str, int | None], ...]


def stale_moves(d: KeyDecision, asg: Assignments, reg: Registry | None) -> StaleMoves | None:
    """`StaleMoves` for a key decision, or None while it still describes the episode: its
    character is gone from the registry, or the current assignments give its sub-clusters (or
    its lock-kept ones) to another character, as after a `registry merge` or `split`."""
    clusters = {c.cluster: c for c in asg.clusters}
    known = {c.id for c in reg.characters()} if reg is not None else None
    wanted = {lb: d.character_id for lb in d.labels if lb not in d.locked}
    wanted.update(d.locked)
    gone: list[int] = []
    if known is not None:
        gone = sorted({c for c in wanted.values() if c is not None and c not in known})
    moved: list[tuple[str, int | None]] = []
    for lb, cid in sorted(wanted.items()):
        if d.how in ("extra", "ambiguous") and lb not in d.locked:
            continue
        a = clusters.get(lb)
        now = a.character_id if a is not None else None
        if now != cid:
            moved.append((lb, now))
    if not gone and not moved:
        return None
    return StaleMoves(tuple(gone), tuple(moved))


def stale_line(m: StaleMoves, ws: Workspace, episode: str) -> str:
    """`stale_moves`' finding as the line `names show` and the season index print."""
    why = [f"character {c} is gone from the registry" for c in m.gone]
    by_now: dict[int | None, list[str]] = defaultdict(list)
    for lb, now in m.moved:
        by_now[now].append(lb)
    for now, labels in sorted(by_now.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        who = f"character {now}" if now is not None else "nobody"
        why.append(f"the assignments give {', '.join(labels)} to {who}")
    return (
        f"stale: {'; '.join(why)}; rerun the characters stage "
        f"(`fukikae run {ws.root} {episode} --from characters`)"
    )


def stale_decision(
    d: KeyDecision, asg: Assignments, reg: Registry | None, ws: Workspace, episode: str
) -> str | None:
    """Why a `names-report.json` entry no longer describes the episode (`stale_moves`, as
    `stale_line` words it), or None."""
    m = stale_moves(d, asg, reg)
    return None if m is None else stale_line(m, ws, episode)


def names_flag_identities(
    ws: Workspace, episode: str, reg: Registry, asg: Assignments | None = None
) -> dict[int, list[list[Any]]]:
    """Character -> what each of its ``names`` flags on the episode is, with no text in it
    (names, seconds and wording change on a rerun; these don't): a stale key decision is
    ``["stale", key, [[label, character now], ...], [gone character, ...]]`` for each
    character holding the key (as the season index assigns it), a disagreement
    ``["disagree", key, character, [[other character, how, labels], ...]]``. A report from
    before structured disagreements gives its "acoustics disagree" lines as
    ``["disagree-line", line]``. Empty unless the episode used names and has its names report
    and assignments; ``asg`` stands in for the assignments on disk (a merge's remap, before it
    is written)."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import used_names
    from fukikae.models import Assignments
    from fukikae.names.inputs import names_report_path
    from fukikae.names.models import NamesReport

    work = ws.work_dir(episode)
    rpath = names_report_path(work)
    apath = work / "05-characters" / "assignments.json"
    if used_names(work) is None or not rpath.exists():
        return {}
    if asg is None:
        if not apath.exists():
            return {}
        asg = read_json(apath, Assignments)
    report = read_json(rpath, NamesReport)
    out: dict[int, list[list[Any]]] = defaultdict(list)
    for d in report.keys:
        m = stale_moves(d, asg, reg)
        if m is None:
            continue
        holders = {d.character_id} if d.how not in ("extra", "ambiguous") else set()
        holders |= set(d.locked.values())
        item: list[Any] = ["stale", d.key, [list(p) for p in m.moved], list(m.gone)]
        for cid in sorted(h for h in holders if h is not None):
            out[cid].append(item)
    if report.disagreements is None:
        for line in report.lines:
            if "acoustics disagree" not in line:
                continue
            for d in report.keys:
                if d.character_id is not None and line.startswith(f"{d.key!r} -> "):
                    out[d.character_id].append(["disagree-line", line])
    else:
        for x in report.disagreements:
            against = sorted([w.character_id, w.how, sorted(w.labels)] for w in x.against)
            out[x.character_id].append(["disagree", x.key, x.character_id, against])
    return dict(out)


def flags_digest(items: list[list[Any]]) -> str:
    """sha256 of a character's flag identities (`names_flag_identities`), in any order."""
    canon = sorted(json.dumps(i, sort_keys=True) for i in items)
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def names_flag_digest(
    ws: Workspace, episode: str, reg: Registry, character_id: int, asg: Assignments | None = None
) -> str | None:
    """The digest of the character's names flags on the episode (`flags_digest`), or None
    when it has none there: what `studio apply` records with a review and the identity gate
    compares (`fukikae.studio.gate`)."""
    items = names_flag_identities(ws, episode, reg, asg).get(character_id)
    return flags_digest(items) if items else None


@dataclass(frozen=True)
class Disagreement:
    """A named sub-cluster's cosine to its character's centroid from other episodes."""

    cosine: float
    cluster: str
    name: str | None
    character_id: int


def disagreement_rows(
    ws: Workspace, episode: str, asg: Assignments, reg: Registry
) -> list[Disagreement]:
    """Each named sub-cluster with embedding rows that names gave a character, against that
    character's centroid from its exemplars in other episodes, lowest cosine first;
    sub-clusters with no rows, or characters with no other episode, are left out."""
    from fukikae.names.inputs import identity_inputs
    from fukikae.registry.matching import load_clusters

    _, clusters_path, _ = identity_inputs(ws.work_dir(episode))
    if not clusters_path.exists():
        return []
    _, rows = load_clusters(clusters_path)
    chars = {c.id: c for c in reg.characters()}
    others: dict[int, list[np.ndarray]] = defaultdict(list)
    for ex in reg.exemplars():
        if ex.episode != episode:
            others[ex.actor_id].append(ex.embedding)
    out: list[Disagreement] = []
    for a in asg.clusters:
        if a.source != "name" or a.character_id is None or a.character_id not in chars:
            continue
        r = rows.get(a.cluster)
        theirs = others.get(chars[a.character_id].actor_id)
        if r is None or not len(r) or not theirs:
            continue
        mine = r.mean(axis=0)
        ref = np.vstack(theirs).mean(axis=0)
        denom = float(np.linalg.norm(mine) * np.linalg.norm(ref))
        if not denom:
            continue
        cos = float(np.dot(mine, ref)) / denom
        out.append(Disagreement(cos, a.cluster, a.name, a.character_id))
    out.sort(key=lambda d: (d.cosine, d.cluster))
    return out


def disagreement_line(reg: Registry | None, d: Disagreement) -> str:
    return f"{d.cosine:.3f}  {d.cluster} ({d.name}) -> {char_label(reg, d.character_id)}"


def disagreements(
    ws: Workspace, episode: str, asg: Assignments, reg: Registry
) -> list[tuple[float, str]]:
    """(cosine, line) for each `disagreement_rows` row, lowest first."""
    rows = disagreement_rows(ws, episode, asg, reg)
    out = [(d.cosine, disagreement_line(reg, d)) for d in rows]
    out.sort()
    return out


@dataclass(frozen=True)
class MergeHint:
    """A nameless character (``drop``) that kept ``parent``'s unnamed remainder while most of
    its named speech went to the named character ``keep``."""

    keep: int
    drop: int
    keep_actor: int
    drop_actor: int
    parent: str
    remainder_seconds: float
    held: float
    total: float


def merge_hints(
    ws: Workspace, result: NamesResult, asg: Assignments, reg: Registry
) -> list[MergeHint]:
    """A nameless character that kept a parent cluster's unnamed remainder (at least
    ``[names] min_new_seconds`` of it) while that parent's named speech went, mostly
    (`MERGE_SHARE`), to one named character: the two may be one voice, and merging is the
    operator's call (Task 6 ruling). The command is `registry merge KEEP DROP` with the named
    character's actor as KEEP, which folds the leftover into KEEP's lowest-id character, so
    the hint is given only when that is the named character; never for two characters of one
    actor, which no merge joins."""
    floor = ws.config.names.min_new_seconds
    chars = {c.id: c for c in reg.characters()}
    first: dict[int, int] = {}  # actor -> its lowest-id character, the one a merge keeps
    for c in chars.values():
        first[c.actor_id] = min(first.get(c.actor_id, c.id), c.id)
    clusters = {c.cluster: c for c in asg.clusters}
    seconds = label_seconds(asg)
    named_by_parent: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    remainder: dict[str, str] = {}
    for lb, info in result.labels.items():
        a = clusters.get(lb)
        if a is None or a.character_id is None:
            continue
        if info.key is None:
            remainder[info.parent] = lb
        elif a.source == "name":
            named_by_parent[info.parent][a.character_id] += seconds[lb]
    out: list[MergeHint] = []
    seen: set[tuple[int, int]] = set()
    for parent, lb in sorted(remainder.items()):
        x = chars.get(clusters[lb].character_id or -1)
        got = named_by_parent.get(parent)
        if x is None or x.name is not None or not got or seconds[lb] < floor:
            continue
        cid, held = max(got.items(), key=lambda kv: (kv[1], -kv[0]))
        total = sum(got.values())
        y = chars.get(cid)
        if y is None or y.name is None or y.id == x.id or held < MERGE_SHARE * total:
            continue
        if y.actor_id == x.actor_id or first[y.actor_id] != y.id:
            continue
        if (y.id, x.id) in seen:
            continue
        seen.add((y.id, x.id))
        out.append(MergeHint(y.id, x.id, y.actor_id, x.actor_id, parent, seconds[lb], held, total))
    return out


def merge_suggestions(
    ws: Workspace, result: NamesResult, asg: Assignments, reg: Registry
) -> list[str]:
    """One line per `merge_hints` hint, with the `registry merge` command."""
    return [
        f"{char_label(reg, h.drop)} kept {h.parent}'s unnamed remainder "
        f"({h.remainder_seconds:.1f} s) while its named speech went to "
        f"{char_label(reg, h.keep)} ({h.held:.1f} of {h.total:.1f} s); if they are one voice, "
        f"`fukikae registry merge {ws.root} {h.keep_actor} {h.drop_actor}` (actor ids)"
        for h in merge_hints(ws, result, asg, reg)
    ]


def card_lines(ws: Workspace, reg: Registry, characters: list[int]) -> list[str]:
    """The flagged cards of the episode's characters, one line per card with every reason
    (``cards.auto.card_status_lines``: stale references, under the clone floor or thin, impure
    references), then the episodes stale detection skipped."""
    from fukikae.cards import auto
    from fukikae.cards.store import CardStore

    notes: list[str] = []
    store = CardStore(ws.root)
    lines = auto.card_status_lines(
        ws, reg, store, characters, ws.config.target_language, notes=notes
    )
    return lines + notes
