"""`fukikae names`: what the subtitle names say and what the pipeline did with them (D27).

``scan`` runs only the detection test on each episode's candidate tracks, read-only: embedded
tracks are extracted once per source version into ``.fukikae/cache/subtitles/<episode>/names/``
(the identity harness's cache), and nothing under the episode's work dir changes. ``show``
reads one episode's ``names.json``, the characters stage's ``names-report.json`` and
``assignments.json``, the registry and the card store, and prints the verdict, coverage, keys
and where each went, cuts, unvoiced lines, voice disagreements, alias and merge suggestions,
stale or flagged clone cards, and characters still holding exemplars from episodes run without
names. Neither command writes anything but the scan cache, with one exception: opening the
registry may migrate its database to the current schema, which is a write.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import typer

if TYPE_CHECKING:
    from fukikae.config import Config
    from fukikae.models import Assignments
    from fukikae.names.detect import TrackChoice, Verdict
    from fukikae.names.models import KeyDecision, NamedLine, NamesReport, NamesResult
    from fukikae.registry.store import Registry
    from fukikae.workspace import Episode, Workspace

names_app = typer.Typer(
    help="Subtitle speaker names: detection, and what each episode's names did.",
    no_args_is_help=True,
)

_WS = typer.Argument(..., help="Workspace directory.")
MERGE_SHARE = 0.5  # the majority of a parent's named speech one character holds for a merge hint


def _echo(text: str = "") -> None:
    typer.echo(text)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def _episodes(ws: Workspace, given: list[str] | None) -> list[str]:
    if given:
        return given
    base = ws.root / ".fukikae" / "episodes"
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def _char(reg: Registry | None, cid: int | None) -> str:
    if cid is None:
        return "nobody"
    name = None
    if reg is not None:
        name = next((c.name for c in reg.characters() if c.id == cid), None)
    return f"character {cid} ({name})" if name else f"character {cid} (unnamed)"


def suggestion_lines(
    ws: Workspace, reg: Registry | None, pairs: list[tuple[str, str]], counts: Counter[str]
) -> list[str]:
    """One line per (rare, common) key pair (`suggest_aliases`, one episode's counts) with the
    `registry alias` command that makes the rare spelling answer to the common key's
    character (``<character>`` when no character answers to it yet)."""
    out = []
    for rare, common in pairs:
        target = "<character>"
        if reg is not None:
            try:
                c = reg.character_for_key(common)
            except ValueError:
                c = None
            if c is not None:
                target = str(c.id)
        out.append(
            f"{rare!r} ({counts[rare]} lines) is one edit from {common!r} ({counts[common]} "
            f"lines): if one name, `fukikae registry alias {ws.root} {target} {rare.upper()}`"
        )
    return out


def _registry(ws: Workspace) -> Registry | None:
    from fukikae.registry.store import Registry

    return Registry(ws.registry_path) if ws.registry_path.exists() else None


# -- scan --------------------------------------------------------------------------------------


def _scan_tracks(ep: Episode, ws: Workspace, choices: list[TrackChoice]) -> list[Path | None]:
    """Each candidate's file: an operator file as is, an embedded ASS track extracted into the
    scan cache (refreshed when the source file's size, mtime, inode or ctime changed, as the
    stage's copies are); None for a track that can't carry names."""
    from fukikae import media
    from fukikae.artifacts import atomic_path
    from fukikae.names import detect

    cache = ws.root / ".fukikae" / "cache" / "subtitles" / ep.id / "names"
    index_path = cache / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    st = ep.source.stat() if ep.source.is_file() else None
    out: list[Path | None] = []
    extract: list[tuple[media.SubTrack, Path]] = []
    for c in choices:
        if c.file is not None:
            out.append(c.file)
            continue
        if c.track is None or detect.no_name_field(c) is not None or st is None:
            out.append(None)
            continue
        dest = cache / f"track-{c.track.index}.ass"
        # the names stage's stamp for its track copies: inode and ctime too, so a source
        # replaced or edited with the same size and mtime is extracted again
        stamp = [
            str(ep.source.resolve()), st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns,
            c.track.index,
        ]  # fmt: skip
        if index.get(dest.name) != stamp or not dest.is_file():
            extract.append((c.track, dest))
            index[dest.name] = stamp
        out.append(dest)
    if extract:
        cache.mkdir(parents=True, exist_ok=True)
        media.extract_subtitles(ep.source.resolve(), extract)
        with atomic_path(index_path) as tmp:
            tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _where(c: TrackChoice) -> str:
    if c.file is not None:
        return f"{c.how} file {c.file.name}"
    assert c.track is not None
    title = f' "{c.track.title}"' if c.track.title else ""
    return f"{c.how} track {c.track.index} ({c.language or '?'}, {c.track.codec}{title})"


def _verdict_line(v: Verdict, lines: list[NamedLine], hidden: set[int]) -> str:
    counts = Counter(ln.key for ln in lines if ln.key is not None and ln.id not in hidden)
    top = counts.most_common(1)
    text = f"{v.named} of {v.dialogue} dialogue lines named ({v.share:.0%}), {v.distinct} keys" + (
        f", top {top[0][0]!r} on {v.top_share:.0%}" if top else ""
    )
    if v.level != "none":
        text += f"; offset {v.offset_s:+.2f} s, {v.align_share:.0%} of named-line time on speech"
    return text


def scan_episode(ws: Workspace, cfg: Config, ref: str, reg: Registry | None) -> list[str]:
    """The detection test on one episode's candidate tracks, as report lines."""
    from fukikae import media
    from fukikae.artifacts import read_json
    from fukikae.models import Diarization
    from fukikae.names import detect
    from fukikae.names.inputs import DIARIZE_DIR
    from fukikae.names.normalize import suggest_aliases
    from fukikae.stages.names import hide_lines

    ep = ws.find_episode(ref)
    s = cfg.names
    if s.use == "off":
        return [f'{ep.id}: [names] use = "off"; nothing read']
    held = ep.id in s.holdout_episodes
    turns_path = ep.work / DIARIZE_DIR / "turns.json"
    turns = read_json(turns_path, Diarization).turns if turns_path.exists() else None
    probe = media.probe(ep.source) if ep.source.is_file() else media.ProbeResult()
    choices = detect.candidate_tracks(ep.source, probe, cfg, episode=ep.id)
    paths = _scan_tracks(ep, ws, choices)
    out: list[str] = []
    tried: list[str] = []
    chosen: tuple[TrackChoice, Verdict, list[NamedLine], set[int]] | None = None
    for c, path in zip(choices, paths, strict=True):
        why = detect.no_name_field(c)
        if why is not None or path is None:
            tried.append(f"{_where(c)}: none ({why or 'no source file'})")
            continue
        hidden: set[int] = set()

        def hide(lines: list[NamedLine], keep: set[int] = hidden) -> list[int]:
            got = hide_lines(lines, s.hide_fraction, s.hide_seed, ep.id)
            keep.update(got)
            return got

        v, lines = detect.assess_path(
            path,
            turns or [],
            min_coverage=s.min_coverage,
            use=s.use,
            align=s.align and turns is not None,
            hide=hide,
        )
        tried.append(f"{_where(c)}: {v.level}" + (f" ({v.reason})" if v.reason else ""))
        if v.level != "none":
            chosen = (c, v, lines, hidden)
            break
    if chosen is None:
        out.append(f"{ep.id}: none" + ("" if choices else " (no subtitle track to read)"))
    else:
        c, v, lines, hidden = chosen
        out.append(f"{ep.id}: {v.level} from {_where(c)}: {_verdict_line(v, lines, hidden)}")
        counts = Counter(ln.key for ln in lines if ln.key is not None and ln.id not in hidden)
        out += [
            f"  suggestion: {x}" for x in suggestion_lines(ws, reg, suggest_aliases(counts), counts)
        ]
    out += [f"  tried {t}" for t in tried]
    if turns is None:
        out.append("  no diarize turns yet, so the alignment test was skipped")
    if held:
        out.append("  held out ([names] holdout_episodes): the names stage reads none of this")
    return out


@names_app.command("scan")
def scan(
    workspace: Path = _WS,
    episodes: list[str] = typer.Argument(
        None, help="Episode file paths or SxxEyy codes (default: every episode with a work dir)."
    ),
) -> None:
    """Run the names detection test on each episode's subtitle tracks, read-only: the track
    chosen, its named share and keys, the verdict, and alias suggestions."""
    from fukikae.errors import user_errors
    from fukikae.workspace import Workspace

    ws = Workspace(workspace)
    try:
        cfg = ws.config
        reg = _registry(ws)
        try:
            for ref in _episodes(ws, episodes):
                for line in scan_episode(ws, cfg, ref, reg):
                    _echo(line)
        finally:
            if reg is not None:
                reg.close()
    except user_errors() as e:
        raise _fail(str(e)) from e


# -- show --------------------------------------------------------------------------------------


def _label_seconds(asg: Assignments) -> dict[str, float]:
    """Seconds of (refined) turns per label."""
    out: dict[str, float] = defaultdict(float)
    for t in asg.turns or []:
        out[t.speaker] += t.end - t.start
    return out


def _fallback_decisions(result: NamesResult, asg: Assignments) -> NamesReport:
    """How each key went, read back from the assignments, for an episode whose characters
    stage predates ``names-report.json``."""
    from fukikae.names.models import KeyDecision, NamesReport

    by_key: dict[str, list[str]] = defaultdict(list)
    for lb, info in result.labels.items():
        if info.key is not None:
            by_key[info.key].append(lb)
    seconds = _label_seconds(asg)
    clusters = {c.cluster: c for c in asg.clusters}
    keys = []
    for k, labels in by_key.items():
        decided = [clusters[lb] for lb in labels if lb in clusters]
        locked = {c.cluster: c.character_id for c in decided if c.source == "lock"}
        rest = [c for c in decided if c.source == "name"]
        how = (
            "lock"
            if not rest and locked
            else "extra"
            if rest and rest[0].status == "extra"
            else "created"
            if rest and rest[0].status == "new"
            else "known"
            if rest
            else "ambiguous"
        )
        keys.append(
            KeyDecision(
                key=k,
                how=how,  # type: ignore[arg-type]
                character_id=rest[0].character_id if rest else None,
                display=next(
                    (i.display for lb in labels if (i := result.labels[lb]).display), None
                ),
                seconds=round(sum(seconds[lb] for lb in labels), 3),
                labels=sorted(labels),
                locked=locked,
            )
        )
    keys.sort(key=lambda d: (-d.seconds, d.key))
    return NamesReport(episode=result.episode, keys=keys, lines=[])


def _stale_decision(
    d: KeyDecision, asg: Assignments, reg: Registry | None, ws: Workspace, episode: str
) -> str | None:
    """Why a `names-report.json` entry no longer describes the episode, or None: its character
    is gone from the registry, or the current assignments give its sub-clusters (or its
    lock-kept ones) to another character, as after a `registry merge` or `split`."""
    clusters = {c.cluster: c for c in asg.clusters}
    why: list[str] = []
    known = {c.id for c in reg.characters()} if reg is not None else None
    wanted = {lb: d.character_id for lb in d.labels if lb not in d.locked}
    wanted.update(d.locked)
    if known is not None:
        gone = sorted({c for c in wanted.values() if c is not None and c not in known})
        why += [f"character {c} is gone from the registry" for c in gone]
    moved: dict[int | None, list[str]] = defaultdict(list)
    for lb, cid in sorted(wanted.items()):
        a = clusters.get(lb)
        if d.how in ("extra", "ambiguous") and lb not in d.locked:
            continue
        now = a.character_id if a is not None else None
        if now != cid:
            moved[now].append(lb)
    for now, labels in sorted(moved.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        who = f"character {now}" if now is not None else "nobody"
        why.append(f"the assignments give {', '.join(labels)} to {who}")
    if not why:
        return None
    return (
        f"stale: {'; '.join(why)}; rerun the characters stage "
        f"(`fukikae run {ws.root} {episode} --from characters`)"
    )


def _disagreements(
    ws: Workspace, episode: str, asg: Assignments, reg: Registry
) -> list[tuple[float, str]]:
    """(cosine, line) for each named sub-cluster with embedding rows that names gave a
    character, against that character's centroid from its exemplars in other episodes,
    lowest first; sub-clusters with no rows, or characters with no other episode, are left
    out."""
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
    out: list[tuple[float, str]] = []
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
        out.append((cos, f"{cos:.3f}  {a.cluster} ({a.name}) -> {_char(reg, a.character_id)}"))
    out.sort()
    return out


def _merge_suggestions(
    ws: Workspace, result: NamesResult, asg: Assignments, reg: Registry
) -> list[str]:
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
    seconds = _label_seconds(asg)
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
    out: list[str] = []
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
        out.append(
            f"{_char(reg, x.id)} kept {parent}'s unnamed remainder ({seconds[lb]:.1f} s) while "
            f"its named speech went to {_char(reg, y.id)} ({held:.1f} of {total:.1f} s); if "
            f"they are one voice, `fukikae registry merge {ws.root} {y.actor_id} {x.actor_id}` "
            "(actor ids)"
        )
    return out


def _card_lines(ws: Workspace, reg: Registry, characters: list[int]) -> list[str]:
    """Stale or flagged clone cards of the episode's characters (`stale_cards`), with what
    replaces each, then automatic clone cards under the clone floor (`floor_lines`, D32)."""
    from fukikae.cards import auto
    from fukikae.cards.auto import AUTOMATIC_BY, repick_blocker, replace_refusal, stale_cards
    from fukikae.cards.store import CardStore

    cfg = ws.config
    language = cfg.target_language
    notes: list[str] = []
    stale = stale_cards(ws, reg, CardStore(ws.root), characters, language, notes=notes)
    out: list[str] = []
    blocker: list[str | None] = []
    for cid in sorted(stale):
        a = reg.card_for(cid, language)
        if a is None:
            continue
        said = sorted({k for r in stale[cid] for k in r.get("names") or []})
        if a.chosen_by in AUTOMATIC_BY and replace_refusal(a, "auto") is None:
            if not blocker:
                blocker.append(repick_blocker(cfg))
            fix = (
                "the next tts run re-picks it"
                if blocker[0] is None
                else f"kept while {blocker[0]}; `fukikae voices clone` replaces it"
            )
        else:
            fix = "`fukikae voices clone` replaces it"
        out.append(
            f"{_char(reg, cid)}'s card {a.card_id} ({a.chosen_by}) was cloned from turns now "
            f"named {', '.join(k.upper() for k in said) or 'otherwise'}; {fix}"
        )
    rest = [cid for cid in characters if cid not in stale]  # a stale card's line says it all
    floor = auto.floor_lines(ws, reg, CardStore(ws.root), rest, language)
    return out + floor + notes


def _unnamed_exemplars(ws: Workspace, reg: Registry, characters: list[int]) -> list[str]:
    """Characters this episode's names placed that still hold exemplars from episodes whose
    characters stage ran without names (their mixed evidence stays until those rerun)."""
    from fukikae.cards.auto import used_names

    chars = {c.id: c for c in reg.characters()}
    out = []
    used: dict[str, bool] = {}
    for cid in characters:
        c = chars.get(cid)
        if c is None:
            continue
        stale = []
        for e in reg.episode_counts(c.actor_id):
            if e not in used:
                used[e] = used_names(ws.work_dir(e)) is not None
            if not used[e]:
                stale.append(e)
        if stale:
            out.append(
                f"{_char(reg, cid)} holds exemplars from {', '.join(stale)}, run without names; "
                f"rerun them from names (`fukikae run {ws.root} {' '.join(stale)} --from names`)"
            )
    return out


def show_episode(ws: Workspace, ref: str) -> list[str]:
    """`names show` for one episode, as lines."""
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments
    from fukikae.names.inputs import names_path, names_report_path
    from fukikae.names.models import NamesReport, NamesResult

    ep = ws.find_episode(ref)
    path = names_path(ep.work)
    if not path.exists():
        raise ValueError(f"{ep.id} has no names.json; run the names stage (`fukikae run`)")
    result = read_json(path, NamesResult)
    src = result.source or {}
    where = (
        f"{src.get('how')} "
        + (
            f"file {Path(str(src['file'])).name}"
            if src.get("file")
            else f"track {src.get('track')}"
        )
        + f" ({src.get('language') or '?'})"
        if src
        else "no track"
    )
    out = [
        f"{ep.id}: names {result.verdict} from {where}"
        + (f"; {result.reason}" if result.reason else "")
        + (f", offset {result.offset_s:+.2f} s" if result.used else "")
    ]
    for t in result.tried:
        if not result.used or t.get("verdict") == "none":
            out.append(
                f"  tried {t.get('how')} {t.get('language') or t.get('file')}: "
                f"{t.get('verdict')}" + (f" ({t['reason']})" if t.get("reason") else "")
            )
    if not result.used:
        return out
    cov = result.coverage
    out.append(
        f"coverage: {cov.get('named')} of {cov.get('dialogue')} dialogue lines named "
        f"({float(cov.get('share', 0)):.0%}), {cov.get('distinct')} keys; "
        f"{cov.get('named_turns')} of {cov.get('turns')} turns named whole, "
        f"{cov.get('cut_turns')} cut, {cov.get('sub_clusters')} sub-clusters; "
        f"{result.dropped_row_seconds:.1f} s of embedding rows dropped at cuts; "
        f"{len(result.unvoiced)} unvoiced; {len(result.hidden)} hidden"
    )
    asg_path = ep.work / "05-characters" / "assignments.json"
    asg = read_json(asg_path, Assignments) if asg_path.exists() else None
    from fukikae.cards.auto import names_mismatch

    why = names_mismatch(ep.work, result, asg)
    if why is not None or asg is None:
        out.append(f"characters: {why}; run the characters stage (`fukikae run`) to place the keys")
        return out
    rpath = names_report_path(ep.work)
    report = read_json(rpath, NamesReport) if rpath.exists() else _fallback_decisions(result, asg)
    reg = _registry(ws)
    try:
        out.append("keys (named seconds; sub-clusters -> character, how):")
        for d in report.keys:
            target = _char(reg, d.character_id) if d.how not in ("extra", "ambiguous") else d.how
            line = f"  {d.key:<16} {d.seconds:7.1f} s  {', '.join(d.labels)} -> {target}, {d.how}"
            if d.locked and d.how != "lock":
                kept = ", ".join(f"{lb} on {_char(reg, c)}" for lb, c in sorted(d.locked.items()))
                line += f"; kept by lock: {kept}"
            stale = _stale_decision(d, asg, reg, ws, ep.id) if rpath.exists() else None
            if stale is not None:
                line += f" [{stale}]"
            out.append(line)
        cuts: dict[int, list] = defaultdict(list)
        for p in result.pieces:
            cuts[p.turn].append(p)
        cut = {t: ps for t, ps in cuts.items() if len(ps) > 1}
        if cut:
            out.append(f"cut turns ({len(cut)}):")
            for t, ps in sorted(cut.items()):
                ps.sort(key=lambda p: p.start)
                parts = " | ".join(f"{p.start:.2f}-{p.end:.2f} {p.key or '-'}" for p in ps)
                out.append(f"  turn {t}: {parts}")
        lines = {ln.id: ln for ln in result.lines}
        if result.unvoiced:
            out.append(f"unvoiced named lines ({len(result.unvoiced)}):")
            for i in result.unvoiced:
                ln = lines[i]
                out.append(f"  {ln.start:.2f}-{ln.end:.2f} {ln.raw}: {ln.text[:60]}")
        if reg is not None:
            dis = _disagreements(ws, ep.id, asg, reg)
            if dis:
                out.append(
                    "voice disagreements (cosine to the other-episode centroid, lowest first):"
                )
                out += [f"  {line}" for _, line in dis]
        if report.lines:
            out.append("characters stage report:")
            out += [f"  {line}" for line in report.lines]
        counts = Counter(ln.key for ln in result.visible_lines() if ln.key is not None)
        sugg = suggestion_lines(ws, reg, result.suggestions, counts)
        if sugg:
            out.append("suggestions:")
            out += [f"  {line}" for line in sugg]
        if reg is not None:
            merges = _merge_suggestions(ws, result, asg, reg)
            if merges:
                out.append("merge suggestions:")
                out += [f"  {line}" for line in merges]
            placed = sorted(
                {d.character_id for d in report.keys if d.character_id is not None}
                | {c for d in report.keys for c in d.locked.values() if c is not None}
            )
            speaking = sorted({a.character_id for a in asg.clusters if a.character_id is not None})
            cards = _card_lines(ws, reg, speaking)
            if cards:
                out.append("clone cards:")
                out += [f"  {line}" for line in cards]
            mixed = _unnamed_exemplars(ws, reg, placed)
            if mixed:
                out.append("exemplars from episodes run without names:")
                out += [f"  {line}" for line in mixed]
    finally:
        if reg is not None:
            reg.close()
    return out


@names_app.command("show")
def show(
    workspace: Path = _WS,
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
) -> None:
    """Show what an episode's subtitle names did: verdict, coverage, each key's character and
    how it got there, cuts, unvoiced lines, voice disagreements, suggestions and cards."""
    from fukikae.errors import user_errors
    from fukikae.workspace import Workspace

    try:
        for line in show_episode(Workspace(workspace), episode):
            _echo(line)
    except (*user_errors(), ValueError) as e:
        raise _fail(str(e)) from e
