"""`fukikae registry` commands."""

from pathlib import Path
from typing import Any

import typer

from fukikae.names.inputs import identity_inputs
from fukikae.workspace import Episode, Workspace

registry_app = typer.Typer(help="Manage the series character registry.", no_args_is_help=True)


@registry_app.command("rematch")
def rematch(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
) -> None:
    """Forget an episode's registry contributions and match its clusters again.

    Actors only this episode created are deleted first (unless curated), so its clusters
    are judged against the rest of the series, never against their own earlier result.
    """
    from fukikae import job
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments
    from fukikae.stages.characters import CharactersStage

    ws = Workspace(workspace)
    try:
        ep = ws.find_episode(episode)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e

    stage = CharactersStage(rematch=True)
    records = job.load_manifest(ep).stages
    missing = [
        n
        for n in ("extract", "separate", "language", "asr", "diarize")
        if n not in records or records[n].status != "done"
    ]
    if missing or any(not p.exists() for p in identity_inputs(ep.work)[:2]):
        typer.echo(
            f"error: {ep.id}: earlier stages not done ({', '.join(missing) or 'diarize outputs'}); "
            f"run `fukikae run` first",
            err=True,
        )
        raise typer.Exit(1)

    (assignments_path,) = stage.outputs(ep)
    old: dict[str, int | None] = {}
    if assignments_path.exists():
        old = {c.cluster: c.character_id for c in read_json(assignments_path, Assignments).clusters}
    # The stage rebuilds the episode's exemplars in one transaction and replaces
    # assignments.json atomically afterwards, so an interrupted rematch changes nothing.
    job.run_stages(ep, ws.config, [stage], from_stage="characters")

    changed = 0
    for c in read_json(assignments_path, Assignments).clusters:
        before = old.get(c.cluster)
        if before != c.character_id:
            changed += 1
            typer.echo(f"{c.cluster}: {before} -> {c.character_id}")
    if not changed:
        typer.echo("no cluster changed character")


def _open(workspace: Path):  # noqa: ANN202
    from fukikae.registry.store import Registry

    return Registry(Workspace(workspace).registry_path)


def _episode(ws: Workspace, ref: str) -> Episode:
    try:
        return ws.find_episode(ref)
    except FileNotFoundError:
        if ws.work_dir(ref).is_dir():
            return Episode(source=Path(ref), id=ref, work=ws.work_dir(ref), workspace_root=ws.root)
        raise


def _fail(msg: str) -> typer.Exit:
    typer.echo(f"error: {msg}", err=True)
    return typer.Exit(1)


def _same_dims(dims: dict[str, int], hint: str) -> None:
    """Refuse episodes whose clusters.npz rows have different dimensions (the embedder
    changed between their diarize runs); `dims` maps episode -> dimension, 0 for none."""
    found = {e: d for e, d in dims.items() if d}
    if len(set(found.values())) > 1:
        raise _fail(
            "the episodes' clusters come from different embedders ("
            + ", ".join(f"{e}: {d}-dimensional" for e, d in found.items())
            + f"). {hint}"
        )


def _stale(ws: Workspace, episodes: list[str]) -> None:
    """Delete each episode's `characters` manifest record and say so."""
    from fukikae.job import mark_stale

    if not episodes:
        typer.echo("no episode has exemplars under it; nothing is stale")
        return
    for eid in episodes:
        ep = Episode(source=Path(eid), id=eid, work=ws.work_dir(eid), workspace_root=ws.root)
        mark_stale(ep, "characters")
    typer.echo(f"stale (characters will rerun on the next `fukikae run`): {', '.join(episodes)}")


def _remap_assignments(
    ws: Workspace,
    episode: str,
    actor_map: dict[int, int],
    only_cluster: str | None = None,
    character: tuple[int, int] | None = None,
    character_map: dict[int, int] | None = None,
) -> None:
    """Keep an episode's assignments.json in step with a merge or split.

    Pinning copies actor and character ids from it, so stale ids would survive a rerun.
    """
    from fukikae.artifacts import read_json, write_json
    from fukikae.models import Assignments

    path = ws.work_dir(episode) / "05-characters" / "assignments.json"
    if not path.exists():
        return
    a = read_json(path, Assignments)
    out = []
    for c in a.clusters:
        if only_cluster is not None and c.cluster != only_cluster:
            out.append(c)
            continue
        upd: dict[str, int | None] = {}
        if c.actor_id in actor_map:
            upd["actor_id"] = actor_map[c.actor_id]
        if character is not None and c.cluster == only_cluster:
            upd["character_id"] = character[1]
        if character_map and c.character_id in character_map:
            upd["character_id"] = character_map[c.character_id]
        out.append(c.model_copy(update=upd))
    write_json(path, a.model_copy(update={"clusters": out}))


@registry_app.command("cluster")
def cluster(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str | None = typer.Option(
        None, "--episodes", help="Comma-separated SxxEyy codes; default every worked episode."
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="Cosine distance to join clusters; default 1 - match_threshold."
    ),
) -> None:
    """Seed an empty registry from a whole season at once."""
    import numpy as np

    from fukikae.artifacts import read_json, write_json
    from fukikae.job import load_manifest
    from fukikae.models import Assignments, ClusterAssignment, Diarization
    from fukikae.registry.cluster import seed_clusters
    from fukikae.registry.matching import ClusterEvidence, load_clusters, mean_embedding
    from fukikae.stages.characters import CharactersStage

    ws = Workspace(workspace)
    cfg = ws.config
    if episodes:
        refs = [e.strip() for e in episodes.split(",") if e.strip()]
    else:
        root = ws.work_dir("x").parent
        refs = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    if not refs:
        raise _fail("no episodes to cluster")
    try:
        eps = [_episode(ws, r) for r in refs]
    except (FileNotFoundError, ValueError) as e:
        raise _fail(str(e)) from e

    stage = CharactersStage()
    not_done = []
    for ep in eps:
        rec = load_manifest(ep).stages.get("diarize")
        if (
            rec is None
            or rec.status != "done"
            or any(not p.exists() for p in identity_inputs(ep.work)[:2])
        ):
            not_done.append(ep.id)
    if not_done:
        raise _fail(f"diarize stage not done for {', '.join(not_done)}; run `fukikae run` first")

    with _open(workspace) as reg, reg.transaction():  # all of it, or none of it
        if reg.actors():
            raise _fail(
                "the registry already has actors; batch seeding needs an empty one. "
                "Use per-episode matching (`fukikae run`, `registry rematch`)."
            )
        loaded = {ep.id: load_clusters(identity_inputs(ep.work)[1]) for ep in eps}
        _same_dims(
            {
                e: next((r.shape[1] for r in rows.values() if len(r)), 0)
                for e, (_, rows) in loaded.items()
            },
            "Re-run the older episodes' diarize stage with the configured embedder "
            "(`fukikae run`), then cluster",
        )
        groups = seed_clusters(
            {eid: ev for eid, (ev, _) in loaded.items()},
            threshold if threshold is not None else 1.0 - cfg.registry.match_threshold,
            cfg.registry.min_new_seconds,
        )
        member: dict[tuple[str, str], tuple[int, int]] = {}
        review: dict[tuple[str, str], int] = {}
        for g in groups:
            gev: dict[tuple[str, str], ClusterEvidence] = {
                (e, c): next(x for x in loaded[e][0] if x.cluster == c) for e, c in g
            }
            # One character per episode: of a group's clusters in one episode, the one with
            # the most clean speech is matched; the others go to review and add nothing.
            keep: dict[str, str] = {}
            for m in sorted(g, key=lambda m: (-gev[m].clean_seconds, m)):
                keep.setdefault(m[0], m[1])
            kept = sorted((e, c) for e, c in keep.items())
            centroid = mean_embedding(np.concatenate([loaded[e][1][c] for e, c in kept]))
            assert centroid is not None
            aid, cid = reg.create_actor_with_character(centroid)
            for e, c in g:
                if keep[e] != c:
                    review[(e, c)] = aid
                    continue
                reg.add_exemplars(aid, e, c, loaded[e][1][c], gev[(e, c)].clean_seconds)
                member[(e, c)] = (aid, cid)
            reg.recompute_centroid(aid)

    for ep in eps:
        turns_path = identity_inputs(ep.work)[0]
        turns = read_json(turns_path, Diarization)
        evs, _ = loaded[ep.id]
        out = []
        for ev in evs:
            hit = member.get((ep.id, ev.cluster))
            rev = review.get((ep.id, ev.cluster))
            out.append(
                ClusterAssignment(
                    cluster=ev.cluster,
                    actor_id=hit[0] if hit else rev,
                    character_id=hit[1] if hit else None,
                    score=None,
                    clean_seconds=ev.clean_seconds,
                    status="matched" if hit else "review" if rev else "extra",
                )
            )
        (path,) = stage.outputs(ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            Assignments(
                episode=ep.id,
                threshold=cfg.registry.match_threshold,
                clusters=out,
                turns=turns.turns,
            ),
        )
    typer.echo(f"{len(groups)} characters from {len(eps)} episodes")


@registry_app.command("list")
def list_(workspace: Path = typer.Argument(..., help="Workspace directory.")) -> None:
    """List actors with their characters, exemplar counts and lock state."""
    with _open(workspace) as reg:
        chars = reg.characters()
        typer.echo("actor\tname\tcharacters\texemplars\tlocked")
        for a in reg.actors():
            cs = ", ".join(f"{c.id}:{c.name or '-'}" for c in chars if c.actor_id == a.id)
            typer.echo(
                f"{a.id}\t{a.name or '-'}\t{cs}\t{reg.exemplar_count(a.id)}\t"
                f"{'yes' if a.locked else 'no'}"
            )


@registry_app.command("show")
def show(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    actor_id: int = typer.Argument(..., metavar="ID", help="Actor id."),
) -> None:
    """Show one actor's characters and per-episode exemplar counts."""
    with _open(workspace) as reg:
        actor = next((a for a in reg.actors() if a.id == actor_id), None)
        if actor is None:
            raise _fail(f"no actor {actor_id}")
        typer.echo(
            f"actor {actor.id}: {actor.name or '-'} (external id {actor.external_id or '-'}, "
            f"{'locked' if actor.locked else 'unlocked'})"
        )
        for c in reg.characters():
            if c.actor_id == actor_id:
                typer.echo(
                    f"  character {c.id}: {c.name or '-'}"
                    + (f" (name by {c.name_by})" if c.name and c.name_by else "")
                    + (f" (external id {c.external_id})" if c.external_id else "")
                )
                aliases = reg.aliases(c.id)
                if aliases:
                    typer.echo("    aliases: " + ", ".join(f"{a.alias} ({a.by})" for a in aliases))
        for eid, n in reg.episode_counts(actor_id).items():
            typer.echo(f"  {eid}: {n} exemplars")


@registry_app.command("rename")
def rename(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    character_id: int = typer.Argument(..., metavar="CHARACTER", help="Character id."),
    name: str = typer.Argument(..., metavar="NAME", help="New name."),
    external_id: str | None = typer.Option(
        None,
        "--external-id",
        help="The character's id in an outside source (an AniDB character id); '' clears it.",
    ),
) -> None:
    """Name a character (an operator name, which subtitles never replace) and make it answer to
    the name's key; its episodes' characters stage reruns next time, and so does every episode
    whose names stage used that key."""
    from fukikae.registry.store import alias_key

    ws = Workspace(workspace)
    with _open(workspace) as reg:
        char = next((c for c in reg.characters() if c.id == character_id), None)
        if char is None:
            raise _fail(f"no character {character_id}")
        with reg.transaction():
            warning = reg.rename_character(character_id, name)
            if external_id is not None:
                reg.set_character_external_id(character_id, external_id or None)
        affected = list(reg.episode_counts(char.actor_id))
    if warning:
        typer.echo(f"warning: {warning}")
    key = alias_key(name)
    users = _episodes_with_key(ws, key) if key is not None else []
    _stale(ws, sorted(set(affected) | set(users)))


def _episodes_with_key(ws: Workspace, key: str) -> list[str]:
    """Episodes whose names stage used a line of this key."""
    from fukikae.artifacts import read_json
    from fukikae.names.inputs import names_path
    from fukikae.names.models import NamesResult

    out = []
    episodes = ws.root / ".fukikae" / "episodes"
    for work in sorted(episodes.iterdir()) if episodes.is_dir() else []:
        path = names_path(work)
        if not path.is_file():
            continue
        try:
            r = read_json(path, NamesResult)
        except ValueError:
            continue
        if r.used and any(ln.key == key for ln in r.visible_lines()):
            out.append(work.name)
    return out


@registry_app.command("alias")
def alias(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    character: str | None = typer.Argument(
        None, metavar="[CHARACTER]", help="Character id or name; omit to list every alias."
    ),
    name: str | None = typer.Argument(
        None,
        metavar="[NAME]",
        help="A subtitle Name as written (NATSU, MAKAROV (WRAITH)); omit to list the "
        "character's aliases.",
    ),
    remove: bool = typer.Option(False, "--remove", help="Remove the alias instead."),
    replace: bool = typer.Option(
        False, "--replace", help="Move the key from another character's operator alias."
    ),
) -> None:
    """Make a character answer to a subtitle name (an operator alias), remove one, or list them.

    NAME is keyed as the names stage keys a subtitle Name; a name that keys nobody (two
    speakers such as NATSU/ZEREF, a group word, a sign) is refused. An operator alias moves a
    subtitles alias of the same key and another operator's only with --replace; it may never
    be another character's name."""
    from fukikae.registry.store import AliasConflictError, alias_key

    ws = Workspace(workspace)
    with _open(workspace) as reg:
        target = None
        if character is not None:
            try:
                target = reg.character_by_ref(character)
            except ValueError as e:
                raise _fail(str(e)) from e
        if name is None:
            if remove or replace:
                raise _fail("--remove and --replace need a NAME")
            typer.echo("alias	character	by	language")
            for a in reg.aliases(None if target is None else target.id):
                typer.echo(f"{a.alias}	{a.character_id}	{a.by}	{a.language or '-'}")
            return
        assert target is not None
        key = alias_key(name)
        if key is None:
            raise _fail(
                f"{name!r} names nobody (two speakers, a group word, a sign or a song, or blank); "
                "only a single speaker's name can be an alias"
            )
        try:
            before = reg.character_for_key(key) if not remove else None
        except ValueError as e:  # two characters' names share the key
            raise _fail(str(e)) from e
        if remove:
            held = next((a for a in reg.aliases() if a.alias == key), None)
            if held is None:
                raise _fail(f"no alias {key!r}")
            if held.character_id != target.id:
                raise _fail(f"{key!r} is character {held.character_id}'s alias, not {target.id}'s")
            reg.remove_alias(key)
            typer.echo(f"character {target.id}: alias {key!r} removed")
        else:
            try:
                got = reg.add_alias(target.id, key, by="operator", replace=replace)
            except AliasConflictError as e:
                raise _fail(str(e)) from e
            typer.echo(f"character {target.id}: answers to {got.alias!r} (operator)")
        actors = {target.actor_id} | ({before.actor_id} if before is not None else set())
        affected = {e for aid in actors for e in reg.episode_counts(aid)}
    _stale(ws, sorted(affected | set(_episodes_with_key(ws, key))))


@registry_app.command("merge")
def merge(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    keep: int = typer.Argument(..., metavar="KEEP", help="Actor id to keep."),
    drop: int = typer.Argument(..., metavar="DROP", help="Actor id folded into KEEP."),
) -> None:
    """Merge actor DROP into actor KEEP; DROP's characters fold into KEEP's first one."""
    ws = Workspace(workspace)
    with _open(workspace) as reg:
        notes: list[str] = []
        try:
            affected = list(reg.episode_counts(drop))
            chars = reg.merge_actors(keep, drop, report=notes)
        except (KeyError, ValueError) as e:
            raise _fail(f"cannot merge {drop} into {keep}: {e!r}") from e
    for line in notes:
        typer.echo(line)
    for eid in affected:
        _remap_assignments(ws, eid, {drop: keep}, character_map=chars)
    _stale(ws, affected)


@registry_app.command("split")
def split(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Option(..., "--episode", help="Episode code (SxxEyy)."),
    cluster: str = typer.Option(..., "--cluster", help="Cluster label in that episode."),
) -> None:
    """Move one episode cluster to a new actor and character."""
    ws = Workspace(workspace)
    with _open(workspace) as reg:
        holders = reg.cluster_actors(episode, cluster)
        if len(holders) != 1:
            raise _fail(f"no exemplars for {episode} {cluster}")
        old = holders[0]
        new_aid, new_cid = reg.split_cluster(old, episode, cluster)
    _remap_assignments(
        ws, episode, {old: new_aid}, only_cluster=cluster, character=(new_aid, new_cid)
    )
    typer.echo(f"{episode} {cluster}: actor {old} -> {new_aid}, character {new_cid}")
    _stale(ws, [episode])


@registry_app.command("lock")
def lock(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    actor_id: int = typer.Argument(
        ..., metavar="ID", help="Actor id (character id with --character)."
    ),
    unlock: bool = typer.Option(False, "--unlock", help="Unlock instead."),
    character: bool = typer.Option(
        False, "--character", help="ID is a character: lock it against subtitle names."
    ),
) -> None:
    """Lock an actor's centroid (or unlock it); with --character, lock a character, which keeps
    the speech its clusters hold whatever subtitle names say, and is never named by them."""
    what = "character" if character else "actor"
    with _open(workspace) as reg:
        try:
            if character:
                reg.lock_character(actor_id, not unlock)
            else:
                reg.lock_actor(actor_id, not unlock)
        except KeyError as e:
            raise _fail(f"no {what} {actor_id}") from e
    typer.echo(f"{what} {actor_id}: {'unlocked' if unlock else 'locked'}")


@registry_app.command("actor")
def actor(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    actor_id: int = typer.Argument(..., metavar="ID", help="Actor id."),
    name: str | None = typer.Option(None, "--name", help="Actor's name."),
    external_id: str | None = typer.Option(None, "--external-id", help="Id in an outside source."),
) -> None:
    """Set an actor's name or external id."""
    with _open(workspace) as reg:
        try:
            reg.set_actor(actor_id, name=name, external_id=external_id)
        except KeyError as e:
            raise _fail(f"no actor {actor_id}") from e


def _worked_episodes(ws: Workspace, episodes: str) -> list[Episode]:
    """The listed episodes (SxxEyy codes or media paths), each with a done diarize stage in
    this workspace; anything else is refused as not an episode of this workspace."""
    from fukikae.job import load_manifest

    refs = [e.strip() for e in episodes.split(",") if e.strip()]
    if not refs:
        raise _fail("--episodes names no episode")
    eps: list[Episode] = []
    for ref in refs:
        try:
            ep = _episode(ws, ref)
        except (FileNotFoundError, ValueError) as e:
            raise _fail(f"{ref}: not an episode of this workspace ({e})") from e
        rec = load_manifest(ep).stages.get("diarize")
        if (
            rec is None
            or rec.status != "done"
            or any(not p.exists() for p in identity_inputs(ep.work)[:2])
        ):
            raise _fail(
                f"{ref}: not an episode of this workspace with a done diarize stage "
                f"(nothing under {ep.work / '04-diarize'}); run `fukikae run` first"
            )
        eps.append(ep)
    if len({e.id for e in eps}) != len(eps):
        raise _fail("--episodes names an episode twice")
    return eps


@registry_app.command("adapt")
def adapt(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(
        ..., "--episodes", help="Comma-separated labeled episodes (SxxEyy codes or files)."
    ),
    gt: str = typer.Option(
        "ass", "--gt", help="Ground truth: ass (Name field), sdh (speaker tags) or colour."
    ),
    gt_file: Path = typer.Option(
        None, "--gt-file", help="Sidecar subtitle file, or a directory matched by SxxEyy."
    ),
    subtitle_track: int = typer.Option(
        None, "--subtitle-track", help="Absolute stream index of the ground-truth track."
    ),
    alias: list[str] = typer.Option(
        None, "--alias", help="FROM=TO speaker-name alias; repeat. Also labels-aliases.yaml."
    ),
    dims: int = typer.Option(None, "--dims", min=1, help="Components to keep (default all)."),
    force: bool = typer.Option(
        False, "--force", help="Fit even when the subtitles do not line up with the speech."
    ),
) -> None:
    """Fit the series' LDA projection from labeled episodes, replacing the previous one.

    Each episode's identity clusters (the names stage's sub-clusters when names were used,
    else the diarizer's) have their clean spans labeled from its subtitles (as `eval identity`
    labels them), and the labeled rows of the matching clusters.npz, the embeddings the
    characters stage matches, train the projection. A row a names cut moved keeps its old span,
    which is no longer a clean span of the refined turns, so it trains nothing.
    `[characters] matcher = "lda"` then matches in it.
    Leave the held-out episode(s) out: a projection must never be fit on an episode it
    later resolves for evaluation.
    """
    import numpy as np

    from fukikae.artifacts import read_json
    from fukikae.errors import user_errors
    from fukikae.eval.autolabel import speech_turns
    from fukikae.eval.groundtruth import KINDS, check_alignment
    from fukikae.eval.groundtruth.base import load_alias_file
    from fukikae.eval.identity.cli import _ground_truth, parse_aliases
    from fukikae.eval.identity.segments import diarization_segments
    from fukikae.models import Diarization
    from fukikae.registry.projection import KIND_LDA, fit_lda, series_of

    if gt not in KINDS:
        raise _fail(f"unknown --gt {gt!r}; use one of {', '.join(KINDS)}")
    ws = Workspace(workspace)
    eps = _worked_episodes(ws, episodes)
    refs = [e.strip() for e in episodes.split(",") if e.strip()]
    x: list[np.ndarray] = []
    y: list[str] = []
    emb_dims: dict[str, int] = {}
    try:
        aliases = {**load_alias_file(ws.root / "labels-aliases.yaml"), **parse_aliases(alias)}
        for ref, ep in zip(refs, eps, strict=True):
            path, lines = _ground_truth(
                ws, ref, ep.id, gt, gt_file, len(eps), subtitle_track, aliases
            )
            lines, _, _ = check_alignment(
                lines,
                speech_turns(ep.work),
                force=force,
                label=f"{gt} speaker lines in {ep.id} ({path.name})",
            )
            turns_path, npz_path, _ = identity_inputs(ep.work)  # names' sub-clusters when used
            turns = read_json(turns_path, Diarization).turns
            label_of = {
                (s.cluster, round(s.start, 6), round(s.end, 6)): s.label
                for s in diarization_segments(ep.id, turns, lines)
            }
            with np.load(npz_path) as z:
                for cluster in [str(c) for c in z["labels"]]:
                    emb = np.asarray(z[f"{cluster}/embeddings"], dtype=np.float32)
                    if len(emb):
                        emb_dims.setdefault(ep.id, emb.shape[1])
                    spans = np.asarray(z[f"{cluster}/spans"]).reshape(-1, 2)
                    for row, (a, b) in zip(emb, spans, strict=True):
                        lab = label_of.get((cluster, round(float(a), 6), round(float(b), 6)))
                        if lab is not None:
                            x.append(row)
                            y.append(lab)
    except user_errors() as e:
        raise _fail(str(e)) from e
    _same_dims(
        emb_dims,
        "If the registry was built with the old embedder, run `fukikae registry reembed "
        "<workspace>` first; then re-run the older episodes' diarize stage (`fukikae run`) "
        "and adapt",
    )
    projection = fit_lda(np.vstack(x), y, dims) if x else None
    if projection is None:
        raise _fail(
            f"too few labels to fit on: {len(x)} labeled rows; LDA needs two speaker labels "
            "with at least two labeled rows each"
        )
    series = series_of(ws.config)
    with _open(workspace) as reg:
        reg.save_projection(series, KIND_LDA, projection.matrix, projection.mean)
    typer.echo(
        f"lda projection for series {series!r}: {len(x)} labeled rows, {len(set(y))} labels, "
        f"{projection.matrix.shape[1]} dims, from {', '.join(e.id for e in eps)}"
    )


def _dims(c: Any) -> str:
    return ", ".join(f"{n} x {d}-d" for d, n in sorted(c.items())) or "none"


@registry_app.command("reembed")
def reembed(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Check and report counts per dimension; change nothing."
    ),
) -> None:
    """Re-embed every exemplar with the configured embedder after an embedder change.

    Each exemplar is embedded again from its episode's diarize spans and separated stem, then
    every centroid is recomputed and projections of another dimension are dropped, in one
    registry transaction. Names, locks, voices and external ids are kept. Run it before
    re-running diarize with the new embedder (the exemplars' cluster labels refer to the
    existing diarization), then re-run the pipeline.
    """
    from fukikae.errors import user_errors
    from fukikae.registry import reembed as rr

    ws = Workspace(workspace)
    try:
        p = rr.plan(ws)
    except user_errors() as e:
        raise _fail(str(e)) from e
    typer.echo(f"embedder: {p.embedder.backend} ({p.dim}-d)")
    typer.echo(f"exemplars: {_dims(p.exemplar_dims)}")
    typer.echo(f"centroids: {_dims(p.centroid_dims)}")
    drop = [x for x in p.projection_dims if x[2] != p.dim]
    typer.echo(
        "projections: "
        + (", ".join(f"{s}/{k} ({d}-d)" for s, k, d in p.projection_dims) or "none")
        + (f"; {len(drop)} of another dimension to drop" if drop else "")
    )
    typer.echo(f"{p.spans} spans to embed from {len(p.work)} episodes")
    for aid in sorted(p.fixed):
        typer.echo(p.treatment(aid))
    if dry_run:
        typer.echo("dry run: nothing changed")
        return
    try:
        r = rr.run(ws, p)
    except user_errors() as e:
        raise _fail(f"{e}; nothing was changed") from e
    typer.echo(
        f"re-embedded {p.spans} exemplars as {p.dim}-d; {len(r.centroids)} centroids "
        f"recomputed; kept {', '.join(map(str, r.kept)) or 'none'}; mapped "
        f"{', '.join(map(str, r.mapped)) or 'none'}; "
        f"dropped projections {', '.join(f'{s}/{k}' for s, k, _ in r.projections) or 'none'}. "
        "Now re-run the pipeline (`fukikae run`)"
    )


def resolve_settings(
    cfg: Any,
    matcher: str | None,
    threshold: float | None,
    options: dict[str, Any] | None,
    language: str | None = None,
) -> tuple[str, float, dict[str, Any]]:
    """The batch matcher, threshold and matcher options: explicit flags over the ``[resolve]``
    table (workspace over language over profile, Ruling R17) over the built-ins (joint, the
    registry's match_threshold, no options). A ``[resolve]`` table without ``matcher`` means
    joint and still applies its other keys. `language` picks ``[resolve.lang.<code>]``: the
    episodes' language as the characters stage reads it (`episodes_language`)."""
    table = cfg.resolve("resolve", language, key="matcher", default="joint")
    name, opts = table.backend, dict(table.options)
    thr = opts.pop("threshold", None)
    if thr is None:
        thr = cfg.registry.match_threshold
    return (
        matcher or name,
        float(threshold if threshold is not None else thr),
        {**opts, **(options or {})},
    )


def episodes_language(cfg: Any, eps: list[Episode]) -> str | None:
    """The language the characters stage matches these episodes in (each one's detected
    ``language.json``, else ``source_language``); refuses episodes of different languages,
    which belong in separate workspaces."""
    from fukikae.stages.characters import CharactersStage

    langs = {ep.id: CharactersStage._language(ep, cfg) for ep in eps}
    if len(set(langs.values())) > 1:
        raise _fail(
            "the episodes are in different languages ("
            + ", ".join(f"{e}: {lang or 'unknown'}" for e, lang in langs.items())
            + "); each audio language is its own workspace and registry"
        )
    return next(iter(langs.values()), None)


@registry_app.command("resolve")
def resolve(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(
        ..., "--episodes", help="Comma-separated episodes resolved together, e.g. S01E12,S01E13."
    ),
    matcher: str = typer.Option(
        None, "--matcher", help="Batch matcher: joint (default: [resolve] matcher, else joint)."
    ),
    threshold: float = typer.Option(
        None,
        "--threshold",
        help="Cosine to merge groups (default: [resolve] threshold, else match_threshold).",
    ),
    matcher_option: list[str] = typer.Option(
        None,
        "--matcher-option",
        help="k=v joint option: episode_cannot_link (true), linkage (centroid); repeat. "
        "Over the same keys in [resolve].",
    ),
) -> None:
    """Re-resolve several episodes at once with the joint matcher, rewriting the registry and
    their assignments.

    Every listed episode needs a done diarize stage. Actors only these episodes made are
    replaced (unless curated: locked, named, or hand-voiced); every other actor is matched
    against and never merged with another. One registry transaction; assignments.json
    written after it.
    """
    from fukikae.errors import user_errors
    from fukikae.eval.identity.cli import parse_options
    from fukikae.registry.resolve import resolve_episodes

    ws = Workspace(workspace)
    eps = _worked_episodes(ws, episodes)
    try:
        cfg = ws.config
        matcher, thr, opts = resolve_settings(
            cfg,
            matcher,
            threshold,
            parse_options(matcher_option, "--matcher-option"),
            episodes_language(cfg, eps),
        )
    except (ValueError, *user_errors()) as e:
        raise _fail(str(e)) from e
    if matcher != "joint":
        raise _fail(
            f"--matcher {matcher}: resolve runs the batch matcher joint; per-episode matchers "
            "run in the characters stage (`fukikae run`, `registry rematch`)"
        )
    try:
        s = resolve_episodes(ws, eps, thr, options=opts)
    except user_errors() as e:
        raise _fail(str(e)) from e
    typer.echo(
        f"resolved {', '.join(s.episodes)} with joint at {thr:g}: {s.matched} clusters matched "
        f"to existing actors, {len(s.new_actors)} new actors"
        f"{' (' + ', '.join(map(str, s.new_actors)) + ')' if s.new_actors else ''}, "
        f"{s.extras} extras; pruned {', '.join(map(str, s.pruned)) or 'none'}"
    )
