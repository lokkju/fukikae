"""`doppio registry` commands."""

from pathlib import Path

import typer

from doppio.workspace import Episode, Workspace

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
    from doppio import job
    from doppio.artifacts import read_json
    from doppio.models import Assignments
    from doppio.stages.characters import CharactersStage

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
    if missing or any(not p.exists() for p in stage.inputs(ep)):
        typer.echo(
            f"error: {ep.id}: earlier stages not done ({', '.join(missing) or 'diarize outputs'}); "
            f"run `doppio run` first",
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
    from doppio.registry.store import Registry

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


def _stale(ws: Workspace, episodes: list[str]) -> None:
    """Delete each episode's `characters` manifest record and say so."""
    from doppio.job import mark_stale

    if not episodes:
        typer.echo("no episode has exemplars under it; nothing is stale")
        return
    for eid in episodes:
        ep = Episode(source=Path(eid), id=eid, work=ws.work_dir(eid), workspace_root=ws.root)
        mark_stale(ep, "characters")
    typer.echo(f"stale (characters will rerun on the next `doppio run`): {', '.join(episodes)}")


def _remap_assignments(
    ws: Workspace,
    episode: str,
    actor_map: dict[int, int],
    only_cluster: str | None = None,
    character: tuple[int, int] | None = None,
) -> None:
    """Keep an episode's assignments.json in step with a merge or split.

    Pinning copies actor and character ids from it, so stale ids would survive a rerun.
    """
    from doppio.artifacts import read_json, write_json
    from doppio.models import Assignments

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

    from doppio.artifacts import read_json, write_json
    from doppio.job import load_manifest
    from doppio.models import Assignments, ClusterAssignment, Diarization
    from doppio.registry.cluster import seed_clusters
    from doppio.registry.matching import load_clusters, mean_embedding
    from doppio.stages.characters import CharactersStage

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
        if rec is None or rec.status != "done" or any(not p.exists() for p in stage.inputs(ep)):
            not_done.append(ep.id)
    if not_done:
        raise _fail(f"diarize stage not done for {', '.join(not_done)}; run `doppio run` first")

    with _open(workspace) as reg:
        if reg.actors():
            raise _fail(
                "the registry already has actors; batch seeding needs an empty one. "
                "Use per-episode matching (`doppio run`, `registry rematch`)."
            )
        loaded = {ep.id: load_clusters(stage.inputs(ep)[1]) for ep in eps}
        groups = seed_clusters(
            {eid: ev for eid, (ev, _) in loaded.items()},
            threshold if threshold is not None else 1.0 - cfg.registry.match_threshold,
            cfg.registry.min_new_seconds,
        )
        member: dict[tuple[str, str], tuple[int, int]] = {}
        for g in groups:
            allrows = np.concatenate([loaded[e][1][c] for e, c in g])
            centroid = mean_embedding(allrows)
            assert centroid is not None
            aid, cid = reg.create_actor_with_character(centroid)
            for e, c in g:
                ev = next(x for x in loaded[e][0] if x.cluster == c)
                reg.add_exemplars(aid, e, c, loaded[e][1][c], ev.clean_seconds)
                member[(e, c)] = (aid, cid)
            reg.recompute_centroid(aid)

    for ep in eps:
        turns_path = stage.inputs(ep)[0]
        turns = read_json(turns_path, Diarization)
        evs, _ = loaded[ep.id]
        out = []
        for ev in evs:
            hit = member.get((ep.id, ev.cluster))
            out.append(
                ClusterAssignment(
                    cluster=ev.cluster,
                    actor_id=hit[0] if hit else None,
                    character_id=hit[1] if hit else None,
                    score=None,
                    clean_seconds=ev.clean_seconds,
                    status="matched" if hit else "extra",
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
                typer.echo(f"  character {c.id}: {c.name or '-'}")
        for eid, n in reg.episode_counts(actor_id).items():
            typer.echo(f"  {eid}: {n} exemplars")


@registry_app.command("rename")
def rename(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    character_id: int = typer.Argument(..., metavar="CHARACTER", help="Character id."),
    name: str = typer.Argument(..., metavar="NAME", help="New name."),
) -> None:
    """Name a character; its episodes' characters stage reruns next time."""
    ws = Workspace(workspace)
    with _open(workspace) as reg:
        char = next((c for c in reg.characters() if c.id == character_id), None)
        if char is None:
            raise _fail(f"no character {character_id}")
        reg.rename_character(character_id, name)
        affected = list(reg.episode_counts(char.actor_id))
    _stale(ws, affected)


@registry_app.command("merge")
def merge(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    keep: int = typer.Argument(..., metavar="KEEP", help="Actor id to keep."),
    drop: int = typer.Argument(..., metavar="DROP", help="Actor id folded into KEEP."),
) -> None:
    """Merge actor DROP into actor KEEP."""
    ws = Workspace(workspace)
    with _open(workspace) as reg:
        try:
            affected = list(reg.episode_counts(drop))
            reg.merge_actors(keep, drop)
        except (KeyError, ValueError) as e:
            raise _fail(f"cannot merge {drop} into {keep}: {e!r}") from e
    for eid in affected:
        _remap_assignments(ws, eid, {drop: keep})
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
    actor_id: int = typer.Argument(..., metavar="ID", help="Actor id."),
    unlock: bool = typer.Option(False, "--unlock", help="Unlock instead."),
) -> None:
    """Lock an actor's centroid (or unlock it)."""
    with _open(workspace) as reg:
        try:
            reg.lock_actor(actor_id, not unlock)
        except KeyError as e:
            raise _fail(f"no actor {actor_id}") from e
    typer.echo(f"actor {actor_id}: {'unlocked' if unlock else 'locked'}")


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
