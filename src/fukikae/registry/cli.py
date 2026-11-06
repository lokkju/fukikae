"""`fukikae registry` commands."""

from pathlib import Path

import typer

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
    if missing or any(not p.exists() for p in stage.inputs(ep)):
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
        if rec is None or rec.status != "done" or any(not p.exists() for p in stage.inputs(ep)):
            not_done.append(ep.id)
    if not_done:
        raise _fail(f"diarize stage not done for {', '.join(not_done)}; run `fukikae run` first")

    with _open(workspace) as reg, reg.transaction():  # all of it, or none of it
        if reg.actors():
            raise _fail(
                "the registry already has actors; batch seeding needs an empty one. "
                "Use per-episode matching (`fukikae run`, `registry rematch`)."
            )
        loaded = {ep.id: load_clusters(stage.inputs(ep)[1]) for ep in eps}
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
        turns_path = stage.inputs(ep)[0]
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
    """Merge actor DROP into actor KEEP; DROP's characters fold into KEEP's first one."""
    ws = Workspace(workspace)
    with _open(workspace) as reg:
        try:
            affected = list(reg.episode_counts(drop))
            chars = reg.merge_actors(keep, drop)
        except (KeyError, ValueError) as e:
            raise _fail(f"cannot merge {drop} into {keep}: {e!r}") from e
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


def _worked_episodes(ws: Workspace, episodes: str) -> list[Episode]:
    """The listed episodes (SxxEyy codes or media paths), each with a done diarize stage in
    this workspace; anything else is refused as not an episode of this workspace."""
    from fukikae.job import load_manifest
    from fukikae.stages.characters import CharactersStage

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
            or any(not p.exists() for p in CharactersStage().inputs(ep))
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

    Each episode's diarization clean spans are labeled from its subtitles (as `eval identity`
    labels them) and the labeled rows of its clusters.npz, the embeddings the characters
    stage matches, train the projection. `[characters] matcher = "lda"` then matches in it.
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
            turns = read_json(ep.work / "04-diarize" / "turns.json", Diarization).turns
            label_of = {
                (s.cluster, round(s.start, 6), round(s.end, 6)): s.label
                for s in diarization_segments(ep.id, turns, lines)
            }
            with np.load(ep.work / "04-diarize" / "clusters.npz") as z:
                for cluster in [str(c) for c in z["labels"]]:
                    emb = np.asarray(z[f"{cluster}/embeddings"], dtype=np.float32)
                    spans = np.asarray(z[f"{cluster}/spans"]).reshape(-1, 2)
                    for row, (a, b) in zip(emb, spans, strict=True):
                        lab = label_of.get((cluster, round(float(a), 6), round(float(b), 6)))
                        if lab is not None:
                            x.append(row)
                            y.append(lab)
    except user_errors() as e:
        raise _fail(str(e)) from e
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


@registry_app.command("resolve")
def resolve(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(
        ..., "--episodes", help="Comma-separated episodes resolved together, e.g. S01E12,S01E13."
    ),
    matcher: str = typer.Option("joint", "--matcher", help="Batch matcher: joint."),
    threshold: float = typer.Option(
        None, "--threshold", help="Cosine to merge groups (default: registry match_threshold)."
    ),
    matcher_option: list[str] = typer.Option(
        None,
        "--matcher-option",
        help="k=v joint option: episode_cannot_link (true), linkage (centroid); repeat.",
    ),
) -> None:
    """Re-resolve several episodes at once with the joint matcher, rewriting the registry and
    their assignments.

    Every listed episode needs a done diarize stage. Actors only these episodes made are
    replaced (unless locked, named or voiced); every other actor is matched against and
    never merged with another. One registry transaction; assignments.json written after it.
    """
    from fukikae.errors import user_errors
    from fukikae.eval.identity.cli import parse_options
    from fukikae.registry.resolve import resolve_episodes

    if matcher != "joint":
        raise _fail(
            f"--matcher {matcher}: resolve runs the batch matcher joint; per-episode matchers "
            "run in the characters stage (`fukikae run`, `registry rematch`)"
        )
    ws = Workspace(workspace)
    eps = _worked_episodes(ws, episodes)
    try:
        opts = parse_options(matcher_option, "--matcher-option")
        thr = ws.config.registry.match_threshold if threshold is None else threshold
        s = resolve_episodes(ws, eps, thr, options=opts)
    except user_errors() as e:
        raise _fail(str(e)) from e
    typer.echo(
        f"resolved {', '.join(s.episodes)} with joint at {thr:g}: {s.matched} clusters matched "
        f"to existing actors, {len(s.new_actors)} new actors"
        f"{' (' + ', '.join(map(str, s.new_actors)) + ')' if s.new_actors else ''}, "
        f"{s.extras} extras; pruned {', '.join(map(str, s.pruned)) or 'none'}"
    )
