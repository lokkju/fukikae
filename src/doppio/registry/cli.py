"""`doppio registry` commands."""

from pathlib import Path

import typer

registry_app = typer.Typer(help="Manage the series character registry.", no_args_is_help=True)


@registry_app.command("rematch")
def rematch(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
) -> None:
    """Forget an episode's registry contributions and match its clusters again."""
    from doppio.artifacts import read_json
    from doppio.job import _load_manifest, run_stages
    from doppio.models import Assignments
    from doppio.registry.store import Registry
    from doppio.stages.characters import CharactersStage
    from doppio.workspace import Workspace

    ws = Workspace(workspace)
    try:
        ep = ws.find_episode(episode)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e

    stage = CharactersStage()
    records = _load_manifest(ep).stages
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

    assignments_path, prev_turns_path = stage.outputs(ep)
    old: dict[str, int | None] = {}
    if assignments_path.exists():
        old = {c.cluster: c.character_id for c in read_json(assignments_path, Assignments).clusters}
    with Registry(ws.registry_path) as reg:
        reg.delete_episode(ep.id)
    assignments_path.unlink(missing_ok=True)
    prev_turns_path.unlink(missing_ok=True)

    run_stages(ep, ws.config, [stage], from_stage="characters")

    changed = 0
    for c in read_json(assignments_path, Assignments).clusters:
        before = old.get(c.cluster)
        if before != c.character_id:
            changed += 1
            typer.echo(f"{c.cluster}: {before} -> {c.character_id}")
    if not changed:
        typer.echo("no cluster changed character")
