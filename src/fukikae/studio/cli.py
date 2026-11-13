"""`fukikae studio` commands."""

from pathlib import Path

import typer

studio_app = typer.Typer(help="The season studio: identity and voices.", no_args_is_help=True)


@studio_app.command("apply")
def apply(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    file: Path = typer.Argument(
        ...,
        metavar="FILE",
        help="Decisions: a JSON document ({workspace, decisions}), or JSON lines of decisions "
        "or of studio/decisions.jsonl records.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Check each decision and say what it would rerun; change nothing."
    ),
    force: bool = typer.Option(
        False, "--force", help="Apply decisions whose registry rows changed since they were made."
    ),
) -> None:
    """Apply studio decisions to the registry in order, each in its own transaction, as the
    studio page does. A decision made against rows that have changed since is refused (unless
    --force), and so is one the registry refuses; either stops the file, and the decisions
    before it stay applied."""
    from fukikae.studio.apply import DecisionError, apply_file

    try:
        results = apply_file(workspace, file, dry_run=dry_run, force=force)
    except (DecisionError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for r in results:
        for line in r.lines:
            typer.echo(line)
    typer.echo(f"{len(results)} decision(s) {'checked (dry run)' if dry_run else 'applied'}")
