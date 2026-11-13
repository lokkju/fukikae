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
        False,
        "--force",
        help="Skip the digest check for every decision in the file: apply those whose registry "
        "rows changed since they were made, and those that carry no digest. The summary says "
        "how many went unchecked.",
    ),
) -> None:
    """Apply studio decisions to the registry in order, each in its own transaction, as the
    studio page does. A decision made against rows that have changed since is refused, and so
    is one that carries no digest (unless --force), and one the registry refuses; any of
    these stops the file, and the decisions before it stay applied. A dry run checks the
    whole file in one transaction that is rolled back, so later decisions see earlier ones.
    When a decision's registry part lands but a file change after it fails, the decisions log
    records what is left: fix the cause and run `fukikae studio resume`."""
    from fukikae.studio.apply import DecisionError, apply_file

    try:
        results = apply_file(workspace, file, dry_run=dry_run, force=force)
    except (DecisionError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for r in results:
        for line in r.lines:
            typer.echo(line)
    done = "checked (dry run)" if dry_run else "applied"
    typer.echo(f"{len(results)} decision(s) {done}")
    if force and results:
        bare = sum(r.undigested for r in results)
        typer.echo(
            f"--force: {sum(r.unchecked for r in results)} decision(s) {done} without a digest "
            f"check; {bare} carried no digest"
        )


@studio_app.command("resume")
def resume(workspace: Path = typer.Argument(..., help="Workspace directory.")) -> None:
    """Finish decisions whose registry part landed but whose steps after it (stale marks,
    assignments remapped, a pitch split file) failed: the decisions log names what is left,
    and this redoes only that. Every step is idempotent, so running it again is safe; until
    it succeeds, `studio apply` refuses new decisions."""
    from fukikae.studio.apply import DecisionError, resume_incomplete

    try:
        results = resume_incomplete(workspace)
    except (DecisionError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for r in results:
        for line in r.lines:
            typer.echo(line)
    typer.echo(f"{len(results)} incomplete decision(s) finished")
