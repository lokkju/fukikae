"""`fukikae studio` commands."""

import sqlite3
from pathlib import Path
from typing import Any

import typer
from typer.core import TyperGroup


class _ServeByDefault(TyperGroup):
    """`fukikae studio <ws>` is `fukikae studio serve <ws>`: a first argument that names no
    subcommand and isn't ``--help`` goes to ``serve``, options included (``fukikae studio
    --port 0 <ws>``). A workspace directory literally named like a subcommand is reached as
    ``./apply``."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:  # typer's vendored click
        if args and args[0] not in self.commands and args[0] not in ctx.help_option_names:
            args = ["serve", *args]
        return super().parse_args(ctx, args)


studio_app = typer.Typer(
    cls=_ServeByDefault,
    help="The season studio: identity and voices. `fukikae studio <ws>` serves it (the "
    "`serve` command); `apply` and `resume` work on decisions without it.",
    no_args_is_help=True,
)


@studio_app.command("serve")
def serve(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    port: int = typer.Option(8765, "--port", help="Port to listen on (0: any free one)."),
    clips: int = typer.Option(6, "--clips", min=0, help="Sample clips per character."),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Address to bind: loopback unless --allow-non-loopback."
    ),
    allow_non_loopback: bool = typer.Option(
        False,
        "--allow-non-loopback",
        help="Allow --host to name a non-loopback IPv4 address. Anyone who can reach it and "
        "has the token can change the registry.",
    ),
) -> None:
    """Serve the season studio for WORKSPACE on 127.0.0.1: builds or loads the season index,
    prints the URL (this run's token after `#t=`, which the browser keeps to itself) and the
    `ssh -L` line that reaches it from your machine, and serves until Ctrl-C. It opens no
    browser. Every API and audio request needs the token."""
    from fukikae.registry.store import RegistrySchemaError
    from fukikae.studio.server import StudioServer
    from fukikae.workspace import Workspace

    if not (workspace / "fukikae.toml").is_file():
        typer.echo(f"error: {workspace} is not a workspace (no fukikae.toml)", err=True)
        raise typer.Exit(1)
    ws = Workspace(workspace.resolve())
    try:
        server = StudioServer(
            ws, ws.config, host=host, port=port, clips=clips,
            allow_non_loopback=allow_non_loopback,
        )  # fmt: skip
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    except OSError as e:
        typer.echo(f"error: can't listen on {host}:{port}: {e}", err=True)
        raise typer.Exit(1) from e
    try:
        typer.echo("loading the season index ...")
        try:
            doc = server.season()
        except (OSError, ValueError, RegistrySchemaError, sqlite3.DatabaseError) as e:
            typer.echo(f"error: the season index: {e}", err=True)
            raise typer.Exit(1) from e
        ready = sum(e.ready for e in doc.episodes)
        typer.echo(
            f"{len(doc.characters)} character(s); {ready} of {len(doc.episodes)} episode(s) ready"
        )
        typer.echo(f"studio: {server.url()}")
        if server.loopback:
            typer.echo(f"from your machine: {server.tunnel()}")
        typer.echo("Ctrl-C stops it.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            typer.echo("stopped")
    finally:
        server.server_close()


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
    records what is left: fix the cause and run `fukikae studio resume`. Replaying the
    decisions log skips the decisions already applied."""
    from fukikae.studio.apply import DecisionError, apply_file

    try:
        results = apply_file(workspace, file, dry_run=dry_run, force=force)
    except DecisionError as e:
        _report(e.results, dry_run, force)  # what landed before the error
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    except OSError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    _report(results, dry_run, force)


def _report(results: list, dry_run: bool, force: bool) -> None:  # noqa: FBT001
    for r in results:
        for line in r.lines:
            typer.echo(line)
    done = "checked (dry run)" if dry_run else "applied"
    ran = [r for r in results if not r.skipped]
    skipped = len(results) - len(ran)
    typer.echo(
        f"{len(ran)} decision(s) {done}" + (f", {skipped} already applied" if skipped else "")
    )
    if force and ran:
        bare = sum(r.undigested for r in ran)
        typer.echo(
            f"--force: {sum(r.unchecked for r in ran)} decision(s) {done} without a digest "
            f"check; {bare} carried no digest"
        )


@studio_app.command("resume")
def resume(workspace: Path = typer.Argument(..., help="Workspace directory.")) -> None:
    """Finish decisions (and `registry merge` runs) whose registry part landed but whose
    steps after it (stale marks, assignments remapped, a pitch split file) failed or were
    interrupted: the decisions log names what is left, and this redoes only that. A record
    whose registry transaction never committed is logged as aborted, with nothing to redo.
    Every step is idempotent, so running it again is safe; until it succeeds, `studio apply`,
    `registry merge` and `fukikae run` refuse to start."""
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
