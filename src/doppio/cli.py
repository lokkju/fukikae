"""Command line entry point."""

import typer

from doppio import __version__

app = typer.Typer(help="Automatically dub foreign-language TV into English.", no_args_is_help=True)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """doppio command group.

    Args:
        version: Print the version and exit.
    """


backends_app = typer.Typer(help="Inspect installed model backends.", no_args_is_help=True)
app.add_typer(backends_app, name="backends")


@backends_app.command("list")
def backends_list() -> None:
    """List installed backends for each kind."""
    from doppio.backends.base import KINDS
    from doppio.backends.loader import names

    for kind in KINDS:
        found = names(kind)
        typer.echo(f"{kind}: {', '.join(found) if found else '(none)'}")
