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
