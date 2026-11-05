"""Command line entry point."""

import os
from pathlib import Path

import typer

from doppio import __version__
from doppio.eval.cli import eval_app

# torch must number cards the way nvidia-smi and NVML do (gpu_device is an NVML index).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

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


app.add_typer(eval_app, name="eval")

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


@app.command()
def doctor() -> None:
    """Check ffmpeg, the GPU, the Hugging Face token and installed backends."""
    from doppio.doctor import doctor as checks

    for label, ok, detail in checks():
        typer.echo(f"{'ok  ' if ok else 'MISS'} {label}: {detail}")


@app.command()
def run(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
    from_stage: str | None = typer.Option(None, "--from", help="Rerun from this stage."),
    to_stage: str | None = typer.Option(None, "--to", help="Stop after this stage."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
) -> None:
    """Run the pipeline stages for one episode."""
    from doppio import runctx
    from doppio.job import run_stages
    from doppio.media import MediaError
    from doppio.stages.asr import AsrStage
    from doppio.stages.diarize import DiarizeStage
    from doppio.stages.extract import ExtractStage
    from doppio.stages.language import LanguageStage
    from doppio.stages.separate import SeparateStage
    from doppio.workspace import Workspace

    runctx.vram_timeout_s = 0 if no_wait else None
    ws = Workspace(workspace)
    try:
        ep = ws.find_episode(episode)
        stages = [ExtractStage(), SeparateStage(), LanguageStage(), AsrStage(), DiarizeStage()]
        ran = run_stages(ep, ws.config, stages, from_stage, to_stage)
    except (FileNotFoundError, MediaError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{ep.id}: ran {', '.join(ran) if ran else 'nothing (up to date)'}")
