"""Command line entry point."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from fukikae import __version__
from fukikae.cards.cli import voices_app
from fukikae.eval.cli import eval_app
from fukikae.registry.cli import registry_app

if TYPE_CHECKING:
    from fukikae.config import Config
    from fukikae.job import Stage

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
    """fukikae command group.

    Args:
        version: Print the version and exit.
    """


app.add_typer(eval_app, name="eval")
app.add_typer(registry_app, name="registry")
app.add_typer(voices_app, name="voices")

backends_app = typer.Typer(help="Inspect installed model backends.", no_args_is_help=True)
app.add_typer(backends_app, name="backends")


@backends_app.command("list")
def backends_list() -> None:
    """List installed backends for each kind."""
    from fukikae.backends.base import KINDS
    from fukikae.backends.loader import names

    for kind in KINDS:
        found = names(kind)
        typer.echo(f"{kind}: {', '.join(found) if found else '(none)'}")


@app.command()
def doctor() -> None:
    """Check ffmpeg, the GPU, the Hugging Face token and installed backends."""
    from fukikae.doctor import doctor as checks

    for label, ok, detail in checks():
        typer.echo(f"{'ok  ' if ok else 'MISS'} {label}: {detail}")


def configure_logging() -> None:
    """Send fukikae's INFO lines (stage summaries) to stderr, once, unless something already
    configured the root logger (some model libraries do on import)."""
    import logging

    logger = logging.getLogger("fukikae")
    if logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def pipeline_stages(cfg: "Config") -> "list[Stage]":
    """The stages `fukikae run` goes through, in order; `tag` only when `[tag] enabled`."""
    from fukikae.stages.asr import AsrStage
    from fukikae.stages.characters import CharactersStage
    from fukikae.stages.diarize import DiarizeStage
    from fukikae.stages.extract import ExtractStage
    from fukikae.stages.language import LanguageStage
    from fukikae.stages.mix import MixStage
    from fukikae.stages.separate import SeparateStage
    from fukikae.stages.tag import TagStage, tag_enabled
    from fukikae.stages.timefit import TimefitStage
    from fukikae.stages.translate import TranslateStage
    from fukikae.stages.tts import TtsStage

    stages: list[Stage] = [
        ExtractStage(),
        SeparateStage(),
        LanguageStage(),
        AsrStage(),
        DiarizeStage(),
        CharactersStage(),
    ]
    tags = tag_enabled(cfg)
    if tags:
        stages.append(TagStage())
    stages.append(TranslateStage(cfg.target_language, tags=tags, cfg=cfg))
    stages.append(TtsStage(cfg))
    stages.append(TimefitStage(cfg))
    stages.append(MixStage(cfg, tags=tags))
    return stages


@app.command()
def run(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
    from_stage: str | None = typer.Option(None, "--from", help="Rerun from this stage."),
    to_stage: str | None = typer.Option(None, "--to", help="Stop after this stage."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
) -> None:
    """Run the pipeline stages for one episode."""
    from fukikae import job, runctx
    from fukikae.errors import user_errors
    from fukikae.workspace import Workspace

    runctx.vram_timeout_s = 0 if no_wait else None
    configure_logging()
    ws = Workspace(workspace)
    try:
        ep = ws.find_episode(episode)
        stages = pipeline_stages(ws.config)
        ran = job.run_stages(ep, ws.config, stages, from_stage, to_stage)
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{ep.id}: ran {', '.join(ran) if ran else 'nothing (up to date)'}")
