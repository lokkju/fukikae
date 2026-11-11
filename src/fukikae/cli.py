"""Command line entry point."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from fukikae import __version__
from fukikae.cards.cli import voices_app
from fukikae.eval.cli import eval_app
from fukikae.names.cli import names_app
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
app.add_typer(names_app, name="names")

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


@backends_app.command("setup")
def backends_setup(
    name: str = typer.Argument(..., help="A backend with model files to fetch."),
    no_fetch: bool = typer.Option(False, "--no-fetch", help="Sync only; download nothing."),
    force: bool = typer.Option(
        False, "--force", help="Redo the costly steps even when done (indextts2_trt's engines)."
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        help="Fetch the model this workspace's config names (default: the default profile's); "
        "for indextts2_trt, build the engines its [tts.indextts2_trt] asks for.",
    ),
) -> None:
    """Fetch a backend's pinned model files: the only step that downloads (ruling CD24). A
    cloning TTS backend's worker environment is set up first (`uv sync --locked`, the lock's
    hash recorded); an in-process backend (Kokoro, SenseVoice, emotion2vec, the managed
    translator's GGUF) fetches its pinned snapshot into the Hugging Face cache."""
    import subprocess

    from fukikae.backends.base import KINDS
    from fukikae.backends.loader import BackendUnavailable, backend_class, names
    from fukikae.backends.tts.remote import WorkerSynthesizer
    from fukikae.config import ConfigError

    kinds = [k for k in KINDS if name in names(k)]
    if not kinds:
        typer.echo(f"error: no backend named {name!r} (see `fukikae backends list`)", err=True)
        raise typer.Exit(1)
    try:
        cls = backend_class(kinds[0], name)
    except (ConfigError, BackendUnavailable) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    if isinstance(cls, type) and issubclass(cls, WorkerSynthesizer):
        options = _worker_setup_options(name, workspace)
        try:
            cls.setup(fetch=not no_fetch, force=force, options=options)
        except subprocess.CalledProcessError as e:
            typer.echo(f"error: {' '.join(map(str, e.cmd))} exited with {e.returncode}", err=True)
            raise typer.Exit(1) from e
        except (OSError, BackendUnavailable) as e:  # docker (or uv) missing, docker down
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
        typer.echo(f"{name}: set up in {cls(options, 'cpu').environment_location()}")
        return
    fetch = getattr(cls, "fetch", None)
    if fetch is None:
        typer.echo(f"{name}: nothing to fetch")
        return
    if no_fetch:
        typer.echo(f"{name}: --no-fetch, nothing done")
        return
    for path in fetch(_setup_options(kinds[0], name, workspace)):
        typer.echo(f"{name}: {path}")


def _worker_setup_options(name: str, workspace: Path | None) -> dict[str, object]:
    """``[tts.<name>]``'s options in ``workspace``'s config (or the default profile's), outside
    ``device`` and ``force_cpu``: what a cloning backend renders with."""
    import tempfile

    from fukikae.config import load_config
    from fukikae.stages.tts import backend_config

    with tempfile.TemporaryDirectory(prefix="fukikae-setup-") as tmp:
        cfg = load_config(workspace or Path(tmp), None)
        bc = backend_config(cfg, name)
    return dict(bc.options) if bc.backend == name else {}


def _setup_options(kind: str, name: str, workspace: Path | None) -> dict[str, object]:
    """The options ``name`` gets in ``workspace``'s config (or the default profile's)."""
    import tempfile

    from fukikae.config import load_config

    with tempfile.TemporaryDirectory(prefix="fukikae-setup-") as tmp:
        cfg = load_config(workspace or Path(tmp), None)
        try:
            bc = cfg.resolve(kind)
        except Exception:  # noqa: BLE001 - a kind with no table: the backend's defaults
            return {}
    return dict(bc.options) if bc.backend == name else {}


@app.command()
def status(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    as_json: bool = typer.Option(False, "--json", help="Print status.json, with `live` added."),
) -> None:
    """Show what a run in the workspace is doing: each thread's episode, stage, counts and
    ETA, the GPU lock, the last log line; or that no run is in progress."""
    import json

    from fukikae import progress

    data = progress.read_status(workspace)
    if as_json:
        out = dict(data or {})
        out["live"] = bool(data) and progress.is_live(data or {})
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(progress.describe(data))


@app.command()
def doctor() -> None:
    """Check ffmpeg, the GPU, the Hugging Face token and installed backends."""
    from fukikae.doctor import doctor as checks

    for label, ok, detail in checks():
        typer.echo(f"{'ok  ' if ok else 'MISS'} {label}: {detail}")


def configure_logging() -> None:
    """Send fukikae's INFO lines (stage summaries) to stderr, once, unless something already
    configured the root logger. Model libraries may also configure it later, mid-stage (funasr
    calls basicConfig on import), so fukikae's logger stops propagating to the root once it has
    its own handler: each line prints exactly once."""
    import logging

    logger = logging.getLogger("fukikae")
    if logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def pipeline_stages(cfg: "Config") -> "list[Stage]":
    """The stages `fukikae run` goes through, in order; `tag` only when `[tag] enabled`."""
    from fukikae.stages.asr import AsrStage
    from fukikae.stages.characters import CharactersStage
    from fukikae.stages.diarize import DiarizeStage
    from fukikae.stages.extract import ExtractStage
    from fukikae.stages.language import LanguageStage
    from fukikae.stages.mix import MixStage
    from fukikae.stages.names import NamesStage
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
        NamesStage(cfg),
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


# The stages `fukikae run` with several episodes overlaps with the next episode's GPU stages.
CPU_TAIL = ("timefit", "mix")


@app.command()
def run(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: list[str] = typer.Argument(
        ...,
        help="Episode file paths or SxxEyy codes; several run in order, each one's "
        "timefit and mix overlapping the next one's GPU stages.",
    ),
    from_stage: str | None = typer.Option(None, "--from", help="Rerun from this stage."),
    to_stage: str | None = typer.Option(None, "--to", help="Stop after this stage."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
) -> None:
    """Run the pipeline stages for one episode, or several in order."""
    from fukikae import job, progress, runctx
    from fukikae.errors import user_errors
    from fukikae.workspace import Workspace

    runctx.vram_timeout_s = 0 if no_wait else None
    configure_logging()
    ws = Workspace(workspace)
    if len(episodes) > 1:
        try:
            eps = [ws.find_episode(e) for e in episodes]
            with progress.run(ws.root, [ep.id for ep in eps]):
                results = job.run_episodes(
                    eps, ws.config, pipeline_stages(ws.config), tail=CPU_TAIL,
                    from_stage=from_stage, to_stage=to_stage,
                )  # fmt: skip
        except user_errors() as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
        failed = False
        bug: BaseException | None = None
        for ep in eps:
            res = results.get(ep.id)
            if res is None:
                typer.echo(f"{ep.id}: not started (an earlier episode failed)")
                continue
            if res.ran or res.error is None:
                ran = ", ".join(res.ran) if res.ran else "nothing (up to date)"
                typer.echo(f"{ep.id}: ran {ran}")
            if res.error is not None:
                failed = True
                if isinstance(res.error, user_errors()):
                    typer.echo(f"error: {ep.id}: {res.error}", err=True)
                else:  # a bug: every episode is reported first, then its traceback
                    typer.echo(f"error: {ep.id}: {type(res.error).__name__}: {res.error}", err=True)
                    bug = bug or res.error
        if bug is not None:
            raise bug
        if failed:
            raise typer.Exit(1)
        return
    try:
        ep = ws.find_episode(episodes[0])
        stages = pipeline_stages(ws.config)
        with progress.run(ws.root, [ep.id]):
            try:
                ran = job.run_stages(ep, ws.config, stages, from_stage, to_stage)
            except BaseException:
                progress.episode(ep.id, "failed")
                raise
            progress.episode(ep.id, "done")
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{ep.id}: ran {', '.join(ran) if ran else 'nothing (up to date)'}")
