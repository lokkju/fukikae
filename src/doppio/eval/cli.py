"""`doppio eval` commands."""

from pathlib import Path

import typer

eval_app = typer.Typer(help="Evaluate backends against reference data.", no_args_is_help=True)


@eval_app.command("asr")
def asr(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
    subtitle_track: int = typer.Option(..., "--subtitle-track", help="Absolute stream index."),
    backend: list[str] = typer.Option(None, "--backend", help="ASR backend; repeat for several."),
) -> None:
    """Score ASR backends by character error rate against a subtitle track."""
    from doppio.backends.loader import names
    from doppio.eval.asr import evaluate
    from doppio.media import MediaError
    from doppio.workspace import Workspace

    ws = Workspace(workspace)
    try:
        cfg = ws.config
        ep = ws.find_episode(episode)
        backends = list(backend) if backend else names("asr")
        language = cfg.source_language or "en"
        scores = evaluate(ep, cfg, subtitle_track, backends, language)
    except (FileNotFoundError, MediaError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{'backend':<20}{'CER':>8}{'runtime':>10}")
    for s in scores:
        runtime = "cached" if s.runtime_s is None else f"{s.runtime_s:.0f}s"
        typer.echo(f"{s.backend:<20}{s.cer:>8.4f}{runtime:>10}")
