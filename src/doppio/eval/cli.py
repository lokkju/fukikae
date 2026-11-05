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
    from doppio.errors import user_errors
    from doppio.eval import asr as eval_asr
    from doppio.workspace import Workspace

    ws = Workspace(workspace)
    try:
        cfg = ws.config
        ep = ws.find_episode(episode)
        backends = list(backend) if backend else names("asr")
        language = cfg.source_language or "en"
        scores = eval_asr.evaluate(ep, cfg, subtitle_track, backends, language)
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{'backend':<20}{'CER':>8}{'runtime':>10}")
    for s in scores:
        runtime = "cached" if s.runtime_s is None else f"{s.runtime_s:.0f}s"
        typer.echo(f"{s.backend:<20}{s.cer:>8.4f}{runtime:>10}")


@eval_app.command("sample")
def sample(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(..., "--episodes", help="Comma-separated, e.g. S01E01,S01E02."),
    per_cluster: int = typer.Option(4, "--per-cluster", min=1, help="Clips per cluster."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    force: bool = typer.Option(False, "--force", help="Overwrite a labels.yaml that has labels."),
) -> None:
    """Export clips per diarized cluster and a labels.yaml to fill in, blind to the registry."""
    from doppio.eval import speakers

    refs = [e.strip() for e in episodes.split(",") if e.strip()]
    labels_path = out / "labels.yaml"
    try:
        if not force and speakers.labels_have_content(labels_path):
            raise ValueError(f"{labels_path} already has labels; pass --force to overwrite")
        clips = speakers.sample(workspace, refs, per_cluster, out)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"wrote {len(clips)} clips and {labels_path}")


@eval_app.command("speakers")
def speakers_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    labels: Path = typer.Option(..., "--labels", help="The filled-in labels.yaml."),
) -> None:
    """Score hand labels against the registry's assignments and print the verdict."""
    from doppio.eval import speakers

    try:
        clips = speakers.load_labels(labels)
        s = speakers.score(clips, speakers.load_assignments(workspace, {c.episode for c in clips}))
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"recurring: {s.recurring}")
    typer.echo(f"consistent: {s.consistent}")
    typer.echo(f"fraction: {s.fraction:.2f}")
    if s.merged_pairs:
        for (a, b), cid in zip(s.merged_pairs, s.merged_ids or [], strict=False):
            typer.echo(f"merged: {a} + {b} (character {cid})")
    else:
        typer.echo("merged pairs: none")
    typer.echo(f"verdict: {speakers.verdict(s)}")
