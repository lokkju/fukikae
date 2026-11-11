"""`fukikae eval` commands."""

from pathlib import Path
from typing import Any

import typer

eval_app = typer.Typer(help="Evaluate backends against reference data.", no_args_is_help=True)


@eval_app.command("asr")
def asr(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode file path or SxxEyy code."),
    subtitle_track: int = typer.Option(..., "--subtitle-track", help="Absolute stream index."),
    backend: list[str] = typer.Option(None, "--backend", help="ASR backend; repeat for several."),
    split: str = typer.Option(
        "tune", "--split", help="Ledger split of this episode: tune or holdout."
    ),
    test_set: str = typer.Option(
        None, "--set", help="Ledger test-set name (default: the workspace directory's name)."
    ),
    tag: str = typer.Option(None, "--tag", help="Free-form ledger tag for this run."),
    no_ledger: bool = typer.Option(
        False, "--no-ledger", help="Do not append rows to the results ledger."
    ),
    ledger: Path = typer.Option(
        None,
        "--ledger",
        help="Results ledger (default: $FUKIKAE_RESULTS, else "
        "$XDG_DATA_HOME/fukikae/results.jsonl).",
    ),
) -> None:
    """Score ASR backends by character error rate against a subtitle track."""
    from fukikae.backends.loader import names
    from fukikae.errors import user_errors
    from fukikae.eval import asr as eval_asr
    from fukikae.eval.identity.cli import write_ledger
    from fukikae.eval.ledger import SPLITS
    from fukikae.workspace import Workspace

    if split not in SPLITS:
        typer.echo(f"error: --split {split!r}: use one of {', '.join(SPLITS)}", err=True)
        raise typer.Exit(2)
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
    if not no_ledger:
        name = test_set or ws.root.resolve().name
        write_ledger(
            lambda: _asr_rows(scores, cfg, language, ep.id, name, split, tag, ws.root), ledger
        )


def _asr_rows(
    scores: list[Any],
    cfg: Any,
    language: str,
    episode_id: str,
    test_set: str,
    split: str,
    tag: str | None,
    root: Path,
) -> list[Any]:
    """One `asr` ledger row per backend. Model, revision and options come from the
    workspace's resolved ASR config when it names the same backend; another backend ran with
    its defaults, so it records none."""
    from fukikae.config import BackendConfig, ConfigError
    from fukikae.eval.ledger import make_row, model_identity

    try:
        resolved: BackendConfig | None = cfg.resolve("asr", language)
    except ConfigError:
        resolved = None
    rows = []
    for s in scores:
        same = resolved is not None and resolved.backend == s.backend
        options = dict(resolved.options) if same and resolved is not None else {}
        model, revision = (
            model_identity("asr", resolved) if same and resolved is not None else (None, None)
        )
        metrics: dict[str, float | int | str] = {"cer": s.cer}
        if s.runtime_s is not None:
            metrics["runtime_s"] = s.runtime_s
        rows.append(
            make_row(
                stage="asr",
                backend=s.backend,
                model=model,
                revision=revision,
                options=options,
                language=language,
                genre=cfg.series.genre,
                series=cfg.series.title,
                test_set=test_set,
                split=split,
                episodes=[episode_id],
                metrics=metrics,
                workspace=str(root),
                tag=tag,
            )
        )
    return rows


@eval_app.command("tts")
def tts_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(..., "--episodes", help="Comma-separated, e.g. S01E11,S01E12."),
    backend: str = typer.Option(
        None,
        "--backend",
        help="Render with this TTS backend: clone cards from each character's reference "
        "clips for a cloning backend, else stock cards (default: each character's assigned "
        "card).",
    ),
    per_character: int = typer.Option(8, "--per-character", min=1, help="Cues per character."),
    seed: int = typer.Option(0, "--seed", help="Seed of the cue sample."),
    characters: str = typer.Option(
        None,
        "--characters",
        help="Score only these character ids (comma-separated); each keeps the cues a full "
        "run samples for it.",
    ),
    min_slot: float = typer.Option(
        1.0, "--min-slot", min=0.0, help="Leave out cues whose slot is shorter (seconds)."
    ),
    floor_subs: Path = typer.Option(
        None,
        "--floor-subs",
        help="Subtitle file (or a directory matched by SxxEyy) with the text of natural "
        "English speech, for the WER floor: on the reference workspace's audio at the sampled "
        "cues, else on --floor-workspace's --floor-episode as a fixed clip set.",
    ),
    floor_workspace: Path = typer.Option(
        None, "--floor-workspace", help="Workspace whose vocals the fixed WER floor set uses."
    ),
    floor_episode: str = typer.Option(
        None, "--floor-episode", help="Episode of the fixed WER floor set, e.g. S01E01."
    ),
    reference_workspace: Path = typer.Option(
        None,
        "--reference-workspace",
        help="Workspace of the original-language track on the same timeline (BBT's English): "
        "adds duration, emotion and voice against the original line.",
    ),
    test_set: str = typer.Option(
        None, "--set", help="Ledger test-set name (default: the workspace directory's name)."
    ),
    tag: str = typer.Option(None, "--tag", help="Free-form ledger tag for this run."),
    split: str = typer.Option("tune", "--split", help="Ledger split: tune or holdout."),
    no_ledger: bool = typer.Option(
        False, "--no-ledger", help="Do not append rows to the results ledger."
    ),
    ledger: Path = typer.Option(
        None,
        "--ledger",
        help="Results ledger (default: $FUKIKAE_RESULTS, else "
        "$XDG_DATA_HOME/fukikae/results.jsonl).",
    ),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
) -> None:
    """Render a sample of cues per character and score speaker similarity, intelligibility,
    duration fit, emotion carry-over and per-card speech rate."""
    from fukikae import runctx
    from fukikae.errors import user_errors
    from fukikae.eval import tts as ev
    from fukikae.eval.identity.cli import write_ledger
    from fukikae.eval.ledger import SPLITS
    from fukikae.workspace import Workspace

    if split not in SPLITS:
        typer.echo(f"error: --split {split!r}: use one of {', '.join(SPLITS)}", err=True)
        raise typer.Exit(2)
    runctx.vram_timeout_s = 0 if no_wait else None
    refs = [e.strip().upper() for e in episodes.split(",") if e.strip()]
    ws = Workspace(workspace)
    try:
        cfg = ws.config
        result = ev.evaluate(
            ws.root,
            refs,
            backend=backend,
            per_character=per_character,
            seed=seed,
            min_slot=min_slot,
            reference_workspace=reference_workspace,
            floor_subs=floor_subs,
            floor_workspace=floor_workspace,
            floor_episode=floor_episode,
            characters=(
                [int(x) for x in characters.split(",") if x.strip()] if characters else None
            ),
        )
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for note in result.notes:
        typer.echo(f"note: {note}", err=True)
    for b, summary in result.summaries.items():
        typer.echo(ev.format_summary(summary, result.names, b))
        typer.echo("")
    if not no_ledger:
        name = test_set or ws.root.resolve().name
        options = {
            "mode": result.mode,
            "per_character": per_character,
            "seed": seed,
            "min_slot": min_slot,
            "asr": f"{ev.ASR.backend}:{ev.ASR.options['model']}",
            "embedder": ev.EMBEDDER.backend,
            "reference": reference_workspace is not None,
            "emotion_backend": result.tagger,
            "wer_floor": result.floor,
        }

        def rows() -> list[Any]:
            out: list[Any] = []
            for b, summary in result.summaries.items():
                out += ev.tts_rows(
                    summary,
                    cfg=cfg,
                    backend_cfg=result.backend_cfgs[b],
                    source_language=result.source_language,
                    test_set=name,
                    split=split,
                    episodes=result.episodes,
                    tag=tag,
                    workspace=ws.root,
                    options=options,
                    notes=result.notes,
                )
            return out

        write_ledger(rows, ledger)


@eval_app.command("gender")
def gender_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episode: str = typer.Argument(..., help="Episode code, e.g. S01E01."),
    similarity: bool = typer.Option(
        False, "--similarity", help="Also score render-vs-source speaker cosine (WeSpeaker, CPU)."
    ),
    before: Path = typer.Option(
        None,
        "--before",
        help="An earlier episode's 07-tts/plan.json (beside its 06-translate): report the cues "
        "it voiced with a stock card apart, matched by source text and start.",
    ),
    json_out: Path = typer.Option(None, "--json", help="Write the report as JSON here."),
    md_out: Path = typer.Option(None, "--md", help="Write the report as Markdown here."),
    pitch_cache: Path = typer.Option(
        None,
        "--pitch-cache",
        help="Root of the pitch cache (default: the workspace's own .fukikae/pitch-cache); "
        "name one to leave a read-only workspace untouched.",
    ),
    min_voiced: float = typer.Option(
        None,
        "--min-voiced",
        min=0.0,
        help="Seconds voiced below which a line or render is u (default: pitch.LINE_MIN_VOICED_S).",
    ),
    test_set: str = typer.Option(
        None, "--set", help="Ledger test-set name (default: the workspace directory's name)."
    ),
    tag: str = typer.Option(None, "--tag", help="Free-form ledger tag for this run."),
    split: str = typer.Option("tune", "--split", help="Ledger split: tune or holdout."),
    no_ledger: bool = typer.Option(
        False, "--no-ledger", help="Do not append rows to the results ledger."
    ),
    ledger: Path = typer.Option(
        None,
        "--ledger",
        help="Results ledger (default: $FUKIKAE_RESULTS, else "
        "$XDG_DATA_HOME/fukikae/results.jsonl).",
    ),
) -> None:
    """Score a dubbed episode's voice-register consistency: source line against render,
    reference purity and tiers, stock-voiced cues, gated lines and wrong-gender emotion
    prompts (Part A evaluation; CPU only)."""
    import json

    from fukikae import pitch
    from fukikae.errors import user_errors
    from fukikae.eval import gender as ev
    from fukikae.eval.identity.cli import write_ledger
    from fukikae.eval.ledger import SPLITS
    from fukikae.workspace import Workspace

    if split not in SPLITS:
        typer.echo(f"error: --split {split!r}: use one of {', '.join(SPLITS)}", err=True)
        raise typer.Exit(2)
    ws = Workspace(workspace)
    try:
        cfg = ws.config
        report = ev.measure(
            ws.root,
            episode,
            similarity=similarity,
            before=before,
            cache_root=pitch_cache,
            min_voiced_s=pitch.LINE_MIN_VOICED_S if min_voiced is None else min_voiced,
        )
    except (*user_errors(), FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    text = ev.format_report(report)
    typer.echo(text)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(ev.report_json(report), ensure_ascii=False, indent=1, default=str) + "\n",
            encoding="utf-8",
        )
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(text, encoding="utf-8")
    if not no_ledger:
        name = test_set or ws.root.resolve().name
        write_ledger(
            lambda: ev.gender_rows(report, cfg=cfg, test_set=name, split=split, tag=tag), ledger
        )


@eval_app.command("tts-ab")
def tts_ab_cmd(
    spec: Path = typer.Argument(..., help="Pack spec (TOML): seed, [[set]] tables."),
    out: Path = typer.Option(..., "-o", "--out", help="New or empty folder for the pack."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
) -> None:
    """Build a blind A/B listening pack of the TTS candidates (the D15 checkpoint): per cue the
    original line and each candidate under a shuffled letter, a scene per set with
    [set.scene], index.html, README.txt, verdict.toml and the separate key.toml."""
    from fukikae import runctx
    from fukikae.errors import user_errors
    from fukikae.eval import ab

    runctx.vram_timeout_s = 0 if no_wait else None
    try:
        pack, notes = ab.build_pack(ab.load_spec(spec), out.expanduser(), log=typer.echo)
    except (*user_errors(), FileExistsError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for note in notes:
        typer.echo(f"note: {note}", err=True)
    typer.echo(f"pack: {pack}")


@eval_app.command("report")
def report_cmd(
    ledger: Path = typer.Option(
        None,
        "--ledger",
        help="Results ledger (default: $FUKIKAE_RESULTS, else "
        "$XDG_DATA_HOME/fukikae/results.jsonl).",
    ),
    stage: list[str] = typer.Option(None, "--stage", help="Only this stage; repeat."),
    backend: list[str] = typer.Option(None, "--backend", help="Only this backend; repeat."),
    test_set: list[str] = typer.Option(None, "--set", help="Only this test set; repeat."),
    split: list[str] = typer.Option(None, "--split", help="Only this split; repeat."),
    tag: list[str] = typer.Option(None, "--tag", help="Only this tag; repeat."),
    metric: str = typer.Option(
        None,
        "--metric",
        help="Metric shown in each cell (default: asr cer, identity fraction with clean and "
        "merge counts, embedder nc_overall, tts speaker_similarity).",
    ),
    fmt: str = typer.Option("text", "--format", help="text or md."),
    export: Path = typer.Option(
        None, "--export", help="Also write the filtered rows to this file as JSONL."
    ),
) -> None:
    """Render the results ledger as a matrix: configurations by language and genre."""
    from fukikae.eval.ledger import read_rows, row_json
    from fukikae.eval.report import build_matrix, filter_rows, render

    if fmt not in ("text", "md"):
        typer.echo(f"error: --format {fmt!r}: use text or md", err=True)
        raise typer.Exit(2)
    try:
        rows, warnings = read_rows(ledger)
    except OSError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for w in warnings:
        typer.echo(f"warning: {w}", err=True)
    rows = filter_rows(
        rows,
        stages=stage or (),
        backends=backend or (),
        sets=test_set or (),
        splits=split or (),
        tags=tag or (),
    )
    if export is not None:
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text("".join(row_json(r) + "\n" for r in rows), encoding="utf-8")
        typer.echo(f"exported {len(rows)} rows to {export}", err=True)
    if not rows:
        typer.echo("no ledger rows match")
        return
    typer.echo(render(build_matrix(rows, metric), "md" if fmt == "md" else "text"))


@eval_app.command("retract")
def retract_cmd(
    run_id: str = typer.Argument(..., help="The run_id of the ledger row to retract."),
    reason: str = typer.Option(..., "--reason", help="Why the row is wrong (kept in the ledger)."),
    ledger: Path = typer.Option(None, "--ledger", help="Ledger file (default: the user ledger)."),
) -> None:
    """Retract a ledger row: append a record that hides it from `eval report`; the row stays."""
    from fukikae.eval.ledger import retract

    try:
        path = retract(run_id, reason, ledger)
    except (ValueError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"retracted {run_id} in {path}")


@eval_app.command("sample")
def sample(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(..., "--episodes", help="Comma-separated, e.g. S01E01,S01E02."),
    per_cluster: int = typer.Option(4, "--per-cluster", min=1, help="Clips per cluster."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite a labels.yaml that has labels, and remove old clips."
    ),
    video: bool = typer.Option(False, "--video", help="Export mp4 clips with the picture."),
) -> None:
    """Export clips per diarized cluster and a labels.yaml to fill in, blind to the registry."""
    from fukikae.eval import speakers
    from fukikae.media import MediaError

    refs = [e.strip() for e in episodes.split(",") if e.strip()]
    labels_path = out / "labels.yaml"
    try:
        if not force and speakers.labels_have_content(labels_path):
            raise ValueError(f"{labels_path} already has labels; pass --force to overwrite")
        if force:
            speakers.remove_clips(out)
        clips = speakers.sample(workspace, refs, per_cluster, out, video=video)
    except (FileNotFoundError, ValueError, MediaError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"wrote {len(clips)} clips and {labels_path}")


@eval_app.command("speakers")
def speakers_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    labels: Path = typer.Option(..., "--labels", help="The filled-in labels.yaml."),
) -> None:
    """Score hand labels against the registry's assignments and print the verdict."""
    from fukikae.eval import speakers

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


@eval_app.command("auto-label")
def auto_label_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(..., "--episodes", help="Comma-separated, e.g. S01E11,S01E12."),
    out: Path = typer.Option(..., "--out", help="Output directory for labels.yaml."),
    subtitle_track: int = typer.Option(
        None, "--subtitle-track", help="Absolute stream index of an ASS track naming speakers."
    ),
    min_overlap: float = typer.Option(
        0.6, "--min-overlap", min=0.0, max=1.0, help="Share of a span one name must cover."
    ),
    alias: list[str] = typer.Option(
        None, "--alias", help="FROM=TO speaker-name alias; repeat. Also labels-aliases.yaml."
    ),
    report: bool = typer.Option(
        False, "--report", help="Print per-cluster purity and name spread."
    ),
    source: str = typer.Option(
        "ass",
        "--source",
        help="Ground truth: ass (Name field), sdh (NAME:/(NAME) tags) or colour (text colours).",
    ),
    gt_file: Path = typer.Option(
        None,
        "--gt-file",
        help="Sidecar subtitle file (or a directory of them, matched by SxxEyy) instead of "
        "an embedded track.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Score even when the subtitles do not line up with the speech."
    ),
    colour_anonymous: str = typer.Option(
        None,
        "--colour-anonymous",
        help="With --source colour and a labels-colours.yaml: drop (default), series "
        "(colour:<name>) or episode (<name>@<episode>) for colours the map lacks.",
    ),
) -> None:
    """Label every clean span from the speaker names, tags or colours in the episodes' subtitles."""
    from fukikae.eval import autolabel
    from fukikae.media import MediaError

    refs = [e.strip() for e in episodes.split(",") if e.strip()]
    try:
        aliases: dict[str, str] = {}
        for item in alias or []:
            src, sep, dst = item.partition("=")
            if not sep or not src.strip() or not dst.strip():
                raise ValueError(f"--alias {item!r}: expected FROM=TO")
            aliases[src.strip()] = dst.strip()
        result = autolabel.auto_label(
            workspace, refs, out, subtitle_track, min_overlap, aliases,
            source=source, gt_file=gt_file, force=force, colour_anonymous=colour_anonymous,
        )  # fmt: skip
    except (FileNotFoundError, ValueError, MediaError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    kept = sum(1 for c in result.clips if c.label != autolabel.UNKNOWN)
    typer.echo(f"wrote {len(result.clips)} spans ({kept} labeled) to {out / 'labels.yaml'}")
    for ep_id, rep, applied in result.alignment:
        if applied:
            typer.echo(
                f"{ep_id}: subtitles ran {rep.best_offset_s:+.1f} s against the speech; "
                f"moved back ({rep.overlap_share:.0%} -> {rep.share_at_offset:.0%} overlap)"
            )
    if report:
        typer.echo(autolabel.format_report(result))


@eval_app.command("subalign")
def subalign_cmd(
    labelled: Path = typer.Argument(..., help="Subtitle file to retime (ASS/SSA, SRT, WebVTT)."),
    out: Path = typer.Option(..., "--out", help="Where to write the retimed file."),
    reference: Path = typer.Option(
        None, "--reference", help="Subtitle file already timed to the target capture."
    ),
    speech: Path = typer.Option(
        None, "--speech", help="Workspace whose diarized speech is the target timeline."
    ),
    episode: str = typer.Option(None, "--episode", help="SxxEyy, with --speech."),
    language: str = typer.Option(
        None,
        "--language",
        help="Text language for anchor matching (default: guessed from the labelled file).",
    ),
    force: bool = typer.Option(
        False, "--force", help="Write the file even when the fit fails its checks."
    ),
) -> None:
    """Find the offset and linear drift from LABELLED to a reference subtitle file (by text) or
    to a workspace's detected speech, and write LABELLED retimed to OUT."""
    import re

    from fukikae.eval.autolabel import speech_turns
    from fukikae.eval.groundtruth.base import read_cues
    from fukikae.eval.groundtruth.subalign import (
        check_fit,
        fit_to_reference,
        fit_to_speech,
        guess_language,
        retime_file,
    )
    from fukikae.workspace import Workspace

    if (reference is None) == (speech is None) or (speech is not None and not episode):
        typer.echo(
            "error: give either --reference REF or --speech WORKSPACE --episode SxxEyy", err=True
        )
        raise typer.Exit(2)
    try:
        cues = read_cues(labelled)
        if reference is not None:
            rep = fit_to_reference(cues, read_cues(reference), language or guess_language(cues))
        else:
            ws = Workspace(speech)
            m = re.fullmatch(r"[Ss](\d+)[Ee](\d+)", episode.strip())
            eid = f"S{int(m[1]):02d}E{int(m[2]):02d}" if m else ws.find_episode(episode).id
            work = ws.work_dir(eid)
            if not (work / "04-diarize").is_dir():
                raise FileNotFoundError(f"{eid}: no diarization under {work / '04-diarize'}")
            rep = fit_to_speech(cues, speech_turns(work))
        s = rep.tmap
        line = f"mode {rep.mode}  scale {s.scale:.6f}  offset {s.offset:+.2f} s  "
        if rep.mode == "text":
            share = rep.inliers / rep.anchors if rep.anchors else 0.0
            line += f"anchors {rep.anchors}  inliers {rep.inliers} ({share:.0%})  "
            line += f"rms {rep.rms_s:.3f} s  max residual {rep.max_resid_s:.3f} s"
        else:
            line += f"cues {rep.anchors}  over speech {rep.inliers}  share {rep.share or 0:.0%}"
        typer.echo(line)
        check_fit(rep, force=force)
        done, clamped = retime_file(labelled, out, rep.tmap)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"wrote {out}: {done} cues retimed, {clamped} clamped at 0")


@eval_app.command("transcript-align")
def transcript_align_cmd(
    transcripts: Path = typer.Option(
        ..., "--transcripts", help="Transcript file or directory (matched by SxxEyy)."
    ),
    subtitles: Path = typer.Option(
        ..., "--subtitles", help="Timed subtitle file or directory (matched by SxxEyy)."
    ),
    out: Path = typer.Option(..., "--out", help="Output directory for <SxxEyy>.ass."),
    episodes: str = typer.Option(..., "--episodes", help="Comma-separated, e.g. S01E01,S01E02."),
    alias: list[str] = typer.Option(
        None, "--alias", help="FROM=TO speaker-name alias; repeat. Also labels-aliases.yaml."
    ),
    language: str = typer.Option("en", "--language", help="Language of the subtitle words."),
) -> None:
    """Name the cues of timed subtitles from untimed, speaker-named transcripts."""
    from fukikae.eval.autolabel import gt_path_for
    from fukikae.eval.groundtruth.base import Namer, load_alias_file, read_cues
    from fukikae.eval.groundtruth.transcript import (
        align_transcript,
        read_transcript,
        write_named_ass,
    )

    refs = [e.strip().upper() for e in episodes.split(",") if e.strip()]
    try:
        aliases = load_alias_file(out.parent / "labels-aliases.yaml")
        for item in alias or []:
            src, sep, dst = item.partition("=")
            if not sep or not src.strip() or not dst.strip():
                raise ValueError(f"--alias {item!r}: expected FROM=TO")
            aliases[src.strip()] = dst.strip()
        results = []
        for ref in refs:
            tr = gt_path_for(transcripts, ref, len(refs))
            sub = gt_path_for(subtitles, ref, len(refs))
            res = align_transcript(read_transcript(tr), read_cues(sub), Namer(aliases), language)
            results.append((ref, res))
        out.mkdir(parents=True, exist_ok=True)
        for ref, res in results:
            write_named_ass(out / f"{ref}.ass", res.named)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for ref, res in results:
        who = ", ".join(f"{n} {c}" for n, c in sorted(res.speakers.items(), key=lambda x: -x[1]))
        typer.echo(f"{ref}: cues {res.cue_share:.0%}, words {res.word_share:.0%}; {who}")


@eval_app.command("bazinga-labels")
def bazinga_labels_cmd(
    annotations: list[Path] = typer.Argument(..., help="Bazinga annotation .txt files."),
    out: Path = typer.Option(..., "--out", help="Output directory for <SxxEyy>.ass."),
    min_confidence: float = typer.Option(
        0.9, "--min-confidence", help="Words below this confidence never set a turn's span."
    ),
    time_scale: float = typer.Option(
        1.0, "--time-scale", help="Multiply every time (1.04271 for audio slowed to film speed)."
    ),
) -> None:
    """Named-ASS ground truth from Bazinga's word-level speaker annotations."""
    from fukikae.eval.groundtruth.bazinga import episode_code, read_words, words_to_turns
    from fukikae.eval.groundtruth.transcript import write_named_ass

    try:
        done: dict[str, tuple[Path, Any, Any]] = {}
        for path in annotations:
            words = read_words(path)
            if not words:
                raise ValueError(f"{path}: no annotation lines")
            ref = episode_code(words[0].episode)
            if ref in done:
                raise ValueError(f"{path} and {done[ref][0]} are both {ref}")
            lines, stats = words_to_turns(words, min_confidence, time_scale)
            done[ref] = (path, lines, stats)
        out.mkdir(parents=True, exist_ok=True)
        for ref, (_, lines, _) in done.items():
            write_named_ass(out / f"{ref}.ass", lines)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for ref, (_, _, s) in sorted(done.items()):
        why = ", ".join(f"{k} {v}" for k, v in sorted(s.dropped.items())) or "none"
        typer.echo(
            f"{ref}: words kept {s.kept}, dropped {why}; turns {s.turns} "
            f"({s.turns_dropped} with no confident word dropped); speakers {len(s.speakers)}"
        )


@eval_app.command("colour-map")
def colour_map_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    episodes: str = typer.Option(..., "--episodes", help="Comma-separated, e.g. S01E01,S01E02."),
    gt_file: Path = typer.Option(
        ..., "--gt-file", help="Colour-caption file, or a directory matched by SxxEyy."
    ),
    alias: list[str] = typer.Option(
        None, "--alias", help="FROM=TO speaker-name alias; repeat. Also labels-aliases.yaml."
    ),
    write: bool = typer.Option(False, "--write", help="Write WORKSPACE/labels-colours.yaml."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing labels-colours.yaml."),
) -> None:
    """Map each caption colour to a character from the named lines of all the episodes."""
    from fukikae.eval.autolabel import gt_path_for
    from fukikae.eval.groundtruth.base import Namer, load_alias_file, read_cues
    from fukikae.eval.groundtruth.colour import (
        COLOUR_MAP_FILE,
        colour_counts,
        propose_colour_map,
        write_colour_map,
    )

    refs = [e.strip().upper() for e in episodes.split(",") if e.strip()]
    target = workspace / COLOUR_MAP_FILE
    try:
        aliases = load_alias_file(workspace / "labels-aliases.yaml")
        for item in alias or []:
            src, sep, dst = item.partition("=")
            if not sep or not src.strip() or not dst.strip():
                raise ValueError(f"--alias {item!r}: expected FROM=TO")
            aliases[src.strip()] = dst.strip()
        if write and target.exists() and not force:
            raise ValueError(f"{target} exists; --force overwrites it")
        cues = {ref: read_cues(gt_path_for(gt_file, ref, len(refs))) for ref in refs}
        counts = colour_counts(cues, Namer(aliases))
        mapping, conflicts = propose_colour_map(counts)
        colours = sorted({c for per in counts.values() for c in per})
        for col in colours:
            cells = []
            for ref in refs:
                c = counts[ref].get(col)
                if c:
                    cells.append(f"{ref} " + ", ".join(f"{n} {k}" for n, k in c.most_common()))
            typer.echo(f"{col}: " + ("; ".join(cells) if cells else "no tagged lines"))
        typer.echo("map: " + (", ".join(f"{c} -> {n}" for c, n in mapping.items()) or "(empty)"))
        for line in conflicts:
            typer.echo(f"conflict: {line}")
        if write:
            write_colour_map(target, mapping)
            typer.echo(f"wrote {target}")
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


@eval_app.command("embedders")
def embedders(
    clipset: Path = typer.Argument(..., help="Clip set: one folder per character."),
    embedder: list[str] = typer.Option(
        ..., "--embedder", help="Embed backend to score; repeat for several."
    ),
    language: str = typer.Option(..., "--language", help="Language of the clips (ledger column)."),
    genre: str = typer.Option("live-action", "--genre", help="Ledger genre."),
    test_set: str = typer.Option(
        None, "--set", help="Ledger test-set name (default: the clip set directory's name)."
    ),
    exclude: list[str] = typer.Option(
        None, "--exclude", help="Character folder to leave out; repeat for several."
    ),
    episode_pattern: str = typer.Option(
        None,
        "--episode-pattern",
        help="Regex whose first group is the episode number in a file name.",
    ),
    tag: str = typer.Option(None, "--tag", help="Free-form ledger tag for this run."),
    no_ledger: bool = typer.Option(
        False, "--no-ledger", help="Do not append rows to the results ledger."
    ),
    ledger: Path = typer.Option(
        None,
        "--ledger",
        help="Results ledger (default: $FUKIKAE_RESULTS, else "
        "$XDG_DATA_HOME/fukikae/results.jsonl).",
    ),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
) -> None:
    """Score embedders by nearest-centroid accuracy on a folder-per-character clip set."""
    from fukikae import runctx
    from fukikae.backends.loader import names
    from fukikae.config import BackendConfig, Config
    from fukikae.errors import user_errors
    from fukikae.eval import clipset as cs
    from fukikae.eval.identity.cache import ResidentEmbedder
    from fukikae.eval.identity.cli import write_ledger
    from fukikae.eval.identity.metrics import OVERALL
    from fukikae.eval.ledger import make_row, model_identity

    runctx.vram_timeout_s = 0 if no_wait else None
    root = clipset.expanduser()
    try:
        known = names("embed")
        for name in embedder:
            if name not in known:
                raise ValueError(f"unknown embedder {name!r}; use one of {', '.join(known)}")
        if not root.is_dir():
            raise FileNotFoundError(f"clip set {root} is not a directory")
        clips = cs.scan_clipset(
            root, episode_pattern or cs.DEFAULT_EPISODE_PATTERN, exclude=exclude or ()
        )
        train, test, rule, skipped = cs.split_clips(clips)
        missing = sum(c.episode is None for c in clips)
        if missing and rule != "episode":
            typer.echo(
                f"warning: {missing} of {len(clips)} clips carry no episode number; the whole "
                f"set uses the {rule!r} split instead of the episode split",
                err=True,
            )
        if not train or not test:
            raise ValueError(f"{root}: no character has clips on both sides of the split")
        cache = root / cs.CACHE_DIR
        audio = cs.prepare_audio(train + test, cache)
        name = test_set or root.resolve().name
        results: list[tuple[Any, dict[str, float]]] = []
        with ResidentEmbedder(Config(gpu_device=0)) as resident:
            for emb_name in embedder:
                cfg = BackendConfig(emb_name, {})
                ident = cs.cache_identity(cfg)
                tr = cs.embed_clips(train, audio, cfg, resident, cache, ident)
                te = cs.embed_clips(test, audio, cfg, resident, cache, ident)
                results.append((cfg, cs.clipset_accuracy(train, test, tr, te, audio=audio)))
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    scored = sorted({c.character for c in test} - set(skipped))
    typer.echo(f"{'embedder':<16}{'accuracy':>9}{'chars':>7}{'train':>7}{'test':>6}  split")
    for cfg, acc in results:
        typer.echo(
            f"{cfg.backend:<16}{acc.get(OVERALL, 0.0):>9.3f}{len(acc) - 1 if acc else 0:>7}"
            f"{len(train):>7}{len(test):>6}  {rule}"
        )
    if rule != "episode":
        typer.echo(
            "note: folder/half splits cut across a character's folders (e.g. emotions), so "
            "they also measure robustness to them; check class balance before reading accuracy.",
            err=True,
        )
    if skipped:
        typer.echo(f"not scored (no clip on one side): {', '.join(skipped)}", err=True)
    if not no_ledger:

        def rows() -> list[Any]:
            out = []
            for cfg, acc in results:
                model, revision = model_identity("embed", cfg)
                per_char = {k: round(v, 6) for k, v in acc.items() if k != OVERALL}
                out.append(
                    make_row(
                        stage="embedder",
                        backend=cfg.backend,
                        model=model,
                        revision=revision,
                        options={**cfg.options, "split_rule": rule},
                        language=language,
                        genre=genre,
                        series=name,
                        test_set=name,
                        split="tune",
                        episodes=[],
                        metrics={
                            "nc_overall": acc.get(OVERALL, 0.0),
                            "characters": len(per_char),
                            "train_clips": len(train),
                            "test_clips": len(test),
                        },
                        detail={
                            "per_character": per_char,
                            "not_scored": skipped,
                            "excluded": list(exclude or []),
                            "characters": scored,
                        },
                        workspace=str(root.resolve()),
                        tag=tag,
                    )
                )
            return out

        write_ledger(rows, ledger)


def _register_identity() -> None:
    from fukikae.eval.identity.cli import identity_cmd

    eval_app.command("identity")(identity_cmd)


def _register_names() -> None:
    from fukikae.eval.names import names_cmd

    eval_app.command("names")(names_cmd)


_register_identity()
_register_names()
