"""`fukikae eval identity`: score cross-episode speaker identity against subtitle ground truth.

The tuning episodes are resolved in order against an in-memory registry (the first builds it,
each later one resolves against what came before) and scored together. A held-out episode is
resolved against the registry the tuning episodes built (Ruling R7) and scored separately:
each character's id is its modal id in the tuning episodes, and its share, purity, coverage
and merges come from the held-out speech alone (`metrics.score(reference=...)`). Its recurring
characters have at least `--min-speech` (default 5 s, Ruling R6) in the tuning episodes
together and in the held-out one.

Reads pipeline artifacts (``04-diarize/turns.json``, ``01-separate/vocals_16k.wav``) and the
subtitles; writes only the embedding cache, extracted subtitle tracks under
``.fukikae/cache/subtitles/`` and the optional ``--json`` file. Never touches the registry.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import typer

if TYPE_CHECKING:
    import numpy as np

    from fukikae.config import BackendConfig, Config
    from fukikae.eval.groundtruth import AlignmentReport, GroundTruthLine
    from fukikae.eval.identity.metrics import IdentityScore
    from fukikae.eval.identity.segments import Segment
    from fukikae.registry.matchers import Matcher, RegistryState, SegmentLike
    from fukikae.workspace import Workspace

# Heavy imports (numpy, scipy, the backends) wait for the command to run, so `fukikae`
# itself starts fast.

POOLED = "tune"
_POOL_GAP_S = 1.0e6  # tuning episodes laid end to end in the pooled view, far apart


@dataclass
class EpisodeInfo:
    episode: str
    gt_path: str
    lines: int
    segments: int
    labeled: int
    alignment: AlignmentReport
    offset_applied: bool
    statuses: dict[str, int] = field(default_factory=dict)


@dataclass
class IdentityRun:
    settings: dict[str, Any]
    episodes: list[EpisodeInfo]
    tune: IdentityScore
    holdout: IdentityScore | None
    holdout_episode: str | None
    cache_hits: int
    cache_misses: int
    # Nearest-centroid accuracy (Ruling R8): the first tuning episode trains, the other
    # tuning episodes test; for the holdout, all tuning episodes train.
    tune_nc: dict[str, float] = field(default_factory=dict)
    holdout_nc: dict[str, float] = field(default_factory=dict)


class _Recording:
    """Wraps a matcher to keep each episode's cluster statuses, when it reports them."""

    name: ClassVar[str] = "recording"

    def __init__(self, inner: Matcher) -> None:
        self.inner = inner
        self.statuses: dict[str, dict[str, int]] = {}

    def fit_episode(
        self, state: RegistryState, episode: str, embs: np.ndarray, segs: Sequence[SegmentLike]
    ) -> list[int | None]:
        from collections import Counter

        ids = self.inner.fit_episode(state, episode, embs, segs)
        last = getattr(self.inner, "last", [])
        self.statuses[episode] = dict(Counter(a.status for a in last))
        return ids


def parse_options(items: list[str] | None, flag: str = "--embedder-option") -> dict[str, Any]:
    """`k=v` pairs; a value that reads as a TOML scalar (8, 0.5, true) becomes one."""
    out: dict[str, Any] = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"{flag} {item!r}: expected k=v")
        try:
            out[k.strip()] = tomllib.loads(f"v = {v.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            out[k.strip()] = v.strip()
    return out


def parse_aliases(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        src, sep, dst = item.partition("=")
        if not sep or not src.strip() or not dst.strip():
            raise ValueError(f"--alias {item!r}: expected FROM=TO")
        out[src.strip()] = dst.strip()
    return out


def embedder_config(cfg: Config, backend: str | None, options: dict[str, Any]) -> BackendConfig:
    """`backend` with exactly `options`; with no backend, the workspace's embedder (as the
    diarize stage resolves it) with `options` laid over its own."""
    from fukikae.config import BackendConfig

    if backend:
        return BackendConfig(backend, dict(options))

    base = cfg.resolve("embed")
    return BackendConfig(base.backend, {**base.options, **options})


def _ground_truth(
    ws: Workspace,
    ref: str,
    eid: str,
    kind: str,
    gt_file: Path | None,
    n_episodes: int,
    subtitle_track: int | None,
    aliases: dict[str, str],
) -> tuple[Path, list[GroundTruthLine]]:
    from fukikae.eval import autolabel
    from fukikae.eval.groundtruth import Kind, read_ground_truth
    from fukikae.media import probe

    if gt_file is not None:
        path = autolabel.gt_path_for(gt_file, eid, n_episodes)
    else:
        ep = ws.find_episode(ref)
        scratch = ws.root / ".fukikae" / "cache" / "subtitles" / eid
        pr = probe(ep.source)
        if kind == "ass":
            _, path = autolabel.choose_subtitle_track(ep.source, pr, scratch, subtitle_track)
        else:
            _, path = autolabel.choose_text_track(
                ep.source, pr, scratch, kind, ws.config.source_language, subtitle_track
            )
    return path, read_ground_truth(path, cast(Kind, kind), aliases, episode=eid)


def _pooled(
    tune: list[str],
    holdout: str,
    segs: dict[str, list[Segment]],
    assign: dict[tuple[str, int], int | None],
    gt: dict[str, list[GroundTruthLine]],
) -> tuple[dict[str, list[Segment]], dict[tuple[str, int], int | None], dict[str, list[Any]]]:
    """The tuning episodes as one episode (laid end to end, far apart) beside `holdout`."""
    p_segs: list[Segment] = []
    p_gt: list[GroundTruthLine] = []
    p_assign: dict[tuple[str, int], int | None] = {}
    for k, ep in enumerate(tune):
        off = k * _POOL_GAP_S
        for i, s in enumerate(segs[ep]):
            p_assign[(POOLED, len(p_segs))] = assign.get((ep, i))
            p_segs.append(
                dataclasses.replace(s, episode=POOLED, start=s.start + off, end=s.end + off)
            )
        p_gt += [dataclasses.replace(g, start=g.start + off, end=g.end + off) for g in gt[ep]]
    for i in range(len(segs[holdout])):
        p_assign[(holdout, i)] = assign.get((holdout, i))
    return (
        {POOLED: p_segs, holdout: list(segs[holdout])},
        p_assign,
        {POOLED: p_gt, holdout: gt[holdout]},
    )


def run_identity(
    root: Path,
    tune: list[str],
    holdout: str | None = None,
    *,
    gt_kind: str = "ass",
    segmentation: str = "diarization",
    embedder: str | None = None,
    embedder_options: dict[str, Any] | None = None,
    matcher: str = "centroid",
    matcher_options: dict[str, Any] | None = None,
    segment_embedder: str | None = None,
    segment_embedder_options: dict[str, Any] | None = None,
    segmentation_options: dict[str, Any] | None = None,
    threshold: float | None = None,
    min_new_seconds: float | None = None,
    min_speech: float | None = None,
    gt_file: Path | None = None,
    subtitle_track: int | None = None,
    aliases: dict[str, str] | None = None,
    force: bool = False,
    from_assignments: bool = False,
) -> IdentityRun:
    import numpy as np

    from fukikae.eval.autolabel import speech_turns
    from fukikae.eval.groundtruth import KINDS as GT_KINDS
    from fukikae.eval.groundtruth import check_alignment
    from fukikae.eval.groundtruth.base import load_alias_file
    from fukikae.eval.identity.cache import EmbeddingCache, ResidentEmbedder, cache_root
    from fukikae.eval.identity.metrics import MIN_SPEECH_S, nearest_centroid_accuracy, score
    from fukikae.eval.identity.segments import KINDS as SEG_KINDS
    from fukikae.eval.identity.segments import Kind as SegKind
    from fukikae.eval.identity.segments import Segment as Seg
    from fukikae.eval.identity.segments import segments as make_segments
    from fukikae.eval.identity.simulate import simulate
    from fukikae.registry.matchers import MATCHERS, make_matcher
    from fukikae.workspace import Workspace, episode_id

    if gt_kind not in GT_KINDS:
        raise ValueError(f"unknown --gt {gt_kind!r}; use one of {', '.join(GT_KINDS)}")
    if segmentation not in SEG_KINDS:
        raise ValueError(
            f"unknown segmentation {segmentation!r}; use one of {', '.join(SEG_KINDS)}"
        )
    if from_assignments:
        if segmentation != "diarization":
            raise ValueError("--from-assignments scores diarization clusters; use diarization")
        matcher = "assignments"
    elif matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; use one of {', '.join(sorted(MATCHERS))}")
    if not tune:
        raise ValueError("--tune names no episode")
    ids = [episode_id(Path(r)) for r in tune]
    if len(set(ids)) != len(ids):
        raise ValueError("--tune names an episode twice")
    hold_id = episode_id(Path(holdout)) if holdout else None
    if hold_id is not None and hold_id in ids:
        raise ValueError(f"held-out episode {hold_id} is also a tuning episode")
    if hold_id is not None and hasattr(MATCHERS.get(matcher), "fit_all"):
        raise ValueError(
            f"matcher {matcher} clusters every episode at once and keeps no registry to "
            "resolve a held-out episode against; score it without --holdout"
        )
    if segmentation != "asr" and (segment_embedder or segmentation_options):
        raise ValueError("--segment-embedder and --segmentation-option apply to asr only")
    m_opts = dict(matcher_options or {})
    for reserved in ("threshold", "min_new_seconds"):
        if reserved in m_opts:
            raise ValueError(f"--matcher-option {reserved}: use --{reserved.replace('_', '-')}")

    ws = Workspace(root)
    cfg = ws.config
    emb_cfg = embedder_config(cfg, embedder, embedder_options or {})
    seg_emb_cfg = (
        embedder_config(cfg, segment_embedder, segment_embedder_options or {})
        if segment_embedder
        else emb_cfg
    )
    thr = cfg.registry.match_threshold if threshold is None else float(threshold)
    min_new = cfg.registry.min_new_seconds if min_new_seconds is None else float(min_new_seconds)
    the_matcher: Any = None
    if not from_assignments:
        try:
            the_matcher = make_matcher(matcher, threshold=thr, min_new_seconds=min_new, **m_opts)
        except TypeError as e:
            raise ValueError(f"matcher {matcher}: {e}") from e
    merged_aliases = {**load_alias_file(ws.root / "labels-aliases.yaml"), **(aliases or {})}

    refs = list(tune) + ([holdout] if holdout else [])
    order = ids + ([hold_id] if hold_id else [])
    gt: dict[str, list[GroundTruthLine]] = {}
    segs: dict[str, list[Segment]] = {}
    infos: dict[str, EpisodeInfo] = {}
    cache = EmbeddingCache(cache_root(ws.root))
    embs: dict[str, np.ndarray] = {}

    def audio_of(eid: str) -> Path:
        audio = ws.work_dir(eid) / "01-separate" / "vocals_16k.wav"
        if not audio.is_file():
            raise FileNotFoundError(f"{eid}: no separated vocals at {audio}")
        return audio

    with ResidentEmbedder(cfg) as embed:
        for ref, eid in zip(refs, order, strict=True):
            work = ws.work_dir(eid)
            if not (work / "04-diarize").is_dir():
                raise FileNotFoundError(f"{eid}: no diarization under {work / '04-diarize'}")
            path, lines = _ground_truth(
                ws, ref, eid, gt_kind, gt_file, len(refs), subtitle_track, merged_aliases
            )
            lines, rep, applied = check_alignment(
                lines,
                speech_turns(work),
                force=force,
                label=f"{gt_kind} speaker lines in {eid} ({path.name})",
            )
            gt[eid] = lines

            def window_embed(spans: list[tuple[float, float]], eid: str = eid) -> np.ndarray:
                windows = [Seg(eid, a, b, None) for a, b in spans]
                return cache.get_or_compute(eid, windows, seg_emb_cfg, audio_of(eid), embed=embed)

            segs[eid] = make_segments(
                ws,
                eid,
                cast(SegKind, segmentation),
                lines,
                window_embed=window_embed if segmentation == "asr" else None,
                options=segmentation_options,
            )
            infos[eid] = EpisodeInfo(
                eid,
                str(path),
                len(lines),
                len(segs[eid]),
                sum(s.label is not None for s in segs[eid]),
                rep,
                applied,
            )
        for eid in order:
            embs[eid] = cache.get_or_compute(eid, segs[eid], emb_cfg, audio_of(eid), embed=embed)

    if from_assignments:
        from fukikae.artifacts import read_json
        from fukikae.models import Assignments

        assign = {}
        for eid in order:
            path = ws.work_dir(eid) / "05-characters" / "assignments.json"
            if not path.is_file():
                raise FileNotFoundError(f"{eid}: no assignments at {path}")
            a = read_json(path, Assignments)
            char = {c.cluster: c.character_id for c in a.clusters}
            counts: dict[str, int] = {}
            for c in a.clusters:
                counts[c.status] = counts.get(c.status, 0) + 1
            infos[eid].statuses = counts
            for i, s in enumerate(segs[eid]):
                assign[(eid, i)] = char.get(str(s.cluster))
    elif hasattr(the_matcher, "fit_all"):
        assign = simulate(order, the_matcher, segs, embs)
    else:
        rec = _Recording(the_matcher)
        assign = simulate(order, rec, segs, embs)
        for eid, counts in rec.statuses.items():
            infos[eid].statuses = counts

    floor = MIN_SPEECH_S if min_speech is None else float(min_speech)
    tune_score = score(
        {k: v for k, v in assign.items() if k[0] in ids},
        {e: segs[e] for e in ids},
        min_speech=floor,
        gt={e: gt[e] for e in ids},
    )

    def stacked(eps: list[str]) -> tuple[list[Segment], np.ndarray]:
        rows = [embs[e] for e in eps if len(segs[e])]
        return [s for e in eps for s in segs[e]], (
            np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        )

    tune_nc: dict[str, float] = {}
    if len(ids) >= 2:
        tune_nc = nearest_centroid_accuracy(*stacked(ids[:1]), *stacked(ids[1:]))
    hold_score = None
    hold_nc: dict[str, float] = {}
    if hold_id is not None:
        p_segs, p_assign, p_gt = _pooled(ids, hold_id, segs, assign, gt)
        hold_score = score(p_assign, p_segs, min_speech=floor, gt=p_gt, reference=[POOLED])
        hold_nc = nearest_centroid_accuracy(*stacked(ids), *stacked([hold_id]))
    settings = {
        "workspace": str(ws.root),
        "tune": ids,
        "holdout": hold_id,
        "gt": gt_kind,
        "segmentation": segmentation,
        "embedder": {"backend": emb_cfg.backend, "options": emb_cfg.options},
        "matcher": matcher,
        "matcher_options": m_opts,
        "threshold": thr,
        "min_new_seconds": min_new,
        "min_speech": floor,
    }
    if segmentation == "asr":
        settings["segment_embedder"] = {
            "backend": seg_emb_cfg.backend,
            "options": seg_emb_cfg.options,
        }
        settings["segmentation_options"] = dict(segmentation_options or {})
    return IdentityRun(
        settings,
        [infos[e] for e in order],
        tune_score,
        hold_score,
        hold_id,
        cache.hits,
        cache.misses,
        tune_nc,
        hold_nc,
    )


# ---- output ----


def _id(cid: int | None) -> str:
    return "-" if cid is None else str(cid)


def format_nc(nc: dict[str, float]) -> list[str]:
    from fukikae.eval.identity.metrics import OVERALL

    if not nc:
        return ["  nearest-centroid accuracy: nothing to score"]
    per = ", ".join(f"{n} {v:.0%}" for n, v in nc.items() if n != OVERALL)
    return [f"  nearest-centroid accuracy: {nc[OVERALL]:.0%} overall (speech-weighted); {per}"]


def format_score(
    s: IdentityScore, title: str, min_speech: float, nc: dict[str, float] | None = None
) -> str:
    rows = [title]
    for ep in s.episodes:
        rows.append(f"  {ep}")
        rows.append(f"    {'character':<20}{'speech':>8}{'id':>5}{'here':>7}{'share':>7}  bar")
        for c in s.characters:
            if ep not in c.by_episode:
                continue
            flag = ("pass" if c.passing else "FAIL") if c.recurring else ""
            rows.append(
                f"    {c.name[:19]:<20}{c.by_episode[ep]:>7.1f}s{_id(c.modal_id):>5}"
                f"{c.share_by_episode.get(ep, 0.0):>7.0%}{c.share:>7.0%}  {flag}"
            )
    rows.append("  character ids (purity)")
    for cid, p in sorted(s.purity_by_id.items(), key=lambda kv: -kv[1].speech):
        others = ", ".join(f"{n} {v:.0%}" for n, v in list(p.others.items())[:4])
        rows.append(
            f"    {cid:>4} {p.speech:>7.1f}s  {p.dominant[:19]:<20}{p.purity:>5.0%}  {others}"
        )
    rows.append(
        f"  recurring (>= {min_speech:g} s in each): {s.recurring}  passing: {s.passing}  "
        f"fraction: {s.fraction:.2f}"
    )
    if s.merges:
        rows.append("  merges: " + "; ".join(f"{a} + {b} (id {cid})" for a, b, cid in s.merges))
    else:
        rows.append("  merges: none")
    rows.append(
        f"  labeled in every episode: {s.labeled_recurring}  "
        f"speech-weighted on modal id: {s.weighted_share:.0%}  strict: {s.strict_fraction:.2f}"
    )
    rows.append(f"  coverage: {s.coverage:.0%}")
    if nc is not None:
        rows += format_nc(nc)
    rows.append(f"  verdict: {s.verdict}")
    return "\n".join(rows)


def format_run(r: IdentityRun) -> str:
    st = r.settings
    opts = ", ".join(f"{k}={v}" for k, v in st["embedder"]["options"].items())
    rows = [
        f"segmentation {st['segmentation']}, embedder {st['embedder']['backend']}"
        f"{f' ({opts})' if opts else ''}, matcher {st['matcher']}, threshold {st['threshold']:g}, "
        f"ground truth {st['gt']}",
        f"embeddings: {r.cache_hits} cached, {r.cache_misses} computed",
    ]
    for e in r.episodes:
        a = e.alignment
        align = (
            f"offset {a.best_offset_s:+.1f} s applied ({a.overlap_share:.0%} -> "
            f"{a.share_at_offset:.0%})"
            if e.offset_applied
            else f"alignment {a.overlap_share:.0%}"
        )
        status = ", ".join(f"{n} {k}" for k, n in sorted(e.statuses.items()))
        rows.append(
            f"{e.episode}: {e.lines} lines, {e.segments} segments ({e.labeled} labeled), "
            f"{align}{f'; clusters: {status}' if status else ''}"
        )
    floor = st["min_speech"]
    rows.append(
        format_score(
            r.tune,
            f"tune: {', '.join(st['tune'])}",
            floor,
            r.tune_nc if len(st["tune"]) >= 2 else None,
        )
    )
    if r.holdout is not None:
        title = (
            f"held out: {r.holdout_episode} (against the registry the tuning episodes built; "
            "ids from the tuning episodes, shares from the held-out speech)"
        )
        rows.append(format_score(r.holdout, title, floor, r.holdout_nc))
    return "\n".join(rows)


def to_json(r: IdentityRun) -> dict[str, Any]:
    def plain(x: Any) -> Any:
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {("none" if k is None else str(k)): plain(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [plain(v) for v in x]
        return x

    return {
        "settings": r.settings,
        "episodes": plain(r.episodes),
        "cache": {"hits": r.cache_hits, "misses": r.cache_misses},
        "tune": {
            "episodes": r.settings["tune"],
            "score": plain(r.tune),
            "nearest_centroid": r.tune_nc,
        },
        "holdout": None
        if r.holdout is None
        else {
            "episode": r.holdout_episode,
            "score": plain(r.holdout),
            "nearest_centroid": r.holdout_nc,
        },
    }


def identity_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    tune: str = typer.Option(
        ..., "--tune", help="Tuning episodes in registry order, e.g. S01E12,S01E13."
    ),
    holdout: str = typer.Option(
        None,
        "--holdout",
        help="Held-out episode, resolved against the registry all tuning episodes built "
        "(pooled into one registry, Ruling R7) and scored separately: each character's id is "
        "its modal id in the tuning episodes, and its share is the held-out speech on that id. "
        "A recurring character needs --min-speech in the tuning episodes together and in the "
        "held-out one.",
    ),
    gt: str = typer.Option(
        "ass", "--gt", help="Ground truth: ass (Name field), sdh (speaker tags) or colour."
    ),
    gt_file: Path = typer.Option(
        None, "--gt-file", help="Sidecar subtitle file, or a directory matched by SxxEyy."
    ),
    subtitle_track: int = typer.Option(
        None, "--subtitle-track", help="Absolute stream index of the ground-truth track."
    ),
    alias: list[str] = typer.Option(
        None, "--alias", help="FROM=TO speaker-name alias; repeat. Also labels-aliases.yaml."
    ),
    segmentation: str = typer.Option(
        "diarization",
        "--segmentation",
        help="diarization (clean spans), subtitles, or asr (transcript sentences split where "
        "word-window voices differ).",
    ),
    segment_embedder: str = typer.Option(
        None,
        "--segment-embedder",
        help="asr only: embed backend for the word windows (default: --embedder's).",
    ),
    segmentation_option: list[str] = typer.Option(
        None,
        "--segmentation-option",
        help="asr only: k=v, one of window (s, 1.5), split (cosine distance, 0.8), "
        "max_gap (s, 1.0); repeat.",
    ),
    embedder: str = typer.Option(
        None,
        "--embedder",
        help="Embed backend (default: the workspace's [embed], else its profile's).",
    ),
    embedder_option: list[str] = typer.Option(
        None, "--embedder-option", help="k=v embedder option; repeat."
    ),
    matcher: str = typer.Option(
        "centroid",
        "--matcher",
        help="Matcher: centroid, knn (k), multi-centroid (max_centroids, min_rows), lda (dims; "
        "fit on earlier episodes' labels), joint (seed; all episodes at once, no --holdout).",
    ),
    matcher_option: list[str] = typer.Option(
        None, "--matcher-option", help="k=v matcher option, e.g. k=5; repeat."
    ),
    min_speech: float = typer.Option(
        5.0,
        "--min-speech",
        min=0.0,
        help="Seconds of labeled speech a character needs in each episode compared to recur.",
    ),
    threshold: float = typer.Option(
        None, "--threshold", help="Match threshold (default: the workspace's registry setting)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Score even when the subtitles do not line up with the speech."
    ),
    no_wait: bool = typer.Option(False, "--no-wait", help="Fail at once if the GPU lacks VRAM."),
    json_out: Path = typer.Option(None, "--json", help="Also write the full result as JSON."),
    from_assignments: bool = typer.Option(
        False,
        "--from-assignments",
        help="Score the pipeline's 05-characters/assignments.json (the characters stage or "
        "`registry resolve`) instead of running a matcher; diarization segmentation only.",
    ),
) -> None:
    """Score cross-episode speaker identity against subtitle ground truth, offline."""
    from fukikae import runctx
    from fukikae.artifacts import atomic_path
    from fukikae.errors import user_errors

    runctx.vram_timeout_s = 0 if no_wait else None
    try:
        result = run_identity(
            workspace,
            [e.strip() for e in tune.split(",") if e.strip()],
            holdout.strip() if holdout else None,
            gt_kind=gt,
            segmentation=segmentation,
            embedder=embedder,
            embedder_options=parse_options(embedder_option),
            matcher=matcher,
            matcher_options=parse_options(matcher_option, "--matcher-option"),
            segment_embedder=segment_embedder,
            segmentation_options=parse_options(segmentation_option, "--segmentation-option"),
            threshold=threshold,
            min_speech=min_speech,
            gt_file=gt_file,
            subtitle_track=subtitle_track,
            aliases=parse_aliases(alias),
            force=force,
            from_assignments=from_assignments,
        )
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(format_run(result))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(json_out) as tmp:
            tmp.write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")
