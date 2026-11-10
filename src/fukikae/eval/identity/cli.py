"""`fukikae eval identity`: score cross-episode speaker identity against subtitle ground truth.

The tuning episodes are resolved in order against an in-memory registry (the first builds it,
each later one resolves against what came before) and scored together. A held-out episode is
resolved against the registry the tuning episodes built (Ruling R7) and scored separately:
each character's id is its modal id in the tuning episodes, and its share, purity, coverage
and merges come from the held-out speech alone (`metrics.score(reference=...)`). Its recurring
characters have at least `--min-speech` (default 5 s, Ruling R6) in the tuning episodes
together and in the held-out one.

Reads pipeline artifacts (the identity turns, ``01-separate/vocals_16k.wav``) and the
subtitles; writes only the embedding cache, extracted subtitle tracks under
``.fukikae/cache/subtitles/``, the optional ``--json`` file and the results ledger's rows
(one per split, unless ``--no-ledger``). Never touches the registry.
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
    from fukikae.eval.ledger import LedgerRow
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
    # `--ceiling` episodes: scored as the held-out one is, without `check_held_out` (their
    # names were read, so each is graded on its own input), ledger split `ceiling`
    ceiling: dict[str, IdentityScore] = field(default_factory=dict)
    ceiling_nc: dict[str, dict[str, float]] = field(default_factory=dict)


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
    colour_names: dict[str, str] | None = None,
    colour_anonymous: str | None = None,
) -> tuple[Path, list[GroundTruthLine]]:
    from fukikae.eval import autolabel
    from fukikae.eval.groundtruth import Kind, read_ground_truth
    from fukikae.eval.groundtruth.colour import Anonymous
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
    lines = read_ground_truth(
        path,
        cast(Kind, kind),
        aliases,
        episode=eid,
        colour_names=colour_names,
        anonymous=cast(Anonymous, colour_anonymous or "drop") if colour_names else None,
    )
    return path, lines


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
    colour_anonymous: str | None = None,
    ceiling: list[str] | None = None,
) -> IdentityRun:
    """Score ``tune`` and the held-out episode (see the module docstring). Each ``ceiling``
    episode is scored as a held-out one, resolved against the registry the tuning episodes
    built, but without `check_held_out`: protocol A's ceiling arm, whose names were read."""
    import copy

    import numpy as np

    from fukikae.eval.autolabel import speech_turns
    from fukikae.eval.groundtruth import KINDS as GT_KINDS
    from fukikae.eval.groundtruth import check_alignment
    from fukikae.eval.groundtruth.base import load_alias_file
    from fukikae.eval.groundtruth.colour import workspace_colour_map
    from fukikae.eval.identity.cache import EmbeddingCache, ResidentEmbedder, cache_root
    from fukikae.eval.identity.metrics import MIN_SPEECH_S, nearest_centroid_accuracy, score
    from fukikae.eval.identity.segments import KINDS as SEG_KINDS
    from fukikae.eval.identity.segments import Kind as SegKind
    from fukikae.eval.identity.segments import Segment as Seg
    from fukikae.eval.identity.segments import segments as make_segments
    from fukikae.eval.identity.simulate import simulate
    from fukikae.names.inputs import check_held_out
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
    ceil_refs = list(ceiling or [])
    ceil_ids = [episode_id(Path(r)) for r in ceil_refs]
    if len(set(ceil_ids)) != len(ceil_ids) or set(ceil_ids) & (set(ids) | {hold_id}):
        raise ValueError("an episode is named twice across --tune, --holdout and --ceiling")
    if (hold_id is not None or ceil_ids) and hasattr(MATCHERS.get(matcher), "fit_all"):
        raise ValueError(
            f"matcher {matcher} clusters every episode at once and keeps no registry to "
            "resolve a held-out episode against; score it without --holdout or --ceiling"
        )
    if segmentation != "asr" and (segment_embedder or segmentation_options):
        raise ValueError("--segment-embedder and --segmentation-option apply to asr only")
    m_opts = dict(matcher_options or {})
    for reserved in ("threshold", "min_new_seconds"):
        if reserved in m_opts:
            raise ValueError(f"--matcher-option {reserved}: use --{reserved.replace('_', '-')}")

    ws = Workspace(root)
    if hold_id is not None:
        check_held_out(ws.work_dir(hold_id), hold_id)
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
    colour_map = workspace_colour_map(ws.root, gt_kind, colour_anonymous)

    refs = list(tune) + ([holdout] if holdout else []) + ceil_refs
    extra = ([hold_id] if hold_id else []) + ceil_ids  # each resolved against tune alone
    order = ids + extra
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

    recorded: dict[str, Any] = {}
    if from_assignments:  # segment the turns the assignments label, sub-clusters included
        from fukikae.artifacts import read_json
        from fukikae.models import Assignments

        for eid in order:
            path = ws.work_dir(eid) / "05-characters" / "assignments.json"
            if not path.is_file():
                raise FileNotFoundError(f"{eid}: no assignments at {path}")
            recorded[eid] = read_json(path, Assignments)  # turns None (older): identity turns

    with ResidentEmbedder(cfg) as embed:
        for ref, eid in zip(refs, order, strict=True):
            work = ws.work_dir(eid)
            if not (work / "04-diarize").is_dir():
                raise FileNotFoundError(f"{eid}: no diarization under {work / '04-diarize'}")
            path, lines = _ground_truth(
                ws, ref, eid, gt_kind, gt_file, len(refs), subtitle_track, merged_aliases,
                colour_map, colour_anonymous,
            )  # fmt: skip
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
                turns=recorded[eid].turns if from_assignments else None,
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
        assign = {}
        for eid in order:
            a = recorded[eid]
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
        from fukikae.registry.matchers import RegistryState

        rec = _Recording(the_matcher)
        state = RegistryState()
        assign = simulate(ids, rec, segs, embs, state=state)
        for x in extra:  # never against another held-out or ceiling episode
            assign.update(simulate([x], rec, segs, embs, state=copy.deepcopy(state)))
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
    ceil_scores: dict[str, IdentityScore] = {}
    ceil_nc: dict[str, dict[str, float]] = {}
    for c in ceil_ids:
        p_segs, p_assign, p_gt = _pooled(ids, c, segs, assign, gt)
        ceil_scores[c] = score(p_assign, p_segs, min_speech=floor, gt=p_gt, reference=[POOLED])
        ceil_nc[c] = nearest_centroid_accuracy(*stacked(ids), *stacked([c]))
    settings = {
        "workspace": str(ws.root),
        "tune": ids,
        "holdout": hold_id,
        "ceiling": ceil_ids,
        "gt": gt_kind,
        "segmentation": segmentation,
        "embedder": {"backend": emb_cfg.backend, "options": emb_cfg.options},
        "matcher": matcher,
        "matcher_options": m_opts,
        "threshold": thr,
        "min_new_seconds": min_new,
        "min_speech": floor,
        "profile": cfg.profile,
    }
    if colour_map:
        settings["colour_map"] = colour_map
        settings["colour_anonymous"] = colour_anonymous or "drop"
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
        ceil_scores,
        ceil_nc,
    )


# ---- ledger ----

_NOT_OPTIONS = ("workspace", "tune", "holdout", "ceiling")


def identity_rows(run: IdentityRun, cfg: Config, test_set: str, tag: str | None) -> list[LedgerRow]:
    """Ledger rows for a run: one `tune` row, and a `holdout` row when it has one. Both
    carry the same options (the run's settings without the workspace and episodes, plus the
    profile's hash), so they share a matrix row."""
    from fukikae.config import BackendConfig
    from fukikae.eval.identity.metrics import OVERALL, clean_count
    from fukikae.eval.ledger import make_row, model_identity, options_hash

    st = run.settings
    options = {k: v for k, v in st.items() if k not in _NOT_OPTIONS}
    options.setdefault("profile", cfg.profile)
    options["profile_hash"] = options_hash(cfg.profile_raw)
    emb = st["embedder"]
    backend = str(emb["backend"])
    model, revision = model_identity("embed", BackendConfig(backend, dict(emb["options"])))

    def metrics(s: IdentityScore, nc: dict[str, float]) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            "fraction": s.fraction,
            "passing": s.passing,
            "recurring": s.recurring,
            "clean": clean_count(s),
            "merges": len(s.merges),
            "strict_fraction": s.strict_fraction,
            "weighted_share": s.weighted_share,
            "coverage": s.coverage,
            "verdict": s.verdict,
        }
        if OVERALL in nc:
            out["nc_overall"] = nc[OVERALL]
        return out

    splits: list[tuple[str, list[str], IdentityScore, dict[str, float]]] = [
        ("tune", list(st["tune"]), run.tune, run.tune_nc)
    ]
    if run.holdout is not None and run.holdout_episode is not None:
        splits.append(("holdout", [run.holdout_episode], run.holdout, run.holdout_nc))
    for c, s in run.ceiling.items():
        splits.append(("ceiling", [c], s, run.ceiling_nc.get(c, {})))
    return [
        make_row(
            stage="identity",
            backend=backend,
            model=model,
            revision=revision,
            options=options,
            language=cfg.source_language or "und",
            genre=cfg.series.genre,
            series=cfg.series.title,
            test_set=test_set,
            split=split,
            episodes=episodes,
            metrics=metrics(score, nc),
            detail={
                "merges": [list(m) for m in score.merges],
                "nc": dict(nc),
                # each recurring character's share of its speech on its modal id
                "characters": [
                    {
                        "name": c.name,
                        "modal_id": c.modal_id,
                        "share": round(c.share, 4),
                        "passing": c.passing,
                    }
                    for c in score.characters
                    if c.recurring
                ],
                **({"ceiling": True} if split == "ceiling" else {}),
            },
            workspace=st.get("workspace"),
            tag=tag,
        )
        for split, episodes, score, nc in splits
    ]


def write_ledger(rows_of: Any, ledger: Path | None) -> None:
    """Append the rows `rows_of()` builds; any failure is a warning, never an exit code,
    so a ledger problem never hides a result already printed."""
    from fukikae.eval.ledger import append_rows

    try:
        path = append_rows(rows_of(), ledger)
    except Exception as e:  # noqa: BLE001 - the result is out; the ledger is best effort
        typer.echo(f"warning: ledger not written: {e}", err=True)
        return
    typer.echo(f"ledger: appended to {path}", err=True)


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
    for c, s in r.ceiling.items():
        title = (
            f"ceiling: {c} (graded on its own names: resolved against the registry the tuning "
            "episodes built, ids from the tuning episodes, no held-out check)"
        )
        rows.append(format_score(s, title, floor, r.ceiling_nc.get(c)))
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
        "ceiling": [
            {"episode": c, "score": plain(s), "nearest_centroid": r.ceiling_nc.get(c, {})}
            for c, s in r.ceiling.items()
        ],
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
    ceiling: str = typer.Option(
        None,
        "--ceiling",
        help="Episodes scored as --holdout is, comma-separated, but without its check that "
        "they read no subtitle names: protocol A's ceiling arm, graded on its own input "
        "(ledger split ceiling). Each is resolved against the tuning episodes alone.",
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
    colour_anonymous: str = typer.Option(
        None,
        "--colour-anonymous",
        help="With --gt colour and a labels-colours.yaml: what to do with colours the map "
        "lacks. drop (default) excludes them; series keeps them as colour:<name>; episode as "
        "<name>@<episode>.",
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
    test_set: str = typer.Option(
        None,
        "--set",
        "--test-set",
        help="Ledger test-set name (default: the workspace directory's name).",
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
            colour_anonymous=colour_anonymous,
            force=force,
            from_assignments=from_assignments,
            ceiling=[e.strip() for e in ceiling.split(",") if e.strip()] if ceiling else None,
        )
    except user_errors() as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(format_run(result))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(json_out) as tmp:
            tmp.write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")
    if not no_ledger:
        from fukikae.workspace import Workspace

        name = test_set or Path(workspace).resolve().name
        write_ledger(lambda: identity_rows(result, Workspace(workspace).config, name, tag), ledger)
