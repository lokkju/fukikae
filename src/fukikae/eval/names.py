"""`fukikae eval names`: what subtitle speaker names (D27) bought, scored offline on CPU.

Two measures, per the spec's protocols:

*Cue attribution accuracy.* Every spoken dialogue cue of the ground-truth subtitle track with
one usable ground-truth name (the names covering half its length or more agree) is attributed
as the subtitle path would: ``cues.assign_subtitle_speakers`` over the episode's
``assignments.json``, with the name map from its name decisions and the names stage's hidden
lines (so a held-out episode, whose names stage read nothing, and a hidden line never see their
own names). A cue is correct when its character's modal ground-truth name over the tuning
episodes' speech (each diarize turn labelled by the name covering 60% of it, the harness's
convention) is the cue's name. Speech-weighted (cue seconds), per split: ``tune``,
``holdout`` (protocol A: every held-out episode passes ``check_held_out`` first) and
``hidden`` (protocol B: only the cues of lines ``[names] hide_fraction`` withheld; under
``use = "off"`` the same lines are drawn again from the ground-truth track, so the baseline
arm scores the same cues).

*Reference purity.* Per character, the share of the reference seconds ``character_references``
would pick (computed, not rendered) whose ground-truth label (60% cover) is the character's
modal name: Natsu cloned from Wendy's turns is a purity failure. Reported on the tune row.

Coverage (named share, sub-clusters, cut turns, dropped row seconds, unvoiced and hidden lines)
comes from each episode's ``names.json``. Rows go to the results ledger with ``stage =
"names"``, ``backend = "subtitle-names"`` and the resolved ``[names]`` table as options, one per
split. Reads the workspace only; the ground-truth tracks are cached under
``.fukikae/cache/subtitles/`` as the identity harness caches them.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from fukikae.config import Config
    from fukikae.eval.groundtruth.base import GroundTruthLine
    from fukikae.eval.ledger import LedgerRow
    from fukikae.models import Assignments, DubCue
    from fukikae.names.models import NamesResult
    from fukikae.workspace import Workspace

LABEL_SHARE = 0.6  # a span's ground-truth label: the name covering this much of it
CUE_SHARE = 0.5  # a cue's ground-truth names: those covering this much of it
BACKEND = "subtitle-names"


@dataclass
class NamesScore:
    """One split's scores. ``per_character`` is cue accuracy per ground-truth name (how much
    of that speaker's speech went to a character whose modal name it is);
    ``purity_per_character`` is keyed by the registry character (its name, else
    ``character <id>``). A measure with nothing to score is None."""

    cue_accuracy: float | None
    cue_seconds: float
    cues: int
    per_character: dict[str, float]
    reference_purity: float | None = None
    reference_seconds: float = 0.0
    purity_per_character: dict[str, float] = field(default_factory=dict)
    hidden_accuracy: float | None = None
    coverage: dict[str, float] = field(default_factory=dict)


@dataclass
class NamesRun:
    settings: dict[str, Any]
    tune: NamesScore
    holdout: NamesScore | None
    hidden: NamesScore | None
    modal: dict[int, str]  # character -> modal ground-truth name over the tuning episodes
    holdout_episodes: list[str] = field(default_factory=list)
    hidden_episodes: list[str] = field(default_factory=list)


def _cover(gt: Sequence[GroundTruthLine], start: float, end: float) -> dict[str, float]:
    from fukikae.eval.speakers import label_key

    out: dict[str, float] = defaultdict(float)
    for g in gt:
        ov = min(end, g.end) - max(start, g.start)
        if ov > 0:
            out[label_key(g.speaker)] += ov
    return out


def gt_label(gt: Sequence[GroundTruthLine], start: float, end: float) -> str | None:
    """The name covering `LABEL_SHARE` of the span, else None."""
    cover = _cover(gt, start, end)
    if not cover or end <= start:
        return None
    best = max(cover, key=lambda k: (cover[k], k))
    return best if cover[best] >= LABEL_SHARE * (end - start) else None


def cue_key(gt: Sequence[GroundTruthLine], start: float, end: float) -> str | None:
    """The cue's one usable ground-truth name: every name covering `CUE_SHARE` of it agrees;
    None for no name or two."""
    cover = _cover(gt, start, end)
    keys = {k for k, s in cover.items() if s >= CUE_SHARE * (end - start)}
    return keys.pop() if len(keys) == 1 else None


def _characters(asg: Assignments) -> dict[str, int]:
    return {
        a.cluster: a.character_id
        for a in asg.clusters
        if a.character_id is not None and a.status not in ("extra", "review")
    }


def modal_names(
    episodes: Sequence[tuple[Assignments, Sequence[GroundTruthLine]]],
) -> dict[int, str]:
    """Character -> the ground-truth name over most of its speech in these episodes."""
    seconds: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for asg, gt in episodes:
        who = _characters(asg)
        for t in asg.turns or []:
            cid = who.get(t.speaker)
            label = gt_label(gt, t.start, t.end) if cid is not None else None
            if cid is not None and label is not None:
                seconds[cid][label] += t.end - t.start
    return {cid: max(by, key=lambda k: (by[k], k)) for cid, by in sorted(seconds.items()) if by}


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    ov = min(a[1], b[1]) - max(a[0], b[0])
    return ov > 0 and ov >= 0.5 * min(a[1] - a[0], b[1] - b[0])


@dataclass
class _Episode:
    id: str
    work: Path
    asg: Assignments
    gt: list[GroundTruthLine]
    gt_path: Path
    names: NamesResult | None
    hidden: list[tuple[float, float]]  # hidden lines in cue time


@dataclass
class _Tally:
    right: float = 0.0
    total: float = 0.0
    n: int = 0
    by: dict[str, list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))

    def add(self, key: str, seconds: float, ok: bool) -> None:
        self.total += seconds
        self.n += 1
        self.by[key][1] += seconds
        if ok:
            self.right += seconds
            self.by[key][0] += seconds

    def accuracy(self) -> float | None:
        return round(self.right / self.total, 6) if self.total else None

    def per(self) -> dict[str, float]:
        return {k: round(r / t, 6) for k, (r, t) in sorted(self.by.items()) if t}


def attributed_cues(ws: Workspace, cfg: Config, e: _Episode) -> list[DubCue]:
    """The ground-truth track's cues attributed the subtitle path's way, honouring the names
    stage's hidden lines and its offset; a held-out or names-off episode gets no name map."""
    from fukikae.artifacts import read_json
    from fukikae.config import ConfigError
    from fukikae.cues import assign_subtitle_speakers, name_map
    from fukikae.models import Tags, Transcript
    from fukikae.stages.translate import TranslateStage, speech_cps
    from fukikae.subtitles import RULES, read_cues

    tr = e.work / "03-asr" / "transcript.json"
    tg = e.work / "04b-tag" / "tags.json"
    transcript = read_json(tr, Transcript) if tr.exists() else None
    tags = read_json(tg, Tags) if tg.exists() else None
    try:
        _, st = TranslateStage()._settings(ws.find_episode(e.id), cfg)
        max_speed, rules, cps = st["max_speed"], st["subs_filter"], st["cps"]
    except (ConfigError, FileNotFoundError, ValueError):
        max_speed, rules, cps = 20.0, list(RULES), speech_cps(cfg)
    nm = None
    hidden: list[tuple[float, float, str | None]] = []
    offset = 0.0
    n = e.names
    if n is not None and n.used and (n.source or {}).get("how") != "other-language":
        nm = name_map(e.asg) or None
        visible = {ln.id for ln in n.visible_lines()}
        hidden = [(ln.start, ln.end, ln.key) for ln in n.lines if ln.id not in visible]
        offset = n.offset_s
    return assign_subtitle_speakers(
        read_cues(e.gt_path), e.asg.turns or [], e.asg, transcript, tags, {},
        cfg.target_language, max_speed, rules, cps=cps, name_map=nm, hidden=hidden,
        offset_s=offset,
    )  # fmt: skip


def _hidden_spans(cfg: Config, eid: str, gt_path: Path, names: NamesResult | None) -> list:
    """The lines ``[names] hide_fraction`` withheld, in cue time: from ``names.json`` when the
    stage read names, else drawn again from the ground-truth track with the same seed (the
    baseline arm), else none."""
    from fukikae.names import detect
    from fukikae.stages.names import hide_lines

    if names is not None and names.used:
        visible = {ln.id for ln in names.visible_lines()}
        o = names.offset_s
        return [(ln.start + o, ln.end + o) for ln in names.lines if ln.id not in visible]
    s = cfg.names
    if s.hide_fraction <= 0 or eid in s.holdout_episodes:
        return []
    got = detect.read_track(gt_path)
    if isinstance(got, str):
        return []
    lines, _ = got
    ids = set(hide_lines(lines, s.hide_fraction, s.hide_seed, eid))
    return [(ln.start, ln.end) for ln in lines if ln.id in ids]


def _coverage(episodes: Sequence[_Episode]) -> dict[str, float]:
    used = [e.names for e in episodes if e.names is not None and e.names.used]
    out: dict[str, float] = {"episodes": len(episodes), "names_used": len(used)}
    if not used:
        return out
    out["named_share"] = round(sum(float(n.coverage.get("share", 0)) for n in used) / len(used), 6)
    for k in ("sub_clusters", "cut_turns", "hidden"):
        out[k] = sum(int(n.coverage.get(k, 0)) for n in used)
    out["dropped_row_seconds"] = round(sum(n.dropped_row_seconds for n in used), 3)
    out["unvoiced"] = sum(len(n.unvoiced) for n in used)
    return out


def _purity(
    ws: Workspace, episodes: Sequence[_Episode], modal: dict[int, str]
) -> tuple[float | None, float, dict[str, float]]:
    """(overall purity, reference seconds, per character) of the references
    ``character_references`` picks over these episodes for each character with a modal name."""
    from fukikae.cards.auto import character_references
    from fukikae.registry.store import Registry

    gt = {e.id: e.gt for e in episodes}
    order = [e.id for e in episodes]
    names: dict[int, str] = {}
    if ws.registry_path.exists():
        with Registry(ws.registry_path) as reg:
            names = {c.id: c.name for c in reg.characters() if c.name}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    loaded: dict = {}
    pure_all = total_all = 0.0
    per: dict[str, float] = {}
    for cid, name in sorted(modal.items()):
        got = character_references(ws, order, cid, exclusions, loaded=loaded)
        if got is None:
            continue
        e, picks, _ = got
        total = sum(p.end - p.start for p in picks)
        pure = sum(p.end - p.start for p in picks if gt_label(gt[e], p.start, p.end) == name)
        if total <= 0:
            continue
        per[names.get(cid) or f"character {cid}"] = round(pure / total, 6)
        pure_all += pure
        total_all += total
    return (round(pure_all / total_all, 6) if total_all else None), round(total_all, 3), per


def names_eval(
    root: Path,
    tune: Sequence[str],
    holdout: Sequence[str] | str | None = None,
    *,
    gt_kind: str = "ass",
    gt_file: Path | None = None,
    subtitle_track: int | None = None,
    aliases: dict[str, str] | None = None,
) -> NamesRun:
    """Score cue attribution and reference purity over ``tune`` and the held-out episodes."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import checked_names
    from fukikae.eval.groundtruth.base import load_alias_file
    from fukikae.eval.identity.cli import _ground_truth
    from fukikae.models import Assignments
    from fukikae.names.inputs import check_held_out, names_path
    from fukikae.names.models import NamesResult
    from fukikae.workspace import Workspace, episode_id

    if gt_kind != "ass":
        raise ValueError("eval names grades on the ASS Name field: use --gt ass")
    held = [holdout] if isinstance(holdout, str) else list(holdout or [])
    ids = [episode_id(Path(r)) for r in tune]
    hold_ids = [episode_id(Path(r)) for r in held]
    if not ids:
        raise ValueError("--tune names no episode")
    if len(set(ids + hold_ids)) != len(ids) + len(hold_ids):
        raise ValueError("an episode is named twice across --tune and --holdout")
    ws = Workspace(root)
    cfg = ws.config
    for h in sorted(set(hold_ids) | (set(ids) & set(cfg.names.holdout_episodes))):
        check_held_out(ws.work_dir(h), h)  # every held-out episode, before anything is read
    merged = {**load_alias_file(ws.root / "labels-aliases.yaml"), **(aliases or {})}
    refs = list(tune) + held
    order = ids + hold_ids
    eps: dict[str, _Episode] = {}
    for ref, eid in zip(refs, order, strict=True):
        work = ws.work_dir(eid)
        asg_path = work / "05-characters" / "assignments.json"
        if not asg_path.exists():
            raise ValueError(f"{eid} has no assignments.json; run it through characters first")
        asg = read_json(asg_path, Assignments)
        _, why = checked_names(work, asg)
        if why is not None:
            raise ValueError(
                f"{eid}: {why}; run the characters stage first "
                f"(`fukikae run {ws.root} {eid} --from characters`)"
            )
        path, gt = _ground_truth(ws, ref, eid, gt_kind, gt_file, len(order), subtitle_track, merged)
        npath = names_path(work)
        names = read_json(npath, NamesResult) if npath.exists() else None
        eps[eid] = _Episode(
            eid, work, asg, gt, path, names,
            _hidden_spans(cfg, eid, path, names),
        )  # fmt: skip
    modal = modal_names([(eps[e].asg, eps[e].gt) for e in ids])
    tallies = {"tune": _Tally(), "holdout": _Tally(), "hidden": _Tally()}
    hidden_eps: list[str] = []
    for eid in order:
        e = eps[eid]
        split = "tune" if eid in ids else "holdout"
        for c in attributed_cues(ws, cfg, e):
            if c.skip is not None or not c.text or c.end <= c.start:
                continue
            key = cue_key(e.gt, c.start, c.end)
            if key is None:
                continue
            ok = c.character_id is not None and modal.get(c.character_id) == key
            seconds = c.end - c.start
            tallies[split].add(key, seconds, ok)
            if any(_overlaps((c.start, c.end), h) for h in e.hidden):
                tallies["hidden"].add(key, seconds, ok)
                if eid not in hidden_eps:
                    hidden_eps.append(eid)
    purity, ref_seconds, per_purity = _purity(ws, [eps[e] for e in order], modal)
    t = tallies["tune"]
    hid = tallies["hidden"]
    tune_score = NamesScore(
        t.accuracy(), round(t.total, 3), t.n, t.per(), purity, ref_seconds, per_purity,
        hid.accuracy(), _coverage([eps[e] for e in ids]),
    )  # fmt: skip
    hold_score = None
    if hold_ids:
        h = tallies["holdout"]
        hold_score = NamesScore(
            h.accuracy(), round(h.total, 3), h.n, h.per(),
            coverage=_coverage([eps[e] for e in hold_ids]),
        )  # fmt: skip
    hidden_score = None
    if hid.n:
        hidden_score = NamesScore(hid.accuracy(), round(hid.total, 3), hid.n, hid.per())
    settings = {
        "workspace": str(ws.root.resolve()),
        "tune": ids,
        "holdout": hold_ids,
        "gt": gt_kind,
        "names": names_options(cfg),
    }
    return NamesRun(settings, tune_score, hold_score, hidden_score, modal, hold_ids, hidden_eps)


def names_options(cfg: Config) -> dict[str, Any]:
    """The resolved ``[names]`` table, as the ledger's options."""
    out = dataclasses.asdict(cfg.names)
    out["holdout_episodes"] = list(out["holdout_episodes"])
    return out


# ---- ledger ----


def names_rows(run: NamesRun, cfg: Config, test_set: str, tag: str | None) -> list[LedgerRow]:
    """One row per split the run scored: ``tune``, ``holdout`` and ``hidden``. Options are the
    resolved ``[names]`` table; metrics the split's cue accuracy and seconds, and on the tune
    row reference purity, hidden accuracy and coverage; detail the per-name and per-character
    breakdowns and the modal names."""
    from fukikae.eval.ledger import make_row

    def metrics(s: NamesScore) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {"cue_seconds": s.cue_seconds, "cues": s.cues}
        for k, v in (
            ("cue_accuracy", s.cue_accuracy),
            ("reference_purity", s.reference_purity),
            ("hidden_accuracy", s.hidden_accuracy),
        ):
            if v is not None:
                out[k] = v
        if s.reference_purity is not None:
            out["reference_seconds"] = s.reference_seconds
        out.update(s.coverage)
        return out

    splits: list[tuple[str, list[str], NamesScore]] = [("tune", run.settings["tune"], run.tune)]
    if run.holdout is not None:
        splits.append(("holdout", run.holdout_episodes, run.holdout))
    if run.hidden is not None:
        splits.append(("hidden", run.hidden_episodes, run.hidden))
    return [
        make_row(
            stage="names",
            backend=BACKEND,
            model=None,
            revision=None,
            options=run.settings["names"],
            language=cfg.source_language or "und",
            genre=cfg.series.genre,
            series=cfg.series.title,
            test_set=test_set,
            split=split,
            episodes=episodes,
            metrics=metrics(score),
            detail={
                "per_character": score.per_character,
                "purity_per_character": score.purity_per_character,
                "modal": {str(k): v for k, v in run.modal.items()},
                "gt": run.settings["gt"],
            },
            workspace=run.settings["workspace"],
            tag=tag,
        )
        for split, episodes, score in splits
    ]


# ---- output ----


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def format_run(r: NamesRun) -> str:
    lines = []
    for label, s in (("tune", r.tune), ("holdout", r.holdout), ("hidden", r.hidden)):
        if s is None:
            continue
        lines.append(
            f"{label}: cue accuracy {_pct(s.cue_accuracy)} over {s.cues} cues "
            f"({s.cue_seconds:.1f} s)"
        )
        if s.reference_purity is not None:
            lines.append(
                f"  reference purity {_pct(s.reference_purity)} of {s.reference_seconds:.1f} s"
            )
        if s.coverage:
            lines.append("  coverage: " + ", ".join(f"{k} {v}" for k, v in s.coverage.items()))
        for name, acc in sorted(s.per_character.items(), key=lambda kv: (kv[1], kv[0])):
            lines.append(f"  {name:<20} {acc:.1%}")
        for name, p in sorted(s.purity_per_character.items(), key=lambda kv: (kv[1], kv[0])):
            lines.append(f"  purity {name:<20} {p:.1%}")
    return "\n".join(lines)


def names_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    tune: str = typer.Option(
        ..., "--tune", help="Tuning episodes, e.g. S01E11,S01E12,S01E13 (their names may be used)."
    ),
    holdout: str = typer.Option(
        None,
        "--holdout",
        help="Held-out episodes, comma-separated (protocol A): each must have read no names "
        "([names] holdout_episodes), checked before scoring.",
    ),
    gt: str = typer.Option("ass", "--gt", help="Ground truth: ass (the Name field)."),
    gt_file: Path = typer.Option(
        None, "--gt-file", help="Sidecar subtitle file, or a directory matched by SxxEyy."
    ),
    subtitle_track: int = typer.Option(
        None, "--subtitle-track", help="Absolute stream index of the ground-truth track."
    ),
    alias: list[str] = typer.Option(
        None, "--alias", help="FROM=TO speaker-name alias; repeat. Also labels-aliases.yaml."
    ),
    json_out: Path = typer.Option(None, "--json", help="Also write the full result as JSON."),
    test_set: str = typer.Option(
        None,
        "--set",
        "--test-set",
        help="Ledger test-set name (default: the workspace directory's name).",
    ),
    tag: str = typer.Option(None, "--tag", help="Free-form ledger tag, e.g. names-baseline."),
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
    """Score subtitle-name attribution (cue accuracy) and clone-reference purity, offline."""
    from fukikae.artifacts import atomic_path
    from fukikae.errors import user_errors
    from fukikae.eval.identity.cli import parse_aliases, write_ledger
    from fukikae.names.inputs import HeldOutNamesError
    from fukikae.workspace import Workspace

    try:
        result = names_eval(
            workspace,
            [e.strip() for e in tune.split(",") if e.strip()],
            [e.strip() for e in holdout.split(",") if e.strip()] if holdout else None,
            gt_kind=gt,
            gt_file=gt_file,
            subtitle_track=subtitle_track,
            aliases=parse_aliases(alias),
        )
    except (*user_errors(), ValueError, HeldOutNamesError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(format_run(result))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(json_out) as tmp:
            tmp.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
    if not no_ledger:
        name = test_set or Path(workspace).resolve().name
        write_ledger(lambda: names_rows(result, Workspace(workspace).config, name, tag), ledger)
