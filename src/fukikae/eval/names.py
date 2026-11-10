"""`fukikae eval names`: what subtitle speaker names (D27) bought, scored offline on CPU.

Two measures, per the spec's protocols:

*Cue attribution accuracy.* Every spoken dialogue cue of the ground-truth subtitle track with
one usable ground-truth name (the names covering half its length or more agree) is attributed
as the subtitle path would: ``cues.assign_subtitle_speakers`` over the episode's
``assignments.json``, with the name map from its name decisions and the names stage's hidden
lines (so a held-out episode, whose names stage read nothing, and a hidden line never see their
own names). A cue is correct when its character's modal ground-truth name is the cue's name.
The modal map comes from the tuning episodes' speech alone (each diarize turn labelled by the
name covering 60% of it, the harness's convention), leaving out turns that overlap a hidden
line, so the lines the hidden split scores never choose the name they are scored against.
Several characters can share one modal name (an over-split registry): each of them scores as
that name, so ``characters_per_name`` is reported beside the accuracy.

Speech-weighted (cue seconds), per split:

- ``tune``: the tuning episodes. Where their names were used, the run is graded on the names
  it read, so the row is labelled a ceiling (``detail.ceiling``).
- ``holdout`` (protocol A): every held-out episode passes ``check_held_out`` first (no names in
  its ``names.json`` or its assignments).
- ``ceiling``: episodes the operator names with ``--ceiling``, scored as held out but without
  that check, against the tuning episodes' modal map: protocol A's ceiling arm, graded on its
  own input and labelled so.
- ``hidden`` (protocol B): only the cues of lines ``[names] hide_fraction`` withheld (a cue
  counts when a hidden line of its own key overlaps it, `cues._hidden`'s test); under
  ``use = "off"`` the same lines are drawn again from the ground-truth track, so the baseline
  arm scores the same cues. When ``names.json`` records its track, the lines it hid must be the
  lines the ground-truth track gives the same draw, or the run refuses (the two arms would
  score different cues).

*Reference purity.* Per character with a modal name, the references ``character_references``
would pick over the split's episodes (computed, not rendered), measured in ground-truth
seconds: ``pure`` seconds lie under lines of the character's modal name only, ``foreign``
seconds under lines of another name (a second where both overlap counts as foreign: the clip
carries another voice), ``unverifiable`` seconds under no line. Purity is pure / (pure +
foreign); the unverifiable share of all reference seconds is reported beside it. Natsu cloned
from Wendy's turns is a purity failure.

Ground-truth lines are compared with turns and references only after
``groundtruth.check_alignment`` against the episode's identity turns (the offset the identity
harness applies; ``--force`` scores a track that doesn't line up); cues and lines are compared
in subtitle time. Cues are attributed to turns moved by the names stage's offset where names
were used, else by that alignment shift (Fairy Tail's are all 0.0).

Coverage (named share, sub-clusters, cut turns, dropped row seconds, unvoiced and hidden lines)
comes from each episode's ``names.json``. Rows go to the results ledger with ``stage =
"names"``, ``backend = "subtitle-names"`` and the resolved ``[names]`` table as options, one per
split. Reads the workspace only; the ground-truth tracks are cached under
``.fukikae/cache/subtitles/`` as the identity harness caches them.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
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
HIDDEN_MATCH_S = 0.002  # names.json's hidden lines against the ground-truth track's draw
BACKEND = "subtitle-names"
SPLIT_ORDER = ("tune", "holdout", "ceiling", "hidden")

type Spans = list[tuple[float, float]]
type KeyedSpans = list[tuple[float, float, str | None]]  # (start, end, name key)


@dataclass
class NamesScore:
    """One split's scores. ``per_character`` is cue accuracy per ground-truth name (how much
    of that speaker's speech went to a character whose modal name it is);
    ``purity_per_character`` and ``purity_seconds_per_character`` are keyed by the registry
    character (its name, else ``character <id>``). A measure with nothing to score is None.
    ``ceiling`` says the split was graded on names its run read."""

    cue_accuracy: float | None
    cue_seconds: float
    cues: int
    per_character: dict[str, float]
    reference_purity: float | None = None
    reference_seconds: float = 0.0
    pure_seconds: float = 0.0
    foreign_seconds: float = 0.0
    unverifiable_seconds: float = 0.0
    unverifiable_share: float | None = None
    purity_per_character: dict[str, float] = field(default_factory=dict)
    purity_seconds_per_character: dict[str, dict[str, float]] = field(default_factory=dict)
    hidden_accuracy: float | None = None
    coverage: dict[str, float] = field(default_factory=dict)
    ceiling: bool = False


@dataclass
class NamesRun:
    settings: dict[str, Any]
    tune: NamesScore
    holdout: NamesScore | None
    hidden: NamesScore | None
    modal: dict[int, str]  # character -> modal ground-truth name over the tuning episodes
    holdout_episodes: list[str] = field(default_factory=list)
    hidden_episodes: list[str] = field(default_factory=list)
    ceiling: NamesScore | None = None
    ceiling_episodes: list[str] = field(default_factory=list)
    characters_per_name: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


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


def reference_split(
    gt: Sequence[GroundTruthLine], start: float, end: float, name: str
) -> tuple[float, float, float]:
    """(pure, foreign, unverifiable) seconds of the span: under lines of ``name`` only, under a
    line of another name (with or without ``name``'s), under no line."""
    from fukikae.eval.speakers import label_key

    spans = [
        (max(start, g.start), min(end, g.end), label_key(g.speaker))
        for g in gt
        if min(end, g.end) > max(start, g.start)
    ]
    edges = sorted({start, end, *(a for a, _, _ in spans), *(b for _, b, _ in spans)})
    pure = foreign = unverifiable = 0.0
    for a, b in zip(edges, edges[1:], strict=False):
        if b <= a:
            continue
        keys = {k for x, y, k in spans if x <= a and y >= b}
        if keys - {name}:
            foreign += b - a
        elif keys:
            pure += b - a
        else:
            unverifiable += b - a
    return pure, foreign, unverifiable


def _characters(asg: Assignments) -> dict[str, int]:
    return {
        a.cluster: a.character_id
        for a in asg.clusters
        if a.character_id is not None and a.status not in ("extra", "review")
    }


def modal_names(
    episodes: Sequence[tuple[Assignments, Sequence[GroundTruthLine]]],
    hidden: Sequence[Spans] | None = None,
) -> dict[int, str]:
    """Character -> the ground-truth name over most of its speech in these episodes, leaving
    out turns that overlap one of the episode's ``hidden`` spans (same time base as the
    turns): those are the lines the hidden split scores."""
    seconds: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for k, (asg, gt) in enumerate(episodes):
        who = _characters(asg)
        skip = hidden[k] if hidden is not None else []
        for t in asg.turns or []:
            cid = who.get(t.speaker)
            if cid is None or any(min(t.end, b) > max(t.start, a) for a, b in skip):
                continue
            label = gt_label(gt, t.start, t.end)
            if label is not None:
                seconds[cid][label] += t.end - t.start
    return {cid: max(by, key=lambda k: (by[k], k)) for cid, by in sorted(seconds.items()) if by}


def characters_per_name(modal: dict[int, str]) -> dict[str, int]:
    """Modal name -> how many characters it stands for (more than one: an over-split)."""
    return dict(sorted(Counter(modal.values()).items()))


@dataclass
class _Episode:
    id: str
    work: Path
    asg: Assignments
    gt: list[GroundTruthLine]  # subtitle (cue) time
    gt_path: Path
    names: NamesResult | None
    hidden: KeyedSpans  # hidden lines in cue time, with their keys
    shift: float = 0.0  # the alignment offset: turn time = cue time - shift
    gt_audio: list[GroundTruthLine] = field(default_factory=list)  # in turn time

    @property
    def hidden_audio(self) -> Spans:
        return [(a - self.shift, b - self.shift) for a, b, _ in self.hidden]

    @property
    def names_used(self) -> bool:
        return self.names is not None and self.names.used


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


def attributed_cues(
    ws: Workspace, cfg: Config, e: _Episode, warnings: list[str] | None = None
) -> list[DubCue]:
    """The ground-truth track's cues attributed the subtitle path's way, honouring the names
    stage's hidden lines and its offset; a held-out or names-off episode gets no name map and
    its cues are moved back by the alignment shift instead (the subtitle path would use none,
    but every arm should be graded on aligned turns, so a names arm doesn't win by its
    offset alone).
    The subtitle path's settings come from ``TranslateStage.settings``; when they can't be
    read, the stage's defaults stand in and a line goes to ``warnings``."""
    from fukikae.artifacts import read_json
    from fukikae.config import ConfigError
    from fukikae.cues import assign_subtitle_speakers, name_map
    from fukikae.models import Tags, Transcript
    from fukikae.stages.translate import DEFAULT_MAX_SPEED, TranslateStage, speech_cps
    from fukikae.subtitles import RULES, read_cues

    tr = e.work / "03-asr" / "transcript.json"
    tg = e.work / "04b-tag" / "tags.json"
    transcript = read_json(tr, Transcript) if tr.exists() else None
    tags = read_json(tg, Tags) if tg.exists() else None
    try:
        st = TranslateStage().settings(ws.find_episode(e.id), cfg)
        max_speed, rules, cps = st["max_speed"], st["subs_filter"], st["cps"]
    except (ConfigError, FileNotFoundError, ValueError) as err:
        max_speed, rules, cps = DEFAULT_MAX_SPEED, list(RULES), speech_cps(cfg)
        if warnings is not None:
            warnings.append(
                f"{e.id}: the translate settings can't be read ({err}); cues are attributed "
                f"with the stage's defaults (max_speed {DEFAULT_MAX_SPEED:g}, every subtitle "
                "filter rule)"
            )
    nm = None
    hidden: list[tuple[float, float, str | None]] = []
    offset = e.shift
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


def _track_hidden(cfg: Config, eid: str, gt_path: Path) -> KeyedSpans | None:
    """The lines ``[names] hide_fraction`` draws from the ground-truth track, in cue time, with
    their keys; None when the track can't carry names."""
    from fukikae.names import detect
    from fukikae.stages.names import hide_lines

    s = cfg.names
    got = detect.read_track(gt_path)
    if isinstance(got, str):
        return None
    lines, _ = got
    ids = set(hide_lines(lines, s.hide_fraction, s.hide_seed, eid))
    return sorted(((ln.start, ln.end, ln.key) for ln in lines if ln.id in ids), key=_span_order)


def _span_order(x: tuple[float, float, str | None]) -> tuple[float, float, str]:
    return x[0], x[1], x[2] or ""


def _hidden_spans(cfg: Config, eid: str, gt_path: Path, names: NamesResult | None) -> KeyedSpans:
    """The lines ``[names] hide_fraction`` withheld, in cue time, with their keys (a cue is
    hidden only by a line of its own key, as `cues._hidden` has it): from ``names.json`` when the
    stage read names, else drawn again from the ground-truth track with the same seed (the
    baseline arm), else none.

    When ``names.json`` records the track or file it read and hid lines, the ground-truth track must
    give the same draw (`_track_hidden`), so the names arm and the off arm score the same
    lines: a ValueError otherwise."""
    if names is not None and names.used:
        visible = {ln.id for ln in names.visible_lines()}
        o = names.offset_s
        out = sorted(
            ((ln.start + o, ln.end + o, ln.key) for ln in names.lines if ln.id not in visible),
            key=_span_order,
        )
        src = names.source or {}
        if out and (src.get("track") is not None or src.get("file")):
            drawn = _track_hidden(cfg, eid, gt_path) or []
            same = len(drawn) == len(out) and all(
                abs(a - c) <= HIDDEN_MATCH_S and abs(b - d) <= HIDDEN_MATCH_S
                for (a, b, _), (c, d, _) in zip(out, drawn, strict=True)
            )
            if not same:
                where = (
                    f"file {Path(str(src['file'])).name}"
                    if src.get("file")
                    else f"track {src.get('track')}"
                )
                raise ValueError(
                    f"{eid}: the names stage hid {len(out)} lines of {src.get('how')} {where}, "
                    f"but the same draw on the ground-truth track {gt_path.name} gives "
                    f'{len(drawn)} other lines: a use = "off" arm would score other cues. '
                    "Grade with the track the names stage read (--subtitle-track or --gt-file)"
                )
        return out
    s = cfg.names
    if s.hide_fraction <= 0 or eid in s.holdout_episodes:
        return []
    return _track_hidden(cfg, eid, gt_path) or []


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


@dataclass
class _Purity:
    pure: float = 0.0
    foreign: float = 0.0
    unverifiable: float = 0.0

    def add(self, other: tuple[float, float, float]) -> None:
        self.pure += other[0]
        self.foreign += other[1]
        self.unverifiable += other[2]

    @property
    def total(self) -> float:
        return self.pure + self.foreign + self.unverifiable

    def purity(self) -> float | None:
        verified = self.pure + self.foreign
        return round(self.pure / verified, 6) if verified > 0 else None

    def unverifiable_share(self) -> float | None:
        return round(self.unverifiable / self.total, 6) if self.total > 0 else None

    def seconds(self) -> dict[str, float]:
        out = {
            "pure_seconds": round(self.pure, 3),
            "foreign_seconds": round(self.foreign, 3),
            "unverifiable_seconds": round(self.unverifiable, 3),
        }
        share = self.unverifiable_share()
        if share is not None:
            out["unverifiable_share"] = share
        return out


def _purity(
    ws: Workspace,
    episodes: Sequence[_Episode],
    modal: dict[int, str],
    loaded: dict[str, Any] | None = None,
) -> tuple[_Purity, dict[str, _Purity]]:
    """(all characters, per character) purity seconds of the references
    ``character_references`` picks over these episodes for each character with a modal name,
    against each episode's aligned ground truth."""
    from fukikae.cards.auto import character_references
    from fukikae.registry.store import Registry

    gt = {e.id: e.gt_audio for e in episodes}
    order = [e.id for e in episodes]
    names: dict[int, str] = {}
    if ws.registry_path.exists():
        with Registry(ws.registry_path) as reg:
            names = {c.id: c.name for c in reg.characters() if c.name}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    loaded = loaded if loaded is not None else {}
    overall = _Purity()
    per: dict[str, _Purity] = {}
    if not order:
        return overall, per
    for cid, name in sorted(modal.items()):
        got = character_references(ws, order, cid, exclusions, loaded=loaded)
        if got is None:
            continue
        e, picks, _ = got
        mine = _Purity()
        for p in picks:
            mine.add(reference_split(gt[e], p.start, p.end, name))
        if mine.total <= 0:
            continue
        per[names.get(cid) or f"character {cid}"] = mine
        overall.add((mine.pure, mine.foreign, mine.unverifiable))
    return overall, per


def _score(
    tally: _Tally,
    purity: tuple[_Purity, dict[str, _Purity]],
    *,
    coverage: dict[str, float] | None = None,
    hidden_accuracy: float | None = None,
    ceiling: bool = False,
) -> NamesScore:
    overall, per = purity
    return NamesScore(
        cue_accuracy=tally.accuracy(),
        cue_seconds=round(tally.total, 3),
        cues=tally.n,
        per_character=tally.per(),
        reference_purity=overall.purity(),
        reference_seconds=round(overall.total, 3),
        pure_seconds=round(overall.pure, 3),
        foreign_seconds=round(overall.foreign, 3),
        unverifiable_seconds=round(overall.unverifiable, 3),
        unverifiable_share=overall.unverifiable_share(),
        purity_per_character={
            k: p for k, v in sorted(per.items()) if (p := v.purity()) is not None
        },
        purity_seconds_per_character={k: v.seconds() for k, v in sorted(per.items())},
        hidden_accuracy=hidden_accuracy,
        coverage=coverage or {},
        ceiling=ceiling,
    )


def _ids(refs: Sequence[str] | str | None) -> tuple[list[str], list[str]]:
    from fukikae.workspace import episode_id

    given = [refs] if isinstance(refs, str) else list(refs or [])
    return given, [episode_id(Path(r)) for r in given]


def names_eval(
    root: Path,
    tune: Sequence[str],
    holdout: Sequence[str] | str | None = None,
    *,
    ceiling: Sequence[str] | str | None = None,
    gt_kind: str = "ass",
    gt_file: Path | None = None,
    subtitle_track: int | None = None,
    aliases: dict[str, str] | None = None,
    force: bool = False,
) -> NamesRun:
    """Score cue attribution and reference purity over ``tune``, the held-out episodes and
    the ``ceiling`` episodes (held out without `check_held_out`, labelled a ceiling)."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import checked_names
    from fukikae.cues import _hidden, merge_layers
    from fukikae.eval.autolabel import speech_turns
    from fukikae.eval.groundtruth import check_alignment
    from fukikae.eval.groundtruth.base import load_alias_file
    from fukikae.eval.identity.cli import _ground_truth
    from fukikae.models import Assignments
    from fukikae.names.inputs import check_held_out, names_path
    from fukikae.names.models import NamesResult
    from fukikae.names.normalize import parse_name
    from fukikae.subtitles import read_cues
    from fukikae.workspace import Workspace

    if gt_kind != "ass":
        raise ValueError("eval names grades on the ASS Name field: use --gt ass")
    tune_refs, ids = _ids(tune)
    held, hold_ids = _ids(holdout)
    ceil_refs, ceil_ids = _ids(ceiling)
    if not ids:
        raise ValueError("--tune names no episode")
    every = ids + hold_ids + ceil_ids
    if len(set(every)) != len(every):
        raise ValueError("an episode is named twice across --tune, --holdout and --ceiling")
    ws = Workspace(root)
    cfg = ws.config
    for h in sorted(set(hold_ids) | (set(ids) & set(cfg.names.holdout_episodes))):
        check_held_out(ws.work_dir(h), h)  # every held-out episode, before anything is read
    merged = {**load_alias_file(ws.root / "labels-aliases.yaml"), **(aliases or {})}
    refs = tune_refs + held + ceil_refs
    eps: dict[str, _Episode] = {}
    for ref, eid in zip(refs, every, strict=True):
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
        path, gt = _ground_truth(ws, ref, eid, gt_kind, gt_file, len(every), subtitle_track, merged)
        npath = names_path(work)
        names = read_json(npath, NamesResult) if npath.exists() else None
        turns = asg.turns if asg.turns is not None else speech_turns(work)  # identity turns
        label = f"{gt_kind} speaker lines in {eid} ({path.name})"
        aligned, rep, applied = check_alignment(gt, turns, force=force, label=label)
        eps[eid] = _Episode(
            eid, work, asg, gt, path, names, _hidden_spans(cfg, eid, path, names),
            shift=rep.best_offset_s if applied else 0.0, gt_audio=aligned,
        )  # fmt: skip
    modal = modal_names(
        [(eps[e].asg, eps[e].gt_audio) for e in ids], [eps[e].hidden_audio for e in ids]
    )
    warnings: list[str] = []
    tallies = {k: _Tally() for k in SPLIT_ORDER}
    hidden_eps: list[str] = []
    for eid in every:
        e = eps[eid]
        split = "tune" if eid in ids else "holdout" if eid in hold_ids else "ceiling"
        raw = merge_layers(read_cues(e.gt_path))  # one per attributed cue, in order
        for c, rc in zip(attributed_cues(ws, cfg, e, warnings), raw, strict=True):
            if c.skip is not None or not c.text or c.end <= c.start:
                continue
            key = cue_key(e.gt, c.start, c.end)
            if key is None:
                continue
            ok = c.character_id is not None and modal.get(c.character_id) == key
            seconds = c.end - c.start
            tallies[split].add(key, seconds, ok)
            own = parse_name(rc.name).key if rc.name else None
            if _hidden(rc, own, e.hidden, 0.0):  # the subtitle path's test; both in cue time
                tallies["hidden"].add(key, seconds, ok)
                if eid not in hidden_eps:
                    hidden_eps.append(eid)
    loaded: dict[str, Any] = {}

    def purity(split_ids: Sequence[str]) -> tuple[_Purity, dict[str, _Purity]]:
        return _purity(ws, [eps[e] for e in split_ids], modal, loaded)

    hid = tallies["hidden"]
    tune_score = _score(
        tallies["tune"], purity(ids), coverage=_coverage([eps[e] for e in ids]),
        hidden_accuracy=hid.accuracy(), ceiling=any(eps[e].names_used for e in ids),
    )  # fmt: skip
    hold_score = None
    if hold_ids:
        hold_score = _score(
            tallies["holdout"], purity(hold_ids), coverage=_coverage([eps[e] for e in hold_ids])
        )
    ceil_score = None
    if ceil_ids:
        ceil_score = _score(
            tallies["ceiling"], purity(ceil_ids),
            coverage=_coverage([eps[e] for e in ceil_ids]), ceiling=True,
        )  # fmt: skip
    hidden_score = _score(hid, purity(hidden_eps)) if hid.n else None
    settings = {
        "workspace": str(ws.root.resolve()),
        "tune": ids,
        "holdout": hold_ids,
        "ceiling": ceil_ids,
        "gt": gt_kind,
        "names": names_options(cfg),
    }
    return NamesRun(
        settings=settings,
        tune=tune_score,
        holdout=hold_score,
        hidden=hidden_score,
        modal=modal,
        holdout_episodes=hold_ids,
        hidden_episodes=hidden_eps,
        ceiling=ceil_score,
        ceiling_episodes=ceil_ids,
        characters_per_name=characters_per_name(modal),
        warnings=warnings,
    )


def names_options(cfg: Config) -> dict[str, Any]:
    """The resolved ``[names]`` table, as the ledger's options."""
    out = dataclasses.asdict(cfg.names)
    out["holdout_episodes"] = list(out["holdout_episodes"])
    return out


def _splits(run: NamesRun) -> list[tuple[str, list[str], NamesScore]]:
    out: list[tuple[str, list[str], NamesScore]] = [("tune", run.settings["tune"], run.tune)]
    if run.holdout is not None:
        out.append(("holdout", run.holdout_episodes, run.holdout))
    if run.ceiling is not None:
        out.append(("ceiling", run.ceiling_episodes, run.ceiling))
    if run.hidden is not None:
        out.append(("hidden", run.hidden_episodes, run.hidden))
    return out


# ---- ledger ----


def names_rows(run: NamesRun, cfg: Config, test_set: str, tag: str | None) -> list[LedgerRow]:
    """One row per split the run scored: ``tune``, ``holdout``, ``ceiling`` and ``hidden``.
    Options are the resolved ``[names]`` table; metrics the split's cue accuracy and seconds,
    its reference purity with pure, foreign and unverifiable seconds and the unverifiable
    share, the most characters any modal name stands for, and on the tune row hidden accuracy
    and coverage; detail the per-name and per-character breakdowns, the modal names, the
    characters per name and whether the split is a ceiling (graded on names its run read)."""
    from fukikae.eval.ledger import make_row

    per_name = run.characters_per_name

    def metrics(s: NamesScore) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {"cue_seconds": s.cue_seconds, "cues": s.cues}
        for k, v in (
            ("cue_accuracy", s.cue_accuracy),
            ("reference_purity", s.reference_purity),
            ("unverifiable_share", s.unverifiable_share),
            ("hidden_accuracy", s.hidden_accuracy),
        ):
            if v is not None:
                out[k] = v
        if s.reference_seconds > 0:
            out["reference_seconds"] = s.reference_seconds
            out["pure_seconds"] = s.pure_seconds
            out["foreign_seconds"] = s.foreign_seconds
            out["unverifiable_seconds"] = s.unverifiable_seconds
        if per_name:
            out["max_characters_per_name"] = max(per_name.values())
        out.update(s.coverage)
        return out

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
                "ceiling": score.ceiling,
                "per_character": score.per_character,
                "purity_per_character": score.purity_per_character,
                "purity_seconds_per_character": score.purity_seconds_per_character,
                "modal": {str(k): v for k, v in run.modal.items()},
                "characters_per_name": per_name,
                "gt": run.settings["gt"],
                "hide_fraction": run.settings["names"]["hide_fraction"],
                "hide_seed": run.settings["names"]["hide_seed"],
            },
            workspace=run.settings["workspace"],
            tag=tag,
        )
        for split, episodes, score in _splits(run)
    ]


# ---- output ----


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


_TITLES = {
    "tune": "tune",
    "holdout": "holdout",
    "ceiling": "ceiling (graded on its own names)",
    "hidden": "hidden",
}


def format_run(r: NamesRun) -> str:
    lines = []
    for label, episodes, s in _splits(r):
        title = _TITLES[label]
        if label == "tune" and s.ceiling:
            title = "tune (ceiling: graded on the names its run read)"
        lines.append(
            f"{title} {', '.join(episodes)}: cue accuracy {_pct(s.cue_accuracy)} over {s.cues} "
            f"cues ({s.cue_seconds:.1f} s)"
        )
        if label == "tune" and r.characters_per_name:
            shared = {n: k for n, k in r.characters_per_name.items() if k > 1}
            lines.append(
                f"  characters per name: {len(r.modal)} characters on "
                f"{len(r.characters_per_name)} names"
                + (
                    "; several characters on one name each score as it: "
                    + ", ".join(f"{n} {k}" for n, k in shared.items())
                    if shared
                    else ""
                )
            )
        if s.reference_seconds > 0:
            lines.append(
                f"  reference purity {_pct(s.reference_purity)} of {s.reference_seconds:.1f} s "
                f"(pure {s.pure_seconds:.1f} s, foreign {s.foreign_seconds:.1f} s, "
                f"unverifiable {s.unverifiable_seconds:.1f} s = {_pct(s.unverifiable_share)})"
            )
        if s.coverage:
            lines.append("  coverage: " + ", ".join(f"{k} {v}" for k, v in s.coverage.items()))
        for name, acc in sorted(s.per_character.items(), key=lambda kv: (kv[1], kv[0])):
            lines.append(f"  {name:<20} {acc:.1%}")
        for name, sec in sorted(
            s.purity_seconds_per_character.items(),
            key=lambda kv: (s.purity_per_character.get(kv[0], 2.0), kv[0]),
        ):
            lines.append(
                f"  purity {name:<20} {_pct(s.purity_per_character.get(name))} (pure "
                f"{sec['pure_seconds']:.1f} s, foreign {sec['foreign_seconds']:.1f} s, "
                f"unverifiable {sec['unverifiable_seconds']:.1f} s)"
            )
    return "\n".join(lines)


def _list(value: str | None) -> list[str] | None:
    return [e.strip() for e in value.split(",") if e.strip()] if value else None


def names_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory."),
    tune: str = typer.Option(
        ..., "--tune", help="Tuning episodes, e.g. S01E11,S01E12,S01E13 (their names may be used)."
    ),
    holdout: str = typer.Option(
        None,
        "--holdout",
        help="Held-out episodes, comma-separated (protocol A): each must have read no names "
        "([names] holdout_episodes) in its names.json or its assignments, checked before "
        "scoring.",
    ),
    ceiling: str = typer.Option(
        None,
        "--ceiling",
        help="Episodes scored as held out without that check, comma-separated: protocol A's "
        "ceiling arm (their names were read), labelled a ceiling; the modal names still come "
        "from --tune alone.",
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
    force: bool = typer.Option(
        False, "--force", help="Score even when the subtitles do not line up with the speech."
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
            _list(tune) or [],
            _list(holdout),
            ceiling=_list(ceiling),
            gt_kind=gt,
            gt_file=gt_file,
            subtitle_track=subtitle_track,
            aliases=parse_aliases(alias),
            force=force,
        )
    except (*user_errors(), ValueError, HeldOutNamesError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for w in result.warnings:
        typer.echo(f"warning: {w}", err=True)
    typer.echo(format_run(result))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(json_out) as tmp:
            tmp.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
    if not no_ledger:
        name = test_set or Path(workspace).resolve().name
        write_ledger(lambda: names_rows(result, Workspace(workspace).config, name, tag), ledger)
