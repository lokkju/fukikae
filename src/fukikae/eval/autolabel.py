"""Ground-truth speaker labels for diarized spans, from the speaker names in a subtitle track:
ASS Name fields, SDH/CC speaker tags or colour-coded subtitles (see `eval.groundtruth`)."""

import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from fukikae.artifacts import atomic_path, read_json
from fukikae.eval.groundtruth import (
    KINDS,
    AlignmentReport,
    GroundTruthLine,
    Kind,
    check_alignment,
    read_ground_truth,
)
from fukikae.eval.groundtruth.ass_names import (
    IGNORED_NAMES,
    UNKNOWN,
    Line,
    canonical_name,
    label_spans,
    load_alias_file,
    normalize_aliases,
    parse_ass,
)
from fukikae.eval.groundtruth.base import read_cues
from fukikae.eval.groundtruth.colour import has_colour
from fukikae.eval.speakers import LabeledClip, label_key, write_labels
from fukikae.media import MediaError, ProbeResult, SubTrack, _norm_lang, probe, run_ffmpeg
from fukikae.models import Diarization, Turn
from fukikae.workspace import Episode, Workspace

__all__ = [
    "IGNORED_NAMES",
    "UNKNOWN",
    "Line",
    "canonical_name",
    "label_spans",
    "load_alias_file",
    "normalize_aliases",
    "parse_ass",
]

_ASS_CODECS = {"ass", "ssa"}
_TEXT_CODECS = _ASS_CODECS | {"subrip", "srt", "webvtt", "mov_text", "text", "ttml"}
_SDH_TITLE = re.compile(r"sdh|\bcc\b|closed|hearing|слабослыш|자막", re.IGNORECASE)


def _extract(source: Path, index: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(dest) as tmp:
        run_ffmpeg(["-i", str(source), "-map", f"0:{index}", "-c:s", "copy", "-f", "ass", str(tmp)])


def choose_subtitle_track(
    source: Path, pr: ProbeResult, work: Path, override: int | None = None
) -> tuple[int, Path]:
    """Extract the speaker-naming subtitle track to `work/eval/subtitles.ass`: `override`, else
    the non-forced English ASS stream with the most named Dialogue lines. When every English
    ASS stream is flagged forced (a release that flags its only track), they compete instead."""
    out = work / "eval" / "subtitles.ass"
    if override is not None:
        if override not in {s.index for s in pr.subtitles}:
            raise MediaError(
                f"subtitle track {override} not found; have {[s.index for s in pr.subtitles]}"
            )
        _extract(source, override, out)
        return override, out
    english = [
        s for s in pr.subtitles if s.codec in _ASS_CODECS and _norm_lang(s.language) == "eng"
    ]
    cands = [s for s in english if not s.forced] or english
    if not cands:
        raise MediaError("no English ASS subtitle stream; pass --subtitle-track")
    best: tuple[int, SubTrack, Path] | None = None
    for s in cands:
        tmp = work / "eval" / f"subtitles-{s.index}.ass"
        _extract(source, s.index, tmp)
        n = len(parse_ass(tmp.read_text(encoding="utf-8", errors="replace")))
        if best is None or n > best[0]:
            best = (n, s, tmp)
    assert best is not None
    if best[0] == 0:
        raise MediaError("no English ASS stream names its speakers; pass --subtitle-track")
    shutil.copyfile(best[2], out)
    for s in cands:
        (work / "eval" / f"subtitles-{s.index}.ass").unlink(missing_ok=True)
    return best[1].index, out


def _extract_text(source: Path, index: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(dest) as tmp:
        run_ffmpeg(["-i", str(source), "-map", f"0:{index}", "-c:s", "srt", "-f", "srt", str(tmp)])


def choose_text_track(
    source: Path,
    pr: ProbeResult,
    work: Path,
    kind: str,
    source_language: str | None = None,
    override: int | None = None,
) -> tuple[int, Path]:
    """Extract the SDH or colour-coded text track to `work/eval/subtitles.srt`: `override`, else
    among the non-forced text tracks (all of them when every one is flagged forced, as on Grid)
    those marked SDH/CC in their title, including `[CC]`, and in the episode's source language
    (else either; else all of them), the one with the most tagged (`sdh`) or coloured
    (`colour`) lines."""
    out = work / "eval" / "subtitles.srt"
    texts = [s for s in pr.subtitles if s.codec in _TEXT_CODECS]
    if override is not None:
        if override not in {s.index for s in pr.subtitles}:
            raise MediaError(
                f"subtitle track {override} not found; have {[s.index for s in pr.subtitles]}"
            )
        _extract_text(source, override, out)
        return override, out
    want = _norm_lang(source_language)
    cands: list[SubTrack] = [s for s in texts if not s.forced] or texts
    marked = [s for s in cands if _SDH_TITLE.search(s.title or "")]
    spoken = [s for s in cands if want and _norm_lang(s.language) == want]
    both = [s for s in marked if s in spoken]
    cands = both or sorted({*marked, *spoken}, key=lambda s: s.index) or cands
    if not cands:
        raise MediaError(f"no text subtitle stream for {kind} ground truth; pass --subtitle-track")
    best: tuple[int, SubTrack, Path] | None = None
    for s in cands:
        tmp = work / "eval" / f"subtitles-{s.index}.srt"
        _extract_text(source, s.index, tmp)
        if kind == "colour":
            n = has_colour(read_cues(tmp))
        else:
            n = len(read_ground_truth(tmp, "sdh", {}))
        if best is None or n > best[0]:
            best = (n, s, tmp)
    assert best is not None
    if best[0] == 0:
        raise MediaError(
            f"no subtitle stream carries {kind} speaker information; pass --subtitle-track "
            "or --gt-file"
        )
    shutil.copyfile(best[2], out)
    for s in cands:
        (work / "eval" / f"subtitles-{s.index}.srt").unlink(missing_ok=True)
    return best[1].index, out


def prepare_text_subtitles(ep: Episode, track: int | None, kind: str, language: str | None) -> Path:
    _, path = choose_text_track(ep.source, probe(ep.source), ep.work, kind, language, track)
    return path


def gt_path_for(gt_file: Path, ep_id: str, n_episodes: int) -> Path:
    """A sidecar subtitle file: the file itself for one episode, or the file in a directory
    whose name contains the episode code."""
    gt_file = Path(gt_file)
    if gt_file.is_dir():
        found = sorted(
            p for p in gt_file.iterdir() if p.is_file() and ep_id.casefold() in p.name.casefold()
        )
        if not found:
            raise FileNotFoundError(f"{ep_id}: no file with that code in {gt_file}")
        return found[0]
    if n_episodes > 1:
        raise ValueError("--gt-file names one file; give a directory for several episodes")
    if not gt_file.is_file():
        raise FileNotFoundError(f"{gt_file}: no such file")
    return gt_file


def speech_turns(work: Path) -> list[Turn]:
    """The diarizer's turns, else the cluster spans."""
    d = work / "04-diarize"
    if (d / "turns.json").is_file():
        return read_json(d / "turns.json", Diarization).turns
    out: list[Turn] = []
    with np.load(d / "clusters.npz") as z:
        for cluster in [str(x) for x in z["labels"]]:
            for a, b in np.asarray(z[f"{cluster}/spans"]).reshape(-1, 2):
                out.append(Turn(start=float(a), end=float(b), speaker=cluster))
    return out


def prepare_subtitles(ep: Episode, track: int | None) -> Path:
    _, path = choose_subtitle_track(ep.source, probe(ep.source), ep.work, track)
    return path


@dataclass(frozen=True)
class ClusterReport:
    episode: str
    cluster: str
    spans: int
    labeled: int
    majority: str | None
    purity: float | None


@dataclass(frozen=True)
class AutoLabelResult:
    clips: list[LabeledClip]
    clusters: list[ClusterReport]
    # name -> (speech seconds, clusters it spreads over as (episode, cluster))
    names: dict[str, tuple[float, set[tuple[str, str]]]]
    collapsed: list[tuple[str, list[str]]] = field(default_factory=list)
    aliased: list[tuple[str, str, int]] = field(default_factory=list)
    suggestions: list[tuple[str, str]] = field(default_factory=list)
    # per episode: (episode, alignment of the subtitles with the speech, offset applied)
    alignment: list[tuple[str, AlignmentReport, bool]] = field(default_factory=list)


def _within_one(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


def _strip_paren(name: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", name).strip()


def suggest_aliases(names: dict[str, str], amap: dict[str, str]) -> list[tuple[str, str]]:
    """Pairs of canonical names (key -> display) that look like one speaker and that no alias
    already joins: equal once a parenthetical is removed, or one edit apart at 5+ letters."""

    def final(k: str) -> str:
        return label_key(amap.get(k, k))

    keys = sorted(names)
    out: list[tuple[str, str]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if final(a) == final(b):
                continue
            pa, pb = label_key(_strip_paren(names[a])), label_key(_strip_paren(names[b]))
            if pa == pb or (len(a) >= 5 and len(b) >= 5 and _within_one(a, b)):
                out.append((names[a], names[b]))
    return out


def auto_label(
    root: Path,
    episodes: list[str],
    out: Path,
    subtitle_track: int | None = None,
    min_overlap: float = 0.6,
    aliases: dict[str, str] | None = None,
    *,
    source: str = "ass",
    gt_file: Path | None = None,
    force: bool = False,
) -> AutoLabelResult:
    """Label every diarized span from the ground truth read with `source` (ass, sdh, colour).
    The subtitles must line up with the detected speech (`check_alignment`); `force` scores a
    misaligned episode anyway."""
    if source not in KINDS:
        raise ValueError(f"unknown --source {source!r}; use one of {', '.join(KINDS)}")
    ws = Workspace(root)
    merged = {**load_alias_file(Path(root) / "labels-aliases.yaml"), **(aliases or {})}
    seen: dict[str, Counter[str]] = {}
    hits: Counter[tuple[str, str]] = Counter()
    clips: list[LabeledClip] = []
    reports: list[ClusterReport] = []
    names: dict[str, tuple[float, set[tuple[str, str]]]] = {}
    display: dict[str, str] = {}
    aligned: list[tuple[str, AlignmentReport, bool]] = []
    for ref in episodes:
        ep = ws.find_episode(ref)
        npz = ep.work / "04-diarize" / "clusters.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"{ep.id}: no diarization at {npz}")
        if gt_file is not None:
            path = gt_path_for(gt_file, ep.id, len(episodes))
        elif source == "ass":
            path = prepare_subtitles(ep, subtitle_track)
        else:
            path = prepare_text_subtitles(ep, subtitle_track, source, ws.config.source_language)
        truth: list[GroundTruthLine] = read_ground_truth(
            path,
            cast(Kind, source),
            merged,
            episode=ep.id,
            seen=seen,
            hits=hits,
        )
        truth, report, applied = check_alignment(
            truth,
            speech_turns(ep.work),
            force=force,
            label=f"{source} speaker lines in {ep.id} ({path.name})",
        )
        aligned.append((ep.id, report, applied))
        lines = [Line(t.start, t.end, t.speaker, "") for t in truth]
        with np.load(npz) as z:
            for cluster in [str(x) for x in z["labels"]]:
                spans = [
                    (float(a), float(b))
                    for a, b in np.asarray(z[f"{cluster}/spans"]).reshape(-1, 2)
                ]
                labels = label_spans(spans, lines, min_overlap)
                tally: dict[str, int] = defaultdict(int)
                for k, ((a, b), lab) in enumerate(zip(spans, labels, strict=True)):
                    clips.append(LabeledClip(ep.id, cluster, f"{cluster}-{k}", a, b, lab))
                    if lab != UNKNOWN:
                        key = label_key(lab)
                        display.setdefault(key, lab)
                        tally[key] += 1
                        secs, where = names.get(key, (0.0, set()))
                        names[key] = (secs + (b - a), where | {(ep.id, cluster)})
                top = max(tally, key=lambda k: tally[k], default=None)
                n = sum(tally.values())
                reports.append(
                    ClusterReport(
                        ep.id,
                        cluster,
                        len(spans),
                        n,
                        None if top is None else display[top],
                        None if top is None else tally[top] / n,
                    )
                )
    write_labels(out / "labels.yaml", clips)
    shown = {k: canonical_name(next(iter(c))) for k, c in seen.items()}
    collapsed = [
        (shown[k], sorted(c)) for k, c in sorted(seen.items()) if len({label_key(r) for r in c}) > 1
    ]
    aliased = [(a, b, n) for (a, b), n in sorted(hits.items())]
    return AutoLabelResult(
        clips,
        reports,
        {display[k]: v for k, v in names.items()},
        collapsed,
        aliased,
        suggest_aliases(shown, normalize_aliases(merged)),
        aligned,
    )


def format_report(result: AutoLabelResult) -> str:
    rows: list[str] = []
    current = None
    for r in result.clusters:
        if r.episode != current:
            current = r.episode
            rows.append(f"{r.episode}")
        frac = r.labeled / r.spans if r.spans else 0.0
        who = "-" if r.majority is None else r.majority
        pur = "-" if r.purity is None else f"{r.purity:.2f}"
        rows.append(
            f"  {r.cluster:<14} spans {r.spans:>4}  labeled {frac:>5.0%}  "
            f"majority {who:<16} purity {pur}"
        )
    if result.alignment:
        rows.append("subtitle alignment (share of subtitle time over detected speech)")
        for ep_id, rep, applied in result.alignment:
            note = (
                f"offset {rep.best_offset_s:+.1f} s applied, "
                f"{rep.overlap_share:.0%} -> {rep.share_at_offset:.0%}"
                if applied
                else f"{rep.overlap_share:.0%}, best offset {rep.best_offset_s:+.1f} s "
                f"({rep.share_at_offset:.0%}), not applied"
            )
            rows.append(f"  {ep_id}: {note}")
    rows.append("names")
    for name, (secs, where) in sorted(result.names.items(), key=lambda kv: -kv[1][0]):
        rows.append(f"  {name:<16} {secs:>8.1f}s  {len(where)} clusters")
    if result.collapsed:
        rows.append("spelling variants merged")
        for name, raws in result.collapsed:
            rows.append(f"  {name}: {', '.join(raws)}")
    if result.aliased:
        rows.append("aliases applied")
        for a, b, n in result.aliased:
            rows.append(f"  {a} -> {b} ({n} lines)")
    if result.suggestions:
        rows.append("alias suggestions (not applied; use --alias FROM=TO)")
        for a, b in result.suggestions:
            rows.append(f"  {a} = {b}")
    return "\n".join(rows)
