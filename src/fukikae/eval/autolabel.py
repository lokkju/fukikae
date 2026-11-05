"""Ground-truth speaker labels from the speaker names in a subtitle track's ASS dialogue."""

import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from fukikae.artifacts import atomic_path
from fukikae.eval.speakers import LabeledClip, label_key, write_labels
from fukikae.media import MediaError, ProbeResult, SubTrack, _norm_lang, probe, run_ffmpeg
from fukikae.workspace import Episode, Workspace

IGNORED_NAMES = {"sign", "signs", "title", "caption", "note", "song", "op", "ed", "lyrics"}
_MULTI = re.compile(r"[&/,]|\band\b", re.IGNORECASE)
_DEFAULT_FORMAT = "Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
_ASS_CODECS = {"ass", "ssa"}
UNKNOWN = "?"


@dataclass(frozen=True)
class Line:
    start: float
    end: float
    name: str
    text: str


def _seconds(stamp: str) -> float:
    h, m, s = stamp.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_THOUGHT = re.compile(r"\s*(?:/\s*(?:I|OS|OC|VO)|\((?:V\.O\.|O\.S\.)\))\s*$", re.IGNORECASE)


def canonical_name(name: str) -> str:
    """The name without a trailing thought/off-screen marker (`/I`, `/OS`, `(V.O.)`), with
    whitespace collapsed and spacing around parentheses normalized."""
    name = _THOUGHT.sub("", name.strip())
    name = re.sub(r"\s*\(\s*", " (", name)
    name = re.sub(r"\s*\)", ")", name)
    return re.sub(r"\s+", " ", name).strip()


def _usable(name: str) -> bool:
    if not name or label_key(name) in IGNORED_NAMES:
        return False
    return not _MULTI.search(name)


def normalize_aliases(aliases: dict[str, str] | None) -> dict[str, str]:
    """FROM (canonical, case-insensitive) -> TO (canonical spelling)."""
    return {label_key(canonical_name(a)): canonical_name(b) for a, b in (aliases or {}).items()}


def load_alias_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of FROM: TO")
    return {str(a): str(b) for a, b in data.items()}


def parse_ass(
    text: str,
    aliases: dict[str, str] | None = None,
    seen: dict[str, Counter[str]] | None = None,
    hits: Counter[tuple[str, str]] | None = None,
) -> list[Line]:
    """Dialogue lines that name exactly one speaker, in file order. Names are canonicalized,
    then mapped through `aliases`. `seen` collects raw spellings per canonical key and `hits`
    counts alias uses as (canonical FROM, TO), for the report."""
    amap = normalize_aliases(aliases)
    fields = [f.strip().casefold() for f in _DEFAULT_FORMAT.split(",")]
    lines: list[Line] = []
    in_events = False
    for raw in text.splitlines():
        row = raw.strip()
        if row.startswith("["):
            in_events = row.casefold() == "[events]"
            continue
        if not in_events:
            continue
        if row.casefold().startswith("format:"):
            fields = [f.strip().casefold() for f in row.split(":", 1)[1].split(",")]
        elif row.casefold().startswith("dialogue:"):
            parts = row.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
            if len(parts) < len(fields):
                continue
            d = dict(zip(fields, parts, strict=True))
            try:
                start, end = _seconds(d["start"]), _seconds(d["end"])
            except (KeyError, ValueError):
                continue
            raw = d.get("name", "").strip()
            name = canonical_name(raw)
            if _usable(name):
                if seen is not None:
                    seen.setdefault(label_key(name), Counter())[raw] += 1
                target = amap.get(label_key(name))
                if target is not None and label_key(target) != label_key(name):
                    if hits is not None:
                        hits[(name, target)] += 1
                    name = target
                lines.append(Line(start, end, name, d.get("text", "")))
    return lines


def label_spans(
    spans: list[tuple[float, float]], lines: list[Line], min_overlap: float
) -> list[str]:
    """One label per span: the name covering at least `min_overlap` of it, else `?`."""
    spelling: dict[str, str] = {}
    for ln in sorted(lines, key=lambda x: x.start):
        spelling.setdefault(label_key(ln.name), ln.name)
    out: list[str] = []
    for a, b in spans:
        dur = b - a
        cover: dict[str, float] = defaultdict(float)
        for ln in lines:
            ov = min(b, ln.end) - max(a, ln.start)
            if ov > 0:
                cover[label_key(ln.name)] += ov
        best = max(cover, key=lambda k: cover[k], default=None)
        if best is not None and dur > 0 and min(cover[best], dur) / dur >= min_overlap:
            out.append(spelling[best])
        else:
            out.append(UNKNOWN)
    return out


def _extract(source: Path, index: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(dest) as tmp:
        run_ffmpeg(["-i", str(source), "-map", f"0:{index}", "-c:s", "copy", "-f", "ass", str(tmp)])


def choose_subtitle_track(
    source: Path, pr: ProbeResult, work: Path, override: int | None = None
) -> tuple[int, Path]:
    """Extract the speaker-naming subtitle track to `work/eval/subtitles.ass`: `override`, else
    the non-forced English ASS stream with the most named Dialogue lines."""
    out = work / "eval" / "subtitles.ass"
    if override is not None:
        if override not in {s.index for s in pr.subtitles}:
            raise MediaError(
                f"subtitle track {override} not found; have {[s.index for s in pr.subtitles]}"
            )
        _extract(source, override, out)
        return override, out
    cands: list[SubTrack] = [
        s
        for s in pr.subtitles
        if s.codec in _ASS_CODECS and _norm_lang(s.language) == "eng" and not s.forced
    ]
    if not cands:
        raise MediaError("no non-forced English ASS subtitle stream; pass --subtitle-track")
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
) -> AutoLabelResult:
    ws = Workspace(root)
    merged = {**load_alias_file(Path(root) / "labels-aliases.yaml"), **(aliases or {})}
    seen: dict[str, Counter[str]] = {}
    hits: Counter[tuple[str, str]] = Counter()
    clips: list[LabeledClip] = []
    reports: list[ClusterReport] = []
    names: dict[str, tuple[float, set[tuple[str, str]]]] = {}
    display: dict[str, str] = {}
    for ref in episodes:
        ep = ws.find_episode(ref)
        npz = ep.work / "04-diarize" / "clusters.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"{ep.id}: no diarization at {npz}")
        ass = prepare_subtitles(ep, subtitle_track)
        lines = parse_ass(ass.read_text(encoding="utf-8", errors="replace"), merged, seen, hits)
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
