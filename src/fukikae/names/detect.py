"""Which subtitle track names the speakers, and whether its Name column is usable.

`candidate_tracks` lists the tracks to try in the spec's order; `dialogue_lines` turns a
track's cues into classified lines; `assess` decides `full`, `partial` or `none` with the
failing test named, and moves the lines back by the alignment offset it applied. A Name column
that holds one value, the style names or a track that doesn't line up with the speech never
passes under `use = "auto"` (Review Focus 2)."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from fukikae import media
from fukikae.config import Config
from fukikae.cues import merge_layers
from fukikae.models import Turn
from fukikae.names.align import MIN_SHARE, check_alignment
from fukikae.names.models import NamedLine
from fukikae.names.normalize import NameClass, fold, parse_name
from fukikae.subtitles import (
    RULES,
    Cue,
    filter_cue,
    parse_ass_cues,
    parse_ass_styles,
    read_text,
    spoken_text,
)

FULL_SHARE = 0.9  # named share for a `full` verdict
MAX_TOP_SHARE = 0.8  # the most common key on more named lines than this is not speakers
MAX_STYLE_SHARE = 0.5  # keys equal to a style name on this share of named lines or more
MIN_COVERAGE = 0.5  # `[names] min_coverage` default
SAME_SPAN_DIGITS = 2  # events whose start and end agree to 10 ms share a span

Use = Literal["auto", "on", "off"]
ASS_CODECS = frozenset({"ass", "ssa"})
ASS_SUFFIXES = frozenset({".ass", ".ssa"})


@dataclass(frozen=True)
class Verdict:
    level: Literal["full", "partial", "none"]
    reason: str | None  # the failing test for `none`
    dialogue: int  # dialogue lines
    named: int  # lines with a named or role key
    share: float  # named / dialogue
    distinct: int  # distinct keys
    top_share: float  # the most common key's share of named lines
    offset_s: float  # the offset applied (lines = audio + offset); 0 when none was
    align_share: float  # share of named-line time over speech at the offset used


@dataclass(frozen=True)
class TrackChoice:
    file: Path | None
    track: media.SubTrack | None
    language: str | None  # ISO 639-1 where there is one; None for an operator file
    how: Literal["file", "track", "target", "other-language"]


def _merge_by_name(cues: list[Cue]) -> list[Cue]:
    """`merge_layers` within each Name, so two speakers saying the same words at once stay
    two lines."""
    groups: dict[str, list[Cue]] = defaultdict(list)
    for c in cues:
        groups[c.name.strip()].append(c)
    out = [c for group in groups.values() for c in merge_layers(group)]
    return sorted(out, key=lambda c: (c.start, c.end))


def dialogue_lines(cues: list[Cue], rules: Sequence[str] = RULES) -> list[NamedLine]:
    """A track's dialogue lines: layers of one line merged, furigana, filtered (sign, song,
    sdh) and empty lines dropped, each Name parsed. Two events with the same span and
    different keys both become `MULTI` (two speakers at once name nobody)."""
    kept: list[tuple[Cue, str]] = []
    for c in _merge_by_name(cues):
        if c.ruby or filter_cue(c, rules) is not None:
            continue
        text = spoken_text(c)
        if text:
            kept.append((c, text))
    parsed = [parse_name(c.name) for c, _ in kept]
    span_keys: dict[tuple[float, float], set[str]] = defaultdict(set)
    for (c, _), p in zip(kept, parsed, strict=True):
        if p.key is not None:
            span_keys[_span(c)].add(p.key)
    out: list[NamedLine] = []
    for i, ((c, text), p) in enumerate(zip(kept, parsed, strict=True)):
        multi = p.key is not None and len(span_keys[_span(c)]) > 1
        out.append(
            NamedLine(
                id=i,
                start=c.start,
                end=c.end,
                raw=c.name,
                key=None if multi else p.key,
                display=None if multi else p.display,
                cls=NameClass.MULTI if multi else p.cls,
                inner=p.inner,
                text=text,
            )
        )
    return out


def _span(c: Cue) -> tuple[float, float]:
    return round(c.start, SAME_SPAN_DIGITS), round(c.end, SAME_SPAN_DIGITS)


def _none(reason: str) -> Verdict:
    return Verdict("none", reason, 0, 0, 0.0, 0, 0.0, 0.0, 0.0)


def assess(
    lines: list[NamedLine],
    styles: set[str],
    speech: list[Turn],
    *,
    min_coverage: float = MIN_COVERAGE,
    use: Use = "auto",
    align: bool = True,
) -> tuple[Verdict, list[NamedLine]]:
    """The verdict on a track's dialogue lines against the diarize speech, and the lines moved
    back by the offset applied (all of them, named or not). `use = "auto"` needs a named share
    of `min_coverage`, two keys with none on more than MAX_TOP_SHARE of named lines, keys equal
    to a style name on under MAX_STYLE_SHARE of named lines, and named-line time over speech of
    MIN_SHARE at the best offset. `use = "on"` needs two keys and the alignment only (none of
    it when `align` is false). `use = "off"` reads nothing."""
    if use == "off":
        return _none('[names] use = "off"'), []
    named = [ln for ln in lines if ln.key is not None]
    counts = Counter(ln.key for ln in named)
    dialogue = len(lines)
    share = len(named) / dialogue if dialogue else 0.0
    top_key, top_n = counts.most_common(1)[0] if counts else (None, 0)
    top_share = top_n / len(named) if named else 0.0
    stats = Verdict("none", None, dialogue, len(named), share, len(counts), top_share, 0.0, 0.0)
    if not dialogue:
        return replace(stats, reason="no dialogue lines"), lines
    if use == "auto":
        if share < min_coverage:
            return replace(
                stats,
                reason=f"named share {share:.0%} ({len(named)} of {dialogue} lines) is under "
                f"min_coverage {min_coverage:.0%}",
            ), lines
        if len(counts) < 2 or top_share > MAX_TOP_SHARE:
            return replace(
                stats,
                reason=f"one name is dominant: {top_key!r} on {top_share:.0%} of named lines "
                f"(limit {MAX_TOP_SHARE:.0%}); a Name column with one value is not speakers",
            ), lines
        folded_styles = {fold(s) for s in styles}
        as_style = sum(1 for ln in named if ln.key in folded_styles) / len(named)
        if as_style >= MAX_STYLE_SHARE:
            return replace(
                stats,
                reason=f"names equal a style name on {as_style:.0%} of named lines "
                f"(limit under {MAX_STYLE_SHARE:.0%}); the Name column holds styles",
            ), lines
    elif len(counts) < 2:
        return replace(stats, reason=f"fewer than two distinct names ({len(counts)})"), lines
    offset, align_share = 0.0, 0.0
    if align:
        _, rep, applied = check_alignment(named, speech, force=True)
        if rep.share_at_offset < MIN_SHARE:
            return replace(
                stats,
                reason=f"names do not line up with the speech: {rep.overlap_share:.0%} of "
                f"named-line time over speech as is, {rep.share_at_offset:.0%} at the best offset "
                f"({rep.best_offset_s:+.1f} s; needs {MIN_SHARE:.0%})",
                align_share=rep.share_at_offset,
            ), lines
        if applied:
            offset, align_share = rep.best_offset_s, rep.share_at_offset
            lines = [replace(ln, start=ln.start - offset, end=ln.end - offset) for ln in lines]
        else:
            align_share = rep.overlap_share
    level = "full" if share >= FULL_SHARE else "partial"
    return replace(stats, level=level, offset_s=offset, align_share=align_share), lines


def _is_ass(path: Path, text: str) -> bool:
    head = text.lstrip()[:200]
    return (
        path.suffix.casefold() in ASS_SUFFIXES
        or "[script info]" in head.casefold()
        or head.startswith("[Events]")
    )


def assess_path(
    path: Path,
    speech: list[Turn],
    *,
    min_coverage: float = MIN_COVERAGE,
    use: Use = "auto",
    align: bool = True,
    rules: Sequence[str] = RULES,
) -> tuple[Verdict, list[NamedLine]]:
    """`assess` on a subtitle file. Only ASS/SSA has a Name field; any other format is `none`
    at once."""
    if use == "off":
        return _none('[names] use = "off"'), []
    text = read_text(path)
    if not _is_ass(path, text):
        fmt = path.suffix.lstrip(".").casefold() or "text"
        return _none(f"format has no Name field ({fmt})"), []
    lines = dialogue_lines(parse_ass_cues(text), rules)
    return assess(
        lines, parse_ass_styles(text), speech, min_coverage=min_coverage, use=use, align=align
    )


def no_name_field(choice: TrackChoice) -> str | None:
    """Why an embedded candidate can't carry names (its codec isn't ASS), or None. A file is
    judged by `assess_path` once read."""
    if choice.track is not None and choice.track.codec not in ASS_CODECS:
        return f"format has no Name field ({choice.track.codec})"
    return None


def candidate_tracks(
    source: Path, probe: media.ProbeResult, cfg: Config, episode: str | None = None
) -> list[TrackChoice]:
    """The tracks to try, in order: `[names] file` (`{episode}` replaced, `~` expanded,
    relative to the source's folder), `[names] track` (a stream index), the target language's
    subtitles as translate finds them (`media.find_target_subtitles` with `[subs]`), then each
    other language's dialogue track in probe order. A track is offered once. Non-ASS candidates
    are kept (they assess `none`). A configured file or track that isn't there is a
    MediaError."""
    source = Path(source)
    names = dict(cfg.raw.get("names", {}) or {})
    out: list[TrackChoice] = []
    file = names.get("file")
    if file:
        text = str(file).replace("{episode}", episode) if episode else str(file)
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        if not path.is_file():
            raise media.MediaError(f"[names] file {path} not found")
        out.append(TrackChoice(path, None, None, "file"))
    track = names.get("track")
    if track is not None:
        found = next((t for t in probe.subtitles if t.index == int(track)), None)
        if found is None:
            raise media.MediaError(
                f"[names] track: no subtitle track {track} in {source.name}; "
                f"have {[(t.index, t.language, t.codec) for t in probe.subtitles]}"
            )
        out.append(TrackChoice(None, found, media.short_language(found.language), "track"))
    subs = dict(cfg.raw.get("subs", {}) or {})
    if subs.get("file") and episode:
        subs["file"] = str(subs["file"]).replace("{episode}", episode)
    tfile, ttrack = media.find_target_subtitles(source, probe, cfg.target_language, subs)
    if tfile is not None or ttrack is not None:
        out.append(TrackChoice(tfile, ttrack, media.short_language(cfg.target_language), "target"))
    seen = {c.track.index for c in out if c.track is not None}
    target = media.short_language(cfg.target_language)
    languages: list[str] = []
    for t in probe.subtitles:
        lang = media.short_language(t.language)
        if lang and lang != target and lang not in languages:
            languages.append(lang)
    for lang in languages:
        t = media.dialogue_track(probe, media.normalize_language(lang))
        if t is not None and t.index not in seen:
            seen.add(t.index)
            out.append(TrackChoice(None, t, lang, "other-language"))
    return out
