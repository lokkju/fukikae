"""ffmpeg/ffprobe wrappers. Always argument lists, never a shell string."""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class MediaError(Exception):
    """ffmpeg or ffprobe failed, or the media isn't what we need."""


@dataclass(frozen=True)
class AudioTrack:
    index: int
    codec: str
    channels: int
    language: str | None
    title: str | None
    default: bool


@dataclass(frozen=True)
class SubTrack:
    index: int
    codec: str
    language: str | None
    title: str | None
    forced: bool
    cues: int | None = None  # the container's NUMBER_OF_FRAMES statistics tag, when present
    hearing_impaired: bool = False


@dataclass(frozen=True)
class ProbeResult:
    audio: list[AudioTrack] = field(default_factory=list)
    subtitles: list[SubTrack] = field(default_factory=list)
    duration: float | None = None  # seconds


_ISO1_TO_3 = {
    "en": "eng", "ja": "jpn", "ko": "kor", "zh": "zho", "es": "spa", "fr": "fra", "de": "deu",
    "it": "ita", "pt": "por", "ru": "rus", "hi": "hin", "ar": "ara", "tr": "tur", "pl": "pol",
    "nl": "nld", "sv": "swe", "th": "tha", "vi": "vie", "id": "ind", "uk": "ukr",
}  # fmt: skip
_ISO3_ALIASES = {"chi": "zho", "fre": "fra", "ger": "deu"}


def _norm_lang(code: str | None) -> str | None:
    if not code:
        return None
    c = code.lower().split("-")[0].split("_")[0]
    c = _ISO1_TO_3.get(c, c)
    return _ISO3_ALIASES.get(c, c)


_ISO3_TO_1 = {v: k for k, v in _ISO1_TO_3.items()}


def normalize_language(code: str | None) -> str | None:
    """A language code as the ISO 639-2 form ``dialogue_track`` matches (``en``, ``ger`` ->
    ``eng``, ``deu``)."""
    return _norm_lang(code)


def short_language(code: str | None) -> str | None:
    """A language code as ISO 639-1 when it has one (``eng``, ``ger``, ``en-US`` -> ``en``,
    ``de``, ``en``), else its ISO 639-2 form; None for none."""
    c = _norm_lang(code)
    return _ISO3_TO_1.get(c, c) if c else None


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(  # noqa: S603 - argument list, no shell
            cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL
        )
    except FileNotFoundError as e:
        raise MediaError(f"{cmd[0]} not found on PATH") from e
    if proc.returncode != 0:
        raise MediaError(f"{cmd[0]} exited {proc.returncode}:\n{_tail(proc.stderr)}")
    return proc


def probe(path: Path) -> ProbeResult:
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=duration"
            ":stream=index,codec_type,codec_name,channels:stream_tags"
            ":stream_disposition=default,forced,hearing_impaired",
            "-of", "json",
            str(Path(path).resolve()),
        ]
    )  # fmt: skip
    return parse_probe(json.loads(proc.stdout))


def _count(tags: dict[str, Any]) -> int | None:
    """The NUMBER_OF_FRAMES statistics tag (mkvmerge writes it, maybe with a language suffix)."""
    for key, value in tags.items():
        if key.upper().split("-", 1)[0] == "NUMBER_OF_FRAMES":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def parse_probe(doc: dict[str, Any]) -> ProbeResult:
    """ffprobe's JSON (format duration, streams with tags and dispositions) as a ProbeResult."""
    audio: list[AudioTrack] = []
    subs: list[SubTrack] = []
    for s in doc.get("streams", []):
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        if s.get("codec_type") == "audio":
            audio.append(
                AudioTrack(
                    s["index"],
                    s.get("codec_name", ""),
                    int(s.get("channels", 0)),
                    tags.get("language"),
                    tags.get("title"),
                    bool(disp.get("default")),
                )  # fmt: skip
            )
        elif s.get("codec_type") == "subtitle":
            subs.append(
                SubTrack(
                    s["index"],
                    s.get("codec_name", ""),
                    tags.get("language"),
                    tags.get("title"),
                    bool(disp.get("forced")),
                    cues=_count(tags),
                    hearing_impaired=bool(disp.get("hearing_impaired")),
                )  # fmt: skip
            )
    raw = (doc.get("format") or {}).get("duration")
    try:
        duration: float | None = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        duration = None
    return ProbeResult(audio=audio, subtitles=subs, duration=duration)


def choose_audio_track(
    probe_result: ProbeResult,
    source_language: str | None,
    override: int | None,
    title: str | None = None,
) -> AudioTrack:
    """Override wins; else source language plus title (a title nothing matches is an error);
    else language; else first non-English."""
    tracks = probe_result.audio
    if not tracks:
        raise MediaError("no audio tracks")
    if override is not None:
        for t in tracks:
            if t.index == override:
                return t
        raise MediaError(f"audio track {override} not found; have {[t.index for t in tracks]}")
    want = _norm_lang(source_language)
    if want and title:
        needle = title.lower()
        for t in tracks:
            if _norm_lang(t.language) == want and needle in (t.title or "").lower():
                return t
        raise MediaError(
            f"no {want} audio track has {title!r} in its title; "
            f"have {[(t.index, t.language, t.title) for t in tracks]}"
        )
    if want:
        for t in tracks:
            if _norm_lang(t.language) == want:
                return t
    for t in tracks:
        if _norm_lang(t.language) not in ("eng", None):
            return t
    return tracks[0]


def run_ffmpeg(args: list[str]) -> None:
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args])


# Subtitle codecs ffmpeg can turn into text; image subtitles (PGS, VobSub, DVB) need OCR.
TEXT_SUBTITLE_CODECS = frozenset({"ass", "ssa", "subrip", "srt", "webvtt", "mov_text", "text"})
SIDECAR_SUFFIXES = (".ass", ".ssa", ".srt", ".vtt")


def extract_subtitle(source: Path, track: SubTrack, out: Path) -> Path:
    """The subtitle stream at ``track`` written to ``out`` plus ``.ass`` for an ASS/SSA track,
    copied as is, else ``.srt`` (a subtitle suffix already on ``out`` is replaced, any other
    dot in the name kept: ``subs.en`` -> ``subs.en.ass``). Returns the written path."""
    ass = track.codec in ("ass", "ssa")
    out = Path(out)
    stem = out.with_suffix("") if out.suffix.casefold() in SIDECAR_SUFFIXES else out
    dest = stem.with_name(stem.name + (".ass" if ass else ".srt"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i", str(Path(source).resolve()),
            "-map", f"0:{track.index}",
            "-c:s", "copy" if ass else "srt",
            str(dest),
        ]
    )  # fmt: skip
    return dest


def _text_track(track: SubTrack) -> bool:
    return track.codec in TEXT_SUBTITLE_CODECS


def find_target_subtitles(
    source: Path, probe_result: ProbeResult, language: str, configured: dict[str, Any]
) -> tuple[Path | None, SubTrack | None]:
    """Subtitles in ``language`` for the subtitle-first path, as (file, None) or (None, track),
    or (None, None) when there are none. In order: ``[subs] file`` (``~`` expanded, relative to
    the source's folder), ``[subs] track`` (a stream index), a sidecar ``<stem>.<lang>.<ext>``
    (two- or three-letter code; ass, ssa, srt, vtt), then the embedded dialogue track of the
    language (``dialogue_track``, ruling CD25: never a forced or signs-and-songs track). A
    configured file or track that isn't there, or names an image track, is a MediaError."""
    source = Path(source)
    file = configured.get("file")
    if file:
        path = Path(str(file)).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        if not path.is_file():
            raise MediaError(f"[subs] file {path} not found")
        return path, None
    track = configured.get("track")
    if track is not None:
        for t in probe_result.subtitles:
            if t.index == int(track):
                if not _text_track(t):
                    raise MediaError(
                        f"[subs] track {t.index} is an image subtitle ({t.codec}); "
                        "fukikae reads text subtitles only"
                    )
                return None, t
        raise MediaError(
            f"[subs] track: no subtitle track {track} in {source.name}; "
            f"have {[(t.index, t.language, t.codec) for t in probe_result.subtitles]}"
        )
    want = _norm_lang(language)
    codes = [language, *([want] if want and want != language else [])]
    for code in codes:
        for suffix in SIDECAR_SUFFIXES:
            side = source.with_name(f"{source.stem}.{code}{suffix}")
            if side.is_file():
                return side, None
    return None, dialogue_track(probe_result, want)


# Ruling CD25: the subtitle path reads dialogue subtitles.
# - A title naming forced is never dialogue; nor is one naming signs or songs, unless it also
#   says full, dialogue or complete ("Full Subtitles (incl. Signs)", "Dialogue + Songs"), and
#   then only when its cue count is known and dialogue-sized.
# - Dialogue-sized: at least DIALOGUE_MIN_PER_MINUTE cues a minute and DIALOGUE_SHARE of the
#   densest plausible dialogue track (unflagged, title not ruling it out), its rate capped at
#   DIALOGUE_BAR_CAP_PER_MINUTE, so a typeset signs track with a thousand styled events never
#   sets the bar; a track denser than the cap ranks after the plausible ones.
# - A forced flag is believed unless the track reaches FORCED_DIALOGUE_MIN_PER_MINUTE as well
#   (FT E11's only English track is flagged forced and carries every line at 14.4 a minute);
#   a flagged track of unknown size is skipped.
# - Among dialogue tracks a plain one wins over closed captions (CC, SDH, or the
#   hearing_impaired disposition: sound descriptions), then an unflagged one, then stream order.
NOT_DIALOGUE_TITLE = re.compile(r"\bforced\b", re.IGNORECASE)
SONGS_TITLE = re.compile(r"\b(signs?|songs?|lyrics|karaoke)\b", re.IGNORECASE)
FULL_TITLE = re.compile(r"\b(full|dialogue|dialog|complete)\b", re.IGNORECASE)
CAPTIONS_TITLE = re.compile(r"\b(cc|sdh)\b", re.IGNORECASE)
DIALOGUE_MIN_PER_MINUTE = 3.0
FORCED_DIALOGUE_MIN_PER_MINUTE = 8.0
DIALOGUE_SHARE = 0.4
DIALOGUE_BAR_CAP_PER_MINUTE = 30.0


def _per_minute(track: SubTrack, duration: float | None) -> float | None:
    if track.cues is None or not duration or duration <= 0:
        return None
    return track.cues / (duration / 60.0)


def _title_rules_out(title: str | None) -> str | None:
    """Why a title says the track is not the dialogue, or None."""
    t = title or ""
    if NOT_DIALOGUE_TITLE.search(t):
        return f"its title {title!r} says it is not the dialogue"
    if SONGS_TITLE.search(t) and not FULL_TITLE.search(t):
        return f"its title {title!r} says it holds signs or songs only"
    return None


def dialogue_track(probe_result: ProbeResult, language: str | None) -> SubTrack | None:
    """The embedded text track of ``language`` (ISO 639-2) that carries the dialogue (ruling
    CD25), or None when there is none."""
    tracks = [
        t for t in probe_result.subtitles if _norm_lang(t.language) == language and _text_track(t)
    ]
    rates = {t.index: _per_minute(t, probe_result.duration) for t in tracks}
    plausible = [
        r
        for t in tracks
        if not t.forced and _title_rules_out(t.title) is None
        and (r := rates[t.index]) is not None
    ]  # fmt: skip
    densest = min(max(plausible, default=0.0), DIALOGUE_BAR_CAP_PER_MINUTE)
    keep: list[SubTrack] = []
    for t in tracks:
        rate = rates[t.index]
        floor = FORCED_DIALOGUE_MIN_PER_MINUTE if t.forced else DIALOGUE_MIN_PER_MINUTE
        why = _title_rules_out(t.title)
        if why is None and rate is not None and (rate < floor or rate < DIALOGUE_SHARE * densest):
            why = f"{t.cues} cues ({rate:.1f} a minute) is too few for dialogue"
        elif why is None and rate is None and t.forced:
            why = "it is flagged forced and its cue count is unknown"
        elif why is None and rate is None and SONGS_TITLE.search(t.title or ""):
            why = f"its title {t.title!r} mentions signs or songs and its cue count is unknown"
        if why is not None:
            log.info("subtitle track %d (%s): skipped, %s", t.index, t.language, why)
            continue
        if t.forced:
            log.info(
                "subtitle track %d (%s): flagged forced, but %d cues (%.1f a minute) is "
                "dialogue, so it counts as a dialogue track", t.index, t.language, t.cues,
                rate,
            )  # fmt: skip
        keep.append(t)

    def captions(t: SubTrack) -> bool:
        return t.hearing_impaired or bool(CAPTIONS_TITLE.search(t.title or ""))

    def too_dense(t: SubTrack) -> bool:
        rate = rates[t.index]
        return rate is not None and rate > DIALOGUE_BAR_CAP_PER_MINUTE

    keep.sort(key=lambda t: (captions(t), t.forced, too_dense(t), tracks.index(t)))
    return keep[0] if keep else None


_FFMPEG_VERSION: str | None = None


def ffmpeg_version() -> str:
    """The first line of ``ffmpeg -version`` (memoized), for the cache keys of stages whose
    output ffmpeg's filters shape; ``"unavailable"`` when ffmpeg can't be run."""
    global _FFMPEG_VERSION
    if _FFMPEG_VERSION is None:
        try:
            out = _run(["ffmpeg", "-hide_banner", "-version"]).stdout
            _FFMPEG_VERSION = out.splitlines()[0].strip() if out.strip() else "unknown"
        except MediaError:
            _FFMPEG_VERSION = "unavailable"
    return _FFMPEG_VERSION


def run_ffmpeg_log(args: list[str]) -> str:
    """Run ffmpeg at log level info and return its stderr, for filters that report on it
    (loudnorm's JSON)."""
    return _run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-y", *args]
    ).stderr
