"""ffmpeg/ffprobe wrappers. Always argument lists, never a shell string."""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class ProbeResult:
    audio: list[AudioTrack] = field(default_factory=list)
    subtitles: list[SubTrack] = field(default_factory=list)


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
            "stream=index,codec_type,codec_name,channels:stream_tags=language,title"
            ":stream_disposition=default,forced",
            "-of", "json",
            str(Path(path).resolve()),
        ]
    )  # fmt: skip
    audio: list[AudioTrack] = []
    subs: list[SubTrack] = []
    for s in json.loads(proc.stdout).get("streams", []):
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
                )  # fmt: skip
            )
    return ProbeResult(audio=audio, subtitles=subs)


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
    (two- or three-letter code; ass, ssa, srt, vtt), then the first non-forced text track of
    the language whose title doesn't say forced. A configured file or track that isn't there,
    or names an image track, is a MediaError."""
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
    for t in probe_result.subtitles:
        if (
            _norm_lang(t.language) == want
            and not t.forced
            and "forced" not in (t.title or "").casefold()
            and _text_track(t)
        ):
            return None, t
    return None, None


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
