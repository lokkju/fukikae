"""ffmpeg/ffprobe wrappers. Always argument lists, never a shell string."""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


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
    """Override wins; else source language plus title; else language; else first non-English."""
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
