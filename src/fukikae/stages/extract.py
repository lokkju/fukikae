"""Extract stage: the source-language audio track as 44.1 kHz stereo wav."""

from pathlib import Path
from typing import Any

from fukikae import media
from fukikae.artifacts import atomic_path
from fukikae.config import Config
from fukikae.workspace import Episode


def source_track(cfg: Config, source: Path, *, timeout: float | None = None) -> media.AudioTrack:
    """The source-language audio track of ``source`` the extract stage takes: ``[extract]
    audio_track`` when set, else the source language (and ``audio_title``), else the first
    non-English track (`media.choose_audio_track`). ``timeout`` bounds the probe."""
    table = cfg.raw.get("extract", {})
    override = table.get("audio_track")
    return media.choose_audio_track(
        media.probe(source, timeout=timeout),
        cfg.source_language,
        None if override is None else int(override),
        table.get("audio_title"),
    )


def extract_config(cfg: Config, track: int) -> dict[str, Any]:
    """The extract stage's config key when it takes audio track ``track``."""
    return {
        "track": track,
        "audio_title": cfg.raw.get("extract", {}).get("audio_title"),
        "rate": 44100,
        "channels": 2,
    }


class ExtractStage:
    name = "extract"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.source]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "00-extract" / "audio.wav"]

    def _track(self, cfg: Config, ep: Episode) -> media.AudioTrack:
        return source_track(cfg, ep.source)

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        return extract_config(cfg, self._track(cfg, ep).index)

    def run(self, ep: Episode, cfg: Config) -> None:
        track = self._track(cfg, ep)
        (out,) = self.outputs(ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(out) as tmp:
            media.run_ffmpeg(
                [
                    "-i", str(ep.source.resolve()),
                    "-map", f"0:{track.index}",
                    "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
                    str(tmp),
                ]
            )  # fmt: skip
