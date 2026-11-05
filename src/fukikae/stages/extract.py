"""Extract stage: the source-language audio track as 44.1 kHz stereo wav."""

from pathlib import Path
from typing import Any

from fukikae import media
from fukikae.artifacts import atomic_path
from fukikae.config import Config
from fukikae.workspace import Episode


class ExtractStage:
    name = "extract"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.source]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "00-extract" / "audio.wav"]

    def _track(self, cfg: Config, ep: Episode) -> media.AudioTrack:
        override = cfg.raw.get("extract", {}).get("audio_track")
        return media.choose_audio_track(
            media.probe(ep.source), cfg.source_language, None if override is None else int(override)
        )

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        return {"track": self._track(cfg, ep).index, "rate": 44100, "channels": 2}

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
